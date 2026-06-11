# Speaker-gate measurement (Batch 2c)

The per-turn speaker gate (`talkback.turn_gate`) verifies every new turn against
the enrolled primary speaker. Its threshold (default `0.50`) was unvalidated:
the first live run showed the *enrolled* speaker scoring only **0.05–0.39**
against their own primary in clean conditions, so the gate rejected everyone.

A follow-up run added segment durations and showed the low scores are **not** a
short-segment artifact — 1.2–1.7s segments also score near zero, and the scores
are **bimodal** (a ~0.3–0.57 cluster and a ~0.0 cluster) for the *same* speaker.
That points to either an unrepresentative primary snapshot or contaminated
segments in the live pipeline — not segment length.

## ROOT CAUSE (found 2026-06-10): ECAPA needs 2–3s, turns are ~1s

Measured on the user's own clean recording, cut to fixed windows (cosine
self-similarity, median): 1.0s → **0.291** (min −0.052), 1.5s → 0.384,
2.0s → 0.481, 3.0s → 0.575; long 1–7s segments → ~0.62. Conversational turns
are ~1s, so even the enrolled speaker scores ~0.3 against their own primary —
and at 1s self/non-self both sit near zero and don't separate. This is **not** a
capture bug (SNR 35–49 dB, correct scaling, no contamination in clean sessions).

Best single-shot threshold is length-dependent (self vs a different speaker):
1.2s → 0.19 (80% accept-self / 95% reject-other), 1.5s → 0.20 (91% / 98%),
2.0s → 0.30 (100% / 100%).

**Shipped config (rolling window):** a single live turn proved *unverifiable* —
across 9 sessions, genuine-length turns (1.2–2.0s) scored anywhere from −0.14 to
0.60, so no per-turn `(threshold, min_length)` separates self from a bystander,
and a `min_verify_ms` skip just let short bystander turns through. Instead,
consecutive turn audio **accumulates** into a rolling buffer; once it reaches
`verify_window_ms` (2000) the whole window is embedded and scored once
(`speaker_threshold` 0.30, where 2s windows give ~100%/100% offline). Sub-window
turns are served provisionally (logged `turn_gate_pending`) and fed into the next
window, so a brief utterance never false-rejects the real user. A bystander leaks
until the window fills, then the window rejects and the M-of-N lockout
(`lockout`, 1-of-3 rejecting windows) ends the session. Enroll a longer primary
for an even better reference. The steps below are the tools used to reach and
confirm this.

## Step 0 — isolate ECAPA from the pipeline (do this first)

Before tuning anything, find out whether ECAPA on your mic is even capable of
consistent embeddings, with the live talkback loop (asyncio, TTS, AEC, VAD
state) removed. Record plain WAVs and run them through the *same* VAD + ECAPA
the kiosk uses:

```
cd target-vad
arecord -f S16_LE -r 16000 -c 1 -d 25 self.wav      # you, several clear sentences
arecord -f S16_LE -r 16000 -c 1 -d 25 other.wav     # a second person, same

python3 bench/ecapa_selftest.py self.wav             # within-speaker consistency
python3 bench/ecapa_selftest.py self.wav other.wav   # + self-vs-other separation
```

Read the verdict:

- **within-self mean HIGH (~0.6–0.9)** → ECAPA + mic are fine. The low *live*
  scores are a **pipeline bug** (talkback captures/processes turn segments
  wrong). Fix that, not the threshold.
- **within-self mean LOW (~0.3) or bimodal** → ECAPA on this voice/mic is the
  ceiling. Per-turn gating needs longer windows / M-of-N smoothing / a longer,
  cleaner enrollment utterance.
- With `other.wav` it also prints the achievable self-vs-other separation and a
  recommended threshold (or "OVERLAPPING").

If Step 0 shows ECAPA is fine (within-self median ~0.6) but the live kiosk still
scores near zero, the live pipeline is degrading turn audio. Capture it directly:

## Step 0.5 — dump live segments and re-embed them offline

Run the kiosk with audio dumping on; it writes the primary snapshot and every
gated turn segment to WAV:

```
cd target-vad
TVAD_DEBUG_AUDIO_DIR=debug_audio ./kiosk-stack.sh start
# wake it, speak several turns, Ctrl-C
python3 bench/ecapa_selftest.py --dump-dir debug_audio --clean self.wav
```

This re-embeds the live segments (no VAD — they're already segments) and reports:

- **turn-vs-primary (re-embedded)** — should match the live `turn_gate` scores.
- **turn-vs-turn (live)** — do the live turns embed consistently with each other?
- **live-turn vs CLEAN-self** — the decider: **high** means the live turns ARE
  you, so the live *primary snapshot* is a bad reference (fix enrollment);
  **low** means the live turn audio itself is corrupted (fix the capture path —
  e.g. the mic ring buffer dropping chunks, or TTS bleed without AEC).

## Live two-session protocol (pipeline-level)

## Why two sessions

A single session can't be labeled — the log can't tell who spoke. So run two
**separately labeled** sessions. The kiosk locks the primary to whoever speaks
first after the wake word, so in both sessions **you** wake it and become the
primary; the labeling differs in who speaks the *follow-up* turns.

The gate stays strict (rejecting) during measurement — that's fine. We only read
the `turn_gate` scores from the log; turns don't need to be answered.

## Protocol

Start the kiosk: `./kiosk-stack.sh start`

**Session A — SELF.** Say "hey mycroft", then speak **5–8 clear sentences of
2–3 seconds each**, pausing ~1s between them, all in your own voice. Let it end
(silence timeout) or Ctrl-C. Every `turn_gate` event here is **self**.

**Session B — NON-SELF.** Say "hey mycroft" yourself (so *you* are still the
primary), then **stay silent** while a **second person** speaks 5–8 sentences.
Every `turn_gate` event here is **non-self**.

Note the two session IDs (the hex in the log filenames, newest two):

```
ls -t target-vad/logs/kiosk-*.jsonl | head -2
```

## Analyze

```
cd target-vad
python3 bench/speaker_scores.py logs/kiosk-*.jsonl \
    --source turn_gate --self <SESSION_A> --nonself <SESSION_B>
```

The report prints both distributions and a verdict:

- **CLEANLY SEPARABLE** → it prints a safe threshold band and midpoint. Set
  `kiosk.talkback.turn_gate.speaker_threshold` to that midpoint in `config.yaml`.
- **OVERLAPPING** → no threshold separates self from non-self on these segments.
  Per-turn ECAPA can't gate reliably here; the fix is longer audio windows /
  M-of-N smoothing, or a longer, cleaner enrollment utterance. Re-open the
  design before shipping the gate.

Also useful — distribution of one session alone, or duration vs score (short
segments embed poorly):

```
python3 bench/speaker_scores.py --source turn_gate --group-by session
```
(`turn_gate` events now log `duration_ms` for exactly this check.)
