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

If Step 0 shows ECAPA is fine but the live kiosk still scores low, use the live
two-session protocol below to capture the pipeline's actual `turn_gate` scores.

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
