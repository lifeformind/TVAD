# Speaker-gate measurement (Batch 2c)

The per-turn speaker gate (`talkback.turn_gate`) verifies every new turn against
the enrolled primary speaker. Its threshold (default `0.50`) was unvalidated:
the first live run showed the *enrolled* speaker scoring only **0.05–0.39**
against their own primary in clean conditions, so the gate rejected everyone.

Before picking a threshold (or concluding per-turn gating needs a different
approach), measure real self vs non-self scores in your actual acoustic setup.

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
