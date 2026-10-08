# Director-12 — Decision-model interjection classifier (design)

**Date:** 2026-10-08
**Status:** draft for review
**Depends on:** Director-11 (merged), quick-wins (Parakeet live, AS-Norm shadow).
**Reference:** Unsloth decision models (LoRA + option-scoring head, calibrated
per-option probabilities, prefill-only inference) —
unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth

## 1. Problem

Once an interjection survives the reject ladder (firmware_silent → too_quiet →
out_of_cone → too_short → speaker_mismatch) and the `conf_floor` check, the
CUT/RESTORE call is `classify_interjection()` (`modes/talkback/intent.py`): a
two-stage lexicon with default-to-cut. Known weaknesses:

- **Context-blind.** "Yes" while the kiosk asked a question mid-story is an
  answer (CUT); "yes" mid-narration is a backchannel (RESTORE). The lexicon
  cannot tell them apart — it never sees what the kiosk was saying.
- **Force tokens over-fire.** "No way, really?" and "oh no" contain `no` → CUT.
  "What a story" contains `what` → CUT.
- **Default-to-cut on everything else.** Side talk ("hang on, honey"), a
  partial word, or a Parakeet mis-hearing with decent confidence all cut the
  reply. The auto-resume net that was meant to heal wrong cuts (Plan 06) was
  never built — there is no arbiter, no `NoFollowup`, no resume pop in the
  repo — so every wrong cut is final.

The planned fix in the spec (§6 arbiter: a second warm `LlmClient`) would put
a decode-heavy LLM call next to Gemma on a bandwidth-bound GB10 and only acts
*after* the reflex call. A decision model is prefill-only (no decode), returns
calibrated probabilities, and can sit *in* the decision instead of after it.

## 2. Goals / non-goals

Goals:
1. Replace the lexical decision with a calibrated, context-aware classifier
   when it is confident; fall back to the lexicon when it is not or is absent.
2. Keep the reducer pure, and keep every structural reject gate in front of it
   (the classifier never sees rejected audio and never overrides a reject).
3. Graduated rollout: shadow → on, the same pattern as `score_norm.mode` and
   `lockout.enabled`.
4. Produce the labelled corpus the model needs, from live sessions.

Non-goals:
- New-turn endpointing (Smart Turn second opinion) and garbled-transcript
  routing in LISTENING — same infrastructure, follow-up spec.
- The resume / auto-resume net (Plan 06) — still wanted; orthogonal.
- Any change to the reject ladder, duck reflex, cone, or safety net.

## 3. Label set → action

One `choice` question per interjection:

| Label | Example | Action |
|---|---|---|
| `backchannel` | "mm-hm", "wow", "no way" | RESTORE |
| `continue` | "go on", "keep going" | RESTORE |
| `side_talk` | "hang on, honey", phone call | RESTORE |
| `garbled` | fragment / mis-hearing | RESTORE |
| `answer` | "yes" to a question the kiosk just asked | CUT |
| `stop` | "stop", "pause a minute" | CUT |
| `correction` | "no, I meant the other one" | CUT |
| `new_request` | "what's the weather", "tell me a new story" | CUT |

`p_cut = sum(p[label] for CUT labels)`. Decision band:

- `p_cut >= cut_hi` (default 0.75) → CUT
- `p_cut <= restore_lo` (default 0.25) → RESTORE
- otherwise → **abstain** → today's lexicon decides (unchanged behaviour)

Abstain-to-lexicon matches the cone's fail-open semantics: the new signal only
acts when it is sure, and its absence is exactly Director-11 behaviour.

## 4. Model input

The model's "state" text:

```
KIOSK WAS SAYING: <last ~200 chars of ctx.partial_response>
KIOSK ASKED A QUESTION: <yes|no>   # partial tail ends in '?'
USER SAID: <interjection transcript>
STT CONFIDENCE: <mean_word_prob, 2dp>
```

The kiosk context is the whole point (it is what fixes "yes"). It lives in the
reducer's Context, so the reducer stamps it onto the command — pure, no worker
reads Context.

## 5. Components

### 5.1 Command / event plumbing

- `C.TranscribeInterjection` gains `context_tail: str = ""`. The reducer fills
  it from `ctx.partial_response[-200:]` in `_on_interjection_segment`.
- `E.InterjectionTranscribed` gains `cut_prob: float | None = None` and
  `label: str | None = None` (DIAG only). `None` = no classifier / timed out /
  error. Existing tests and the no-classifier path are untouched (defaults).

### 5.2 `modes/director/decision.py` (NEW, pure)

```python
def decide_interjection(text, cut_prob, cfg) -> Interjection:
    if cut_prob is not None and cfg.decision_mode == "on":
        if cut_prob >= cfg.decision_cut_hi:     return Interjection.INTERRUPT
        if cut_prob <= cfg.decision_restore_lo: return Interjection.BACKCHANNEL
    return classify_interjection(text)          # abstain / shadow / off
```

`_on_interjection_transcribed` calls this instead of `classify_interjection`.
The `conf_floor` guard stays in front of it.

### 5.3 `InterjectionClassifier` protocol + clients

Same pattern as `TurnDetector` / `NullTurnDetector`:

- `classify(text, mean_word_prob, context_tail) -> (label, cut_prob) | None`
- `NullInterjectionClassifier` — always `None` (tests, CI, disabled).
- `DecisionClient` — HTTP to a localhost sidecar, hard timeout
  (`timeout_ms`, default 150). Timeout or any error → `None` and one log line;
  it never raises into the worker.

`SttWorker` gets the classifier at construction. For `TranscribeInterjection`
only: transcribe → classify (awaited, bounded by the timeout) → emit one
`InterjectionTranscribed` carrying both. One event keeps the reducer's
EVALUATING flow unchanged; the added latency is bounded by `timeout_ms`.

### 5.4 Sidecar: `bench/decision_server.py` + `kiosk-stack.sh`

A small separate process (FastAPI or stdlib `http.server`) that loads the
LoRA + scoring head once and serves `POST /classify`. Separate process because
Unsloth patches torch at import and NeMo/Parakeet already lives in the kiosk
process — isolation avoids a dependency fight and makes the classifier
killable/restartable like llama-server. `kiosk-stack.sh` gains
`start/stop/status` for it on its own port. Startup probe in `kiosk.py`:
reachable → `✓ decision model ready`; not → `✗ decision model unavailable —
lexicon decides` (warn-only, fail open; never exit 4).

Base model: **Qwen3.5-0.8B** (≈4 GB VRAM per the Unsloth page). Prefill-only,
so it costs a short burst of bandwidth per interjection and nothing while idle
— unlike a second llama-server slot, which decodes.

## 6. Config

```yaml
barge_in:
  decision_model:
    mode: "shadow"        # off | shadow | on  — shadow = classify + DIAG, lexicon decides
    url: "http://127.0.0.1:8091/classify"
    timeout_ms: 150
    cut_hi: 0.75
    restore_lo: 0.25
```

`mode` and the thresholds map into `DirectorConfig` (`decision_mode`,
`decision_cut_hi`, `decision_restore_lo`); `url`/`timeout_ms` are read by
assembly when it builds the client (config-truth: every key read where used).
`mode: off` → `NullInterjectionClassifier` is built, no HTTP.

## 7. DIAG and the training corpus

The JSONL `EventLogger` exists but is not wired into the Director ("not on
the hot path yet", `assembly.py`). Live DIAG goes to stdout only, so there is
no corpus today. This spec wires one record per resolved interjection:

```json
{"event": "interjection", "text": "...", "mean_word_prob": 0.91,
 "context_tail": "...", "lexicon": "INTERRUPT", "model_label": "backchannel",
 "cut_prob": 0.18, "decision": "INTERRUPT", "mode": "shadow",
 "audio": "debug_audio/<session>/<seq>.wav"}
```

…plus a runtime DIAG line: `[DIAG] interjection lexicon=CUT model=backchannel
p_cut=0.18 (shadow)`. Disagreements between lexicon and model are the rows
worth hand-labelling first.

Corpus sources, in order of value:
1. **Live shadow logs** — real Parakeet output through the real ch0 path.
   Label in the tune console (one keypress per row; follow-up).
2. **Synthetic** — Gemma 4 generates kiosk story/answer tails and user
   interjections per label, run through a text-noise pass that mimics
   Parakeet errors (drops, homophones, truncation).
3. **Public** — the HRI 2025 interruption-handling set (arXiv 2501.01568, code
   released) for interrupt-vs-backchannel coverage.

Train with `DecisionTrainer` (LoRA r16, 2 epochs, head LR 1e-4), then
`calibrate()` on a held-out live slice so `cut_hi` / `restore_lo` mean what
they say.

## 8. Testing

Unit (TDD, no model in CI):
- `decide_interjection`: on/shadow/off × hi/band/lo × `None` → correct path;
  shadow and off are byte-identical to the lexicon.
- Reducer: `context_tail` stamped from `partial_response` (truncated);
  `cut_prob=None` → today's behaviour (no-regression); "yes" with
  `cut_prob=0.9` → CUT; "what a story" with `cut_prob=0.1` → RESTORE.
- `SttWorker`: classifier result lands on the event; timeout → `None`;
  exception → `None`; `UserTurnTranscribed` never calls the classifier.
- `DecisionClient`: timeout honoured against a slow fake server.
- Entrypoint: probe ✓/✗, never exits.

## 9. Live merge gate

1. **Shadow week first.** `mode: shadow` across normal sessions. Pass = latency
   p95 of transcribe+classify ≤ STT p95 + 150 ms on GB10 under Gemma load
   (re-run `bench/reflex_contention.py` with the sidecar), and model–label
   agreement on hand-labelled disagreements clearly beats the lexicon.
2. **Flip to `on`.** Scenarios: "mm-hm"/"wow"/"no way" mid-story → RESTORE;
   kiosk asks a question mid-reply, owner says "yes" → CUT and answered;
   "stop" → CUT; side talk to a companion → RESTORE; "tell me a new story" → CUT.
3. **Fail-open.** Kill the sidecar mid-session → `✗`/timeout lines, behaviour
   identical to Director-11.

## 10. Risks

- **Unsloth on aarch64/GB10** — unverified for decision-model inference; spike
  first. Fallback: export the merged 16-bit model and reimplement the scoring
  head in plain transformers (it is a small head over hidden states).
- **Corpus size** — live interjections are sparse. Synthetic data carries the
  first model; live rows correct it. Keep the band wide until calibration on
  live data supports narrowing.
- **Accuracy ceiling** — the Unsloth page reports ~74–81% on intent benchmarks
  for 0.8B–2B. Acceptable *only* because of the abstain band; never ship
  without it.
- **Latency** — no published per-request numbers. The 150 ms timeout is the
  guard; if p95 lands near it, the classifier is too slow for this seat.
