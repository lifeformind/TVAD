# Kiosk Conversation De-Risk Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the three load-bearing assumptions behind a hand-rolled "Conversation Director" rebuild — before committing to it — by (1) making backchannel-vs-question work today, (2) proving we can borrow specialist models on this GB10, and (3) measuring the reflex latency budget under real GPU contention.

**Architecture:** Three independent spikes against the *existing* `modes/talkback` controller. Spike 1 reorders the SPEAKING-state barge-in path to transcribe-then-decide and adds a pure lexical interjection classifier (zero new models). Spike 2 verifies aarch64 availability and wires a borrowed endpointing model (Smart Turn v3) + a borrowed backchannel classifier, comparing them to Spike 1's lexical baseline. Spike 3 is a standalone benchmark harness measuring reflex-path latency while the main pipeline (gemma + Kokoro + Whisper) loads the single GB10 GPU. Each spike produces working, independently-testable output.

**Tech Stack:** Python 3.12, asyncio, pytest / pytest-asyncio, numpy, faster-whisper (existing STT), gemma-3-4b via llama.cpp server, Kokoro TTS, SpeechBrain ECAPA-TDNN, Silero VAD, WebRTC AEC. New: a pure-Python `modes/talkback/intent.py`; borrowed models (Smart Turn v3 / LiveKit turn-detector, a Krisp-style backchannel classifier) evaluated in Spike 2; `bench/reflex_contention.py` in Spike 3.

**Scope note:** Spike 1 delivers requirement 3's "keep talking through 'okay'" half immediately. It does NOT make a *short* "why?" cut — that still hits the existing `verify_window_ms` floor and is deferred to the later rolling-barge-in-window work (review rec R5). Spike 1's win is that a long verified affirmation no longer cuts, and a long verified question does.

---

## File Structure

- `modes/talkback/intent.py` *(new)* — pure interjection classifier: `Interjection` enum + `classify_interjection(text)`. One responsibility: map a transcript string to BACKCHANNEL vs INTERRUPT. No I/O, no state — trivially unit-testable.
- `modes/talkback/controller.py` *(modify, SPEAKING branch of `_handle_segment`, ~lines 636-719)* — reorder to transcribe-then-classify-then-cut|restore; add backchannel short-circuit.
- `tests/kiosk/talkback/test_intent.py` *(new)* — unit tests for the classifier (lexicons, empty/garbage guard, mixed cases).
- `tests/kiosk/talkback/test_barge_in.py` *(modify)* — add integration cases asserting backchannel keeps SPEAKING + does not pollute history; question cuts.
- `bench/reflex_contention.py` *(new, Spike 3)* — contention benchmark harness.
- `docs/superpowers/specs/` — Spike 2 & 3 each append a short findings section to the eventual rebuild spec (or a `docs/notes/` file); see those tasks.

---

## Task 1: Pure interjection classifier (`intent.py`)

**Files:**
- Create: `modes/talkback/intent.py`
- Test: `tests/kiosk/talkback/test_intent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/kiosk/talkback/test_intent.py
"""Tests for the pure backchannel-vs-question interjection classifier."""

import pytest

from modes.talkback.intent import Interjection, classify_interjection


@pytest.mark.parametrize("text", [
    "okay", "ok", "yeah", "yep", "mhm", "right", "sure", "cool",
    "got it", "yeah got it", "uh-huh", "mm-hmm", "i see", "makes sense",
])
def test_pure_backchannels_do_not_interrupt(text):
    assert classify_interjection(text) == Interjection.BACKCHANNEL


@pytest.mark.parametrize("text", ["", "   ", ".", "?!"])
def test_empty_or_punctuation_only_is_backchannel(text):
    # whisper-tiny near-silence -> never cut on garbage
    assert classify_interjection(text) == Interjection.BACKCHANNEL


@pytest.mark.parametrize("text", ["you", "thank you", "thanks"])
def test_common_stt_hallucinations_are_backchannel(text):
    assert classify_interjection(text) == Interjection.BACKCHANNEL


@pytest.mark.parametrize("text", [
    "why", "why?", "wait", "stop", "what do you mean", "hold on",
    "can you repeat that", "but is that the fast one", "actually no",
])
def test_questions_and_commands_interrupt(text):
    assert classify_interjection(text) == Interjection.INTERRUPT


def test_force_token_wins_over_backchannel_tokens():
    # a force-interrupt token anywhere forces a cut even amid backchannels
    assert classify_interjection("okay but why") == Interjection.INTERRUPT


def test_multiple_content_words_interrupt():
    # not in either lexicon, >=2 content words -> treat as a real turn
    assert classify_interjection("tell me about the blue door") == Interjection.INTERRUPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/kiosk/talkback/test_intent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modes.talkback.intent'`

- [ ] **Step 3: Implement the classifier**

```python
# modes/talkback/intent.py
"""Pure backchannel-vs-question classifier for barge-in interjections.

Maps a transcript to BACKCHANNEL (the user is just acknowledging — keep
talking) or INTERRUPT (a genuine question/command — cut and respond). No I/O,
no state; the SPEAKING-branch barge-in path calls classify_interjection() on
the (already speaker-verified) interjection transcript BEFORE deciding to cut.

Design (review rec R2):
- Stage 0 EMPTY/GARBAGE GUARD (mandatory): empty / punctuation-only / known
  whisper-tiny hallucinations -> BACKCHANNEL. Never cut on garbage STT.
- Stage 1 LEXICAL: any FORCE_INTERRUPT token wins -> INTERRUPT; else if every
  token is a known backchannel -> BACKCHANNEL; else (real content) -> INTERRUPT
  (default-to-cut; a future auto-resume net self-heals wrong cuts).
"""

import enum
import re


class Interjection(enum.Enum):
    BACKCHANNEL = "BACKCHANNEL"
    INTERRUPT = "INTERRUPT"


# A force token anywhere forces a cut (questions / commands / repair).
FORCE_INTERRUPT = {
    "why", "what", "how", "when", "where", "who", "which",
    "wait", "stop", "no", "hold", "pause", "sorry", "repeat", "again",
    "but", "actually", "pardon", "explain", "mean",
}

# If EVERY token is here (and none are force tokens), it's an acknowledgment.
# Includes common whisper-tiny near-silence hallucinations (you/thanks/i).
BACKCHANNEL = {
    "okay", "ok", "yeah", "yep", "yup", "yes", "uhhuh", "mhm", "mm", "hmm",
    "right", "sure", "got", "it", "gotcha", "cool", "nice", "wow", "oh", "ah",
    "totally", "exactly", "makes", "sense", "fair", "true", "go", "on",
    "continue", "i", "see", "you", "thanks", "thank", "alright", "great",
}


def _tokenize(text: str) -> list[str]:
    t = text.lower()
    # normalize multi-word backchannels to single known tokens
    t = (t.replace("uh-huh", "uhhuh").replace("uh huh", "uhhuh")
          .replace("mm-hmm", "mhm").replace("mm hmm", "mhm")
          .replace("mhmm", "mhm"))
    return re.findall(r"[a-z]+", t)


def classify_interjection(text: str) -> Interjection:
    tokens = _tokenize(text or "")
    if not tokens:                                    # Stage 0: garbage guard
        return Interjection.BACKCHANNEL
    if any(tok in FORCE_INTERRUPT for tok in tokens):  # Stage 1: force wins
        return Interjection.INTERRUPT
    if all(tok in BACKCHANNEL for tok in tokens):      # all-acknowledgment
        return Interjection.BACKCHANNEL
    return Interjection.INTERRUPT                       # default-to-cut
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/kiosk/talkback/test_intent.py -v`
Expected: PASS (all parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add modes/talkback/intent.py tests/kiosk/talkback/test_intent.py
git commit -m "feat(talkback): pure backchannel-vs-question interjection classifier"
```

---

## Task 2: Reorder the SPEAKING barge-in path to transcribe-then-decide

**Files:**
- Modify: `modes/talkback/controller.py` (add import near line 29; rewrite SPEAKING branch ~lines 682-719 of `_handle_segment`)
- Test: `tests/kiosk/talkback/test_barge_in.py`

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/kiosk/talkback/test_barge_in.py`:

```python
class TestBackchannelVsQuestion:
    """The SPEAKING-branch reorder: classify the interjection before cutting."""

    def _make_ctrl(self):
        from modes.talkback.conversation import ConversationManager
        ctrl = TalkbackController(
            stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
            player=MagicMock(), logger=MagicMock(),
        )
        ctrl.state = TalkbackState.SPEAKING
        ctrl._conversation = ConversationManager(system_prompt="sys")
        ctrl._primary_embedding = np.ones(192, dtype=np.float32) / np.sqrt(192)
        ctrl._embedder = MagicMock()
        ctrl._embedder.extract = MagicMock(
            return_value=np.ones(192, dtype=np.float32) / np.sqrt(192))
        ctrl._talkback_config = {
            "barge_in": {"enabled": True, "require_speaker_match": True,
                         "min_speech_ms": 120, "verify_window_ms": 700,
                         "speaker_threshold": 0.20},
            "resume": {"enabled": True},
        }
        ctrl._barge_duck_enabled = True
        ctrl._proximity_rms = 0.0  # disable proximity gate for the test
        ctrl._restore_volume = MagicMock()
        ctrl._drain_playback = AsyncMock()
        ctrl._response_task = None
        return ctrl

    def _segment(self, ms=900, rms=0.5):
        n = int(16000 * ms / 1000)
        audio = np.full(n, rms, dtype=np.float32)
        seg = MagicMock()
        seg.audio = audio
        seg.duration_ms = ms
        return seg

    @pytest.mark.asyncio
    async def test_backchannel_keeps_speaking_and_does_not_pollute_history(self):
        ctrl = self._make_ctrl()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="yeah got it")

        task = await ctrl._handle_segment(self._segment())

        assert task is None
        assert ctrl.state == TalkbackState.SPEAKING       # never cut
        ctrl._restore_volume.assert_called_once()          # un-ducked
        ctrl._drain_playback.assert_not_called()           # no cut
        # CONTENT assertion (turn_count would not move anyway, conversation.py:24)
        assert all(m["content"] != "yeah got it"
                   for m in ctrl._conversation.get_messages())

    @pytest.mark.asyncio
    async def test_question_cuts_and_starts_new_turn(self):
        ctrl = self._make_ctrl()
        ctrl._stt.transcribe_segment = AsyncMock(return_value="wait why is that")
        ctrl._generate_and_speak = AsyncMock()

        task = await ctrl._handle_segment(self._segment())

        ctrl._drain_playback.assert_awaited_once()         # cut happened
        assert any(m["content"] == "wait why is that"
                   for m in ctrl._conversation.get_messages())
```

Add the import at the top of the test file (with the other imports):

```python
from unittest.mock import AsyncMock
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/kiosk/talkback/test_barge_in.py::TestBackchannelVsQuestion -v`
Expected: FAIL — `test_backchannel_keeps_speaking...` fails because the current code cuts on any verified segment regardless of transcript.

- [ ] **Step 3: Add the import**

In `modes/talkback/controller.py`, immediately after the existing line
`from modes.talkback.watchdog import AsyncWatchdog` (line 29), add:

```python
from modes.talkback.intent import Interjection, classify_interjection
```

- [ ] **Step 4: Rewrite the SPEAKING branch tail**

In `_handle_segment`, the SPEAKING branch currently runs the speaker gate
(`if barge_cfg.get("require_speaker_match", True): ... else: score = 1.0`,
ending ~line 681), then cuts (`await self._drain_playback()` ...), then
transcribes (~line 702). Replace everything FROM the cut (`# Verified
registered user — CUT...` / `await self._drain_playback()`, ~line 683)
THROUGH the end of the SPEAKING branch (the `return task` at ~line 719) with:

```python
            # Speaker-verified primary interjection. Transcribe BEFORE deciding
            # to cut so backchannels ("okay") keep us talking while questions
            # ("why?") cut. (Reorder: STT used to run only after the cut.)
            text = await self._stt.transcribe_segment(segment.audio)

            if classify_interjection(text) is Interjection.BACKCHANNEL:
                # Acknowledgment, not an interruption: un-duck, keep talking.
                # No cut, no history pollution, no resume offer.
                self._restore_volume()
                self._emit("barge_in_backchannel_ignored", {"text": text})
                return None

            # Genuine question/command — CUT the in-flight reply. Stop the old
            # playback thread and wait for it to exit before the new response
            # writes the stream, so two threads never touch it at once.
            await self._drain_playback()
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                try:
                    await self._response_task
                except asyncio.CancelledError:
                    pass

            self._handle_barge_in(
                primary_score=float(score), speech_ms=segment.duration_ms
            )

            # Remember the interrupted answer so we can offer to resume it.
            self._store_interruption()

            self._gain = 1.0
            self._ducked = False
            turn = self._conversation.turn_count + 1
            self._emit("user_turn_complete", {"text": text, "turn_number": turn})
            self._conversation.add_user_turn(text)
            self._current_query = text
            self._transition(TalkbackState.SPEAKING)
            self._emit("turn_started", {"turn_number": turn})
            task = asyncio.create_task(
                self._generate_and_speak(self._conversation, self._talkback_config)
            )
            self._response_task = task
            return task
```

Note: the old post-cut `if not text: -> LISTENING` guard is removed because an
empty transcript now classifies as BACKCHANNEL above and returns early (never
reaching the cut), so `text` is always non-empty here.

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pytest tests/kiosk/talkback/test_barge_in.py::TestBackchannelVsQuestion -v`
Expected: PASS (both cases)

- [ ] **Step 6: Run the full talkback suite to catch regressions**

Run: `pytest tests/kiosk/talkback/ -v`
Expected: PASS. If `test_multi_turn.py::TestBargeInGate` fails, confirm its mock
`transcribe_segment` returns a string that classifies as INTERRUPT (e.g.
`"interrupt"` classifies INTERRUPT — it is neither a force token nor an
all-backchannel string, so it cuts as before). Fix any test that assumed STT
runs only after the cut by setting `transcribe_segment` on the mock before the
call.

- [ ] **Step 7: Commit**

```bash
git add modes/talkback/controller.py tests/kiosk/talkback/test_barge_in.py
git commit -m "feat(talkback): classify barge-in (backchannel vs question) before cutting"
```

---

## Task 3: Manual live validation of Spike 1

**Files:** none (runtime validation)

- [ ] **Step 1: Run the kiosk and exercise both paths**

Use the existing launcher. Run: `./kiosk-stack.sh` (starts the llama.cpp server
+ kiosk per the existing script).

- [ ] **Step 2: Verify behavior by speaking**

While the kiosk is replying, say a backchannel ("yeah", "mhm", "okay") and
confirm it KEEPS TALKING (event log shows `barge_in_backchannel_ignored`, no
cut). Then, while it is replying, ask a question ("wait, why?") long enough to
clear the verify floor and confirm it CUTS and answers.

Run: `grep -E "barge_in_backchannel_ignored|barge_in\b|user_turn_complete" logs/kiosk-*.jsonl | tail -20`
Expected: backchannels emit `barge_in_backchannel_ignored`; questions emit
`barge_in` + `user_turn_complete`.

- [ ] **Step 3: Record the finding**

Append a short note (date, what worked, the verify-floor limitation for short
questions) to `docs/superpowers/specs/2026-06-12-verified-bargein-resume-design.md`
under a new "Spike 1 results" heading, then commit.

```bash
git add docs/superpowers/specs/2026-06-12-verified-bargein-resume-design.md
git commit -m "docs(talkback): record Spike 1 backchannel-vs-question live results"
```

---

## Task 4: Verify aarch64/GB10 availability of borrowed specialists (Spike 2 gate)

**Files:**
- Create: `docs/notes/2026-06-21-specialist-model-availability.md`

This is an investigation task: a spike's first job is to confirm the borrowed
models install and run on this ARM/GB10 box BEFORE wiring them in.

- [ ] **Step 1: Probe Smart Turn v3 availability on aarch64**

Smart Turn v3 ships as an ONNX/int8 model (~8MB) usable via the `pipecat-ai`
LocalSmartTurnAnalyzerV3 or directly through `onnxruntime`. Check that
`onnxruntime` imports on this aarch64 Python and that the model can be fetched.

Run: `python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"`
Expected: prints a version and providers (CPUExecutionProvider at minimum;
CUDAExecutionProvider if the aarch64 CUDA build is present).

- [ ] **Step 2: Probe a backchannel/turn classifier path**

Evaluate options in order of least friction: (a) Smart Turn v3 (endpointing /
complete-vs-incomplete), (b) LiveKit `turn-detector` v1-mini (CPU ONNX), (c) a
small HF audio classifier. For each candidate, attempt `pip install` into the
project venv and record success/failure + wheel arch.

Run: `pip download --no-deps --dest /tmp/wheelprobe pipecat-ai 2>&1 | tail -5`
Expected: resolves a wheel (note if it pulls `onnxruntime`); record any
aarch64 gap.

- [ ] **Step 3: Write the availability findings**

Create `docs/notes/2026-06-21-specialist-model-availability.md` documenting,
for Smart Turn v3 + the chosen backchannel classifier: install command that
worked, model size, where weights came from, and whether inference ran on CPU
and/or CUDA on this GB10. If NONE install cleanly on aarch64, record that as the
finding — it directly informs the rebuild decision (we keep the lexical
baseline from Task 1 and revisit).

- [ ] **Step 4: Commit**

```bash
git add docs/notes/2026-06-21-specialist-model-availability.md
git commit -m "docs: aarch64/GB10 availability of borrowed turn-taking specialists"
```

---

## Task 5: Wire Smart Turn v3 endpointing behind a clean interface (Spike 2)

**Files:**
- Create: `modes/talkback/endpointing.py` — `TurnDetector` protocol + `SmartTurnDetector` (ONNX) + `NullTurnDetector` fallback.
- Test: `tests/kiosk/talkback/test_endpointing.py`

> Only execute if Task 4 confirmed Smart Turn (or an equivalent) installs on
> aarch64. If not, mark this task skipped in the findings doc and proceed to
> Task 6 with the lexical baseline.

- [ ] **Step 1: Write the interface test**

```python
# tests/kiosk/talkback/test_endpointing.py
import numpy as np
from modes.talkback.endpointing import NullTurnDetector


def test_null_detector_reports_complete():
    det = NullTurnDetector()
    # contract: returns a float prob in [0,1]; Null always "complete"
    p = det.endpoint_prob(np.zeros(8000, dtype=np.float32), sample_rate=16000)
    assert 0.0 <= p <= 1.0
    assert p == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/kiosk/talkback/test_endpointing.py -v`
Expected: FAIL — module/class missing.

- [ ] **Step 3: Implement the interface + Null fallback (+ Smart Turn if available)**

```python
# modes/talkback/endpointing.py
"""Pluggable end-of-turn detector. NullTurnDetector is the always-complete
fallback; SmartTurnDetector wraps the borrowed Smart Turn v3 ONNX model when
available on this platform (see Task 4 findings)."""

from typing import Protocol

import numpy as np


class TurnDetector(Protocol):
    def endpoint_prob(self, audio: np.ndarray, sample_rate: int) -> float:
        """Probability in [0,1] that the speaker has finished their turn."""
        ...


class NullTurnDetector:
    """Fallback: treat every endpointed segment as a complete turn."""

    def endpoint_prob(self, audio: np.ndarray, sample_rate: int) -> float:
        return 1.0


class SmartTurnDetector:
    """Smart Turn v3 ONNX wrapper. Lazy-loads the model on first call.

    model_path: path to the smart-turn-v3 int8 onnx confirmed in Task 4.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._session = None

    def _ensure(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            self._model_path, providers=["CPUExecutionProvider"])

    def endpoint_prob(self, audio: np.ndarray, sample_rate: int) -> float:
        self._ensure()
        # Smart Turn expects 16kHz mono float32; preprocessing per the model
        # card recorded in Task 4 findings. Returns P(complete).
        feats = audio.astype(np.float32)
        out = self._session.run(None, {self._session.get_inputs()[0].name:
                                       feats[None, :]})
        return float(np.asarray(out[0]).reshape(-1)[-1])
```

- [ ] **Step 4: Run to verify the Null path passes**

Run: `pytest tests/kiosk/talkback/test_endpointing.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test SmartTurn on real audio (if model present)**

Run: `python -c "import numpy as np; from modes.talkback.endpointing import SmartTurnDetector; d=SmartTurnDetector('<path-from-task4>'); print(d.endpoint_prob(np.zeros(16000,dtype=np.float32),16000))"`
Expected: prints a float in [0,1] without error. Record latency with a `time`
wrapper for input to Spike 3.

- [ ] **Step 6: Commit**

```bash
git add modes/talkback/endpointing.py tests/kiosk/talkback/test_endpointing.py
git commit -m "feat(talkback): pluggable turn detector (Null + Smart Turn v3)"
```

---

## Task 6: Compare borrowed classifier vs lexical baseline on a labeled set (Spike 2)

**Files:**
- Create: `bench/backchannel_eval.py`
- Create: `bench/backchannel_labels.json` — small labeled set of interjection transcripts (backchannel vs question/command), seeded from `logs/kiosk-*.jsonl` `user_turn_complete` texts + hand-written cases.

- [ ] **Step 1: Build the labeled set**

Create `bench/backchannel_labels.json` as a list of `{"text": ..., "label": "BACKCHANNEL"|"INTERRUPT"}`. Seed ~40 cases: real ones mined from
`logs/kiosk-*.jsonl` plus the edge cases from Task 1's tests and ambiguous ones
("okay so what next", "no", "right?", "sure but how").

- [ ] **Step 2: Write the eval harness**

```python
# bench/backchannel_eval.py
"""Compare the lexical classifier (and, if available, a borrowed model) against
labeled interjections. Prints accuracy + confusion for each."""

import json
import sys

from modes.talkback.intent import Interjection, classify_interjection


def main(path: str = "bench/backchannel_labels.json") -> None:
    cases = json.load(open(path))
    correct = 0
    confusion = {"BB": 0, "BI": 0, "IB": 0, "II": 0}
    for c in cases:
        pred = classify_interjection(c["text"]).value
        gold = c["label"]
        correct += pred == gold
        confusion[gold[0] + pred[0]] += 1
    n = len(cases)
    print(f"lexical accuracy: {correct}/{n} = {correct / n:.1%}")
    print(f"confusion (gold,pred): {confusion}")


if __name__ == "__main__":
    main(*sys.argv[1:])
```

- [ ] **Step 3: Run the eval**

Run: `python bench/backchannel_eval.py`
Expected: prints lexical accuracy + confusion. Record the number.

- [ ] **Step 4: Record findings + decision**

Append to `docs/notes/2026-06-21-specialist-model-availability.md`: lexical
accuracy on the labeled set, whether a borrowed classifier beat it enough to
justify the dependency, and the recommendation (keep lexical for v0 vs adopt the
model). Commit.

```bash
git add bench/backchannel_eval.py bench/backchannel_labels.json docs/notes/2026-06-21-specialist-model-availability.md
git commit -m "bench: backchannel classifier eval (lexical vs borrowed)"
```

---

## Task 7: Reflex-latency-under-contention benchmark (Spike 3)

**Files:**
- Create: `bench/reflex_contention.py`
- Append findings to: `docs/notes/2026-06-21-specialist-model-availability.md` (or a new `2026-06-21-gb10-contention.md`)

**Goal:** Measure whether the reflex path can decide quickly while the main
pipeline loads the single GB10 GPU. The duck-at-onset itself is model-free and
instant; what we must bound is the *keep-vs-escalate* reflex (a short-clip
inference) and the cut-decision components (whisper-tiny STT + ECAPA embed)
under concurrent gemma + Kokoro load.

- [ ] **Step 1: Write the benchmark harness**

```python
# bench/reflex_contention.py
"""Measure reflex-path component latency on this GB10 under main-pipeline load.

Reports p50/p95 for each component (a) idle and (b) while a background thread
hammers the GPU with the main-pipeline ops (gemma generation via the llama.cpp
server + Kokoro synth), so we know the real hot-path budget under contention.
Run on the GB10 with the llama.cpp server already up (./kiosk-stack.sh)."""

import statistics
import threading
import time

import numpy as np


def _percentiles(xs):
    xs = sorted(xs)
    p50 = statistics.median(xs)
    p95 = xs[max(0, int(len(xs) * 0.95) - 1)]
    return p50, p95


def time_op(fn, n=30):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return _percentiles(out)


def make_whisper():
    from modes.talkback.stt import StreamingStt
    stt = StreamingStt(model="tiny", compute_type="int8", device="cuda")
    stt._ensure_model()
    chunk = np.zeros(int(16000 * 0.2), dtype=np.float32)  # 200ms partial proxy
    return lambda: stt._transcribe_sync(chunk)


def make_ecapa():
    # Use the project's embedder exactly as the controller does.
    from core.speaker.embedder import EcapaEmbedder  # adjust to real class
    emb = EcapaEmbedder()
    seg = np.zeros(int(16000 * 0.8), dtype=np.float32)
    return lambda: emb.extract(seg)


def gpu_load(stop_evt):
    """Background main-pipeline pressure: repeated short LLM completions."""
    import urllib.request, json
    body = json.dumps({"model": "gemma-3-4b-it", "prompt": "Say hello.",
                       "max_tokens": 64, "stream": False}).encode()
    while not stop_evt.is_set():
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8080/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            time.sleep(0.2)


def bench(label, op):
    print(f"  {label:14s} idle   p50/p95 = %.1f / %.1f ms" % time_op(op))
    stop = threading.Event()
    t = threading.Thread(target=gpu_load, args=(stop,), daemon=True)
    t.start()
    time.sleep(1.0)  # let load ramp
    print(f"  {label:14s} loaded p50/p95 = %.1f / %.1f ms" % time_op(op))
    stop.set(); t.join(timeout=2)


def main():
    print("whisper-tiny (200ms chunk):")
    bench("whisper", make_whisper())
    print("ecapa embed (800ms):")
    bench("ecapa", make_ecapa())
    # If Task 5 produced a SmartTurnDetector, add it here on CPU to confirm the
    # reflex specialist is insulated from GPU contention.


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the embedder import path**

The harness references `core.speaker.embedder`. Verify the real class name.

Run: `python -c "import core.speaker.embedder as m; print([n for n in dir(m) if n[0].isupper()])"`
Expected: prints the embedder class names; fix `make_ecapa()` to use the real
one if it differs.

- [ ] **Step 3: Run the benchmark with the server up**

Run (server already started via `./kiosk-stack.sh`): `python bench/reflex_contention.py`
Expected: prints idle vs loaded p50/p95 for whisper-tiny and ECAPA.

- [ ] **Step 4: Interpret against the budget + record**

Record the numbers and the conclusion in the findings doc:
- The model-free duck-at-onset is instant regardless (no GPU).
- If whisper-tiny/ECAPA p95 stays low under load → the cut-decision path is
  viable on-GPU. If it inflates badly → the reflex specialists (Smart Turn /
  Krisp) should run on **CPU** (they are CPU models) to stay insulated, which is
  the recommended design; note this explicitly as a rebuild constraint.
- State whether the <100 ms keep-vs-escalate reflex budget holds, and under what
  placement (CPU vs GPU) — this is the go/no-go input for owning the loop.

- [ ] **Step 5: Commit**

```bash
git add bench/reflex_contention.py docs/notes/
git commit -m "bench: reflex-path latency under GB10 GPU contention (Spike 3)"
```

---

## Self-Review

- **Spec coverage:** De-risk strategy items map to tasks — reorder for req3 (Tasks 1-3), borrow specialists + aarch64 verification (Tasks 4-6), contention benchmark (Task 7). ✓
- **Placeholders:** Task 1 & 2 contain full code. Tasks 4-7 are spikes: concrete commands + acceptance criteria, with the two genuinely environment-dependent unknowns (exact Smart Turn preprocessing, embedder class name) called out as explicit verification steps rather than hidden. ✓
- **Type consistency:** `Interjection`/`classify_interjection` used identically in Tasks 1, 2, 6. `endpoint_prob(audio, sample_rate)` consistent across Task 5. `transcribe_segment` (async, returns str), `_restore_volume`, `_store_interruption`, `_handle_barge_in(primary_score, speech_ms)` match the current controller signatures. ✓
- **Known scope limit (documented, not a gap):** short questions under `verify_window_ms` still won't cut after Spike 1 — deferred to the rolling-barge-in-window work (review R5), noted in the scope note and Task 3.
