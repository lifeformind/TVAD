# Wake-Word Kiosk Talkback (S2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming wake-word activated kiosk pipeline. System idles until any user says a wake phrase, then captures a fresh voiceprint snapshot from the wake-word audio, locks the session to that voice for its duration, and forwards only same-speaker speech segments to a downstream callback. Sessions end on silence or hard timeout, returning the system to idle.

**Architecture:** State machine (IDLE → CAPTURING → ACTIVE_SESSION → ENDING → IDLE) driven by a single mic loop that routes incoming audio chunks based on current state. In IDLE, chunks feed the openwakeword detector. In ACTIVE_SESSION, chunks feed the existing Silero VAD and emerging speech segments are scored via ECAPA against the captured snapshot, with M-of-N decision smoothing before invoking the primary-speech callback.

**Tech Stack:** Python 3.14 (`py -3.14`), openwakeword (~50 MB, new dep), reuses [core/speaker/embedder.py](../../target-vad/core/speaker/embedder.py) (ECAPA), [core/vad/silero_vad.py](../../target-vad/core/vad/silero_vad.py), [core/audio/mic_stream.py](../../target-vad/core/audio/mic_stream.py), [core/speaker/verifier.py](../../target-vad/core/speaker/verifier.py)::`cosine_similarity`. Tests use pytest + `unittest.mock`. Same-condition matching (snapshot vs. session segments captured seconds apart in same room/mic state) is the architectural fix for the Nuroum C10's voiceprint-drift problem.

**Reference spec:** [`docs/superpowers/specs/2026-05-14-kiosk-talkback-design.md`](../specs/2026-05-14-kiosk-talkback-design.md)

**Working directory:** All commands assume `cwd = c:\repos\TVAD\target-vad\` (the project root inside the repo). When git commands need it, prefix with `target-vad/` from repo root or `cd target-vad &&`. Always invoke Python as `py -3.14`.

---

## Pre-flight notes

- The shared core/+modes/ refactor is **already complete** (commit `f547f1e` and earlier). [target-vad/core/](../../target-vad/core/) holds all primitives, [target-vad/modes/kiosk/](../../target-vad/modes/kiosk/) is an empty placeholder ready to receive new files.
- **Commit after every task.** Each task's regression net is "all existing tests pass + new tests for this task pass."
- **Never use `git add -A`** — stage explicit paths only.
- **Test isolation rule:** new tests live under `tests/core/` (for shared primitives like DecisionSmoother) or `tests/kiosk/` (for kiosk-specific code). Existing 23 tests must continue to pass after every task.
- **Mocking policy:** mock the embedder (slow model load), wake detector, and mic. Use the real SileroVAD or DecisionSmoother where their deterministic logic helps test integration.
- **CWD warning:** the bash shell's CWD has been observed to revert to repo root between commands. When running git commands or python from `target-vad/`, always lead with `cd target-vad &&` to be safe.

---

## Task 1: Install openwakeword + verify API surface

**Why first:** openwakeword is the one new dependency. If it doesn't install or behaves unexpectedly on Python 3.14 / Windows, we want to know before sinking effort into wrapping it.

**Files:**
- Modify: `target-vad/requirements.txt`
- Create (temporary, deleted before commit): `target-vad/_oww_smoke.py`

- [ ] **Step 1: Install openwakeword via pip**

Run: `cd target-vad && py -3.14 -m pip install openwakeword`
Expected: install succeeds. The package will probably also install `tflite-runtime` or `onnxruntime` as a model-runtime dep (we already have `onnxruntime` per [requirements.txt](../../target-vad/requirements.txt) so this should be a no-op).

If pip errors about wheel availability for Python 3.14, **STOP and report to the user** — we'll need to evaluate alternatives (`pvporcupine` is the main commercial alternative; `precise-runner` is another open-source option but less maintained).

- [ ] **Step 2: Write a smoke-test script**

Create `target-vad/_oww_smoke.py` with content:

```python
"""Smoke test: verify openwakeword loads a bundled model and produces predictions."""
from core import compat  # noqa: F401
import numpy as np
from openwakeword.model import Model

# Load bundled hey_jarvis model. First call may download model files.
oww = Model(wakeword_models=["hey_jarvis"])
print("Loaded models:", list(oww.models.keys()))

# Feed 1 second of silence (16000 samples at 16kHz, int16 PCM)
silence = np.zeros(16000, dtype=np.int16)
preds = oww.predict(silence)
print("Predictions for silence:", preds)

# Verify the prediction dict has at least one key with our model name as a prefix
assert any("hey_jarvis" in k.lower() for k in preds.keys()), \
    f"Expected a 'hey_jarvis' key, got: {list(preds.keys())}"
print("API surface OK: predict() returns dict keyed by model name")
```

- [ ] **Step 3: Run the smoke test**

Run: `cd target-vad && py -3.14 _oww_smoke.py`
Expected: prints `Loaded models: [...]`, prints `Predictions for silence: {...}`, prints `API surface OK: ...`. First run may download model files (~10 MB total) — that's fine, just takes a few seconds.

If the model name in the prediction dict differs from `hey_jarvis` (e.g., is `hey_jarvis_v0.1` or has a path prefix), **note the actual key format** — the WakeWordDetector wrapper in Task 6 will need to match it.

- [ ] **Step 4: Add openwakeword to requirements.txt**

In `target-vad/requirements.txt`, append at the end (after the existing `rich` line):

```
openwakeword>=0.6.0
```

- [ ] **Step 5: Delete the smoke-test script**

Run: `cd target-vad && rm _oww_smoke.py`

- [ ] **Step 6: Run existing tests as regression net**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`. Adding the dep should not affect anything.

- [ ] **Step 7: Commit**

```bash
cd target-vad && git add requirements.txt && git commit -m "$(cat <<'EOF'
chore(deps): add openwakeword for kiosk wake-word detection

Smoke-tested on Python 3.14: bundled hey_jarvis model loads, predict()
returns a dict keyed by model name. Existing 23 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `kiosk:` block to config.yaml

**Files:**
- Modify: `target-vad/config.yaml`

- [ ] **Step 1: Append the `kiosk:` block**

Append this to `target-vad/config.yaml` (after the existing `core:` block, at top level):

```yaml

kiosk:
  wake_phrase: "hey_jarvis"
  wake_threshold: 0.5
  wake_capture_tail_seconds: 1.0
  session_primary_threshold: 0.60
  session_silence_timeout_s: 10
  session_hard_timeout_s: 300
  decision_smoother:
    window_size: 3
    min_matches: 2
    threshold: 0.60
```

- [ ] **Step 2: Verify YAML still parses**

Run: `cd target-vad && py -3.14 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print('core keys:', list(c['core'].keys())); print('kiosk keys:', list(c['kiosk'].keys()))"`
Expected:
```
core keys: ['vad', 'speaker', 'audio', 'paths']
kiosk keys: ['wake_phrase', 'wake_threshold', 'wake_capture_tail_seconds', 'session_primary_threshold', 'session_silence_timeout_s', 'session_hard_timeout_s', 'decision_smoother']
```

- [ ] **Step 3: Run existing tests**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `23 passed`. (Tests don't read config.yaml.)

- [ ] **Step 4: Commit**

```bash
cd target-vad && git add config.yaml && git commit -m "$(cat <<'EOF'
feat(config): add kiosk: block with wake-word and session tunables

Defaults from the kiosk talkback design spec: hey_jarvis wake phrase,
0.5 wake threshold, 0.60 session-primary cosine threshold, 10s silence
timeout, 5min hard timeout, 3-window 2-of-3 decision smoother.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Refactor SileroVAD to expose stateful `process_chunk()`

**Why:** The kiosk's state machine routes chunks per state. The existing `SileroVAD.process_stream()` is a generator over an iterable, which couples consumption to a fixed source. We need chunk-by-chunk processing with explicit segment return so the kiosk can interleave silence/timeout checks. This is a small mechanical refactor with `process_stream` reimplemented as a thin wrapper over `process_chunk` — backward-compatible.

**Files:**
- Modify: `target-vad/core/vad/silero_vad.py`
- Modify: `target-vad/tests/test_vad.py` (add new test for `process_chunk`)

- [ ] **Step 1: Write the new test**

In `target-vad/tests/test_vad.py`, append at the end of the file (after `class TestSileroVADStream`):

```python
class TestSileroVADChunkAPI:
    def test_process_chunk_returns_list(self, vad):
        """process_chunk() returns a list (possibly empty) per call."""
        silence = np.zeros(480, dtype=np.float32)
        result = vad.process_chunk(silence)
        assert isinstance(result, list)

    def test_process_chunk_silence_yields_no_segments(self, vad):
        """Feeding silence chunks one at a time should not yield segments."""
        vad.reset()
        silence = np.zeros(480, dtype=np.float32)
        all_segments = []
        for _ in range(int(2.0 * 16000 / 480)):  # ~2 seconds
            all_segments.extend(vad.process_chunk(silence))
        assert all_segments == []

    def test_process_stream_uses_process_chunk(self, vad):
        """process_stream is a thin wrapper — must yield same as iterating process_chunk."""
        vad.reset()
        chunks = [np.zeros(480, dtype=np.float32) for _ in range(20)]
        via_stream = list(vad.process_stream(iter(chunks)))
        vad.reset()
        via_chunk = []
        for c in chunks:
            via_chunk.extend(vad.process_chunk(c))
        assert len(via_stream) == len(via_chunk)
```

- [ ] **Step 2: Run new tests — should fail (process_chunk not defined)**

Run: `cd target-vad && py -3.14 -m pytest tests/test_vad.py::TestSileroVADChunkAPI -v`
Expected: 3 failures, all with `AttributeError: 'SileroVAD' object has no attribute 'process_chunk'`.

- [ ] **Step 3: Refactor `silero_vad.py` — extract `process_chunk` from `process_stream`**

In `target-vad/core/vad/silero_vad.py`, replace the entire `process_stream` method (lines 75–131) with:

```python
    def process_chunk(self, chunk: np.ndarray) -> list:
        """Stateful: feed one audio chunk, return any speech segments completed by this chunk.

        Buffers incoming audio to 512-sample Silero frames, tracks speech/silence
        transitions, and returns a list of completed SpeechSegments (possibly empty).
        Caller is responsible for the loop. See process_stream() for a generator-style adapter.
        """
        completed: list[SpeechSegment] = []
        self._input_buffer = np.concatenate([self._input_buffer, chunk])

        while len(self._input_buffer) >= self.SILERO_CHUNK_SAMPLES:
            frame = self._input_buffer[: self.SILERO_CHUNK_SAMPLES]
            self._input_buffer = self._input_buffer[self.SILERO_CHUNK_SAMPLES:]

            prob = self.is_speech(frame)

            if prob >= self.speech_threshold:
                if not self._is_speaking:
                    # Speech onset
                    self._is_speaking = True
                    self._speech_start_sample = self._current_sample
                    if len(self._pre_speech_buffer) > 0:
                        pad = self._pre_speech_buffer[-self._padding_samples:]
                        self._speech_buffer = [pad]
                    else:
                        self._speech_buffer = []
                self._speech_buffer.append(frame)
            else:
                if self._is_speaking:
                    # Speech offset — add padding then yield
                    self._speech_buffer.append(frame)
                    self._is_speaking = False

                    speech_audio = np.concatenate(self._speech_buffer)
                    duration_samples = len(speech_audio)
                    duration_ms = duration_samples / self.sample_rate * 1000

                    if duration_ms >= self.min_speech_duration_ms:
                        start_ms = self._speech_start_sample / self.sample_rate * 1000
                        end_ms = start_ms + duration_ms
                        completed.append(SpeechSegment(
                            audio=speech_audio,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            duration_ms=duration_ms,
                        ))
                    self._speech_buffer = []

            # Keep a rolling pre-speech buffer for padding
            self._pre_speech_buffer = np.concatenate(
                [self._pre_speech_buffer, frame]
            )
            max_pre = self._padding_samples * 2
            if len(self._pre_speech_buffer) > max_pre:
                self._pre_speech_buffer = self._pre_speech_buffer[-max_pre:]

            self._current_sample += self.SILERO_CHUNK_SAMPLES

        return completed

    def process_stream(self, audio_gen: Iterator[np.ndarray]) -> Iterator[SpeechSegment]:
        """Process an audio stream and yield complete speech segments.

        Thin generator wrapper over process_chunk(). Maintained for backward
        compatibility with the legacy pipeline.
        """
        for chunk in audio_gen:
            for segment in self.process_chunk(chunk):
                yield segment
```

- [ ] **Step 4: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/test_vad.py::TestSileroVADChunkAPI -v`
Expected: 3 passed.

- [ ] **Step 5: Run full regression suite**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `26 passed` (23 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
cd target-vad && git add core/vad/silero_vad.py tests/test_vad.py && git commit -m "$(cat <<'EOF'
refactor(vad): extract stateful process_chunk() from process_stream()

The kiosk pipeline needs chunk-by-chunk control to interleave wake-word
routing in IDLE state and silence/timeout checks in ACTIVE_SESSION.
process_stream() is reimplemented as a thin generator wrapper over
process_chunk() — backward compatible. 3 new tests cover the new
chunk API and the wrapper equivalence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement DecisionSmoother (TDD)

**Files:**
- Create: `target-vad/core/speaker/decision_smoother.py`
- Create: `target-vad/tests/core/__init__.py` (empty)
- Create: `target-vad/tests/core/test_decision_smoother.py`

- [ ] **Step 1: Create the tests dir and __init__.py**

Create `target-vad/tests/core/__init__.py` with content:
```python
```
(empty file)

- [ ] **Step 2: Write the failing tests**

Create `target-vad/tests/core/test_decision_smoother.py` with:

```python
"""Tests for DecisionSmoother — sliding-window M-of-N threshold counter."""

import pytest

from core.speaker.decision_smoother import DecisionSmoother


@pytest.fixture
def smoother():
    """Default 2-of-3 at threshold 0.6 (matches kiosk default)."""
    return DecisionSmoother(window_size=3, min_matches=2, threshold=0.6)


class TestDecisionSmootherBasics:
    def test_first_update_below_min_matches(self, smoother):
        """One score above threshold, but min_matches=2 not yet hit."""
        assert smoother.update(0.9) is False

    def test_two_above_threshold_triggers(self, smoother):
        """Two consecutive above-threshold scores hit M=2 in window=3."""
        assert smoother.update(0.9) is False
        assert smoother.update(0.8) is True

    def test_below_threshold_does_not_count(self, smoother):
        """Scores below threshold are not crossings."""
        assert smoother.update(0.5) is False
        assert smoother.update(0.4) is False
        assert smoother.update(0.3) is False

    def test_mixed_window(self, smoother):
        """Three scores: 0.9, 0.5, 0.7 → 2 crossings → True."""
        assert smoother.update(0.9) is False  # 1 crossing
        assert smoother.update(0.5) is False  # still 1 crossing
        assert smoother.update(0.7) is True   # 2 crossings

    def test_window_slides(self, smoother):
        """When window fills, oldest score drops out."""
        # Fill window with one crossing + two non-crossings
        smoother.update(0.9)  # crossing — window: [0.9]
        smoother.update(0.5)  # window: [0.9, 0.5]
        smoother.update(0.4)  # window: [0.9, 0.5, 0.4] — 1 crossing
        # Next call evicts 0.9
        assert smoother.update(0.4) is False  # window: [0.5, 0.4, 0.4] — 0 crossings


class TestDecisionSmootherEdgeCases:
    def test_threshold_inclusive(self):
        """A score exactly equal to threshold is a crossing (>=)."""
        s = DecisionSmoother(window_size=2, min_matches=2, threshold=0.5)
        s.update(0.5)
        assert s.update(0.5) is True

    def test_min_matches_one(self):
        """min_matches=1 fires on the very first crossing."""
        s = DecisionSmoother(window_size=3, min_matches=1, threshold=0.5)
        assert s.update(0.6) is True

    def test_reset_clears_window(self, smoother):
        """reset() empties the window — fresh M-of-N from zero."""
        smoother.update(0.9)
        smoother.update(0.9)  # would be True
        smoother.reset()
        assert smoother.update(0.9) is False  # only 1 in window now


class TestDecisionSmootherKioskDefaults:
    def test_realistic_self_speech_pattern(self):
        """Self-speech scores from the C10 (0.40, 0.72, 0.60): 2-of-3 at 0.55 fires on segment 3."""
        # Note: kiosk default threshold is 0.60, but the spec also documents 0.55 as a
        # tunable lower bound. This test uses 0.55 to mirror the spec's example math.
        s = DecisionSmoother(window_size=3, min_matches=2, threshold=0.55)
        assert s.update(0.40) is False  # below
        assert s.update(0.72) is False  # 1 crossing
        assert s.update(0.60) is True   # 2 crossings (0.72 + 0.60)

    def test_three_random_ambient_segments_silent(self):
        """Three sub-threshold ambient scores → never triggers."""
        s = DecisionSmoother(window_size=3, min_matches=2, threshold=0.6)
        for score in (0.21, 0.43, 0.30):
            assert s.update(score) is False
```

- [ ] **Step 3: Run new tests — should fail (module not found)**

Run: `cd target-vad && py -3.14 -m pytest tests/core/test_decision_smoother.py -v`
Expected: collection errors / `ModuleNotFoundError: No module named 'core.speaker.decision_smoother'`.

- [ ] **Step 4: Implement DecisionSmoother**

Create `target-vad/core/speaker/decision_smoother.py` with:

```python
"""Sliding-window M-of-N decision smoother for noisy per-segment scores."""

from collections import deque
from typing import Deque


class DecisionSmoother:
    """Counts threshold-crossings in a sliding window of recent scores.

    Per-instance state. Use one smoother per session (kiosk creates one when
    a session starts, discards it on session end). The smoother is intentionally
    dumb — it just tracks rolling crossings; consumers decide what to do with
    the True/False signal it returns.
    """

    def __init__(self, window_size: int, min_matches: int, threshold: float):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_matches < 1 or min_matches > window_size:
            raise ValueError(
                f"min_matches must be in [1, window_size]={window_size}, got {min_matches}"
            )
        self.window_size = window_size
        self.min_matches = min_matches
        self.threshold = threshold
        self._scores: Deque[float] = deque(maxlen=window_size)

    def update(self, score: float) -> bool:
        """Append a score, return True if window has min_matches >= threshold."""
        self._scores.append(score)
        crossings = sum(1 for s in self._scores if s >= self.threshold)
        return crossings >= self.min_matches

    def reset(self) -> None:
        """Empty the window. Next update starts fresh."""
        self._scores.clear()
```

- [ ] **Step 5: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/core/test_decision_smoother.py -v`
Expected: 9 passed (the count of test methods above).

- [ ] **Step 6: Run full regression suite**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `35 passed` (26 from before + 9 new).

- [ ] **Step 7: Commit**

```bash
cd target-vad && git add core/speaker/decision_smoother.py tests/core/__init__.py tests/core/test_decision_smoother.py && git commit -m "$(cat <<'EOF'
feat(core): add DecisionSmoother — sliding-window M-of-N threshold counter

Used by the kiosk pipeline to smooth noisy per-segment ECAPA scores
before deciding whether to invoke the primary-speech callback. Pure
logic, no external deps. Defaults validated against the spec's worked
example: 2-of-3 at threshold 0.55 fires on the C10's [0.40, 0.72, 0.60]
self-speech pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement Session dataclass

**Files:**
- Create: `target-vad/modes/kiosk/session.py`
- Create: `target-vad/tests/kiosk/__init__.py` (empty)
- Create: `target-vad/tests/kiosk/test_session.py`

- [ ] **Step 1: Create kiosk tests dir**

Create `target-vad/tests/kiosk/__init__.py` with content:
```python
```
(empty file)

- [ ] **Step 2: Write the failing tests**

Create `target-vad/tests/kiosk/test_session.py` with:

```python
"""Tests for the Session dataclass — holds session-primary state."""

import numpy as np

from core.speaker.decision_smoother import DecisionSmoother
from modes.kiosk.session import Session


def make_session(now: float = 1000.0) -> Session:
    return Session(
        primary_embedding=np.ones(192, dtype=np.float32) / np.sqrt(192),
        smoother=DecisionSmoother(window_size=3, min_matches=2, threshold=0.6),
        started_at=now,
        last_speech_at=now,
    )


class TestSession:
    def test_silence_duration(self):
        s = make_session(now=1000.0)
        assert s.silence_duration(now=1003.5) == 3.5

    def test_session_duration(self):
        s = make_session(now=1000.0)
        assert s.session_duration(now=1042.0) == 42.0

    def test_silence_resets_when_last_speech_advances(self):
        s = make_session(now=1000.0)
        s.last_speech_at = 1010.0
        assert s.silence_duration(now=1012.0) == 2.0
```

- [ ] **Step 3: Run new tests — should fail (module not found)**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_session.py -v`
Expected: `ModuleNotFoundError: No module named 'modes.kiosk.session'`.

- [ ] **Step 4: Implement Session**

Create `target-vad/modes/kiosk/session.py` with:

```python
"""Session — per-session state for the active kiosk speaker lock."""

from dataclasses import dataclass

import numpy as np

from core.speaker.decision_smoother import DecisionSmoother


@dataclass
class Session:
    """All state for a single ACTIVE_SESSION. Discarded on session end."""
    primary_embedding: np.ndarray   # 192-dim L2-normalized snapshot from wake-word audio
    smoother: DecisionSmoother
    started_at: float               # time.monotonic() at session start
    last_speech_at: float           # time.monotonic() at most recent matched speech

    def silence_duration(self, now: float) -> float:
        """Seconds since last matched primary-speech segment."""
        return now - self.last_speech_at

    def session_duration(self, now: float) -> float:
        """Seconds since session start."""
        return now - self.started_at
```

- [ ] **Step 5: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_session.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run full regression suite**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `38 passed` (35 + 3).

- [ ] **Step 7: Commit**

```bash
cd target-vad && git add modes/kiosk/session.py tests/kiosk/__init__.py tests/kiosk/test_session.py && git commit -m "$(cat <<'EOF'
feat(kiosk): add Session dataclass for per-session primary-speaker state

Holds the captured primary embedding, the decision smoother instance,
and timing state. Helper methods for silence and session duration.
Discarded on session end (state lives only as long as the session).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement WakeWordDetector wrapper

**Why a wrapper:** isolates the kiosk pipeline from openwakeword API quirks (model name suffixes, prediction dict shape, frame size requirements). Wrapper presents a small surface: feed a chunk, get back `Optional[float]` (confidence if above threshold, else `None`).

**Files:**
- Create: `target-vad/modes/kiosk/wake_word.py`
- Create: `target-vad/tests/kiosk/test_wake_word.py`

**Note on openwakeword frame size:** openwakeword expects 1280-sample (80ms) frames at 16 kHz. Our mic produces 480-sample chunks. The wrapper handles buffering internally so callers can feed any chunk size.

- [ ] **Step 1: Write the tests (using mock openwakeword Model)**

Create `target-vad/tests/kiosk/test_wake_word.py` with:

```python
"""Tests for WakeWordDetector — wraps openwakeword with a clean Optional[float] API."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_model():
    """A fake openwakeword Model that returns a controllable predictions dict."""
    m = MagicMock()
    m.predict.return_value = {"hey_jarvis_v0.1": 0.0}
    return m


@pytest.fixture
def detector(fake_model):
    from modes.kiosk.wake_word import WakeWordDetector
    with patch("modes.kiosk.wake_word.Model", return_value=fake_model):
        det = WakeWordDetector(model_name="hey_jarvis", threshold=0.5)
    return det


class TestWakeWordDetector:
    def test_below_threshold_returns_none(self, detector, fake_model):
        fake_model.predict.return_value = {"hey_jarvis_v0.1": 0.3}
        # Feed enough audio to trigger one prediction (1280 samples at 16kHz)
        chunks_needed = (1280 + 479) // 480
        result = None
        for _ in range(chunks_needed):
            result = detector.process(np.zeros(480, dtype=np.float32))
        assert result is None

    def test_above_threshold_returns_score(self, detector, fake_model):
        fake_model.predict.return_value = {"hey_jarvis_v0.1": 0.87}
        chunks_needed = (1280 + 479) // 480
        result = None
        for _ in range(chunks_needed):
            result = detector.process(np.zeros(480, dtype=np.float32))
        assert result == pytest.approx(0.87)

    def test_threshold_inclusive(self, detector, fake_model):
        fake_model.predict.return_value = {"hey_jarvis_v0.1": 0.5}
        chunks_needed = (1280 + 479) // 480
        result = None
        for _ in range(chunks_needed):
            result = detector.process(np.zeros(480, dtype=np.float32))
        assert result == pytest.approx(0.5)

    def test_unrelated_model_keys_ignored(self, detector, fake_model):
        fake_model.predict.return_value = {"alexa_v0.1": 0.99, "hey_jarvis_v0.1": 0.3}
        chunks_needed = (1280 + 479) // 480
        result = None
        for _ in range(chunks_needed):
            result = detector.process(np.zeros(480, dtype=np.float32))
        assert result is None  # alexa is high but we only care about hey_jarvis

    def test_partial_buffer_no_prediction(self, detector, fake_model):
        """Less than 1280 samples buffered yet — no predict call, returns None."""
        # 480 samples is less than 1280
        result = detector.process(np.zeros(480, dtype=np.float32))
        assert result is None
        fake_model.predict.assert_not_called()

    def test_reset_clears_buffer(self, detector, fake_model):
        # Partially fill the internal buffer
        detector.process(np.zeros(480, dtype=np.float32))
        detector.reset()
        # After reset, need full 1280 samples again
        result = detector.process(np.zeros(480, dtype=np.float32))
        assert result is None
        fake_model.predict.assert_not_called()
```

- [ ] **Step 2: Run new tests — should fail (module not found)**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_wake_word.py -v`
Expected: `ModuleNotFoundError: No module named 'modes.kiosk.wake_word'`.

- [ ] **Step 3: Implement WakeWordDetector**

Create `target-vad/modes/kiosk/wake_word.py` with:

```python
"""WakeWordDetector — thin wrapper over openwakeword with a clean Optional[float] API."""

from typing import Optional

import numpy as np
from openwakeword.model import Model


class WakeWordDetector:
    """Buffers audio chunks to openwakeword's 1280-sample frame size and
    returns the wake-phrase confidence when it crosses the threshold.

    Handles model-name suffix variation (e.g. 'hey_jarvis_v0.1') by matching
    on substring. Other models in the predictions dict are ignored.
    """

    OWW_FRAME_SAMPLES = 1280  # 80ms at 16 kHz, what openwakeword expects

    def __init__(self, model_name: str, threshold: float):
        self.model_name = model_name
        self.threshold = threshold
        self._model = Model(wakeword_models=[model_name])
        self._buffer = np.array([], dtype=np.float32)

    def process(self, chunk: np.ndarray) -> Optional[float]:
        """Append chunk to internal buffer, run predict on full 1280-sample frames.

        Returns the highest matching confidence above threshold, or None.
        """
        self._buffer = np.concatenate([self._buffer, chunk])
        best_score: Optional[float] = None
        while len(self._buffer) >= self.OWW_FRAME_SAMPLES:
            frame_f32 = self._buffer[: self.OWW_FRAME_SAMPLES]
            self._buffer = self._buffer[self.OWW_FRAME_SAMPLES :]
            # openwakeword expects int16 PCM
            frame_i16 = (frame_f32 * 32767.0).clip(-32768, 32767).astype(np.int16)
            preds = self._model.predict(frame_i16)
            for key, score in preds.items():
                if self.model_name in key.lower():
                    if score >= self.threshold:
                        if best_score is None or score > best_score:
                            best_score = float(score)
        return best_score

    def reset(self) -> None:
        """Clear the internal audio buffer and reset openwakeword model state."""
        self._buffer = np.array([], dtype=np.float32)
        if hasattr(self._model, "reset"):
            self._model.reset()
```

- [ ] **Step 4: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_wake_word.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run full regression suite**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `44 passed` (38 + 6).

- [ ] **Step 6: Commit**

```bash
cd target-vad && git add modes/kiosk/wake_word.py tests/kiosk/test_wake_word.py && git commit -m "$(cat <<'EOF'
feat(kiosk): add WakeWordDetector wrapper over openwakeword

Buffers arbitrary-size chunks to openwakeword's 1280-sample frame size,
matches model name as a substring (handles version-suffixed keys like
hey_jarvis_v0.1), returns Optional[float] confidence above threshold.
Tested with mocked Model — 6 cases covering buffer math, threshold
behavior, multi-model dict, and reset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement KioskPipeline state machine

**The big one.** Multi-step task because the state machine has four distinct behaviors. Each step is its own TDD cycle but all the code lives in one file with one tested class.

**Files:**
- Create: `target-vad/modes/kiosk/pipeline.py`
- Create: `target-vad/tests/kiosk/test_pipeline.py`

**Design notes for the implementer:**
- Constructor accepts `_mic`, `_vad`, `_embedder`, `_wake_detector` underscore-kwargs for test injection. Production code passes nothing and the pipeline builds defaults from config.
- The state is a string (`"IDLE"`, `"CAPTURING"`, `"ACTIVE_SESSION"`, `"ENDING"`) on `self._state` for simplicity. Transitions are explicit: `self._state = "..."`.
- Time uses `time.monotonic()` so tests can patch `time.monotonic` to drive the clock deterministically.
- The mic loop runs in `run()` and is single-threaded. `stop()` flips a flag the loop checks.
- During CAPTURING the wake-word audio is not actually captured by us — the wake detector consumed those frames already. Per spec, the snapshot is "wake_audio + tail_seconds." For simplicity in v1, we capture the **next `wake_capture_tail_seconds` of audio after the wake fires** and use that as the snapshot. The wake-word audio itself is discarded — the same speaker should still produce a representative embedding from a 1-second tail (ECAPA needs ≥800 ms anyway; 1000 ms tail meets that).
- Errors in the downstream callback do NOT crash the pipeline (per spec: "the kiosk should be resilient to downstream bugs").

- [ ] **Step 1: Write the tests for `__init__` and IDLE wake-detection**

Create `target-vad/tests/kiosk/test_pipeline.py` with:

```python
"""Tests for KioskPipeline state machine. Mocks all I/O dependencies."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.speaker.decision_smoother import DecisionSmoother


@pytest.fixture
def base_config():
    return {
        "core": {
            "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 480},
            "vad": {
                "sample_rate": 16000,
                "speech_threshold": 0.5,
                "min_speech_duration_ms": 300,
                "padding_ms": 200,
            },
        },
        "kiosk": {
            "wake_phrase": "hey_jarvis",
            "wake_threshold": 0.5,
            "wake_capture_tail_seconds": 1.0,
            "session_primary_threshold": 0.6,
            "session_silence_timeout_s": 10,
            "session_hard_timeout_s": 300,
            "decision_smoother": {"window_size": 3, "min_matches": 2, "threshold": 0.6},
        },
    }


@pytest.fixture
def fake_mic():
    """Mic that yields a fixed list of chunks then stops."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=None)
    return m


@pytest.fixture
def fake_vad():
    m = MagicMock()
    m.process_chunk = MagicMock(return_value=[])
    m.reset = MagicMock()
    return m


@pytest.fixture
def fake_embedder():
    m = MagicMock()
    # Return a unit vector — same one every call by default
    m.extract = MagicMock(return_value=np.ones(192, dtype=np.float32) / np.sqrt(192))
    return m


@pytest.fixture
def fake_wake():
    m = MagicMock()
    m.process = MagicMock(return_value=None)  # no wake by default
    m.reset = MagicMock()
    return m


def make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                  on_primary=None, on_started=None, on_ended=None):
    from modes.kiosk.pipeline import KioskPipeline
    return KioskPipeline(
        config=base_config,
        on_primary_speech=on_primary or (lambda seg, emb: None),
        on_session_started=on_started or (lambda: None),
        on_session_ended=on_ended or (lambda reason: None),
        _mic=fake_mic,
        _vad=fake_vad,
        _embedder=fake_embedder,
        _wake_detector=fake_wake,
    )


class TestKioskPipelineInit:
    def test_starts_in_idle(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        assert p._state == "IDLE"
        assert p._session is None

    def test_stop_sets_running_false(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._running = True
        p.stop()
        assert p._running is False
```

- [ ] **Step 2: Run tests — should fail (module not found)**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_pipeline.py -v`
Expected: `ModuleNotFoundError: No module named 'modes.kiosk.pipeline'`.

- [ ] **Step 3: Implement minimal KioskPipeline (just enough to pass init tests)**

Create `target-vad/modes/kiosk/pipeline.py` with:

```python
"""KioskPipeline — state machine for the wake-word talkback kiosk."""

import time
from typing import Any, Callable, Optional

import numpy as np

from core.audio.mic_stream import MicrophoneStream
from core.speaker.decision_smoother import DecisionSmoother
from core.speaker.embedder import EmbeddingExtractor
from core.speaker.verifier import cosine_similarity
from core.vad.silero_vad import SileroVAD, SpeechSegment
from modes.kiosk.session import Session
from modes.kiosk.wake_word import WakeWordDetector


class KioskPipeline:
    """Wake-word activated streaming kiosk with session-locked speaker filter.

    States:
      IDLE             — feeding chunks to wake detector
      CAPTURING        — collecting wake_capture_tail_seconds of audio for snapshot
      ACTIVE_SESSION   — feeding chunks to VAD, scoring segments against snapshot
      ENDING           — transient cleanup before returning to IDLE
    """

    def __init__(
        self,
        config: dict,
        on_primary_speech: Callable[[SpeechSegment, np.ndarray], None],
        on_session_started: Callable[[], None] = lambda: None,
        on_session_ended: Callable[[str], None] = lambda reason: None,
        # Underscore kwargs are for test injection. Production code omits them.
        _mic: Optional[Any] = None,
        _vad: Optional[Any] = None,
        _embedder: Optional[Any] = None,
        _wake_detector: Optional[Any] = None,
    ):
        self.config = config
        self.on_primary_speech = on_primary_speech
        self.on_session_started = on_session_started
        self.on_session_ended = on_session_ended

        kiosk_cfg = config["kiosk"]
        self._tail_samples = int(
            kiosk_cfg["wake_capture_tail_seconds"] * config["core"]["audio"]["sample_rate"]
        )
        self._session_primary_threshold = kiosk_cfg["session_primary_threshold"]
        self._silence_timeout_s = kiosk_cfg["session_silence_timeout_s"]
        self._hard_timeout_s = kiosk_cfg["session_hard_timeout_s"]
        self._smoother_cfg = kiosk_cfg["decision_smoother"]

        self.mic = _mic or MicrophoneStream(config["core"]["audio"])
        self.vad = _vad or SileroVAD(config["core"]["vad"])
        self.embedder = _embedder or EmbeddingExtractor()
        self.wake_detector = _wake_detector or WakeWordDetector(
            kiosk_cfg["wake_phrase"], kiosk_cfg["wake_threshold"]
        )

        self._state = "IDLE"
        self._session: Optional[Session] = None
        self._capture_buffer = np.array([], dtype=np.float32)
        self._running = False

    def stop(self) -> None:
        """Signal run() to exit cleanly on the next loop iteration."""
        self._running = False

    def run(self) -> None:
        """Main mic loop. Blocks until stop() is called or KeyboardInterrupt."""
        self._running = True
        try:
            with self.mic:
                for chunk in self.mic.stream():
                    if not self._running:
                        break
                    self._handle_chunk(chunk)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            if self._state == "ACTIVE_SESSION":
                self._end_session("stopped")

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        """Route a single mic chunk based on current state."""
        if self._state == "IDLE":
            self._handle_idle_chunk(chunk)
        elif self._state == "CAPTURING":
            self._handle_capturing_chunk(chunk)
        elif self._state == "ACTIVE_SESSION":
            self._handle_active_chunk(chunk)
        # ENDING is transient — handled inline by _end_session

    def _handle_idle_chunk(self, chunk: np.ndarray) -> None:
        wake_score = self.wake_detector.process(chunk)
        if wake_score is not None:
            self._state = "CAPTURING"
            self._capture_buffer = np.array([], dtype=np.float32)

    def _handle_capturing_chunk(self, chunk: np.ndarray) -> None:
        self._capture_buffer = np.concatenate([self._capture_buffer, chunk])
        if len(self._capture_buffer) >= self._tail_samples:
            snapshot_audio = self._capture_buffer[: self._tail_samples]
            self._start_session(snapshot_audio)

    def _start_session(self, snapshot_audio: np.ndarray) -> None:
        try:
            embedding = self.embedder.extract(snapshot_audio)
        except Exception:
            # Snapshot failed (e.g. all silence) — abort session, return to IDLE
            self._state = "IDLE"
            self._capture_buffer = np.array([], dtype=np.float32)
            self.wake_detector.reset()
            return
        now = time.monotonic()
        self._session = Session(
            primary_embedding=embedding,
            smoother=DecisionSmoother(**self._smoother_cfg),
            started_at=now,
            last_speech_at=now,
        )
        self.vad.reset()
        self._state = "ACTIVE_SESSION"
        self._safe_callback(self.on_session_started)

    def _handle_active_chunk(self, chunk: np.ndarray) -> None:
        # Check timeouts before processing more audio
        now = time.monotonic()
        assert self._session is not None
        if self._session.silence_duration(now) >= self._silence_timeout_s:
            self._end_session("silence_timeout")
            return
        if self._session.session_duration(now) >= self._hard_timeout_s:
            self._end_session("hard_timeout")
            return

        # Feed chunk to VAD, process any completed speech segments
        for segment in self.vad.process_chunk(chunk):
            self._process_session_segment(segment)

    def _process_session_segment(self, segment: SpeechSegment) -> None:
        assert self._session is not None
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception:
            return  # skip this segment, session continues
        score = cosine_similarity(embedding, self._session.primary_embedding)
        matched = self._session.smoother.update(score)
        if matched:
            self._session.last_speech_at = time.monotonic()
            self._safe_callback(self.on_primary_speech, segment, embedding)

    def _end_session(self, reason: str) -> None:
        self._session = None
        self._state = "IDLE"
        self._capture_buffer = np.array([], dtype=np.float32)
        self.wake_detector.reset()
        self._safe_callback(self.on_session_ended, reason)

    def _safe_callback(self, fn: Callable, *args) -> None:
        """Invoke a callback; swallow + log exceptions so a buggy downstream
        handler doesn't crash the pipeline (per spec)."""
        try:
            fn(*args)
        except Exception as e:
            # In production, route to logger. Print to stderr for now.
            import sys
            print(f"[kiosk callback error] {type(e).__name__}: {e}", file=sys.stderr)
```

- [ ] **Step 4: Run init tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_pipeline.py::TestKioskPipelineInit -v`
Expected: 2 passed.

- [ ] **Step 5: Add tests for IDLE state wake detection + CAPTURING → ACTIVE_SESSION transition**

Append to `target-vad/tests/kiosk/test_pipeline.py`:

```python
class TestIdleAndCapturing:
    def test_idle_no_wake_stays_idle(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = None
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"

    def test_idle_wake_transitions_to_capturing(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "CAPTURING"

    def test_capturing_buffers_until_tail_reached(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → CAPTURING
        # Need 16000 samples (1.0s at 16kHz), feeding 480 at a time
        # 16000 / 480 = 33.33 → 34 chunks needed to first cross threshold
        for _ in range(33):
            p._handle_chunk(np.zeros(480, dtype=np.float32))
            assert p._state == "CAPTURING"
        # 34th chunk pushes past 16000, triggers session start
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "ACTIVE_SESSION"

    def test_session_start_invokes_on_session_started(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        started = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_started=started)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → CAPTURING
        # Push past tail buffer
        for _ in range(34):
            p._handle_chunk(np.zeros(480, dtype=np.float32))
        started.assert_called_once_with()

    def test_failed_snapshot_returns_to_idle(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        fake_embedder.extract.side_effect = RuntimeError("snapshot failed")
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → CAPTURING
        for _ in range(34):
            p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"
        assert p._session is None
```

- [ ] **Step 6: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_pipeline.py::TestIdleAndCapturing -v`
Expected: 5 passed.

- [ ] **Step 7: Add tests for ACTIVE_SESSION processing + smoothed callback firing**

Append to `target-vad/tests/kiosk/test_pipeline.py`:

```python
def make_segment(duration_ms: float = 1000.0) -> "SpeechSegment":
    from core.vad.silero_vad import SpeechSegment
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0,
        end_ms=duration_ms,
        duration_ms=duration_ms,
    )


def force_active_session(pipeline, fake_wake):
    """Helper: drive pipeline from IDLE through CAPTURING to ACTIVE_SESSION."""
    fake_wake.process.return_value = 0.87
    pipeline._handle_chunk(np.zeros(480, dtype=np.float32))  # → CAPTURING
    for _ in range(34):
        pipeline._handle_chunk(np.zeros(480, dtype=np.float32))
    assert pipeline._state == "ACTIVE_SESSION"


class TestActiveSession:
    def test_matched_segment_invokes_callback(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        # Embedder returns the same vector every call → cosine = 1.0 always → smoother fires
        on_primary = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        force_active_session(p, fake_wake)
        # During ACTIVE_SESSION, vad.process_chunk yields a segment each time
        fake_vad.process_chunk.return_value = [make_segment()]
        # Need 2 of 3 in window: feed 2 chunks
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        # On the second matched segment, smoother fires → callback invoked
        assert on_primary.called

    def test_unmatched_segment_does_not_invoke_callback(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        # Embedder returns an orthogonal vector → cosine ~0 → never crosses threshold
        on_primary = MagicMock()
        snapshot = np.ones(192, dtype=np.float32) / np.sqrt(192)
        orthogonal = np.zeros(192, dtype=np.float32)
        orthogonal[0] = 1.0
        # First call returns snapshot (during CAPTURING), then orthogonal forever
        fake_embedder.extract.side_effect = [snapshot] + [orthogonal] * 10
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        force_active_session(p, fake_wake)
        fake_vad.process_chunk.return_value = [make_segment()]
        for _ in range(5):
            p._handle_chunk(np.zeros(480, dtype=np.float32))
        on_primary.assert_not_called()

    def test_callback_exception_does_not_crash(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        on_primary = MagicMock(side_effect=RuntimeError("downstream broken"))
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        force_active_session(p, fake_wake)
        fake_vad.process_chunk.return_value = [make_segment()]
        # Should not raise
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "ACTIVE_SESSION"  # session continues
```

- [ ] **Step 8: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_pipeline.py::TestActiveSession -v`
Expected: 3 passed.

- [ ] **Step 9: Add tests for session timeouts and session-end callback**

Append to `target-vad/tests/kiosk/test_pipeline.py`:

```python
class TestSessionEnd:
    def test_silence_timeout_ends_session(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
        on_ended = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_ended=on_ended)
        # Drive into ACTIVE_SESSION at t=1000.0
        clock = [1000.0]
        monkeypatch.setattr("modes.kiosk.pipeline.time.monotonic", lambda: clock[0])
        force_active_session(p, fake_wake)
        assert p._state == "ACTIVE_SESSION"
        # Jump clock past silence timeout (10s)
        clock[0] = 1011.0
        fake_vad.process_chunk.return_value = []
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"
        on_ended.assert_called_once_with("silence_timeout")

    def test_hard_timeout_ends_session(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
        on_ended = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_ended=on_ended)
        clock = [1000.0]
        monkeypatch.setattr("modes.kiosk.pipeline.time.monotonic", lambda: clock[0])
        force_active_session(p, fake_wake)
        # Embedder always returns snapshot → matched segments keep silence_timer reset
        # but hard timeout wins
        clock[0] = 1301.0  # 301 s, past 300 s hard_timeout
        fake_vad.process_chunk.return_value = []
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"
        on_ended.assert_called_once_with("hard_timeout")

    def test_explicit_end_session_stopped_invokes_callback(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """The 'stopped' reason is emitted by run()'s finally clause when the loop exits.
        We test the unit (_end_session) directly rather than the threaded stop()
        interaction (which would require multi-thread test rigging for marginal value)."""
        on_ended = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_ended=on_ended)
        force_active_session(p, fake_wake)
        p._end_session("stopped")
        on_ended.assert_called_with("stopped")
        assert p._state == "IDLE"
        assert p._session is None
```

- [ ] **Step 10: Run new tests — should pass**

Run: `cd target-vad && py -3.14 -m pytest tests/kiosk/test_pipeline.py::TestSessionEnd -v`
Expected: 3 passed.

- [ ] **Step 11: Run full regression suite**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `55 passed` (44 + 11 new pipeline tests).

- [ ] **Step 12: Commit**

```bash
cd target-vad && git add modes/kiosk/pipeline.py tests/kiosk/test_pipeline.py && git commit -m "$(cat <<'EOF'
feat(kiosk): add KioskPipeline state machine

States: IDLE → CAPTURING → ACTIVE_SESSION → IDLE. In IDLE, frames feed
the wake detector. On wake, capture wake_capture_tail_seconds of audio
as the session-primary snapshot embedding. In ACTIVE_SESSION, route
through Silero VAD; score each segment via cosine vs snapshot, pass
through DecisionSmoother (M-of-N), invoke on_primary_speech callback
when smoother fires. Session ends on silence_timeout, hard_timeout, or
explicit stop(). Downstream callback exceptions are caught so a buggy
handler can't crash the kiosk. 11 tests with mocked I/O cover all
states and transitions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Implement `kiosk.py` CLI entry point

**Files:**
- Create: `target-vad/kiosk.py`

**No tests:** the CLI is pure orchestration over the well-tested KioskPipeline. Tests would be theater.

- [ ] **Step 1: Create the entry point**

Create `target-vad/kiosk.py` with:

```python
"""Kiosk talkback entry point — wake-word activated speaker-locked session."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse
import time

import numpy as np
import yaml
from rich.console import Console

from core.vad.silero_vad import SpeechSegment
from modes.kiosk.pipeline import KioskPipeline

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_dryrun_callbacks():
    """Print events to console; do not forward audio anywhere."""
    def on_primary_speech(segment: SpeechSegment, embedding: np.ndarray):
        console.print(
            f"[bold green][PRIMARY][/] {segment.duration_ms:.0f}ms "
            f"emb_norm={float(np.linalg.norm(embedding)):.3f}"
        )

    def on_session_started():
        console.print("[bold cyan][SESSION STARTED][/] Primary speaker locked")

    def on_session_ended(reason: str):
        console.print(f"[bold yellow][SESSION ENDED][/] reason={reason}\n")
        console.print('[dim][IDLE] Listening for wake phrase...[/]')

    return on_primary_speech, on_session_started, on_session_ended


def main():
    parser = argparse.ArgumentParser(description="Target VAD — Kiosk Talkback")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--wake-phrase",
        help="Override wake phrase (default from config). Bundled options: hey_jarvis, alexa, hey_mycroft.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print events instead of forwarding to a real downstream handler.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.wake_phrase:
        config["kiosk"]["wake_phrase"] = args.wake_phrase

    if args.dry_run:
        on_primary, on_started, on_ended = make_dryrun_callbacks()
    else:
        # No real downstream handler is configured yet — fall back to dry-run
        # behavior with a warning.
        console.print(
            "[yellow]No downstream handler configured. Running in dry-run mode.[/]"
        )
        on_primary, on_started, on_ended = make_dryrun_callbacks()

    console.print(
        f"[bold][IDLE][/] Listening for [bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
    )
    pipeline = KioskPipeline(
        config=config,
        on_primary_speech=on_primary,
        on_session_started=on_started,
        on_session_ended=on_ended,
    )
    try:
        pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check the file**

Run: `cd target-vad && py -3.14 -c "import ast; ast.parse(open('kiosk.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify CLI help works (no mic interaction)**

Run: `cd target-vad && py -3.14 kiosk.py --help`
Expected: argparse help text printed, exit code 0.

- [ ] **Step 4: Run full regression suite**

Run: `cd target-vad && py -3.14 -m pytest -q`
Expected: `55 passed`. (Adding kiosk.py doesn't add or break any tests.)

- [ ] **Step 5: Commit**

```bash
cd target-vad && git add kiosk.py && git commit -m "$(cat <<'EOF'
feat(kiosk): add kiosk.py CLI entry point

Args: --config, --wake-phrase override, --dry-run. Wires up
KioskPipeline with rich-formatted printer callbacks. No real downstream
handler exists yet so all modes effectively run as dry-run with an info
message. Manual smoke test next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final verification + manual smoke test

**Files:** none modified; verification and documentation only.

- [ ] **Step 1: Final automated test sweep**

Run: `cd target-vad && py -3.14 -m pytest -v`
Expected: 55 tests passing — 23 original + 3 new VAD chunk-API + 9 smoother + 3 session + 6 wake-word + 11 pipeline. No skips that weren't skipped originally.

- [ ] **Step 2: Verify file tree**

Run: `cd target-vad && ls modes/kiosk core/speaker tests/core tests/kiosk`
Expected:
- `modes/kiosk/`: `__init__.py`, `pipeline.py`, `session.py`, `wake_word.py`
- `core/speaker/`: `__init__.py`, `decision_smoother.py`, `embedder.py`, `enrollment_store.py`, `verifier.py`
- `tests/core/`: `__init__.py`, `test_decision_smoother.py`
- `tests/kiosk/`: `__init__.py`, `test_pipeline.py`, `test_session.py`, `test_wake_word.py`

- [ ] **Step 3: Verify the import graph is clean**

Run: `cd target-vad && py -3.14 -c "from modes.kiosk.pipeline import KioskPipeline; print('imports OK')"`
Expected: `imports OK`. (This pulls in openwakeword, ECAPA, Silero, etc. — first run may take several seconds.)

- [ ] **Step 4: Manual smoke test (USER drives this — not for an agent)**

Tell the user:

> The kiosk pipeline is now in place. To smoke-test it live:
>
> ```
> cd c:\repos\TVAD\target-vad
> py -3.14 kiosk.py --dry-run
> ```
>
> Wait for the `[IDLE] Listening for "hey_jarvis"...` prompt. Then say **"Hey Jarvis"** in your normal speaking voice. Expected sequence:
>
> 1. `[SESSION STARTED] Primary speaker locked`
> 2. `[PRIMARY] {duration}ms emb_norm=1.000` for each segment of your continued speech
> 3. After 10 s of silence: `[SESSION ENDED] reason=silence_timeout`
> 4. Returns to `[IDLE] Listening...`
>
> Things to specifically test:
> - Say "Hey Jarvis" then keep talking — primary segments should fire.
> - Say "Hey Jarvis" then have someone else speak (or play a recording of someone else) — their segments should NOT fire as primary (will be silently ignored by the smoother).
> - Say "Hey Jarvis", then go silent — session should end after ~10 s with `silence_timeout`.
> - Hit Ctrl+C — should exit cleanly.
>
> If wake-word isn't firing: lower `kiosk.wake_threshold` in `config.yaml` (try 0.4, then 0.3) and restart. The C10's noise suppression may be munching the wake phrase.

- [ ] **Step 5: No commit needed for verification task**

If the user reports the manual smoke test results, follow up with any tuning commits as needed (separate from this plan).

---

## Self-review notes

**Spec coverage:**
- Purpose / state machine / pipeline → Task 7 (pipeline.py)
- Components table → Tasks 4 (DecisionSmoother), 5 (Session), 6 (WakeWordDetector), 7 (KioskPipeline), 8 (kiosk.py)
- KioskPipeline interface signature → Task 7 step 3 matches the spec exactly
- CLI flags → Task 8 covers --config, --wake-phrase, --dry-run; **--log is intentionally NOT implemented** because the structured event logger is deferred to a later mode that consumes it (per shared spec's YAGNI deferral)
- Configuration block → Task 2
- Error handling → Task 7 covers callback resilience and snapshot failure; mic disconnection and openwakeword model-missing are inherited from the underlying libraries (will surface as exceptions during init or run)
- Testing approach → Tasks 4–7 cover all unit tests called out in the spec; the integration test ("test_end_to_end.py.skip") becomes Task 9's manual smoke test, not an automated test

**Placeholders:** none. Every step has concrete code or commands.

**Type/name consistency:**
- `DecisionSmoother(window_size, min_matches, threshold)` used the same way in tests and KioskPipeline init.
- `WakeWordDetector(model_name, threshold)` matches between test and pipeline.
- `Session(primary_embedding, smoother, started_at, last_speech_at)` consistent.
- `KioskPipeline.run()`, `.stop()`, `_handle_chunk()`, `_state` strings all consistent.

**Known deferrals (out of scope per spec):**
- `--log` flag and structured event logging
- Custom wake-phrase training
- Variant B (auth via enrolled voiceprint match)
- Downstream STT/LLM/TTS — `--dry-run` is the only working mode
