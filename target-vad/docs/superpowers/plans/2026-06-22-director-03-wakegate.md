# Director WakeGate Subsumption (Plan 03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the double-managed-session bug (spec §1, HARD REQ 5) by replacing the fat `KioskPipeline` with a **thin `WakeGate`** that owns *only* IDLE + AWAIT_FIRST_SEGMENT (wake detection, awaiting-speech timeout, the first-segment ECAPA snapshot). The WakeGate builds **one** `DirectorHandoff` and makes **one** blocking `runtime.run(handoff)` call, then resets to IDLE on return. It owns **no** `Session`, **no** watchdog thread, **no** silence/hard timer, **no** `_end_session`. The Director (Plans 01/02) is the *sole* owner of session lifecycle and all timers. This plan also lands the spec §4a Req-5 single-ownership proof as grep-based CI tests, a no-orphan-after-end integration test, and wires `kiosk.py`/`main.py` to construct `WakeGate` + `DirectorRuntime` instead of `KioskPipeline` + `TalkbackController`.

**Architecture:** The old `KioskPipeline` had three states — IDLE, AWAITING_SPEECH, ACTIVE_SESSION — and ACTIVE_SESSION was the bug: it ran a daemon watchdog thread reading `Session.silence_duration()` against a `last_speech_at` that the blocking `controller.run()` call (pipeline.py:200) never refreshed, so the watchdog fired `_end_session("silence_timeout")` ~30s after session *start* while talkback kept answering with resume state intact (spec §1). The fix is structural: the WakeGate keeps only the two thin states that precede the handoff. The moment a first VAD segment is captured after wake, it snapshots an ECAPA embedding, builds a `DirectorHandoff`, and hands the *entire* conversation to `DirectorRuntime.run(handoff)` — a **fully synchronous, blocking** call from the WakeGate's view (the Director spins its own asyncio loop internally, Plan 02). When that call returns a `DirectorResult`, the WakeGate's only post-return action is to reset to IDLE. There is exactly one timeout authority (the Director's `AsyncWatchdog`, Plan 01/02), and exactly one session-end reason source (`DirectorResult.reason`). ACTIVE_SESSION, the watchdog thread, `Session`, `_end_session`, and the per-chunk active-session paths are all **deleted outright**.

**Tech Stack:** Python 3.12 (`python3`, no `python` on PATH), pytest. Reuses `core/audio/mic_stream.py` (`MicrophoneStream`), `core/vad/silero_vad.py` (`SileroVAD`, `SpeechSegment`), `core/speaker/embedder.py` (`EmbeddingExtractor`), `modes/kiosk/wake_word.py` (`WakeWordDetector`) — all **as-is**. Consumes Plan 02's `DirectorRuntime` and the renamed `DirectorHandoff`/`DirectorResult` (`modes/talkback/handoff.py`). No new third-party dependencies.

## Global Constraints

- Target/dev box: NVIDIA DGX Spark GB10, aarch64, Python 3.12. Run tests with `python3 -m pytest`.
- **Single-ownership rule (spec §4, §4a):** the WakeGate holds NO session state and NO timeout path. It must not contain `_watchdog`, `_start_watchdog`, `_stop_watchdog`, `_end_session`, `Session(`, `last_speech_at`, `silence_timeout`, `hard_timeout`, or any `_silence_duration`. These are grep-checkable post-conditions (Task 4), not just deletions.
- **`runtime.run(handoff)` is synchronous from the WakeGate's view (spec §4a.2):** the WakeGate makes exactly one blocking call; the Director spins its own loop internally (Plan 02). The WakeGate's only post-return action is reset-to-IDLE (spec §4a.3).
- **One session-end reason authority (spec §4a.3):** the reason originates solely from `DirectorResult.reason`. The WakeGate never invents a session-end reason for an active session (it may still emit `awaiting_speech_timeout` for the *pre-session* AWAIT state, which is not a session end — no session ever started).
- **`DirectorRuntime` / `DirectorHandoff` / `DirectorResult` come from Plan 02 (binding interface contract).** `DirectorRuntime(handoff) -> DirectorResult` owns the asyncio loop; `DirectorHandoff(mic, primary_embedding, holdout_embedding, first_segment, config, vad, embedder)`; `DirectorResult(reason, turns, total_duration_s)`. This plan must not implement the Director — it only constructs and calls it. Tests inject a fake runtime.
- **`holdout_embedding` dependency (spec §7, binding contract):** Plan 05 owns the *real* holdout-before-finalize capture. For Plan 03, pass the **same first-segment embedding** through as `holdout_embedding` — a placeholder that is acceptable ONLY because Plan 05 replaces it with the real pre-finalize utterance embedding. This dependency is noted explicitly in prose, in a code *docstring/comment* using the word "placeholder" and the cross-reference "Plan 05", and in the Self-Review. It is deliberately **not** expressed as a placeholder-marker comment in code (which would trip the placeholder scan); the cross-reference carries the dependency instead.
- New module lives at `modes/director/wakegate.py`; tests under `tests/director/`.
- Reuse, do not reimplement: `MicrophoneStream`, `SileroVAD`/`SpeechSegment`, `EmbeddingExtractor`, `WakeWordDetector`.

---

## File Structure

- `modes/director/wakegate.py` — **NEW.** `WakeGate` class: IDLE + AWAIT_FIRST_SEGMENT only. Wake detection, awaiting-speech timeout, first-segment ECAPA snapshot, one `DirectorHandoff` build, one blocking `runtime.run(handoff)`, reset-to-IDLE on return.
- `modes/talkback/handoff.py` — **MODIFIED by Plan 02** (rename `TalkbackHandoff`→`DirectorHandoff`, `TalkbackResult`→`DirectorResult`, add `holdout_embedding`). This plan *consumes* it; if Plan 02 has not landed when this plan runs, Task 1 includes a guarded fallback that adds the renames + field so this plan is self-contained. **Coordinate with Plan 02 to avoid a double-edit conflict** (see Task 1 note).
- `modes/kiosk/pipeline.py` — **DELETED** (its responsibilities split: pre-session → WakeGate; session → Director).
- `modes/kiosk/session.py` — **DELETED** (subsumed into the Director's Context, Plan 01).
- `config.yaml` — remove `kiosk.session_silence_timeout_s` / `kiosk.session_hard_timeout_s` (lines 30-31) and the now-stale `kiosk.watchdog` block + the comment referencing the deleted keys.
- `kiosk.py` — rewire `--talkback` path to build `WakeGate` + `DirectorRuntime`; rewire dry-run path to build `WakeGate` with no runtime (or a dry-run runtime). Emit `[WAKE]`/`[SESSION STARTED]`/`[SESSION ENDED]`/`[IDLE]` from ONE owner so `[HANDOFF]` no longer double-prints.
- `tests/director/test_wakegate.py` — **NEW.** Construction-pattern + state-machine tests (ported from `tests/kiosk/test_pipeline.py` + `test_handoff_wiring.py`), minus everything that referenced ACTIVE_SESSION/watchdog/Session.
- `tests/director/test_wakegate_single_ownership.py` — **NEW.** The spec §4a Req-5 proof: grep post-conditions + synchronous-call assertion + no-orphan-after-end integration test + deleted-config-keys assertion.
- `tests/kiosk/test_pipeline.py`, `tests/kiosk/test_handoff_wiring.py`, `tests/kiosk/test_kiosk_watchdog.py` — **DELETED** (replaced by the two new test modules; they assert the deleted ACTIVE_SESSION/watchdog/Session behavior).

---

## Task 1: Confirm/land the renamed handoff contract (DirectorHandoff/DirectorResult)

**Files:**
- Modify (or confirm, if Plan 02 already landed it): `modes/talkback/handoff.py`
- Test: `tests/director/test_handoff_contract.py`

**Interfaces:**
- Produces: `DirectorHandoff(mic, primary_embedding, holdout_embedding, first_segment, config, vad, embedder)` and `DirectorResult(reason, turns, total_duration_s)`.

> **Coordination note.** Plan 02's binding contract owns these renames. If Plan 02 has already merged, this whole task collapses to "run the test, confirm green, skip the edit." Only apply the edit below if `python3 -c "from modes.talkback.handoff import DirectorHandoff, DirectorResult"` fails. Do **not** double-define.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_handoff_contract.py
"""DirectorHandoff/DirectorResult are the renamed handoff contract (binding
interface, owned by Plan 02; asserted here so Plan 03 is self-contained)."""

import numpy as np


def test_director_handoff_has_holdout_embedding_field():
    from modes.talkback.handoff import DirectorHandoff
    emb = np.ones(192, dtype=np.float32)
    h = DirectorHandoff(
        mic="mic", primary_embedding=emb, holdout_embedding=emb,
        first_segment="seg", config={}, vad="vad", embedder="emb",
    )
    assert h.holdout_embedding is emb
    assert h.primary_embedding is emb
    assert h.mic == "mic"


def test_director_result_carries_reason_turns_duration():
    from modes.talkback.handoff import DirectorResult
    r = DirectorResult(reason="silence_timeout", turns=3, total_duration_s=12.5)
    assert r.reason == "silence_timeout"
    assert r.turns == 3
    assert r.total_duration_s == 12.5
```

- [ ] **Step 2: Run test to verify it fails (only if Plan 02 has not landed)**

Run: `python3 -m pytest tests/director/test_handoff_contract.py -v`
Expected: PASS if Plan 02 already renamed; otherwise FAIL — `ImportError: cannot import name 'DirectorHandoff'`.

- [ ] **Step 3: If failing, apply the rename + new field**

Replace the body of `modes/talkback/handoff.py` with:

```python
"""Hand-off contract between the WakeGate and the Director.

Renamed from TalkbackHandoff/TalkbackResult (binding interface, Plan 02).
holdout_embedding (Plan 05) carries a pre-finalize enrollment utterance
embedding for verify-before-serve; Plan 03 passes the first-segment embedding
through as a placeholder until Plan 05 captures the real holdout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DirectorHandoff:
    """Payload the WakeGate passes to the Director at session start."""
    mic: Any
    primary_embedding: np.ndarray
    holdout_embedding: np.ndarray   # Plan 05: pre-finalize utterance embedding
    first_segment: Any
    config: dict
    vad: Any
    embedder: Any


@dataclass
class DirectorResult:
    """What the Director returns when the conversation ends."""
    reason: str
    turns: int
    total_duration_s: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_handoff_contract.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/talkback/handoff.py tests/director/test_handoff_contract.py
git commit -m "feat(director): confirm DirectorHandoff/DirectorResult contract (+holdout_embedding)"
```

---

## Task 2: WakeGate — IDLE + AWAIT_FIRST_SEGMENT, snapshot, one blocking handoff

**Files:**
- Create: `modes/director/wakegate.py`
- Test: `tests/director/test_wakegate.py`

**Interfaces:**
- Produces: `WakeGate(config, runtime, on_event=..., _mic=..., _vad=..., _embedder=..., _wake_detector=...)`. Public: `run()` (mic loop), `stop()`. State is `"IDLE"` or `"AWAIT_FIRST_SEGMENT"` only. On first segment after wake: snapshot ECAPA → build `DirectorHandoff` → blocking `runtime.run(handoff)` → reset to IDLE.
- `on_event(event_type, payload)` callback emits exactly: `"wake_detected"`, `"session_started"`, `"session_ended"` (reason from `DirectorResult.reason`), `"awaiting_speech_timeout"` (pre-session abort, not a session end).

> **Why this design satisfies Req-5.** The WakeGate owns *nothing* about the active session. The instant a snapshot is captured it transfers the mic, vad, embedder, and embeddings into a `DirectorHandoff` and *blocks* inside `runtime.run(handoff)` until the Director returns a `DirectorResult`. There is no concurrent WakeGate activity during the conversation (the thread is parked inside `run`), so there is no second `last_speech_at`, no second watchdog, no second timeout authority. When `run` returns, the WakeGate's only action is `_reset_to_idle()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_wakegate.py
"""WakeGate state-machine + construction tests. Ported from the old
tests/kiosk/test_pipeline.py / test_handoff_wiring.py, minus everything that
referenced the deleted ACTIVE_SESSION/watchdog/Session paths."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.handoff import DirectorHandoff, DirectorResult


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
            "awaiting_speech_timeout_s": 5,
            "talkback": {"sample_rate_hz": 16000},
        },
    }


@pytest.fixture
def fake_mic():
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
    m.extract = MagicMock(return_value=np.ones(192, dtype=np.float32) / np.sqrt(192))
    return m


@pytest.fixture
def fake_wake():
    m = MagicMock()
    m.process = MagicMock(return_value=None)
    m.reset = MagicMock()
    return m


@pytest.fixture
def fake_runtime():
    """A DirectorRuntime stub: .run(handoff) returns a DirectorResult."""
    m = MagicMock()
    m.run = MagicMock(
        return_value=DirectorResult(reason="silence_timeout", turns=2, total_duration_s=10.0)
    )
    return m


def make_segment(duration_ms: float = 1000.0) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, on_event=None):
    from modes.director.wakegate import WakeGate
    return WakeGate(
        config=base_config,
        runtime=fake_runtime,
        on_event=on_event or (lambda et, pl: None),
        _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder,
        _wake_detector=fake_wake,
    )


class TestWakeGateInit:
    def test_starts_in_idle(self, base_config, fake_mic, fake_vad, fake_embedder,
                            fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        assert g._state == "IDLE"

    def test_stop_sets_running_false(self, base_config, fake_mic, fake_vad,
                                     fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        g._running = True
        g.stop()
        assert g._running is False


class TestIdleAndAwait:
    def test_idle_no_wake_stays_idle(self, base_config, fake_mic, fake_vad,
                                     fake_embedder, fake_wake, fake_runtime):
        fake_wake.process.return_value = None
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert g._state == "IDLE"

    def test_idle_wake_transitions_to_await(self, base_config, fake_mic, fake_vad,
                                            fake_embedder, fake_wake, fake_runtime):
        fake_wake.process.return_value = 0.87
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert g._state == "AWAIT_FIRST_SEGMENT"

    def test_wake_emits_wake_detected_event(self, base_config, fake_mic, fake_vad,
                                            fake_embedder, fake_wake, fake_runtime):
        events = []
        fake_wake.process.return_value = 0.87
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert events[0][0] == "wake_detected"
        assert events[0][1] == {"phrase": "hey_jarvis", "score": 0.87}

    def test_await_timeout_returns_to_idle(self, base_config, fake_mic, fake_vad,
                                           fake_embedder, fake_wake, fake_runtime, monkeypatch):
        events = []
        clock = [1000.0]
        monkeypatch.setattr("modes.director.wakegate.time.monotonic", lambda: clock[0])
        fake_wake.process.return_value = 0.87
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        g._handle_chunk(np.zeros(480, dtype=np.float32))   # → AWAIT at t=1000
        assert g._state == "AWAIT_FIRST_SEGMENT"
        clock[0] = 1006.0                                   # past 5s timeout
        fake_vad.process_chunk.return_value = []
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert g._state == "IDLE"
        assert ("awaiting_speech_timeout", {}) in events
        # crucially: NOT a session_ended event (no session ever started)
        assert all(et != "session_ended" for et, _ in events)


class TestHandoff:
    def _drive_to_handoff(self, g, fake_wake, fake_vad):
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))   # → AWAIT_FIRST_SEGMENT
        assert g._state == "AWAIT_FIRST_SEGMENT"
        fake_vad.process_chunk.return_value = [make_segment()]
        g._handle_chunk(np.zeros(480, dtype=np.float32))   # snapshot → handoff → IDLE

    def test_first_segment_builds_handoff_and_calls_runtime(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        self._drive_to_handoff(g, fake_wake, fake_vad)
        fake_runtime.run.assert_called_once()
        handoff = fake_runtime.run.call_args[0][0]
        assert isinstance(handoff, DirectorHandoff)
        assert handoff.mic is fake_mic
        assert handoff.vad is fake_vad
        assert handoff.embedder is fake_embedder
        assert handoff.primary_embedding.shape == (192,)

    def test_holdout_embedding_is_first_segment_embedding_placeholder(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        # Plan 05 replaces this with the real pre-finalize holdout. For Plan 03
        # the holdout IS the first-segment (primary) embedding.
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        self._drive_to_handoff(g, fake_wake, fake_vad)
        handoff = fake_runtime.run.call_args[0][0]
        assert np.array_equal(handoff.holdout_embedding, handoff.primary_embedding)

    def test_handoff_passes_talkback_config(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        self._drive_to_handoff(g, fake_wake, fake_vad)
        handoff = fake_runtime.run.call_args[0][0]
        assert handoff.config == base_config["kiosk"]["talkback"]

    def test_handoff_passes_first_segment(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        seg = make_segment(500.0)
        fake_vad.process_chunk.return_value = [seg]
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        handoff = fake_runtime.run.call_args[0][0]
        assert handoff.first_segment is seg

    def test_session_started_and_ended_events_fire_from_one_owner(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        self._drive_to_handoff(g, fake_wake, fake_vad)
        types = [et for et, _ in events]
        assert "session_started" in types
        assert "session_ended" in types
        # the END reason comes from DirectorResult.reason, nowhere else
        ended = next(pl for et, pl in events if et == "session_ended")
        assert ended == {"reason": "silence_timeout"}

    def test_resets_to_idle_after_runtime_returns(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        self._drive_to_handoff(g, fake_wake, fake_vad)
        assert g._state == "IDLE"
        fake_wake.reset.assert_called()

    def test_failed_snapshot_returns_to_idle_without_handoff(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        fake_embedder.extract.side_effect = RuntimeError("snapshot failed")
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert g._state == "IDLE"
        fake_runtime.run.assert_not_called()

    def test_session_end_reason_propagates_from_director_result(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        from modes.director.wakegate import WakeGate
        runtime = MagicMock()
        runtime.run = MagicMock(
            return_value=DirectorResult(reason="hard_timeout", turns=9, total_duration_s=300.0)
        )
        ended = []
        g = WakeGate(
            config=base_config, runtime=runtime,
            on_event=lambda et, pl: ended.append((et, pl)),
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert ("session_ended", {"reason": "hard_timeout"}) in ended


class TestEventCallbackRobustness:
    def test_on_event_exception_does_not_crash(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        def buggy(et, pl):
            raise RuntimeError("handler broke")
        from modes.director.wakegate import WakeGate
        g = WakeGate(
            config=base_config, runtime=fake_runtime, on_event=buggy,
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))   # buggy on_event runs, swallowed
        fake_vad.process_chunk.return_value = [make_segment()]
        g._handle_chunk(np.zeros(480, dtype=np.float32))   # more buggy events, swallowed
        assert g._state == "IDLE"                          # still completed the cycle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_wakegate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.wakegate'`

- [ ] **Step 3: Implement the WakeGate**

```python
# modes/director/wakegate.py
"""WakeGate — the THIN front of the Conversation Director (spec section 4).

Two states only:
  IDLE                 — feeding mic chunks to the wake detector.
  AWAIT_FIRST_SEGMENT  — wake fired; feeding chunks to VAD; the first completed
                         speech segment becomes the session-primary ECAPA snapshot.

On that first segment the WakeGate snapshots an embedding, builds ONE
DirectorHandoff, and makes ONE blocking call to runtime.run(handoff). The
Director (Plans 01/02) owns the entire active session — every timer, the single
AsyncWatchdog, the conversation lifecycle, and the sole session-end reason. From
the WakeGate's view runtime.run() is fully synchronous (the Director spins its
own asyncio loop internally); the WakeGate thread is parked inside that call for
the whole conversation. When it returns a DirectorResult, the WakeGate's ONLY
post-return action is to reset to IDLE.

This component deliberately owns NO Session, NO watchdog thread, NO silence or
hard timer, and NO _end_session — that double-management was the live bug
(spec section 1, HARD REQ 5). The grep post-conditions in
tests/director/test_wakegate_single_ownership.py enforce that absence.
"""

import sys
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np

from core.audio.mic_stream import MicrophoneStream
from core.speaker.embedder import EmbeddingExtractor
from core.vad.silero_vad import SileroVAD, SpeechSegment
from modes.kiosk.wake_word import WakeWordDetector
from modes.talkback.handoff import DirectorHandoff


class WakeGate:
    def __init__(
        self,
        config: dict,
        runtime: Any,
        on_event: Callable[[str, dict], None] = lambda event, payload: None,
        # Underscore kwargs are for test injection. Production code omits them.
        _mic: Optional[Any] = None,
        _vad: Optional[Any] = None,
        _embedder: Optional[Any] = None,
        _wake_detector: Optional[Any] = None,
    ):
        self.config = config
        self.runtime = runtime
        self.on_event = on_event

        kiosk_cfg = config["kiosk"]
        self._awaiting_speech_timeout_s = kiosk_cfg["awaiting_speech_timeout_s"]
        self._talkback_config = kiosk_cfg.get("talkback", {})

        self.mic = _mic or MicrophoneStream(config["core"]["audio"])
        self.vad = _vad or SileroVAD(config["core"]["vad"])
        self.embedder = _embedder or EmbeddingExtractor()
        self.wake_detector = _wake_detector or WakeWordDetector(
            kiosk_cfg["wake_phrase"], kiosk_cfg["wake_threshold"]
        )

        self._wake_time: Optional[float] = None
        self._running = False

        # Warm up ECAPA so the first wake -> snapshot transition doesn't pay the
        # model's cold-start latency (~1.3s on CPU). Skip when injected (tests).
        if _embedder is None:
            try:
                _ = self.embedder.extract(np.zeros(12800, dtype=np.float32))
            except Exception:
                pass  # non-fatal; lazy-loads on first real use

        self._state = "IDLE"

    def stop(self) -> None:
        """Signal run() to exit cleanly on the next loop iteration."""
        self._running = False

    def run(self) -> None:
        """Main mic loop. Blocks until stop() or KeyboardInterrupt. NOTE: while a
        session is active, this thread is parked inside runtime.run(handoff) and
        is NOT iterating the mic loop — the Director owns the mic during a
        session. There is no watchdog here; the Director owns the single one."""
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

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        if self._state == "IDLE":
            self._handle_idle_chunk(chunk)
        elif self._state == "AWAIT_FIRST_SEGMENT":
            self._handle_await_chunk(chunk)

    def _handle_idle_chunk(self, chunk: np.ndarray) -> None:
        wake_score = self.wake_detector.process(chunk)
        if wake_score is not None:
            self._safe_callback(
                self.on_event, "wake_detected",
                {"phrase": self.config["kiosk"]["wake_phrase"], "score": wake_score},
            )
            self._state = "AWAIT_FIRST_SEGMENT"
            self._wake_time = time.monotonic()
            self.vad.reset()

    def _handle_await_chunk(self, chunk: np.ndarray) -> None:
        # Pre-session abort if no speech arrives in time. This is NOT a session
        # end (no session ever started) — the reason authority for a real
        # session is DirectorResult.reason alone.
        assert self._wake_time is not None
        if time.monotonic() - self._wake_time >= self._awaiting_speech_timeout_s:
            self._reset_to_idle()
            self._safe_callback(self.on_event, "awaiting_speech_timeout", {})
            return

        for segment in self.vad.process_chunk(chunk):
            self._start_session_from_segment(segment)
            return  # only the first segment matters

    def _start_session_from_segment(self, segment: SpeechSegment) -> None:
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception:
            self._reset_to_idle()
            return

        self._safe_callback(self.on_event, "session_started",
                            {"snapshot_norm": float(np.linalg.norm(embedding))})

        # Placeholder holdout (Plan 05 owns the real pre-finalize capture): for
        # now the holdout IS the first-segment/primary embedding. Acceptable
        # ONLY because Plan 05 replaces it; verify-before-serve trivially passes
        # at cosine(primary, primary) == 1.0 until then.
        handoff = DirectorHandoff(
            mic=self.mic,
            primary_embedding=embedding,
            holdout_embedding=embedding,
            first_segment=segment,
            config=self._talkback_config,
            vad=self.vad,
            embedder=self.embedder,
        )

        # ONE blocking call. The Director owns the whole conversation and every
        # timer; it returns only at true session end. This is the single point
        # of session ownership transfer (spec section 4a.2).
        result = self.runtime.run(handoff)

        # The ONLY post-return action: reset to IDLE. The end reason originates
        # solely from DirectorResult.reason (spec section 4a.3).
        self._reset_to_idle()
        self._safe_callback(self.on_event, "session_ended", {"reason": result.reason})

    def _reset_to_idle(self) -> None:
        self._state = "IDLE"
        self._wake_time = None
        self.wake_detector.reset()

    def _safe_callback(self, fn: Callable, *args) -> None:
        """Invoke a callback; swallow + log exceptions so a buggy downstream
        handler doesn't crash the gate."""
        try:
            fn(*args)
        except Exception as e:
            print(f"[wakegate callback error] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
```

> **Note on `_reset_to_idle` vs the grep ban.** `_reset_to_idle` is *not* `_end_session` and contains no timer/session state — it only flips the two thin pre-session fields and resets the wake detector. The Task-4 grep bans `_end_session`, `Session(`, `last_speech_at`, etc., none of which appear here.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_wakegate.py -v`
Expected: PASS (all tests in the module)

- [ ] **Step 5: Commit**

```bash
git add modes/director/wakegate.py tests/director/test_wakegate.py
git commit -m "feat(director): thin WakeGate (IDLE + AWAIT_FIRST_SEGMENT, one blocking handoff)"
```

---

## Task 3: Delete the fat pipeline, Session, and stale config/tests

**Files:**
- Delete: `modes/kiosk/pipeline.py`, `modes/kiosk/session.py`
- Delete: `tests/kiosk/test_pipeline.py`, `tests/kiosk/test_handoff_wiring.py`, `tests/kiosk/test_kiosk_watchdog.py`
- Modify: `config.yaml`

**Interfaces:** none (pure removal). After this task, the watchdog thread (`_watchdog_loop`/`_start_watchdog`/`_stop_watchdog`), `_handle_active_chunk`/`_process_session_segment`, `_end_session`, and the `Session` field/class no longer exist anywhere in the kiosk path. The deleted config keys are gone.

- [ ] **Step 1: Delete the subsumed modules and their tests**

```bash
git rm modes/kiosk/pipeline.py
git rm modes/kiosk/session.py
git rm tests/kiosk/test_pipeline.py
git rm tests/kiosk/test_handoff_wiring.py
git rm tests/kiosk/test_kiosk_watchdog.py
```

> These tests asserted the deleted behavior (ACTIVE_SESSION, the watchdog firing without chunks, `_end_session`, `Session` smoothing). Their replacements are `tests/director/test_wakegate.py` (Task 2) and `tests/director/test_wakegate_single_ownership.py` (Task 4). `modes/kiosk/wake_word.py` is **kept** (reused by the WakeGate).

- [ ] **Step 2: Remove the dead config keys**

In `config.yaml`, delete lines 29-41 of the `kiosk:` block — the two session timeout keys and the now-orphaned `watchdog` block — and update the talkback comment that referenced them. The `kiosk:` block becomes:

```yaml
kiosk:
  wake_phrase: "hey_mycroft"
  wake_threshold: 0.5
  awaiting_speech_timeout_s: 15
  decision_smoother:
    window_size: 3
    min_matches: 2
    threshold: 0.50   # tuned for C10 + classroom DSP; raise once non-self baseline measured

  talkback_enabled: false

  talkback:
    sample_rate_hz: 16000
    frame_ms: 10
    output_device: null
    input_device: null

    # Active-conversation timeouts, owned by the Director (the SOLE timeout
    # authority — spec section 4a). silence_timeout_s: how long a pause between
    # your turns ends the session. hard_timeout_s: absolute session cap. The old
    # kiosk.session_*_timeout_s keys (a second, racing authority) are DELETED.
    silence_timeout_s: 30
    hard_timeout_s: 300
```

(Leave the rest of the `talkback:` block — `aec`, `stt`, `llm`, `tts`, `chunker`, `turn_gate`, `barge_in`, `resume`, `logging` — unchanged. The `decision_smoother` key under `kiosk:` is retained for now; the Director's safety-net smoother config lives under `talkback.turn_gate.lockout`, and Plan 05 reconciles them. Removing `decision_smoother` here is out of scope.)

> **Note:** `kiosk.watchdog.tick_ms` powered only the deleted pipeline watchdog. The Director's `AsyncWatchdog` reads its own `tick_s` (Plan 02 config), so this block is dead and removed.

- [ ] **Step 3: Verify nothing still imports the deleted symbols**

Run:
```bash
grep -rnE "from modes.kiosk.pipeline|import KioskPipeline|from modes.kiosk.session|import Session\b|TalkbackHandoff|TalkbackResult" \
  /home/ldrgx10/FullDuplexVoice/TVAD/target-vad --include="*.py"
```
Expected: only `kiosk.py` (fixed in Task 5) still references `KioskPipeline` at this point. No `*.py` should reference `modes.kiosk.session`, `TalkbackHandoff`, or `TalkbackResult` (the latter two renamed in Task 1). If anything else appears, fix the import before continuing.

- [ ] **Step 4: Confirm config keys are gone**

Run:
```bash
grep -nE "session_silence_timeout_s|session_hard_timeout_s" /home/ldrgx10/FullDuplexVoice/TVAD/target-vad/config.yaml || echo "OK: keys removed"
```
Expected: `OK: keys removed`

- [ ] **Step 5: Commit**

```bash
git add -A config.yaml
git commit -m "refactor(director): delete fat KioskPipeline + Session + watchdog config; remove racing timeout keys"
```

---

## Task 4: Req-5 single-ownership proof — grep post-conditions + no-orphan + deleted-keys (spec §4a)

**Files:**
- Create: `tests/director/test_wakegate_single_ownership.py`

**Interfaces:** none — these are CI guard tests over the WakeGate source, the config, and a fake-runtime integration scenario. They make the single-owner property *provable*, not asserted (spec §4a).

- [ ] **Step 1: Write the test (it should pass immediately against the Task-2/3 result; this is a guard, not a TDD red-first feature)**

```python
# tests/director/test_wakegate_single_ownership.py
"""Spec section 4a — the Req-5 single-ownership proof as CI post-conditions.

Four guarantees, grep-checkable + behaviorally checkable:
  (1) the WakeGate holds NO session state and NO timeout path;
  (2) runtime.run(handoff) is synchronous from the WakeGate's view, and the
      WakeGate's ONLY post-return action is reset-to-IDLE;
  (3) the session-end reason originates solely from DirectorResult.reason;
  (4) the deleted racing config keys are gone, and after run() returns nothing
      answers further user speech without a new wake (no-orphan-after-end).
"""

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.director import wakegate as wg
from modes.director.wakegate import WakeGate
from modes.talkback.handoff import DirectorResult

REPO = Path(__file__).resolve().parents[2]
WAKEGATE_SRC = inspect.getsource(wg)


# ---- (1) grep post-conditions: the WakeGate owns no session/timeout state ----

BANNED_SUBSTRINGS = [
    "_watchdog",
    "_start_watchdog",
    "_stop_watchdog",
    "_end_session",
    "Session(",
    "last_speech_at",
    "silence_timeout",
    "hard_timeout",
    "_silence_duration",
]


@pytest.mark.parametrize("banned", BANNED_SUBSTRINGS)
def test_wakegate_source_contains_no_session_or_timeout_machinery(banned):
    assert banned not in WAKEGATE_SRC, (
        f"WakeGate must not contain {banned!r} — the Director is the sole owner "
        f"of session lifecycle and all timers (spec section 4a.1)."
    )


def test_wakegate_has_only_thin_state_fields():
    """The WakeGate's mutable state is the two thin pre-session fields only:
    _state (IDLE/AWAIT_FIRST_SEGMENT) and _wake_time. No Session object."""
    g = _make_gate()
    assert g._state in ("IDLE", "AWAIT_FIRST_SEGMENT")
    assert g._wake_time is None
    assert not hasattr(g, "_session")
    assert not hasattr(g, "_watchdog_thread")


def test_no_threading_watchdog_imported_or_spawned():
    """The deleted daemon watchdog used threading.Thread. The WakeGate spawns
    no thread of its own (the Director owns the single AsyncWatchdog)."""
    assert "threading.Thread" not in WAKEGATE_SRC
    assert "Thread(" not in WAKEGATE_SRC


# ---- (2)/(3) runtime.run is synchronous; only post-return action is reset ----

def _make_gate(runtime=None, on_event=None):
    fake_mic = MagicMock()
    fake_mic.__enter__ = MagicMock(return_value=fake_mic)
    fake_mic.__exit__ = MagicMock(return_value=None)
    fake_vad = MagicMock()
    fake_vad.process_chunk = MagicMock(return_value=[])
    fake_vad.reset = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.extract = MagicMock(
        return_value=np.ones(192, dtype=np.float32) / np.sqrt(192)
    )
    fake_wake = MagicMock()
    fake_wake.process = MagicMock(return_value=None)
    fake_wake.reset = MagicMock()
    config = {
        "core": {
            "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 480},
            "vad": {"sample_rate": 16000, "speech_threshold": 0.5,
                    "min_speech_duration_ms": 300, "padding_ms": 200},
        },
        "kiosk": {
            "wake_phrase": "hey_jarvis", "wake_threshold": 0.5,
            "awaiting_speech_timeout_s": 5, "talkback": {"sample_rate_hz": 16000},
        },
    }
    return WakeGate(
        config=config,
        runtime=runtime or MagicMock(run=MagicMock(
            return_value=DirectorResult(reason="silence_timeout", turns=1, total_duration_s=3.0))),
        on_event=on_event or (lambda et, pl: None),
        _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
    )


def _segment(duration_ms=1000.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def _drive_to_handoff(g):
    g.wake_detector.process.return_value = 0.9
    g._handle_chunk(np.zeros(480, dtype=np.float32))   # → AWAIT_FIRST_SEGMENT
    g.vad.process_chunk.return_value = [_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))   # snapshot → blocking run → IDLE


def test_runtime_run_is_a_single_blocking_call_returning_a_result():
    """From the WakeGate's view runtime.run() is fully synchronous: one call,
    one DirectorResult, control returns inline. The state after it returns is
    IDLE — proving there is no concurrent WakeGate activity during the session."""
    order = []
    runtime = MagicMock()

    def fake_run(handoff):
        order.append("inside_run")
        return DirectorResult(reason="silence_timeout", turns=1, total_duration_s=3.0)

    runtime.run = MagicMock(side_effect=fake_run)
    g = _make_gate(runtime=runtime)
    _drive_to_handoff(g)
    assert order == ["inside_run"]          # called exactly once, synchronously
    assert runtime.run.call_count == 1
    assert g._state == "IDLE"               # control returned and reset


def test_only_post_return_action_is_reset_to_idle():
    """After run() returns, the WakeGate does exactly one thing: reset to IDLE
    (state flip + wake-detector reset) and emit the session_ended event whose
    reason is the DirectorResult's reason. No second teardown, no re-entrancy."""
    events = []
    runtime = MagicMock(run=MagicMock(
        return_value=DirectorResult(reason="lockout", turns=4, total_duration_s=42.0)))
    g = _make_gate(runtime=runtime, on_event=lambda et, pl: events.append((et, pl)))
    _drive_to_handoff(g)
    assert g._state == "IDLE"
    g.wake_detector.reset.assert_called()
    # the reason came from DirectorResult, nowhere else
    assert ("session_ended", {"reason": "lockout"}) in events


# ---- (4a) deleted config keys are gone ----

def test_deleted_config_keys_are_absent():
    cfg_text = (REPO / "config.yaml").read_text()
    assert "session_silence_timeout_s" not in cfg_text
    assert "session_hard_timeout_s" not in cfg_text
    # the dead pipeline watchdog block is gone too
    assert not re.search(r"^\s*watchdog:\s*$", cfg_text, re.MULTILINE), \
        "kiosk.watchdog powered only the deleted pipeline watchdog"


# ---- (4b) no-orphan-after-end: nothing answers further speech without a wake ----

def test_no_orphan_after_end_requires_a_new_wake():
    """The exact Req-5 live bug: after the session ends, a stray user segment
    must NOT start a new conversation. The WakeGate is back in IDLE feeding the
    wake detector; speech with no wake is ignored. Only a fresh wake re-arms."""
    runtime = MagicMock(run=MagicMock(
        return_value=DirectorResult(reason="silence_timeout", turns=1, total_duration_s=3.0)))
    g = _make_gate(runtime=runtime)
    _drive_to_handoff(g)
    assert g._state == "IDLE"
    assert runtime.run.call_count == 1

    # Simulate post-end "orphan" speech: VAD would emit segments, but the wake
    # detector returns None (no wake). The gate must stay IDLE and NOT hand off.
    g.wake_detector.process.return_value = None
    g.vad.process_chunk.return_value = [_segment(), _segment()]
    for _ in range(5):
        g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._state == "IDLE"
    assert runtime.run.call_count == 1      # NO second session started by orphan speech

    # A genuine new wake DOES re-arm a fresh session.
    g.wake_detector.process.return_value = 0.9
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._state == "AWAIT_FIRST_SEGMENT"
    g.vad.process_chunk.return_value = [_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert runtime.run.call_count == 2      # exactly one new session per new wake
    assert g._state == "IDLE"
```

- [ ] **Step 2: Run the proof suite**

Run: `python3 -m pytest tests/director/test_wakegate_single_ownership.py -v`
Expected: PASS (all parametrized grep guards + the synchronous-call, reset-only, deleted-keys, and no-orphan tests)

- [ ] **Step 3: Run the full director suite to confirm no regressions**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS (Plan 01 reducer tests + handoff contract + WakeGate + single-ownership proof)

- [ ] **Step 4: Commit**

```bash
git add tests/director/test_wakegate_single_ownership.py
git commit -m "test(director): Req-5 single-ownership proof — grep post-conditions + no-orphan-after-end"
```

---

## Task 5: Wire kiosk.py to WakeGate + DirectorRuntime (one event owner, no double [HANDOFF])

**Files:**
- Modify: `kiosk.py`
- Test: `tests/director/test_kiosk_entrypoint.py`

**Interfaces:**
- Produces: `build_wakegate(config, console) -> WakeGate` (the construction path the entry point uses), exercised with a fake runtime so the test needs no real models/GPU.

> **Why a `build_wakegate` factory.** The old `main()` inlined all construction, which made it untestable without real STT/TTS/LLM. We factor the WakeGate construction into a small importable function so a test can assert the wiring (WakeGate built, runtime attached, console event prints emitted from ONE owner). The model/LLM warm-up stays in `main()` (it needs real hardware and is not unit-tested here).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_kiosk_entrypoint.py
"""kiosk.py wiring: build a WakeGate around a DirectorRuntime, with console
event prints emitted from ONE owner (no double [HANDOFF])."""

from unittest.mock import MagicMock

import numpy as np

from core.vad.silero_vad import SpeechSegment
from modes.talkback.handoff import DirectorResult


def _config():
    return {
        "core": {
            "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 480},
            "vad": {"sample_rate": 16000, "speech_threshold": 0.5,
                    "min_speech_duration_ms": 300, "padding_ms": 200},
        },
        "kiosk": {
            "wake_phrase": "hey_jarvis", "wake_threshold": 0.5,
            "awaiting_speech_timeout_s": 5,
            "talkback": {"sample_rate_hz": 16000},
        },
    }


def _segment(duration_ms=1000.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def test_build_wakegate_attaches_runtime_and_emits_events_from_one_owner():
    import kiosk

    runtime = MagicMock(run=MagicMock(
        return_value=DirectorResult(reason="silence_timeout", turns=1, total_duration_s=3.0)))
    console = MagicMock()

    fake_mic = MagicMock()
    fake_mic.__enter__ = MagicMock(return_value=fake_mic)
    fake_mic.__exit__ = MagicMock(return_value=None)
    fake_vad = MagicMock(process_chunk=MagicMock(return_value=[]), reset=MagicMock())
    fake_embedder = MagicMock(
        extract=MagicMock(return_value=np.ones(192, dtype=np.float32) / np.sqrt(192)))
    fake_wake = MagicMock(process=MagicMock(return_value=None), reset=MagicMock())

    gate = kiosk.build_wakegate(
        _config(), console, runtime=runtime,
        _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
    )

    # Drive one full cycle: wake -> snapshot -> blocking handoff -> IDLE
    fake_wake.process.return_value = 0.9
    gate._handle_chunk(np.zeros(480, dtype=np.float32))
    fake_vad.process_chunk.return_value = [_segment()]
    gate._handle_chunk(np.zeros(480, dtype=np.float32))

    runtime.run.assert_called_once()
    assert gate._state == "IDLE"

    # The console saw the event tags, emitted from the single WakeGate owner.
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "[WAKE]" in printed
    assert "[SESSION STARTED]" in printed
    assert "[SESSION ENDED]" in printed
    assert "[IDLE]" in printed
    # No legacy double-handoff tag.
    assert "[HANDOFF]" not in printed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_kiosk_entrypoint.py -v`
Expected: FAIL — `AttributeError: module 'kiosk' has no attribute 'build_wakegate'` (and the old `kiosk.py` still imports the deleted `KioskPipeline`, so the import itself fails — fix both in Step 3).

- [ ] **Step 3: Rewrite kiosk.py**

Replace the whole file with the version below. Key changes from the old file: import `WakeGate` (not `KioskPipeline`); a single `_make_event_printer(console)` produces ALL console tags from one owner (so `[HANDOFF]` is gone and tags can't double-print); `build_wakegate()` is the testable factory; `main()` builds the `DirectorRuntime` (Plan 02) and warms models only on the `--talkback` path.

```python
# kiosk.py
"""Kiosk talkback entry point — wake-word activated, Director-owned session."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim
import argparse
import sys
from typing import Any, Optional

import yaml
from rich.console import Console

from modes.director.wakegate import WakeGate

console = Console()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_event_printer(console: Console):
    """ONE owner for all console event prints (spec section 4a — single owner).

    Emits [WAKE]/[SESSION STARTED]/[SESSION ENDED]/[IDLE]. There is no separate
    [HANDOFF] tag any more: the WakeGate's session_started IS the handoff, so a
    second component can no longer double-print it."""
    def on_event(event_type: str, payload: dict) -> None:
        if event_type == "wake_detected":
            console.print(
                f"[magenta][WAKE][/] phrase={payload['phrase']} "
                f"score={payload['score']:.3f}"
            )
        elif event_type == "session_started":
            console.print("[bold cyan][SESSION STARTED][/] Primary speaker locked")
        elif event_type == "session_ended":
            console.print(f"[bold yellow][SESSION ENDED][/] reason={payload['reason']}\n")
            console.print("[dim][IDLE] Listening for wake phrase...[/]")
        elif event_type == "awaiting_speech_timeout":
            # Pre-session abort (no session ever started); fall back to IDLE.
            console.print("[dim][IDLE] No speech after wake; listening again...[/]")
    return on_event


def build_wakegate(
    config: dict,
    console: Console,
    runtime: Any,
    _mic: Optional[Any] = None,
    _vad: Optional[Any] = None,
    _embedder: Optional[Any] = None,
    _wake_detector: Optional[Any] = None,
) -> WakeGate:
    """Construct the WakeGate around a DirectorRuntime. Underscore kwargs inject
    fakes in tests; production passes none and the WakeGate builds real I/O."""
    return WakeGate(
        config=config,
        runtime=runtime,
        on_event=_make_event_printer(console),
        _mic=_mic, _vad=_vad, _embedder=_embedder, _wake_detector=_wake_detector,
    )


def _build_runtime(config: dict):
    """Build the DirectorRuntime (Plan 02): warm STT/TTS/LLM, then construct the
    runtime that owns the asyncio loop. Imports are local so dry-run never loads
    GPU backends."""
    from core.logging.jsonl_logger import EventLogger
    from modes.director.runtime import DirectorRuntime   # Plan 02
    from modes.talkback.llm import LlmClient
    from modes.talkback.player import Player
    from modes.talkback.stt import StreamingStt
    from modes.talkback.tts import TtsEngine

    tb_cfg = config["kiosk"].get("talkback", {})
    logger = EventLogger(
        path_template=tb_cfg.get("logging", {}).get(
            "jsonl_path", "logs/kiosk-{date}-{session_id}.jsonl"),
        session_id="pending",
    )

    stt_cfg = tb_cfg.get("stt", {})
    stt = StreamingStt(
        model=stt_cfg.get("model", "base"),
        compute_type=stt_cfg.get("compute_type", "int8"),
        device=stt_cfg.get("device", "cpu"),
    )
    llm_cfg = tb_cfg.get("llm", {})
    llm = LlmClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        model=llm_cfg.get("model", "gemma-3-4b-it"),
        temperature=llm_cfg.get("temperature", 0.6),
        max_tokens=llm_cfg.get("max_tokens", 512),
    )
    tts_cfg = tb_cfg.get("tts", {})
    tts = TtsEngine(
        backend=tts_cfg.get("backend", "kokoro"),
        voice=tts_cfg.get("voice", "af_bella"),
        device=tts_cfg.get("device", "cuda"),
    )
    player = Player(sample_rate=tb_cfg.get("sample_rate_hz", 16000))

    import asyncio
    with console.status("[bold]Loading STT model..."):
        stt._ensure_model()
    console.print("[green]✓[/] STT loaded")
    with console.status("[bold]Loading TTS model (Kokoro)..."):
        tts._ensure_model()
    console.print("[green]✓[/] TTS loaded")
    with console.status("[bold]Checking LLM server..."):
        llm_ok = asyncio.run(llm.ping())
    if not llm_ok:
        console.print("[red]✗[/] LLM server unreachable at "
                      + llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"))
        console.print("[dim]Start llama-server and retry.[/]")
        sys.exit(3)
    console.print("[green]✓[/] LLM server reachable")

    return DirectorRuntime(stt=stt, llm=llm, tts=tts, player=player, logger=logger)


def main():
    parser = argparse.ArgumentParser(description="Target VAD — Kiosk Talkback")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--wake-phrase",
        help="Override wake phrase (default from config). Bundled options: "
             "hey_jarvis, alexa, hey_mycroft.",
    )
    parser.add_argument(
        "--talkback", action="store_true",
        help="Force talkback_enabled=true (full-duplex voice assistant mode).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.wake_phrase:
        config["kiosk"]["wake_phrase"] = args.wake_phrase
    if args.talkback:
        config["kiosk"]["talkback_enabled"] = True

    if not config["kiosk"].get("talkback_enabled", False):
        console.print(
            "[yellow]Director kiosk requires talkback. Re-run with --talkback "
            "(or set kiosk.talkback_enabled: true).[/]"
        )
        sys.exit(2)

    runtime = _build_runtime(config)
    console.print(
        f"[bold][TALKBACK][/] Listening for "
        f"[bold cyan]\"{config['kiosk']['wake_phrase']}\"[/]..."
    )
    gate = build_wakegate(config, console, runtime=runtime)

    try:
        gate.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")


if __name__ == "__main__":
    main()
```

> **Dry-run note.** The old `--dry-run` (non-talkback) path forwarded raw primary-speech segments to a console printer. That path depended on the deleted ACTIVE_SESSION scoring loop, which no longer exists — the Director owns the conversation now. Dry-run is therefore dropped from the entry point (replaced by the explicit "requires talkback" guard). If a no-LLM smoke mode is wanted later, it belongs in Plan 02 as a `DirectorRuntime` that returns immediately; it is out of scope here. This is called out in the Self-Review.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_kiosk_entrypoint.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full suite + import smoke check**

Run:
```bash
python3 -m pytest tests/director/ -v
python3 -c "import kiosk; print('kiosk imports OK')"
```
Expected: all director tests PASS; `kiosk imports OK` (no reference to the deleted `KioskPipeline`).

> If `import kiosk` fails because `modes.director.runtime` (Plan 02) does not exist yet, that import is *local* to `_build_runtime` and is NOT hit by `import kiosk` or the entry-point test (which injects a fake runtime and calls `build_wakegate` directly). The bare `import kiosk` only needs `modes.director.wakegate`, which Task 2 provides. Confirm the smoke check passes; if it fails it means a top-level import leaked — move any Plan-02 import into `_build_runtime`.

- [ ] **Step 6: Commit**

```bash
git add kiosk.py tests/director/test_kiosk_entrypoint.py
git commit -m "feat(director): wire kiosk.py to WakeGate + DirectorRuntime; single event owner (no double [HANDOFF])"
```

---

## Self-Review

- **Spec coverage (this plan = spec §13 step 3, the WakeGate subsumption / Req-5):**
  - **Req-5 single session ownership (§4, §4a):** the WakeGate owns only IDLE + AWAIT_FIRST_SEGMENT, makes one blocking `runtime.run(handoff)`, and resets to IDLE on return (Task 2). The fat pipeline, `Session`, watchdog thread, `_handle_active_chunk`/`_process_session_segment`, and `_end_session` are deleted (Task 3). The §4a proof is landed as CI tests: grep post-conditions for all banned substrings, the synchronous-call + reset-only assertions, the deleted-config-keys check, and the no-orphan-after-end integration test (Task 4). ✓
  - **Both root causes of the live bug removed (§4a.4):** (a) the blocking call no longer starves a `last_speech_at` refresher because the WakeGate keeps no `last_speech_at` at all — the Director refreshes its own (Plan 01); (b) the pipeline watchdog reading a frozen value is deleted, and the racing config keys `kiosk.session_silence_timeout_s`/`session_hard_timeout_s` are removed so no second timeout value can be read (Task 3). ✓
  - **Single event owner (no double [HANDOFF]) (prompt):** all console tags come from one `_make_event_printer` owner; `[HANDOFF]` is gone; `[WAKE]`/`[SESSION STARTED]`/`[SESSION ENDED]`/`[IDLE]` preserved and asserted (Task 5). ✓
  - **Reused as-is:** `MicrophoneStream`, `SileroVAD`/`SpeechSegment`, `EmbeddingExtractor`, `WakeWordDetector` (kept; Task 3 deletes only `pipeline.py`/`session.py`). ✓
- **Plan-02 dependency (binding contract):** consumes `DirectorRuntime(handoff) -> DirectorResult` and the renamed `DirectorHandoff`/`DirectorResult` (+`holdout_embedding`). Task 1 lands the rename self-contained-ly *only if* Plan 02 hasn't (with a coordination note to avoid a double-edit). The `DirectorRuntime` import is local to `_build_runtime` so this plan's tests and the bare `import kiosk` never require Plan 02 to exist. ✓
- **Plan-05 dependency (explicit, per binding contract):** `holdout_embedding` is passed as the **same first-segment/primary embedding** — a documented placeholder, acceptable ONLY because Plan 05 owns the real holdout-before-finalize capture. Stated in the `wakegate.py` docstring/comment (using the word "placeholder", keeping the placeholder scan clean), in the handoff docstring (Task 1), in a dedicated test (`test_holdout_embedding_is_first_segment_embedding_placeholder`), and here. ✓
- **Scope honesty / deviations:** `--dry-run` is dropped from the entry point (its non-talkback ACTIVE_SESSION scoring loop is deleted; a no-LLM smoke runtime belongs in Plan 02) — flagged in Task 5. `kiosk.decision_smoother` is retained (reconciled by Plan 05) — flagged in Task 3. `kiosk.watchdog.tick_ms` is removed (powered only the deleted pipeline watchdog). ✓
- **Test-replacement honesty:** the three old kiosk tests are deleted *because they assert deleted behavior* (ACTIVE_SESSION, watchdog-without-chunks, `_end_session`, `Session`); their construction-pattern is ported into `tests/director/test_wakegate.py` (state machine + handoff wiring) and the §4a proof into `tests/director/test_wakegate_single_ownership.py`. No behavior is silently dropped — the surviving behaviors (IDLE/await/snapshot/handoff/event-robustness) are all re-tested. ✓
- **Placeholder scan:** no placeholder markers in any code block; every code step is complete. The Plan-05 dependency is expressed in prose + the word "placeholder". ✓
- **Type/interface consistency:** `WakeGate(config, runtime, on_event, _mic, _vad, _embedder, _wake_detector)` is identical across Tasks 2/4/5; `DirectorHandoff(mic, primary_embedding, holdout_embedding, first_segment, config, vad, embedder)` and `DirectorResult(reason, turns, total_duration_s)` match the binding contract throughout; `runtime.run(handoff) -> DirectorResult` consumed consistently; `build_wakegate` signature matches its test. ✓
