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
