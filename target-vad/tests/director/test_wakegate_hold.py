# tests/director/test_wakegate_hold.py
"""Director-09 post-eject WakeGate quiet-hold (port of lockout.py's idle half).

After a session ends with reason "speaker_mismatch" (suspected hijacker), the
WakeGate ignores wake detections until the near field has been quiet for
lockout_idle_after_s continuous seconds. Never permanent: quiet always clears
it, and every other end reason never engages the hold.

Fixtures/helpers live in tests/director/conftest.py (shared with
test_wakegate.py)."""

from unittest.mock import MagicMock

import numpy as np

from modes.talkback.handoff import DirectorResult
from tests.director.conftest import make_gate, drive_one_cycle


def _mismatch_runtime(prox=0.5):
    m = MagicMock()
    m.run = MagicMock(return_value=DirectorResult(
        reason="speaker_mismatch", turns=1, total_duration_s=1.0,
        proximity_rms=prox))
    return m


def _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, clock,
            monkeypatch, runtime=None):
    monkeypatch.setattr("modes.director.wakegate.time.monotonic",
                        lambda: clock[0])
    g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                  runtime or _mismatch_runtime())
    drive_one_cycle(g, fake_wake, fake_vad)
    fake_wake.process.return_value = None      # no accidental wakes below
    return g


def test_speaker_mismatch_result_engages_hold(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    g = _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                clock, monkeypatch)
    assert g._hold_floor == 0.5
    fake_wake.process.reset_mock()
    g._handle_chunk(np.full(480, 0.9, dtype=np.float32))    # loud: still held
    fake_wake.process.assert_not_called()                    # wake never consulted


def test_hold_clears_after_quiet_period(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    g = _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                clock, monkeypatch)
    quiet = np.zeros(480, dtype=np.float32)
    g._handle_chunk(quiet)                     # holding; quiet clock running
    clock[0] = 1006.0                          # > lockout_idle_after_s (5s)
    g._handle_chunk(quiet)                     # clears + falls through to wake
    assert g._hold_floor is None
    fake_wake.process.reset_mock()
    fake_wake.process.return_value = 0.9
    g._handle_chunk(quiet)                     # wake works again
    fake_wake.process.assert_called_once()
    assert g._state == "AWAIT_FIRST_SEGMENT"


def test_loud_chunk_resets_the_quiet_clock(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    g = _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                clock, monkeypatch)
    quiet = np.zeros(480, dtype=np.float32)
    clock[0] = 1004.0
    g._handle_chunk(np.full(480, 0.9, dtype=np.float32))    # loud at t=1004: reset
    clock[0] = 1008.0
    g._handle_chunk(quiet)                     # only 4s since loud -> still held
    assert g._hold_floor is not None
    clock[0] = 1009.5
    g._handle_chunk(quiet)                     # 5.5s since loud -> clears
    assert g._hold_floor is None


def test_other_end_reasons_never_hold(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("modes.director.wakegate.time.monotonic",
                        lambda: clock[0])
    for reason in ("silence_timeout", "enroll_verify_failed"):
        rt = MagicMock()
        rt.run = MagicMock(return_value=DirectorResult(
            reason=reason, turns=0, total_duration_s=1.0, proximity_rms=0.5))
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder,
                      fake_wake, rt)
        drive_one_cycle(g, fake_wake, fake_vad)
        assert g._hold_floor is None
