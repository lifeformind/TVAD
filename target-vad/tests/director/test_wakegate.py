# tests/director/test_wakegate.py
"""WakeGate state-machine + construction tests. Ported from the old
tests/kiosk/test_pipeline.py / test_handoff_wiring.py, minus everything that
referenced the deleted ACTIVE_SESSION/watchdog/Session paths. Fixtures live
in tests/director/conftest.py (shared with test_wakegate_hold.py)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.handoff import DirectorHandoff, DirectorResult
from tests.director.conftest import make_segment, make_gate, drive_one_cycle


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
        drive_one_cycle(g, fake_wake, fake_vad)

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

    def test_handoff_passes_talkback_config(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        self._drive_to_handoff(g, fake_wake, fake_vad)
        handoff = fake_runtime.run.call_args[0][0]
        assert handoff.config == base_config["kiosk"]["talkback"]

    def test_handoff_passes_first_segment(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        seg = make_segment(500.0)
        drive_one_cycle(g, fake_wake, fake_vad, seg=seg)
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

    def test_failed_snapshot_stays_awaiting_for_retry(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
        # 2026-07-07 19:38 live: silent reset-to-IDLE on a bad seed made the
        # kiosk deaf right after the wake. Infra failure on one segment now
        # keeps AWAIT alive so the next utterance retries without a re-wake
        # (awaiting_speech_timeout still bounds the phase).
        fake_embedder.extract.side_effect = RuntimeError("snapshot failed")
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime)
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        g._handle_chunk(np.zeros(480, dtype=np.float32))
        assert g._state == "AWAIT_FIRST_SEGMENT"
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
        drive_one_cycle(g, fake_wake, fake_vad)
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
        drive_one_cycle(g, fake_wake, fake_vad)            # buggy events swallowed
        assert g._state == "IDLE"                          # still completed the cycle


class _SplitEmbedder:
    """Full-segment extract -> normalized ones; halves -> scripted vectors."""
    def __init__(self, half_a, half_b):
        self._returns = [np.ones(192, dtype=np.float32) / np.sqrt(192),
                         half_a, half_b]
        self.calls = 0

    def extract(self, audio):
        v = self._returns[min(self.calls, len(self._returns) - 1)]
        self.calls += 1
        return v


def _orthogonal_pair():
    a = np.zeros(192, dtype=np.float32); a[0] = 1.0
    b = np.zeros(192, dtype=np.float32); b[1] = 1.0
    return a, b                                     # cosine == 0.0 < 0.80


class TestVerifyBeforeServe:
    def test_split_half_mismatch_refuses_session(
            self, base_config, fake_mic, fake_vad, fake_wake, fake_runtime):
        a, b = _orthogonal_pair()
        emb = _SplitEmbedder(a, b)
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, emb, fake_wake, fake_runtime,
                      on_event=lambda et, pl: events.append((et, pl)))
        drive_one_cycle(g, fake_wake, fake_vad, seg=make_segment(2200.0))
        fake_runtime.run.assert_not_called()
        assert g._state == "AWAIT_FIRST_SEGMENT"   # retry, not reset (19:38 live)
        types = [et for et, _ in events]
        assert "verify_refused" in types and "session_started" not in types
        refused = next(pl for et, pl in events if et == "verify_refused")
        assert refused["score"] == pytest.approx(0.0, abs=1e-6)

    def test_split_half_match_serves(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        drive_one_cycle(g, fake_wake, fake_vad, seg=make_segment(2200.0))
        fake_runtime.run.assert_called_once()
        assert "session_started" in [et for et, _ in events]

    def test_short_first_segment_skips_split_half(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        drive_one_cycle(
            make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime),
            fake_wake, fake_vad, seg=make_segment(500.0))
        assert fake_embedder.extract.call_count == 1    # full segment only
        fake_runtime.run.assert_called_once()

    def test_one_second_segment_skips_split_half(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        # Live 2026-07-13: 1.0-1.1s seeds in a QUIET room were refused three
        # times in a row (scores 0.15-0.34) — their ~500ms halves are below
        # what ECAPA can embed honestly. The verify floor is on the HALVES:
        # only segments >= 2.0s (halves >= 1.0s) are split-half checked.
        drive_one_cycle(
            make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime),
            fake_wake, fake_vad, seg=make_segment(1128.0))
        assert fake_embedder.extract.call_count == 1    # full segment only
        fake_runtime.run.assert_called_once()

    def test_split_half_length_gate_uses_configured_sample_rate(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        # The >=1.0s gate reads core.audio.sample_rate (was hardcoded 16000):
        # at 4 kHz the same 500ms/8000-sample segment counts as 2s -> split-half
        # runs (full + two halves = 3 extracts) instead of being skipped.
        base_config["core"]["audio"]["sample_rate"] = 4000
        drive_one_cycle(
            make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime),
            fake_wake, fake_vad, seg=make_segment(500.0))
        assert fake_embedder.extract.call_count == 3
        fake_runtime.run.assert_called_once()

    def test_half_embed_failure_resets_to_idle(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        calls = {"n": 0}

        def _extract(audio):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("half embed failed")
            return np.ones(192, dtype=np.float32)

        fake_embedder.extract = MagicMock(side_effect=_extract)
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        drive_one_cycle(g, fake_wake, fake_vad, seg=make_segment(2200.0))
        fake_runtime.run.assert_not_called()
        assert g._state == "AWAIT_FIRST_SEGMENT"   # retry, not reset (19:38 live)
        # infra failure, not a verdict: no verify_refused event
        assert "verify_refused" not in [et for et, _ in events]


class TestSeedRetry:
    def test_refused_seed_then_clean_seed_enrolls_without_rewake(
            self, base_config, fake_mic, fake_vad, fake_wake, fake_runtime):
        # First utterance fails split-half (contaminated); the SECOND utterance
        # in the same AWAIT window enrolls — no re-wake needed (19:38 live fix).
        a = np.zeros(192, dtype=np.float32); a[0] = 1.0
        b = np.zeros(192, dtype=np.float32); b[1] = 1.0
        good = np.ones(192, dtype=np.float32)
        seq = [good, a, b,          # segment 1: full, half_a, half_b -> refused
               good, good, good]    # segment 2: self-similar -> enrolls
        emb = MagicMock()
        emb.extract = MagicMock(side_effect=lambda audio: seq.pop(0))
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, emb, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        fake_wake.process.return_value = 0.87
        g._handle_chunk(np.zeros(480, dtype=np.float32))       # wake
        fake_wake.process.return_value = None
        fake_vad.process_chunk.return_value = [make_segment(2200.0)]
        g._handle_chunk(np.zeros(480, dtype=np.float32))       # seed 1: refused
        assert g._pending_handoff is None
        assert g._state == "AWAIT_FIRST_SEGMENT"
        g._handle_chunk(np.zeros(480, dtype=np.float32))       # seed 2: enrolls
        assert g._pending_handoff is not None
        types = [et for et, _ in events]
        assert "verify_refused" in types and "session_started" in types
