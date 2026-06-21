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


def drive_one_cycle(g, fake_wake, fake_vad, seg=None):
    """Drive ONE full wake->session cycle through g.run() with a finite mic.
    The mic yields a wake chunk then a first-segment chunk, then is exhausted, so
    g.run() runs the single session and exits. runtime.run is called from run()
    AFTER the wake mic generator is closed (single-consumer handoff)."""
    seg = seg or make_segment()
    g.mic.stream = MagicMock(return_value=iter([
        np.zeros(480, dtype=np.float32),   # chunk 1 -> wake detected
        np.zeros(480, dtype=np.float32),   # chunk 2 -> first speech segment
    ]))
    fake_wake.process.return_value = 0.87
    fake_vad.process_chunk.return_value = [seg]
    g.run()
    return seg


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
