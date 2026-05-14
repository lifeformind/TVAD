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
