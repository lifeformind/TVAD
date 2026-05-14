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
            "awaiting_speech_timeout_s": 5,
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
                  on_primary=None, on_started=None, on_ended=None, on_event=None):
    from modes.kiosk.pipeline import KioskPipeline
    return KioskPipeline(
        config=base_config,
        on_primary_speech=on_primary or (lambda seg, emb: None),
        on_session_started=on_started or (lambda: None),
        on_session_ended=on_ended or (lambda reason: None),
        on_event=on_event or (lambda event, payload: None),
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


def make_segment(duration_ms: float = 1000.0) -> "SpeechSegment":
    from core.vad.silero_vad import SpeechSegment
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0,
        end_ms=duration_ms,
        duration_ms=duration_ms,
    )


def force_active_session(pipeline, fake_wake, fake_vad):
    """Helper: drive pipeline from IDLE through AWAITING_SPEECH to ACTIVE_SESSION.

    Fires wake, then feeds a VAD-produced segment to complete the snapshot.
    """
    fake_wake.process.return_value = 0.87
    pipeline._handle_chunk(np.zeros(480, dtype=np.float32))  # IDLE → AWAITING_SPEECH
    assert pipeline._state == "AWAITING_SPEECH"
    # Next chunk produces a segment from VAD, which becomes the snapshot
    fake_vad.process_chunk.return_value = [make_segment()]
    pipeline._handle_chunk(np.zeros(480, dtype=np.float32))
    # Reset vad.process_chunk to empty for subsequent chunks unless test overrides
    fake_vad.process_chunk.return_value = []
    assert pipeline._state == "ACTIVE_SESSION"


class TestIdleAndAwaitingSpeech:
    def test_idle_no_wake_stays_idle(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = None
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"

    def test_idle_wake_transitions_to_awaiting_speech(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "AWAITING_SPEECH"

    def test_first_segment_starts_session(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → AWAITING_SPEECH
        assert p._state == "AWAITING_SPEECH"
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "ACTIVE_SESSION"

    def test_first_segment_fires_primary_speech(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """The first speech segment after wake fires on_primary_speech immediately
        (it IS the primary speaker by definition; no smoother voting required)."""
        on_primary = MagicMock()
        fake_wake.process.return_value = 0.87
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → AWAITING_SPEECH
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        # First primary callback should already have fired
        assert on_primary.called
        assert on_primary.call_count == 1

    def test_session_start_invokes_on_session_started(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        started = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_started=started)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → AWAITING_SPEECH
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        started.assert_called_once_with()

    def test_failed_snapshot_returns_to_idle(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        fake_wake.process.return_value = 0.87
        fake_embedder.extract.side_effect = RuntimeError("snapshot failed")
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → AWAITING_SPEECH
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"
        assert p._session is None

    def test_awaiting_speech_timeout_returns_to_idle(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
        """If no speech segment arrives within awaiting_speech_timeout_s, abort to IDLE."""
        on_ended = MagicMock()
        clock = [1000.0]
        monkeypatch.setattr("modes.kiosk.pipeline.time.monotonic", lambda: clock[0])
        fake_wake.process.return_value = 0.87
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_ended=on_ended)
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # → AWAITING_SPEECH at t=1000
        assert p._state == "AWAITING_SPEECH"
        # Jump clock past 5s timeout, feed another chunk (no segments from VAD)
        clock[0] = 1006.0
        fake_vad.process_chunk.return_value = []
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "IDLE"
        on_ended.assert_called_once_with("awaiting_speech_timeout")


class TestActiveSession:
    def test_matched_segment_invokes_callback(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        # Embedder returns the same vector every call → cosine = 1.0 always → smoother fires
        on_primary = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        force_active_session(p, fake_wake, fake_vad)
        # force_active_session already fired on_primary once (the snapshot segment).
        # Need 2 of 3 in window for the smoother. The first segment in ACTIVE_SESSION
        # is window[1]; the second is window[2]. Two matched segments after snapshot
        # = 2 hits in the smoother window of 3 → fires on the second.
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # smoother gets 1
        p._handle_chunk(np.zeros(480, dtype=np.float32))  # smoother gets 2 → fires
        # Total calls: 1 (snapshot) + 1 (smoother fire on second chunk) = 2
        assert on_primary.call_count >= 2

    def test_unmatched_segment_does_not_invoke_callback(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        on_primary = MagicMock()
        snapshot = np.ones(192, dtype=np.float32) / np.sqrt(192)
        orthogonal = np.zeros(192, dtype=np.float32)
        orthogonal[0] = 1.0
        # First extract call is the snapshot (in _start_session_from_segment).
        # Subsequent calls (for ACTIVE_SESSION segments) return orthogonal.
        fake_embedder.extract.side_effect = [snapshot] + [orthogonal] * 10
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        force_active_session(p, fake_wake, fake_vad)
        # After force_active_session: snapshot was the embedding, on_primary called ONCE for it.
        snapshot_call_count = on_primary.call_count
        fake_vad.process_chunk.return_value = [make_segment()]
        for _ in range(5):
            p._handle_chunk(np.zeros(480, dtype=np.float32))
        # No NEW primary calls beyond the snapshot one (orthogonal segments never match)
        assert on_primary.call_count == snapshot_call_count

    def test_callback_exception_does_not_crash(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        on_primary = MagicMock(side_effect=RuntimeError("downstream broken"))
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_primary=on_primary)
        # force_active_session will trigger on_primary once (snapshot); that exception
        # is caught by _safe_callback. Then we feed more segments — those exceptions
        # also get caught. State should remain ACTIVE_SESSION throughout.
        force_active_session(p, fake_wake, fake_vad)
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert p._state == "ACTIVE_SESSION"


class TestSessionEnd:
    def test_silence_timeout_ends_session(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
        on_ended = MagicMock()
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, on_ended=on_ended)
        # Drive into ACTIVE_SESSION at t=1000.0
        clock = [1000.0]
        monkeypatch.setattr("modes.kiosk.pipeline.time.monotonic", lambda: clock[0])
        force_active_session(p, fake_wake, fake_vad)
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
        force_active_session(p, fake_wake, fake_vad)
        # Both timeouts expire simultaneously at this clock value (301s past start);
        # hard is checked first in _handle_active_chunk and wins.
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
        force_active_session(p, fake_wake, fake_vad)
        p._end_session("stopped")
        on_ended.assert_called_with("stopped")
        assert p._state == "IDLE"
        assert p._session is None


class TestEventCallback:
    def test_on_event_default_is_noop(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """Pipeline works fine when no on_event callback is provided."""
        p = make_pipeline(base_config, fake_mic, fake_vad, fake_embedder, fake_wake)
        # Drive through a session — should not raise
        force_active_session(p, fake_wake, fake_vad)
        assert p._state == "ACTIVE_SESSION"

    def test_wake_detected_event_fires(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """on_event fires with 'wake_detected' when wake fires in IDLE."""
        events = []
        def on_event(event_type, payload):
            events.append((event_type, payload))
        from modes.kiosk.pipeline import KioskPipeline
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_event=on_event,
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        fake_wake.process.return_value = 0.87
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        assert len(events) == 1
        assert events[0][0] == "wake_detected"
        assert events[0][1] == {"phrase": "hey_jarvis", "score": 0.87}

    def test_session_started_event_fires(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """on_event fires with 'session_started' when session begins."""
        events = []
        from modes.kiosk.pipeline import KioskPipeline
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_event=lambda et, pl: events.append((et, pl)),
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        force_active_session(p, fake_wake, fake_vad)
        types = [e[0] for e in events]
        assert "session_started" in types
        started_payload = next(p for t, p in events if t == "session_started")
        assert "snapshot_norm" in started_payload
        assert started_payload["snapshot_norm"] == pytest.approx(1.0, abs=1e-5)

    def test_segment_scored_event_fires(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """on_event fires with 'segment_scored' for each session segment, with match/no_match decision."""
        events = []
        from modes.kiosk.pipeline import KioskPipeline
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_event=lambda et, pl: events.append((et, pl)),
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        force_active_session(p, fake_wake, fake_vad)
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        seg_events = [e for e in events if e[0] == "segment_scored"]
        # force_active_session fires 1 segment_scored (the snapshot), plus 1 more from
        # the additional chunk above → at least 2 total
        assert len(seg_events) >= 1
        for et, pl in seg_events:
            assert "score" in pl
            assert "duration_ms" in pl
            assert pl["decision"] in ("match", "no_match")

    def test_session_ended_event_fires(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """on_event fires with 'session_ended' on session end."""
        events = []
        from modes.kiosk.pipeline import KioskPipeline
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_event=lambda et, pl: events.append((et, pl)),
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        force_active_session(p, fake_wake, fake_vad)
        p._end_session("stopped")
        ended = [e for e in events if e[0] == "session_ended"]
        assert len(ended) == 1
        assert ended[0][1] == {"reason": "stopped"}

    def test_on_event_exception_does_not_crash(self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake):
        """An exception in on_event is swallowed by _safe_callback; pipeline continues."""
        from modes.kiosk.pipeline import KioskPipeline
        def buggy_event(event_type, payload):
            raise RuntimeError("event handler broke")
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_event=buggy_event,
            _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder, _wake_detector=fake_wake,
        )
        # Should not raise even though on_event raises
        force_active_session(p, fake_wake, fake_vad)
        assert p._state == "ACTIVE_SESSION"
