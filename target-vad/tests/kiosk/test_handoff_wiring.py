"""Tests for KioskPipeline → TalkbackController hand-off wiring."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult


@pytest.fixture
def talkback_config():
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
            "talkback_enabled": True,
            "talkback": {"sample_rate_hz": 16000},
        },
    }


@pytest.fixture
def disabled_config():
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
            "talkback_enabled": False,
        },
    }


def make_segment(duration_ms: float = 1000.0) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def make_fakes():
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
    return fake_mic, fake_vad, fake_embedder, fake_wake


class TestHandoffWiring:
    def test_talkback_enabled_calls_controller_run(self, talkback_config):
        from modes.kiosk.pipeline import KioskPipeline

        fake_mic, fake_vad, fake_embedder, fake_wake = make_fakes()
        fake_controller = MagicMock()
        fake_controller.run = MagicMock(
            return_value=TalkbackResult(reason="silence_timeout", turns=2, total_duration_s=10.0)
        )
        on_primary = MagicMock()

        p = KioskPipeline(
            config=talkback_config,
            on_primary_speech=on_primary,
            _mic=fake_mic,
            _vad=fake_vad,
            _embedder=fake_embedder,
            _wake_detector=fake_wake,
            _talkback_controller=fake_controller,
        )

        fake_wake.process.return_value = 0.87
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))

        fake_controller.run.assert_called_once()
        handoff = fake_controller.run.call_args[0][0]
        assert isinstance(handoff, TalkbackHandoff)
        assert handoff.mic is fake_mic
        assert handoff.primary_embedding.shape == (192,)
        on_primary.assert_not_called()
        assert p._state == "IDLE"

    def test_talkback_disabled_fires_on_primary_speech(self, disabled_config):
        from modes.kiosk.pipeline import KioskPipeline

        fake_mic, fake_vad, fake_embedder, fake_wake = make_fakes()
        on_primary = MagicMock()

        p = KioskPipeline(
            config=disabled_config,
            on_primary_speech=on_primary,
            _mic=fake_mic,
            _vad=fake_vad,
            _embedder=fake_embedder,
            _wake_detector=fake_wake,
        )

        fake_wake.process.return_value = 0.87
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))

        on_primary.assert_called_once()
        assert p._state == "ACTIVE_SESSION"

    def test_handoff_payload_contains_first_segment(self, talkback_config):
        from modes.kiosk.pipeline import KioskPipeline

        fake_mic, fake_vad, fake_embedder, fake_wake = make_fakes()
        fake_controller = MagicMock()
        fake_controller.run = MagicMock(
            return_value=TalkbackResult(reason="stopped", turns=0, total_duration_s=1.0)
        )
        segment = make_segment(500.0)

        p = KioskPipeline(
            config=talkback_config,
            on_primary_speech=lambda s, e: None,
            _mic=fake_mic,
            _vad=fake_vad,
            _embedder=fake_embedder,
            _wake_detector=fake_wake,
            _talkback_controller=fake_controller,
        )

        fake_wake.process.return_value = 0.87
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [segment]
        p._handle_chunk(np.zeros(480, dtype=np.float32))

        handoff = fake_controller.run.call_args[0][0]
        assert handoff.first_segment is segment

    def test_handoff_result_reason_propagates_to_session_ended(self, talkback_config):
        from modes.kiosk.pipeline import KioskPipeline

        fake_mic, fake_vad, fake_embedder, fake_wake = make_fakes()
        fake_controller = MagicMock()
        fake_controller.run = MagicMock(
            return_value=TalkbackResult(reason="device_lost", turns=1, total_duration_s=5.0)
        )
        ended_reasons = []

        p = KioskPipeline(
            config=talkback_config,
            on_primary_speech=lambda s, e: None,
            on_session_ended=lambda reason: ended_reasons.append(reason),
            _mic=fake_mic,
            _vad=fake_vad,
            _embedder=fake_embedder,
            _wake_detector=fake_wake,
            _talkback_controller=fake_controller,
        )

        fake_wake.process.return_value = 0.87
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))

        assert "device_lost" in ended_reasons
