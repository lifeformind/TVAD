"""Tests for KioskPipeline watchdog thread (F4).

The watchdog fires silence/hard timeouts even when the mic stops producing chunks.
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment


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
            "session_silence_timeout_s": 0.3,
            "session_hard_timeout_s": 300,
            "decision_smoother": {"window_size": 3, "min_matches": 2, "threshold": 0.6},
            "watchdog": {"tick_ms": 50},
        },
    }


def make_segment(duration_ms: float = 1000.0) -> SpeechSegment:
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestKioskWatchdog:
    def test_silence_timeout_fires_without_chunks(self, base_config):
        """Watchdog fires silence_timeout even when mic produces no chunks."""
        from modes.kiosk.pipeline import KioskPipeline

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

        ended_reasons = []
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_session_ended=lambda reason: ended_reasons.append(reason),
            _mic=fake_mic,
            _vad=fake_vad,
            _embedder=fake_embedder,
            _wake_detector=fake_wake,
        )

        # Drive into ACTIVE_SESSION
        fake_wake.process.return_value = 0.87
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = [make_segment()]
        p._handle_chunk(np.zeros(480, dtype=np.float32))
        fake_vad.process_chunk.return_value = []
        assert p._state == "ACTIVE_SESSION"

        # Start watchdog, then wait for silence timeout (0.3s) + margin
        p._start_watchdog()
        try:
            time.sleep(0.5)
        finally:
            p._stop_watchdog()

        assert p._state == "IDLE"
        assert "silence_timeout" in ended_reasons

    def test_watchdog_does_not_fire_when_not_in_active_session(self, base_config):
        """Watchdog should not fire when pipeline is in IDLE state."""
        from modes.kiosk.pipeline import KioskPipeline

        fake_mic = MagicMock()
        fake_mic.__enter__ = MagicMock(return_value=fake_mic)
        fake_mic.__exit__ = MagicMock(return_value=None)

        ended_reasons = []
        p = KioskPipeline(
            config=base_config,
            on_primary_speech=lambda s, e: None,
            on_session_ended=lambda reason: ended_reasons.append(reason),
            _mic=fake_mic,
            _vad=MagicMock(),
            _embedder=MagicMock(),
            _wake_detector=MagicMock(),
        )
        assert p._state == "IDLE"

        p._start_watchdog()
        try:
            time.sleep(0.2)
        finally:
            p._stop_watchdog()

        assert p._state == "IDLE"
        assert len(ended_reasons) == 0
