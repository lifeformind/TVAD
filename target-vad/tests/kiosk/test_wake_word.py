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
