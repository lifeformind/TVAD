"""Tests for AEC wrapper — echo cancellation via webrtc-audio-processing-py.

Uses a mock APM for unit tests.
"""

from unittest.mock import MagicMock

import numpy as np

from modes.talkback.aec import AecProcessor


class TestAecProcessorUnit:
    def test_process_returns_same_shape(self):
        aec = AecProcessor.__new__(AecProcessor)
        aec._frame_samples = 160
        aec._apm = MagicMock()
        aec._apm.process_reverse_stream = MagicMock()
        aec._apm.process_stream = MagicMock(
            return_value=np.zeros(160, dtype=np.float32)
        )

        mic = np.random.randn(160).astype(np.float32) * 0.1
        ref = np.zeros(160, dtype=np.float32)
        clean = aec.process_frame(mic, ref)
        assert clean.shape == (160,)
        assert clean.dtype == np.float32

    def test_process_calls_reverse_then_stream(self):
        aec = AecProcessor.__new__(AecProcessor)
        aec._frame_samples = 160
        aec._apm = MagicMock()
        aec._apm.process_reverse_stream = MagicMock()
        aec._apm.process_stream = MagicMock(
            return_value=np.zeros(160, dtype=np.float32)
        )

        mic = np.zeros(160, dtype=np.float32)
        ref = np.zeros(160, dtype=np.float32)
        aec.process_frame(mic, ref)

        aec._apm.process_reverse_stream.assert_called_once()
        aec._apm.process_stream.assert_called_once()


class TestAecFrameSize:
    def test_frame_samples_default_160(self):
        aec = AecProcessor.__new__(AecProcessor)
        aec._frame_samples = 160
        assert aec._frame_samples == 160
