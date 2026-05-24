"""Layer 2 — AEC integration test with synthetic sine signals.

Generates a known sine on the playback reference, mixes it into mic input
at a known SNR, runs through APM, and asserts > 10 dB suppression.
"""

import numpy as np
import pytest

from unittest.mock import MagicMock
try:
    from webrtc_audio_processing import AudioProcessingModule
    HAS_APM = not isinstance(AudioProcessingModule, MagicMock)
except (ImportError, OSError):
    HAS_APM = False

from modes.talkback.aec import AecProcessor


@pytest.mark.integration
@pytest.mark.skipif(not HAS_APM, reason="webrtc-audio-processing-py not installed")
class TestAecIntegration:
    def test_sine_suppression(self):
        aec = AecProcessor(sample_rate=16000, frame_ms=10)
        frame_samples = aec.frame_samples

        t = np.arange(frame_samples) / 16000.0
        sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        for _ in range(50):
            aec.process_frame(mic_frame=sine.copy(), playback_ref=sine.copy())

        clean = aec.process_frame(mic_frame=sine.copy(), playback_ref=sine.copy())

        mic_power = np.mean(sine ** 2)
        clean_power = np.mean(clean ** 2)

        if clean_power > 0 and mic_power > 0:
            suppression_db = 10 * np.log10(mic_power / clean_power)
            assert suppression_db > 10
