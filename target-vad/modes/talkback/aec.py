"""AEC — acoustic echo cancellation via webrtc-audio-processing-py.

Wraps the WebRTC APM module (AEC3). Processes 10 ms / 160-sample frames
at 16 kHz. Takes (mic_frame, playback_ref_frame) → clean_mic_frame.
"""

import numpy as np


class AecProcessor:
    """Wraps webrtc-audio-processing-py for echo cancellation."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 10):
        self._sample_rate = sample_rate
        self._frame_samples = int(sample_rate * frame_ms / 1000)

        try:
            from webrtc_audio_processing import AudioProcessingModule
        except ImportError:
            raise RuntimeError(
                "webrtc-audio-processing-py is required for AEC. "
                "Install: pip install webrtc-audio-processing-py "
                "(may need libwebrtc-audio-processing-dev on aarch64)"
            )

        self._apm = AudioProcessingModule(
            sample_rate_hz=sample_rate,
            num_channels=1,
        )
        self._apm.enable_echo_cancellation(True)
        self._apm.enable_noise_suppression(True)

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def process_frame(
        self, mic_frame: np.ndarray, playback_ref: np.ndarray
    ) -> np.ndarray:
        self._apm.process_reverse_stream(playback_ref)
        clean = self._apm.process_stream(mic_frame)
        return clean
