"""AEC — acoustic echo cancellation via WebRTC APM.

Wraps the WebRTC APM module (AEC). Processes 10 ms / 160-sample frames
at 16 kHz. Takes (mic_frame, playback_ref_frame) → clean_mic_frame.

Backends (tried in order):
  1. webrtc-audio-processing-py (pip package, if available)
  2. libaec_shim.so (ctypes wrapper around system libwebrtc_audio_processing)
"""

import ctypes
import os

import numpy as np


class _CtypesApm:
    """ctypes bridge to libaec_shim.so wrapping the system WebRTC APM."""

    def __init__(self, lib_path: str, sample_rate: int, frame_ms: int):
        self._lib = ctypes.CDLL(lib_path)
        self._lib.aec_create.restype = ctypes.c_void_p
        self._lib.aec_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.aec_destroy.argtypes = [ctypes.c_void_p]
        self._lib.aec_process.restype = ctypes.c_int
        self._lib.aec_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._handle = self._lib.aec_create(sample_rate, frame_ms)
        self._frame_samples = sample_rate * frame_ms // 1000

    def process_reverse_stream(self, ref: np.ndarray) -> None:
        self._ref = ref

    def process_stream(self, mic: np.ndarray) -> np.ndarray:
        out = np.zeros(self._frame_samples, dtype=np.float32)
        ref = self._ref if hasattr(self, '_ref') else np.zeros_like(mic)
        self._lib.aec_process(
            self._handle,
            mic.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.aec_destroy(self._handle)


class AecProcessor:
    """Wraps WebRTC APM for echo cancellation."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 10):
        self._sample_rate = sample_rate
        self._frame_samples = int(sample_rate * frame_ms / 1000)

        # Try Python package first, then ctypes shim
        try:
            from unittest.mock import MagicMock
            from webrtc_audio_processing import AudioProcessingModule
            if isinstance(AudioProcessingModule, MagicMock):
                raise ImportError("mocked module")
            self._apm = AudioProcessingModule(
                sample_rate_hz=sample_rate, num_channels=1,
            )
            self._apm.enable_echo_cancellation(True)
            self._apm.enable_noise_suppression(True)
        except (ImportError, OSError):
            shim_path = os.path.join(
                os.path.expanduser("~"), ".local", "lib", "libaec_shim.so"
            )
            if not os.path.exists(shim_path):
                raise RuntimeError(
                    "No AEC backend available. Install webrtc-audio-processing-py "
                    "or build libaec_shim.so from the project's aec_shim/ directory."
                )
            self._apm = _CtypesApm(shim_path, sample_rate, frame_ms)

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def process_frame(
        self, mic_frame: np.ndarray, playback_ref: np.ndarray
    ) -> np.ndarray:
        self._apm.process_reverse_stream(playback_ref)
        clean = self._apm.process_stream(mic_frame)
        return clean
