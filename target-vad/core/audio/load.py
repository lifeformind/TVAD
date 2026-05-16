"""Shared audio loader - read a WAV file and return mono float32 at 16 kHz.

Used by diarize.py (S1), transcribe.py (Phase 2A), and prosody.py (Phase 4).
The function was duplicated in those entry points pre-Phase-4; this module
deduplicates it so all consumers get identical resampling and channel-mixing
semantics.
"""

from math import gcd

import numpy as np
import soundfile as sf


def load_audio_as_mono16k(path: str) -> np.ndarray:
    """Read a WAV file and return mono float32 at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = gcd(sr, 16000)
        audio = resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
    return audio.astype(np.float32, copy=False)
