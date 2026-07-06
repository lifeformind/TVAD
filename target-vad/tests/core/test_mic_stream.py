# tests/core/test_mic_stream.py
"""MicrophoneStream channel selection (Director-10): the ReSpeaker must be
opened at all 6 channels with only column 0 (the XVF-3000's processed,
hardware-AEC'd output) kept — a 1-channel open makes PipeWire downmix raw
capsules and the ch5 playback reference into the mono stream (measured
2026-07-06: >2x the own-TTS bleed of pure ch0)."""

import numpy as np

from core.audio.mic_stream import MicrophoneStream


def test_callback_keeps_only_the_configured_channel():
    m = MicrophoneStream({"channels": 6, "use_channel": 0, "chunk_size": 4})
    frame = np.arange(24, dtype=np.float32).reshape(4, 6)   # 4 samples x 6 ch
    m._audio_callback(frame, 4, None, None)
    chunk = m._buffer.popleft()
    np.testing.assert_array_equal(chunk, frame[:, 0])
    assert chunk.ndim == 1                                   # stream stays mono


def test_nonzero_use_channel_selects_that_column():
    m = MicrophoneStream({"channels": 6, "use_channel": 3})
    frame = np.arange(12, dtype=np.float32).reshape(2, 6)
    m._audio_callback(frame, 2, None, None)
    np.testing.assert_array_equal(m._buffer.popleft(), frame[:, 3])


def test_default_is_channel_zero_legacy_mono_unchanged():
    m = MicrophoneStream({"channels": 1})
    frame = np.full((4, 1), 0.5, dtype=np.float32)
    m._audio_callback(frame, 4, None, None)
    np.testing.assert_array_equal(m._buffer.popleft(), frame[:, 0])
