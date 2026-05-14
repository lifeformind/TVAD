"""Tests for Silero VAD wrapper."""

import numpy as np
import pytest

from core.vad.silero_vad import SileroVAD, SpeechSegment


@pytest.fixture
def vad():
    config = {
        "sample_rate": 16000,
        "speech_threshold": 0.5,
        "min_speech_duration_ms": 300,
        "padding_ms": 200,
    }
    return SileroVAD(config)


def make_sine_wave(freq_hz: float, duration_s: float, sr: int = 16000) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration_s)) / sr
    return (0.3 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def chunk_audio(audio: np.ndarray, chunk_size: int = 480):
    """Yield chunks of audio."""
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) == chunk_size:
            yield chunk


class TestSileroVADBasic:
    def test_is_speech_returns_float(self, vad):
        """is_speech should return a float probability."""
        chunk = np.zeros(512, dtype=np.float32)
        prob = vad.is_speech(chunk)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_silence_low_probability(self, vad):
        """Silence should have low speech probability."""
        chunk = np.zeros(512, dtype=np.float32)
        prob = vad.is_speech(chunk)
        assert prob < 0.5, f"Silence scored {prob}, expected < 0.5"

    def test_sine_wave_low_probability(self, vad):
        """A pure sine wave (not speech) should generally not trigger VAD.

        Note: This is a heuristic test — Silero may occasionally give
        moderate scores to tonal signals. We test that it's not confidently
        classified as speech.
        """
        sine = make_sine_wave(440.0, 0.032)  # 32ms = 512 samples
        prob = vad.is_speech(sine[:512])
        # Sine waves typically score low but we use a lenient threshold
        assert prob < 0.8, f"Sine wave scored {prob}, expected < 0.8"

    def test_reset_clears_state(self, vad):
        """reset() should not raise and should allow reprocessing."""
        chunk = np.zeros(512, dtype=np.float32)
        vad.is_speech(chunk)
        vad.reset()
        prob = vad.is_speech(chunk)
        assert isinstance(prob, float)


class TestSileroVADStream:
    def test_silence_yields_no_segments(self, vad):
        """A stream of silence should yield no speech segments."""
        silence = make_silence(2.0)
        segments = list(vad.process_stream(chunk_audio(silence)))
        assert len(segments) == 0

    def test_segment_has_correct_fields(self, vad):
        """Any yielded segment should have all required fields."""
        # We can't guarantee a segment from synthetic audio,
        # so we test the dataclass directly
        seg = SpeechSegment(
            audio=np.zeros(16000, dtype=np.float32),
            start_ms=0.0,
            end_ms=1000.0,
            duration_ms=1000.0,
        )
        assert seg.duration_ms == 1000.0
        assert len(seg.audio) == 16000
