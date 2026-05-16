"""Tests for the prosody analyzer pure function."""

import numpy as np
import pytest

from modes.prosody import analyzer


SR = 16000
DEFAULT_CFG = {
    "pitch_min_hz": 80,
    "pitch_max_hz": 400,
    "frame_length_ms": 25,
    "hop_length_ms": 10,
}


def _sine(freq_hz: float, duration_s: float, sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    """Generate a pure sine tone at given frequency."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _word(start: float, end: float, w: str = "x") -> dict:
    return {"start": start, "end": end, "word": w, "probability": 0.9}


class TestAnalyzeSegment:
    def test_pure_200hz_sine_centered_pitch(self):
        audio = _sine(200.0, 1.5)
        result = analyzer.analyze_segment(audio, SR, words=[_word(0, 1.5)], segment_duration=1.5, cfg=DEFAULT_CFG)
        assert result["pitch_hz_median"] == pytest.approx(200.0, abs=5.0)
        assert result["pitch_hz_std"] < 5.0
        assert result["pitch_range_hz"] < 5.0

    def test_silence_gives_null_pitch_and_low_energy(self):
        audio = np.zeros(int(SR * 1.0), dtype=np.float32)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=1.0, cfg=DEFAULT_CFG)
        assert result["pitch_hz_median"] is None
        assert result["pitch_hz_std"] is None
        assert result["pitch_range_hz"] is None
        # librosa.amplitude_to_db on zero RMS returns very low dB (clipped near -80 to -100)
        assert result["energy_db_mean"] is not None
        assert result["energy_db_mean"] < -50.0

    def test_concatenated_sines_reflect_range(self):
        # 1s of 100Hz then 1s of 300Hz - non-harmonic pair so pyin detects both distinctly.
        # (200+400Hz collapses: pyin folds 400Hz down to 200Hz subharmonic.)
        audio = np.concatenate([_sine(100.0, 1.0), _sine(300.0, 1.0)])
        result = analyzer.analyze_segment(audio, SR, words=[_word(0, 2.0)], segment_duration=2.0, cfg=DEFAULT_CFG)
        # 5th-95th percentile spans most of the range; allow generous tolerance for pyin noise.
        assert result["pitch_range_hz"] > 100.0

    def test_empty_words_gives_null_rate_fields(self):
        audio = _sine(200.0, 1.0)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=1.0, cfg=DEFAULT_CFG)
        assert result["speech_rate_wps"] is None
        assert result["pause_ratio"] is None

    def test_three_words_over_two_seconds_rate_and_pause(self):
        # 3 words spanning 0-0.3, 0.4-0.7, 0.8-1.1 (total word duration = 0.9s, segment = 2.0s)
        words = [_word(0.0, 0.3), _word(0.4, 0.7), _word(0.8, 1.1)]
        audio = _sine(200.0, 2.0)
        result = analyzer.analyze_segment(audio, SR, words=words, segment_duration=2.0, cfg=DEFAULT_CFG)
        assert result["speech_rate_wps"] == pytest.approx(1.5)  # 3 words / 2.0s
        assert result["pause_ratio"] == pytest.approx((2.0 - 0.9) / 2.0)  # 0.55

    def test_words_exceeding_segment_duration_clamps_pause_to_zero(self):
        # Total word duration > segment duration -> pause_ratio clamped to 0.0
        words = [_word(0.0, 1.5), _word(1.0, 2.5)]  # 1.5 + 1.5 = 3.0s across 2.0s segment
        audio = _sine(200.0, 2.0)
        result = analyzer.analyze_segment(audio, SR, words=words, segment_duration=2.0, cfg=DEFAULT_CFG)
        assert result["pause_ratio"] == 0.0

    def test_zero_length_audio_chunk(self):
        audio = np.array([], dtype=np.float32)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=0.0, cfg=DEFAULT_CFG)
        assert result["pitch_hz_median"] is None
        assert result["pitch_hz_std"] is None
        assert result["pitch_range_hz"] is None
        assert result["energy_db_mean"] is None
        assert result["energy_db_range"] is None
        assert result["speech_rate_wps"] is None
        assert result["pause_ratio"] is None

    def test_pitch_config_range_honored(self):
        # Configured 50-100 Hz range; 200 Hz sine is above fmax.
        # pyin clamps detected pitch to fmax (100.0) rather than returning NaN, so
        # the median should be <= fmax and not at the true frequency of 200 Hz.
        cfg = {"pitch_min_hz": 50, "pitch_max_hz": 100, "frame_length_ms": 25, "hop_length_ms": 10}
        audio = _sine(200.0, 1.0)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=1.0, cfg=cfg)
        # Either null (no voiced frames) or clamped to fmax -- never at the true 200 Hz.
        if result["pitch_hz_median"] is not None:
            assert result["pitch_hz_median"] <= 100.0
