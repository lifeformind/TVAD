"""Integration test for prosody analyzer on real speech.

Exercises analyze_segment end-to-end with real librosa.pyin + real
librosa.feature.rms (not stubbed). Asserts that pitch and energy
are in plausible ranges for an adult speech sample. Range-based
assertions make the test robust to small library version drift.

Slower than the synthetic-sine unit tests (~2-5 seconds for pyin's
numba JIT compile on first run) but worth the coverage gap closure.

Configuration note: frame_length_ms=50 (800 samples at 16kHz) is used
here rather than the unit-test default of 25ms (400 samples). With
fmin=80 Hz, pyin requires at least 2 periods of fmin to fit in the
frame (min ~401 samples); 25ms yields 0 voiced frames on real speech.
50ms satisfies the constraint and matches the minimum recommended
frame length for a male-voice pitch floor of 80 Hz.
"""

from pathlib import Path

import librosa
import pytest

from modes.prosody import analyzer


FIXTURE = Path(__file__).parent / "fixtures" / "speech_sample.wav"

DEFAULT_CFG = {
    "pitch_min_hz": 80,
    "pitch_max_hz": 400,
    "frame_length_ms": 50,   # 800 samples at 16kHz; satisfies pyin's 2-period requirement for fmin=80
    "hop_length_ms": 10,
}


def test_real_speech_pitch_and_energy_in_plausible_ranges():
    """analyze_segment on a 1s adult-speech clip produces plausible pitch + energy."""
    assert FIXTURE.exists(), f"Missing fixture: {FIXTURE}"

    # Load via librosa so the test exercises the loader path too.
    audio, sr = librosa.load(str(FIXTURE), sr=16000)
    assert sr == 16000
    duration = len(audio) / sr
    assert 0.5 <= duration <= 2.0  # ~1 second

    # Pretend we have one word covering the full clip — exercises the rate path too.
    words = [{"start": 0.0, "end": duration, "word": "x", "probability": 0.9}]

    result = analyzer.analyze_segment(audio, sr, words, duration, DEFAULT_CFG)

    # Pitch should be in adult-speech range.
    # Observed value: ~90 Hz (adult male) from Voice 001 short.wav at 5-6s.
    assert result["pitch_hz_median"] is not None, "Expected voiced frames in real speech"
    assert 60.0 < result["pitch_hz_median"] < 250.0, (
        f"Pitch median {result['pitch_hz_median']} Hz outside adult speech range"
    )
    assert result["pitch_hz_std"] is not None
    assert result["pitch_hz_std"] >= 0.0
    assert result["pitch_range_hz"] is not None
    assert result["pitch_range_hz"] >= 0.0

    # Energy should be negative dB (relative to amplitude 1.0) but not silent.
    # Observed value: ~-41.5 dB for this clip.
    assert result["energy_db_mean"] is not None
    assert -80.0 < result["energy_db_mean"] < 0.0, (
        f"Energy {result['energy_db_mean']} dB outside plausible speech range"
    )
    assert result["energy_db_range"] is not None
    assert result["energy_db_range"] >= 0.0

    # Rate fields should be populated (we provided words).
    assert result["speech_rate_wps"] is not None
    assert result["speech_rate_wps"] == pytest.approx(1.0 / duration, abs=0.1)
    assert result["pause_ratio"] is not None
    assert 0.0 <= result["pause_ratio"] <= 1.0
