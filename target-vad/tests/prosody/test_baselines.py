"""Tests for compute_baselines - pure aggregation over a segment list."""

import pytest

from modes.prosody import baselines


def _seg(speaker_id: str, prosody=None) -> dict:
    return {"speaker_id": speaker_id, "speaker": speaker_id, "prosody": prosody}


def _p(pitch_median=None, pitch_std=None, pitch_range=None,
       energy_db_mean=None, energy_db_range=None,
       speech_rate_wps=None, pause_ratio=None) -> dict:
    """Build a 7-field prosody dict - defaults to all-null."""
    return {
        "pitch_hz_median": pitch_median, "pitch_hz_std": pitch_std,
        "pitch_range_hz": pitch_range, "energy_db_mean": energy_db_mean,
        "energy_db_range": energy_db_range,
        "speech_rate_wps": speech_rate_wps, "pause_ratio": pause_ratio,
    }


class TestComputeBaselines:
    def test_two_speakers_three_segments_each(self):
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", _p(pitch_median=145, energy_db_mean=-23)),
            _seg("alice", _p(pitch_median=150, energy_db_mean=-21)),
            _seg("bob", _p(pitch_median=170, energy_db_mean=-22)),
            _seg("bob", _p(pitch_median=175, energy_db_mean=-20)),
            _seg("bob", _p(pitch_median=180, energy_db_mean=-18)),
        ]
        result = baselines.compute_baselines(segments)
        assert set(result.keys()) == {"alice", "bob"}
        # Alice: pitch median = 145; IQR = p75 - p25 = 147.5 - 142.5 = 5.0
        assert result["alice"]["pitch_hz_median"] == pytest.approx(145.0)
        assert result["alice"]["pitch_hz_iqr"] == pytest.approx(5.0)
        assert result["alice"]["energy_db_median"] == pytest.approx(-23.0)
        assert result["alice"]["segment_count"] == 3
        # Bob: pitch median = 175
        assert result["bob"]["pitch_hz_median"] == pytest.approx(175.0)
        assert result["bob"]["segment_count"] == 3

    def test_all_null_prosody_speaker_omitted(self):
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("bob", None),
            _seg("bob", _p()),  # all-null dict - bob has no non-null fields
        ]
        result = baselines.compute_baselines(segments)
        assert "alice" in result
        assert "bob" not in result

    def test_mixed_null_segments_partial_baseline(self):
        # Alice has 2 segments with prosody, 1 with prosody: null.
        # Baseline computed from the 2 valid ones; segment_count = 2.
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", None),
            _seg("alice", _p(pitch_median=160, energy_db_mean=-21)),
        ]
        result = baselines.compute_baselines(segments)
        assert result["alice"]["pitch_hz_median"] == pytest.approx(150.0)  # median of [140, 160]
        assert result["alice"]["segment_count"] == 2

    def test_iqr_constant_sequence_zero(self):
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
        ]
        result = baselines.compute_baselines(segments)
        assert result["alice"]["pitch_hz_iqr"] == 0.0
        assert result["alice"]["energy_db_iqr"] == 0.0
