"""Tests for the centroid-sampling helper."""

import pytest

from modes.diarization.sampling import sample_cluster_segments


class TestSampleClusterSegments:
    def test_short_cluster_returns_all_segments(self):
        """If total duration is <= max_seconds, return everything unchanged."""
        segments = [(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)]  # total 8s
        result = sample_cluster_segments(segments, max_seconds=30.0)
        assert result == segments

    def test_exact_max_returns_all(self):
        """Total duration exactly equal to max_seconds — return all."""
        segments = [(0.0, 15.0), (20.0, 35.0)]  # total 30s
        result = sample_cluster_segments(segments, max_seconds=30.0)
        assert result == segments

    def test_long_cluster_samples_evenly(self):
        """Total > max_seconds: pick evenly-spaced segments until cap reached."""
        # 12 segments of 5s each = 60s total, cap 30s → 6 segments expected
        segments = [(i * 5.0, i * 5.0 + 5.0) for i in range(12)]
        result = sample_cluster_segments(segments, max_seconds=30.0)
        # Total duration of result must be <= max_seconds
        total = sum(end - start for start, end in result)
        assert total <= 30.0
        # Should not be empty
        assert len(result) > 0
        # Result is a subset of the input, in original order
        for seg in result:
            assert seg in segments
        # Result is sorted by start time
        assert result == sorted(result, key=lambda s: s[0])

    def test_long_cluster_evenly_spaced(self):
        """Selected indices should be roughly evenly spaced across the cluster."""
        # 10 segments of 6s each = 60s total, cap 30s → 5 segments expected
        segments = [(i * 6.0, i * 6.0 + 6.0) for i in range(10)]
        result = sample_cluster_segments(segments, max_seconds=30.0)
        selected_indices = [segments.index(s) for s in result]
        # Indices should be spread across the range (not all clustered at the start)
        assert min(selected_indices) <= 2
        assert max(selected_indices) >= 7

    def test_empty_input(self):
        """Empty segment list returns empty."""
        assert sample_cluster_segments([], max_seconds=30.0) == []

    def test_single_segment_under_cap(self):
        assert sample_cluster_segments([(0.0, 10.0)], max_seconds=30.0) == [(0.0, 10.0)]

    def test_single_segment_over_cap(self):
        """A single segment longer than cap is still returned as-is (we don't slice within a segment)."""
        # Design choice: we sample at segment granularity, not sub-segment.
        # ECAPA can ingest any length; centroid sampling is about which segments to include.
        result = sample_cluster_segments([(0.0, 60.0)], max_seconds=30.0)
        assert result == [(0.0, 60.0)]
