"""Tests for DecisionSmoother — sliding-window M-of-N threshold counter."""

import pytest

from core.speaker.decision_smoother import DecisionSmoother


@pytest.fixture
def smoother():
    """Default 2-of-3 at threshold 0.6 (matches kiosk default)."""
    return DecisionSmoother(window_size=3, min_matches=2, threshold=0.6)


class TestDecisionSmootherBasics:
    def test_first_update_below_min_matches(self, smoother):
        """One score above threshold, but min_matches=2 not yet hit."""
        assert smoother.update(0.9) is False

    def test_two_above_threshold_triggers(self, smoother):
        """Two consecutive above-threshold scores hit M=2 in window=3."""
        assert smoother.update(0.9) is False
        assert smoother.update(0.8) is True

    def test_below_threshold_does_not_count(self, smoother):
        """Scores below threshold are not crossings."""
        assert smoother.update(0.5) is False
        assert smoother.update(0.4) is False
        assert smoother.update(0.3) is False

    def test_mixed_window(self, smoother):
        """Three scores: 0.9, 0.5, 0.7 → 2 crossings → True."""
        assert smoother.update(0.9) is False  # 1 crossing
        assert smoother.update(0.5) is False  # still 1 crossing
        assert smoother.update(0.7) is True   # 2 crossings

    def test_window_slides(self, smoother):
        """When window fills, oldest score drops out."""
        # Fill window with one crossing + two non-crossings
        smoother.update(0.9)  # crossing — window: [0.9]
        smoother.update(0.5)  # window: [0.9, 0.5]
        smoother.update(0.4)  # window: [0.9, 0.5, 0.4] — 1 crossing
        # Next call evicts 0.9
        assert smoother.update(0.4) is False  # window: [0.5, 0.4, 0.4] — 0 crossings


class TestDecisionSmootherEdgeCases:
    def test_threshold_inclusive(self):
        """A score exactly equal to threshold is a crossing (>=)."""
        s = DecisionSmoother(window_size=2, min_matches=2, threshold=0.5)
        s.update(0.5)
        assert s.update(0.5) is True

    def test_min_matches_one(self):
        """min_matches=1 fires on the very first crossing."""
        s = DecisionSmoother(window_size=3, min_matches=1, threshold=0.5)
        assert s.update(0.6) is True

    def test_reset_clears_window(self, smoother):
        """reset() empties the window — fresh M-of-N from zero."""
        smoother.update(0.9)
        smoother.update(0.9)  # would be True
        smoother.reset()
        assert smoother.update(0.9) is False  # only 1 in window now


class TestDecisionSmootherKioskDefaults:
    def test_realistic_self_speech_pattern(self):
        """Self-speech scores from the C10 (0.40, 0.72, 0.60): 2-of-3 at 0.55 fires on segment 3."""
        # Note: kiosk default threshold is 0.60, but the spec also documents 0.55 as a
        # tunable lower bound. This test uses 0.55 to mirror the spec's example math.
        s = DecisionSmoother(window_size=3, min_matches=2, threshold=0.55)
        assert s.update(0.40) is False  # below
        assert s.update(0.72) is False  # 1 crossing
        assert s.update(0.60) is True   # 2 crossings (0.72 + 0.60)

    def test_three_random_ambient_segments_silent(self):
        """Three sub-threshold ambient scores → never triggers."""
        s = DecisionSmoother(window_size=3, min_matches=2, threshold=0.6)
        for score in (0.21, 0.43, 0.30):
            assert s.update(score) is False
