"""Sliding-window M-of-N decision smoother for noisy per-segment scores."""

from collections import deque
from typing import Deque


class DecisionSmoother:
    """Counts threshold-crossings in a sliding window of recent scores.

    Per-instance state. Use one smoother per session (kiosk creates one when
    a session starts, discards it on session end). The smoother is intentionally
    dumb — it just tracks rolling crossings; consumers decide what to do with
    the True/False signal it returns.
    """

    def __init__(self, window_size: int, min_matches: int, threshold: float):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_matches < 1 or min_matches > window_size:
            raise ValueError(
                f"min_matches must be in [1, window_size]={window_size}, got {min_matches}"
            )
        self.window_size = window_size
        self.min_matches = min_matches
        self.threshold = threshold
        self._scores: Deque[float] = deque(maxlen=window_size)

    def update(self, score: float) -> bool:
        """Append a score, return True if window has min_matches >= threshold."""
        self._scores.append(score)
        crossings = sum(1 for s in self._scores if s >= self.threshold)
        return crossings >= self.min_matches

    def reset(self) -> None:
        """Empty the window. Next update starts fresh."""
        self._scores.clear()
