"""Director-11 wiring: the owner bearing is calibrated from a lookback window
over the wake utterance (the handoff's first_segment has no timestamps), and
no tracker -> None -> the cone abstains all session."""

import time
from types import SimpleNamespace

import pytest

from modes.director.assembly import _calibrate_owner_bearing


class _FakeDoa:
    def __init__(self, median):
        self._median = median
        self.windows = []

    def median_between(self, t0, t1):
        self.windows.append((t0, t1))
        return self._median


def _first_segment(duration_ms=904.0):
    return SimpleNamespace(duration_ms=duration_ms)


def test_no_tracker_returns_none():
    assert _calibrate_owner_bearing(None, _first_segment()) is None


def test_bearing_is_median_over_wake_lookback_window():
    doa = _FakeDoa(median=97.0)
    before = time.monotonic()
    bearing = _calibrate_owner_bearing(doa, _first_segment(duration_ms=1500.0))
    after = time.monotonic()
    assert bearing == 97.0
    (t0, t1), = doa.windows
    # window ends "now" and reaches back max(dur_s, 1.0) + 1.0 = 2.5s
    assert before <= t1 <= after
    assert (t1 - t0) == pytest.approx(2.5, abs=0.01)


def test_short_seed_still_gets_a_full_lookback():
    doa = _FakeDoa(median=45.0)
    _calibrate_owner_bearing(doa, _first_segment(duration_ms=200.0))
    (t0, t1), = doa.windows
    assert (t1 - t0) == pytest.approx(2.0, abs=0.01)   # max(0.2, 1.0) + 1.0


def test_tracker_with_no_samples_returns_none():
    class _Empty:
        def median_between(self, t0, t1):
            return None
    assert _calibrate_owner_bearing(_Empty(), _first_segment()) is None
