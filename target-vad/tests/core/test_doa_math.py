"""Circular angle math (Director-11): the 359/1 wraparound is 2 degrees."""

import pytest

from core.audio.doa_math import circular_distance, circular_median, circular_ema


def test_distance_plain():
    assert circular_distance(90.0, 110.0) == 20.0


def test_distance_wraps_the_short_way():
    assert circular_distance(359.0, 1.0) == 2.0
    assert circular_distance(1.0, 359.0) == 2.0


def test_distance_max_is_180():
    assert circular_distance(0.0, 180.0) == 180.0


def test_median_plain():
    assert circular_median([10.0, 20.0, 30.0]) == 20.0


def test_median_across_wraparound():
    # Naive sorting would put 358 last and pick a garbage middle; circular
    # median must land on the cluster center 0.
    assert circular_median([358.0, 0.0, 2.0]) == 0.0


def test_median_ignores_nothing_and_is_a_sample():
    # Definition: the SAMPLE angle minimizing summed circular distance.
    assert circular_median([97.0]) == 97.0
    assert circular_median([90.0, 100.0]) in (90.0, 100.0)


def test_median_empty_raises():
    with pytest.raises(ValueError):
        circular_median([])


def test_ema_plain():
    assert circular_ema(90.0, 100.0, 0.3) == pytest.approx(93.0)


def test_ema_shortest_arc_across_zero():
    # 350 -> 10 is +20 the short way; half-step lands on 0, not 180.
    assert circular_ema(350.0, 10.0, 0.5) == pytest.approx(0.0)
    assert circular_ema(10.0, 350.0, 0.5) == pytest.approx(0.0)


def test_ema_result_wrapped_to_0_360():
    assert 0.0 <= circular_ema(355.0, 15.0, 0.9) < 360.0
