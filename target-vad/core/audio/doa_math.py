"""Circular (0-360 degree) angle math for the DOA cone vote (Director-11).

Dependency-free on purpose: the pure reducer imports these without pulling
pyusb, and DoaTracker uses the same definitions so "distance" means one
thing everywhere."""


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def circular_median(angles) -> float:
    """The sample angle minimizing summed circular distance to all samples.
    O(n^2), fine for the tracker's <=600-sample buffer. Raises ValueError on
    empty input — callers translate "no samples" to None before calling."""
    angles = list(angles)
    if not angles:
        raise ValueError("circular_median of empty sequence")
    return min(angles, key=lambda c: sum(circular_distance(c, a) for a in angles))


def circular_ema(current: float, sample: float, alpha: float) -> float:
    """EMA along the shortest arc from current toward sample, wrapped to [0, 360)."""
    delta = ((sample - current) + 180.0) % 360.0 - 180.0
    return (current + alpha * delta) % 360.0
