import numpy as np
from core.speaker.calibration import AsNorm


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)

def _cohort(n=100, d=8, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, d)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def test_same_embedding_scores_higher_than_orthogonal():
    norm = AsNorm(_cohort())
    a = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    b = _unit([0, 1, 0, 0, 0, 0, 0, 0])
    assert norm.score(a, a) > norm.score(a, b)

def test_normalized_scale_is_zscore_like():
    # A raw score equal to the cohort mean should normalize to ~0;
    # a genuine match lands far above.
    norm = AsNorm(_cohort())
    a = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    assert norm.score(a, a) > 3.0          # many std devs above imposters

def test_top_k_larger_than_cohort_is_clamped():
    small = _cohort(n=5)
    norm = AsNorm(small, top_k=50)
    a = _unit([1, 1, 0, 0, 0, 0, 0, 0])
    assert np.isfinite(norm.score(a, a))

def test_raw_matches_cosine():
    norm = AsNorm(_cohort())
    a = _unit([1, 2, 3, 0, 0, 0, 0, 0])
    b = _unit([3, 2, 1, 0, 0, 0, 0, 0])
    assert abs(norm.raw(a, b) - float(np.dot(a, b))) < 1e-6
