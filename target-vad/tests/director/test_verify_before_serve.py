import numpy as np
import pytest
from modes.director.verify import verify_before_serve


def _emb(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_matching_holdout_passes():
    p = _emb(1.0, 0.05, 0.0)
    h = _emb(1.0, 0.0, 0.0)
    ok, score = verify_before_serve(p, h, threshold=0.80)
    assert ok is True
    assert score >= 0.80


def test_orthogonal_holdout_refused():
    ok, score = verify_before_serve(_emb(1, 0), _emb(0, 1), threshold=0.80)
    assert ok is False
    assert score == pytest.approx(0.0, abs=1e-6)


def test_threshold_boundary_just_below_refused():
    # cosine ~0.78 < 0.80 -> refuse.
    p = _emb(1.0, 0.0)
    h = _emb(1.0, 0.80)   # cos = 1/sqrt(1+0.64) ~= 0.78
    ok, score = verify_before_serve(p, h, threshold=0.80)
    assert ok is False
    assert 0.75 < score < 0.80
