"""verify_before_serve math (Director-09 spec s5): scores two embeddings
against each other. The WakeGate (tests/director/test_wakegate.py,
TestVerifyBeforeServe) calls this with the two HALVES of the first speech
segment — same-utterance self-similarity, not a holdout enrollment utterance.
These tests exercise the pure cosine-threshold math the split-half check
relies on."""

import numpy as np
import pytest
from modes.director.verify import verify_before_serve


def _emb(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_matching_halves_pass():
    a = _emb(1.0, 0.05, 0.0)
    b = _emb(1.0, 0.0, 0.0)
    ok, score = verify_before_serve(a, b, threshold=0.80)
    assert ok is True
    assert score >= 0.80


def test_orthogonal_halves_refused():
    ok, score = verify_before_serve(_emb(1, 0), _emb(0, 1), threshold=0.80)
    assert ok is False
    assert score == pytest.approx(0.0, abs=1e-6)


def test_threshold_boundary_just_below_refused():
    # cosine ~0.78 < 0.80 -> refuse.
    a = _emb(1.0, 0.0)
    b = _emb(1.0, 0.80)   # cos = 1/sqrt(1+0.64) ~= 0.78
    ok, score = verify_before_serve(a, b, threshold=0.80)
    assert ok is False
    assert 0.75 < score < 0.80
