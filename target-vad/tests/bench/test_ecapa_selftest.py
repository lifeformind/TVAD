"""Tests for bench/ecapa_selftest.py numeric helpers (no audio/model needed)."""
import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_MOD = os.path.normpath(os.path.join(_HERE, "..", "..", "bench", "ecapa_selftest.py"))
_spec = importlib.util.spec_from_file_location("ecapa_selftest", _MOD)
est = importlib.util.module_from_spec(_spec)
sys.modules["ecapa_selftest"] = est
_spec.loader.exec_module(est)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestPairwiseCross:
    def test_pairwise_count_and_value(self):
        e = [_unit([1, 0, 0]), _unit([1, 0, 0]), _unit([0, 1, 0])]
        sims = est.pairwise(e)
        assert len(sims) == 3  # C(3,2)
        assert max(sims) == 1.0   # the two identical vectors
        assert min(sims) == 0.0   # orthogonal pair

    def test_cross_all_pairs(self):
        a = [_unit([1, 0, 0]), _unit([0, 1, 0])]
        b = [_unit([1, 0, 0])]
        sims = est.cross(a, b)
        assert len(sims) == 2
        assert sims[0] == 1.0 and sims[1] == 0.0


class TestStats:
    def test_stats_basic(self):
        st = est.stats([0.2, 0.4, 0.6])
        assert st["n"] == 3
        assert st["mean"] == pytest.approx(0.4)
        assert st["min"] == pytest.approx(0.2) and st["max"] == pytest.approx(0.6)

    def test_stats_empty(self):
        assert est.stats([]) == {"n": 0}


class TestSeparation:
    def test_separable(self):
        out = est.separation(within=[0.7, 0.8], crossed=[0.1, 0.2])
        assert "CLEANLY SEPARABLE" in out
        assert "midpoint" in out

    def test_overlapping(self):
        out = est.separation(within=[0.1, 0.5], crossed=[0.2, 0.6])
        assert "OVERLAPPING" in out

    def test_missing_group(self):
        assert "need both" in est.separation([], [0.1])
