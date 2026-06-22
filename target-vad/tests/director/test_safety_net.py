import numpy as np
import pytest
from modes.director.safety_net import SafetyNet, SafetyVerdict


class _FakeEmbedder:
    """Returns a fixed embedding so cosine vs primary is deterministic."""
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)
    def extract(self, audio, sample_rate=16000):
        return self._vec


def _emb(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_non_target_audio_is_dropped():
    sn = SafetyNet(_FakeEmbedder(_emb(1, 0)), _emb(1, 0), verify_window_ms=100)
    sn.accumulate(np.ones(16000, dtype=np.float32), is_target=False)
    assert sn.maybe_verify() is None


def test_matching_speaker_passes_smoother():
    primary = _emb(1, 0)
    sn = SafetyNet(_FakeEmbedder(_emb(1, 0)), primary,
                   verify_window_ms=100, threshold=0.30,
                   window_size=3, min_matches=1)
    sn.accumulate(np.ones(3200, dtype=np.float32), is_target=True)  # 200ms
    v = sn.maybe_verify()
    assert isinstance(v, SafetyVerdict)
    assert v.score == pytest.approx(1.0)
    assert v.smoother_ok is True


def test_mismatched_speaker_fails_score():
    primary = _emb(1, 0)
    sn = SafetyNet(_FakeEmbedder(_emb(0, 1)), primary,    # orthogonal
                   verify_window_ms=100, threshold=0.30,
                   window_size=3, min_matches=1)
    sn.accumulate(np.ones(3200, dtype=np.float32), is_target=True)
    v = sn.maybe_verify()
    assert v.score == pytest.approx(0.0, abs=1e-6)
    assert v.smoother_ok is False


def test_buffer_resets_after_verify():
    # verify_window_ms=200 -> need == 3200 samples; accumulate exactly one window,
    # so after it's consumed the next maybe_verify has nothing left.
    sn = SafetyNet(_FakeEmbedder(_emb(1, 0)), _emb(1, 0), verify_window_ms=200)
    sn.accumulate(np.ones(3200, dtype=np.float32), is_target=True)
    assert sn.maybe_verify() is not None        # one full window -> verdict
    assert sn.maybe_verify() is None             # window consumed -> not ready
