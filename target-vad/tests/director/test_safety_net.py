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


def test_verdict_carries_window_rms():
    # embedder/primary fakes follow this file's existing pattern
    class _Emb:
        def extract(self, audio, sample_rate=16000):
            return np.ones(4, dtype=np.float32)
    net = SafetyNet(_Emb(), np.ones(4, dtype=np.float32),
                    verify_window_ms=100, sr=16000)
    net.accumulate(np.full(1600, 0.5, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v is not None
    assert abs(v.window_rms - 0.5) < 1e-6


class _Emb:
    """Fake embedder: extract -> unit ones (Task 7 AS-Norm wiring tests)."""
    def extract(self, audio, sample_rate=16000):
        return np.ones(4, dtype=np.float32)


class _FixedNorm:
    def score(self, enroll, test):
        return 7.0


def test_shadow_mode_logs_norm_but_raw_decides():
    emb = _Emb()          # existing fake: extract -> unit ones
    primary = emb.extract(np.zeros(4))
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.5,
                    normalizer=_FixedNorm(), norm_decides=False, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.norm_score == 7.0
    assert v.score == pytest.approx(1.0)      # raw cosine, and it decided
    assert v.smoother_ok is True


def test_shadow_mode_raw_decides_even_when_norm_disagrees():
    # Discriminates the norm_decides gate: norm score (-1.0) sits on the
    # OPPOSITE side of threshold 0.5 from raw cosine (1.0). If a regression
    # ever dropped the `if self._norm_decides` check (feeding norm_score to
    # the smoother unconditionally), this would flip to smoother_ok=False.
    emb = _Emb()
    primary = emb.extract(np.zeros(4))
    class _OppositeNorm:
        def score(self, enroll, test): return -1.0
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.5,
                    normalizer=_OppositeNorm(), norm_decides=False, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.norm_score == -1.0
    assert v.smoother_ok is True              # raw cosine 1.0 >= 0.5 decided, not norm


def test_on_mode_normalized_score_feeds_smoother():
    emb = _Emb()
    primary = emb.extract(np.zeros(4))
    class _LowNorm:
        def score(self, enroll, test): return -3.0
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.0,
                    normalizer=_LowNorm(), norm_decides=True, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.smoother_ok is False             # -3.0 < threshold 0.0 despite raw cosine 1.0


def test_no_normalizer_is_todays_behavior():
    emb = _Emb()
    net = SafetyNet(emb, emb.extract(np.zeros(4)), verify_window_ms=100, threshold=0.5)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v.norm_score is None and v.smoother_ok is True


# --- Task 8: running enrollment centroid (margin-guarded EMA) ---

class _DriftEmb:
    def extract(self, audio, sample_rate=16000):
        v = np.array([1.0, 0.3, 0.0, 0.0], dtype=np.float32)
        return v / np.linalg.norm(v)


def test_centroid_moves_toward_confident_matches():
    emb = _DriftEmb()
    primary = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.5,
                    update_alpha=0.5, update_margin=0.1, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    net.maybe_verify()
    assert net._primary[1] > 0.0                     # moved toward the window
    assert np.linalg.norm(net._primary) == pytest.approx(1.0, abs=1e-5)


def test_no_update_below_margin():
    class _WeakEmb:
        def extract(self, audio, sample_rate=16000):
            v = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)  # cosine 0.6 vs primary
            return v / np.linalg.norm(v)
    primary = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    net = SafetyNet(_WeakEmb(), primary, verify_window_ms=100, threshold=0.55,
                    update_alpha=0.5, update_margin=0.1, sr=16000)   # 0.6 < 0.55+0.1
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    net.maybe_verify()
    assert np.allclose(net._primary, primary)


def test_alpha_zero_never_updates():
    # same _DriftEmb as above, update_alpha left at default 0.0
    emb = _DriftEmb()
    primary = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    net = SafetyNet(emb, primary, verify_window_ms=100, threshold=0.5,
                    update_margin=0.1, sr=16000)
    net.accumulate(np.ones(1600, dtype=np.float32), is_target=True)
    net.maybe_verify()
    assert np.allclose(net._primary, primary)
