"""Tests for VADStream — the per-session ONNX pVAD driver.

Uses a fake onnxruntime-like session (returns a fixed sigmoid prob, echoes the
state buffers) so the logic runs without the real model.
"""

import numpy as np
import pytest

from modes.director.pvad.stream import VADStream
from modes.director.pvad.types import SpeakerFrame

SR = 16000


class _FakeSession:
    """Mimics an onnxruntime InferenceSession.run(output_names, feeds)."""

    def __init__(self, prob):
        self._prob = prob
        self.calls = 0
        self.seen_spk = None

    def run(self, output_names, feeds):
        self.calls += 1
        self.seen_spk = feeds["spkemb"]
        # echo state back as *_out (streaming), return the fixed prob
        return [np.array([[self._prob]], dtype=np.float32),
                feeds["mel_buffer"], feeds["gru_buffer"]]


def _emb():
    return np.ones(192, dtype=np.float32)


def test_update_speaker_required_before_push():
    vs = VADStream(_FakeSession(0.9))
    with pytest.raises(RuntimeError):
        vs.push(np.zeros(3200, dtype=np.float32), ts=0.0)


def test_high_prob_is_target_true():
    vs = VADStream(_FakeSession(0.9), agg_ms=50, threshold=0.5)
    vs.update_speaker(_emb())
    out = vs.push(np.ones(int(SR * 0.200), dtype=np.float32), ts=1.0)
    assert out and all(isinstance(f, SpeakerFrame) for f in out)
    assert all(f.is_target for f in out)
    assert all(f.confidence == pytest.approx(0.9) for f in out)
    assert all(f.rms > 0.0 and f.ts == 1.0 for f in out)


def test_low_prob_is_target_false():
    vs = VADStream(_FakeSession(0.1), agg_ms=50, threshold=0.5)
    vs.update_speaker(_emb())
    out = vs.push(np.ones(int(SR * 0.200), dtype=np.float32), ts=1.0)
    assert out and all(not f.is_target for f in out)


def test_aggregation_groups_frames():
    # 200ms @ 10ms/frame = 20 frames; aggregated to 50ms -> ~4 SpeakerFrames.
    vs = VADStream(_FakeSession(0.9), agg_ms=50, threshold=0.5)
    vs.update_speaker(_emb())
    out = vs.push(np.ones(int(SR * 0.200), dtype=np.float32), ts=0.0)
    assert 3 <= len(out) <= 5
    # 20 frames -> exactly 20 session.run calls
    assert vs._sess.calls == 20


def test_speaker_embedding_is_l2_normalized():
    sess = _FakeSession(0.9)
    vs = VADStream(sess)
    vs.update_speaker(np.full(192, 3.0, dtype=np.float32))
    vs.push(np.ones(160, dtype=np.float32), ts=0.0)
    assert sess.seen_spk.shape == (1, 192)
    assert float(np.linalg.norm(sess.seen_spk)) == pytest.approx(1.0, abs=1e-5)


def test_streaming_is_continuous_across_pushes():
    # Non-frame-aligned splits must not drop or duplicate audio: the leftover
    # tail carries to the next push.
    one = VADStream(_FakeSession(0.9)); one.update_speaker(_emb())
    n_single = one._sess.calls  # 0
    one.push(np.ones(3200, dtype=np.float32), ts=0.0)
    total_single = one._sess.calls  # 20

    split = VADStream(_FakeSession(0.9)); split.update_speaker(_emb())
    split.push(np.ones(170, dtype=np.float32), ts=0.0)   # 1 frame, 10 leftover
    split.push(np.ones(3030, dtype=np.float32), ts=0.0)  # +10 buffered = 3040 -> 19 frames
    assert split._sess.calls == total_single  # 20 total, none dropped


def test_reset_clears_state():
    vs = VADStream(_FakeSession(0.9))
    vs.update_speaker(_emb())
    vs.push(np.ones(170, dtype=np.float32), ts=0.0)
    vs.reset()
    with pytest.raises(RuntimeError):
        vs.push(np.ones(160, dtype=np.float32), ts=0.0)


def _pvad_cached():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="FireRedTeam/FireRedChat-pvad",
                        filename="pvad.onnx", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pvad_cached(), reason="pvad.onnx not cached")
def test_real_onnx_session_streams_and_threads_state():
    # The fake can't catch an output-name/order mismatch; the real session can.
    from modes.director.pvad.loader import load_pvad
    vs = VADStream(load_pvad(), agg_ms=50, threshold=0.5)
    spk = np.random.RandomState(0).randn(192).astype(np.float32)
    vs.update_speaker(spk)
    out = vs.push(np.zeros(int(SR * 0.200), dtype=np.float32), ts=2.0)
    assert 3 <= len(out) <= 5
    assert all(isinstance(f, SpeakerFrame) for f in out)
    assert all(0.0 <= f.confidence <= 1.0 for f in out)  # finite prob in [0,1]
