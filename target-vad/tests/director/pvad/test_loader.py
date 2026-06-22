"""Tests for the pVAD ONNX loader + SpeakerFrame type.

The frozen-dataclass test always runs. The real-load test downloads/loads the
ONNX checkpoint and is skipped when it isn't cached (keeps CI green offline).
"""

import pytest

from modes.director.pvad.types import SpeakerFrame


def _pvad_cached():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="FireRedTeam/FireRedChat-pvad",
                        filename="pvad.onnx", local_files_only=True)
        return True
    except Exception:
        return False


requires_pvad = pytest.mark.skipif(
    not _pvad_cached(), reason="pvad.onnx not cached (run online once to fetch)")


def test_speaker_frame_is_frozen():
    f = SpeakerFrame(ts=1.0, is_target=True, confidence=0.9, rms=0.05)
    assert (f.ts, f.is_target, f.confidence, f.rms) == (1.0, True, 0.9, 0.05)
    with pytest.raises(Exception):
        f.is_target = False


@requires_pvad
def test_load_pvad_returns_cpu_streaming_session():
    from modes.director.pvad.loader import load_pvad
    sess = load_pvad()
    assert "CPUExecutionProvider" in sess.get_providers()
    names = {i.name for i in sess.get_inputs()}
    assert {"input_audio", "spkemb", "mel_buffer", "gru_buffer"} <= names
    out_names = {o.name for o in sess.get_outputs()}
    assert {"sigmoid_out", "mel_buffer_out", "gru_buffer_out"} <= out_names
