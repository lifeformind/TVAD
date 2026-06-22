"""Real-CUDA integration test for the re-backed StreamingStt.

SKIPS (does not fail) when torch/CUDA or openai-whisper is unavailable, so CI on
CPU-only / x86 boxes stays green. Run on GB10 to verify the binding contract:
transcribe_segment(audio) -> TranscriptResult(text, mean_word_prob in [0,1]).
"""

import os

import numpy as np
import pytest

from modes.talkback.stt import StreamingStt, TranscriptResult

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_WAV = os.path.join(_REPO_ROOT, "self.wav")


def _cuda_and_whisper_available():
    try:
        import torch
        import whisper  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _cuda_and_whisper_available(),
    reason="CUDA + openai-whisper required (GB10 only)",
)


def _load_clip(secs=3.0, sr=16000):
    import soundfile as sf

    data, file_sr = sf.read(_WAV, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert file_sr == sr, f"{_WAV} is {file_sr}Hz"
    return data[: int(secs * sr)]


@requires_gpu
@pytest.mark.asyncio
async def test_transcribe_real_clip_returns_valid_result():
    if not os.path.exists(_WAV):
        pytest.skip(f"fixture {_WAV} missing")
    stt = StreamingStt(model="tiny", device="cuda")
    clip = _load_clip(secs=3.0)
    out = await stt.transcribe_segment(clip)
    assert isinstance(out, TranscriptResult)
    assert isinstance(out.text, str)
    assert out.text != ""  # 3s of real speech -> non-empty transcript
    assert 0.0 <= out.mean_word_prob <= 1.0
    assert out.mean_word_prob > 0.0  # real speech -> some word confidence


@requires_gpu
@pytest.mark.asyncio
async def test_transcribe_silence_low_or_empty():
    stt = StreamingStt(model="tiny", device="cuda")
    silence = np.zeros(48000, dtype=np.float32)
    out = await stt.transcribe_segment(silence)
    assert isinstance(out, TranscriptResult)
    # Silence either transcribes to empty (mean_word_prob == 0.0) or to
    # low-confidence garbage; either way the Director's empty/low-conf guard
    # (conf_floor=0.5) must be able to RESTORE on it.
    assert out.text == "" or out.mean_word_prob < 0.5
