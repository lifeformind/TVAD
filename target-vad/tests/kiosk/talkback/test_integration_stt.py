"""Layer 2 — STT integration test (requires openai-whisper + CUDA).

Re-backed off faster-whisper (no aarch64 CUDA wheel) onto openai-whisper; the
binding contract is now transcribe_segment(audio) -> TranscriptResult. Skips
(does not fail) without CUDA/openai-whisper so CI stays green. The richer
real-clip assertions live in test_stt_cuda.py.
"""

import numpy as np
import pytest

from modes.talkback.stt import StreamingStt, TranscriptResult


def _cuda_and_whisper_available():
    try:
        import torch
        import whisper  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return torch.cuda.is_available()


HAS_WHISPER = _cuda_and_whisper_available()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_WHISPER, reason="openai-whisper + CUDA required")
class TestSttIntegration:
    @pytest.mark.asyncio
    async def test_transcribe_speech_fixture(self):
        stt = StreamingStt(model="tiny", device="cuda")
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        out = await stt.transcribe_segment(audio)
        assert isinstance(out, TranscriptResult)
        assert isinstance(out.text, str)
        assert 0.0 <= out.mean_word_prob <= 1.0
