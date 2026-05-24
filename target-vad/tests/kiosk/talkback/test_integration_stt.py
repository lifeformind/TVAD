"""Layer 2 — STT integration test (requires faster-whisper + CUDA)."""

import numpy as np
import pytest

from unittest.mock import MagicMock
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = not isinstance(WhisperModel, MagicMock)
except (ImportError, OSError):
    HAS_WHISPER = False

from modes.talkback.stt import StreamingStt


@pytest.mark.integration
@pytest.mark.skipif(not HAS_WHISPER, reason="faster-whisper not installed")
class TestSttIntegration:
    @pytest.mark.asyncio
    async def test_transcribe_speech_fixture(self):
        stt = StreamingStt(model="tiny", compute_type="float32", device="cpu")
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        text = await stt.transcribe_segment(audio)
        assert isinstance(text, str)
