"""Tests for streaming STT wrapper around faster-whisper."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.stt import StreamingStt


class FakeWhisperSegment:
    def __init__(self, text: str):
        self.text = text


class TestStreamingStt:
    @pytest.mark.asyncio
    async def test_transcribe_segment_returns_text(self):
        stt = StreamingStt.__new__(StreamingStt)
        stt._model = MagicMock()
        stt._model.transcribe = MagicMock(
            return_value=([FakeWhisperSegment(" hello world ")], {"language": "en"})
        )

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        text = await stt.transcribe_segment(audio)
        assert text == "hello world"

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_returns_empty(self):
        stt = StreamingStt.__new__(StreamingStt)
        stt._model = MagicMock()
        stt._model.transcribe = MagicMock(return_value=([], {"language": "en"}))

        audio = np.zeros(16000, dtype=np.float32)
        text = await stt.transcribe_segment(audio)
        assert text == ""

    @pytest.mark.asyncio
    async def test_transcribe_concatenates_segments(self):
        stt = StreamingStt.__new__(StreamingStt)
        stt._model = MagicMock()
        stt._model.transcribe = MagicMock(
            return_value=(
                [FakeWhisperSegment(" first"), FakeWhisperSegment(" second")],
                {"language": "en"},
            )
        )

        audio = np.random.randn(32000).astype(np.float32) * 0.1
        text = await stt.transcribe_segment(audio)
        assert text == "first second"


class TestStreamingSttInit:
    def test_config_stored(self):
        stt = StreamingStt.__new__(StreamingStt)
        stt._model_name = "large-v3"
        stt._compute_type = "float16"
        stt._device = "cuda"
        assert stt._model_name == "large-v3"
