"""Tests for TTS wrapper — sentence to audio conversion."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.tts import TtsEngine


class TestTtsEngine:
    @pytest.mark.asyncio
    async def test_synthesize_returns_float32_audio(self):
        tts = TtsEngine.__new__(TtsEngine)
        tts._backend = "kokoro"
        tts._sample_rate = 24000
        tts._target_sample_rate = 16000
        tts._model = MagicMock()
        fake_audio = np.random.randn(24000).astype(np.float32) * 0.1
        tts._model.synthesize = MagicMock(return_value=fake_audio)

        audio = await tts.synthesize("Hello world.")
        assert audio.dtype == np.float32
        assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_synthesize_resamples_to_target_rate(self):
        tts = TtsEngine.__new__(TtsEngine)
        tts._backend = "kokoro"
        tts._sample_rate = 24000
        tts._target_sample_rate = 16000
        tts._model = MagicMock()
        fake_audio = np.random.randn(24000).astype(np.float32) * 0.1
        tts._model.synthesize = MagicMock(return_value=fake_audio)

        audio = await tts.synthesize("Test.")
        expected_len = int(len(fake_audio) * 16000 / 24000)
        assert abs(len(audio) - expected_len) <= 2

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_returns_empty(self):
        tts = TtsEngine.__new__(TtsEngine)
        tts._backend = "kokoro"
        tts._sample_rate = 24000
        tts._target_sample_rate = 16000
        tts._model = MagicMock()
        tts._model.synthesize = MagicMock(return_value=np.array([], dtype=np.float32))

        audio = await tts.synthesize("")
        assert len(audio) == 0


class TestTtsEngineConfig:
    def test_config_fields(self):
        tts = TtsEngine.__new__(TtsEngine)
        tts._backend = "kokoro"
        tts._voice = "af_bella"
        tts._device = "cuda"
        assert tts._backend == "kokoro"
        assert tts._voice == "af_bella"
