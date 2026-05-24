"""Tests for TTS wrapper — sentence to audio conversion."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.talkback.tts import TtsEngine


class FakeResult:
    def __init__(self, audio):
        self.audio = audio


def make_fake_tts(fake_audio):
    """Create a TtsEngine with a mock model that yields fake audio."""
    tts = TtsEngine.__new__(TtsEngine)
    tts._backend = "kokoro"
    tts._voice = "af_bella"
    tts._device = "cpu"
    tts._sample_rate = 24000
    tts._target_sample_rate = 16000
    tts._model = MagicMock()
    tts._model.return_value = iter([FakeResult(fake_audio)])
    return tts


class TestTtsEngine:
    @pytest.mark.asyncio
    async def test_synthesize_returns_float32_audio(self):
        fake_audio = np.random.randn(24000).astype(np.float32) * 0.1
        tts = make_fake_tts(fake_audio)

        audio = await tts.synthesize("Hello world.")
        assert audio.dtype == np.float32
        assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_synthesize_resamples_to_target_rate(self):
        fake_audio = np.random.randn(24000).astype(np.float32) * 0.1
        tts = make_fake_tts(fake_audio)

        audio = await tts.synthesize("Test.")
        expected_len = int(len(fake_audio) * 16000 / 24000)
        assert abs(len(audio) - expected_len) <= 2

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_returns_empty(self):
        tts = make_fake_tts(np.array([], dtype=np.float32))

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
