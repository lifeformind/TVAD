"""Layer 2 — TTS integration test (requires Kokoro installed)."""

import numpy as np
import pytest

from unittest.mock import MagicMock
try:
    import kokoro
    HAS_KOKORO = not isinstance(kokoro, MagicMock)
except (ImportError, OSError):
    HAS_KOKORO = False

from modes.talkback.tts import TtsEngine


@pytest.mark.integration
@pytest.mark.skipif(not HAS_KOKORO, reason="kokoro not installed")
class TestTtsIntegration:
    @pytest.mark.asyncio
    async def test_synthesize_sentence(self):
        tts = TtsEngine(backend="kokoro", voice="af_bella", device="cpu")
        audio = await tts.synthesize("Hello, this is a test.")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        assert 8000 < len(audio) < 80000
