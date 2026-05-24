"""TTS wrapper — sentence to audio synthesis.

Default: Kokoro-82M (GPU, 24kHz output, resampled to 16kHz).
Fallback: Piper (CPU, 22kHz output, resampled to 16kHz).
"""

import asyncio

import numpy as np
from scipy import signal


class TtsEngine:
    """Synthesize text to float32 audio at 16 kHz."""

    def __init__(
        self,
        backend: str = "kokoro",
        voice: str = "af_bella",
        device: str = "cuda",
        target_sample_rate: int = 16000,
    ):
        self._backend = backend
        self._voice = voice
        self._device = device
        self._target_sample_rate = target_sample_rate
        self._model = None

        if backend == "kokoro":
            self._sample_rate = 24000
        else:
            self._sample_rate = 22050

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._backend == "kokoro":
            import kokoro
            self._model = kokoro.KokoroTTS(voice=self._voice, device=self._device)
        else:
            raise ValueError(f"Unsupported TTS backend: {self._backend}")

    async def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.array([], dtype=np.float32)
        self._ensure_model()
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, self._synthesize_sync, text)
        return self._resample(audio)

    def _synthesize_sync(self, text: str) -> np.ndarray:
        return self._model.synthesize(text)

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
        if self._sample_rate == self._target_sample_rate:
            return audio
        num_samples = int(len(audio) * self._target_sample_rate / self._sample_rate)
        return signal.resample(audio, num_samples).astype(np.float32)
