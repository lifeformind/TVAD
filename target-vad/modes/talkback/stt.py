"""Streaming STT wrapper around faster-whisper.

Accepts completed speech segments (from VAD) and returns final transcripts.
Runs faster-whisper inference in a thread pool to avoid blocking the async loop.
"""

import asyncio

import numpy as np


class StreamingStt:
    """Wraps faster-whisper for segment-level transcription."""

    def __init__(
        self,
        model: str = "large-v3",
        compute_type: str = "float16",
        device: str = "cuda",
    ):
        self._model_name = model
        self._compute_type = compute_type
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

    async def transcribe_segment(self, audio: np.ndarray) -> str:
        self._ensure_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            vad_filter=False,
        )
        parts = []
        for seg in segments:
            parts.append(seg.text.strip())
        return " ".join(parts).strip()
