"""Streaming STT wrapper, re-backed onto openai-whisper (torch, CUDA).

faster-whisper / CTranslate2 has NO aarch64 CUDA wheel on this GB10 (DGX Spark)
and falls back to ~270ms CPU. This module keeps the StreamingStt class name and
async transcribe_segment interface (callers unchanged) but swaps the internals to
openai-whisper, and returns the canonical TranscriptResult(text, mean_word_prob)
(owned by modes/director/transcript.py, Plan 02) so the Director can RESTORE on
empty / low-confidence transcripts (spec Section 6).

See docs/notes/2026-06-22-stt-backend.md for the backend-selection verdict.
"""

import asyncio

import numpy as np

from modes.director.transcript import TranscriptResult  # canonical type (Plan 02)


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
