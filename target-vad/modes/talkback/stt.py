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
    """Segment-level STT over openai-whisper (torch, CUDA on GB10).

    Keeps the original async transcribe_segment interface so Plan 02's SttWorker
    is unchanged, but returns TranscriptResult(text, mean_word_prob) instead of a
    bare str. faster-whisper is gone (no aarch64 CUDA wheel on this box).
    """

    def __init__(
        self,
        model: str = "base.en",
        device: str = "cuda",
    ):
        self._model_name = model
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import whisper  # openai-whisper (torch-native), NOT the CTranslate2 backend

        self._model = whisper.load_model(self._model_name, device=self._device)

    @staticmethod
    def _mean_word_prob(result: dict) -> float:
        """Mean per-word probability over the transcript; 0.0 if no words."""
        probs = []
        for seg in result.get("segments", []) or []:
            for w in seg.get("words", []) or []:
                p = w.get("probability")
                if p is not None:
                    probs.append(float(p))
        if not probs:
            return 0.0
        mean = sum(probs) / len(probs)
        return max(0.0, min(1.0, mean))

    async def transcribe_segment(self, audio: np.ndarray) -> "TranscriptResult":
        self._ensure_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> "TranscriptResult":
        result = self._model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            fp16=(self._device == "cuda"),
        )
        text = (result.get("text", "") or "").strip()
        mean_word_prob = self._mean_word_prob(result)
        return TranscriptResult(text=text, mean_word_prob=mean_word_prob)
