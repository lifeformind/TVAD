"""WhisperRunner — thin wrapper around faster-whisper's WhisperModel.

The model is loaded lazily on first transcribe() call: constructing the model
downloads weights on cold cache (~244 MB for small, ~1.5 GB for large-v3), so
deferring lets unit tests construct the runner without triggering a network call.

The two preset shortcuts 'small' and 'large' map to faster-whisper's bundled
model names ('small' and 'large-v3'). Any other string is passed through
unchanged so future models (e.g. 'large-v4', custom HuggingFace paths, local
CTranslate2 directories) work via config alone.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from faster_whisper import WhisperModel


_PRESET_ALIASES = {
    "small": "small",
    "large": "large-v3",
}


class WhisperRunner:
    """Transcribes a single audio slice via faster-whisper, returning (text, words)."""

    def __init__(
        self,
        model: str,
        language: Optional[str],
        device: str,
        compute_type: str,
    ):
        self._model_name = _PRESET_ALIASES.get(model, model)
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str,
    ) -> Tuple[str, List[Dict]]:
        """Transcribe a mono 16 kHz float32 audio slice.

        Returns (text, words) where:
            text   — concatenated text across whisper-emitted segments, leading/trailing space stripped
            words  — flat list of {start, end, word, probability} dicts in order

        Empty audio → ("", []).
        """
        model = self._ensure_model()
        kwargs = {
            "initial_prompt": initial_prompt or None,
            "word_timestamps": True,
        }
        if self._language is not None:
            kwargs["language"] = self._language
        segments_iter, _info = model.transcribe(audio, **kwargs)

        chunks: List[str] = []
        words: List[Dict] = []
        for seg in segments_iter:
            chunks.append(seg.text)
            if seg.words:
                for w in seg.words:
                    words.append({
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word.strip(),
                        "probability": float(w.probability),
                    })
        text = "".join(chunks).strip()
        return text, words
