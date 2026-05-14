"""ECAPA-TDNN speaker embedding extractor via SpeechBrain."""

import os
import sys
from typing import Optional

import numpy as np
import torch


class EmbeddingExtractor:
    """Extracts 192-dim speaker embeddings using ECAPA-TDNN."""

    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/target-vad/speechbrain")
    MIN_DURATION_SAMPLES = 12800  # 800ms at 16kHz

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._model = None

    def _ensure_model(self):
        """Lazy-load the SpeechBrain ECAPA-TDNN model."""
        if self._model is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        kwargs = {
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "savedir": self.cache_dir,
            "run_opts": {"device": "cpu"},
        }
        # Windows without developer mode can't create symlinks — use copy strategy
        if sys.platform == "win32":
            from speechbrain.utils.fetching import LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY

        self._model = EncoderClassifier.from_hparams(**kwargs)

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 192-dim L2-normalized embedding from audio.

        Args:
            audio: float32 mono audio signal
            sample_rate: sample rate (must be 16kHz)

        Returns:
            192-dim L2-normalized numpy embedding vector
        """
        self._ensure_model()

        audio = self._pad_if_short(audio, self.MIN_DURATION_SAMPLES)

        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        with torch.no_grad():
            embedding = self._model.encode_batch(waveform)

        emb = embedding.squeeze().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def _pad_if_short(self, audio: np.ndarray, min_samples: int) -> np.ndarray:
        """Pad short audio with reflected signal to reach minimum length."""
        if len(audio) >= min_samples:
            return audio
        # Reflect-pad to reach minimum length
        pad_needed = min_samples - len(audio)
        if len(audio) == 0:
            return np.zeros(min_samples, dtype=np.float32)
        padded = np.pad(audio, (0, pad_needed), mode="reflect")
        return padded
