"""Diarizer — thin wrapper around pyannote.audio's speaker-diarization-3.1 pipeline.

The pipeline is loaded lazily (first .diarize() call) because instantiation downloads
the model the first time and is slow even when cached. Subsequent calls reuse the
loaded pipeline.

Output is normalized from pyannote's Annotation object into a plain dict:
    {cluster_id: [(start_s, end_s), ...]}
sorted by start time within each cluster. Cluster IDs are pyannote's own strings
(e.g. "SPEAKER_00"), preserved as-is.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Diarizer:
    def __init__(self, pipeline_name: str, hf_token: str):
        if not hf_token:
            raise ValueError("Diarizer requires a non-empty HuggingFace token")
        self.pipeline_name = pipeline_name
        self.hf_token = hf_token
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline
        logger.info("Loading pyannote pipeline %s (first call may download model)", self.pipeline_name)
        self._pipeline = Pipeline.from_pretrained(self.pipeline_name, use_auth_token=self.hf_token)

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Run diarization on a mono float32 waveform.

        Returns a dict mapping cluster_id → list of (start_s, end_s) tuples sorted by start.
        Empty dict if pyannote finds no speech.
        """
        self._ensure_pipeline()

        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).unsqueeze(0).float()
        else:
            waveform = torch.from_numpy(audio).float()

        annotation = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        clusters: Dict[str, List[Tuple[float, float]]] = {}
        for segment, _, label in annotation.itertracks(yield_label=True):
            clusters.setdefault(label, []).append((float(segment.start), float(segment.end)))

        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda s: s[0])

        return clusters
