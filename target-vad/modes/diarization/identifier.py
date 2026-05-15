"""ClusterIdentifier — match pyannote clusters to enrolled speakers via centroid cosine.

For each cluster, the identifier:
  1. Picks an evenly-spaced ≤ N-second sample of the cluster's segments.
  2. Slices and concatenates that audio from the full waveform.
  3. Embeds via the injected ECAPA EmbeddingExtractor → 192-dim L2-normalized vector.
     (EmbeddingExtractor already L2-normalizes; concat-then-embed-once produces a
     single embedding that is itself a kind of centroid because ECAPA pools internally.)
  4. Cosine-matches against all enrolled voiceprints; assigns the best-scoring id
     if score >= threshold else the literal string "unknown". Display name lookup
     is the caller's responsibility (via store.get_name(id) or SessionEnrollmentView).

If embedding fails for any cluster, that cluster is labeled "unknown" and processing
continues for the rest.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from core.speaker.verifier import cosine_similarity
from modes.diarization.sampling import sample_cluster_segments

logger = logging.getLogger(__name__)


class ClusterIdentifier:
    def __init__(
        self,
        embedder,
        enrollment_store,
        threshold: float = 0.55,
        max_sample_seconds: float = 30.0,
    ):
        self.embedder = embedder
        self.enrollment_store = enrollment_store
        self.threshold = threshold
        self.max_sample_seconds = max_sample_seconds

    def label_clusters(
        self,
        audio: np.ndarray,
        sample_rate: int,
        clusters: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, str]:
        """Return a {cluster_id: label} map. Label is enrolled name or 'unknown'.

        Args:
            audio: full waveform (float32 mono).
            sample_rate: must match what the embedder expects (16000).
            clusters: pyannote output as {cluster_id: [(start_s, end_s), ...]}.
        """
        if not clusters:
            return {}

        voiceprints = self.enrollment_store.get_all()
        if not voiceprints:
            # Nothing to compare against — every cluster is unknown without embedding.
            return {cid: "unknown" for cid in clusters}

        labels: Dict[str, str] = {}
        for cluster_id, segments in clusters.items():
            try:
                cluster_audio = self._extract_cluster_audio(audio, sample_rate, segments)
                embedding = self.embedder.extract(cluster_audio, sample_rate=sample_rate)
                labels[cluster_id] = self._best_label(embedding, voiceprints)
            except Exception as exc:  # pragma: no cover — exercised via mocks
                logger.warning("Embedding failed for cluster %s: %s — labeling unknown", cluster_id, exc)
                labels[cluster_id] = "unknown"

        return labels

    def _extract_cluster_audio(
        self, audio: np.ndarray, sample_rate: int, segments: List[Tuple[float, float]]
    ) -> np.ndarray:
        sampled = sample_cluster_segments(segments, max_seconds=self.max_sample_seconds)
        chunks = []
        for start_s, end_s in sampled:
            start_i = max(0, int(start_s * sample_rate))
            end_i = min(len(audio), int(end_s * sample_rate))
            if end_i > start_i:
                chunks.append(audio[start_i:end_i])
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def _best_label(self, embedding: np.ndarray, voiceprints: Dict[str, np.ndarray]) -> str:
        """Return the best-matching enrolled id, or 'unknown' if no match clears threshold."""
        best_id = None
        best_score = -1.0
        for id, vp in voiceprints.items():
            score = cosine_similarity(embedding, vp)
            if score > best_score:
                best_score = score
                best_id = id
        if best_id is not None and best_score >= self.threshold - 1e-6:
            return best_id
        return "unknown"
