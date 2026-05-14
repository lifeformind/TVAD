"""Speaker verification — cosine similarity against enrolled voiceprints."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from speaker.enrollment_store import EnrollmentStore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class VerificationResult:
    """Result of speaker verification."""
    is_registered: bool
    matched_user: Optional[str]
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)


class SpeakerVerifier:
    """Verifies speaker identity against enrolled voiceprints."""

    def __init__(self, store: EnrollmentStore, threshold: float = 0.75):
        self.store = store
        self.threshold = threshold

    def verify(self, embedding: np.ndarray) -> VerificationResult:
        """Verify a speaker embedding against all enrolled voiceprints.

        Returns the best match if above threshold.
        """
        voiceprints = self.store.get_all()
        if not voiceprints:
            return VerificationResult(
                is_registered=False,
                matched_user=None,
                confidence=0.0,
                all_scores={},
            )

        scores = {}
        for username, vp in voiceprints.items():
            scores[username] = cosine_similarity(embedding, vp)

        best_user = max(scores, key=scores.get)
        best_score = scores[best_user]

        return VerificationResult(
            is_registered=best_score >= self.threshold,
            matched_user=best_user if best_score >= self.threshold else None,
            confidence=best_score,
            all_scores=scores,
        )

    def update_threshold(self, threshold: float):
        """Update the verification threshold."""
        self.threshold = threshold
