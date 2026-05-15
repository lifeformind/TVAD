"""Speaker verification — cosine similarity against enrolled voiceprints."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from core.speaker.enrollment_store import EnrollmentStore


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
    """Result of speaker verification.

    matched_id is the stable storage key; matched_name is the display label
    looked up via EnrollmentStore.get_name(matched_id). Both are None when
    is_registered is False.
    """
    is_registered: bool
    matched_id: Optional[str]
    matched_name: Optional[str]
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
                matched_id=None,
                matched_name=None,
                confidence=0.0,
                all_scores={},
            )

        scores = {user_id: cosine_similarity(embedding, vp) for user_id, vp in voiceprints.items()}
        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]
        is_match = best_score >= self.threshold

        return VerificationResult(
            is_registered=is_match,
            matched_id=best_id if is_match else None,
            matched_name=self.store.get_name(best_id) if is_match else None,
            confidence=best_score,
            all_scores=scores,
        )

    def update_threshold(self, threshold: float) -> None:
        """Update the verification threshold."""
        self.threshold = threshold
