"""Verify-before-serve gate (spec section 7).

Scores the finalized primary embedding against a holdout utterance embedding
captured BEFORE finalize_enrollment deleted the per-utterance file
(enrollment_store.holdout_utterance_embedding). Below threshold -> the Director
refuses to start (return to IDLE, re-enroll). 0.80 matches the ~2% EER operating
point on >=5s cumulative enrollment audio.
"""

import numpy as np


def verify_before_serve(primary: np.ndarray, holdout: np.ndarray,
                        threshold: float = 0.80) -> tuple:
    """Return (ok, score). ok == score >= threshold."""
    a = np.asarray(primary, dtype=np.float32)
    b = np.asarray(holdout, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    score = 0.0 if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))
    return (score >= threshold, score)
