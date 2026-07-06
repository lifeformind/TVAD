"""Verify-before-serve gate (Director-09 spec s5).

Scores two embeddings against each other; the WakeGate calls it with the two
HALVES of the first segment (same-utterance self-similarity — the only
comparison where 0.80 is statistically honest on short audio). Below threshold
-> refuse to serve (no session). Cross-utterance verification is the safety
net's window-1 job."""

import numpy as np


def verify_before_serve(primary: np.ndarray, holdout: np.ndarray,
                        threshold: float = 0.80) -> tuple:
    """Return (ok, score). ok == score >= threshold."""
    a = np.asarray(primary, dtype=np.float32)
    b = np.asarray(holdout, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    score = 0.0 if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))
    return (score >= threshold, score)
