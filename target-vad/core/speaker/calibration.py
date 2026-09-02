"""AS-Norm: adaptive symmetric score normalization for speaker scores.

Raw cosine on the far-field array channel is miscalibrated (owner band
0.23-0.47 vs stranger 0.07 live, forcing threshold 0.15 — see spec
Appendix C). Normalizing each score against a cohort of imposter
embeddings pushed through the same channel converts it to a z-like score
with a much wider genuine/imposter margin. Standard in VoxSRC-winning
systems (DKU-MSXF 2023 et al.).
"""
import numpy as np


class AsNorm:
    def __init__(self, cohort: np.ndarray, top_k: int = 50):
        # cohort: (N, D) L2-normalized imposter embeddings (build_cohort.py)
        self._cohort = np.asarray(cohort, dtype=np.float32)
        self._top_k = max(1, min(int(top_k), len(self._cohort)))

    def _stats(self, emb: np.ndarray) -> tuple[float, float]:
        scores = self._cohort @ emb
        top = np.sort(scores)[-self._top_k:]
        return float(top.mean()), float(top.std() + 1e-6)

    def raw(self, enroll: np.ndarray, test: np.ndarray) -> float:
        return float(np.dot(enroll, test))

    def score(self, enroll: np.ndarray, test: np.ndarray) -> float:
        s = self.raw(enroll, test)
        mu_e, sd_e = self._stats(enroll)
        mu_t, sd_t = self._stats(test)
        return 0.5 * ((s - mu_e) / sd_e + (s - mu_t) / sd_t)
