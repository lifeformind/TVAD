"""Accumulated-window ECAPA verifier for the Director-09 hijack/verify ladder
(off the hot path).

SafetyNet accumulates ONLY is_target audio, embeds every verify_window_ms
(108ms p95 ECAPA, run by the caller in an executor — fine off the hot path),
and runs the M-of-N DecisionSmoother to catch a different person taking over
for >1 window. It only produces a SpeakerWindowVerdict; the WARN/EJECT
decision lives in the reducer (_on_speaker_window_verdict), and the
post-eject quiet-hold lives in the WakeGate. ECAPA is unreliable on <2-3s
segments (MEMORY: ecapa-short-segment-unreliable.md), so a SINGLE miss never
ejects.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.speaker.decision_smoother import DecisionSmoother


@dataclass(frozen=True)
class SafetyVerdict:
    score: float
    smoother_ok: bool
    window_rms: float                # RMS of the exact window audio consumed
    norm_score: Optional[float] = None  # AS-Norm score (Task 7); None = normalizer off


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SafetyNet:
    def __init__(self, embedder, primary_embedding, *, verify_window_ms=2000,
                 threshold=0.30, window_size=3, min_matches=1, sr=16000,
                 normalizer=None, norm_decides: bool = False):
        self._embedder = embedder
        self._primary = np.asarray(primary_embedding, dtype=np.float32)
        self._need = int(sr * verify_window_ms / 1000)
        self._sr = sr
        self._smoother = DecisionSmoother(window_size, min_matches, threshold)
        self._threshold = threshold
        self._buf = np.zeros(0, dtype=np.float32)
        self._normalizer = normalizer          # AsNorm (Task 5/7); None = today's raw cosine
        self._norm_decides = norm_decides      # True (mode "on") -> norm_score feeds the smoother

    def accumulate(self, audio: np.ndarray, is_target: bool) -> None:
        if not is_target:
            return                                   # drop non-target audio
        self._buf = np.concatenate([self._buf, audio.astype(np.float32)])

    def maybe_verify(self) -> Optional[SafetyVerdict]:
        if self._buf.size < self._need:
            return None
        window = self._buf[: self._need]
        self._buf = self._buf[self._need:]           # consume the window
        emb = self._embedder.extract(window, sample_rate=self._sr)
        score = _cosine(emb, self._primary)
        norm_score = None
        if self._normalizer is not None:
            norm_score = float(self._normalizer.score(self._primary, emb))
        deciding = norm_score if (self._norm_decides and norm_score is not None) else score
        smoother_ok = self._smoother.update(deciding)
        window_rms = float(np.sqrt(np.mean(np.square(window))))
        # .score stays the raw cosine always (DIAG continuity); the smoother's
        # threshold meaning (raw vs normalized scale) is chosen by assembly.py.
        return SafetyVerdict(score=score, smoother_ok=smoother_ok,
                             window_rms=window_rms, norm_score=norm_score)
