"""pVAD worker: VADStream wrapper with the crash-fallback crowd filter.

If the pVAD stream raises, FOCUS degrades to the RMS proximity gate
(is_target := rms >= proximity_rms) — identical to the spike-failure degraded
mode — and a worker_failed event is emitted once. The Plan 02 Ingestion worker
calls process() once per mic chunk and stamps is_target onto its events. Pure
(no asyncio) so it is unit-testable.
"""

import numpy as np

from .types import SpeakerFrame


class PvadWorker:
    def __init__(self, stream, proximity_rms: float, emit):
        self._stream = stream
        self._proximity_rms = proximity_rms
        self._emit = emit
        self._failed = False

    def update_speaker(self, embedding: np.ndarray) -> None:
        self._stream.update_speaker(embedding)

    def process(self, chunk: np.ndarray, ts: float) -> list:
        try:
            return self._stream.push(chunk, ts)
        except Exception as exc:   # noqa: BLE001 — degrade, never crash the loop
            if not self._failed:
                self._failed = True
                self._emit("worker_failed", {"worker": "pvad", "error": str(exc)})
            return [self._rms_fallback(chunk, ts)]

    def _rms_fallback(self, chunk: np.ndarray, ts: float) -> SpeakerFrame:
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        return SpeakerFrame(
            ts=ts, is_target=rms >= self._proximity_rms,
            confidence=0.0, rms=rms,
        )
