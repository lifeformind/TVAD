"""pVAD output type emitted onto the event bus."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerFrame:
    """One aggregated ~50ms target-speaker decision.

    is_target: enrolled speaker speaking now? confidence: mean pVAD prob over
    the aggregated frames. rms: chunk RMS (used by the proximity/near-field
    gate and the crash-fallback).
    """
    ts: float
    is_target: bool
    confidence: float
    rms: float
