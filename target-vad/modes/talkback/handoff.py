"""Hand-off contract between the WakeGate and the Director.

Renamed from TalkbackHandoff/TalkbackResult (binding interface, Plan 02).
Verify-before-serve is the WakeGate's split-half check (Director-09); no
holdout travels in the handoff.

The legacy TalkbackHandoff/TalkbackResult names are retained as aliases so the
not-yet-deleted modes/talkback/controller.py path (and its tests) keep importing.
The alias handoff omits holdout_embedding via a default so old callers that build
a TalkbackHandoff without it still construct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class DirectorHandoff:
    """Payload the WakeGate passes to the Director at session start."""
    mic: Any
    primary_embedding: np.ndarray
    first_segment: Any
    config: dict
    vad: Any
    embedder: Any


@dataclass(frozen=True)
class DirectorResult:
    """What the Director returns when the conversation ends.

    Frozen: a session result is an immutable record. This is the SINGLE
    DirectorResult type — modes/director/result.py re-exports it so the runtime
    and the WakeGate share one class (no isinstance mismatch across the seam)."""
    reason: str
    turns: int
    total_duration_s: float
    # Director-09: the ended session's calibrated proximity floor, so the
    # WakeGate's post-eject quiet-hold has a threshold. 0.0 == unknown -> no hold.
    proximity_rms: float = 0.0


@dataclass
class TalkbackHandoff:
    """Legacy alias for the pre-rename handoff (modes/talkback/controller.py).

    holdout_embedding defaults to None so old callers that never supplied it
    still construct; the Director path always uses DirectorHandoff instead."""
    mic: Any
    primary_embedding: np.ndarray
    first_segment: Any
    config: dict
    vad: Any
    embedder: Any
    holdout_embedding: Optional[np.ndarray] = None


# DirectorResult and the legacy TalkbackResult are structurally identical.
TalkbackResult = DirectorResult
