"""Hand-off contract between the WakeGate and the Director.

Renamed from TalkbackHandoff/TalkbackResult (binding interface, Plan 02).
holdout_embedding (Plan 05) carries a pre-finalize enrollment utterance
embedding for verify-before-serve; Plan 03 passes the first-segment embedding
through as a placeholder until Plan 05 captures the real holdout.

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
    holdout_embedding: np.ndarray   # Plan 05: pre-finalize utterance embedding
    first_segment: Any
    config: dict
    vad: Any
    embedder: Any


@dataclass
class DirectorResult:
    """What the Director returns when the conversation ends."""
    reason: str
    turns: int
    total_duration_s: float


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
