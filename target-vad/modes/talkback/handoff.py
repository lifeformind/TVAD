"""Hand-off contract between KioskPipeline and TalkbackController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.audio.mic_stream import MicrophoneStream
    from core.vad.silero_vad import SpeechSegment


@dataclass
class TalkbackHandoff:
    """Payload KioskPipeline passes to TalkbackController at session start."""
    mic: Any
    primary_embedding: np.ndarray
    first_segment: Any
    config: dict


@dataclass
class TalkbackResult:
    """What TalkbackController returns when the conversation ends."""
    reason: str
    turns: int
    total_duration_s: float
