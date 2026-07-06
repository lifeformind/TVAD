"""Commands — outputs the reducer emits for workers (Plan 02) to execute. Frozen."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Duck:
    level: float


@dataclass(frozen=True)
class Restore:
    pass


@dataclass(frozen=True)
class Cut:
    """Drain playback + cancel the main LLM generation (race-fixed teardown, Plan 02)."""
    gen_id: int


@dataclass(frozen=True)
class StartGeneration:
    gen_id: int
    messages: list
    steer: Optional[str]


@dataclass(frozen=True)
class TranscribeUserTurn:
    seq: int = 0                     # staged-audio sequence (events.SegmentEndpointed)


@dataclass(frozen=True)
class TranscribeInterjection:
    seq: int = 0


@dataclass(frozen=True)
class SpeakNudge:
    pass


@dataclass(frozen=True)
class EndSession:
    reason: str


@dataclass(frozen=True)
class AccumulateSpeakerAudio:
    """Feed the staged segment audio into the safety-net rolling buffer
    (Director-09). Emitted ONLY for served/plausibly-owner speech. Carries no
    audio — worker staging, same discipline as the Transcribe* commands; seq
    (echoed from the segment event) tells the worker WHICH staged audio is
    this command's, so a stale command can't consume a later segment's."""
    seq: int = 0
