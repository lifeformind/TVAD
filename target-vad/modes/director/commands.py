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
    pass


@dataclass(frozen=True)
class TranscribeInterjection:
    pass


@dataclass(frozen=True)
class SpeakNudge:
    pass


@dataclass(frozen=True)
class EndSession:
    reason: str
