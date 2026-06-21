"""DirectorResult — what DirectorRuntime returns at true session end.

Mirrors modes/talkback/handoff.py:27 (TalkbackResult) so the WakeGate (Plan 03)
consumes the same shape it does today."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectorResult:
    reason: str
    turns: int
    total_duration_s: float
