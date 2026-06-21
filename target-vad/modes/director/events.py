"""Events — inputs the reducer consumes. Workers (Plan 02) emit these. Frozen."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tick:
    now: float                       # injected clock; the reducer's only time source


@dataclass(frozen=True)
class SegmentEndpointed:
    """A completed VAD segment in LISTENING, with reflex signals attached."""
    duration_ms: float
    rms: float
    is_target: bool                  # pVAD/ECAPA verdict (Plan 05 fills it; True for now)
    endpoint_prob: float             # Smart Turn endpoint probability


@dataclass(frozen=True)
class UserTurnTranscribed:
    text: str
    mean_word_prob: float


@dataclass(frozen=True)
class NearFieldOnset:
    """Voiced onset during SPEAKING — the duck-at-onset reflex trigger."""
    rms: float
    is_target: bool


@dataclass(frozen=True)
class InterjectionSegment:
    """The ducked capture in EVALUATING endpointed; gate-ladder inputs attached."""
    duration_ms: float
    rms: float
    is_target: bool
    speaker_score: float             # ECAPA/pVAD score vs primary (off hot path)


@dataclass(frozen=True)
class InterjectionTranscribed:
    text: str
    mean_word_prob: float


@dataclass(frozen=True)
class FirstTtsFrame:
    gen_id: int                      # first audible frame written for this generation


@dataclass(frozen=True)
class ReplyComplete:
    gen_id: int
    assistant_text: str
