"""Events — inputs the reducer consumes. Workers (Plan 02) emit these. Frozen."""

import enum
from dataclasses import dataclass


class PresenceStatus(enum.Enum):
    PRESENT = "PRESENT"          # the enrolled owner's face is in frame
    ABSENT = "ABSENT"            # no owner: empty frame OR a stranger (identity fail)
    UNAVAILABLE = "UNAVAILABLE"  # camera can't judge (glitch / not yet enrolled)


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
    # Staging sequence: ingestion stamps the same seq on this event and on the
    # audio it stages into the STT/safety workers; the reducer echoes it into
    # the consuming commands so a command can never grab a LATER segment's
    # staged audio (overwrite-last race). 0 = the assembly-seeded first segment.
    seq: int = 0
    doa_angles: "tuple | None" = None  # speech-flagged DOA samples over the segment span (Director-11); None/empty = no signal


@dataclass(frozen=True)
class UserTurnTranscribed:
    text: str
    mean_word_prob: float


@dataclass(frozen=True)
class NearFieldOnset:
    """Voiced onset during SPEAKING — the duck-at-onset reflex trigger."""
    rms: float
    is_target: bool
    doa_angle: "float | None" = None   # latest speech-flagged sample at onset (Director-11)


@dataclass(frozen=True)
class InterjectionSegment:
    """The ducked capture in EVALUATING endpointed; gate-ladder inputs attached."""
    duration_ms: float
    rms: float
    is_target: bool
    speaker_score: float             # ECAPA/pVAD score vs primary (off hot path)
    seq: int = 0                     # staging sequence (see SegmentEndpointed)
    doa_angles: "tuple | None" = None  # speech-flagged DOA samples over the segment span (Director-11)


@dataclass(frozen=True)
class InterjectionTranscribed:
    text: str
    mean_word_prob: float


@dataclass(frozen=True)
class FirstTtsFrame:
    gen_id: int                      # first audible frame written for this generation


@dataclass(frozen=True)
class AssistantPartial:
    """Spoken-so-far text for the in-flight reply (emitted per sentence). Lets the
    reducer track partial_response so a barge-in records what was actually said
    (keeps history strictly alternating; feeds Plan 06 resume)."""
    gen_id: int
    text: str


@dataclass(frozen=True)
class ReplyComplete:
    gen_id: int
    assistant_text: str


@dataclass(frozen=True)
class OwnerPresenceEvent:
    """Camera floor-control signal (Director-07). Emitted by VisionWorker on a
    DEBOUNCED status change only. Pure state-recording in the reducer — never a
    transition; the owner-absent decision happens on a Tick."""
    status: PresenceStatus
    now: float


@dataclass(frozen=True)
class SpeakerWindowVerdict:
    """Accumulated-window ECAPA verdict from the SafetyNetWorker (Director-09).
    Pure data; the reducer owns the WARN/EJECT decision. No clock field — the
    ladder is not time-based (the post-eject quiet clock lives in the WakeGate)."""
    score: float                     # cosine(window embedding, primary)
    smoother_ok: bool                # M-of-N smoother output for this window
    window_rms: float                # RMS over the window audio (eject rms check)
    norm_score: "float | None" = None  # AS-Norm score (Task 7); None = normalizer off
