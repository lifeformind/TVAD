"""Context — the Director's mutable session blackboard. Only reduce() mutates it."""

from dataclasses import dataclass, field
from typing import Optional

from modes.director.config import DirectorConfig
from modes.talkback.conversation import ConversationManager


@dataclass
class Context:
    cfg: DirectorConfig
    conversation: ConversationManager
    proximity_rms: float
    now: float                      # latest injected clock (from Tick events)
    started_at: float               # session start (for hard timeout)
    last_speech_at: float           # last time we began waiting for the user
    gen_id: int = 0                 # monotone; tags every generation
    nudged_cycle: bool = False      # already nudged this LISTENING cycle?
    ducked: bool = False            # is TTS currently ducked?
    current_query: str = ""         # the request being answered (for resume)
    partial_response: str = ""      # assistant text spoken so far this turn
    pending_steer: Optional[str] = None  # one-shot LLM steer (resume), Plan 06 fills it
    interrupted_stack: list = field(default_factory=list)  # bounded in Plan 06


def new_context(cfg: DirectorConfig, conversation: ConversationManager,
                now: float, proximity_rms: float) -> Context:
    return Context(
        cfg=cfg, conversation=conversation, proximity_rms=proximity_rms,
        now=now, started_at=now, last_speech_at=now,
    )
