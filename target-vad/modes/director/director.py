"""Director — thin stateful shell over the pure reducer. Holds (state, ctx) and
is the single mutator entry point: dispatch(event) runs reduce(), stores the new
state, returns the commands for the worker layer (Plan 02) to execute. Still no
I/O — workers feed it events and run its commands."""

from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import Context, new_context
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


class Director:
    def __init__(self, cfg: DirectorConfig, conversation: ConversationManager,
                 now: float, proximity_rms: float, owner_bearing=None):
        self.ctx: Context = new_context(cfg, conversation, now, proximity_rms,
                                        owner_bearing=owner_bearing)
        self.state: State = State.LISTENING

    def dispatch(self, event) -> list:
        self.state, commands = reduce(self.state, self.ctx, event)
        return commands
