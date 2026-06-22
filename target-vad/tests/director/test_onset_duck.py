from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx():
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=0.0, proximity_rms=0.02)


def test_near_field_target_onset_ducks_and_evaluates():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.NearFieldOnset(rms=0.5, is_target=True))
    assert state is State.EVALUATING
    assert cmds == [C.Duck(0.35)]
    assert ctx.ducked is True


def test_far_onset_does_not_duck():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.NearFieldOnset(rms=0.001, is_target=True))
    assert state is State.SPEAKING and cmds == [] and ctx.ducked is False


def test_non_target_onset_does_not_duck():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.NearFieldOnset(rms=0.5, is_target=False))
    assert state is State.SPEAKING and cmds == [] and ctx.ducked is False


def test_onset_ignored_when_not_speaking():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, E.NearFieldOnset(rms=0.5, is_target=True))
    assert state is State.LISTENING and cmds == []
