from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce, silence_duration
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=now, proximity_rms=0.02)


def test_silence_accrues_only_in_listening():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    ctx.now = 60.0
    assert silence_duration(State.LISTENING, ctx) == 60.0
    for s in (State.THINKING, State.SPEAKING, State.EVALUATING):
        assert silence_duration(s, ctx) == 0.0


def test_no_timeout_while_speaking_even_after_long_reply():
    ctx = _ctx(now=0.0)
    state, cmds = reduce(State.SPEAKING, ctx, E.Tick(now=120.0))
    assert state is State.SPEAKING
    assert cmds == []          # a 120s reply never ends the session


def test_silence_timeout_fires_in_listening():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=30.0))
    assert state is State.IDLE
    assert cmds == [C.EndSession("silence_timeout")]


def test_hard_timeout_fires_in_any_state_and_beats_silence():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    state, cmds = reduce(State.SPEAKING, ctx, E.Tick(now=300.0))
    assert state is State.IDLE
    assert cmds == [C.EndSession("hard_timeout")]


def test_nudge_fires_once_at_lead_and_does_not_end_session():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    # nudge_lead_s=5, silence_timeout_s=30 => nudge at silence>=25
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=25.0))
    assert state is State.LISTENING
    assert cmds == [C.SpeakNudge()]
    assert ctx.nudged_cycle is True
    # next tick before timeout: no second nudge
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=27.0))
    assert cmds == []
    # then the real timeout still fires
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=30.0))
    assert state is State.IDLE and cmds == [C.EndSession("silence_timeout")]
