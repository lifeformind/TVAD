from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=now, proximity_rms=0.02)


def test_first_tts_frame_moves_thinking_to_speaking():
    ctx = _ctx(); ctx.gen_id = 1
    state, cmds = reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=1))
    assert state is State.SPEAKING and cmds == []


def test_stale_first_tts_frame_is_dropped():
    ctx = _ctx(); ctx.gen_id = 2
    state, cmds = reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=1))  # old turn
    assert state is State.THINKING and cmds == []


def test_reply_complete_yields_floor_and_records_assistant():
    ctx = _ctx(now=10.0); ctx.gen_id = 1; ctx.last_speech_at = 0.0; ctx.nudged_cycle = True
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.ReplyComplete(gen_id=1, assistant_text="once upon a time"))
    assert state is State.LISTENING and cmds == []
    msgs = ctx.conversation.get_messages()
    assert {"role": "assistant", "content": "once upon a time"} in msgs
    assert ctx.last_speech_at == 10.0     # floor reset on yield
    assert ctx.nudged_cycle is False      # nudge re-armed


def test_stale_reply_complete_is_dropped():
    ctx = _ctx(); ctx.gen_id = 2
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.ReplyComplete(gen_id=1, assistant_text="stale"))
    assert state is State.SPEAKING and cmds == []
    assert ctx.conversation.get_messages() == [{"role": "system", "content": "s"}]
