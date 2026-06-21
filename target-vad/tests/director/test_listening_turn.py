from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=now, proximity_rms=0.02)


def test_complete_target_segment_requests_transcription():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.SegmentEndpointed(900.0, 0.4, is_target=True, endpoint_prob=0.8))
    assert state is State.LISTENING
    assert cmds == [C.TranscribeUserTurn()]


def test_incomplete_turn_keeps_accumulating():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.SegmentEndpointed(300.0, 0.4, is_target=True, endpoint_prob=0.2))
    assert state is State.LISTENING and cmds == []


def test_bystander_segment_is_ignored():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.SegmentEndpointed(900.0, 0.4, is_target=False, endpoint_prob=0.9))
    assert state is State.LISTENING and cmds == []


def test_segment_updates_silence_clock_so_session_survives():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    ctx.now = 29.0
    reduce(State.LISTENING, ctx,
           E.SegmentEndpointed(900.0, 0.4, is_target=True, endpoint_prob=0.8))
    assert ctx.last_speech_at == 29.0   # user activity resets the grace window


def test_transcribed_turn_starts_generation_and_thinks():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
    assert state is State.THINKING
    assert len(cmds) == 1 and isinstance(cmds[0], C.StartGeneration)
    assert cmds[0].gen_id == 1            # bumped from 0
    assert ctx.gen_id == 1
    assert ctx.current_query == "tell me a story"
    assert {"role": "user", "content": "tell me a story"} in ctx.conversation.get_messages()
    assert cmds[0].messages == ctx.conversation.get_messages()


def test_empty_or_low_confidence_user_turn_is_dropped():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="", mean_word_prob=0.9))
    assert state is State.LISTENING and cmds == []
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="hello", mean_word_prob=0.2))
    assert state is State.LISTENING and cmds == []   # below conf_floor
