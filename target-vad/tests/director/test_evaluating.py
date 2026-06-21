from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx():
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                      now=0.0, proximity_rms=0.02)
    ctx.gen_id = 1
    ctx.ducked = True
    ctx.current_query = "tell me a story"
    ctx.partial_response = "once upon a time"
    return ctx


def _seg(duration_ms=900.0, rms=0.5, is_target=True, score=0.9):
    return E.InterjectionSegment(duration_ms, rms, is_target, score)


def test_too_short_interjection_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(duration_ms=300.0))  # < 700ms
    assert state is State.SPEAKING and cmds == [C.Restore()] and ctx.ducked is False


def test_far_interjection_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(rms=0.001))
    assert state is State.SPEAKING and cmds == [C.Restore()]


def test_speaker_mismatch_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(score=0.05))  # < 0.20
    assert state is State.SPEAKING and cmds == [C.Restore()]


def test_passing_segment_requests_transcription():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg())
    assert state is State.EVALUATING and cmds == [C.TranscribeInterjection()]


def test_backchannel_restores_and_keeps_talking():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="yeah got it", mean_word_prob=0.9))
    assert state is State.SPEAKING and cmds == [C.Restore()] and ctx.ducked is False
    # backchannel must NOT pollute history
    assert all(m["content"] != "yeah got it" for m in ctx.conversation.get_messages())


def test_empty_interjection_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="", mean_word_prob=0.9))
    assert state is State.SPEAKING and cmds == [C.Restore()]


def test_assistant_partial_updates_partial_response():
    ctx = _ctx()
    ctx.gen_id = 7
    ctx.partial_response = ""
    state, cmds = reduce(State.SPEAKING, ctx, E.AssistantPartial(gen_id=7, text="once upon a"))
    assert state is State.SPEAKING and cmds == []
    assert ctx.partial_response == "once upon a"
    # a stale gen_id must NOT clobber the current partial
    reduce(State.SPEAKING, ctx, E.AssistantPartial(gen_id=6, text="STALE"))
    assert ctx.partial_response == "once upon a"


def test_cut_with_empty_partial_still_records_an_assistant_turn():
    """The Req-3 live bug: if the barge-in lands before any sentence was spoken,
    partial_response is empty — but we must STILL add an assistant turn, else the
    history has two user turns in a row and the LLM returns empty for every later
    reply (verified against gemma)."""
    ctx = _ctx()
    ctx.partial_response = ""                         # nothing spoken yet
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert state is State.THINKING
    msgs = ctx.conversation.get_messages()
    # strictly alternating: the cut placeholder sits between the two user turns
    assert {"role": "assistant", "content": "[interrupted]"} in msgs
    roles = [m["role"] for m in msgs if m["role"] in ("user", "assistant")]
    assert all(a != b for a, b in zip(roles, roles[1:])), f"non-alternating: {roles}"


def test_question_cuts_and_starts_new_turn():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="wait why is that", mean_word_prob=0.9))
    assert state is State.THINKING
    assert isinstance(cmds[0], C.Cut) and cmds[0].gen_id == 1      # cut the OLD generation
    assert isinstance(cmds[1], C.Restore)                          # un-duck for the new reply
    assert isinstance(cmds[2], C.StartGeneration) and cmds[2].gen_id == 2
    assert ctx.ducked is False
    msgs = ctx.conversation.get_messages()
    # interrupted partial preserved + the new user turn added
    assert {"role": "assistant", "content": "once upon a time [interrupted]"} in msgs
    assert {"role": "user", "content": "wait why is that"} in msgs
    # resume frame pushed
    assert ctx.interrupted_stack[-1] == {"query": "tell me a story",
                                         "partial": "once upon a time"}
