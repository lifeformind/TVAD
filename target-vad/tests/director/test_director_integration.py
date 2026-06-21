from modes.director.director import Director
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _director():
    return Director(DirectorConfig(), ConversationManager(system_prompt="s"),
                    now=0.0, proximity_rms=0.02)


def test_dispatch_applies_state_and_returns_commands():
    d = _director()
    assert d.state is State.LISTENING
    cmds = d.dispatch(E.UserTurnTranscribed(text="hi", mean_word_prob=0.9))
    assert d.state is State.THINKING
    assert isinstance(cmds[0], C.StartGeneration)


def test_full_conversation_with_backchannel_question_and_resume_push():
    d = _director()
    # 1. user asks for a story
    d.dispatch(E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
    assert d.state is State.THINKING
    gen = d.ctx.gen_id
    d.dispatch(E.FirstTtsFrame(gen_id=gen)); assert d.state is State.SPEAKING
    d.ctx.partial_response = "once upon a time"
    # 2. backchannel mid-story -> keep talking
    d.dispatch(E.NearFieldOnset(rms=0.5, is_target=True)); assert d.state is State.EVALUATING
    d.dispatch(E.InterjectionSegment(900.0, 0.5, True, 0.9))   # passes gates
    cmds = d.dispatch(E.InterjectionTranscribed(text="mhm", mean_word_prob=0.9))
    assert d.state is State.SPEAKING and cmds == [C.Restore()]
    # 3. a real question mid-story -> cut + answer
    d.dispatch(E.NearFieldOnset(rms=0.5, is_target=True)); assert d.state is State.EVALUATING
    d.dispatch(E.InterjectionSegment(900.0, 0.5, True, 0.9))
    cmds = d.dispatch(E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert d.state is State.THINKING
    assert isinstance(cmds[0], C.Cut) and isinstance(cmds[1], C.StartGeneration)
    assert d.ctx.interrupted_stack[-1]["query"] == "tell me a story"
    # 4. answer the question, THEN prove silence is suspended while speaking.
    d.dispatch(E.FirstTtsFrame(gen_id=d.ctx.gen_id))     # -> SPEAKING
    assert d.state is State.SPEAKING
    # 100s elapse WHILE speaking: silence must NOT accrue, session must NOT end.
    assert d.dispatch(E.Tick(now=100.0)) == []
    assert d.state is State.SPEAKING
    # reply ends -> yield floor; the grace window restarts from now (=100).
    d.dispatch(E.ReplyComplete(gen_id=d.ctx.gen_id, assistant_text="because..."))
    assert d.state is State.LISTENING
    assert d.ctx.last_speech_at == 100.0
    # only real LISTENING silence ends the session: 24s is fine, 30s ends it.
    assert d.dispatch(E.Tick(now=124.0)) == []           # silence 24s < 25s nudge lead
    cmds = d.dispatch(E.Tick(now=130.0))                 # silence 30s -> end
    assert d.state is State.IDLE and cmds == [C.EndSession("silence_timeout")]
