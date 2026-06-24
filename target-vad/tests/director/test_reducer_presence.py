from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    cfg = DirectorConfig(owner_absent_grace_s=3.0, active_talk_guard_s=3.0,
                         silence_timeout_s=30.0, hard_timeout_s=300.0)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=0.0)
    return ctx


def test_presence_event_records_without_transition():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.OwnerPresenceEvent(PresenceStatus.PRESENT, now=2.0))
    assert state is State.SPEAKING and cmds == []
    assert ctx.presence_status is PresenceStatus.PRESENT
    assert ctx.presence_since == 2.0


def test_absent_sustained_past_grace_ends_session():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0                     # no recent speech (guard satisfied at t>=3)
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    # tick at 10 + grace; last_speech_at far in the past
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=13.0))
    assert state is State.IDLE
    assert cmds == [C.EndSession("owner_absent")]


def test_absent_within_grace_does_not_end():
    ctx = _ctx()
    ctx.last_speech_at = 0.0
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=12.0))   # only 2s < 3s grace
    assert state is State.LISTENING and cmds == []


def test_active_talk_guard_suppresses_owner_absent():
    ctx = _ctx()
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    ctx.last_speech_at = 12.5                    # owner spoke recently
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=14.0))  # absent 4s>grace, but spoke 1.5s ago<guard
    assert state is State.LISTENING and cmds == []


def test_unavailable_falls_back_to_silence_timeout():
    ctx = _ctx()
    ctx.last_speech_at = 0.0
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.UNAVAILABLE, now=1.0))
    # well under silence_timeout, camera unavailable => no owner-absent end
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=5.0))
    assert state is State.LISTENING and cmds == []
    # but the unchanged silence timeout still fires at 30s
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=31.0))
    assert state is State.IDLE and cmds == [C.EndSession("silence_timeout")]


def test_present_does_not_extend_silence_timeout():
    ctx = _ctx()
    ctx.last_speech_at = 0.0
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.PRESENT, now=1.0))
    # decision 1 (add-on): a present-but-silent owner STILL times out at 30s
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=31.0))
    assert state is State.IDLE and cmds == [C.EndSession("silence_timeout")]
