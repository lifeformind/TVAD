from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


def _ctx(reject, proximity_rms=0.5, presence=PresenceStatus.UNAVAILABLE, now=5.0):
    cfg = DirectorConfig(reject_bystanders=reject, endpoint_threshold=0.5)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=proximity_rms)
    ctx.presence_status = presence
    ctx.last_speech_at = 0.0          # distinct from now=5.0 so we can see resets
    return ctx


def _seg(rms=1.0, is_target=True, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=500.0, rms=rms,
                               is_target=is_target, endpoint_prob=endpoint)


# ---- flag OFF: no-regression (proximity/presence ignored; clock always resets) ----

def test_off_complete_target_accepts_even_if_quiet_and_absent():
    ctx = _ctx(reject=False, proximity_rms=0.9, presence=PresenceStatus.ABSENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=0.0001, endpoint=0.9))
    assert state is State.LISTENING and cmds == [C.TranscribeUserTurn()]
    assert ctx.last_speech_at == 5.0          # reset (legacy)


def test_off_nontarget_no_transcribe_but_resets():
    ctx = _ctx(reject=False)
    state, cmds = reduce(State.LISTENING, ctx, _seg(is_target=False))
    assert cmds == []
    assert ctx.last_speech_at == 5.0          # legacy resets even for non-target


def test_off_incomplete_accumulates_and_resets():
    ctx = _ctx(reject=False)
    state, cmds = reduce(State.LISTENING, ctx, _seg(endpoint=0.1))
    assert cmds == []
    assert ctx.last_speech_at == 5.0


# ---- flag ON: reject-by-default ----

def test_on_quiet_rejected_no_reset():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=0.1, endpoint=0.9))
    assert cmds == []
    assert ctx.last_speech_at == 0.0          # NOT reset


def test_on_owner_absent_rejected_no_reset():
    ctx = _ctx(reject=True, proximity_rms=0.0, presence=PresenceStatus.ABSENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert cmds == []
    assert ctx.last_speech_at == 0.0


def test_on_present_proximate_accepts_and_resets():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert cmds == [C.TranscribeUserTurn()]
    assert ctx.last_speech_at == 5.0


def test_on_unavailable_proximate_accepts_failsafe():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.UNAVAILABLE)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert cmds == [C.TranscribeUserTurn()]   # camera can't judge -> allow


def test_on_nontarget_rejected_no_reset():
    ctx = _ctx(reject=True, proximity_rms=0.0, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(is_target=False, rms=1.0))
    assert cmds == []
    assert ctx.last_speech_at == 0.0


def test_on_incomplete_owner_accumulates_and_resets():
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.1))
    assert cmds == []                          # not complete yet
    assert ctx.last_speech_at == 5.0           # but plausibly-owner -> reset


def test_on_rejected_chatter_does_not_block_owner_absent_end():
    # owner present, then leaves; bystander chatter (rejected) must NOT keep the
    # kiosk alive / block the Director-07 owner-absent end.
    ctx = _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT, now=0.0)
    # advance clock to t=11 and mark owner ABSENT at t=10
    reduce(State.LISTENING, ctx, E.OwnerPresenceEvent(PresenceStatus.ABSENT, now=10.0))
    reduce(State.LISTENING, ctx, E.Tick(now=11.0))         # ctx.now -> 11; 11-10<grace, no end
    # loud bystander chatter while owner ABSENT -> rejected, last_speech_at stays 0.0
    reduce(State.LISTENING, ctx, _seg(rms=1.0, endpoint=0.9))
    assert ctx.last_speech_at == 0.0
    # owner-absent end now fires (grace + guard satisfied) instead of being blocked
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=13.5))
    assert state is State.IDLE and cmds == [C.EndSession("owner_absent")]
