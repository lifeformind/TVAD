from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director.events import PresenceStatus
from modes.director.reducer import gate_diag_reason
from modes.talkback.conversation import ConversationManager


def _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT):
    cfg = DirectorConfig(reject_bystanders=reject, endpoint_threshold=0.5)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=5.0, proximity_rms=proximity_rms)
    ctx.presence_status = presence
    return ctx


def _seg(rms=1.0, is_target=True, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=500.0, rms=rms,
                               is_target=is_target, endpoint_prob=endpoint)


def test_reason_for_too_quiet():
    ctx = _ctx(proximity_rms=0.5)
    assert gate_diag_reason(ctx, _seg(rms=0.1)) == "too_quiet"


def test_reason_for_owner_absent():
    ctx = _ctx(presence=PresenceStatus.ABSENT, proximity_rms=0.0)
    assert gate_diag_reason(ctx, _seg(rms=1.0)) == "owner_absent_frame"


def test_reason_none_when_accepted():
    ctx = _ctx(presence=PresenceStatus.PRESENT, proximity_rms=0.5)
    assert gate_diag_reason(ctx, _seg(rms=1.0, endpoint=0.9)) is None


def test_reason_none_while_accumulating():
    ctx = _ctx(presence=PresenceStatus.PRESENT, proximity_rms=0.5)
    assert gate_diag_reason(ctx, _seg(rms=1.0, endpoint=0.1)) is None
