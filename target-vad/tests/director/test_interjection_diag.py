"""interjection_reject_reason — the pure DIAG mirror of the interjection reject
ladder. The 2026-09-02 live test showed Restore storms with no reason logged;
this makes each RESTORE explain itself (and the reducer's ladder uses the same
function, so the diag can never drift from the decision)."""
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import interjection_reject_reason, reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(**cfg):
    return new_context(DirectorConfig(**cfg), ConversationManager(system_prompt="s"),
                       now=0.0, proximity_rms=0.05)


def _seg(**kw):
    defaults = dict(duration_ms=900.0, rms=0.5, is_target=True, speaker_score=0.5)
    defaults.update(kw)
    return E.InterjectionSegment(**defaults)


def test_reason_too_quiet():
    assert interjection_reject_reason(_ctx(), _seg(rms=0.01)) == "too_quiet"


def test_reason_too_short():
    assert interjection_reject_reason(_ctx(), _seg(duration_ms=300.0)) == "too_short"


def test_reason_speaker_mismatch():
    assert interjection_reject_reason(_ctx(), _seg(speaker_score=0.05)) == "speaker_mismatch"


def test_reason_none_when_accepted():
    assert interjection_reject_reason(_ctx(), _seg()) is None


def test_ladder_order_quiet_wins_over_short():
    # Mirrors the reducer's ladder order: proximity is checked first.
    assert interjection_reject_reason(_ctx(), _seg(rms=0.01, duration_ms=100.0)) == "too_quiet"


def test_diag_line_formats_reason_with_evidence():
    from modes.director.reducer import interjection_diag_line
    line = interjection_diag_line(_ctx(), _seg(rms=0.01, speaker_score=0.123))
    assert line == "interjection REJECT=too_quiet rms=0.0100 dur=900ms score=0.123"


def test_diag_line_none_when_accepted():
    from modes.director.reducer import interjection_diag_line
    assert interjection_diag_line(_ctx(), _seg()) is None


def test_reducer_restore_agrees_with_reason():
    # The same segment the diag calls rejected must produce Restore, and an
    # accepted one must transcribe — decision and diag share one ladder.
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(duration_ms=300.0))
    assert cmds == [C.Restore()]
    ctx2 = _ctx()
    state2, cmds2 = reduce(State.EVALUATING, ctx2, _seg())
    assert any(isinstance(c, C.TranscribeInterjection) for c in cmds2)
