from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.reducer import reduce, safety_diag_line
from modes.talkback.conversation import ConversationManager


def _ctx(lockout=True):
    cfg = DirectorConfig(lockout_enabled=lockout)
    return new_context(cfg, ConversationManager(system_prompt="x"),
                       now=5.0, proximity_rms=0.5)


def _run(ctx, ev):
    state, cmds = reduce(State.LISTENING, ctx, ev)
    return safety_diag_line(ctx, ev, cmds)


def test_passing_window_line():
    line = _run(_ctx(), E.SpeakerWindowVerdict(0.85, True, 0.4))
    assert line == ("safety-net window=1 score=0.850 smoother_ok=True "
                    "streak=0 rms=0.4000")


def test_warn_line():
    ctx = _ctx()
    _run(ctx, E.SpeakerWindowVerdict(0.9, True, 0.4))
    line = _run(ctx, E.SpeakerWindowVerdict(0.1, False, 0.4))
    assert "WARN" in line and "streak=1" in line and "shadow" not in line


def test_shadow_warn_line_is_marked():
    line = _run(_ctx(lockout=False), E.SpeakerWindowVerdict(0.1, False, 0.4))
    assert "WARN" in line and "shadow" in line


def test_eject_line_carries_reason():
    ctx = _ctx()
    line = _run(ctx, E.SpeakerWindowVerdict(0.1, False, 0.4))   # window-1 fail
    assert "EJECT" in line and "enroll_verify_failed" in line


def test_shadow_window_one_fail_shows_would_end():
    line = _run(_ctx(lockout=False), E.SpeakerWindowVerdict(0.1, False, 0.4))
    assert "WARN (shadow) would_end=enroll_verify_failed" in line


def test_shadow_eject_condition_shows_would_end():
    ctx = _ctx(lockout=False)
    _run(ctx, E.SpeakerWindowVerdict(0.9, True, 0.4))     # window 1 passes
    _run(ctx, E.SpeakerWindowVerdict(0.1, False, 0.1))    # streak 1
    line = _run(ctx, E.SpeakerWindowVerdict(0.1, False, 0.1))  # streak 2, quiet
    assert "WARN (shadow) would_end=speaker_mismatch" in line
