from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


def _ctx(lockout=True, proximity_rms=0.5, now=5.0):
    cfg = DirectorConfig(lockout_enabled=lockout)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=proximity_rms)
    ctx.last_speech_at = 0.0
    return ctx


def _verdict(score=0.9, ok=True, rms=1.0):
    return E.SpeakerWindowVerdict(score=score, smoother_ok=ok, window_rms=rms)


# ---- passing windows ----

def test_pass_counts_window_and_resets_streak():
    ctx = _ctx()
    ctx.miss_streak = 1
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=True))
    assert state is State.LISTENING and cmds == []
    assert ctx.windows_seen == 1 and ctx.miss_streak == 0


def test_verdict_never_touches_the_silence_clock():
    ctx = _ctx()
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    reduce(State.LISTENING, ctx, _verdict(ok=False))
    assert ctx.last_speech_at == 0.0


# ---- window-1 fail == bad enrollment ----

def test_window_one_fail_ends_enroll_verify_failed():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False))
    assert state is State.IDLE
    assert cmds == [C.EndSession("enroll_verify_failed")]


def test_window_one_pass_then_fail_is_not_enrollment():
    ctx = _ctx()
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False))
    assert cmds == []                                  # WARN, not enroll_verify_failed


# ---- WARN -> EJECT ladder ----

def test_first_midsession_fail_is_warn_only():
    ctx = _ctx()
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    assert state is State.LISTENING and cmds == []
    assert ctx.miss_streak == 1


def test_two_fails_below_proximity_ejects():
    ctx = _ctx(proximity_rms=0.5)
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    assert state is State.IDLE
    assert cmds == [C.EndSession("speaker_mismatch")]


def test_two_fails_but_loud_never_ejects():
    # rms >= proximity floor -> someone IS at the kiosk -> WARN only (spec s11)
    ctx = _ctx(proximity_rms=0.5)
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    reduce(State.LISTENING, ctx, _verdict(ok=False, rms=1.0))
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False, rms=1.0))
    assert cmds == [] and ctx.miss_streak == 2


def test_passing_window_resets_the_streak_midladder():
    ctx = _ctx(proximity_rms=0.5)
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    reduce(State.LISTENING, ctx, _verdict(ok=True))          # streak resets
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    assert cmds == [] and ctx.miss_streak == 1               # back to WARN


# ---- shadow mode (lockout_enabled=False): counters advance, nothing ends ----

def test_shadow_window_one_fail_does_not_end():
    ctx = _ctx(lockout=False)
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False))
    assert state is State.LISTENING and cmds == []
    assert ctx.windows_seen == 1 and ctx.miss_streak == 1


def test_shadow_eject_condition_does_not_end():
    ctx = _ctx(lockout=False, proximity_rms=0.5)
    reduce(State.LISTENING, ctx, _verdict(ok=True))
    reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    state, cmds = reduce(State.LISTENING, ctx, _verdict(ok=False, rms=0.1))
    assert cmds == [] and ctx.miss_streak == 2
