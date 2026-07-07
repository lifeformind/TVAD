"""Loud-bystander suspension (Director-10 live fix, 2026-07-07): a podcast's
segment rms (0.049-0.107) overlapped the owner's seed (0.085), so no proximity
factor separates them — but the ECAPA safety net did (owner windows 0.22-0.40
vs podcast 0.01-0.14, streak hit 4) and could only WARN because the eject rule
requires sub-floor rms. Fix: while miss_streak >= 2, keep feeding the verifier
(one passing window — the owner speaking — unlocks) but serve nothing, and let
the silence clock run so a podcast cannot hold the session open."""

from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce, gate_diag_reason
from modes.talkback.conversation import ConversationManager


def _ctx(lockout=True, reject=True, streak=2, now=5.0):
    cfg = DirectorConfig(lockout_enabled=lockout, reject_bystanders=reject,
                         endpoint_threshold=0.5)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=0.04)
    ctx.presence_status = PresenceStatus.PRESENT
    ctx.last_speech_at = 0.0          # distinct from now=5.0 so we can see resets
    ctx.windows_seen = 5              # mid-session: window-1 enroll logic is past
    ctx.miss_streak = streak
    return ctx


def _seg(rms=1.0, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=1500.0, rms=rms,
                               is_target=True, endpoint_prob=endpoint)


def test_streak_two_accumulates_but_never_serves():
    ctx = _ctx(streak=2)
    state, cmds = reduce(State.LISTENING, ctx, _seg())
    assert state is State.LISTENING
    assert cmds == [C.AccumulateSpeakerAudio()]        # verifier keeps watching
    assert ctx.last_speech_at == 0.0                   # podcast can't hold session open


def test_streak_one_still_serves():
    ctx = _ctx(streak=1)
    state, cmds = reduce(State.LISTENING, ctx, _seg())
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_shadow_mode_never_suspends():
    ctx = _ctx(lockout=False, streak=4)
    state, cmds = reduce(State.LISTENING, ctx, _seg())
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_passing_window_unlocks_serving():
    ctx = _ctx(streak=2)
    reduce(State.LISTENING, ctx, E.SpeakerWindowVerdict(
        score=0.9, smoother_ok=True, window_rms=1.0))   # owner spoke: streak -> 0
    state, cmds = reduce(State.LISTENING, ctx, _seg())
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_legacy_reject_off_also_suspends():
    # The suspension is D09 lockout-domain, orthogonal to the D08 bystander gate.
    ctx = _ctx(reject=False, streak=2)
    state, cmds = reduce(State.LISTENING, ctx, _seg())
    assert cmds == [C.AccumulateSpeakerAudio()]


def test_gate_diag_reason_reports_speaker_unverified():
    ctx = _ctx(streak=2)
    assert gate_diag_reason(ctx, _seg()) == "speaker_unverified"
