"""Director-11 DOA cone vote: a segment is only owner-speech if it comes from
the owner's direction. None anywhere in the chain = abstain (fail open) —
the other gates decide, exactly D10 behavior."""

import pytest

from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce, in_owner_cone, gate_diag_reason
from modes.talkback.conversation import ConversationManager


def _ctx(bearing=97.0, cone=20.0, alpha=0.3, now=5.0):
    cfg = DirectorConfig(reject_bystanders=True, endpoint_threshold=0.5,
                         doa_cone_deg=cone, doa_bearing_ema_alpha=alpha)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=0.04, owner_bearing=bearing)
    ctx.presence_status = PresenceStatus.PRESENT
    ctx.last_speech_at = 0.0
    return ctx


def _seg(doa=97.0, rms=1.0, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=1500.0, rms=rms, is_target=True,
                               endpoint_prob=endpoint, doa_angle=doa)


# ---- in_owner_cone ----

def test_cone_abstains_without_angle_or_bearing():
    assert in_owner_cone(_ctx(), None) is None
    assert in_owner_cone(_ctx(bearing=None), 97.0) is None


def test_cone_accepts_inside_including_wraparound():
    assert in_owner_cone(_ctx(bearing=97.0), 110.0) is True
    assert in_owner_cone(_ctx(bearing=5.0), 350.0) is True     # 15 deg the short way


def test_cone_rejects_outside():
    assert in_owner_cone(_ctx(bearing=97.0), 193.0) is False   # the spike's podcast


# ---- new turns ----

def test_out_of_cone_turn_rejected_no_serve_no_clock_reset():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, _seg(doa=193.0))
    assert cmds == []
    assert ctx.last_speech_at == 0.0
    assert gate_diag_reason(ctx, _seg(doa=193.0)) == "out_of_cone"


def test_abstain_turn_still_served():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, _seg(doa=None))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_in_cone_turn_served_and_bearing_tracks():
    ctx = _ctx(bearing=97.0, alpha=0.3)
    state, cmds = reduce(State.LISTENING, ctx, _seg(doa=107.0))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]
    assert ctx.owner_bearing == pytest.approx(100.0)           # 97 + 0.3*10


def test_rejected_and_abstained_turns_never_move_the_bearing():
    ctx = _ctx(bearing=97.0)
    reduce(State.LISTENING, ctx, _seg(doa=193.0))              # out of cone
    reduce(State.LISTENING, ctx, _seg(doa=None))               # abstain (served)
    reduce(State.LISTENING, ctx, _seg(doa=97.0, rms=0.001))    # too_quiet reject
    assert ctx.owner_bearing == 97.0


def test_bearing_ema_wraps_across_zero():
    ctx = _ctx(bearing=355.0, cone=20.0, alpha=0.5)
    reduce(State.LISTENING, ctx, _seg(doa=5.0))                # 10 deg inside cone
    assert ctx.owner_bearing == pytest.approx(0.0)


# ---- duck-at-onset ----

def test_out_of_cone_onset_does_not_duck():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.NearFieldOnset(rms=1.0, is_target=True, doa_angle=193.0))
    assert state is State.SPEAKING and cmds == []


def test_abstain_onset_still_ducks():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.NearFieldOnset(rms=1.0, is_target=True, doa_angle=None))
    assert state is State.EVALUATING and cmds == [C.Duck(ctx.cfg.duck_level)]


def test_in_cone_onset_ducks():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.NearFieldOnset(rms=1.0, is_target=True, doa_angle=99.0))
    assert state is State.EVALUATING and cmds == [C.Duck(ctx.cfg.duck_level)]


# ---- interjections ----

def _interjection(doa, score=0.9, dur=2200.0, rms=1.0):
    return E.InterjectionSegment(duration_ms=dur, rms=rms, is_target=True,
                                 speaker_score=score, doa_angle=doa)


def test_out_of_cone_interjection_restores_never_cuts():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _interjection(doa=193.0))
    assert cmds == [C.Restore()]


def test_in_cone_interjection_proceeds_to_transcription():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _interjection(doa=99.0))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeInterjection()]


def test_abstain_interjection_proceeds():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _interjection(doa=None))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeInterjection()]
