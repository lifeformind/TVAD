"""Director-11 DOA cone vote: a segment is only owner-speech if it comes from
the owner's direction. None/empty anywhere in the chain = abstain (fail open)
— the other gates decide, exactly D10 behavior.

Segments vote by in-cone FRACTION, not median (live 2026-07-07 18:42):
continuous background speech makes the VAD merge the owner's utterance into a
bystander-dominated segment, and a duration-majority median votes the
bystander — the owner got REJECT=out_of_cone while the array's LED tracked
them. The vote asks "did the owner speak during this segment", i.e. enough
speech-flagged samples point at them. The instantaneous duck-at-onset reflex
keeps the scalar in_owner_cone."""

import pytest

from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import (reduce, in_owner_cone, cone_vote,
                                    cone_diag, gate_diag_reason)
from modes.talkback.conversation import ConversationManager

PODCAST = 193.0                       # the spike's off-axis podcast bearing


def _ctx(bearing=97.0, cone=20.0, alpha=0.3, now=5.0):
    cfg = DirectorConfig(reject_bystanders=True, endpoint_threshold=0.5,
                         doa_cone_deg=cone, doa_bearing_ema_alpha=alpha)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=now, proximity_rms=0.04, owner_bearing=bearing)
    ctx.presence_status = PresenceStatus.PRESENT
    ctx.last_speech_at = 0.0
    return ctx


def _seg(doa=(97.0, 95.0, 100.0), rms=1.0, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=1500.0, rms=rms, is_target=True,
                               endpoint_prob=endpoint, doa_angles=doa)


# ---- in_owner_cone (scalar, onset path) ----

def test_scalar_cone_abstains_without_angle_or_bearing():
    assert in_owner_cone(_ctx(), None) is None
    assert in_owner_cone(_ctx(bearing=None), 97.0) is None


def test_scalar_cone_accepts_inside_including_wraparound():
    assert in_owner_cone(_ctx(bearing=97.0), 110.0) is True
    assert in_owner_cone(_ctx(bearing=5.0), 350.0) is True     # 15 deg the short way


def test_scalar_cone_rejects_outside():
    assert in_owner_cone(_ctx(bearing=97.0), PODCAST) is False


# ---- cone_vote (fraction, segment path) ----

def test_vote_abstains_on_none_empty_or_no_bearing():
    assert cone_vote(_ctx(), None) is None
    assert cone_vote(_ctx(), ()) is None
    assert cone_vote(_ctx(bearing=None), (97.0,) * 5) is None


def test_vote_passes_pure_owner_segment():
    assert cone_vote(_ctx(), (97.0, 95.0, 100.0, 99.0)) is True


def test_vote_rejects_pure_bystander_segment():
    assert cone_vote(_ctx(), (PODCAST,) * 8) is False


def test_vote_passes_owner_minority_in_merged_segment():
    # The live failure: owner's 3 samples (~450ms) inside a podcast-dominated
    # segment. Median would say 193; the fraction vote (3/12 = 25%) passes.
    mixed = (PODCAST,) * 9 + (97.0, 95.0, 100.0)
    assert cone_vote(_ctx(), mixed) is True


def test_vote_needs_min_samples_even_at_fraction():
    # 2 of 8 = 25% fraction but < 3 samples: a lone DOA blip isn't the owner.
    assert cone_vote(_ctx(), (PODCAST,) * 6 + (97.0, 95.0)) is False


def test_vote_needs_min_fraction_even_with_samples():
    # 3 of 20 = 15% < 25%: too small a share of a long segment.
    assert cone_vote(_ctx(), (PODCAST,) * 17 + (97.0, 95.0, 100.0)) is False


# ---- new turns ----

def test_pure_bystander_turn_rejected_no_serve_no_clock_reset():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, _seg(doa=(PODCAST,) * 8))
    assert cmds == []
    assert ctx.last_speech_at == 0.0
    assert gate_diag_reason(ctx, _seg(doa=(PODCAST,) * 8)) == "out_of_cone"


def test_merged_segment_with_owner_minority_is_served():
    ctx = _ctx()
    mixed = (PODCAST,) * 9 + (97.0, 95.0, 100.0)
    state, cmds = reduce(State.LISTENING, ctx, _seg(doa=mixed))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_abstain_turn_still_served():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, _seg(doa=None))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_served_turn_tracks_bearing_toward_in_cone_median_only():
    # Mixed served segment: EMA moves toward the IN-CONE samples' median
    # (107), never toward the bystander's share of the segment.
    ctx = _ctx(bearing=97.0, alpha=0.3)
    mixed = (PODCAST,) * 9 + (107.0, 107.0, 109.0)
    reduce(State.LISTENING, ctx, _seg(doa=mixed))
    assert ctx.owner_bearing == pytest.approx(100.0)           # 97 + 0.3*(107-97)


def test_rejected_and_abstained_turns_never_move_the_bearing():
    ctx = _ctx(bearing=97.0)
    reduce(State.LISTENING, ctx, _seg(doa=(PODCAST,) * 8))     # out of cone
    reduce(State.LISTENING, ctx, _seg(doa=None))               # abstain (served)
    reduce(State.LISTENING, ctx, _seg(doa=(97.0,) * 4, rms=0.001))  # too_quiet
    assert ctx.owner_bearing == 97.0


def test_bearing_ema_wraps_across_zero():
    ctx = _ctx(bearing=355.0, cone=20.0, alpha=0.5)
    reduce(State.LISTENING, ctx, _seg(doa=(5.0, 5.0, 7.0)))    # in-cone median 5
    assert ctx.owner_bearing == pytest.approx(0.0)


# ---- duck-at-onset (scalar) ----

def test_out_of_cone_onset_does_not_duck():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.NearFieldOnset(rms=1.0, is_target=True, doa_angle=PODCAST))
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
                                 speaker_score=score, doa_angles=doa)


def test_pure_bystander_interjection_restores_never_cuts():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _interjection(doa=(PODCAST,) * 8))
    assert cmds == [C.Restore()]


def test_owner_minority_interjection_proceeds_to_transcription():
    ctx = _ctx()
    mixed = (PODCAST,) * 9 + (97.0, 95.0, 100.0)
    state, cmds = reduce(State.EVALUATING, ctx, _interjection(doa=mixed))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeInterjection()]


def test_abstain_interjection_proceeds():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _interjection(doa=None))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeInterjection()]


# ---- cone_diag (REJECT-line visibility) ----

def test_cone_diag_shows_the_votes_inputs():
    line = cone_diag(_ctx(bearing=162.0), (PODCAST,) * 9 + (160.0, 158.0, 164.0))
    assert "n=12" in line and "in_cone=3" in line and "bearing=162" in line


def test_cone_diag_degrades_without_signal():
    assert cone_diag(_ctx(bearing=None), (97.0,)) == "doa=[no bearing]"
    assert "n=0" in cone_diag(_ctx(bearing=97.0), ())
