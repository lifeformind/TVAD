"""Firmware-silence echo guard (2026-09-02 live): the XVF-3000 subtracts its own
playback from SPEECHDETECTED, so a segment whose DOA sample tuple is PRESENT BUT
EMPTY during/just after TTS was heard only by the software VAD — it's the
kiosk's own echo, not a person (LED confirmed: reacts to humans, never to the
aux speaker). Tracker absent (None) always abstains — exact legacy behavior."""
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import (TurnVerdict, classify_new_turn,
                                    gate_diag_reason, interjection_reject_reason,
                                    reduce)
from modes.director import events as E
from modes.talkback.conversation import ConversationManager


def _ctx(**cfg):
    cfg.setdefault("reject_bystanders", True)
    return new_context(DirectorConfig(**cfg), ConversationManager(system_prompt="s"),
                       now=100.0, proximity_rms=0.05)


def _interj(**kw):
    d = dict(duration_ms=900.0, rms=0.5, is_target=True, speaker_score=0.5)
    d.update(kw)
    return E.InterjectionSegment(**d)


def _turn(**kw):
    d = dict(duration_ms=900.0, rms=0.5, is_target=True, endpoint_prob=0.9)
    d.update(kw)
    return E.SegmentEndpointed(**d)


# ---- interjections (always overlap TTS) ----

def test_interjection_empty_doa_is_firmware_silent():
    ctx = _ctx(echo_guard_tail_s=2.0)
    assert interjection_reject_reason(ctx, _interj(doa_angles=())) == "firmware_silent"


def test_interjection_guard_off_when_tail_zero():
    ctx = _ctx()   # echo_guard_tail_s default 0.0 = no-regression
    assert interjection_reject_reason(ctx, _interj(doa_angles=())) is None


def test_interjection_no_tracker_abstains():
    ctx = _ctx(echo_guard_tail_s=2.0)
    assert interjection_reject_reason(ctx, _interj(doa_angles=None)) is None


def test_interjection_real_speech_passes_guard():
    ctx = _ctx(echo_guard_tail_s=2.0)
    assert interjection_reject_reason(ctx, _interj(doa_angles=(101.0, 103.0))) is None


# ---- new turns (guard only inside the post-reply tail) ----

def test_new_turn_empty_doa_in_tail_is_echo():
    ctx = _ctx(echo_guard_tail_s=2.0)
    ctx.last_reply_done_at = 99.0        # 1s ago, inside the 2s tail
    assert classify_new_turn(ctx, _turn(doa_angles=())) is TurnVerdict.REJECT_ECHO
    assert gate_diag_reason(ctx, _turn(doa_angles=())) == "echo_firmware_silent"


def test_new_turn_outside_tail_is_served():
    ctx = _ctx(echo_guard_tail_s=2.0)
    ctx.last_reply_done_at = 90.0        # 10s ago
    assert classify_new_turn(ctx, _turn(doa_angles=())) is TurnVerdict.ACCEPT


def test_new_turn_with_speech_samples_in_tail_is_served():
    ctx = _ctx(echo_guard_tail_s=2.0)
    ctx.last_reply_done_at = 99.0
    assert classify_new_turn(ctx, _turn(doa_angles=(100.0,))) is TurnVerdict.ACCEPT


def test_new_turn_no_tracker_in_tail_is_served():
    ctx = _ctx(echo_guard_tail_s=2.0)
    ctx.last_reply_done_at = 99.0
    assert classify_new_turn(ctx, _turn(doa_angles=None)) is TurnVerdict.ACCEPT


def test_new_turn_guard_off_when_tail_zero():
    ctx = _ctx()
    ctx.last_reply_done_at = 100.0       # right now — but guard disabled
    assert classify_new_turn(ctx, _turn(doa_angles=())) is TurnVerdict.ACCEPT


def test_reply_complete_stamps_last_reply_done_at():
    ctx = _ctx(echo_guard_tail_s=2.0)
    ctx.gen_id = 1
    state, _ = reduce(State.SPEAKING, ctx,
                      E.ReplyComplete(gen_id=1, assistant_text="done."))
    assert state is State.LISTENING
    assert ctx.last_reply_done_at == ctx.now
