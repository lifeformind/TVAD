from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


def _ctx(reject=True, proximity_rms=0.5, presence=PresenceStatus.PRESENT):
    cfg = DirectorConfig(reject_bystanders=reject, endpoint_threshold=0.5,
                         verify_window_ms=100.0, speaker_threshold=0.2)
    ctx = new_context(cfg, ConversationManager(system_prompt="x"),
                      now=5.0, proximity_rms=proximity_rms)
    ctx.presence_status = presence
    ctx.last_speech_at = 0.0
    return ctx


def _seg(rms=1.0, is_target=True, endpoint=0.9, seq=0):
    return E.SegmentEndpointed(duration_ms=500.0, rms=rms,
                               is_target=is_target, endpoint_prob=endpoint,
                               seq=seq)


def test_accept_emits_accumulate_then_transcribe():
    state, cmds = reduce(State.LISTENING, _ctx(), _seg(endpoint=0.9))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_accumulate_verdict_emits_accumulate_only():
    state, cmds = reduce(State.LISTENING, _ctx(), _seg(endpoint=0.1))
    assert cmds == [C.AccumulateSpeakerAudio()]


def test_rejected_quiet_segment_never_accumulates():
    state, cmds = reduce(State.LISTENING, _ctx(proximity_rms=0.5), _seg(rms=0.1))
    assert cmds == []


def test_rejected_owner_absent_never_accumulates():
    ctx = _ctx(proximity_rms=0.0, presence=PresenceStatus.ABSENT)
    state, cmds = reduce(State.LISTENING, ctx, _seg())
    assert cmds == []


def test_legacy_mode_accept_also_accumulates():
    state, cmds = reduce(State.LISTENING, _ctx(reject=False), _seg(endpoint=0.9))
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]


def test_legacy_mode_nontarget_does_not_accumulate():
    state, cmds = reduce(State.LISTENING, _ctx(reject=False), _seg(is_target=False))
    assert cmds == []


def test_gate_passing_interjection_accumulates():
    ctx = _ctx()
    ctx.ducked = True
    ev = E.InterjectionSegment(duration_ms=500.0, rms=1.0,
                               is_target=True, speaker_score=0.9)
    state, cmds = reduce(State.EVALUATING, ctx, ev)
    assert cmds == [C.AccumulateSpeakerAudio(), C.TranscribeInterjection()]


def test_rejected_interjection_restores_without_accumulate():
    ctx = _ctx()
    ctx.ducked = True
    ev = E.InterjectionSegment(duration_ms=500.0, rms=0.1,     # below proximity
                               is_target=True, speaker_score=0.9)
    state, cmds = reduce(State.EVALUATING, ctx, ev)
    assert cmds == [C.Restore()]


def test_accepted_segment_seq_echoes_into_commands():
    # The reducer echoes the event's staging seq into the commands so the
    # workers consume exactly THIS segment's staged audio (overwrite-last fix).
    state, cmds = reduce(State.LISTENING, _ctx(), _seg(endpoint=0.9, seq=7))
    assert cmds == [C.AccumulateSpeakerAudio(seq=7), C.TranscribeUserTurn(seq=7)]


def test_accumulated_segment_seq_echoes_into_command():
    state, cmds = reduce(State.LISTENING, _ctx(), _seg(endpoint=0.1, seq=3))
    assert cmds == [C.AccumulateSpeakerAudio(seq=3)]


def test_interjection_seq_echoes_into_commands():
    ctx = _ctx()
    ctx.ducked = True
    ev = E.InterjectionSegment(duration_ms=500.0, rms=1.0,
                               is_target=True, speaker_score=0.9, seq=9)
    state, cmds = reduce(State.EVALUATING, ctx, ev)
    assert cmds == [C.AccumulateSpeakerAudio(seq=9), C.TranscribeInterjection(seq=9)]
