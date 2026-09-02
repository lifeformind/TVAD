"""The Director's pure decision core (spec sections 4-6). reduce() is a single
synchronous transition function: the ONLY mutator of State/Context. No I/O, no
await, no clock — 'now' arrives via Tick events. Workers (Plan 02) translate the
returned Commands into effects."""

import enum

from modes.director.state import State
from modes.director.context import Context
from modes.director import events as E
from modes.director import commands as C
from modes.director.events import PresenceStatus
from modes.talkback.intent import Interjection, classify_interjection
from core.audio.doa_math import (circular_distance, circular_ema,
                                 circular_median, fraction_vote)


def silence_duration(state: State, ctx: Context) -> float:
    """User-silence seconds — accrues ONLY while waiting for the user (spec s5)."""
    if state is not State.LISTENING:
        return 0.0
    return ctx.now - ctx.last_speech_at


def reduce(state: State, ctx: Context, event) -> tuple:
    if isinstance(event, E.Tick):
        return _on_tick(state, ctx, event)
    if isinstance(event, E.SegmentEndpointed) and state is State.LISTENING:
        return _on_user_segment(ctx, event)
    if isinstance(event, E.UserTurnTranscribed) and state is State.LISTENING:
        return _on_user_transcribed(ctx, event)
    if isinstance(event, E.FirstTtsFrame):
        if state is State.THINKING and event.gen_id == ctx.gen_id:
            return State.SPEAKING, []
        return state, []
    if isinstance(event, E.AssistantPartial):
        if event.gen_id == ctx.gen_id:
            ctx.partial_response = event.text       # spoken-so-far, for barge-in record
        return state, []
    if isinstance(event, E.ReplyComplete):
        if state in (State.THINKING, State.SPEAKING) and event.gen_id == ctx.gen_id:
            if event.assistant_text:
                ctx.conversation.add_assistant_turn(event.assistant_text)
            ctx.last_reply_done_at = ctx.now   # echo-guard tail anchor
            _enter_listening(ctx)
            return State.LISTENING, []
        if state is State.EVALUATING and event.gen_id == ctx.gen_id:
            # The duck coincided with the reply finishing. Record the turn now
            # (else history desyncs) and remember it's done, but stay EVALUATING
            # until the interjection is resolved. reply_done steers the exit:
            # a reject must go to LISTENING (no more ReplyComplete is coming),
            # NOT back to SPEAKING where the kiosk would sit mute until the cap.
            if event.assistant_text:
                ctx.conversation.add_assistant_turn(event.assistant_text)
            ctx.reply_done = True
            ctx.last_reply_done_at = ctx.now   # echo-guard tail anchor
            return State.EVALUATING, []
        return state, []
    if isinstance(event, E.NearFieldOnset) and state is State.SPEAKING:
        onset_floor = max(ctx.proximity_rms, ctx.cfg.onset_floor_speaking)
        if (event.is_target and event.rms >= onset_floor
                and in_owner_cone(ctx, event.doa_angle) is not False):
            ctx.ducked = True
            return State.EVALUATING, [C.Duck(ctx.cfg.duck_level)]
        return State.SPEAKING, []
    if isinstance(event, E.InterjectionSegment) and state is State.EVALUATING:
        return _on_interjection_segment(ctx, event)
    if isinstance(event, E.InterjectionTranscribed) and state is State.EVALUATING:
        return _on_interjection_transcribed(ctx, event)
    if isinstance(event, E.OwnerPresenceEvent):
        ctx.presence_status = event.status      # pure state-recording: NO transition
        ctx.presence_since = event.now          # the owner-absent decision is on Tick
        return state, []
    if isinstance(event, E.SpeakerWindowVerdict):
        return _on_speaker_window_verdict(state, ctx, event)
    return state, []


def _on_tick(state: State, ctx: Context, ev: E.Tick) -> tuple:
    ctx.now = ev.now
    # Hard cap first (spec s5): beats silence when both expire on one tick.
    if ctx.now - ctx.started_at >= ctx.cfg.hard_timeout_s:
        return State.IDLE, [C.EndSession("hard_timeout")]
    sil = silence_duration(state, ctx)
    if sil >= ctx.cfg.silence_timeout_s:
        return State.IDLE, [C.EndSession("silence_timeout")]
    if sil >= (ctx.cfg.silence_timeout_s - ctx.cfg.nudge_lead_s) and not ctx.nudged_cycle:
        ctx.nudged_cycle = True
        return state, [C.SpeakNudge()]      # non-terminal: stay in LISTENING
    # Camera floor control (Director-07): owner physically gone -> free the kiosk
    # fast. Add-on only — runs AFTER the unchanged hard/silence/nudge checks, and
    # the watchdog is still the sole timeout authority (a condition, not a timer).
    if (ctx.presence_status is PresenceStatus.ABSENT
            and ctx.now - ctx.presence_since >= ctx.cfg.owner_absent_grace_s
            and ctx.now - ctx.last_speech_at >= ctx.cfg.active_talk_guard_s):
        return State.IDLE, [C.EndSession("owner_absent")]
    return state, []


def _enter_listening(ctx: Context) -> None:
    """Yield the floor: restart the silence grace window and re-arm the nudge."""
    ctx.last_speech_at = ctx.now
    ctx.nudged_cycle = False
    ctx.reply_done = False


def _on_speaker_window_verdict(state: State, ctx: Context,
                               ev: E.SpeakerWindowVerdict) -> tuple:
    """Director-09 hijack/verify ladder (spec s4) — port of lockout.py's decision
    half. Window 1 failing == bad enrollment (verify-before-serve semantics, no
    wake hold). Later failures: first smoother-fail == WARN (log-only, runtime
    DIAG); a second consecutive fail that is ALSO below the proximity floor ==
    EJECT (silent — never answer a stranger). A passing window resets the streak.
    cfg.lockout_enabled False == shadow mode: counters advance, nothing ends.
    Never touches the silence clock — verdicts are not user speech."""
    ctx.windows_seen += 1
    if ev.smoother_ok:
        ctx.miss_streak = 0
        return state, []
    ctx.miss_streak += 1
    if not ctx.cfg.lockout_enabled:
        return state, []
    if ctx.windows_seen == 1:
        return State.IDLE, [C.EndSession("enroll_verify_failed")]
    if ctx.miss_streak >= 2 and ev.window_rms < ctx.proximity_rms:
        return State.IDLE, [C.EndSession("speaker_mismatch")]
    return state, []


class TurnVerdict(enum.Enum):
    ACCEPT = "accept"                       # complete owner turn -> transcribe
    ACCUMULATE = "accumulate"               # plausibly-owner, endpoint not yet met
    REJECT_NOT_TARGET = "not_target"        # pVAD bystander
    REJECT_TOO_QUIET = "too_quiet"          # below proximity floor (distant bystander)
    REJECT_OWNER_ABSENT = "owner_absent_frame"  # owner not in camera frame
    REJECT_ECHO = "echo_firmware_silent"        # own-TTS echo: DOA tuple present but empty
    REJECT_SPEAKER_UNVERIFIED = "speaker_unverified"  # safety-net streak active
    REJECT_OUT_OF_CONE = "out_of_cone"      # DOA outside the owner cone (Director-11)


def in_owner_cone(ctx: Context, doa_angle):
    """Scalar direction vote (Director-11), used by the duck-at-onset reflex:
    None = abstain (no signal — fail open, the other gates decide), True =
    within the cone, False = out of cone."""
    if doa_angle is None or ctx.owner_bearing is None:
        return None
    return circular_distance(doa_angle, ctx.owner_bearing) <= ctx.cfg.doa_cone_deg


def _in_cone_samples(ctx: Context, doa_angles) -> list:
    return [a for a in doa_angles
            if circular_distance(a, ctx.owner_bearing) <= ctx.cfg.doa_cone_deg]


def cone_vote(ctx: Context, doa_angles):
    """Segment direction vote (Director-11): did the OWNER speak during this
    segment — NOT who spoke most. With continuous background speech the VAD
    merges the owner's utterance into a bystander-dominated segment, so a
    duration-majority median votes the bystander and rejects the owner (live
    2026-07-07 18:42). None = abstain: no samples, no bearing, or fewer than
    min_in_cone_samples total (thin evidence never rejects — fail open)."""
    if not doa_angles or ctx.owner_bearing is None:
        return None
    return fraction_vote(doa_angles, ctx.owner_bearing, ctx.cfg.doa_cone_deg,
                         ctx.cfg.doa_min_in_cone_fraction,
                         ctx.cfg.doa_min_in_cone_samples)


def cone_diag(ctx: Context, doa_angles) -> str:
    """DIAG-only: the cone vote's inputs, so a live REJECT line shows WHY.
    Pure — the runtime prints."""
    if ctx.owner_bearing is None:
        return "doa=[no bearing]"
    if not doa_angles:
        return f"doa=[n=0] bearing={ctx.owner_bearing:.0f}°"
    hits = _in_cone_samples(ctx, doa_angles)
    return (f"doa=[n={len(doa_angles)} in_cone={len(hits)} "
            f"med={circular_median(doa_angles):.0f}°] "
            f"bearing={ctx.owner_bearing:.0f}°")


def classify_new_turn(ctx: Context, ev: E.SegmentEndpointed) -> TurnVerdict:
    """Pure verdict for a LISTENING SegmentEndpointed. Single source of truth for the
    reducer (decide) and the runtime (DIAG). reject_bystanders off -> legacy verdict."""
    complete = ev.endpoint_prob >= ctx.cfg.endpoint_threshold
    if not ev.is_target:
        return TurnVerdict.REJECT_NOT_TARGET
    # Loud-bystander suspension (2026-07-07 live): a podcast's rms overlapped the
    # owner's seed, so the proximity floor can't separate them — but the ECAPA
    # windows do. While the safety net's miss streak is active, nothing is served
    # until one passing window (the owner speaking) resets it.
    if ctx.cfg.lockout_enabled and ctx.miss_streak >= 2:
        return TurnVerdict.REJECT_SPEAKER_UNVERIFIED
    # Firmware-silence echo guard (see DirectorConfig.echo_guard_tail_s): only
    # inside the post-reply tail, only when the tracker EXISTS and saw nothing.
    if (ctx.cfg.echo_guard_tail_s > 0.0
            and ev.doa_angles is not None and len(ev.doa_angles) == 0
            and ctx.now - ctx.last_reply_done_at <= ctx.cfg.echo_guard_tail_s):
        return TurnVerdict.REJECT_ECHO
    if not ctx.cfg.reject_bystanders:
        return TurnVerdict.ACCEPT if complete else TurnVerdict.ACCUMULATE
    if ev.rms < ctx.proximity_rms:
        return TurnVerdict.REJECT_TOO_QUIET
    if cone_vote(ctx, ev.doa_angles) is False:
        return TurnVerdict.REJECT_OUT_OF_CONE
    if ctx.presence_status is PresenceStatus.ABSENT:
        return TurnVerdict.REJECT_OWNER_ABSENT
    return TurnVerdict.ACCEPT if complete else TurnVerdict.ACCUMULATE


def gate_diag_reason(ctx: Context, ev: E.SegmentEndpointed):
    """DIAG-only: the reject reason for a new-turn segment, or None if accepted /
    still accumulating. None for non-reject verdicts keeps the log quiet."""
    v = classify_new_turn(ctx, ev)
    if v in (TurnVerdict.REJECT_NOT_TARGET, TurnVerdict.REJECT_TOO_QUIET,
             TurnVerdict.REJECT_OWNER_ABSENT, TurnVerdict.REJECT_SPEAKER_UNVERIFIED,
             TurnVerdict.REJECT_OUT_OF_CONE, TurnVerdict.REJECT_ECHO):
        return v.value
    return None


def safety_diag_line(ctx: Context, ev, commands) -> str:
    """DIAG-only formatting for a SpeakerWindowVerdict (spec s7). Pure — the
    runtime prints. Call AFTER dispatch: ctx counters are already advanced."""
    line = (f"safety-net window={ctx.windows_seen} score={ev.score:.3f} "
            f"smoother_ok={ev.smoother_ok} streak={ctx.miss_streak} "
            f"rms={ev.window_rms:.4f}")
    if ev.norm_score is not None:
        line = f"{line} norm={ev.norm_score:.2f}"
    ends = [c for c in commands if isinstance(c, C.EndSession)]
    if ends:
        return f"{line} EJECT reason={ends[0].reason}"
    if not ev.smoother_ok:
        if ctx.cfg.lockout_enabled:
            return f"{line} WARN"
        would = ""
        if ctx.windows_seen == 1:
            would = " would_end=enroll_verify_failed"
        elif ctx.miss_streak >= 2 and ev.window_rms < ctx.proximity_rms:
            would = " would_end=speaker_mismatch"
        return f"{line} WARN (shadow){would}"
    return line


def _on_user_segment(ctx: Context, ev: E.SegmentEndpointed) -> tuple:
    v = classify_new_turn(ctx, ev)
    if v is TurnVerdict.REJECT_SPEAKER_UNVERIFIED:
        # Suspended, not ejected: keep feeding the verifier so the owner's next
        # utterance can unlock, but never serve and never reset the silence
        # clock — a podcast must not hold the session open.
        return State.LISTENING, [C.AccumulateSpeakerAudio(seq=ev.seq)]
    if v in (TurnVerdict.ACCEPT, TurnVerdict.ACCUMULATE):
        # Plausibly-owner speech: reset the silence clock and feed the safety
        # net (Director-09) — only served speech reaches the hijack buffer, so
        # D08-rejected bystander chatter can never eject the owner (spec s3.2).
        ctx.last_speech_at = ctx.now
        cmds = [C.AccumulateSpeakerAudio(seq=ev.seq)]
        if v is TurnVerdict.ACCEPT:
            cmds.append(C.TranscribeUserTurn(seq=ev.seq))
            if cone_vote(ctx, ev.doa_angles) is True:
                # Track toward the median of the IN-CONE samples only — a
                # served mixed segment must not drag the bearing toward the
                # bystander's share of it.
                ctx.owner_bearing = circular_ema(
                    ctx.owner_bearing,
                    circular_median(_in_cone_samples(ctx, ev.doa_angles)),
                    ctx.cfg.doa_bearing_ema_alpha)
        return State.LISTENING, cmds
    # Rejected. Legacy mode (reject_bystanders off) keeps its historical clock
    # reset on ANY voiced segment; reject-by-default does not.
    if not ctx.cfg.reject_bystanders:
        ctx.last_speech_at = ctx.now
    return State.LISTENING, []


def _on_user_transcribed(ctx: Context, ev: E.UserTurnTranscribed) -> tuple:
    if not ev.text.strip() or ev.mean_word_prob < ctx.cfg.conf_floor:
        return State.LISTENING, []               # empty/garbage STT: drop, keep listening
    return _start_generation(ctx, ev.text)


def _start_generation(ctx: Context, query: str) -> tuple:
    ctx.conversation.add_user_turn(query)
    ctx.current_query = query
    ctx.partial_response = ""
    ctx.reply_done = False
    ctx.gen_id += 1
    steer = ctx.pending_steer
    ctx.pending_steer = None                      # one-shot
    return State.THINKING, [C.StartGeneration(ctx.gen_id,
                                              ctx.conversation.get_messages(), steer)]


def _restore_speaking(ctx: Context) -> tuple:
    """Un-duck and exit EVALUATING. If the reply finished while we were ducked,
    there's nothing left to speak -> go LISTENING; otherwise resume SPEAKING."""
    ctx.ducked = False
    if ctx.reply_done:
        _enter_listening(ctx)                # also clears reply_done
        return State.LISTENING, [C.Restore()]
    return State.SPEAKING, [C.Restore()]


def interjection_reject_reason(ctx: Context, ev: E.InterjectionSegment):
    """The interjection reject ladder (spec s6), as a pure reason. None = accept.
    Single source of truth: _on_interjection_segment decides with it, and the
    runtime's DIAG line prints it — decision and diagnosis can't drift."""
    if (ctx.cfg.echo_guard_tail_s > 0.0
            and ev.doa_angles is not None and len(ev.doa_angles) == 0):
        return "firmware_silent"                          # own-TTS echo (guard doc: config.py)
    if ev.rms < ctx.proximity_rms:                       # proximity pre-gate
        return "too_quiet"
    if cone_vote(ctx, ev.doa_angles) is False:           # wrong direction (Director-11)
        return "out_of_cone"
    if ev.duration_ms < ctx.cfg.verify_window_ms:        # too short to verify
        return "too_short"
    if ev.speaker_score < ctx.cfg.speaker_threshold:     # not the primary speaker
        return "speaker_mismatch"
    return None


def interjection_diag_line(ctx: Context, ev: E.InterjectionSegment):
    """DIAG-only formatting for a rejected interjection (None when accepted).
    Pure — the runtime prints. Live 2026-09-02: Restore storms were unexplained."""
    reason = interjection_reject_reason(ctx, ev)
    if reason is None:
        return None
    return (f"interjection REJECT={reason} rms={ev.rms:.4f} "
            f"dur={ev.duration_ms:.0f}ms score={ev.speaker_score:.3f}")


def _on_interjection_segment(ctx: Context, ev: E.InterjectionSegment) -> tuple:
    # NEVER cut on a rejected interjection; always RESTORE.
    if interjection_reject_reason(ctx, ev) is not None:
        return _restore_speaking(ctx)
    return State.EVALUATING, [C.AccumulateSpeakerAudio(seq=ev.seq),
                              C.TranscribeInterjection(seq=ev.seq)]


def _on_interjection_transcribed(ctx: Context, ev: E.InterjectionTranscribed) -> tuple:
    if ev.mean_word_prob < ctx.cfg.conf_floor:           # low-confidence garbage
        return _restore_speaking(ctx)
    if classify_interjection(ev.text) is Interjection.BACKCHANNEL:
        return _restore_speaking(ctx)                    # keep talking; no history change
    # INTERRUPT: cut the in-flight reply and answer the new question.
    old_gen = ctx.gen_id
    if not ctx.reply_done:
        # Record an assistant turn for the cut reply, even if nothing was spoken
        # yet — the LLM returns an EMPTY response for non-alternating history (two
        # user turns in a row), which silently breaks every later reply. When
        # reply_done, the ReplyComplete handler ALREADY recorded the full turn —
        # recording again here would put two assistant turns in a row (same bug,
        # mirrored), and a completed reply was not actually interrupted.
        spoken = ctx.partial_response.strip()
        ctx.conversation.add_assistant_turn(
            f"{spoken} [interrupted]" if spoken else "[interrupted]")
        ctx.interrupted_stack.append({"query": ctx.current_query,
                                      "partial": ctx.partial_response})
    ctx.ducked = False
    state, cmds = _start_generation(ctx, ev.text)        # adds user turn, bumps gen_id, THINKING
    # Restore() un-ducks the playback gain (Duck dropped it to duck_level when we
    # entered EVALUATING). Without it the gain stays pinned at duck_level and the
    # new reply — and every reply after — plays near-silent.
    return state, [C.Cut(old_gen), C.Restore()] + cmds
