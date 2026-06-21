"""The Director's pure decision core (spec sections 4-6). reduce() is a single
synchronous transition function: the ONLY mutator of State/Context. No I/O, no
await, no clock — 'now' arrives via Tick events. Workers (Plan 02) translate the
returned Commands into effects."""

from modes.director.state import State
from modes.director.context import Context
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.intent import Interjection, classify_interjection


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
    if isinstance(event, E.ReplyComplete):
        if state in (State.THINKING, State.SPEAKING) and event.gen_id == ctx.gen_id:
            if event.assistant_text:
                ctx.conversation.add_assistant_turn(event.assistant_text)
            _enter_listening(ctx)
            return State.LISTENING, []
        return state, []
    if isinstance(event, E.NearFieldOnset) and state is State.SPEAKING:
        if event.is_target and event.rms >= ctx.proximity_rms:
            ctx.ducked = True
            return State.EVALUATING, [C.Duck(ctx.cfg.duck_level)]
        return State.SPEAKING, []
    if isinstance(event, E.InterjectionSegment) and state is State.EVALUATING:
        return _on_interjection_segment(ctx, event)
    if isinstance(event, E.InterjectionTranscribed) and state is State.EVALUATING:
        return _on_interjection_transcribed(ctx, event)
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
    return state, []


def _enter_listening(ctx: Context) -> None:
    """Yield the floor: restart the silence grace window and re-arm the nudge."""
    ctx.last_speech_at = ctx.now
    ctx.nudged_cycle = False


def _on_user_segment(ctx: Context, ev: E.SegmentEndpointed) -> tuple:
    ctx.last_speech_at = ctx.now                 # any user activity resets the clock
    if not ev.is_target:
        return State.LISTENING, []               # bystander: ignore
    if ev.endpoint_prob < ctx.cfg.endpoint_threshold:
        return State.LISTENING, []               # turn not complete: keep accumulating
    return State.LISTENING, [C.TranscribeUserTurn()]


def _on_user_transcribed(ctx: Context, ev: E.UserTurnTranscribed) -> tuple:
    if not ev.text.strip() or ev.mean_word_prob < ctx.cfg.conf_floor:
        return State.LISTENING, []               # empty/garbage STT: drop, keep listening
    return _start_generation(ctx, ev.text)


def _start_generation(ctx: Context, query: str) -> tuple:
    ctx.conversation.add_user_turn(query)
    ctx.current_query = query
    ctx.partial_response = ""
    ctx.gen_id += 1
    steer = ctx.pending_steer
    ctx.pending_steer = None                      # one-shot
    return State.THINKING, [C.StartGeneration(ctx.gen_id,
                                              ctx.conversation.get_messages(), steer)]


def _restore_speaking(ctx: Context) -> tuple:
    """Un-duck and keep talking — the non-cut EVALUATING exits."""
    ctx.ducked = False
    return State.SPEAKING, [C.Restore()]


def _on_interjection_segment(ctx: Context, ev: E.InterjectionSegment) -> tuple:
    # Reject ladder (spec s6) — NEVER cut on these; always RESTORE.
    if ev.rms < ctx.proximity_rms:                       # proximity pre-gate
        return _restore_speaking(ctx)
    if ev.duration_ms < ctx.cfg.verify_window_ms:        # too short to verify
        return _restore_speaking(ctx)
    if ev.speaker_score < ctx.cfg.speaker_threshold:     # not the primary speaker
        return _restore_speaking(ctx)
    return State.EVALUATING, [C.TranscribeInterjection()]


def _on_interjection_transcribed(ctx: Context, ev: E.InterjectionTranscribed) -> tuple:
    if ev.mean_word_prob < ctx.cfg.conf_floor:           # low-confidence garbage
        return _restore_speaking(ctx)
    if classify_interjection(ev.text) is Interjection.BACKCHANNEL:
        return _restore_speaking(ctx)                    # keep talking; no history change
    # INTERRUPT: cut the in-flight reply and answer the new question.
    old_gen = ctx.gen_id
    if ctx.partial_response:
        ctx.conversation.add_assistant_turn(ctx.partial_response + " [interrupted]")
    ctx.interrupted_stack.append({"query": ctx.current_query,
                                  "partial": ctx.partial_response})
    ctx.ducked = False
    state, cmds = _start_generation(ctx, ev.text)        # adds user turn, bumps gen_id, THINKING
    return state, [C.Cut(old_gen)] + cmds
