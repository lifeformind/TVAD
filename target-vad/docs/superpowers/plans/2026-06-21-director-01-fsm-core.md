# Director FSM Core (Plan 01) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Conversation Director's decision core as a pure, deterministic `reduce(state, event) -> (state, commands)` reducer with the 5-state FSM, the suspend-during-speech silence timeout, the once-per-cycle "still there?" nudge, and the EVALUATING backchannel-vs-question ladder — fully unit-tested with synthetic events and **zero I/O** (no models, audio, threads, or asyncio).

**Architecture:** A single synchronous transition function is the Director's brain. It owns FSM `State` + a `Context` blackboard and is the sole mutator. Workers (built in Plan 02) will translate audio/STT/LLM/TTS into `Event`s and execute the returned `Command`s; this plan builds and tests the brain in isolation. Time is injected via `Tick(now)` events so the reducer never calls a clock — making every timeout/nudge path deterministic.

**Tech Stack:** Python 3.12 (`python3`, no `python` on PATH), pytest. Reuses `modes/talkback/conversation.py` (`ConversationManager`) and `modes/talkback/intent.py` (`classify_interjection`, `Interjection`) verbatim. No new third-party dependencies.

## Global Constraints

- Target/dev box: NVIDIA DGX Spark GB10, aarch64, Python 3.12. Run tests with `python3 -m pytest`.
- **Single-mutator rule:** only `reduce()` mutates `State`/`Context`. No `await`, no I/O, no threads, no `time.monotonic()`, no `asyncio` anywhere in this plan's modules. The reducer is a pure function of `(state, ctx, event)`.
- **Time is injected:** the reducer reads "now" only from `Tick(now: float)` events (stored as `ctx.now`); it never calls a clock.
- **Silence accrues only in LISTENING** (spec §5): `silence_duration` is `ctx.now - ctx.last_speech_at` iff `state is State.LISTENING`, else `0.0`.
- **gen_id discipline (spec §11):** every generation carries a monotone `ctx.gen_id`; events tagged with a stale `gen_id` are dropped.
- New package lives under `modes/director/`; tests under `tests/director/`.
- Reuse, do not reimplement: `ConversationManager` (`modes/talkback/conversation.py:7`), `classify_interjection`/`Interjection` (`modes/talkback/intent.py:57`/`:10`).

---

## File Structure

- `modes/director/__init__.py` — empty package marker.
- `modes/director/state.py` — `State` enum (the 5 FSM states).
- `modes/director/config.py` — `DirectorConfig` frozen dataclass (all thresholds, copied from spec/config.yaml).
- `modes/director/context.py` — `Context` dataclass (the mutable session blackboard) + `new_context()` factory.
- `modes/director/events.py` — frozen `Event` dataclasses (inputs the reducer consumes).
- `modes/director/commands.py` — frozen `Command` dataclasses (outputs the reducer emits).
- `modes/director/reducer.py` — `reduce(state, ctx, event) -> tuple[State, list[Command]]` (the pure brain) + small private helpers.
- `modes/director/director.py` — `Director` thin wrapper: holds `(state, ctx)`, `dispatch(event)` calls `reduce` and applies the new state (single mutator entry point). Still no I/O.
- `tests/director/__init__.py`, `tests/director/test_*.py` — one test module per task.

---

## Task 1: Core types — State, DirectorConfig, Context

**Files:**
- Create: `modes/director/__init__.py` (empty), `tests/director/__init__.py` (empty)
- Create: `modes/director/state.py`, `modes/director/config.py`, `modes/director/context.py`
- Test: `tests/director/test_core_types.py`

**Interfaces:**
- Produces: `State` (enum: IDLE, LISTENING, THINKING, SPEAKING, EVALUATING). `DirectorConfig` (frozen dataclass, fields below). `Context` (dataclass, fields below). `new_context(cfg, conversation, now, proximity_rms) -> Context`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_core_types.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import Context, new_context
from modes.talkback.conversation import ConversationManager


def test_state_has_five_members():
    assert {s.name for s in State} == {
        "IDLE", "LISTENING", "THINKING", "SPEAKING", "EVALUATING"}


def test_config_defaults_match_spec():
    c = DirectorConfig()
    assert c.silence_timeout_s == 30.0
    assert c.hard_timeout_s == 300.0
    assert c.nudge_lead_s == 5.0
    assert 0.0 < c.nudge_lead_s < c.silence_timeout_s
    assert c.endpoint_threshold == 0.5
    assert c.verify_window_ms == 700.0
    assert c.min_speech_ms == 120.0
    assert c.speaker_threshold == 0.20
    assert c.conf_floor == 0.5
    assert c.duck_level == 0.15


def test_new_context_starts_a_session_clock():
    cfg = DirectorConfig()
    conv = ConversationManager(system_prompt="sys")
    ctx = new_context(cfg, conv, now=100.0, proximity_rms=0.02)
    assert ctx.now == 100.0
    assert ctx.started_at == 100.0
    assert ctx.last_speech_at == 100.0
    assert ctx.gen_id == 0
    assert ctx.nudged_cycle is False
    assert ctx.ducked is False
    assert ctx.interrupted_stack == []
    assert ctx.pending_steer is None
    assert ctx.proximity_rms == 0.02
    assert ctx.conversation is conv
    assert ctx.cfg is cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_core_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director'`

- [ ] **Step 3: Create the package + state**

```python
# modes/director/__init__.py
```
```python
# tests/director/__init__.py
```
```python
# modes/director/state.py
"""The Director's five explicit FSM states (spec section 4)."""

import enum


class State(enum.Enum):
    IDLE = "IDLE"            # no session (lives in the thin WakeGate, Plan 03)
    LISTENING = "LISTENING"  # waiting for / capturing the user's turn
    THINKING = "THINKING"    # LLM generating, no audible TTS yet
    SPEAKING = "SPEAKING"    # TTS audio playing out
    EVALUATING = "EVALUATING"  # near-field onset during SPEAKING: ducked, deciding
```

- [ ] **Step 4: Create the config**

```python
# modes/director/config.py
"""DirectorConfig — all decision thresholds, copied from spec section 5/6 and
config.yaml (kiosk.talkback.*). Frozen: the reducer never mutates config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectorConfig:
    # Timeouts / nudge (spec section 5)
    silence_timeout_s: float = 30.0      # config.yaml:52
    hard_timeout_s: float = 300.0        # config.yaml:53
    nudge_lead_s: float = 5.0            # NEW: nudge fires at silence == 30-5 = 25s
    # Turn-taking (spec section 6)
    endpoint_threshold: float = 0.5      # Smart Turn endpoint_prob >= this => turn complete
    min_speech_ms: float = 120.0         # barge_in.min_speech_ms (config.yaml)
    verify_window_ms: float = 700.0      # barge_in.verify_window_ms (config.yaml:131)
    speaker_threshold: float = 0.20      # barge_in.speaker_threshold (config.yaml:130)
    conf_floor: float = 0.5             # NEW: mean_word_prob below this => RESTORE
    duck_level: float = 0.15            # barge_in.duck_level (config.yaml)
```

- [ ] **Step 5: Create the context**

```python
# modes/director/context.py
"""Context — the Director's mutable session blackboard. Only reduce() mutates it."""

from dataclasses import dataclass, field
from typing import Optional

from modes.director.config import DirectorConfig
from modes.talkback.conversation import ConversationManager


@dataclass
class Context:
    cfg: DirectorConfig
    conversation: ConversationManager
    proximity_rms: float
    now: float                      # latest injected clock (from Tick events)
    started_at: float               # session start (for hard timeout)
    last_speech_at: float           # last time we began waiting for the user
    gen_id: int = 0                 # monotone; tags every generation
    nudged_cycle: bool = False      # already nudged this LISTENING cycle?
    ducked: bool = False            # is TTS currently ducked?
    current_query: str = ""         # the request being answered (for resume)
    partial_response: str = ""      # assistant text spoken so far this turn
    pending_steer: Optional[str] = None  # one-shot LLM steer (resume), Plan 06 fills it
    interrupted_stack: list = field(default_factory=list)  # bounded in Plan 06


def new_context(cfg: DirectorConfig, conversation: ConversationManager,
                now: float, proximity_rms: float) -> Context:
    return Context(
        cfg=cfg, conversation=conversation, proximity_rms=proximity_rms,
        now=now, started_at=now, last_speech_at=now,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_core_types.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add modes/director/__init__.py modes/director/state.py modes/director/config.py modes/director/context.py tests/director/__init__.py tests/director/test_core_types.py
git commit -m "feat(director): core FSM types — State, DirectorConfig, Context"
```

---

## Task 2: Event and Command types

**Files:**
- Create: `modes/director/events.py`, `modes/director/commands.py`
- Test: `tests/director/test_event_command_types.py`

**Interfaces:**
- Produces (Events): `Tick(now)`, `SegmentEndpointed(duration_ms, rms, is_target, endpoint_prob)`, `UserTurnTranscribed(text, mean_word_prob)`, `NearFieldOnset(rms, is_target)`, `InterjectionSegment(duration_ms, rms, is_target, speaker_score)`, `InterjectionTranscribed(text, mean_word_prob)`, `FirstTtsFrame(gen_id)`, `ReplyComplete(gen_id, assistant_text)`.
- Produces (Commands): `Duck(level)`, `Restore()`, `Cut(gen_id)`, `StartGeneration(gen_id, messages, steer)`, `TranscribeUserTurn()`, `TranscribeInterjection()`, `SpeakNudge()`, `EndSession(reason)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_event_command_types.py
import pytest

from modes.director import events as E
from modes.director import commands as C


def test_events_construct_and_are_frozen():
    seg = E.SegmentEndpointed(duration_ms=900.0, rms=0.4, is_target=True,
                              endpoint_prob=0.8)
    assert seg.is_target is True
    with pytest.raises(Exception):
        seg.is_target = False  # frozen


def test_commands_construct_and_are_frozen():
    cmd = C.StartGeneration(gen_id=3, messages=[{"role": "user", "content": "hi"}],
                            steer=None)
    assert cmd.gen_id == 3
    with pytest.raises(Exception):
        cmd.gen_id = 4  # frozen


def test_tick_and_reply_carry_their_payloads():
    assert E.Tick(now=12.5).now == 12.5
    rc = E.ReplyComplete(gen_id=2, assistant_text="done")
    assert rc.gen_id == 2 and rc.assistant_text == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_event_command_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.events'`

- [ ] **Step 3: Create the events**

```python
# modes/director/events.py
"""Events — inputs the reducer consumes. Workers (Plan 02) emit these. Frozen."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tick:
    now: float                       # injected clock; the reducer's only time source


@dataclass(frozen=True)
class SegmentEndpointed:
    """A completed VAD segment in LISTENING, with reflex signals attached."""
    duration_ms: float
    rms: float
    is_target: bool                  # pVAD/ECAPA verdict (Plan 05 fills it; True for now)
    endpoint_prob: float             # Smart Turn endpoint probability


@dataclass(frozen=True)
class UserTurnTranscribed:
    text: str
    mean_word_prob: float


@dataclass(frozen=True)
class NearFieldOnset:
    """Voiced onset during SPEAKING — the duck-at-onset reflex trigger."""
    rms: float
    is_target: bool


@dataclass(frozen=True)
class InterjectionSegment:
    """The ducked capture in EVALUATING endpointed; gate-ladder inputs attached."""
    duration_ms: float
    rms: float
    is_target: bool
    speaker_score: float             # ECAPA/pVAD score vs primary (off hot path)


@dataclass(frozen=True)
class InterjectionTranscribed:
    text: str
    mean_word_prob: float


@dataclass(frozen=True)
class FirstTtsFrame:
    gen_id: int                      # first audible frame written for this generation


@dataclass(frozen=True)
class ReplyComplete:
    gen_id: int
    assistant_text: str
```

- [ ] **Step 4: Create the commands**

```python
# modes/director/commands.py
"""Commands — outputs the reducer emits for workers (Plan 02) to execute. Frozen."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Duck:
    level: float


@dataclass(frozen=True)
class Restore:
    pass


@dataclass(frozen=True)
class Cut:
    """Drain playback + cancel the main LLM generation (race-fixed teardown, Plan 02)."""
    gen_id: int


@dataclass(frozen=True)
class StartGeneration:
    gen_id: int
    messages: list
    steer: Optional[str]


@dataclass(frozen=True)
class TranscribeUserTurn:
    pass


@dataclass(frozen=True)
class TranscribeInterjection:
    pass


@dataclass(frozen=True)
class SpeakNudge:
    pass


@dataclass(frozen=True)
class EndSession:
    reason: str
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_event_command_types.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add modes/director/events.py modes/director/commands.py tests/director/test_event_command_types.py
git commit -m "feat(director): event and command types"
```

---

## Task 3: reduce() foundation + silence timeout & nudge (Tick handling)

**Files:**
- Create: `modes/director/reducer.py`
- Test: `tests/director/test_timeout_nudge.py`

**Interfaces:**
- Consumes: `State`, `Context`, `Tick` (Task 1, 2).
- Produces: `reduce(state: State, ctx: Context, event) -> tuple[State, list[Command]]`. Helper `silence_duration(state, ctx) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_timeout_nudge.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce, silence_duration
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=now, proximity_rms=0.02)


def test_silence_accrues_only_in_listening():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    ctx.now = 60.0
    assert silence_duration(State.LISTENING, ctx) == 60.0
    for s in (State.THINKING, State.SPEAKING, State.EVALUATING):
        assert silence_duration(s, ctx) == 0.0


def test_no_timeout_while_speaking_even_after_long_reply():
    ctx = _ctx(now=0.0)
    state, cmds = reduce(State.SPEAKING, ctx, E.Tick(now=120.0))
    assert state is State.SPEAKING
    assert cmds == []          # a 120s reply never ends the session


def test_silence_timeout_fires_in_listening():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=30.0))
    assert state is State.IDLE
    assert cmds == [C.EndSession("silence_timeout")]


def test_hard_timeout_fires_in_any_state_and_beats_silence():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    state, cmds = reduce(State.SPEAKING, ctx, E.Tick(now=300.0))
    assert state is State.IDLE
    assert cmds == [C.EndSession("hard_timeout")]


def test_nudge_fires_once_at_lead_and_does_not_end_session():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    # nudge_lead_s=5, silence_timeout_s=30 => nudge at silence>=25
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=25.0))
    assert state is State.LISTENING
    assert cmds == [C.SpeakNudge()]
    assert ctx.nudged_cycle is True
    # next tick before timeout: no second nudge
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=27.0))
    assert cmds == []
    # then the real timeout still fires
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=30.0))
    assert state is State.IDLE and cmds == [C.EndSession("silence_timeout")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_timeout_nudge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.reducer'`

- [ ] **Step 3: Implement reduce() with the Tick branch**

```python
# modes/director/reducer.py
"""The Director's pure decision core (spec sections 4-6). reduce() is a single
synchronous transition function: the ONLY mutator of State/Context. No I/O, no
await, no clock — 'now' arrives via Tick events. Workers (Plan 02) translate the
returned Commands into effects."""

from modes.director.state import State
from modes.director.context import Context
from modes.director import events as E
from modes.director import commands as C


def silence_duration(state: State, ctx: Context) -> float:
    """User-silence seconds — accrues ONLY while waiting for the user (spec s5)."""
    if state is not State.LISTENING:
        return 0.0
    return ctx.now - ctx.last_speech_at


def reduce(state: State, ctx: Context, event) -> tuple[State, list]:
    if isinstance(event, E.Tick):
        return _on_tick(state, ctx, event)
    return state, []


def _on_tick(state: State, ctx: Context, ev: E.Tick) -> tuple[State, list]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_timeout_nudge.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/reducer.py tests/director/test_timeout_nudge.py
git commit -m "feat(director): reduce() foundation + suspend-during-speech timeout & once-per-cycle nudge"
```

---

## Task 4: LISTENING → THINKING (user-turn capture and dispatch)

**Files:**
- Modify: `modes/director/reducer.py`
- Test: `tests/director/test_listening_turn.py`

**Interfaces:**
- Consumes: `SegmentEndpointed`, `UserTurnTranscribed`; `ConversationManager.add_user_turn`/`get_messages`.
- Produces: in `reduce()`, handling for `SegmentEndpointed` (LISTENING) → `[TranscribeUserTurn()]` when target+complete, else `[]`; and `UserTurnTranscribed` (LISTENING) → adds user turn, bumps `gen_id`, returns `THINKING, [StartGeneration(gen_id, messages, steer)]`. Helper `_enter_listening(ctx)` resets the floor.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_listening_turn.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=now, proximity_rms=0.02)


def test_complete_target_segment_requests_transcription():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.SegmentEndpointed(900.0, 0.4, is_target=True, endpoint_prob=0.8))
    assert state is State.LISTENING
    assert cmds == [C.TranscribeUserTurn()]


def test_incomplete_turn_keeps_accumulating():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.SegmentEndpointed(300.0, 0.4, is_target=True, endpoint_prob=0.2))
    assert state is State.LISTENING and cmds == []


def test_bystander_segment_is_ignored():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.SegmentEndpointed(900.0, 0.4, is_target=False, endpoint_prob=0.9))
    assert state is State.LISTENING and cmds == []


def test_segment_updates_silence_clock_so_session_survives():
    ctx = _ctx(now=0.0)
    ctx.last_speech_at = 0.0
    ctx.now = 29.0
    reduce(State.LISTENING, ctx,
           E.SegmentEndpointed(900.0, 0.4, is_target=True, endpoint_prob=0.8))
    assert ctx.last_speech_at == 29.0   # user activity resets the grace window


def test_transcribed_turn_starts_generation_and_thinks():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
    assert state is State.THINKING
    assert len(cmds) == 1 and isinstance(cmds[0], C.StartGeneration)
    assert cmds[0].gen_id == 1            # bumped from 0
    assert ctx.gen_id == 1
    assert ctx.current_query == "tell me a story"
    assert {"role": "user", "content": "tell me a story"} in ctx.conversation.get_messages()
    assert cmds[0].messages == ctx.conversation.get_messages()


def test_empty_or_low_confidence_user_turn_is_dropped():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="", mean_word_prob=0.9))
    assert state is State.LISTENING and cmds == []
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="hello", mean_word_prob=0.2))
    assert state is State.LISTENING and cmds == []   # below conf_floor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_listening_turn.py -v`
Expected: FAIL — `SegmentEndpointed`/`UserTurnTranscribed` not handled (returns `(state, [])`, so assertions on commands fail)

- [ ] **Step 3: Implement the LISTENING branches**

In `modes/director/reducer.py`, add the imports already present and extend `reduce()` plus add helpers. Replace the body of `reduce` with:

```python
def reduce(state: State, ctx: Context, event) -> tuple[State, list]:
    if isinstance(event, E.Tick):
        return _on_tick(state, ctx, event)
    if isinstance(event, E.SegmentEndpointed) and state is State.LISTENING:
        return _on_user_segment(ctx, event)
    if isinstance(event, E.UserTurnTranscribed) and state is State.LISTENING:
        return _on_user_transcribed(ctx, event)
    return state, []


def _enter_listening(ctx: Context) -> None:
    """Yield the floor: restart the silence grace window and re-arm the nudge."""
    ctx.last_speech_at = ctx.now
    ctx.nudged_cycle = False


def _on_user_segment(ctx: Context, ev: E.SegmentEndpointed) -> tuple[State, list]:
    ctx.last_speech_at = ctx.now                 # any user activity resets the clock
    if not ev.is_target:
        return State.LISTENING, []               # bystander: ignore
    if ev.endpoint_prob < ctx.cfg.endpoint_threshold:
        return State.LISTENING, []               # turn not complete: keep accumulating
    return State.LISTENING, [C.TranscribeUserTurn()]


def _on_user_transcribed(ctx: Context, ev: E.UserTurnTranscribed) -> tuple[State, list]:
    if not ev.text.strip() or ev.mean_word_prob < ctx.cfg.conf_floor:
        return State.LISTENING, []               # empty/garbage STT: drop, keep listening
    return _start_generation(ctx, ev.text)


def _start_generation(ctx: Context, query: str) -> tuple[State, list]:
    ctx.conversation.add_user_turn(query)
    ctx.current_query = query
    ctx.partial_response = ""
    ctx.gen_id += 1
    steer = ctx.pending_steer
    ctx.pending_steer = None                      # one-shot
    return State.THINKING, [C.StartGeneration(ctx.gen_id,
                                              ctx.conversation.get_messages(), steer)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_listening_turn.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/reducer.py tests/director/test_listening_turn.py
git commit -m "feat(director): LISTENING turn capture -> THINKING (target/endpoint/conf gates)"
```

---

## Task 5: THINKING → SPEAKING → LISTENING (generation lifecycle + gen_id staleness)

**Files:**
- Modify: `modes/director/reducer.py`
- Test: `tests/director/test_speaking_lifecycle.py`

**Interfaces:**
- Consumes: `FirstTtsFrame(gen_id)`, `ReplyComplete(gen_id, assistant_text)`; `ConversationManager.add_assistant_turn`.
- Produces: `reduce()` handles `FirstTtsFrame` (THINKING→SPEAKING iff gen_id matches), `ReplyComplete` (THINKING|SPEAKING→LISTENING iff gen_id matches, records assistant turn, yields floor). Stale gen_id events are dropped.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_speaking_lifecycle.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=now, proximity_rms=0.02)


def test_first_tts_frame_moves_thinking_to_speaking():
    ctx = _ctx(); ctx.gen_id = 1
    state, cmds = reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=1))
    assert state is State.SPEAKING and cmds == []


def test_stale_first_tts_frame_is_dropped():
    ctx = _ctx(); ctx.gen_id = 2
    state, cmds = reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=1))  # old turn
    assert state is State.THINKING and cmds == []


def test_reply_complete_yields_floor_and_records_assistant():
    ctx = _ctx(now=10.0); ctx.gen_id = 1; ctx.last_speech_at = 0.0; ctx.nudged_cycle = True
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.ReplyComplete(gen_id=1, assistant_text="once upon a time"))
    assert state is State.LISTENING and cmds == []
    msgs = ctx.conversation.get_messages()
    assert {"role": "assistant", "content": "once upon a time"} in msgs
    assert ctx.last_speech_at == 10.0     # floor reset on yield
    assert ctx.nudged_cycle is False      # nudge re-armed


def test_stale_reply_complete_is_dropped():
    ctx = _ctx(); ctx.gen_id = 2
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.ReplyComplete(gen_id=1, assistant_text="stale"))
    assert state is State.SPEAKING and cmds == []
    assert ctx.conversation.get_messages() == [{"role": "system", "content": "s"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_speaking_lifecycle.py -v`
Expected: FAIL — `FirstTtsFrame`/`ReplyComplete` unhandled

- [ ] **Step 3: Implement the lifecycle branches**

In `reduce()`, add before the final `return state, []`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_speaking_lifecycle.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/reducer.py tests/director/test_speaking_lifecycle.py
git commit -m "feat(director): generation lifecycle THINKING->SPEAKING->LISTENING with gen_id staleness drop"
```

---

## Task 6: SPEAKING → EVALUATING (duck-at-onset reflex)

**Files:**
- Modify: `modes/director/reducer.py`
- Test: `tests/director/test_onset_duck.py`

**Interfaces:**
- Consumes: `NearFieldOnset(rms, is_target)`.
- Produces: `reduce()` handles `NearFieldOnset` (only in SPEAKING): near-field target onset (`rms >= ctx.proximity_rms`) → `EVALUATING, [Duck(duck_level)]`, sets `ctx.ducked=True`; far/non-target → stay SPEAKING, no duck.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_onset_duck.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx():
    return new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                       now=0.0, proximity_rms=0.02)


def test_near_field_target_onset_ducks_and_evaluates():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.NearFieldOnset(rms=0.5, is_target=True))
    assert state is State.EVALUATING
    assert cmds == [C.Duck(0.15)]
    assert ctx.ducked is True


def test_far_onset_does_not_duck():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.NearFieldOnset(rms=0.001, is_target=True))
    assert state is State.SPEAKING and cmds == [] and ctx.ducked is False


def test_non_target_onset_does_not_duck():
    ctx = _ctx()
    state, cmds = reduce(State.SPEAKING, ctx, E.NearFieldOnset(rms=0.5, is_target=False))
    assert state is State.SPEAKING and cmds == [] and ctx.ducked is False


def test_onset_ignored_when_not_speaking():
    ctx = _ctx()
    state, cmds = reduce(State.LISTENING, ctx, E.NearFieldOnset(rms=0.5, is_target=True))
    assert state is State.LISTENING and cmds == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_onset_duck.py -v`
Expected: FAIL — `NearFieldOnset` unhandled

- [ ] **Step 3: Implement the onset branch**

In `reduce()`, add before the final `return state, []`:

```python
    if isinstance(event, E.NearFieldOnset) and state is State.SPEAKING:
        if event.is_target and event.rms >= ctx.proximity_rms:
            ctx.ducked = True
            return State.EVALUATING, [C.Duck(ctx.cfg.duck_level)]
        return State.SPEAKING, []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_onset_duck.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/reducer.py tests/director/test_onset_duck.py
git commit -m "feat(director): SPEAKING->EVALUATING duck-at-onset reflex"
```

---

## Task 7: EVALUATING gate ladder + backchannel-vs-question decision

**Files:**
- Modify: `modes/director/reducer.py`
- Test: `tests/director/test_evaluating.py`

**Interfaces:**
- Consumes: `InterjectionSegment(duration_ms, rms, is_target, speaker_score)`, `InterjectionTranscribed(text, mean_word_prob)`; `classify_interjection`/`Interjection` (`modes/talkback/intent.py`); `ConversationManager.add_assistant_turn`.
- Produces: in `reduce()`, EVALUATING handling. `InterjectionSegment` runs the reject ladder → `Restore()` + back to SPEAKING (un-duck) OR `[TranscribeInterjection()]` (stay EVALUATING). `InterjectionTranscribed` → BACKCHANNEL/empty/low-conf → `Restore()` + SPEAKING; INTERRUPT → cut: finalize partial as `"... [interrupted]"`, push resume frame, `Cut(old_gen_id)` then `StartGeneration` → THINKING. Helper `_restore_speaking(ctx)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_evaluating.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx():
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                      now=0.0, proximity_rms=0.02)
    ctx.gen_id = 1
    ctx.ducked = True
    ctx.current_query = "tell me a story"
    ctx.partial_response = "once upon a time"
    return ctx


def _seg(duration_ms=900.0, rms=0.5, is_target=True, score=0.9):
    return E.InterjectionSegment(duration_ms, rms, is_target, score)


def test_too_short_interjection_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(duration_ms=300.0))  # < 700ms
    assert state is State.SPEAKING and cmds == [C.Restore()] and ctx.ducked is False


def test_far_interjection_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(rms=0.001))
    assert state is State.SPEAKING and cmds == [C.Restore()]


def test_speaker_mismatch_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg(score=0.05))  # < 0.20
    assert state is State.SPEAKING and cmds == [C.Restore()]


def test_passing_segment_requests_transcription():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx, _seg())
    assert state is State.EVALUATING and cmds == [C.TranscribeInterjection()]


def test_backchannel_restores_and_keeps_talking():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="yeah got it", mean_word_prob=0.9))
    assert state is State.SPEAKING and cmds == [C.Restore()] and ctx.ducked is False
    # backchannel must NOT pollute history
    assert all(m["content"] != "yeah got it" for m in ctx.conversation.get_messages())


def test_empty_interjection_restores():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="", mean_word_prob=0.9))
    assert state is State.SPEAKING and cmds == [C.Restore()]


def test_question_cuts_and_starts_new_turn():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="wait why is that", mean_word_prob=0.9))
    assert state is State.THINKING
    assert isinstance(cmds[0], C.Cut) and cmds[0].gen_id == 1      # cut the OLD generation
    assert isinstance(cmds[1], C.StartGeneration) and cmds[1].gen_id == 2
    assert ctx.ducked is False
    msgs = ctx.conversation.get_messages()
    # interrupted partial preserved + the new user turn added
    assert {"role": "assistant", "content": "once upon a time [interrupted]"} in msgs
    assert {"role": "user", "content": "wait why is that"} in msgs
    # resume frame pushed
    assert ctx.interrupted_stack[-1] == {"query": "tell me a story",
                                         "partial": "once upon a time"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_evaluating.py -v`
Expected: FAIL — EVALUATING events unhandled

- [ ] **Step 3: Implement the EVALUATING branches**

At the top of `modes/director/reducer.py`, add the import:

```python
from modes.talkback.intent import Interjection, classify_interjection
```

In `reduce()`, add before the final `return state, []`:

```python
    if isinstance(event, E.InterjectionSegment) and state is State.EVALUATING:
        return _on_interjection_segment(ctx, event)
    if isinstance(event, E.InterjectionTranscribed) and state is State.EVALUATING:
        return _on_interjection_transcribed(ctx, event)
```

Add the helpers at the end of the module:

```python
def _restore_speaking(ctx: Context) -> tuple[State, list]:
    """Un-duck and keep talking — the non-cut EVALUATING exits."""
    ctx.ducked = False
    return State.SPEAKING, [C.Restore()]


def _on_interjection_segment(ctx: Context, ev: E.InterjectionSegment) -> tuple[State, list]:
    # Reject ladder (spec s6) — NEVER cut on these; always RESTORE.
    if ev.rms < ctx.proximity_rms:                       # proximity pre-gate
        return _restore_speaking(ctx)
    if ev.duration_ms < ctx.cfg.verify_window_ms:        # too short to verify
        return _restore_speaking(ctx)
    if ev.speaker_score < ctx.cfg.speaker_threshold:     # not the primary speaker
        return _restore_speaking(ctx)
    return State.EVALUATING, [C.TranscribeInterjection()]


def _on_interjection_transcribed(ctx: Context, ev: E.InterjectionTranscribed) -> tuple[State, list]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_evaluating.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/reducer.py tests/director/test_evaluating.py
git commit -m "feat(director): EVALUATING gate ladder + backchannel/question cut-or-restore"
```

---

## Task 8: Director wrapper + full synthetic-conversation integration test

**Files:**
- Create: `modes/director/director.py`
- Test: `tests/director/test_director_integration.py`

**Interfaces:**
- Consumes: `reduce`, `State`, `Context`, `new_context`, all events/commands.
- Produces: `Director(cfg, conversation, now, proximity_rms)` with `.state`, `.ctx`, and `dispatch(event) -> list[Command]` (the single mutator entry point: calls `reduce`, stores the new state, returns commands).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_director_integration.py
from modes.director.director import Director
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _director():
    return Director(DirectorConfig(), ConversationManager(system_prompt="s"),
                    now=0.0, proximity_rms=0.02)


def test_dispatch_applies_state_and_returns_commands():
    d = _director()
    assert d.state is State.LISTENING
    cmds = d.dispatch(E.UserTurnTranscribed(text="hi", mean_word_prob=0.9))
    assert d.state is State.THINKING
    assert isinstance(cmds[0], C.StartGeneration)


def test_full_conversation_with_backchannel_question_and_resume_push():
    d = _director()
    # 1. user asks for a story
    d.dispatch(E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
    assert d.state is State.THINKING
    gen = d.ctx.gen_id
    d.dispatch(E.FirstTtsFrame(gen_id=gen)); assert d.state is State.SPEAKING
    d.ctx.partial_response = "once upon a time"
    # 2. backchannel mid-story -> keep talking
    d.dispatch(E.NearFieldOnset(rms=0.5, is_target=True)); assert d.state is State.EVALUATING
    d.dispatch(E.InterjectionSegment(900.0, 0.5, True, 0.9))   # passes gates
    cmds = d.dispatch(E.InterjectionTranscribed(text="mhm", mean_word_prob=0.9))
    assert d.state is State.SPEAKING and cmds == [C.Restore()]
    # 3. a real question mid-story -> cut + answer
    d.dispatch(E.NearFieldOnset(rms=0.5, is_target=True)); assert d.state is State.EVALUATING
    d.dispatch(E.InterjectionSegment(900.0, 0.5, True, 0.9))
    cmds = d.dispatch(E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert d.state is State.THINKING
    assert isinstance(cmds[0], C.Cut) and isinstance(cmds[1], C.StartGeneration)
    assert d.ctx.interrupted_stack[-1]["query"] == "tell me a story"
    # 4. answer the question, THEN prove silence is suspended while speaking.
    d.dispatch(E.FirstTtsFrame(gen_id=d.ctx.gen_id))     # -> SPEAKING
    assert d.state is State.SPEAKING
    # 100s elapse WHILE speaking: silence must NOT accrue, session must NOT end.
    assert d.dispatch(E.Tick(now=100.0)) == []
    assert d.state is State.SPEAKING
    # reply ends -> yield floor; the grace window restarts from now (=100).
    d.dispatch(E.ReplyComplete(gen_id=d.ctx.gen_id, assistant_text="because..."))
    assert d.state is State.LISTENING
    assert d.ctx.last_speech_at == 100.0
    # only real LISTENING silence ends the session: 24s is fine, 30s ends it.
    assert d.dispatch(E.Tick(now=124.0)) == []           # silence 24s < 25s nudge lead
    cmds = d.dispatch(E.Tick(now=130.0))                 # silence 30s -> end
    assert d.state is State.IDLE and cmds == [C.EndSession("silence_timeout")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_director_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.director'`

- [ ] **Step 3: Implement the Director wrapper**

```python
# modes/director/director.py
"""Director — thin stateful shell over the pure reducer. Holds (state, ctx) and
is the single mutator entry point: dispatch(event) runs reduce(), stores the new
state, returns the commands for the worker layer (Plan 02) to execute. Still no
I/O — workers feed it events and run its commands."""

from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import Context, new_context
from modes.director.reducer import reduce
from modes.talkback.conversation import ConversationManager


class Director:
    def __init__(self, cfg: DirectorConfig, conversation: ConversationManager,
                 now: float, proximity_rms: float):
        self.ctx: Context = new_context(cfg, conversation, now, proximity_rms)
        self.state: State = State.LISTENING

    def dispatch(self, event) -> list:
        self.state, commands = reduce(self.state, self.ctx, event)
        return commands
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_director_integration.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full director suite**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 6: Commit**

```bash
git add modes/director/director.py tests/director/test_director_integration.py
git commit -m "feat(director): Director wrapper (single mutator entry) + full synthetic-conversation test"
```

---

## Self-Review

- **Spec coverage (this plan = spec §13.1–2, the pure reducer):**
  - Req-3 backchannel-vs-question (§6): Task 7 (classify ladder, default-to-cut, empty/low-conf RESTORE). ✓
  - Req-4 timeout/nudge (§5): Task 3 (silence only in LISTENING, hard-cap-first, once-per-cycle nudge, re-arm on yield in Tasks 4/5/7 via `_enter_listening`/`_restore` paths). ✓
  - 5-state FSM with EVALUATING as a real state (§4): Tasks 1, 6, 7. ✓
  - gen_id staleness (§11): Task 5. ✓
  - Resume frame push on cut (§8, the Director-side part): Task 7 (`interrupted_stack` + `[interrupted]` partial). The *consuming* side (LLM-steered continuation, bounded cap, auto-resume) is Plan 06. ✓ (scoped)
  - Single-mutator brain (§3, §11): Task 8 `Director.dispatch`. ✓
  - NOT in this plan (correctly deferred): workers/audio/STT/LLM/TTS/event-bus (Plan 02), WakeGate + double-manage deletion (Plan 03), STT re-backing (Plan 04), pVAD/ECAPA focus (Plan 05) — `is_target`/`speaker_score` arrive as event fields here. The race-fixed teardown invariants (§10) are executed by the Playback worker in Plan 02; `Cut` is just a command here.
- **Placeholder scan:** no TBD/TODO; every code step has complete code. ✓
- **Type consistency:** `reduce(state, ctx, event) -> tuple[State, list]` used identically across Tasks 3–7; `_start_generation` (Task 4) reused by Task 7; `_enter_listening` (Task 4) reused by Task 5; `_restore_speaking` (Task 7); event/command field names match Task 2 definitions throughout (`gen_id`, `mean_word_prob`, `is_target`, `speaker_score`, `endpoint_prob`, `assistant_text`). `Cut(old_gen)` carries the pre-bump gen_id; `StartGeneration` the post-bump one. ✓
- **Ambiguity check:** time is only ever `Tick.now`; the integration test's silence assertion derives the deadline from `ctx.last_speech_at` (set at the `ReplyComplete` tick) rather than a hard-coded number, so it stays correct regardless of when the floor was last yielded. ✓
