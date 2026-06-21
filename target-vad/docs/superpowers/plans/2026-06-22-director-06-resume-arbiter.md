# Director Resume & Arbiter Polish (Plan 06) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the interruption-resume loop (spec §8) and the arbiter wiring (spec §6, §9) on top of the Plan-01 reducer and the Plan-02 worker layer. Plan 01 already pushes a resume frame onto `ctx.interrupted_stack` and one-shot-injects `ctx.pending_steer` on a CUT. This plan adds the **consume** side: the LLM-steered "As I was saying…" continuation on the next turn, a **bounded** interrupted-stack, an **auto-resume safety net** that recovers wrong cuts, a **second warm arbiter `LlmClient`** with the fixed close-then-ping lifecycle (main closed on cut, arbiter never closed mid-turn), and the **off-hot-path arbiter query/verdict** wiring. Everything decision-shaped is added to the pure reducer and tested with synthetic events; the two-client lifecycle is tested against fake clients.

**Architecture:** The reducer stays the single synchronous mutator (Plan 01). This plan extends it with: (1) a resume steer injected into the *next* `StartGeneration.steer` after a clarifier is answered and the floor returns to LISTENING; (2) a stack cap applied at push time; (3) a resume-timeout / no-followup event path that pops the resume frame and auto-continues the prior point; (4) an arbiter-verdict event that can flag a likely-wrong cut early. The two `LlmClient`s live in the Plan-02 worker layer; this plan defines and tests their lifecycle. The spec's hard rule — *the arbiter never sits on the duck-reaction path; it is consulted only AFTER the reflex has already made its safe call* — is preserved by making every arbiter interaction an asynchronous event the reducer consumes, never a synchronous call inside the EVALUATING ladder.

**Tech Stack:** Python 3.12 (`python3`, no `python` on PATH), pytest. Extends `modes/director/` (Plans 01-02). Reuses `modes/talkback/llm.py` (`LlmClient`) verbatim; ports the steer strings from `modes/talkback/controller.py` `_store_interruption` (:753-777) and `_maybe_inject_resume_steer` (:779-796). No new third-party dependencies.

## Global Constraints

- Target/dev box: NVIDIA DGX Spark GB10, aarch64, Python 3.12. Run tests with `python3 -m pytest`.
- **Single-mutator rule (Plan 01):** only `reduce()` mutates `State`/`Context`. No `await`, no I/O, no threads, no `time.monotonic()`, no `asyncio` in the reducer modules. Time arrives only via `Tick(now)`.
- **Binding contract — do not redefine:**
  - Plan 01 owns `reduce(state, ctx, event) -> tuple[State, list]`, `Context.interrupted_stack` / `pending_steer` / `current_query` / `partial_response`, `_start_generation` (injects `ctx.pending_steer` one-shot and clears it), the CUT path that finalizes the partial as `"… [interrupted]"` and pushes `{"query", "partial"}` onto `interrupted_stack` (Task 7).
  - Plan 02 owns `GenerationWorker(main_llm, ...)` and the worker layer that executes commands and emits events. This plan adds the **arbiter** `LlmClient` as the *second* warm client and specifies its lifecycle; Plan 02's worker constructor signature gains an `arbiter_llm` parameter (additive, documented in Task 6).
- **Arbiter is never on the hot path (spec §6/§9):** no arbiter call inside the EVALUATING reject ladder or the duck-at-onset path. The reflex makes the safe call first; the arbiter verdict arrives later as an `ArbiterVerdict` event.
- **Arbiter is never closed mid-turn (spec §9):** on CUT only `main_llm` is closed/cancelled; `arbiter_llm` stays warm. Both are close-then-pinged independently at session start.
- New code lives under `modes/director/`; tests under `tests/director/`.
- Reuse, do not reimplement: `LlmClient` (`modes/talkback/llm.py:13`), `classify_interjection`/`Interjection` (`modes/talkback/intent.py:57`/`:20`), `ConversationManager` (`modes/talkback/conversation.py:7`).

---

## File Structure

- Modify: `modes/director/config.py` — add `interrupted_stack_cap`, `resume_timeout_s`, `arbiter_enabled`.
- Modify: `modes/director/context.py` — add `resume_armed_at` (resume-window clock) field.
- Modify: `modes/director/events.py` — add `NoFollowup`, `ArbiterVerdict` events.
- Modify: `modes/director/commands.py` — add `QueryArbiter` command.
- Modify: `modes/director/reducer.py` — bounded push, resume-steer injection on yield, resume-timeout / no-followup auto-resume, arbiter verdict handling, optional `QueryArbiter` emission on a borderline cut.
- Create: `modes/director/llm_lifecycle.py` — `LlmPair` (main + arbiter) with `start()` (per-client close-then-ping) / `on_cut()` (main only) / `close()` (both). No reducer coupling; this is the worker-side lifecycle helper Plan 02 consumes.
- Tests: `tests/director/test_resume_steer.py`, `tests/director/test_stack_cap.py`, `tests/director/test_auto_resume.py`, `tests/director/test_arbiter.py`, `tests/director/test_llm_lifecycle.py`.

---

## Task 1: Config + Context extensions for resume & arbiter

**Files:**
- Modify: `modes/director/config.py`, `modes/director/context.py`
- Test: `tests/director/test_resume_config.py`

**Interfaces:**
- Produces: `DirectorConfig.interrupted_stack_cap` (int, default 3), `.resume_timeout_s` (float, default 8.0), `.arbiter_enabled` (bool, default True). `Context.resume_armed_at` (`Optional[float]`, default `None`) — the monotonic `now` at which the post-cut LISTENING resume window started, or `None` when no resume is armed.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_resume_config.py
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.talkback.conversation import ConversationManager


def test_resume_and_arbiter_config_defaults():
    c = DirectorConfig()
    assert c.interrupted_stack_cap == 3
    assert c.resume_timeout_s == 8.0
    assert c.arbiter_enabled is True
    # invariant: a resume window must be shorter than a full silence timeout
    assert 0.0 < c.resume_timeout_s < c.silence_timeout_s


def test_context_starts_with_no_resume_armed():
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                      now=5.0, proximity_rms=0.02)
    assert ctx.resume_armed_at is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_resume_config.py -v`
Expected: FAIL — `AttributeError: ... 'interrupted_stack_cap'` / `'resume_armed_at'`.

- [ ] **Step 3: Extend the config**

In `modes/director/config.py`, add these fields to `DirectorConfig` (after `duck_level`):

```python
    # Resume & arbiter (spec section 8 / section 6, Plan 06)
    interrupted_stack_cap: int = 3       # bound nested digressions; drop oldest on overflow
    resume_timeout_s: float = 8.0        # post-cut resume window (no genuine followup => auto-resume)
    arbiter_enabled: bool = True         # consult the small arbiter LLM off the hot path
```

- [ ] **Step 4: Extend the context**

In `modes/director/context.py`, add the field to `Context` (after `interrupted_stack`):

```python
    resume_armed_at: Optional[float] = None  # monotonic 'now' a post-cut resume window opened
```

`Optional` is already imported (Plan 01 uses it for `pending_steer`). `new_context()` needs no change — the field defaults to `None`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_resume_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add modes/director/config.py modes/director/context.py tests/director/test_resume_config.py
git commit -m "feat(director): resume/arbiter config + resume-window context field"
```

---

## Task 2: Bounded interrupted-stack (drop oldest on overflow)

**Files:**
- Modify: `modes/director/reducer.py`
- Test: `tests/director/test_stack_cap.py`

**Interfaces:**
- Modify: the CUT path in `_on_interjection_transcribed` (Plan 01 Task 7) pushes through a new helper `_push_interrupted(ctx, frame)` that appends then trims `ctx.interrupted_stack` to `ctx.cfg.interrupted_stack_cap`, dropping the **oldest** (front) on overflow. Behaviour for the common ≤cap case is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_stack_cap.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.talkback.conversation import ConversationManager


def _ctx(cap=3):
    import dataclasses
    cfg = dataclasses.replace(DirectorConfig(), interrupted_stack_cap=cap)
    ctx = new_context(cfg, ConversationManager(system_prompt="s"),
                      now=0.0, proximity_rms=0.02)
    ctx.gen_id = 1
    ctx.ducked = True
    return ctx


def _cut(ctx, query, partial):
    """Drive one INTERRUPT cut: set the live turn, then transcribe a question."""
    ctx.current_query = query
    ctx.partial_response = partial
    # land in EVALUATING with that live turn, then a genuine question cuts it
    reduce(State.EVALUATING, ctx,
           E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))


def test_stack_keeps_newest_and_drops_oldest_past_cap():
    ctx = _ctx(cap=2)
    _cut(ctx, "q1", "p1")
    _cut(ctx, "q2", "p2")
    _cut(ctx, "q3", "p3")
    # cap=2 => oldest (q1) dropped, order preserved newest-last
    assert [f["query"] for f in ctx.interrupted_stack] == ["q2", "q3"]
    assert len(ctx.interrupted_stack) == 2


def test_stack_within_cap_is_untouched():
    ctx = _ctx(cap=3)
    _cut(ctx, "q1", "p1")
    _cut(ctx, "q2", "p2")
    assert [f["query"] for f in ctx.interrupted_stack] == ["q1", "q2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_stack_cap.py -v`
Expected: FAIL — the unbounded `ctx.interrupted_stack.append(...)` from Plan 01 keeps `q1`, so `test_stack_keeps_newest_and_drops_oldest_past_cap` fails (length 3, oldest retained).

- [ ] **Step 3: Add the bounded-push helper and use it**

In `modes/director/reducer.py`, add the helper at the end of the module:

```python
def _push_interrupted(ctx: Context, frame: dict) -> None:
    """Push a resume frame, bounding the stack (drop oldest on overflow, spec s8)."""
    ctx.interrupted_stack.append(frame)
    overflow = len(ctx.interrupted_stack) - ctx.cfg.interrupted_stack_cap
    if overflow > 0:
        del ctx.interrupted_stack[:overflow]      # drop the oldest frames (front)
```

In `_on_interjection_transcribed` (Plan 01 Task 7), replace the direct append:

```python
    ctx.interrupted_stack.append({"query": ctx.current_query,
                                  "partial": ctx.partial_response})
```

with the bounded push:

```python
    _push_interrupted(ctx, {"query": ctx.current_query,
                            "partial": ctx.partial_response})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_stack_cap.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Verify Plan 01's EVALUATING test still passes**

Run: `python3 -m pytest tests/director/test_evaluating.py -v`
Expected: PASS — the single-cut case is unchanged (`interrupted_stack[-1]` still the pushed frame).

- [ ] **Step 6: Commit**

```bash
git add modes/director/reducer.py tests/director/test_stack_cap.py
git commit -m "feat(director): bound the interrupted-stack (drop oldest past cap)"
```

---

## Task 3: Port the cut-time steer + LLM-steered resume continuation on yield

**Files:**
- Modify: `modes/director/reducer.py`
- Test: `tests/director/test_resume_steer.py`

**Interfaces:**
- Port `_store_interruption`'s steer string (controller.py:769-774) into the CUT path: set `ctx.pending_steer` to the "answer their new question briefly, then ask if they would like you to continue" instruction (so the cut turn itself is steered). Plan 01's `_start_generation` already consumes-and-clears `ctx.pending_steer`; ordering matters — the CUT path must set `pending_steer` **before** calling `_start_generation`.
- Port `_maybe_inject_resume_steer`'s steer string (controller.py:789-793) into the **yield-to-LISTENING** path: when `ReplyComplete` returns the floor to LISTENING **and** `ctx.interrupted_stack` is non-empty, pop the newest resume frame, set `ctx.pending_steer` to the "Earlier you offered to continue explaining … resume that explanation naturally (e.g. 'As I was saying…')" instruction, and arm the resume window (`ctx.resume_armed_at = ctx.now`). The next `_start_generation` injects it one-shot and clears it.
- New helpers: `_cut_steer(query, partial) -> str`, `_resume_steer(query, partial) -> str`, `_arm_resume(ctx)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_resume_steer.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(now=0.0):
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                      now=now, proximity_rms=0.02)
    ctx.gen_id = 1
    ctx.ducked = True
    ctx.current_query = "tell me a story"
    ctx.partial_response = "once upon a time"
    return ctx


def test_cut_turn_carries_the_store_interruption_steer():
    ctx = _ctx()
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert state is State.THINKING
    start = cmds[1]                       # cmds == [Cut(...), StartGeneration(...)]
    assert isinstance(start, C.StartGeneration)
    # the cut turn is steered to answer-then-offer-to-continue (controller.py:769-774)
    assert 'tell me a story' in start.steer
    assert 'once upon a time' in start.steer
    assert 'continue' in start.steer.lower()
    assert ctx.pending_steer is None      # one-shot: consumed by _start_generation


def test_yield_after_cut_arms_resume_and_injects_continuation_steer():
    ctx = _ctx(now=10.0)
    # cut: pushes a resume frame, bumps gen to 2, lands in THINKING
    reduce(State.EVALUATING, ctx,
           E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert ctx.interrupted_stack[-1]["query"] == "tell me a story"
    cut_gen = ctx.gen_id
    reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=cut_gen))   # -> SPEAKING
    # clarifier answered -> yield floor to LISTENING; resume frame popped + armed
    ctx.now = 20.0
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.ReplyComplete(gen_id=cut_gen, assistant_text="because magic"))
    assert state is State.LISTENING and cmds == []
    assert ctx.interrupted_stack == []                 # frame consumed
    assert ctx.resume_armed_at == 20.0                 # resume window opened
    assert ctx.pending_steer is not None
    assert 'tell me a story' in ctx.pending_steer
    assert 'As I was saying' in ctx.pending_steer

    # next genuine user turn injects the resume steer one-shot, then clears it
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="yes please", mean_word_prob=0.9))
    assert state is State.THINKING
    assert 'As I was saying' in cmds[0].steer
    assert ctx.pending_steer is None
    assert ctx.resume_armed_at is None                 # window closed by real followup


def test_yield_with_empty_stack_does_not_arm_resume():
    ctx = _ctx(now=5.0)
    ctx.interrupted_stack = []
    state, cmds = reduce(State.SPEAKING, ctx,
                         E.ReplyComplete(gen_id=1, assistant_text="done"))
    assert state is State.LISTENING
    assert ctx.resume_armed_at is None and ctx.pending_steer is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_resume_steer.py -v`
Expected: FAIL — no steer set on cut; `ReplyComplete` does not arm resume or inject the continuation steer.

- [ ] **Step 3: Add the steer helpers (ported verbatim)**

At the end of `modes/director/reducer.py`, add:

```python
def _cut_steer(query: str, partial: str) -> str:
    """Steer for the interrupting turn (ported from controller.py:769-774)."""
    return (
        f'You were answering the user\'s request: "{query}". So far you had '
        f'said: "{partial}". The user interrupted with a new question. Answer '
        f'their new question briefly, then ask if they would like you to '
        f'continue with the earlier topic.'
    )


def _resume_steer(query: str, partial: str) -> str:
    """Steer for the continuation turn (ported from controller.py:789-793)."""
    return (
        f'Earlier you offered to continue explaining "{query}" (you had '
        f'said: "{partial}"). Interpret the user\'s latest reply: if they '
        f'want you to continue, resume that explanation naturally (e.g. "As '
        f'I was saying...") and finish it; otherwise just respond normally.'
    )


def _arm_resume(ctx: Context) -> None:
    """On yielding the floor with a pending interruption, pop the newest resume
    frame and queue its continuation steer one-shot; open the resume window."""
    frame = ctx.interrupted_stack.pop()
    ctx.pending_steer = _resume_steer(frame["query"], frame["partial"])
    ctx.resume_armed_at = ctx.now
```

- [ ] **Step 4: Set the cut steer on the CUT path**

In `_on_interjection_transcribed` (Plan 01 Task 7), the INTERRUPT branch reads (after Task 2's bounded push):

```python
    old_gen = ctx.gen_id
    if ctx.partial_response:
        ctx.conversation.add_assistant_turn(ctx.partial_response + " [interrupted]")
    _push_interrupted(ctx, {"query": ctx.current_query,
                            "partial": ctx.partial_response})
    ctx.ducked = False
    state, cmds = _start_generation(ctx, ev.text)
    return state, [C.Cut(old_gen)] + cmds
```

Set the cut steer **before** `_start_generation` so it is the one-shot it consumes. Insert immediately above `state, cmds = _start_generation(ctx, ev.text)`:

```python
    ctx.pending_steer = _cut_steer(ctx.current_query, ctx.partial_response)
```

(Plan 01's `_start_generation` reads `ctx.pending_steer`, passes it into `StartGeneration.steer`, and clears it.)

- [ ] **Step 5: Arm resume on the yield-to-LISTENING path**

In `reduce()`'s `ReplyComplete` branch (Plan 01 Task 5), after the floor is yielded, arm resume if a frame is waiting. The branch becomes:

```python
    if isinstance(event, E.ReplyComplete):
        if state in (State.THINKING, State.SPEAKING) and event.gen_id == ctx.gen_id:
            if event.assistant_text:
                ctx.conversation.add_assistant_turn(event.assistant_text)
            _enter_listening(ctx)
            if ctx.interrupted_stack:
                _arm_resume(ctx)          # pop newest frame, queue "As I was saying..."
            return State.LISTENING, []
        return state, []
```

- [ ] **Step 6: Clear the resume window when a genuine followup starts a turn**

`_arm_resume` set `ctx.resume_armed_at`; once a real user turn starts generating, the window is closed. In `_start_generation` (Plan 01 Task 4), add a single line right after `ctx.pending_steer = None` (the one-shot clear):

```python
    ctx.resume_armed_at = None            # a real generation closes any resume window
```

This makes both `pending_steer` and `resume_armed_at` consistent: arming sets them, the next generation clears them.

- [ ] **Step 7: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_resume_steer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Verify prior reducer tests still pass**

Run: `python3 -m pytest tests/director/test_evaluating.py tests/director/test_listening_turn.py tests/director/test_speaking_lifecycle.py -v`
Expected: PASS — Plan 01's cut test now additionally has a steer set (it only asserted commands/history/stack, not `steer`, so it remains green); `_start_generation`'s extra `resume_armed_at = None` is a no-op when unset.

- [ ] **Step 9: Commit**

```bash
git add modes/director/reducer.py tests/director/test_resume_steer.py
git commit -m "feat(director): port cut/resume steer strings; LLM-steered continuation on yield"
```

---

## Task 4: Auto-resume safety net (recover a wrong cut)

**Files:**
- Modify: `modes/director/events.py`, `modes/director/reducer.py`
- Test: `tests/director/test_auto_resume.py`

**Interfaces:**
- New event `NoFollowup(gen_id)` — emitted by the worker layer (Plan 02) when a post-cut LISTENING turn yields no genuine user content (empty / `mean_word_prob < conf_floor` / backchannel), tagged with the resume generation it belongs to.
- Reducer mechanism (two redundant triggers, both pop the newest resume frame and auto-continue the prior point):
  1. **Resume-timeout Tick path:** while in LISTENING with `ctx.resume_armed_at` set, if `ctx.now - ctx.resume_armed_at >= ctx.cfg.resume_timeout_s` (and we are not already past silence/hard timeout), auto-resume: start a generation that continues the prior point. This fires *before* the silence nudge/timeout so a wrong cut self-heals quickly.
  2. **Explicit `NoFollowup` event:** when a backchannel/empty/low-conf turn is detected in LISTENING during an armed resume window, auto-resume immediately (don't wait for the timeout).
- Auto-resume reuses the already-queued `ctx.pending_steer` (the `_resume_steer` set in Task 3) by starting a generation with an empty user query that carries only the steer. New helper `_auto_resume(ctx)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_auto_resume.py
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _armed_ctx(now=20.0):
    """A context already in the post-cut resume window (as Task 3 leaves it)."""
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                      now=now, proximity_rms=0.02)
    ctx.gen_id = 2
    ctx.last_speech_at = now
    ctx.resume_armed_at = now
    ctx.pending_steer = "Earlier you offered to continue explaining \"x\" ... As I was saying..."
    return ctx


def test_resume_timeout_auto_continues_the_prior_point():
    ctx = _armed_ctx(now=20.0)
    # resume_timeout_s = 8 => fires at now >= 28, before silence nudge (lead at 25 sil)
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=28.0))
    assert state is State.THINKING
    assert isinstance(cmds[0], C.StartGeneration)
    assert 'As I was saying' in cmds[0].steer       # the queued resume steer
    assert ctx.gen_id == 3                            # a new generation
    assert ctx.resume_armed_at is None                # window closed
    assert ctx.pending_steer is None                  # one-shot consumed


def test_resume_window_does_not_fire_before_timeout():
    ctx = _armed_ctx(now=20.0)
    state, cmds = reduce(State.LISTENING, ctx, E.Tick(now=25.0))   # only 5s into window
    # not auto-resume yet; this tick is < resume_timeout but >= nudge lead?
    # silence here is now-last_speech_at = 5s, well below nudge lead, so no nudge either
    assert state is State.LISTENING and cmds == []
    assert ctx.resume_armed_at == 20.0


def test_no_followup_event_auto_resumes_immediately():
    ctx = _armed_ctx(now=21.0)
    state, cmds = reduce(State.LISTENING, ctx, E.NoFollowup(gen_id=2))
    assert state is State.THINKING
    assert isinstance(cmds[0], C.StartGeneration)
    assert 'As I was saying' in cmds[0].steer
    assert ctx.resume_armed_at is None


def test_no_followup_ignored_when_no_resume_armed():
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="s"),
                      now=5.0, proximity_rms=0.02)
    state, cmds = reduce(State.LISTENING, ctx, E.NoFollowup(gen_id=1))
    assert state is State.LISTENING and cmds == []


def test_genuine_followup_closes_window_before_timeout():
    ctx = _armed_ctx(now=20.0)
    # a real user turn arrives within the window
    state, cmds = reduce(State.LISTENING, ctx,
                         E.UserTurnTranscribed(text="actually no, never mind that",
                                               mean_word_prob=0.9))
    assert state is State.THINKING
    assert ctx.resume_armed_at is None                # closed by the real generation (Task 3 step 6)
    # the pending resume steer rode along on this turn (one-shot); fine — it's a
    # benign instruction and the user query dominates.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_auto_resume.py -v`
Expected: FAIL — `NoFollowup` does not exist; the resume-timeout Tick path is not implemented.

- [ ] **Step 3: Add the `NoFollowup` event**

In `modes/director/events.py`, add:

```python
@dataclass(frozen=True)
class NoFollowup:
    """A post-cut LISTENING turn yielded no genuine user content (empty /
    low-confidence / backchannel). Triggers auto-resume of the prior point."""
    gen_id: int
```

- [ ] **Step 4: Add the auto-resume helper**

At the end of `modes/director/reducer.py`, add:

```python
def _auto_resume(ctx: Context) -> tuple[State, list]:
    """Recover a wrong cut: continue the prior point with no new user query.

    The resume steer was already queued in ctx.pending_steer when the floor was
    yielded (Task 3 _arm_resume). _start_generation injects+clears it and also
    clears resume_armed_at (Task 3 step 6). We pass an empty query so no spurious
    user turn is added to history; the steer alone drives the continuation."""
    return _start_generation(ctx, "")
```

`_start_generation` (Plan 01) calls `ctx.conversation.add_user_turn(query)`. An empty query must NOT pollute history. Guard it: in `_start_generation`, change the unconditional add to skip empties:

```python
def _start_generation(ctx: Context, query: str) -> tuple[State, list]:
    if query:
        ctx.conversation.add_user_turn(query)     # auto-resume passes "" (steer-only)
    ctx.current_query = query
    ctx.partial_response = ""
    ctx.gen_id += 1
    steer = ctx.pending_steer
    ctx.pending_steer = None                       # one-shot
    ctx.resume_armed_at = None                     # a real generation closes any resume window
    return State.THINKING, [C.StartGeneration(ctx.gen_id,
                                              ctx.conversation.get_messages(), steer)]
```

(The empty-query guard is safe for normal turns: `_on_user_transcribed` already rejects empty text before calling `_start_generation`, so only auto-resume reaches it with `""`.)

- [ ] **Step 5: Wire the resume-timeout into the Tick path**

In `_on_tick` (Plan 01 Task 3), insert the resume-timeout check **after** the hard-cap and silence-timeout checks (those terminal cases win) but **before** the nudge, so a wrong cut self-heals rather than nudging:

```python
def _on_tick(state: State, ctx: Context, ev: E.Tick) -> tuple[State, list]:
    ctx.now = ev.now
    if ctx.now - ctx.started_at >= ctx.cfg.hard_timeout_s:
        return State.IDLE, [C.EndSession("hard_timeout")]
    sil = silence_duration(state, ctx)
    if sil >= ctx.cfg.silence_timeout_s:
        return State.IDLE, [C.EndSession("silence_timeout")]
    # Auto-resume a wrong cut before nudging (spec s8 safety net).
    if (state is State.LISTENING and ctx.resume_armed_at is not None
            and ctx.now - ctx.resume_armed_at >= ctx.cfg.resume_timeout_s):
        return _auto_resume(ctx)
    if sil >= (ctx.cfg.silence_timeout_s - ctx.cfg.nudge_lead_s) and not ctx.nudged_cycle:
        ctx.nudged_cycle = True
        return state, [C.SpeakNudge()]
    return state, []
```

- [ ] **Step 6: Wire the explicit `NoFollowup` event**

In `reduce()`, add before the final `return state, []`:

```python
    if isinstance(event, E.NoFollowup):
        if state is State.LISTENING and ctx.resume_armed_at is not None:
            return _auto_resume(ctx)
        return state, []
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_auto_resume.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Verify prior reducer + steer tests still pass**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS — the `_start_generation` empty-guard and `resume_armed_at = None` are no-ops for all Plan 01 paths.

- [ ] **Step 9: Commit**

```bash
git add modes/director/events.py modes/director/reducer.py tests/director/test_auto_resume.py
git commit -m "feat(director): auto-resume safety net (resume-timeout Tick + NoFollowup event)"
```

---

## Task 5: Arbiter query/verdict wiring (off the hot path)

**Files:**
- Modify: `modes/director/events.py`, `modes/director/commands.py`, `modes/director/reducer.py`
- Test: `tests/director/test_arbiter.py`

**Interfaces:**
- New command `QueryArbiter(gen_id, text, decision)` — emitted *after* a CUT (alongside `Cut` + `StartGeneration`) when `ctx.cfg.arbiter_enabled` and the reflex's call was borderline (default-to-cut on non-lexical content, i.e. `classify_interjection` returned INTERRUPT but the text was not a force token). The worker runs the small arbiter `LlmClient` asynchronously and emits an `ArbiterVerdict`. This is strictly *after* the safe reflex cut — never inside the EVALUATING ladder.
- New event `ArbiterVerdict(gen_id, likely_wrong_cut)` — the arbiter's late opinion on the just-made cut, tagged with the cut's resume gen_id.
- Reducer verdict handling: an `ArbiterVerdict(likely_wrong_cut=True)` for a turn that is still the live cut turn **shortens the resume window to fire now** by treating it like a `NoFollowup` *iff* the floor has already returned to LISTENING with a resume armed; otherwise it sets `ctx.resume_armed_at` back to the past so the next Tick auto-resumes immediately (catch-wrong-cut-early per spec §6/§8). A `likely_wrong_cut=False` verdict is a confirmation and is a no-op (the cut stands).
- Borderline detection helper `_is_borderline_cut(text) -> bool`: True when `classify_interjection(text)` is INTERRUPT **and** no token is in `FORCE_INTERRUPT` (i.e. the default-to-cut branch fired, the genuinely ambiguous case the spec calls out).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_arbiter.py
import dataclasses

from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director.reducer import reduce
from modes.director import events as E
from modes.director import commands as C
from modes.talkback.conversation import ConversationManager


def _ctx(arbiter=True):
    cfg = dataclasses.replace(DirectorConfig(), arbiter_enabled=arbiter)
    ctx = new_context(cfg, ConversationManager(system_prompt="s"),
                      now=0.0, proximity_rms=0.02)
    ctx.gen_id = 1
    ctx.ducked = True
    ctx.current_query = "tell me a story"
    ctx.partial_response = "once upon a time"
    return ctx


def test_borderline_cut_also_queries_arbiter():
    ctx = _ctx()
    # "lets move on" is INTERRUPT via default-to-cut (no force token) => borderline
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="lets move on", mean_word_prob=0.9))
    assert state is State.THINKING
    kinds = [type(c).__name__ for c in cmds]
    assert kinds[0] == "Cut" and kinds[1] == "StartGeneration"  # reflex acts FIRST
    assert "QueryArbiter" in kinds                              # arbiter consulted AFTER
    q = next(c for c in cmds if isinstance(c, C.QueryArbiter))
    assert q.gen_id == ctx.gen_id and q.text == "lets move on"


def test_lexical_question_cut_does_not_query_arbiter():
    ctx = _ctx()
    # "wait why" has force tokens => unambiguous cut => no arbiter
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert not any(isinstance(c, C.QueryArbiter) for c in cmds)


def test_arbiter_disabled_never_queries():
    ctx = _ctx(arbiter=False)
    state, cmds = reduce(State.EVALUATING, ctx,
                         E.InterjectionTranscribed(text="lets move on", mean_word_prob=0.9))
    assert not any(isinstance(c, C.QueryArbiter) for c in cmds)


def test_wrong_cut_verdict_triggers_auto_resume_in_listening():
    ctx = _ctx()
    # drive a borderline cut, answer it, yield -> resume armed (Task 3)
    reduce(State.EVALUATING, ctx,
           E.InterjectionTranscribed(text="lets move on", mean_word_prob=0.9))
    cut_gen = ctx.gen_id
    reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=cut_gen))
    ctx.now = 12.0
    reduce(State.SPEAKING, ctx, E.ReplyComplete(gen_id=cut_gen, assistant_text="ok"))
    assert ctx.resume_armed_at == 12.0
    # arbiter (late) says the cut was likely wrong -> auto-resume now
    state, cmds = reduce(State.LISTENING, ctx,
                         E.ArbiterVerdict(gen_id=cut_gen, likely_wrong_cut=True))
    assert state is State.THINKING
    assert isinstance(cmds[0], C.StartGeneration)
    assert 'As I was saying' in cmds[0].steer


def test_confirming_verdict_is_a_noop():
    ctx = _ctx()
    reduce(State.EVALUATING, ctx,
           E.InterjectionTranscribed(text="lets move on", mean_word_prob=0.9))
    cut_gen = ctx.gen_id
    reduce(State.THINKING, ctx, E.FirstTtsFrame(gen_id=cut_gen))
    ctx.now = 12.0
    reduce(State.SPEAKING, ctx, E.ReplyComplete(gen_id=cut_gen, assistant_text="ok"))
    state, cmds = reduce(State.LISTENING, ctx,
                         E.ArbiterVerdict(gen_id=cut_gen, likely_wrong_cut=False))
    assert state is State.LISTENING and cmds == []
    assert ctx.resume_armed_at == 12.0      # window untouched; cut stands
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_arbiter.py -v`
Expected: FAIL — `QueryArbiter` / `ArbiterVerdict` do not exist; no borderline emission or verdict handling.

- [ ] **Step 3: Add the command and event**

In `modes/director/commands.py`, add:

```python
@dataclass(frozen=True)
class QueryArbiter:
    """Consult the small arbiter LLM AFTER a borderline reflex cut (spec s6/s9).
    Off the hot path: the reflex has already acted; this only flags a wrong cut."""
    gen_id: int
    text: str
    decision: str        # the reflex verdict, e.g. "CUT" (context for the arbiter)
```

In `modes/director/events.py`, add:

```python
@dataclass(frozen=True)
class ArbiterVerdict:
    """The arbiter's late opinion on a just-made cut. likely_wrong_cut=True feeds
    the auto-resume net (recover the cut early); False confirms the cut stands."""
    gen_id: int
    likely_wrong_cut: bool
```

- [ ] **Step 4: Emit `QueryArbiter` on a borderline cut**

At the end of `modes/director/reducer.py`, add the borderline helper (it needs `FORCE_INTERRUPT`):

```python
from modes.talkback.intent import FORCE_INTERRUPT      # add to the existing intent import


def _is_borderline_cut(text: str) -> bool:
    """A default-to-cut: INTERRUPT with no force token — the ambiguous case the
    arbiter exists to second-guess (spec s6). Lexical force-token cuts are sure."""
    if classify_interjection(text) is not Interjection.INTERRUPT:
        return False
    from modes.talkback.intent import _tokenize
    return not any(tok in FORCE_INTERRUPT for tok in _tokenize(text))
```

> Implementer note: prefer adding `FORCE_INTERRUPT` and `_tokenize` to the top-level import line `from modes.talkback.intent import Interjection, classify_interjection` (Plan 01 Task 7) → `from modes.talkback.intent import Interjection, classify_interjection, FORCE_INTERRUPT, _tokenize`, and drop the inline `from ... import _tokenize`. Both are module-level symbols in `intent.py`.

In `_on_interjection_transcribed`'s INTERRUPT branch, append the arbiter query when enabled and borderline. The branch's final return becomes:

```python
    ctx.pending_steer = _cut_steer(ctx.current_query, ctx.partial_response)
    cut_text = ev.text
    state, cmds = _start_generation(ctx, ev.text)
    out = [C.Cut(old_gen)] + cmds                      # reflex acts first
    if ctx.cfg.arbiter_enabled and _is_borderline_cut(cut_text):
        out.append(C.QueryArbiter(ctx.gen_id, cut_text, "CUT"))  # consult AFTER
    return state, out
```

(`ctx.gen_id` here is the post-cut resume generation — the verdict will be tagged to it.)

- [ ] **Step 5: Handle `ArbiterVerdict`**

In `reduce()`, add before the final `return state, []`:

```python
    if isinstance(event, E.ArbiterVerdict):
        if (event.likely_wrong_cut and state is State.LISTENING
                and ctx.resume_armed_at is not None):
            return _auto_resume(ctx)        # recover the wrong cut early
        return state, []                    # confirmation (or stale): cut stands
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_arbiter.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Verify the full reducer suite stays green**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS — `test_evaluating.py`'s `test_question_cuts_and_starts_new_turn` uses `"wait why is that"` (force token), so no `QueryArbiter` is appended and its `cmds[0]`/`cmds[1]` assertions are unchanged.

- [ ] **Step 8: Commit**

```bash
git add modes/director/events.py modes/director/commands.py modes/director/reducer.py tests/director/test_arbiter.py
git commit -m "feat(director): off-hot-path arbiter query on borderline cut + wrong-cut verdict -> auto-resume"
```

---

## Task 6: Two warm LLM clients with the fixed lifecycle

**Files:**
- Create: `modes/director/llm_lifecycle.py`
- Test: `tests/director/test_llm_lifecycle.py`

**Interfaces:**
- `LlmPair(main: LlmClient, arbiter: LlmClient)` — the worker-side holder for the two warm clients (spec §9). Methods:
  - `async start() -> bool` — per-client **close-then-ping** at session start (mirrors `controller.py:264` `await self._llm.close()` then `ping()`, done **independently per client**). Returns True only if **both** ping True. Closing first drops any stale session from a prior run.
  - `on_cut()` — invoked on a CUT: cancels **only** the main client (`main.cancel()`); the arbiter is **never** touched (spec §9 — it has no in-flight generation and is needed for the next ambiguous call). Synchronous (just sets the main cancel flag).
  - `async close()` — graceful teardown: close **both** clients (session end only).
- This module is pure lifecycle plumbing; it does not import the reducer. Plan 02's worker constructor gains `arbiter_llm: LlmClient` and holds an `LlmPair(main_llm, arbiter_llm)`; on a `Cut` command it calls `pair.on_cut()`; at session start it calls `await pair.start()` as the ping gate; the existing single-client `await self._llm.close()` (controller.py:264) is replaced by `await pair.start()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_llm_lifecycle.py
import asyncio

from modes.director.llm_lifecycle import LlmPair


class FakeClient:
    def __init__(self, ping_ok=True):
        self.closed = 0
        self.pinged = 0
        self.cancelled = 0
        self._ping_ok = ping_ok

    async def close(self):
        self.closed += 1

    async def ping(self):
        self.pinged += 1
        return self._ping_ok

    def cancel(self):
        self.cancelled += 1


def test_start_closes_then_pings_each_client_independently():
    main, arb = FakeClient(), FakeClient()
    pair = LlmPair(main, arb)
    ok = asyncio.run(pair.start())
    assert ok is True
    # close-then-ping on BOTH (spec s9): each closed once, each pinged once
    assert main.closed == 1 and main.pinged == 1
    assert arb.closed == 1 and arb.pinged == 1


def test_start_fails_if_either_client_does_not_ping():
    main, arb = FakeClient(ping_ok=True), FakeClient(ping_ok=False)
    pair = LlmPair(main, arb)
    assert asyncio.run(pair.start()) is False


def test_cut_cancels_only_main_never_arbiter():
    main, arb = FakeClient(), FakeClient()
    pair = LlmPair(main, arb)
    pair.on_cut()
    assert main.cancelled == 1
    assert arb.cancelled == 0          # arbiter NEVER cancelled mid-turn (spec s9)
    assert arb.closed == 0


def test_close_closes_both_clients():
    main, arb = FakeClient(), FakeClient()
    pair = LlmPair(main, arb)
    asyncio.run(pair.close())
    assert main.closed == 1 and arb.closed == 1


def test_repeated_cuts_never_touch_arbiter():
    main, arb = FakeClient(), FakeClient()
    pair = LlmPair(main, arb)
    for _ in range(5):
        pair.on_cut()
    assert main.cancelled == 5
    assert arb.cancelled == 0 and arb.closed == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_llm_lifecycle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.llm_lifecycle'`.

- [ ] **Step 3: Implement `LlmPair`**

```python
# modes/director/llm_lifecycle.py
"""Two warm LLM clients with the fixed lifecycle (spec section 9).

main = gemma (the answering model); arbiter = a small model consulted off the
hot path. Both are warm llama.cpp servers behind LlmClient (modes/talkback/llm.py).

Lifecycle rules (spec section 9), enforced here so Plan 02's worker just delegates:
  - At session start: close-then-ping EACH client independently (mirrors the
    single-client controller.py:264 'await self._llm.close()' before ping()).
    Closing first drops any stale aiohttp session from a prior run.
  - On a CUT: cancel ONLY the main client. The arbiter is NEVER closed or
    cancelled mid-turn -- it has no in-flight generation and is needed for the
    next ambiguous call.
  - At session end only: close BOTH.
"""

from modes.talkback.llm import LlmClient


class LlmPair:
    def __init__(self, main: LlmClient, arbiter: LlmClient):
        self.main = main
        self.arbiter = arbiter

    async def start(self) -> bool:
        """Per-client close-then-ping. True iff BOTH clients ping healthy."""
        await self.main.close()
        main_ok = await self.main.ping()
        await self.arbiter.close()
        arbiter_ok = await self.arbiter.ping()
        return main_ok and arbiter_ok

    def on_cut(self) -> None:
        """Cancel ONLY the main generation; leave the arbiter warm (spec s9)."""
        self.main.cancel()

    async def close(self) -> None:
        """Session-end teardown: close both clients."""
        await self.main.close()
        await self.arbiter.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_llm_lifecycle.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/llm_lifecycle.py tests/director/test_llm_lifecycle.py
git commit -m "feat(director): two warm LLM clients — per-client close-then-ping; cut cancels main only"
```

---

## Task 7: Full resume + arbiter integration test

**Files:**
- Test: `tests/director/test_resume_integration.py`

**Interfaces:**
- Consumes only public reducer/`Director`/`LlmPair` surface. No new production code — this task is a green-suite integration proof that the resume loop and arbiter wiring compose end-to-end through `Director.dispatch`.

- [ ] **Step 1: Write the integration test**

```python
# tests/director/test_resume_integration.py
import asyncio

from modes.director.director import Director
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director import events as E
from modes.director import commands as C
from modes.director.llm_lifecycle import LlmPair
from modes.talkback.conversation import ConversationManager


def _director():
    return Director(DirectorConfig(), ConversationManager(system_prompt="s"),
                    now=0.0, proximity_rms=0.02)


def test_interrupt_answer_then_resume_continuation():
    d = _director()
    # 1. user asks for a story; it starts speaking
    d.dispatch(E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
    gen = d.ctx.gen_id
    d.dispatch(E.FirstTtsFrame(gen_id=gen)); assert d.state is State.SPEAKING
    d.ctx.partial_response = "once upon a time"
    # 2. genuine question cuts mid-story
    d.dispatch(E.NearFieldOnset(rms=0.5, is_target=True)); assert d.state is State.EVALUATING
    d.dispatch(E.InterjectionSegment(900.0, 0.5, True, 0.9))
    cmds = d.dispatch(E.InterjectionTranscribed(text="wait why", mean_word_prob=0.9))
    assert d.state is State.THINKING
    assert isinstance(cmds[0], C.Cut) and isinstance(cmds[1], C.StartGeneration)
    assert d.ctx.interrupted_stack[-1]["query"] == "tell me a story"
    cut_gen = d.ctx.gen_id
    # 3. clarifier answered; yield floor -> resume armed + continuation steer queued
    d.dispatch(E.FirstTtsFrame(gen_id=cut_gen))
    d.dispatch(E.Tick(now=5.0))
    d.dispatch(E.ReplyComplete(gen_id=cut_gen, assistant_text="because magic"))
    assert d.state is State.LISTENING
    assert d.ctx.interrupted_stack == []
    assert d.ctx.resume_armed_at == 5.0
    # 4. user says "yes please" -> the continuation steer rides the next turn
    cmds = d.dispatch(E.UserTurnTranscribed(text="yes please", mean_word_prob=0.9))
    assert d.state is State.THINKING
    assert 'As I was saying' in cmds[0].steer
    assert d.ctx.pending_steer is None and d.ctx.resume_armed_at is None


def test_wrong_cut_auto_resumes_on_silence_window():
    d = _director()
    d.dispatch(E.UserTurnTranscribed(text="explain photosynthesis", mean_word_prob=0.9))
    gen = d.ctx.gen_id
    d.dispatch(E.FirstTtsFrame(gen_id=gen))
    d.ctx.partial_response = "plants make energy"
    # a borderline interjection (default-to-cut) wrongly cuts
    d.dispatch(E.NearFieldOnset(rms=0.5, is_target=True))
    d.dispatch(E.InterjectionSegment(900.0, 0.5, True, 0.9))
    cmds = d.dispatch(E.InterjectionTranscribed(text="lets move on", mean_word_prob=0.9))
    assert any(isinstance(c, C.QueryArbiter) for c in cmds)   # arbiter consulted
    cut_gen = d.ctx.gen_id
    d.dispatch(E.FirstTtsFrame(gen_id=cut_gen))
    d.dispatch(E.ReplyComplete(gen_id=cut_gen, assistant_text="okay"))
    assert d.state is State.LISTENING and d.ctx.resume_armed_at == 0.0
    # no genuine followup; resume_timeout_s=8 => auto-resume at now>=8
    d.dispatch(E.Tick(now=4.0)); assert d.state is State.LISTENING
    cmds = d.dispatch(E.Tick(now=8.0))
    assert d.state is State.THINKING
    assert 'As I was saying' in cmds[0].steer


def test_llm_pair_lifecycle_through_a_cut():
    class FakeClient:
        def __init__(self): self.closed = self.pinged = self.cancelled = 0
        async def close(self): self.closed += 1
        async def ping(self): self.pinged += 1; return True
        def cancel(self): self.cancelled += 1

    main, arb = FakeClient(), FakeClient()
    pair = LlmPair(main, arb)
    assert asyncio.run(pair.start()) is True
    pair.on_cut()                                 # simulate the Cut command
    assert main.cancelled == 1 and arb.cancelled == 0   # arbiter stays warm
    asyncio.run(pair.close())
    assert main.closed == 2 and arb.closed == 2          # start-close + end-close
```

- [ ] **Step 2: Run the integration test**

Run: `python3 -m pytest tests/director/test_resume_integration.py -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Run the full director suite**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS (all Plan 01 + Plan 06 tests green)

- [ ] **Step 4: Commit**

```bash
git add tests/director/test_resume_integration.py
git commit -m "test(director): end-to-end resume + arbiter + LLM-pair lifecycle integration"
```

---

## Self-Review

- **Spec coverage (this plan = spec §8 interruption-resume + §6/§9 arbiter & two-client lifecycle):**
  - Bounded interrupted-stack, drop-oldest on overflow (§8): Task 2 (`_push_interrupted`, cap default 3, TDD'd at cap=2). ✓
  - LLM-steered continuation, one-shot, cleared after injection (§8): Task 3 ports both steer strings verbatim from `controller.py` (`_store_interruption` :769-774 cut steer; `_maybe_inject_resume_steer` :789-793 resume steer); arms on yield-to-LISTENING; the next `StartGeneration.steer` carries "As I was saying…" then `pending_steer`/`resume_armed_at` clear. ✓
  - Auto-resume safety net for a wrong cut (§8): Task 4 — *two* triggers, the resume-timeout Tick path (fires before the nudge) and the explicit `NoFollowup` event; both `_auto_resume` (steer-only, empty query, no history pollution). ✓
  - Resume driven from the same EVALUATING→CUT path (§8): Plan 01 pushes on cut; Tasks 3-4 add the consume side on the *next* turn — one stack, one path, no separate "resume mode." ✓
  - Two warm clients, per-client close-then-ping at start, main-only close/cancel on cut, arbiter never closed mid-turn (§9): Task 6 `LlmPair` (`start`/`on_cut`/`close`), TDD'd against fake clients (assert `arbiter.cancelled==0`/`closed==0` on cut; both pinged at start). ✓
  - Arbiter off the hot path, consulted only AFTER the reflex's safe call (§6/§9): Task 5 — `QueryArbiter` appended *after* `Cut`+`StartGeneration`, only on a borderline (default-to-cut, no force token) cut; the verdict (`ArbiterVerdict`) feeds auto-resume (`likely_wrong_cut=True`) or is a no-op confirmation. No arbiter call anywhere in the EVALUATING ladder or duck path. ✓
- **Binding-contract fidelity:** extends Plan 01's `reduce`/`Context`/`_start_generation`/CUT-push and Plan 02's `GenerationWorker(main_llm, …)` (adds `arbiter_llm` additively via `LlmPair`); does not redefine any of them. The only Plan-01 edits are *additive* (`_start_generation` gains an empty-query guard + `resume_armed_at=None`; the CUT path sets `pending_steer` and may append `QueryArbiter`; `ReplyComplete` may `_arm_resume`; `_on_tick` gains the resume-timeout branch) — each guarded so all Plan 01 tests stay green (Task 3 step 8, Task 4 step 8, Task 5 step 7). ✓
- **Single-mutator / purity:** all reducer additions are synchronous, clock-free (time only via `Tick.now`), no I/O. `LlmPair` is the *only* async code and lives outside the reducer (worker-side), tested with `asyncio.run` + fakes. ✓
- **Ordering correctness:** cut steer is set *before* `_start_generation` consumes it; resume is armed *after* `_enter_listening` yields the floor; the resume-timeout check sits *after* terminal timeouts but *before* the nudge so a wrong cut self-heals rather than nudging; `resume_armed_at` is cleared by any real generation so a genuine followup closes the window. ✓
- **Placeholder scan:** no TBD/TODO/FIXME; every step has complete, runnable code. ✓
- **Type consistency:** `reduce(state, ctx, event) -> tuple[State, list]` unchanged; new events `NoFollowup(gen_id)`, `ArbiterVerdict(gen_id, likely_wrong_cut)`; new command `QueryArbiter(gen_id, text, decision)`; field names (`gen_id`, `mean_word_prob`, `steer`, `likely_wrong_cut`) match across tasks and Plan 01/02. ✓
