# Director-09 Safety Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the built-but-dormant ECAPA safety net into the live Director: accumulated-window hijack detection (pure-reducer WARN→EJECT ladder), real verify-before-serve (split-half + window-1), post-eject WakeGate quiet-hold, and a full config truth pass.

**Architecture:** Worker computes, reducer decides. `IngestionWorker` stages segment audio; the reducer emits `AccumulateSpeakerAudio` for served/plausibly-owner speech only; a new `SafetyNetWorker` buffers it, embeds each 2s window off the event loop, and emits `SpeakerWindowVerdict`; the pure reducer applies the ladder (window-1 fail = bad enrollment; 3-miss WARN = log-only; WARN + below-proximity = EJECT, silent). The WakeGate gains a pre-serve split-half verify and a post-`speaker_mismatch` quiet-hold. `lockout.py` is deleted after both halves are ported.

**Tech Stack:** Python 3.12, pytest, asyncio, numpy, existing `SafetyNet`/`DecisionSmoother`/ECAPA embedder. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-02-director-09-safety-net-design.md` (read it first).

## Global Constraints

- Working directory: `/home/ldrgx10/FullDuplexVoice/TVAD/target-vad`. Branch: `feat/director-09-safety-net` (stacked on Director-08).
- **Every git commit message MUST end with the trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- The reducer stays PURE: no I/O, no `await`, no clock reads (`now` only via events). Never `print` in `reducer.py` except nothing — DIAG formatting helpers return strings; the runtime prints.
- Boolean feature flags are STRICT-BOOL: only a real `True` enables (`x is True`); warn to stderr on a present non-bool value (the `enabled: flase` lesson).
- TDD: every behavior lands test-first. Run tests from the repo root with plain `python3 -m pytest`.
- Baseline before Task 1: `python3 -m pytest tests/ -q` → 648 passed, 2 skipped. Never commit with new failures.
- New `EndSession` reasons are exactly the strings `"enroll_verify_failed"` and `"speaker_mismatch"`.

---

### Task 1: Reducer ladder — `SpeakerWindowVerdict` event, ctx counters, `lockout_enabled`, `_on_speaker_window_verdict`

**Files:**
- Modify: `modes/director/events.py` (append event)
- Modify: `modes/director/config.py` (append field)
- Modify: `modes/director/context.py` (two ctx fields)
- Modify: `modes/director/reducer.py` (dispatch line + handler)
- Test: `tests/director/test_reducer_speaker_verdict.py` (new)

**Interfaces:**
- Consumes: existing `reduce(state, ctx, event)`, `Context`, `DirectorConfig`, `C.EndSession`.
- Produces: `E.SpeakerWindowVerdict(score: float, smoother_ok: bool, window_rms: float)` (frozen dataclass); `DirectorConfig.lockout_enabled: bool = False`; `Context.windows_seen: int = 0`, `Context.miss_streak: int = 0`; reducer handler `_on_speaker_window_verdict(state, ctx, ev) -> tuple`. Tasks 2/4/5 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/director/test_reducer_speaker_verdict.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_reducer_speaker_verdict.py -q`
Expected: errors — `AttributeError: ... has no attribute 'SpeakerWindowVerdict'` / `TypeError: ... unexpected keyword argument 'lockout_enabled'`.

- [ ] **Step 3: Implement**

`modes/director/events.py` — append at end of file:

```python
@dataclass(frozen=True)
class SpeakerWindowVerdict:
    """Accumulated-window ECAPA verdict from the SafetyNetWorker (Director-09).
    Pure data; the reducer owns the WARN/EJECT decision. No clock field — the
    ladder is not time-based (the post-eject quiet clock lives in the WakeGate)."""
    score: float                     # cosine(window embedding, primary)
    smoother_ok: bool                # M-of-N smoother output for this window
    window_rms: float                # RMS over the window audio (eject rms check)
```

`modes/director/config.py` — append inside `DirectorConfig` after `reject_bystanders`:

```python
    # Director-09: EJECT authority for the safety-net ladder. False = shadow mode
    # (verdicts + WARN visibility, no session ends). Strict-bool mapped from
    # turn_gate.lockout.enabled.
    lockout_enabled: bool = False
```

`modes/director/context.py` — append inside `Context` after `presence_since`:

```python
    windows_seen: int = 0           # SpeakerWindowVerdict count (Director-09)
    miss_streak: int = 0            # consecutive smoother-fail windows (Director-09)
```

`modes/director/reducer.py` — in `reduce()`, add a dispatch line directly after the `OwnerPresenceEvent` block (before the final `return state, []`):

```python
    if isinstance(event, E.SpeakerWindowVerdict):
        return _on_speaker_window_verdict(state, ctx, event)
```

and append the handler after `_on_tick`:

```python
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
```

- [ ] **Step 4: Run the new tests, then the director suite**

Run: `python3 -m pytest tests/director/test_reducer_speaker_verdict.py -q` → 11 passed.
Run: `python3 -m pytest tests/director/ -q` → all pass (no existing behavior touched).

- [ ] **Step 5: Commit**

```bash
git add modes/director/events.py modes/director/config.py modes/director/context.py \
        modes/director/reducer.py tests/director/test_reducer_speaker_verdict.py
git commit -m "feat(director-09): pure-reducer safety-net ladder (SpeakerWindowVerdict)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `AccumulateSpeakerAudio` command — reducer emission on served speech

**Files:**
- Modify: `modes/director/commands.py` (append command)
- Modify: `modes/director/reducer.py` (`_on_user_segment`, `_on_interjection_segment`)
- Test: `tests/director/test_reducer_accumulate.py` (new)
- Modify: existing tests that assert exact command lists (see Step 4)

**Interfaces:**
- Consumes: `classify_new_turn` / `TurnVerdict` (Director-08), the interjection gate ladder.
- Produces: `C.AccumulateSpeakerAudio()` (frozen, no fields). Emission rule relied on by Tasks 3-5: `[AccumulateSpeakerAudio()]` on ACCUMULATE, `[AccumulateSpeakerAudio(), TranscribeUserTurn()]` on ACCEPT (that exact order), `[AccumulateSpeakerAudio(), TranscribeInterjection()]` on the interjection accept branch. Rejected segments never accumulate. Emitted in BOTH `reject_bystanders` modes (the runtime no-ops when no worker exists).

- [ ] **Step 1: Write the failing tests**

Create `tests/director/test_reducer_accumulate.py`:

```python
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


def _seg(rms=1.0, is_target=True, endpoint=0.9):
    return E.SegmentEndpointed(duration_ms=500.0, rms=rms,
                               is_target=is_target, endpoint_prob=endpoint)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_reducer_accumulate.py -q`
Expected: `AttributeError: module ... has no attribute 'AccumulateSpeakerAudio'`.

- [ ] **Step 3: Implement**

`modes/director/commands.py` — append at end of file:

```python
@dataclass(frozen=True)
class AccumulateSpeakerAudio:
    """Feed the last-staged segment audio into the safety-net rolling buffer
    (Director-09). Emitted ONLY for served/plausibly-owner speech. Carries no
    audio — worker staging, same discipline as the Transcribe* commands."""
    pass
```

`modes/director/reducer.py` — replace the whole `_on_user_segment` with:

```python
def _on_user_segment(ctx: Context, ev: E.SegmentEndpointed) -> tuple:
    v = classify_new_turn(ctx, ev)
    if v in (TurnVerdict.ACCEPT, TurnVerdict.ACCUMULATE):
        # Plausibly-owner speech: reset the silence clock and feed the safety
        # net (Director-09) — only served speech reaches the hijack buffer, so
        # D08-rejected bystander chatter can never eject the owner (spec s3.2).
        ctx.last_speech_at = ctx.now
        cmds = [C.AccumulateSpeakerAudio()]
        if v is TurnVerdict.ACCEPT:
            cmds.append(C.TranscribeUserTurn())
        return State.LISTENING, cmds
    # Rejected. Legacy mode (reject_bystanders off) keeps its historical clock
    # reset on ANY voiced segment; reject-by-default does not.
    if not ctx.cfg.reject_bystanders:
        ctx.last_speech_at = ctx.now
    return State.LISTENING, []
```

(Behavior note: in legacy mode `classify_new_turn` short-circuits to legacy verdicts, so ACCEPT/ACCUMULATE/REJECT_NOT_TARGET are the only possibilities and the clock-reset semantics above are byte-for-byte today's — verified by the existing D08 test file in Step 4.)

`modes/director/reducer.py` — in `_on_interjection_segment`, change only the final line:

```python
    return State.EVALUATING, [C.AccumulateSpeakerAudio(), C.TranscribeInterjection()]
```

- [ ] **Step 4: Fix existing command-list assertions, then run**

Run: `python3 -m pytest tests/director/ tests/kiosk/ -q` and fix every failure that is an exact-command-list assertion now missing `C.AccumulateSpeakerAudio()`. Known files (verify by running — there may be others):
- `tests/director/test_reducer_reject_bystanders.py` — 3 asserts (`test_off_complete_target...`, `test_on_present_proximate...`, `test_on_unavailable_proximate...`) become `[C.AccumulateSpeakerAudio(), C.TranscribeUserTurn()]`; the two "accumulates and resets" tests become `cmds == [C.AccumulateSpeakerAudio()]`.
- `tests/director/test_listening_turn.py`, `tests/director/test_evaluating.py`, `tests/director/test_director_integration.py`, `tests/director/test_runtime.py` — update any `TranscribeUserTurn`/`TranscribeInterjection` exact-list asserts the same way. Do NOT weaken asserts to `in`-checks; keep exact lists.

Run: `python3 -m pytest tests/ -q` → everything passes (648+ + 8 new − 0).

- [ ] **Step 5: Commit**

```bash
git add modes/director/commands.py modes/director/reducer.py tests/director/
git commit -m "feat(director-09): AccumulateSpeakerAudio on served speech only

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `SafetyVerdict.window_rms` + `SafetyNetWorker`

**Files:**
- Modify: `modes/director/safety_net.py` (`SafetyVerdict` + `maybe_verify`)
- Create: `modes/director/workers/safety_net.py`
- Test: `tests/director/test_safety_net.py` (extend), `tests/director/test_safety_net_worker.py` (new)

**Interfaces:**
- Consumes: `SafetyNet.accumulate(audio, is_target)`, `SafetyNet.maybe_verify() -> Optional[SafetyVerdict]`, `EventBus.emit`, `C.AccumulateSpeakerAudio`, `E.SpeakerWindowVerdict` (Task 1).
- Produces: `SafetyVerdict(score, smoother_ok, window_rms)`; `SafetyNetWorker(safety_net, bus)` with `set_pending_audio(audio) -> None` and `async execute(command) -> None`. Tasks 4-5 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/director/test_safety_net.py`:

```python
def test_verdict_carries_window_rms():
    # embedder/primary fakes follow this file's existing pattern
    class _Emb:
        def extract(self, audio, sample_rate=16000):
            return np.ones(4, dtype=np.float32)
    net = SafetyNet(_Emb(), np.ones(4, dtype=np.float32),
                    verify_window_ms=100, sr=16000)
    net.accumulate(np.full(1600, 0.5, dtype=np.float32), is_target=True)
    v = net.maybe_verify()
    assert v is not None
    assert abs(v.window_rms - 0.5) < 1e-6
```

Create `tests/director/test_safety_net_worker.py` (bus draining uses `bus.qsize()` + `await bus.get()`, the pattern from `tests/director/test_ingestion_worker.py`):

```python
from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.director.bus import EventBus
from modes.director.safety_net import SafetyNet
from modes.director.workers.safety_net import SafetyNetWorker
from modes.director import commands as C
from modes.director import events as E


class _Emb:
    def extract(self, audio, sample_rate=16000):
        return np.ones(4, dtype=np.float32)      # cosine vs ones-primary == 1.0


def _worker(verify_window_ms=100):
    bus = EventBus()
    net = SafetyNet(_Emb(), np.ones(4, dtype=np.float32),
                    verify_window_ms=verify_window_ms, threshold=0.30, sr=16000)
    return SafetyNetWorker(net, bus), bus


async def _events(bus):
    return [await bus.get() for _ in range(bus.qsize())]


@pytest.mark.asyncio
async def test_execute_without_staged_audio_is_a_noop():
    worker, bus = _worker()
    await worker.execute(C.AccumulateSpeakerAudio())
    assert await _events(bus) == []


@pytest.mark.asyncio
async def test_pending_is_consumed_once():
    worker, bus = _worker(verify_window_ms=100)   # 1600 samples fills a window
    worker.set_pending_audio(np.full(1600, 0.5, dtype=np.float32))
    await worker.execute(C.AccumulateSpeakerAudio())
    assert len(await _events(bus)) == 1
    await worker.execute(C.AccumulateSpeakerAudio())          # nothing staged now
    assert await _events(bus) == []


@pytest.mark.asyncio
async def test_subwindow_audio_emits_nothing_until_window_fills():
    worker, bus = _worker(verify_window_ms=100)
    worker.set_pending_audio(np.full(800, 0.5, dtype=np.float32))   # half a window
    await worker.execute(C.AccumulateSpeakerAudio())
    assert await _events(bus) == []
    worker.set_pending_audio(np.full(800, 0.5, dtype=np.float32))   # completes it
    await worker.execute(C.AccumulateSpeakerAudio())
    events = await _events(bus)
    assert len(events) == 1 and isinstance(events[0], E.SpeakerWindowVerdict)
    assert events[0].smoother_ok is True and events[0].score > 0.99
    assert events[0].window_rms == pytest.approx(0.5, abs=1e-6)


@pytest.mark.asyncio
async def test_long_audio_drains_multiple_windows_in_order():
    worker, bus = _worker(verify_window_ms=100)
    worker.set_pending_audio(np.full(3300, 0.5, dtype=np.float32))  # 2 windows + rest
    await worker.execute(C.AccumulateSpeakerAudio())
    events = await _events(bus)
    assert len(events) == 2
    assert all(isinstance(e, E.SpeakerWindowVerdict) for e in events)


@pytest.mark.asyncio
async def test_non_accumulate_commands_are_ignored():
    worker, bus = _worker()
    worker.set_pending_audio(np.full(1600, 0.5, dtype=np.float32))
    await worker.execute(C.TranscribeUserTurn())
    assert await _events(bus) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_safety_net.py tests/director/test_safety_net_worker.py -q`
Expected: FAIL — `window_rms` unexpected / `ModuleNotFoundError: modes.director.workers.safety_net`.

- [ ] **Step 3: Implement**

`modes/director/safety_net.py` — replace `SafetyVerdict` and the end of `maybe_verify`:

```python
@dataclass(frozen=True)
class SafetyVerdict:
    score: float
    smoother_ok: bool
    window_rms: float                # RMS of the exact window audio consumed
```

```python
    def maybe_verify(self) -> Optional[SafetyVerdict]:
        if self._buf.size < self._need:
            return None
        window = self._buf[: self._need]
        self._buf = self._buf[self._need:]           # consume the window
        emb = self._embedder.extract(window, sample_rate=self._sr)
        score = _cosine(emb, self._primary)
        smoother_ok = self._smoother.update(score)
        window_rms = float(np.sqrt(np.mean(np.square(window))))
        return SafetyVerdict(score=score, smoother_ok=smoother_ok,
                             window_rms=window_rms)
```

Create `modes/director/workers/safety_net.py`:

```python
"""SafetyNetWorker — executes AccumulateSpeakerAudio (Director-09).

The reducer decides WHICH segments count (served/plausibly-owner only, spec
s3.2); this worker only buffers the last-staged audio and, when a verify window
fills, embeds it OFF the event loop (run_in_executor, ECAPA ~108ms p95) and
emits SpeakerWindowVerdict. No decisions here — the reducer owns the ladder.
An empty pending buffer is a no-op: the assembly factory's seeded first segment
(the enrollment utterance) is deliberately never staged, so window 1 is real
post-enrollment speech (spec s3.2 seed exclusion)."""

import asyncio

from modes.director.bus import EventBus
from modes.director import commands as C
from modes.director import events as E


class SafetyNetWorker:
    def __init__(self, safety_net, bus: EventBus):
        self._net = safety_net
        self._bus = bus
        self._pending = None

    def set_pending_audio(self, audio) -> None:
        self._pending = audio

    async def execute(self, command) -> None:
        if not isinstance(command, C.AccumulateSpeakerAudio):
            return
        audio, self._pending = self._pending, None
        if audio is None or len(audio) == 0:
            return
        loop = asyncio.get_event_loop()
        verdicts = await loop.run_in_executor(None, self._drain, audio)
        for v in verdicts:
            await self._bus.emit(E.SpeakerWindowVerdict(
                score=v.score, smoother_ok=v.smoother_ok,
                window_rms=v.window_rms))

    def _drain(self, audio):
        """Accumulate, then consume EVERY full window (a long turn can complete
        more than one). Runs in the executor: accumulate + embed off the loop."""
        self._net.accumulate(audio, is_target=True)
        out = []
        while True:
            v = self._net.maybe_verify()
            if v is None:
                return out
            out.append(v)
```

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/director/test_safety_net.py tests/director/test_safety_net_worker.py -q` → all pass.
Run: `python3 -m pytest tests/director/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add modes/director/safety_net.py modes/director/workers/safety_net.py \
        tests/director/test_safety_net.py tests/director/test_safety_net_worker.py
git commit -m "feat(director-09): SafetyNetWorker + SafetyVerdict.window_rms

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wiring — assembly build (strict-bool), ingestion staging, runtime routing

**Files:**
- Modify: `modes/director/assembly.py` (`_build_safety_net`, `_director_config_from`, `build_director_runtime`)
- Modify: `modes/director/workers/ingestion.py` (constructor + `_on_segment` staging)
- Modify: `modes/director/runtime.py` (constructor + `_route`)
- Test: `tests/director/test_assembly.py` (extend), `tests/director/test_ingestion_worker.py` (extend), `tests/director/test_runtime.py` (extend)

**Interfaces:**
- Consumes: `SafetyNetWorker` (Task 3), `C.AccumulateSpeakerAudio` (Task 2), `DirectorConfig.lockout_enabled` (Task 1).
- Produces: `_build_safety_net(tb_cfg, primary_embedding, embedder, bus) -> Optional[SafetyNetWorker]`; `IngestionWorker(..., safety_worker=None)` kwarg; `DirectorRuntime(..., safety_worker=None)` kwarg; `_director_config_from` maps `lockout_enabled`. The seeded first segment is NOT staged into the safety worker (factory untouched; the empty-pending no-op that covers the seed is Task 3's `test_execute_without_staged_audio_is_a_noop`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/director/test_assembly.py` (follow its existing fixture style for `tb_cfg` dicts / fake handoffs):

```python
def test_safety_net_requires_strict_bool_true():
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    emb = object()
    prim = np.ones(4, dtype=np.float32)
    bus = EventBus()
    on = {"turn_gate": {"require_speaker_match": True}}
    off = {"turn_gate": {"require_speaker_match": False}}
    truthy_string = {"turn_gate": {"require_speaker_match": "true"}}
    missing = {}
    assert _build_safety_net(on, prim, emb, bus) is not None
    assert _build_safety_net(off, prim, emb, bus) is None
    assert _build_safety_net(truthy_string, prim, emb, bus) is None
    assert _build_safety_net(missing, prim, emb, bus) is None


def test_safety_net_none_without_embedder_or_primary():
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    cfg = {"turn_gate": {"require_speaker_match": True}}
    assert _build_safety_net(cfg, None, object(), EventBus()) is None
    assert _build_safety_net(cfg, np.ones(4), None, EventBus()) is None


def test_lockout_enabled_strict_bool_mapping():
    from modes.director.assembly import _director_config_from
    assert _director_config_from(
        {"turn_gate": {"lockout": {"enabled": True}}}).lockout_enabled is True
    assert _director_config_from(
        {"turn_gate": {"lockout": {"enabled": "true"}}}).lockout_enabled is False
    assert _director_config_from({}).lockout_enabled is False
```

In `tests/director/test_ingestion_worker.py`, extend `make_worker` with a pass-through kwarg — change its signature to
`def make_worker(mic, vad, state, turn_prob=0.9, embedder_score=0.9, pvad=None, safety=None):`
and add `safety_worker=safety,` to its `IngestionWorker(...)` call. Then append:

```python
@pytest.mark.asyncio
async def test_listening_segment_staged_into_safety_worker():
    seg = _seg()
    safety = MagicMock()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.LISTENING, safety=safety)
    await _run_briefly(w)
    safety.set_pending_audio.assert_called_once()
    assert np.array_equal(safety.set_pending_audio.call_args[0][0], seg.audio)


@pytest.mark.asyncio
async def test_evaluating_segment_staged_into_safety_worker():
    seg = _seg()
    safety = MagicMock()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.EVALUATING, safety=safety)
    await _run_briefly(w)
    safety.set_pending_audio.assert_called_once()
```

(The `safety=None` default means every existing test in the file doubles as the
no-safety-worker regression check — no third test needed.)

Append to `tests/director/test_runtime.py` (its `build_runtime` helper does not pass a
safety worker, so `rt._safety` defaults to None; set it directly for the spy test):

```python
@pytest.mark.asyncio
async def test_accumulate_command_routes_to_safety_worker():
    rt, bus, director, playback, generation = build_runtime(lambda: 0.0)
    safety = MagicMock()
    safety.execute = AsyncMock()
    rt._safety = safety
    await rt._route(C.AccumulateSpeakerAudio())
    safety.execute.assert_awaited_once_with(C.AccumulateSpeakerAudio())


@pytest.mark.asyncio
async def test_accumulate_command_with_no_safety_worker_is_noop():
    rt, bus, director, playback, generation = build_runtime(lambda: 0.0)
    await rt._route(C.AccumulateSpeakerAudio())    # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_assembly.py tests/director/test_ingestion_worker.py tests/director/test_runtime.py -q`
Expected: FAIL — `_build_safety_net` missing, unexpected kwarg `safety_worker`, `lockout_enabled` unmapped.

- [ ] **Step 3: Implement**

`modes/director/assembly.py`:

Add import near the other worker imports:

```python
from modes.director.safety_net import SafetyNet
from modes.director.workers.safety_net import SafetyNetWorker
```

Add after `_build_vision`:

```python
def _build_safety_net(tb_cfg: dict, primary_embedding, embedder, bus):
    """Build the Director-09 accumulated-window ECAPA safety net, or None (no
    hijack detection — byte-for-byte Director-08 behavior). Strict-bool enable:
    only a real True builds it (the 'flase' lesson); warn on a present non-bool."""
    tg = tb_cfg.get("turn_gate", {})
    enabled = tg.get("require_speaker_match", False)
    if enabled is not True:
        if "require_speaker_match" in tg and not isinstance(enabled, bool):
            print(f"[director] turn_gate.require_speaker_match is not a boolean "
                  f"(got {enabled!r}) -> safety net disabled",
                  file=sys.stderr, flush=True)
        return None
    if embedder is None or primary_embedding is None:
        return None
    lock = tg.get("lockout", {})
    net = SafetyNet(
        embedder, primary_embedding,
        verify_window_ms=tg.get("verify_window_ms", 2000),
        threshold=tg.get("speaker_threshold", 0.30),
        window_size=lock.get("window_size", 3),
        min_matches=lock.get("min_matches", 1),
        sr=tb_cfg.get("sample_rate_hz", 16000),
    )
    print("[director] safety net ENABLED (accumulated-window ECAPA)",
          file=sys.stderr, flush=True)
    return SafetyNetWorker(net, bus)
```

In `_director_config_from`, add before the closing paren:

```python
        lockout_enabled=tb_cfg.get("turn_gate", {}).get("lockout", {})
                              .get("enabled", False) is True,
```

In `build_director_runtime`, after the `pvad = _build_pvad(...)` line:

```python
    safety = _build_safety_net(tb_cfg, handoff.primary_embedding,
                               handoff.embedder, bus)
```

then pass `safety_worker=safety` to BOTH the `IngestionWorker(...)` and `DirectorRuntime(...)` constructions. Do NOT touch `DirectorRuntimeFactory.run` — the seeded first segment is deliberately not staged into the safety net (spec s3.2 seed exclusion; the reducer's `AccumulateSpeakerAudio` for the seed no-ops on the empty pending buffer).

`modes/director/workers/ingestion.py`: add `safety_worker=None` to `__init__` kwargs and store `self._safety = safety_worker`. In `_on_segment`, in the `State.LISTENING` branch immediately before `self._stt.set_pending_user_audio(seg.audio)` AND in the `State.EVALUATING` branch immediately before `self._stt.set_pending_interjection_audio(seg.audio)`, add:

```python
            if self._safety is not None:
                self._safety.set_pending_audio(seg.audio)
```

`modes/director/runtime.py`: add `safety_worker=None` to `DirectorRuntime.__init__` kwargs, store `self._safety = safety_worker`, and add to `_route` before the `EndSession` branch:

```python
        elif isinstance(command, C.AccumulateSpeakerAudio):
            if self._safety is not None:
                await self._safety.execute(command)
```

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/director/ -q` → all pass.
Run: `python3 -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add modes/director/assembly.py modes/director/workers/ingestion.py \
        modes/director/runtime.py tests/director/
git commit -m "feat(director-09): wire safety net (assembly strict-bool, staging, routing)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Runtime DIAG observability

**Files:**
- Modify: `modes/director/reducer.py` (pure helper `safety_diag_line`)
- Modify: `modes/director/runtime.py` (print after dispatch)
- Test: `tests/director/test_safety_diag.py` (new)

**Interfaces:**
- Consumes: ctx counters (Task 1) post-dispatch, the dispatched event + returned commands.
- Produces: `safety_diag_line(ctx, ev, commands) -> str` (pure; runtime calls it AFTER `dispatch`, so counters are already advanced).

- [ ] **Step 1: Write the failing tests**

Create `tests/director/test_safety_diag.py`:

```python
from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.director import events as E
from modes.director import commands as C
from modes.director.reducer import reduce, safety_diag_line
from modes.talkback.conversation import ConversationManager


def _ctx(lockout=True):
    cfg = DirectorConfig(lockout_enabled=lockout)
    return new_context(cfg, ConversationManager(system_prompt="x"),
                       now=5.0, proximity_rms=0.5)


def _run(ctx, ev):
    state, cmds = reduce(State.LISTENING, ctx, ev)
    return safety_diag_line(ctx, ev, cmds)


def test_passing_window_line():
    line = _run(_ctx(), E.SpeakerWindowVerdict(0.85, True, 0.4))
    assert "window=1" in line and "score=0.850" in line and "WARN" not in line


def test_warn_line():
    ctx = _ctx()
    _run(ctx, E.SpeakerWindowVerdict(0.9, True, 0.4))
    line = _run(ctx, E.SpeakerWindowVerdict(0.1, False, 0.4))
    assert "WARN" in line and "streak=1" in line and "shadow" not in line


def test_shadow_warn_line_is_marked():
    line = _run(_ctx(lockout=False), E.SpeakerWindowVerdict(0.1, False, 0.4))
    assert "WARN" in line and "shadow" in line


def test_eject_line_carries_reason():
    ctx = _ctx()
    line = _run(ctx, E.SpeakerWindowVerdict(0.1, False, 0.4))   # window-1 fail
    assert "EJECT" in line and "enroll_verify_failed" in line
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_safety_diag.py -q`
Expected: `ImportError: cannot import name 'safety_diag_line'`.

- [ ] **Step 3: Implement**

`modes/director/reducer.py` — append after `gate_diag_reason`:

```python
def safety_diag_line(ctx: Context, ev, commands) -> str:
    """DIAG-only formatting for a SpeakerWindowVerdict (spec s7). Pure — the
    runtime prints. Call AFTER dispatch: ctx counters are already advanced."""
    line = (f"safety-net window={ctx.windows_seen} score={ev.score:.3f} "
            f"smoother_ok={ev.smoother_ok} streak={ctx.miss_streak} "
            f"rms={ev.window_rms:.4f}")
    ends = [c for c in commands if isinstance(c, C.EndSession)]
    if ends:
        return f"{line} EJECT reason={ends[0].reason}"
    if not ev.smoother_ok:
        shadow = "" if ctx.cfg.lockout_enabled else " (shadow)"
        return f"{line} WARN{shadow}"
    return line
```

`modes/director/runtime.py` — import `safety_diag_line` alongside `gate_diag_reason`, and in `run_async` directly after the existing `SegmentEndpointed` DIAG block add:

```python
                if _DIAG and isinstance(event, E.SpeakerWindowVerdict):
                    _diag(safety_diag_line(self._director.ctx, event, commands))
```

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/director/test_safety_diag.py tests/director/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add modes/director/reducer.py modes/director/runtime.py \
        tests/director/test_safety_diag.py
git commit -m "feat(director-09): runtime DIAG for safety-net verdicts (safety_diag_line)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: WakeGate split-half verify-before-serve

**Files:**
- Modify: `modes/director/verify.py` (docstring only)
- Modify: `modes/director/wakegate.py` (`_start_session_from_segment`)
- Modify: `modes/talkback/handoff.py` (drop `DirectorHandoff.holdout_embedding`)
- Test: `tests/director/test_verify_before_serve.py` (rewrite for the new role), `tests/director/test_wakegate.py` (extend)

**Interfaces:**
- Consumes: `verify_before_serve(primary, holdout, threshold) -> (ok, score)` (function unchanged).
- Produces: WakeGate `verify_refused` callback event `{"score": float}`; `DirectorHandoff` WITHOUT `holdout_embedding` (Task-7 wakegate code and any test constructing `DirectorHandoff` must not pass it). `TalkbackHandoff` keeps its optional field (legacy controller path untouched).

- [ ] **Step 1: Write the failing tests**

Append to `tests/director/test_wakegate.py`, reusing its fixtures (`base_config`, `fake_mic`, `fake_vad`, `fake_embedder`, `fake_wake`, `fake_runtime`) and helpers (`make_gate`, `make_segment`, `drive_one_cycle`). Note `make_segment(1000.0)` is exactly 1.0s = 16000 samples, so the default segment triggers the split-half path; the existing MagicMock embedder returns the same vector on every call (cosine 1.0), so existing handoff tests keep passing.

```python
class _SplitEmbedder:
    """Full-segment extract -> normalized ones; halves -> scripted vectors."""
    def __init__(self, half_a, half_b):
        self._returns = [np.ones(192, dtype=np.float32) / np.sqrt(192),
                         half_a, half_b]
        self.calls = 0

    def extract(self, audio):
        v = self._returns[min(self.calls, len(self._returns) - 1)]
        self.calls += 1
        return v


def _orthogonal_pair():
    a = np.zeros(192, dtype=np.float32); a[0] = 1.0
    b = np.zeros(192, dtype=np.float32); b[1] = 1.0
    return a, b                                     # cosine == 0.0 < 0.80


class TestVerifyBeforeServe:
    def test_split_half_mismatch_refuses_session(
            self, base_config, fake_mic, fake_vad, fake_wake, fake_runtime):
        a, b = _orthogonal_pair()
        emb = _SplitEmbedder(a, b)
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, emb, fake_wake, fake_runtime,
                      on_event=lambda et, pl: events.append((et, pl)))
        drive_one_cycle(g, fake_wake, fake_vad, seg=make_segment(1000.0))
        fake_runtime.run.assert_not_called()
        assert g._state == "IDLE"
        types = [et for et, _ in events]
        assert "verify_refused" in types and "session_started" not in types
        refused = next(pl for et, pl in events if et == "verify_refused")
        assert refused["score"] == pytest.approx(0.0, abs=1e-6)

    def test_split_half_match_serves(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        drive_one_cycle(g, fake_wake, fake_vad, seg=make_segment(1000.0))
        fake_runtime.run.assert_called_once()
        assert "session_started" in [et for et, _ in events]

    def test_short_first_segment_skips_split_half(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        drive_one_cycle(
            make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime),
            fake_wake, fake_vad, seg=make_segment(500.0))
        assert fake_embedder.extract.call_count == 1    # full segment only
        fake_runtime.run.assert_called_once()

    def test_half_embed_failure_resets_to_idle(
            self, base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
            fake_runtime):
        calls = {"n": 0}

        def _extract(audio):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("half embed failed")
            return np.ones(192, dtype=np.float32)

        fake_embedder.extract = MagicMock(side_effect=_extract)
        events = []
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                      fake_runtime, on_event=lambda et, pl: events.append((et, pl)))
        drive_one_cycle(g, fake_wake, fake_vad, seg=make_segment(1000.0))
        fake_runtime.run.assert_not_called()
        assert g._state == "IDLE"
        # infra failure, not a verdict: no verify_refused event
        assert "verify_refused" not in [et for et, _ in events]
```

Also in this file: DELETE `test_holdout_embedding_is_first_segment_embedding_placeholder` (the placeholder it pins is exactly what this task removes).

Rewrite `tests/director/test_verify_before_serve.py`'s docstring/test names to the split-half role (the function's math tests stay valid — keep them; update any test that constructs `DirectorHandoff` with `holdout_embedding`).

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_wakegate.py -q`
Expected: new tests FAIL (no refusal path exists; embedder called once).

- [ ] **Step 3: Implement**

`modes/director/wakegate.py` — add import:

```python
from modes.director.verify import verify_before_serve
```

Replace `_start_session_from_segment`:

```python
    def _start_session_from_segment(self, segment: SpeechSegment) -> None:
        try:
            embedding = self.embedder.extract(segment.audio)
        except Exception:
            self._reset_to_idle()
            return

        # Verify-before-serve (Director-09 spec s5): split-half self-similarity.
        # Same-utterance halves of one speaker are highly self-similar; a noise/
        # garbage first segment is not — so 0.80 is honest HERE, while cross-
        # utterance verification (too noisy on short audio) is window 1's job.
        # Only for segments >= 1.0s: halves off the 300ms VAD floor are too
        # short to compare honestly and would false-refuse real users.
        sr = 16000
        if len(segment.audio) >= sr:
            thr = self._talkback_config.get("verify_before_serve_threshold", 0.80)
            half = len(segment.audio) // 2
            try:
                emb_a = self.embedder.extract(segment.audio[:half])
                emb_b = self.embedder.extract(segment.audio[half:])
            except Exception:
                self._reset_to_idle()      # infra failure == the existing embed-fail path
                return
            ok, score = verify_before_serve(emb_a, emb_b, thr)
            if not ok:
                self._reset_to_idle()
                self._safe_callback(self.on_event, "verify_refused",
                                    {"score": float(score)})
                return

        self._safe_callback(self.on_event, "session_started",
                            {"snapshot_norm": float(np.linalg.norm(embedding))})

        # Stage the handoff for run() to execute AFTER _collect_handoff closes the
        # wake mic generator — so runtime.run is NOT called from inside a parked
        # generator and the Director's ingestion is the sole mic consumer. run()
        # makes the single blocking runtime.run call and emits session_ended from
        # DirectorResult.reason (spec section 4a — the Director is the sole session
        # owner and the only end-reason authority).
        self._pending_handoff = DirectorHandoff(
            mic=self.mic,
            primary_embedding=embedding,
            first_segment=segment,
            config=self._talkback_config,
            vad=self.vad,
            embedder=self.embedder,
        )
```

`modes/talkback/handoff.py` — delete the `holdout_embedding` field from `DirectorHandoff` (keep `TalkbackHandoff`'s optional one) and update the module docstring: the Plan-05 placeholder paragraph is replaced by one line — "Verify-before-serve is the WakeGate's split-half check (Director-09); no holdout travels in the handoff."

`modes/director/verify.py` — update the docstring to the split-half role (function body unchanged):

```python
"""Verify-before-serve gate (Director-09 spec s5).

Scores two embeddings against each other; the WakeGate calls it with the two
HALVES of the first segment (same-utterance self-similarity — the only
comparison where 0.80 is statistically honest on short audio). Below threshold
-> refuse to serve (no session). Cross-utterance verification is the safety
net's window-1 job."""
```

Then `grep -rn "holdout_embedding" --include="*.py" .` and remove every remaining `DirectorHandoff(... holdout_embedding=...)` argument (tests and any fixtures; `TalkbackHandoff` usages stay).

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/director/ tests/kiosk/ -q` → all pass.
Run: `python3 -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add modes/director/wakegate.py modes/director/verify.py modes/talkback/handoff.py tests/
git commit -m "feat(director-09): real verify-before-serve (split-half) in the WakeGate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Post-eject quiet-hold + delete `lockout.py`

**Files:**
- Modify: `modes/talkback/handoff.py` (`DirectorResult.proximity_rms` field)
- Modify: `modes/director/runtime.py` (return it)
- Modify: `modes/director/wakegate.py` (hold state)
- Delete: `modes/director/lockout.py`, `tests/director/test_lockout.py`
- Test: `tests/director/test_wakegate_hold.py` (new)

**Interfaces:**
- Consumes: `DirectorResult` (frozen), `WakeGate.run` loop, `_handle_idle_chunk`.
- Produces: `DirectorResult.proximity_rms: float = 0.0`; WakeGate hold behavior: engages only on reason `"speaker_mismatch"` with `proximity_rms > 0`; chunk RMS ≥ floor resets the quiet clock; `lockout_idle_after_s` (config, default 5.0) of continuous quiet clears it; every other reason never holds.

- [ ] **Step 1: Write the failing tests**

Create `tests/director/test_wakegate_hold.py`. It imports the fixtures/helpers from `tests/director/test_wakegate.py` by re-declaring the same fixtures (copy the `base_config`/`fake_mic`/`fake_vad`/`fake_embedder`/`fake_wake` fixtures and `make_gate`/`make_segment`/`drive_one_cycle` helpers into this file — pytest fixtures don't import across files without a conftest, and we are not adding one for this). Time control uses the file's existing pattern: `monkeypatch.setattr("modes.director.wakegate.time.monotonic", lambda: clock[0])`.

```python
def _mismatch_runtime(prox=0.5):
    m = MagicMock()
    m.run = MagicMock(return_value=DirectorResult(
        reason="speaker_mismatch", turns=1, total_duration_s=1.0,
        proximity_rms=prox))
    return m


def _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake, clock,
            monkeypatch, runtime=None):
    monkeypatch.setattr("modes.director.wakegate.time.monotonic",
                        lambda: clock[0])
    g = make_gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                  runtime or _mismatch_runtime())
    drive_one_cycle(g, fake_wake, fake_vad)
    fake_wake.process.return_value = None      # no accidental wakes below
    return g


def test_speaker_mismatch_result_engages_hold(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    g = _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                clock, monkeypatch)
    assert g._hold_floor == 0.5
    fake_wake.process.reset_mock()
    g._handle_chunk(np.full(480, 0.9, dtype=np.float32))    # loud: still held
    fake_wake.process.assert_not_called()                    # wake never consulted


def test_hold_clears_after_quiet_period(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    g = _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                clock, monkeypatch)
    quiet = np.zeros(480, dtype=np.float32)
    g._handle_chunk(quiet)                     # holding; quiet clock running
    clock[0] = 1006.0                          # > lockout_idle_after_s (5s)
    g._handle_chunk(quiet)                     # clears + falls through to wake
    assert g._hold_floor is None
    fake_wake.process.reset_mock()
    fake_wake.process.return_value = 0.9
    g._handle_chunk(quiet)                     # wake works again
    fake_wake.process.assert_called_once()
    assert g._state == "AWAIT_FIRST_SEGMENT"


def test_loud_chunk_resets_the_quiet_clock(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    g = _engage(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
                clock, monkeypatch)
    quiet = np.zeros(480, dtype=np.float32)
    clock[0] = 1004.0
    g._handle_chunk(np.full(480, 0.9, dtype=np.float32))    # loud at t=1004: reset
    clock[0] = 1008.0
    g._handle_chunk(quiet)                     # only 4s since loud -> still held
    assert g._hold_floor is not None
    clock[0] = 1009.5
    g._handle_chunk(quiet)                     # 5.5s since loud -> clears
    assert g._hold_floor is None


def test_other_end_reasons_never_hold(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("modes.director.wakegate.time.monotonic",
                        lambda: clock[0])
    for reason in ("silence_timeout", "enroll_verify_failed"):
        rt = MagicMock()
        rt.run = MagicMock(return_value=DirectorResult(
            reason=reason, turns=0, total_duration_s=1.0, proximity_rms=0.5))
        g = make_gate(base_config, fake_mic, fake_vad, fake_embedder,
                      fake_wake, rt)
        drive_one_cycle(g, fake_wake, fake_vad)
        assert g._hold_floor is None
```

(Imports at top of the file: `numpy as np`, `MagicMock`, `DirectorResult` from `modes.talkback.handoff`, plus the copied fixtures/helpers. The `fake_mic` fixture is function-scoped, so the loop in the last test reuses one mic across two `drive_one_cycle` calls — `drive_one_cycle` re-assigns `g.mic.stream` each time, so this is fine.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_wakegate_hold.py -q`
Expected: FAIL — `DirectorResult` has no `proximity_rms`; no hold behavior.

- [ ] **Step 3: Implement**

`modes/talkback/handoff.py` — add to `DirectorResult` (frozen; default keeps every existing constructor working):

```python
    # Director-09: the ended session's calibrated proximity floor, so the
    # WakeGate's post-eject quiet-hold has a threshold. 0.0 == unknown -> no hold.
    proximity_rms: float = 0.0
```

`modes/director/runtime.py` — in `run_async`'s return:

```python
        return DirectorResult(
            reason=self._result_reason or "stopped",
            turns=self._director.ctx.conversation.turn_count,
            total_duration_s=self._clock() - self._started_at,
            proximity_rms=self._director.ctx.proximity_rms,
        )
```

`modes/director/wakegate.py`:

In `__init__`, after `self._talkback_config = ...`:

```python
        # Director-09 post-eject quiet-hold (port of lockout.py's idle half):
        # after a speaker_mismatch end, ignore wakes until the near field has
        # been quiet this long. Never permanent — quiet always clears it.
        self._hold_idle_after_s = float(
            self._talkback_config.get("lockout_idle_after_s", 5.0))
        self._hold_floor: Optional[float] = None
        self._hold_quiet_since: float = 0.0
```

In `run()`, replace the post-result lines (spec s7 asks for hold visibility — add the two DIAG prints; `import os` at the top of the file next to `import sys`):

```python
                    result = self.runtime.run(handoff)
                    self._reset_to_idle()
                    if (result.reason == "speaker_mismatch"
                            and getattr(result, "proximity_rms", 0.0) > 0.0):
                        self._hold_floor = result.proximity_rms
                        self._hold_quiet_since = time.monotonic()
                        if os.environ.get("TVAD_DIAG"):
                            print(f"[DIAG wakegate] hold engaged "
                                  f"floor={self._hold_floor:.4f} "
                                  f"quiet_needed={self._hold_idle_after_s}s",
                                  file=sys.stderr, flush=True)
                    self._safe_callback(self.on_event, "session_ended",
                                        {"reason": result.reason})
```

In `_handle_idle_chunk`, add at the very top (before wake detection):

```python
        if self._hold_floor is not None:
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            now = time.monotonic()
            if rms >= self._hold_floor:
                self._hold_quiet_since = now      # still someone close -> keep holding
                return
            if now - self._hold_quiet_since < self._hold_idle_after_s:
                return                            # quiet, but not long enough yet
            self._hold_floor = None               # cleared; fall through to wake
            if os.environ.get("TVAD_DIAG"):
                print("[DIAG wakegate] hold cleared (near field quiet)",
                      file=sys.stderr, flush=True)
```

Delete `modes/director/lockout.py` and `tests/director/test_lockout.py` (`git rm`): the decision half now lives in `_on_speaker_window_verdict` (Task 1 tests), the idle half here (this task's tests). Check nothing else imports it: `grep -rn "lockout" --include="*.py" modes/ | grep -v talkback` must return nothing.

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/director/ -q` → all pass.
Run: `python3 -m pytest tests/ -q` → all pass (test_lockout.py gone, nothing imports the module).

- [ ] **Step 5: Commit**

```bash
git add -A modes/director/ modes/talkback/handoff.py tests/director/
git commit -m "feat(director-09): post-eject WakeGate quiet-hold; retire lockout.py

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Config truth pass

**Files:**
- Modify: `config.yaml`
- Modify: `modes/director/assembly.py` (`_director_config_from`: map `nudge_lead_s`, `conf_floor`)
- Modify: `modes/director/config.py` (fix stale line-ref comments)
- Test: `tests/director/test_assembly.py` (extend)

**Interfaces:**
- Consumes: everything above.
- Produces: config.yaml that is TRUE — every `kiosk.talkback` key is read by live code; the four silently-defaulted keys exist and are mapped.

- [ ] **Step 1: Write the failing tests**

Append to `tests/director/test_assembly.py`:

```python
def test_nudge_lead_and_conf_floor_are_mapped():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({"nudge_lead_s": 7.5,
                                 "barge_in": {"conf_floor": 0.65}})
    assert cfg.nudge_lead_s == 7.5
    assert cfg.conf_floor == 0.65


def test_shipped_config_yaml_matches_live_readers():
    import yaml
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    tb = full["kiosk"]["talkback"]
    # keys this feature makes/keeps live
    assert tb["turn_gate"]["require_speaker_match"] is True
    assert tb["turn_gate"]["lockout"]["enabled"] is True
    assert tb["turn_gate"]["endpoint_threshold"] == 0.5
    assert tb["verify_before_serve_threshold"] == 0.80
    assert tb["lockout_idle_after_s"] == 5
    assert tb["nudge_lead_s"] == 5.0
    assert tb["barge_in"]["conf_floor"] == 0.5
    assert tb["watchdog"]["tick_ms"] == 500
    # dead keys must be GONE
    assert "decision_smoother" not in full["kiosk"]
    assert "suppression_level" not in tb["aec"]
    assert "partials_every_ms" not in tb["stt"]
    assert "require_speaker_match" not in tb["barge_in"]
    assert "audio_safety_net" not in tb["vision"]
    assert "resume" not in tb
    assert "include_partial_transcripts" not in tb["logging"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/director/test_assembly.py -q` → the two new tests FAIL.

- [ ] **Step 3: Implement**

`modes/director/assembly.py` — in `_director_config_from`, add:

```python
        nudge_lead_s=tb_cfg.get("nudge_lead_s", 5.0),
        conf_floor=tb_cfg.get("barge_in", {}).get("conf_floor", 0.5),
```

`config.yaml` edits (all under `kiosk.talkback` unless noted):
1. DELETE the `kiosk.decision_smoother` block (lines ~34-37).
2. After `hard_timeout_s: 300` ADD:
   ```yaml
       # Nudge fires this many seconds BEFORE silence_timeout ("are you still
       # there?"). Read by the Director reducer; was previously a silent default.
       nudge_lead_s: 5.0
       # Watchdog tick cadence (ms) — the Director's single clock source.
       watchdog:
         tick_ms: 500
   ```
3. `verify_before_serve_threshold` comment: rewrite to the split-half role —
   ```yaml
       # Verify-before-serve (Director-09): split-half self-similarity of the FIRST
       # segment (same-utterance halves; only runs on segments >=1.0s). Below this
       # -> refuse to serve (no session), re-wake immediately. Cross-utterance
       # verification is the safety net's window-1 job (turn_gate below).
       verify_before_serve_threshold: 0.80
       # After a speaker_mismatch EJECT, accept a fresh wake only once the near
       # field has been quiet this long (never a permanent lockout). Director-09.
       lockout_idle_after_s: 5
   ```
4. `aec:` block: delete `suppression_level: "high"`.
5. `stt:` block: delete `partials_every_ms: 300` and its comment line.
6. Rewrite the `turn_gate:` block comment + keys (the accumulated-window prose is now TRUE):
   ```yaml
       # Director-09 speaker safety net: NEW turns already pass the bystander gate
       # (reject_bystanders below); this layer catches what proximity+camera can't —
       # a session hijack / close loud bystander. Consecutive SERVED turn audio
       # accumulates; every verify_window_ms window is ECAPA-scored against the
       # primary (short single turns are never judged alone — see
       # docs/speaker-gate-measurement.md). Window 1 failing = bad enrollment ->
       # session ends (enroll_verify_failed). Later: 3 consecutive failing windows
       # (M-of-N below) = WARN (log-only); a second consecutive smoother-fail that
       # is ALSO below the proximity floor = silent EJECT (speaker_mismatch).
       turn_gate:
         require_speaker_match: true  # master enable (strict-bool; only real `true`)
         speaker_threshold: 0.30      # tuned for >=2s windows (offline ~100%/100%)
         verify_window_ms: 2000       # accumulate served turn audio to this, then verify
         endpoint_threshold: 0.5      # Smart Turn endpoint_prob >= this => turn complete
         # Bystander gate v1 (Director-08): ... (keep the existing comment + key)
         reject_bystanders: true
         # EJECT authority. false = shadow mode: verdicts + WARN DIAG, no session
         # ends — the graduated-rollout knob. window_size/min_matches = the M-of-N
         # smoother over window scores (1-of-3: fails only after 3 consecutive
         # misses ~6s of served non-matching speech).
         lockout:
           enabled: true
           window_size: 3
           min_matches: 1
   ```
7. `barge_in:` block: delete `require_speaker_match: true`; after `speaker_threshold` ADD:
   ```yaml
         conf_floor: 0.5              # interjection mean_word_prob below this => RESTORE
   ```
8. `vision:` block: delete the `audio_safety_net:` sub-block and its "Reserved" comment (Director-09 realized that seam — the safety net lives under turn_gate).
9. DELETE the `resume:` block (the Director's resume steer is always-on; the key was controller-only) and the `include_partial_transcripts` line under `logging:` (keep `jsonl_path`).

`modes/director/config.py` — fix the stale line-ref comments: `config.yaml:52` → `config.yaml:51`, `:53` → `:52`, `:131`/`:130` on `verify_window_ms`/`speaker_threshold` → drop the line numbers entirely (write `barge_in.verify_window_ms (config.yaml)` / `barge_in.speaker_threshold (config.yaml)` — line numbers rot; this task is the proof).

- [ ] **Step 4: Run everything**

Run: `python3 -m pytest tests/ -q` → all pass. (If a talkback/controller test read a deleted key from the REAL config.yaml, it was already coupled wrongly — fix the test to build its own dict, mirroring how the other controller tests do.)

- [ ] **Step 5: Commit**

```bash
git add config.yaml modes/director/assembly.py modes/director/config.py \
        tests/director/test_assembly.py
git commit -m "feat(director-09): config truth pass — every talkback key is read by live code

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Full-suite gate + live validation (human at the kiosk)

**Files:** none (verification only) + `docs/notes/2026-07-XX-director-09-live.md` (verdict note, written after the live runs).

- [ ] **Step 1: Full suite**

Run: `python3 -m pytest tests/ -q`
Expected: baseline 648 − 1 deleted file's tests + ~35 new, 2 skipped, 0 failed. Record exact counts in the ledger.

- [ ] **Step 2: Live validation (HUMAN — merge gate, spec s10)**

All with `TVAD_DIAG=1 ./kiosk-stack.sh start`:
1. **Owner-normal:** multi-turn conversation → expect `safety-net window=N ... smoother_ok=True` lines, ZERO `WARN` lines, no eject. If WARNs appear for the real owner, tune `turn_gate.speaker_threshold` DOWN (0.30 → 0.25) and re-run.
2. **Hijack:** owner starts, second speaker takes over → `WARN` lines, then `EJECT reason=speaker_mismatch`; wake refused until ~5s quiet; owner re-wakes normally after.
3. **Garbage enrollment:** cough/tap as the first "turn" → `verify_refused` callback (or window-1 `EJECT reason=enroll_verify_failed`); immediate re-wake works.
4. **Shadow spot-check:** set `lockout.enabled: false`, repeat the hijack → `WARN (shadow)` lines, session continues; restore `true` after.

- [ ] **Step 3: Verdict note + finish**

Write `docs/notes/<today>-director-09-live.md` (mirror `docs/notes/2026-06-24-director-07-live.md`'s structure: per-check DIAG excerpts + verdict). Commit it. Then use superpowers:finishing-a-development-branch (D08 must merge first — this branch is stacked on it).

---

## Execution notes for the coordinator

- Tasks 1→5 are strictly ordered (each consumes the previous task's interfaces). Task 6 depends only on Task 1's spec context (independent of 2-5); Task 7 depends on 6 (same wakegate function region); Task 8 depends on 4 (mapping) and 7 (key deletions); Task 9 is last.
- The suite is fast (~15s); run the FULL suite at every task's Step 4, not just the named files.
- `bench/spatial_voice_probe.py` is untracked in the working tree — do not `git add -A` at the repo root; stage explicit paths (Task 7's `git add -A modes/director/` is scoped).
