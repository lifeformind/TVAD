# Director-11 DOA Cone Vote Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Direction becomes a fourth bystander gate — a `DoaTracker` samples the ReSpeaker's onboard DOA continuously, and a ±20° cone around a calibrated owner bearing gates new turns, interjections, and the duck-at-onset reflex.

**Architecture:** A daemon-thread sampler (`core/audio/doa_tracker.py`) polls `DOAANGLE`/`SPEECHDETECTED` over the existing USB control module every 150ms into a timestamped ring buffer. Ingestion stamps each event with the circular median over the segment's own time span (`doa_angle: float | None`); the reducer's pure `in_owner_cone` votes True/False/None (None = abstain, fail-open). The owner bearing is calibrated from a lookback window over the wake utterance and EMA-tracked on served turns only.

**Tech Stack:** Python 3.12, pyusb (already a dep via `core/audio/respeaker.py`), pytest + pytest-asyncio. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-07-director-11-doa-cone-vote-design.md` — read it if a requirement here seems ambiguous; the spec governs.

## Global Constraints

- Branch: `feat/director-11-doa-cone-vote` (already exists; work on it).
- Every commit message ends with the trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Run tests as `python3 -m pytest` — never bare `python`/`pytest`.
- Stage explicit paths only — NEVER `git add -A` (`bench/spatial_voice_probe.py` is deliberately untracked).
- Fail-open everywhere: any absence of DOA signal produces `None` and the cone abstains; behavior with `doa.enabled: false` must be byte-identical to Director-10.
- Angles are circular 0–359°: distance/median/EMA must treat 359° and 1° as 2° apart, never 358°.
- Exact config values: `cone_deg: 20` (half-width), `poll_ms: 150`, `bearing_ema_alpha: 0.3`. Strict-bool for `enabled` (`is True` — only a real boolean `true` enables).
- The reducer stays pure (no I/O, no printing) and must NOT import anything that imports pyusb — circular math lives in dependency-free `core/audio/doa_math.py`.
- `DOAANGLE` is the bearing of the CURRENT dominant sound: segments must be scored over their own time span from buffered samples, never by a read at decision time.
- Suite baseline: 726 passed / 2 skipped. Every task ends with the full suite green.

---

### Task 1: Circular angle math (`core/audio/doa_math.py`)

**Files:**
- Create: `core/audio/doa_math.py`
- Test: `tests/core/test_doa_math.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces: `circular_distance(a: float, b: float) -> float`; `circular_median(angles: iterable) -> float` (raises `ValueError` on empty); `circular_ema(current: float, sample: float, alpha: float) -> float`. Tasks 2 and 3 import these exact names from `core.audio.doa_math`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_doa_math.py
"""Circular angle math (Director-11): the 359/1 wraparound is 2 degrees."""

import pytest

from core.audio.doa_math import circular_distance, circular_median, circular_ema


def test_distance_plain():
    assert circular_distance(90.0, 110.0) == 20.0


def test_distance_wraps_the_short_way():
    assert circular_distance(359.0, 1.0) == 2.0
    assert circular_distance(1.0, 359.0) == 2.0


def test_distance_max_is_180():
    assert circular_distance(0.0, 180.0) == 180.0


def test_median_plain():
    assert circular_median([10.0, 20.0, 30.0]) == 20.0


def test_median_across_wraparound():
    # Naive sorting would put 358 last and pick a garbage middle; circular
    # median must land on the cluster center 0.
    assert circular_median([358.0, 0.0, 2.0]) == 0.0


def test_median_ignores_nothing_and_is_a_sample():
    # Definition: the SAMPLE angle minimizing summed circular distance.
    assert circular_median([97.0]) == 97.0
    assert circular_median([90.0, 100.0]) in (90.0, 100.0)


def test_median_empty_raises():
    with pytest.raises(ValueError):
        circular_median([])


def test_ema_plain():
    assert circular_ema(90.0, 100.0, 0.3) == pytest.approx(93.0)


def test_ema_shortest_arc_across_zero():
    # 350 -> 10 is +20 the short way; half-step lands on 0, not 180.
    assert circular_ema(350.0, 10.0, 0.5) == pytest.approx(0.0)
    assert circular_ema(10.0, 350.0, 0.5) == pytest.approx(0.0)


def test_ema_result_wrapped_to_0_360():
    assert 0.0 <= circular_ema(355.0, 15.0, 0.9) < 360.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/core/test_doa_math.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.audio.doa_math'`

- [ ] **Step 3: Write the implementation**

```python
# core/audio/doa_math.py
"""Circular (0-360 degree) angle math for the DOA cone vote (Director-11).

Dependency-free on purpose: the pure reducer imports these without pulling
pyusb, and DoaTracker uses the same definitions so "distance" means one
thing everywhere."""


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def circular_median(angles) -> float:
    """The sample angle minimizing summed circular distance to all samples.
    O(n^2), fine for the tracker's <=600-sample buffer. Raises ValueError on
    empty input — callers translate "no samples" to None before calling."""
    angles = list(angles)
    if not angles:
        raise ValueError("circular_median of empty sequence")
    return min(angles, key=lambda c: sum(circular_distance(c, a) for a in angles))


def circular_ema(current: float, sample: float, alpha: float) -> float:
    """EMA along the shortest arc from current toward sample, wrapped to [0, 360)."""
    delta = ((sample - current) + 180.0) % 360.0 - 180.0
    return (current + alpha * delta) % 360.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/core/test_doa_math.py -v`
Expected: 10 passed

- [ ] **Step 5: Full suite, then commit**

Run: `python3 -m pytest tests/ -q`
Expected: 736 passed, 2 skipped

```bash
git add core/audio/doa_math.py tests/core/test_doa_math.py
git commit -m "feat(director-11): circular angle math (distance, median, EMA)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: DoaTracker (`core/audio/doa_tracker.py`)

**Files:**
- Create: `core/audio/doa_tracker.py`
- Test: `tests/core/test_doa_tracker.py`

**Interfaces:**
- Consumes: `core.audio.respeaker.find` / `read_param` (existing), `core.audio.doa_math.circular_median` (Task 1).
- Produces: class `DoaTracker(poll_s=0.15, maxlen=600, reader=None, finder=None, clock=time.monotonic)` with `start() -> None`, `stop() -> None`, property `available: bool`, `sample_once() -> None`, `latest() -> tuple[float, float, int] | None` (`(t, angle_deg, speech_flag)`), `median_between(t0: float, t1: float) -> float | None`. Tasks 4 and 5 rely on exactly `latest()` and `median_between()` returning `None` whenever the tracker is disabled/unavailable/empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_doa_tracker.py
"""DoaTracker (Director-11): continuous DOA sampling with a hard fail-open
contract — any USB error latches the tracker unavailable and every read
returns None (the cone gate abstains; the kiosk degrades to D10 behavior)."""

import time

import pytest

from core.audio.doa_tracker import DoaTracker


class _Reader:
    """Scripted read_param: returns queued (DOAANGLE, SPEECHDETECTED) pairs;
    a pair of Exception instances raises instead."""
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __call__(self, dev, name):
        if name == "DOAANGLE":
            self._current = self._pairs.pop(0)
            val = self._current[0]
        else:
            val = self._current[1]
        if isinstance(val, Exception):
            raise val
        return val


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _tracker(pairs, clock=None):
    t = DoaTracker(reader=_Reader(pairs), finder=lambda: object(),
                   clock=clock or _Clock())
    return t


def test_start_probe_failure_latches_unavailable():
    t = DoaTracker(reader=_Reader([(RuntimeError("errno 13"), 0)]),
                   finder=lambda: object())
    t.start()
    assert t.available is False
    assert t.latest() is None
    assert t.median_between(0.0, 99.0) is None
    t.stop()   # must be safe even though no thread ever started


def test_missing_device_is_unavailable():
    t = DoaTracker(reader=_Reader([]), finder=lambda: None)
    t.start()
    assert t.available is False
    t.stop()


def test_sample_once_appends_and_latest_reads():
    clock = _Clock(5.0)
    t = _tracker([(97, 1), (140, 0)], clock)
    t._dev, t._available = object(), True      # bypass start(): no thread, no probe read
    t.sample_once()
    assert t.latest() == (5.0, 97.0, 1)
    clock.t = 6.0
    t.sample_once()
    assert t.latest() == (6.0, 140.0, 0)


def test_median_between_filters_time_and_speech_flag():
    clock = _Clock()
    t = _tracker([(90, 1), (200, 0), (100, 1), (95, 1)], clock)
    t._dev, t._available = object(), True
    for ts in (1.0, 2.0, 3.0, 4.0):
        clock.t = ts
        t.sample_once()
    # window [2.0, 4.0]: samples 200(speech=0, dropped), 100, 95 -> median 100 or 95
    assert t.median_between(2.0, 4.0) in (95.0, 100.0)
    # window [0.5, 1.5]: only the 90/speech=1 sample
    assert t.median_between(0.5, 1.5) == 90.0
    # window with no qualifying samples
    assert t.median_between(10.0, 20.0) is None


def test_read_error_latches_unavailable_forever():
    clock = _Clock()
    t = _tracker([(90, 1), (RuntimeError("unplugged"), 0)], clock)
    t._dev, t._available = object(), True
    clock.t = 1.0
    t.sample_once()
    assert t.available is True and t.latest() is not None
    clock.t = 2.0
    t.sample_once()                            # raises inside -> latch
    assert t.available is False
    assert t.latest() is None                  # even though a sample exists
    assert t.median_between(0.0, 9.0) is None


def test_thread_lifecycle_samples_and_stops():
    pairs = [(97, 1)] * 50                     # probe + plenty of polls
    t = DoaTracker(poll_s=0.02, reader=_Reader(pairs), finder=lambda: object())
    t.start()
    assert t.available is True
    time.sleep(0.1)
    t.stop()
    assert t.latest() is not None              # the thread actually sampled
    n = len(t._samples)
    time.sleep(0.05)
    assert len(t._samples) == n                # and actually stopped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/core/test_doa_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.audio.doa_tracker'`

- [ ] **Step 3: Write the implementation**

```python
# core/audio/doa_tracker.py
"""DoaTracker — continuous DOA sampling over the ReSpeaker USB control path.

DOAANGLE is the bearing of the CURRENT dominant sound (spike 2026-07-06), so
direction must be sampled WHILE speech happens and segments scored over their
own time span — a read at decision time sees whatever is loud NOW. A daemon
thread polls DOAANGLE + SPEECHDETECTED every poll_s into a bounded buffer;
readers take the circular median of speech-flagged samples in a window.

Fail-open contract (Director-11 spec s6): ANY USB error — at the start()
probe or mid-session — latches the tracker unavailable (logged once to
stderr) and every read returns None from then on; the cone gate abstains and
the kiosk degrades to Director-10 behavior. No retry: a dead control path
stays dead for the process; the next startup's probe reports it.

Process-lifetime, owned by kiosk.py: the owner bearing is calibrated from
the wake utterance, so the tracker must already be sampling before any
session exists."""

import sys
import threading
import time
from collections import deque

from core.audio import respeaker
from core.audio.doa_math import circular_median


class DoaTracker:
    def __init__(self, poll_s: float = 0.15, maxlen: int = 600,
                 reader=None, finder=None, clock=time.monotonic):
        self._poll_s = poll_s
        self._samples = deque(maxlen=maxlen)   # (t, angle_deg, speech_flag)
        self._reader = reader or respeaker.read_param
        self._finder = finder or respeaker.find
        self._clock = clock
        self._lock = threading.Lock()
        self._dev = None
        self._available = False
        self._running = False
        self._thread = None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        try:
            self._dev = self._finder()
            if self._dev is None:
                raise RuntimeError("ReSpeaker not found on USB (2886:0018)")
            self._reader(self._dev, "DOAANGLE")            # probe read
        except Exception as e:
            print(f"[doa] unavailable ({e}) — cone gate will abstain",
                  file=sys.stderr, flush=True)
            self._available = False
            return
        self._available = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="doa-tracker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            self.sample_once()
            time.sleep(self._poll_s)

    def sample_once(self) -> None:
        if not self._available:
            return
        try:
            angle = float(self._reader(self._dev, "DOAANGLE"))
            speech = int(self._reader(self._dev, "SPEECHDETECTED"))
        except Exception as e:
            print(f"[doa] read failed ({e}) — latching unavailable, "
                  "cone gate will abstain", file=sys.stderr, flush=True)
            self._available = False
            self._running = False
            return
        with self._lock:
            self._samples.append((self._clock(), angle, speech))

    def latest(self):
        """Newest (t, angle_deg, speech_flag) sample, or None."""
        if not self._available:
            return None
        with self._lock:
            return self._samples[-1] if self._samples else None

    def median_between(self, t0: float, t1: float):
        """Circular median of speech-flagged angles in [t0, t1], or None."""
        if not self._available:
            return None
        with self._lock:
            angles = [a for (t, a, s) in self._samples if t0 <= t <= t1 and s]
        return circular_median(angles) if angles else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/core/test_doa_tracker.py -v`
Expected: 6 passed

- [ ] **Step 5: Full suite, then commit**

Run: `python3 -m pytest tests/ -q`
Expected: 742 passed, 2 skipped

```bash
git add core/audio/doa_tracker.py tests/core/test_doa_tracker.py
git commit -m "feat(director-11): DoaTracker — sampled DOA with fail-open latch

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Cone vote in the reducer (+ events, config, context, runtime DIAG)

**Files:**
- Modify: `modes/director/events.py` (three dataclasses gain `doa_angle`)
- Modify: `modes/director/config.py` (two new frozen fields)
- Modify: `modes/director/context.py` (`owner_bearing` field + `new_context` param)
- Modify: `modes/director/director.py` (`owner_bearing` passthrough)
- Modify: `modes/director/reducer.py` (`in_owner_cone`, `REJECT_OUT_OF_CONE`, three gate points, bearing EMA)
- Modify: `modes/director/assembly.py:152-172` (`_director_config_from` maps the two new keys)
- Modify: `modes/director/runtime.py` (DIAG line on bearing update)
- Test: `tests/director/test_reducer_doa_cone.py` (new)

**Interfaces:**
- Consumes: `circular_distance`, `circular_ema` from `core.audio.doa_math` (Task 1).
- Produces: `in_owner_cone(ctx, doa_angle) -> bool | None` in `reducer.py`; `TurnVerdict.REJECT_OUT_OF_CONE` with value `"out_of_cone"`; `DirectorConfig.doa_cone_deg: float = 20.0` and `.doa_bearing_ema_alpha: float = 0.3`; `Context.owner_bearing: Optional[float] = None`; `new_context(..., owner_bearing=None)` and `Director(..., owner_bearing=None)`; `SegmentEndpointed.doa_angle`, `InterjectionSegment.doa_angle`, `NearFieldOnset.doa_angle` all `float | None = None`. Tasks 4–5 rely on all of these names exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/director/test_reducer_doa_cone.py
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
```

Also extend the existing config-mapping coverage — append to `tests/director/test_config_reject_bystanders.py` (it already imports `_director_config_from`; if it does not, import it there the same way its existing tests do):

```python
def test_doa_keys_map_from_turn_gate_doa():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({"turn_gate": {"doa": {
        "cone_deg": 25, "bearing_ema_alpha": 0.5}}})
    assert cfg.doa_cone_deg == 25.0
    assert cfg.doa_bearing_ema_alpha == 0.5


def test_doa_keys_default_when_absent():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({})
    assert cfg.doa_cone_deg == 20.0
    assert cfg.doa_bearing_ema_alpha == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_reducer_doa_cone.py -v`
Expected: FAIL — `ImportError: cannot import name 'in_owner_cone'` (and/or `TypeError: unexpected keyword argument 'doa_angle'`)

- [ ] **Step 3: Implement**

`modes/director/events.py` — add to each of the three dataclasses (as the LAST field; all three additions default `None` so every existing construction keeps working):

```python
# in SegmentEndpointed, after `seq: int = 0`:
    doa_angle: "float | None" = None   # circular median over the segment span (Director-11); None = no signal

# in NearFieldOnset, after `is_target: bool`:
    doa_angle: "float | None" = None   # latest speech-flagged sample at onset (Director-11)

# in InterjectionSegment, after `seq: int = 0`:
    doa_angle: "float | None" = None   # circular median over the segment span (Director-11)
```

`modes/director/config.py` — append to `DirectorConfig`:

```python
    # Director-11: DOA cone vote. Direction is a fourth gate; a None angle
    # always abstains, so there is no enabled flag here — kiosk.py gates
    # tracker construction, and no tracker means every doa_angle is None.
    doa_cone_deg: float = 20.0           # cone half-width, degrees (spike-validated)
    doa_bearing_ema_alpha: float = 0.3   # served-turn bearing tracking rate
```

`modes/director/context.py` — add the field and thread it through:

```python
# in Context, after `miss_streak: int = 0`:
    owner_bearing: Optional[float] = None   # calibrated owner DOA (Director-11); None = cone abstains

# new_context gains the parameter:
def new_context(cfg: DirectorConfig, conversation: ConversationManager,
                now: float, proximity_rms: float,
                owner_bearing: Optional[float] = None) -> Context:
    return Context(
        cfg=cfg, conversation=conversation, proximity_rms=proximity_rms,
        now=now, started_at=now, last_speech_at=now,
        presence_since=now, owner_bearing=owner_bearing,
    )
```

`modes/director/director.py` — passthrough:

```python
class Director:
    def __init__(self, cfg: DirectorConfig, conversation: ConversationManager,
                 now: float, proximity_rms: float, owner_bearing=None):
        self.ctx: Context = new_context(cfg, conversation, now, proximity_rms,
                                        owner_bearing=owner_bearing)
        self.state: State = State.LISTENING
```

`modes/director/reducer.py` — four changes:

(a) import at the top, alongside the existing imports:

```python
from core.audio.doa_math import circular_distance, circular_ema
```

(b) the pure vote, placed just above `classify_new_turn`:

```python
def in_owner_cone(ctx: Context, doa_angle):
    """Direction vote (Director-11): None = abstain (no signal — fail open,
    the other gates decide), True = within the cone, False = out of cone."""
    if doa_angle is None or ctx.owner_bearing is None:
        return None
    return circular_distance(doa_angle, ctx.owner_bearing) <= ctx.cfg.doa_cone_deg
```

(c) the enum member and the new-turn branch. In `TurnVerdict` add:

```python
    REJECT_OUT_OF_CONE = "out_of_cone"      # DOA outside the owner cone (Director-11)
```

In `classify_new_turn`, insert between the proximity check and the presence check (spec s4.3: after proximity):

```python
    if ev.rms < ctx.proximity_rms:
        return TurnVerdict.REJECT_TOO_QUIET
    if in_owner_cone(ctx, ev.doa_angle) is False:
        return TurnVerdict.REJECT_OUT_OF_CONE
    if ctx.presence_status is PresenceStatus.ABSENT:
        return TurnVerdict.REJECT_OWNER_ABSENT
```

In `gate_diag_reason`, add the new verdict to the reject tuple:

```python
    if v in (TurnVerdict.REJECT_NOT_TARGET, TurnVerdict.REJECT_TOO_QUIET,
             TurnVerdict.REJECT_OWNER_ABSENT, TurnVerdict.REJECT_SPEAKER_UNVERIFIED,
             TurnVerdict.REJECT_OUT_OF_CONE):
        return v.value
```

(d) the three gate wirings. Onset (in `reduce`, the `NearFieldOnset` arm):

```python
    if isinstance(event, E.NearFieldOnset) and state is State.SPEAKING:
        if (event.is_target and event.rms >= ctx.proximity_rms
                and in_owner_cone(ctx, event.doa_angle) is not False):
            ctx.ducked = True
            return State.EVALUATING, [C.Duck(ctx.cfg.duck_level)]
        return State.SPEAKING, []
```

Interjection ladder (in `_on_interjection_segment`, a new rung right after the proximity pre-gate):

```python
    if ev.rms < ctx.proximity_rms:                       # proximity pre-gate
        return _restore_speaking(ctx)
    if in_owner_cone(ctx, ev.doa_angle) is False:        # wrong direction (Director-11)
        return _restore_speaking(ctx)
```

Bearing EMA (in `_on_user_segment`, inside the ACCEPT branch — served, in-cone turns are the ONLY thing that moves the bearing, so a bystander can never drag the cone):

```python
        if v is TurnVerdict.ACCEPT:
            cmds.append(C.TranscribeUserTurn(seq=ev.seq))
            if in_owner_cone(ctx, ev.doa_angle) is True:
                ctx.owner_bearing = circular_ema(
                    ctx.owner_bearing, ev.doa_angle,
                    ctx.cfg.doa_bearing_ema_alpha)
```

`modes/director/assembly.py` — in `_director_config_from`, add before the closing paren (read `turn_gate.doa.*`; `enabled`/`poll_ms` are deliberately NOT here — kiosk.py consumes them):

```python
        doa_cone_deg=float(tb_cfg.get("turn_gate", {}).get("doa", {})
                                 .get("cone_deg", 20.0)),
        doa_bearing_ema_alpha=float(tb_cfg.get("turn_gate", {}).get("doa", {})
                                          .get("bearing_ema_alpha", 0.3)),
```

`modes/director/runtime.py` — DIAG visibility for bearing tracking (the reducer is pure and never prints). In `run_async`, around the dispatch:

```python
                bearing_before = self._director.ctx.owner_bearing
                commands = self._director.dispatch(event)
                if _DIAG and self._director.ctx.owner_bearing != bearing_before:
                    _diag(f"owner bearing {bearing_before:.1f}° -> "
                          f"{self._director.ctx.owner_bearing:.1f}°")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_reducer_doa_cone.py tests/director/test_config_reject_bystanders.py -v`
Expected: all pass (14 new in test_reducer_doa_cone.py + 2 new mapping tests)

- [ ] **Step 5: Full suite (regression: nothing existing may break — every new field defaults to abstain), then commit**

Run: `python3 -m pytest tests/ -q`
Expected: 758 passed, 2 skipped

```bash
git add modes/director/events.py modes/director/config.py modes/director/context.py \
        modes/director/director.py modes/director/reducer.py modes/director/assembly.py \
        modes/director/runtime.py tests/director/test_reducer_doa_cone.py \
        tests/director/test_config_reject_bystanders.py
git commit -m "feat(director-11): in_owner_cone vote at all three gate points

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Ingestion stamps doa_angle on events

**Files:**
- Modify: `modes/director/workers/ingestion.py`
- Test: `tests/director/test_ingestion_worker.py` (extend — reuse its `FakeMic`/`FakeVad`/`_seg`/`make_worker`/`_run_briefly` fixtures)

**Interfaces:**
- Consumes: `DoaTracker.latest()` / `.median_between(t0, t1)` contract from Task 2 (only the contract — tests use a fake); `doa_angle` event fields from Task 3.
- Produces: `IngestionWorker(..., doa_tracker=None)` keyword parameter — Task 5's assembly wiring passes the real tracker here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/director/test_ingestion_worker.py`, and extend its `make_worker` helper with a passthrough parameter — change the signature line to:

```python
def make_worker(mic, vad, state, turn_prob=0.9, embedder_score=0.9, pvad=None,
                safety=None, doa_tracker=None):
```

and add `doa_tracker=doa_tracker,` to the `IngestionWorker(...)` call inside it. Then append:

```python
class _FakeDoa:
    """Records the median_between window; returns a scripted angle."""
    def __init__(self, median=97.0, latest=None):
        self._median = median
        self._latest = latest          # (t, angle, speech) or None
        self.windows = []

    def latest(self):
        return self._latest

    def median_between(self, t0, t1):
        self.windows.append((t0, t1))
        return self._median


@pytest.mark.asyncio
async def test_segment_carries_doa_median_over_its_own_span():
    seg = _seg(duration_ms=900.0)
    doa = _FakeDoa(median=97.0)
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.LISTENING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1 and seps[0].doa_angle == 97.0
    (t0, t1), = doa.windows
    assert (t1 - t0) == pytest.approx(0.9, abs=0.05)   # the segment's own span


@pytest.mark.asyncio
async def test_interjection_carries_doa_median():
    seg = _seg(duration_ms=900.0)
    doa = _FakeDoa(median=193.0)
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.EVALUATING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    inters = [e for e in evs if isinstance(e, E.InterjectionSegment)]
    assert len(inters) == 1 and inters[0].doa_angle == 193.0


@pytest.mark.asyncio
async def test_onset_carries_latest_speech_flagged_angle():
    chunk = np.full(512, 0.5, dtype=np.float32)
    doa = _FakeDoa(latest=(1.0, 140.0, 1))
    w, bus, stt = make_worker(FakeMic([chunk]), FakeVad([[]], is_speaking=True),
                              State.SPEAKING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].doa_angle == 140.0


@pytest.mark.asyncio
async def test_onset_nonspeech_latest_sample_stamps_none():
    chunk = np.full(512, 0.5, dtype=np.float32)
    doa = _FakeDoa(latest=(1.0, 140.0, 0))              # speech_flag = 0
    w, bus, stt = make_worker(FakeMic([chunk]), FakeVad([[]], is_speaking=True),
                              State.SPEAKING, doa_tracker=doa)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].doa_angle is None


@pytest.mark.asyncio
async def test_no_tracker_stamps_none_everywhere():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]], is_speaking=True),
                              State.LISTENING, doa_tracker=None)
    await _run_briefly(w)
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1 and seps[0].doa_angle is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_ingestion_worker.py -v -k doa`
Expected: FAIL — `TypeError: IngestionWorker.__init__() got an unexpected keyword argument 'doa_tracker'`

- [ ] **Step 3: Implement in `modes/director/workers/ingestion.py`**

Add `import time` to the imports. Constructor: add `doa_tracker=None` after `safety_worker=None` and store:

```python
        self._doa = doa_tracker            # Director-11 DoaTracker; None -> doa_angle=None
```

Add a helper next to `_target_from`:

```python
    def _onset_doa(self):
        """Latest DOA sample's angle IF it is speech-flagged, else None. The
        onset is instantaneous, so only a current speech bearing counts."""
        if self._doa is None:
            return None
        sample = self._doa.latest()
        if sample is None or not sample[2]:
            return None
        return sample[1]
```

In `_maybe_onset`, stamp the onset event:

```python
        if rms >= self._proximity_rms:
            self._ducked_onset = True
            is_target, _ = self._target_from(self._chunk_frames)
            await self._bus.emit(E.NearFieldOnset(rms=rms, is_target=is_target,
                                                  doa_angle=self._onset_doa()))
```

In `_on_segment`, compute the segment's span FIRST (before any await — the
endpoint-prob executor call would otherwise skew t_end) and stamp both events:

```python
    async def _on_segment(self, seg, state: State) -> None:
        # DOAANGLE tracks the CURRENT dominant sound, so the segment is scored
        # over its own span from buffered samples: the VAD closed it just now,
        # so [now - duration, now] is the voiced run (Director-11).
        doa_angle = None
        if self._doa is not None:
            t_end = time.monotonic()
            doa_angle = self._doa.median_between(
                t_end - seg.duration_ms / 1000.0, t_end)
        rms = _rms(seg.audio)
```

and add `doa_angle=doa_angle,` to both the `E.SegmentEndpointed(...)` and
`E.InterjectionSegment(...)` constructions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_ingestion_worker.py -v`
Expected: all pass (5 new + all pre-existing)

- [ ] **Step 5: Full suite, then commit**

Run: `python3 -m pytest tests/ -q`
Expected: 763 passed, 2 skipped

```bash
git add modes/director/workers/ingestion.py tests/director/test_ingestion_worker.py
git commit -m "feat(director-11): ingestion stamps per-segment DOA medians on events

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wiring — assembly calibration, kiosk lifecycle + probe, config.yaml

**Files:**
- Modify: `modes/director/assembly.py` (bearing calibration; `build_director_runtime` gains `doa_tracker`; passes to `Director` and `IngestionWorker`)
- Modify: `kiosk.py` (`_assert_array_startup` DOA probe; tracker construction/start/stop; `_LazyDirectorRuntime` passthrough)
- Modify: `config.yaml` (the `turn_gate.doa` block)
- Test: `tests/director/test_assembly_doa.py` (new), `tests/director/test_kiosk_entrypoint.py` (extend)

**Interfaces:**
- Consumes: `DoaTracker` (Task 2), `Director(..., owner_bearing=...)` (Task 3), `IngestionWorker(..., doa_tracker=...)` (Task 4), `respeaker.read_param(dev, "DOAANGLE")` (existing).
- Produces: `_calibrate_owner_bearing(doa_tracker, first_segment) -> float | None` in assembly; `build_director_runtime(..., doa_tracker=None)`; `_LazyDirectorRuntime(config, stt, llm, tts, player, logger, doa_tracker=None)`.

- [ ] **Step 1: Write the failing assembly tests**

```python
# tests/director/test_assembly_doa.py
"""Director-11 wiring: the owner bearing is calibrated from a lookback window
over the wake utterance (the handoff's first_segment has no timestamps), and
no tracker -> None -> the cone abstains all session."""

import time
from types import SimpleNamespace

import pytest

from modes.director.assembly import _calibrate_owner_bearing


class _FakeDoa:
    def __init__(self, median):
        self._median = median
        self.windows = []

    def median_between(self, t0, t1):
        self.windows.append((t0, t1))
        return self._median


def _first_segment(duration_ms=904.0):
    return SimpleNamespace(duration_ms=duration_ms)


def test_no_tracker_returns_none():
    assert _calibrate_owner_bearing(None, _first_segment()) is None


def test_bearing_is_median_over_wake_lookback_window():
    doa = _FakeDoa(median=97.0)
    before = time.monotonic()
    bearing = _calibrate_owner_bearing(doa, _first_segment(duration_ms=1500.0))
    after = time.monotonic()
    assert bearing == 97.0
    (t0, t1), = doa.windows
    # window ends "now" and reaches back max(dur_s, 1.0) + 1.0 = 2.5s
    assert before <= t1 <= after
    assert (t1 - t0) == pytest.approx(2.5, abs=0.01)


def test_short_seed_still_gets_a_full_lookback():
    doa = _FakeDoa(median=45.0)
    _calibrate_owner_bearing(doa, _first_segment(duration_ms=200.0))
    (t0, t1), = doa.windows
    assert (t1 - t0) == pytest.approx(2.0, abs=0.01)   # max(0.2, 1.0) + 1.0


def test_tracker_with_no_samples_returns_none():
    class _Empty:
        def median_between(self, t0, t1):
            return None
    assert _calibrate_owner_bearing(_Empty(), _first_segment()) is None
```

And the failing kiosk test — append to `tests/director/test_kiosk_entrypoint.py`, following that file's existing pattern for invoking `_assert_array_startup` with monkeypatched `assembly._pipewire_sinks` and `core.audio.respeaker` (reuse its existing fixtures/helpers; the new tests only add the DOA probe dimension):

```python
def test_doa_probe_failure_warns_but_does_not_exit(monkeypatch, capsys, base_config):
    # Sink resolve + AGC fine; DOAANGLE read raises -> startup must NOT exit
    # (fail-open: the cone abstains; D10 behavior).
    base_config["kiosk"]["talkback"]["turn_gate"] = {"doa": {"enabled": True}}
    import core.audio.respeaker as respeaker

    def _read_param(dev, name):
        raise RuntimeError("errno 13")
    monkeypatch.setattr(respeaker, "read_param", _read_param)
    kiosk._assert_array_startup(base_config, _console())        # no SystemExit
    out = _console_output()
    assert "DOA unavailable" in out


def test_doa_probe_success_prints_bearing(monkeypatch, base_config):
    base_config["kiosk"]["talkback"]["turn_gate"] = {"doa": {"enabled": True}}
    import core.audio.respeaker as respeaker
    monkeypatch.setattr(respeaker, "read_param", lambda dev, name: 143
                        if name == "DOAANGLE" else 0)
    kiosk._assert_array_startup(base_config, _console())
    assert "DOA control readable" in _console_output()
```

(Adapt `base_config`, `_console`, `_console_output` to whatever the file actually names its config fixture and console capture — it already asserts on `✓`/`✗` console output for the sink and AGC paths; mirror that mechanism exactly. The AGC monkeypatching there also patches `respeaker.find`/`write_param`; keep those patches so the AGC assert still passes.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/director/test_assembly_doa.py tests/director/test_kiosk_entrypoint.py -v`
Expected: FAIL — `ImportError: cannot import name '_calibrate_owner_bearing'`; kiosk DOA tests fail on missing output.

- [ ] **Step 3: Implement**

`modes/director/assembly.py` — add below `_calibrate_proximity_rms` (mirror its DIAG style):

```python
def _calibrate_owner_bearing(doa_tracker, first_segment):
    """Owner bearing from the wake utterance (Director-11). The handoff's
    first_segment has no timestamps, so use a lookback window from 'now'
    (assembly runs within moments of the wake endpoint) generously covering
    the utterance; the tracker's speech-flag filter keeps only voiced
    samples. None (no tracker / no samples) -> the cone abstains all session."""
    if doa_tracker is None:
        return None
    dur_s = float(getattr(first_segment, "duration_ms", 0.0) or 0.0) / 1000.0
    t_end = time.monotonic()
    bearing = doa_tracker.median_between(t_end - (max(dur_s, 1.0) + 1.0), t_end)
    if os.environ.get("TVAD_DIAG"):
        shown = f"{bearing:.0f}°" if bearing is not None else "None (abstain)"
        print(f"[DIAG assembly] owner bearing: {shown}",
              file=sys.stderr, flush=True)
    return bearing
```

(Confirm `assembly.py` already imports `time`, `os`, `sys` — it uses all three for the proximity DIAG; add any that are missing.)

In `build_director_runtime`: add the keyword parameter `doa_tracker: Optional[Any] = None` after `_watchdog_tick_s`, then inside:

```python
    owner_bearing = _calibrate_owner_bearing(doa_tracker, handoff.first_segment)
    director = Director(cfg, conversation, now=now, proximity_rms=proximity_rms,
                        owner_bearing=owner_bearing)
```

and add `doa_tracker=doa_tracker,` to the `IngestionWorker(...)` construction.

`kiosk.py` — three changes:

(a) `_assert_array_startup`: extend the existing ReSpeaker try-block. Replace the current AGC block with (structure preserved — one warn-only block for the array's control path, now also probing DOA when enabled):

```python
    try:
        from core.audio import respeaker
        dev = respeaker.find()
        if dev is None:
            raise RuntimeError("ReSpeaker not found on USB (2886:0018)")
        respeaker.write_param(dev, "AGCONOFF", 0)
        console.print("[green]✓[/] ReSpeaker AGC off")
    except Exception as e:
        console.print(
            f"[yellow]![/] ReSpeaker AGC assert failed ({e}); "
            "continuing with AGC on — proximity floors will be less stable")
    # Director-11 DOA probe — warn-only (fail open: no DOA means the cone
    # gate abstains and behavior degrades to Director-10, not a dead kiosk).
    doa_cfg = tb_cfg.get("turn_gate", {}).get("doa", {})
    if doa_cfg.get("enabled", False) is True:
        try:
            from core.audio import respeaker
            dev = respeaker.find()
            if dev is None:
                raise RuntimeError("ReSpeaker not found on USB (2886:0018)")
            angle = respeaker.read_param(dev, "DOAANGLE")
            console.print(f"[green]✓[/] DOA control readable (bearing now {angle}°)")
        except Exception as e:
            console.print(f"[red]✗[/] DOA unavailable ({e}) — cone gate will abstain")
```

(b) `_LazyDirectorRuntime`: constructor gains `doa_tracker=None`, stored as `self._doa_tracker`, and `run()` passes `doa_tracker=self._doa_tracker,` into `build_director_runtime(...)`.

(c) `main()`: construct and start the tracker after `_assert_array_startup`, hand it to the runtime wrapper, stop it on the way out. `_build_runtime(config)` is called before the assert today — keep that order and attach the tracker afterward:

```python
    runtime = _build_runtime(config)
    _assert_array_startup(config, console)
    doa_tracker = None
    doa_cfg = (config["kiosk"].get("talkback", {})
                              .get("turn_gate", {}).get("doa", {}))
    if doa_cfg.get("enabled", False) is True:
        from core.audio.doa_tracker import DoaTracker
        doa_tracker = DoaTracker(poll_s=float(doa_cfg.get("poll_ms", 150)) / 1000.0)
        doa_tracker.start()          # unavailable -> logs once, reads return None
        runtime._doa_tracker = doa_tracker
```

and wrap the wake-loop section so the thread always stops (extend the existing `try/except KeyboardInterrupt` with a `finally`):

```python
    finally:
        if doa_tracker is not None:
            doa_tracker.stop()
```

`config.yaml` — inside the existing `turn_gate:` block (after `reject_bystanders: true`):

```yaml
      # Director-11: DOA cone vote — direction as a fourth gate. The tracker
      # samples DOAANGLE/SPEECHDETECTED continuously (a decision-time read
      # would see whatever is loud NOW); segments are scored by circular
      # median over their own span; the cone abstains whenever any signal is
      # missing (fail open -> Director-10 behavior). Only a real boolean
      # true enables.
      doa:
        enabled: true
        cone_deg: 20             # half-width; ±20° scored 100% on the 2026-07-06 spike
        poll_ms: 150             # firmware update cadence
        bearing_ema_alpha: 0.3   # served-turn bearing tracking rate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/director/test_assembly_doa.py tests/director/test_kiosk_entrypoint.py -v`
Expected: all pass (4 new assembly + 2 new kiosk + all pre-existing entrypoint tests)

- [ ] **Step 5: Full suite, then commit**

Run: `python3 -m pytest tests/ -q`
Expected: 769 passed, 2 skipped

```bash
git add modes/director/assembly.py kiosk.py config.yaml \
        tests/director/test_assembly_doa.py tests/director/test_kiosk_entrypoint.py
git commit -m "feat(director-11): wire DoaTracker — kiosk lifecycle, bearing calibration, config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## After the tasks

Final whole-branch review (most capable model), then the LIVE merge gate from
spec s8 (4 checks: podcast out_of_cone + no ducking; owner normal + bearing
tracks; owner barge-in over podcast; fail-open probe) — the user runs
`TVAD_DIAG=1 ./kiosk-stack.sh start` sessions and pastes logs. Then verdict
note + finishing-a-development-branch.
