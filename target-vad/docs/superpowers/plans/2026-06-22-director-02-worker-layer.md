# Director Worker Layer & Runtime (Plan 02) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Plan-01 Director **runnable on the existing models** by building the async worker layer and the single-loop runtime around the pure reducer. Define and TDD the `EventBus`, the four supervised workers (Ingestion, STT, Generation, Playback) that translate audio/STT/LLM/TTS into Plan-01 `Event`s and execute Plan-01 `Command`s, and the `DirectorRuntime` that owns one asyncio loop (mirroring `TalkbackController.run`/`_run_async`) and drives `event → dispatch → route(command)` to a `DirectorResult`. The **6 race-fixed PortAudio teardown invariants** (spec §10) are copied **verbatim** into the Playback worker and regression-tested. Everything is tested with fakes/mocks — no GPU, no real audio device, no real models.

**Architecture:** Parallelism in the workers, decision-making serialized (spec §3). One asyncio event loop runs the `Director` (Plan 01, the sole state mutator) plus the workers and the `AsyncWatchdog`. Workers communicate with the Director only through an async `EventBus` (`asyncio.Queue`): a worker `await bus.emit(event)`; the runtime `event = await bus.get()`, `cmds = director.dispatch(event)`, and routes each command back to the worker that executes it. The reducer never blocks; short synchronous model calls (ECAPA, Smart Turn) go through `run_in_executor`; the blocking mic generator and the blocking `sd.write` are bridged thread→loop. The Playback worker holds the only `OutputStream` and is the sole owner of `record_reference` + `sd.write` under one `_write_lock` — the Ingestion worker only *reads* the AEC reference ring.

**Tech Stack:** Python 3.12 (`python3`, no `python` on PATH), pytest, pytest-asyncio. Reuses verbatim: `Player` (`modes/talkback/player.py:14`), `AecProcessor` (`modes/talkback/aec.py:54`), `SentenceChunker` (`modes/talkback/chunker.py:17`), `SileroVAD`/`SpeechSegment` (`core/vad/silero_vad.py:20`/`:11`), `TtsEngine` (`modes/talkback/tts.py:13`), `LlmClient` (`modes/talkback/llm.py:13`), `MicrophoneStream` (`core/audio/mic_stream.py:11`), `TurnDetector`/`NullTurnDetector`/`SmartTurnDetector` (`modes/talkback/endpointing.py`), `EmbeddingExtractor` (`core/speaker/embedder.py:12`). Consumes the Plan-01 reducer (`modes/director/`). No new third-party dependencies.

## Global Constraints

- Target/dev box: NVIDIA DGX Spark GB10, aarch64, Python 3.12. Run tests with `python3 -m pytest`.
- **Single-mutator rule (spec §3, §11):** only `Director.dispatch()` (Plan 01) mutates FSM state/Context. Workers NEVER touch the Director's state or Context — they only `await bus.emit(event)` and execute the `Command`s the runtime hands them. A worker may *read* `director.state` (read-only) to decide which event to emit; it must never assign to it.
- **gen_id discipline (spec §11):** every generation carries the Plan-01 `Context.gen_id`. The Generation/Playback workers tag `FirstTtsFrame`/`ReplyComplete` with their `gen_id` and **drop stale work** when a newer `gen_id` supersedes them; `_play_gen` is the playback-side mirror of `gen_id` (spec §10 invariant 2).
- **Teardown invariants are sacred (spec §10):** the 6 race-fixed invariants from `controller.py` are copied **verbatim** into the Playback worker (Task 6). No refactor may split `record_reference` from `sd.write`, drop the `_write_lock`, clear `_play_future` on barge-in, or close the stream without first awaiting the drain.
- **No real I/O in tests:** every worker test uses a fake mic / fake STT / fake LLM / fake TTS / `MagicMock` output stream, mirroring `tests/kiosk/talkback/test_controller.py` and `tests/kiosk/talkback/test_playback.py`. No GPU, no `sd.OutputStream`/`sd.InputStream`, no network.
- **STT signature shim (spec §6, §9):** today `StreamingStt.transcribe_segment(audio) -> str` (a bare string, `stt.py:37-51`). Plan 04 re-backs it to return `TranscriptResult(text, mean_word_prob)`. Plan 02 defines `TranscriptResult` here and ships a `wrap_transcript()` shim that coerces a bare-`str` return into `TranscriptResult(text=<str>, mean_word_prob=1.0)`, so the worker layer composes today and swaps cleanly when Plan 04 lands.
- New runtime/worker code lives under `modes/director/` (package from Plan 01) and `modes/director/workers/`; tests under `tests/director/`.
- Reuse, do not reimplement, the Plan-01 modules (`modes/director/state.py`, `config.py`, `context.py`, `events.py`, `commands.py`, `reducer.py`, `director.py`) and the talkback assets listed in Tech Stack.

---

## File Structure

- `modes/director/bus.py` — `EventBus` (async, `asyncio.Queue`-backed): `await emit(event)`, `await get()`, `qsize()`.
- `modes/director/result.py` — `DirectorResult(reason, turns, total_duration_s)` (the session-end return value; mirrors `TalkbackResult` `handoff.py:27`).
- `modes/director/transcript.py` — `TranscriptResult(text, mean_word_prob)` dataclass + `wrap_transcript(raw)` shim (str-or-TranscriptResult → TranscriptResult).
- `modes/director/watchdog.py` — `AsyncWatchdog` for the runtime: ticks a clock and emits `Tick(now)` onto the bus, calls `on_session_end(reason)` when the loop is told to stop. (Plan 02 scope: Tick-emitter only; the non-terminal nudge extension is Plan 03 work — documented in Task 7.)
- `modes/director/workers/__init__.py` — empty package marker.
- `modes/director/workers/ingestion.py` — `IngestionWorker` (mic → AEC → Silero VAD → RMS → Smart Turn → ECAPA → emits `NearFieldOnset`/`SegmentEndpointed`/`InterjectionSegment`).
- `modes/director/workers/stt_worker.py` — `SttWorker` (executes `TranscribeUserTurn`/`TranscribeInterjection`; emits `UserTurnTranscribed`/`InterjectionTranscribed`).
- `modes/director/workers/generation.py` — `GenerationWorker` (executes `StartGeneration`/`Cut`; streams LLM→chunker→TTS→Playback; emits `FirstTtsFrame`/`ReplyComplete`).
- `modes/director/workers/playback.py` — `PlaybackWorker` (owns the `OutputStream` + `Player`; the 6 teardown invariants; executes `Duck`/`Restore`/`SpeakNudge`; provides `play()`/`drain()`/`close()` to the Generation worker).
- `modes/director/runtime.py` — `DirectorRuntime` (owns one asyncio loop; constructs Director + bus + workers + watchdog; the `event → dispatch → route` loop; returns `DirectorResult`).
- `tests/director/test_*.py` — one test module per task.

---

## Task 1: EventBus + DirectorResult + TranscriptResult shim

**Files:**
- Create: `modes/director/bus.py`, `modes/director/result.py`, `modes/director/transcript.py`
- Test: `tests/director/test_bus_and_types.py`

**Interfaces:**
- Consumes: nothing (Plan-01 events are only used as opaque payloads here).
- Produces: `EventBus` with `async emit(event) -> None`, `async get() -> Any`, `qsize() -> int`. `DirectorResult(reason: str, turns: int, total_duration_s: float)` frozen dataclass. `TranscriptResult(text: str, mean_word_prob: float)` frozen dataclass. `wrap_transcript(raw: str | TranscriptResult) -> TranscriptResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_bus_and_types.py
import asyncio

import pytest

from modes.director.bus import EventBus
from modes.director.result import DirectorResult
from modes.director.transcript import TranscriptResult, wrap_transcript


@pytest.mark.asyncio
async def test_bus_round_trips_events_in_fifo_order():
    bus = EventBus()
    await bus.emit("a")
    await bus.emit("b")
    assert bus.qsize() == 2
    assert await bus.get() == "a"
    assert await bus.get() == "b"
    assert bus.qsize() == 0


@pytest.mark.asyncio
async def test_bus_get_blocks_until_emit():
    bus = EventBus()

    async def producer():
        await asyncio.sleep(0.01)
        await bus.emit("late")

    asyncio.create_task(producer())
    # get() must wait for the producer rather than raising QueueEmpty.
    assert await asyncio.wait_for(bus.get(), timeout=1.0) == "late"


def test_director_result_is_frozen_and_carries_fields():
    r = DirectorResult(reason="silence_timeout", turns=3, total_duration_s=12.5)
    assert r.reason == "silence_timeout" and r.turns == 3 and r.total_duration_s == 12.5
    with pytest.raises(Exception):
        r.reason = "x"  # frozen


def test_transcript_result_fields_and_shim():
    tr = TranscriptResult(text="hello", mean_word_prob=0.8)
    assert tr.text == "hello" and tr.mean_word_prob == 0.8
    # The shim coerces today's bare-str return into a full TranscriptResult.
    wrapped = wrap_transcript("bare string")
    assert isinstance(wrapped, TranscriptResult)
    assert wrapped.text == "bare string" and wrapped.mean_word_prob == 1.0
    # A real TranscriptResult passes through untouched.
    assert wrap_transcript(tr) is tr
    # None / empty coerces to empty text, full confidence (never crashes).
    assert wrap_transcript(None) == TranscriptResult(text="", mean_word_prob=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_bus_and_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.bus'`

- [ ] **Step 3: Create the bus**

```python
# modes/director/bus.py
"""EventBus — the single async channel between workers and the Director.

Workers `await emit(event)`; the runtime `await get()` one event at a time and
feeds it to the sole mutator (Director.dispatch). asyncio.Queue gives FIFO
ordering and back-pressure-free hand-off on one event loop (spec section 3)."""

import asyncio
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    async def emit(self, event: Any) -> None:
        await self._queue.put(event)

    async def get(self) -> Any:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()
```

- [ ] **Step 4: Create the result type**

```python
# modes/director/result.py
"""DirectorResult — what DirectorRuntime returns at true session end.

Mirrors modes/talkback/handoff.py:27 (TalkbackResult) so the WakeGate (Plan 03)
consumes the same shape it does today."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectorResult:
    reason: str
    turns: int
    total_duration_s: float
```

- [ ] **Step 5: Create the transcript type + shim**

```python
# modes/director/transcript.py
"""TranscriptResult — the extended STT return (spec sections 6 & 9).

Today StreamingStt.transcribe_segment returns a bare str (stt.py:37-51). Plan 04
re-backs it to return per-segment confidence. wrap_transcript() bridges the gap:
a bare str becomes TranscriptResult(text, mean_word_prob=1.0) so the empty/low-
confidence RESTORE guard in the reducer composes today and tightens for free
once Plan 04 supplies real word probabilities."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    mean_word_prob: float


def wrap_transcript(raw: Optional[Union[str, TranscriptResult]]) -> TranscriptResult:
    if isinstance(raw, TranscriptResult):
        return raw
    if raw is None:
        return TranscriptResult(text="", mean_word_prob=1.0)
    return TranscriptResult(text=str(raw), mean_word_prob=1.0)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_bus_and_types.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add modes/director/bus.py modes/director/result.py modes/director/transcript.py tests/director/test_bus_and_types.py
git commit -m "feat(director): EventBus, DirectorResult, TranscriptResult shim"
```

---

## Task 2: AsyncWatchdog (Tick-emitter) for the runtime

**Files:**
- Create: `modes/director/watchdog.py`
- Test: `tests/director/test_watchdog.py`

**Interfaces:**
- Consumes: `EventBus` (Task 1), `Tick` (Plan-01 `modes/director/events.py`).
- Produces: `AsyncWatchdog(tick_s, clock, bus, on_session_end)` with `start()`, `async stop()`, and `request_stop(reason)`. Each tick: reads `clock()`, `await bus.emit(Tick(now))`. The reducer (Plan 01) decides `EndSession` from the `Tick`; when the runtime routes an `EndSession`, it calls `request_stop(reason)`, which stops the tick loop and records the reason for `DirectorResult`.

> **Plan 02 scope note (spec §5 / §10 reuse-map "AsyncWatchdog — extended NOT as-is"):** The *terminal* timeout decision lives in the Plan-01 reducer (`_on_tick` returns `EndSession` at hard/silence timeout). The *non-terminal* nudge path (`nudge_lead_s`/`on_nudge`/`is_nudged`/`mark_nudged`) is **also** in the Plan-01 reducer (it emits `SpeakNudge` directly from `_on_tick`), so this Plan-02 watchdog needs **only** to emit `Tick(now)` on a timer — the FSM does the rest. This deliberately keeps the watchdog dumb; spec §5's extended `AsyncWatchdog` contract is satisfied by the reducer + this Tick emitter together. The `on_session_end` callback is the runtime's `EndSession` handler.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_watchdog.py
import asyncio

import pytest

from modes.director.bus import EventBus
from modes.director.watchdog import AsyncWatchdog
from modes.director import events as E


@pytest.mark.asyncio
async def test_watchdog_emits_tick_with_current_clock():
    bus = EventBus()
    now = [100.0]
    wd = AsyncWatchdog(tick_s=0.005, clock=lambda: now[0], bus=bus,
                       on_session_end=lambda reason: None)
    wd.start()
    ev = await asyncio.wait_for(bus.get(), timeout=1.0)
    assert isinstance(ev, E.Tick) and ev.now == 100.0
    now[0] = 200.0
    ev2 = await asyncio.wait_for(bus.get(), timeout=1.0)
    assert ev2.now == 200.0
    await wd.stop()


@pytest.mark.asyncio
async def test_request_stop_halts_ticks_and_reports_reason():
    bus = EventBus()
    captured = []
    wd = AsyncWatchdog(tick_s=0.005, clock=lambda: 0.0, bus=bus,
                       on_session_end=lambda reason: captured.append(reason))
    wd.start()
    await asyncio.wait_for(bus.get(), timeout=1.0)   # at least one tick
    wd.request_stop("silence_timeout")
    assert captured == ["silence_timeout"]
    # After stop, no new ticks accumulate.
    await asyncio.sleep(0.02)
    drained = bus.qsize()
    await asyncio.sleep(0.02)
    assert bus.qsize() == drained                    # loop stopped, no growth
    await wd.stop()


@pytest.mark.asyncio
async def test_stop_is_safe_when_never_started():
    wd = AsyncWatchdog(tick_s=0.01, clock=lambda: 0.0, bus=EventBus(),
                       on_session_end=lambda reason: None)
    await wd.stop()   # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_watchdog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.watchdog'`

- [ ] **Step 3: Implement the watchdog**

```python
# modes/director/watchdog.py
"""AsyncWatchdog — the runtime's single timer. Emits Tick(now) onto the bus on
a fixed cadence; the Plan-01 reducer turns those ticks into EndSession/SpeakNudge
(spec section 5). This is the ONLY timeout authority (spec Req 5): the pipeline
watchdog thread is deleted in Plan 03."""

import asyncio
from typing import Callable

from modes.director.bus import EventBus
from modes.director import events as E


class AsyncWatchdog:
    def __init__(
        self,
        tick_s: float,
        clock: Callable[[], float],
        bus: EventBus,
        on_session_end: Callable[[str], None],
    ):
        self._tick_s = tick_s
        self._clock = clock
        self._bus = bus
        self._on_session_end = on_session_end
        self._task: asyncio.Task | None = None
        self._stopping = False

    def start(self) -> None:
        self._stopping = False
        self._task = asyncio.create_task(self._loop())

    def request_stop(self, reason: str) -> None:
        """Called by the runtime when the reducer emits EndSession: stop ticking
        and record the terminal reason for DirectorResult."""
        if self._stopping:
            return
        self._stopping = True
        self._on_session_end(reason)

    async def stop(self) -> None:
        self._stopping = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._tick_s)
            if self._stopping:
                return
            await self._bus.emit(E.Tick(now=self._clock()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_watchdog.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/watchdog.py tests/director/test_watchdog.py
git commit -m "feat(director): runtime AsyncWatchdog (Tick emitter; reducer owns timeout decisions)"
```

---

## Task 3: SttWorker (TranscribeUserTurn / TranscribeInterjection)

**Files:**
- Create: `modes/director/workers/__init__.py` (empty), `modes/director/workers/stt_worker.py`
- Test: `tests/director/test_stt_worker.py`

**Interfaces:**
- Consumes: `Command`s `TranscribeUserTurn`/`TranscribeInterjection`; the last captured audio (held by the worker, set by the Ingestion worker via `set_pending_user_audio`/`set_pending_interjection_audio`); `StreamingStt.transcribe_segment(audio)` (today → `str`, Plan 04 → `TranscriptResult`).
- Produces: emits `UserTurnTranscribed(text, mean_word_prob)` / `InterjectionTranscribed(text, mean_word_prob)`; exposes `async execute(command) -> None`.

> The Director decides *when* to transcribe (`TranscribeUserTurn`/`TranscribeInterjection` carry no audio — Plan 01 `commands.py:342,346`). The audio to transcribe is whatever the Ingestion worker most recently captured for that purpose; the SttWorker holds those buffers, set by the Ingestion worker the moment it emits the matching segment event. This keeps audio off the bus (events stay small/loggable) while preserving the command→audio pairing.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_stt_worker.py
import numpy as np
import pytest

from modes.director.bus import EventBus
from modes.director.transcript import TranscriptResult
from modes.director.workers.stt_worker import SttWorker
from modes.director import events as E
from modes.director import commands as C


class FakeStt:
    """Mirrors today's bare-str transcribe_segment; Plan 04 returns TranscriptResult."""
    def __init__(self, returns):
        self._returns = returns
        self.calls = []

    async def transcribe_segment(self, audio):
        self.calls.append(audio)
        return self._returns


@pytest.mark.asyncio
async def test_transcribe_user_turn_emits_user_event_via_shim():
    bus = EventBus()
    stt = FakeStt("tell me a story")          # bare str -> shim => prob 1.0
    w = SttWorker(stt, bus)
    audio = np.ones(16000, dtype=np.float32)
    w.set_pending_user_audio(audio)
    await w.execute(C.TranscribeUserTurn())
    ev = await bus.get()
    assert isinstance(ev, E.UserTurnTranscribed)
    assert ev.text == "tell me a story" and ev.mean_word_prob == 1.0
    assert stt.calls and stt.calls[0] is audio


@pytest.mark.asyncio
async def test_transcribe_interjection_passes_real_confidence_through():
    bus = EventBus()
    stt = FakeStt(TranscriptResult(text="wait why", mean_word_prob=0.42))
    w = SttWorker(stt, bus)
    w.set_pending_interjection_audio(np.ones(8000, dtype=np.float32))
    await w.execute(C.TranscribeInterjection())
    ev = await bus.get()
    assert isinstance(ev, E.InterjectionTranscribed)
    assert ev.text == "wait why" and ev.mean_word_prob == 0.42


@pytest.mark.asyncio
async def test_missing_audio_emits_empty_low_noop_transcript():
    """If no audio was staged, emit an empty transcript (reducer RESTOREs/keeps
    listening) rather than crashing the worker."""
    bus = EventBus()
    w = SttWorker(FakeStt("ignored"), bus)
    await w.execute(C.TranscribeUserTurn())          # no set_pending_user_audio
    ev = await bus.get()
    assert isinstance(ev, E.UserTurnTranscribed) and ev.text == ""


@pytest.mark.asyncio
async def test_unknown_command_is_ignored():
    bus = EventBus()
    w = SttWorker(FakeStt("x"), bus)
    await w.execute(C.Restore())     # not an STT command
    assert bus.qsize() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_stt_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.workers'`

- [ ] **Step 3: Implement the SttWorker**

```python
# modes/director/workers/__init__.py
```
```python
# modes/director/workers/stt_worker.py
"""SttWorker — executes the Director's transcription commands.

TranscribeUserTurn / TranscribeInterjection carry no audio (the reducer is pure):
the audio to transcribe is whatever the Ingestion worker last staged here for
that purpose. The worker runs StreamingStt.transcribe_segment and emits the
matching *Transcribed event, coercing today's bare-str return into a
TranscriptResult via wrap_transcript (spec sections 6 & 9). Plan 04 swaps the
engine internals for real per-word confidence with no change here."""

import numpy as np

from modes.director.bus import EventBus
from modes.director.transcript import wrap_transcript
from modes.director import events as E
from modes.director import commands as C


class SttWorker:
    def __init__(self, stt, bus: EventBus):
        self._stt = stt
        self._bus = bus
        self._pending_user_audio = None
        self._pending_interjection_audio = None

    def set_pending_user_audio(self, audio: np.ndarray) -> None:
        self._pending_user_audio = audio

    def set_pending_interjection_audio(self, audio: np.ndarray) -> None:
        self._pending_interjection_audio = audio

    async def execute(self, command) -> None:
        if isinstance(command, C.TranscribeUserTurn):
            audio = self._pending_user_audio
            self._pending_user_audio = None
            tr = await self._transcribe(audio)
            await self._bus.emit(E.UserTurnTranscribed(text=tr.text,
                                                       mean_word_prob=tr.mean_word_prob))
        elif isinstance(command, C.TranscribeInterjection):
            audio = self._pending_interjection_audio
            self._pending_interjection_audio = None
            tr = await self._transcribe(audio)
            await self._bus.emit(E.InterjectionTranscribed(text=tr.text,
                                                          mean_word_prob=tr.mean_word_prob))

    async def _transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return wrap_transcript("")        # empty -> reducer keeps listening / RESTOREs
        raw = await self._stt.transcribe_segment(audio)
        return wrap_transcript(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_stt_worker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/workers/__init__.py modes/director/workers/stt_worker.py tests/director/test_stt_worker.py
git commit -m "feat(director): SttWorker executes Transcribe* commands with TranscriptResult shim"
```

---

## Task 4: PlaybackWorker — race-fixed teardown invariants (spec §10) + Duck/Restore/Nudge

**Files:**
- Create: `modes/director/workers/playback.py`
- Test: `tests/director/test_playback_worker.py`

**Interfaces:**
- Consumes: `Command`s `Duck(level)`/`Restore()`/`SpeakNudge()`; a `TtsEngine` (for the nudge); a `Player` (AEC reference ring). Owns the `sd.OutputStream` (injected/created at `open()`).
- Produces: `async play(audio, gen_id)` (write one utterance frame-by-frame, gen-gated, reference recorded under lock); `async drain()` (`_drain_playback` semantics); `close()` (`_close_out_stream` semantics); `record_reference` co-located with the write under one `_write_lock`; `async execute(command)`; `set_gen(gen_id)`; `gain` property. The 6 invariants are copied **verbatim** from `controller.py`.

> **Spec §10 invariants implemented here (cite controller.py line numbers):**
> 1. `_write_lock` around every `sd.write` (controller.py:221-228) **and** around `stop()/close()` (controller.py:237-244).
> 2. `_play_gen` checked before each frame (controller.py:216) **and** again inside the lock (controller.py:222); bumped on drain/close.
> 3. `record_reference` co-located with the write under **one** lock (controller.py:224-227) — the Ingestion worker only READS the ring via `get_reference_frame`.
> 4. `drain()` bumps `_play_gen` then `await asyncio.shield(_play_future)` before close (controller.py:189-203, 439).
> 5. Two-path teardown: async `drain()` first, then synchronous idempotent `close()` backstop (controller.py:231-244, 254).
> 6. `_play_future` is **not** cleared on barge-in (controller.py:181-187 docstring) — it survives so `drain()` can await the real thread exit.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_playback_worker.py
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.director.workers.playback import PlaybackWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.talkback.player import Player
from modes.director import commands as C


def make_worker(gain=1.0):
    w = PlaybackWorker(
        tts=MagicMock(),
        player=Player(sample_rate=16000, ring_buffer_seconds=2.0),
        cfg=DirectorConfig(),
        bus=EventBus(),
    )
    w._out_stream = MagicMock()       # fake OutputStream (no real device)
    w._running = True
    w._gain = gain
    w._play_gen = 0
    return w


class TestPlayAudioInvariants:
    def test_writes_frames_to_output_stream(self):
        w = make_worker()
        w._play_audio(np.ones(960, dtype=np.float32), 0)
        assert w._out_stream.write.call_count == 2      # 960 / 480 (invariant: framed)

    def test_applies_gain(self):
        w = make_worker(gain=0.15)
        w._play_audio(np.ones(480, dtype=np.float32), 0)
        written = w._out_stream.write.call_args[0][0]
        np.testing.assert_array_almost_equal(written, np.full(480, 0.15, np.float32))

    def test_records_post_gain_frame_as_aec_reference(self):
        # Invariant 3: record + write co-located under one lock; ducked gain recorded.
        w = make_worker(gain=0.15)
        w._play_audio(np.ones(480, dtype=np.float32), 0)
        ref = w._player.get_reference_frame(160)
        np.testing.assert_array_almost_equal(ref, np.full(160, 0.15, np.float32))

    def test_superseded_generation_writes_nothing(self):
        # Invariant 2: _play_gen mismatch stops a stale playback immediately.
        w = make_worker()
        w._play_gen = 5
        w._play_audio(np.ones(960, dtype=np.float32), 0)   # gen 0 stale
        w._out_stream.write.assert_not_called()

    def test_no_output_stream_is_noop(self):
        w = make_worker()
        w._out_stream = None
        w._play_audio(np.ones(960, dtype=np.float32), 0)   # must not raise

    def test_write_and_close_share_one_lock(self):
        w = make_worker()
        assert isinstance(w._write_lock, type(threading.Lock()))


class TestTeardown:
    def test_close_stops_clears_and_halts_playback(self):
        # Invariant 1 + 5: lock-guarded synchronous close; late frame writes nothing.
        w = make_worker()
        stream = w._out_stream
        w.close()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert w._out_stream is None and w._running is False
        w._play_audio(np.ones(480, dtype=np.float32), w._play_gen)
        stream.write.assert_not_called()

    def test_close_is_idempotent(self):
        w = make_worker()
        w.close()
        w.close()     # must not raise

    @pytest.mark.asyncio
    async def test_drain_awaits_future_before_close(self):
        # Invariant 4 + 6: drain bumps gen, awaits the in-flight write, does NOT
        # clear _play_future; close after drain segfault-free (no concurrent write).
        w = make_worker()
        order = []

        def slow_write(frame):
            order.append("write")

        w._out_stream.write.side_effect = slow_write
        await w.play(np.ones(1440, dtype=np.float32), gen_id=0)   # 3 frames
        assert order.count("write") == 3
        future_before = w._play_future
        await w.drain()
        assert w._play_future is future_before          # NOT cleared on barge-in
        w.close()
        order.append("close")
        assert order[-1] == "close"                     # close strictly after drain


class TestCommands:
    @pytest.mark.asyncio
    async def test_duck_sets_gain_and_restore_resets(self):
        w = make_worker()
        await w.execute(C.Duck(level=0.15))
        assert w.gain == 0.15
        await w.execute(C.Restore())
        assert w.gain == 1.0

    @pytest.mark.asyncio
    async def test_speak_nudge_synthesizes_are_you_still_there(self):
        from unittest.mock import AsyncMock
        w = make_worker()
        w._tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
        await w.execute(C.SpeakNudge())
        w._tts.synthesize.assert_awaited_once()
        assert "still there" in w._tts.synthesize.await_args[0][0].lower()
        # nudge audio went to the stream
        assert w._out_stream.write.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_playback_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.workers.playback'`

- [ ] **Step 3: Implement the PlaybackWorker (invariants copied verbatim)**

```python
# modes/director/workers/playback.py
"""PlaybackWorker — sole owner of the OutputStream and the AEC reference ring.

The 6 race-fixed teardown invariants from spec section 10 are copied VERBATIM
from modes/talkback/controller.py (cross-thread sd.write/stream.close segfault
PortAudio). The Ingestion worker only READS the reference ring via
get_reference_frame; record_reference + sd.write stay co-located here under ONE
_write_lock. Also executes Duck/Restore (gain) and SpeakNudge (direct TTS of
"Are you still there?", spec section 5 — no LLM round-trip)."""

import asyncio
import threading

import numpy as np

from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director import commands as C

# 30ms playback frames (controller.py:172): small enough for ~30ms duck latency.
PLAYBACK_FRAME_SAMPLES = 480

NUDGE_TEXT = "Are you still there?"


class PlaybackWorker:
    def __init__(self, tts, player, cfg: DirectorConfig, bus: EventBus):
        self._tts = tts
        self._player = player
        self._cfg = cfg
        self._bus = bus
        self._out_stream = None
        self._running = False
        self._gain = 1.0
        # Invariant 2: generation counter, aligned with Context.gen_id; checked
        # before each frame and inside the lock. Invariant 6: _play_future is the
        # running write job and is NOT cleared on barge-in.
        self._play_gen = 0
        self._play_future = None
        # Invariant 1: held around every write AND around stream close.
        self._write_lock = threading.Lock()

    @property
    def gain(self) -> float:
        return self._gain

    def open(self, out_stream) -> None:
        """Inject the already-started OutputStream (runtime/WakeGate owns device
        creation; tests inject a MagicMock)."""
        self._out_stream = out_stream
        self._running = True

    def set_gen(self, gen_id: int) -> None:
        """Align _play_gen with the Context gen_id at the start of a generation."""
        self._play_gen = gen_id

    async def execute(self, command) -> None:
        if isinstance(command, C.Duck):
            self._gain = command.level
        elif isinstance(command, C.Restore):
            self._gain = 1.0
        elif isinstance(command, C.SpeakNudge):
            await self._speak_nudge()

    async def _speak_nudge(self) -> None:
        audio = await self._tts.synthesize(NUDGE_TEXT)
        if audio is not None and len(audio) > 0:
            await self.play(audio, gen_id=self._play_gen)

    async def play(self, audio: np.ndarray, gen_id: int) -> None:
        """Play one utterance in an executor, tracking the job so a barge-in or
        shutdown can wait for it before the stream is touched again. _play_future
        is intentionally NOT cleared on cancellation (invariant 6): cancelling the
        await does not stop the executor thread (run_in_executor jobs can't be
        cancelled once running), so the reference must survive for drain()."""
        self._play_gen = gen_id
        self._play_future = asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio, gen_id
        )
        await self._play_future

    async def drain(self) -> None:
        """Stop in-flight playback and wait for the write thread to exit
        (invariant 4). Bumping the generation makes _play_audio break at its next
        frame; awaiting the shielded future guarantees no sd.write is running when
        the stream is closed (concurrent PortAudio calls across threads segfault).
        _play_future is NOT cleared (invariant 6)."""
        self._play_gen += 1
        fut = self._play_future
        if fut is not None:
            try:
                await asyncio.shield(fut)
            except Exception:
                pass

    def _play_audio(self, audio: np.ndarray, gen: int) -> None:
        """Write one utterance frame-by-frame (blocking; runs in an executor).
        Applies the current gain (ducking), records each played (post-gain) frame
        as the AEC reference, bails if a barge-in superseded this generation or the
        session ended (invariants 2 & 3)."""
        if self._out_stream is None or len(audio) == 0:
            return
        frame = PLAYBACK_FRAME_SAMPLES
        for i in range(0, len(audio), frame):
            if not self._running or gen != self._play_gen:
                break
            gained = (audio[i:i + frame] * self._gain).astype(np.float32)
            # Invariant 1: the lock makes write and close mutually exclusive;
            # re-check the stream/gen inside it (teardown may have closed while we
            # waited). Invariant 3: record + write co-located under ONE lock.
            with self._write_lock:
                if self._out_stream is None or gen != self._play_gen:
                    break
                if self._player is not None:
                    self._player.record_reference(gained)
                try:
                    self._out_stream.write(gained)
                except Exception:
                    break

    def close(self) -> None:
        """Stop playback and close the output stream — synchronous, lock-guarded,
        idempotent (invariants 1 & 5). Safe even when a KeyboardInterrupt has
        killed the loop (the async drain can't run then)."""
        self._running = False
        self._play_gen += 1
        with self._write_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_playback_worker.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/workers/playback.py tests/director/test_playback_worker.py
git commit -m "feat(director): PlaybackWorker — 6 race-fixed teardown invariants (verbatim) + Duck/Restore/Nudge"
```

---

## Task 5: GenerationWorker — StartGeneration / Cut (LLM→chunker→TTS→Playback)

**Files:**
- Create: `modes/director/workers/generation.py`
- Test: `tests/director/test_generation_worker.py`

**Interfaces:**
- Consumes: `Command`s `StartGeneration(gen_id, messages, steer)`/`Cut(gen_id)`; a `LlmClient` (`stream(messages)` async-iterator + `cancel()`), a `TtsEngine` (`synthesize(text)`), a `SentenceChunker`-factory, the `PlaybackWorker` (Task 4).
- Produces: emits `FirstTtsFrame(gen_id)` when the first post-gain frame is enqueued to the Player, `ReplyComplete(gen_id, assistant_text)` at the end; honors `gen_id` (drops stale frames); `Cut` drains playback + cancels the LLM + bumps gen. Exposes `async execute(command)`.

> Mirrors `controller.py:_generate_response` (461-518): steer injection (467-469), `chunker.feed` + `tts.synthesize` + play loop (488-509), and the cancellable wrapper (`_generate_and_speak` 520-528, `llm.cancel()` on CancelledError). The FSM (Plan 01) already moved to THINKING when it emitted `StartGeneration`; the worker emits `FirstTtsFrame` at the first audible frame (THINKING→SPEAKING) and `ReplyComplete` at the end (→LISTENING). `Cut` reproduces controller.py:715-721 (drain + cancel task) and bumps the gen so late frames are dropped.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_generation_worker.py
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.director.workers.generation import GenerationWorker
from modes.director.workers.playback import PlaybackWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.talkback.player import Player
from modes.talkback.chunker import SentenceChunker
from modes.director import events as E
from modes.director import commands as C


class FakeLlm:
    def __init__(self, tokens):
        self._tokens = tokens
        self.cancelled = False

    async def stream(self, messages):
        for t in self._tokens:
            await asyncio.sleep(0)
            yield t

    def cancel(self):
        self.cancelled = True


def make_playback():
    pw = PlaybackWorker(tts=MagicMock(), player=Player(16000, 2.0),
                        cfg=DirectorConfig(), bus=EventBus())
    pw._out_stream = MagicMock()
    pw._running = True
    return pw


def make_worker(tokens):
    bus = EventBus()
    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
    pw = make_playback()
    w = GenerationWorker(
        llm=FakeLlm(tokens), tts=tts,
        chunker_factory=lambda: SentenceChunker(max_chunk_chars=120),
        playback=pw, bus=bus,
    )
    return w, bus, tts, pw


@pytest.mark.asyncio
async def test_start_generation_emits_first_frame_then_reply_complete():
    w, bus, tts, pw = make_worker(["Hello there. ", "How are you?"])
    await w.execute(C.StartGeneration(gen_id=1,
                                      messages=[{"role": "user", "content": "hi"}],
                                      steer=None))
    events = []
    while bus.qsize():
        events.append(await bus.get())
    first = [e for e in events if isinstance(e, E.FirstTtsFrame)]
    done = [e for e in events if isinstance(e, E.ReplyComplete)]
    assert len(first) == 1 and first[0].gen_id == 1
    assert len(done) == 1 and done[0].gen_id == 1
    assert done[0].assistant_text == "Hello there. How are you?"


@pytest.mark.asyncio
async def test_steer_is_appended_as_system_note_for_this_generation_only():
    w, bus, tts, pw = make_worker(["ok."])
    captured = {}
    orig = w._llm.stream

    async def spy(messages):
        captured["messages"] = list(messages)
        async for t in orig(messages):
            yield t

    w._llm.stream = spy
    await w.execute(C.StartGeneration(gen_id=1, messages=[{"role": "user", "content": "q"}],
                                      steer="continue the earlier topic"))
    assert captured["messages"][-1] == {"role": "system",
                                        "content": "continue the earlier topic"}


@pytest.mark.asyncio
async def test_cut_drains_playback_cancels_llm_and_bumps_gen():
    w, bus, tts, pw = make_worker(["a long answer that streams. "])
    pw.set_gen(1)
    await w.execute(C.Cut(gen_id=1))
    assert w._llm.cancelled is True
    assert pw._play_gen == 2          # drain() bumped the play gen


@pytest.mark.asyncio
async def test_no_first_frame_when_llm_yields_nothing():
    w, bus, tts, pw = make_worker([])
    await w.execute(C.StartGeneration(gen_id=3, messages=[], steer=None))
    events = []
    while bus.qsize():
        events.append(await bus.get())
    assert not [e for e in events if isinstance(e, E.FirstTtsFrame)]
    done = [e for e in events if isinstance(e, E.ReplyComplete)]
    assert len(done) == 1 and done[0].gen_id == 3 and done[0].assistant_text == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_generation_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.workers.generation'`

- [ ] **Step 3: Implement the GenerationWorker**

```python
# modes/director/workers/generation.py
"""GenerationWorker — executes StartGeneration / Cut.

Streams LLM tokens into a SentenceChunker, synthesizes each sentence via the
TtsEngine, and plays it through the PlaybackWorker. Emits FirstTtsFrame(gen_id)
when the first audible frame is written (THINKING->SPEAKING) and
ReplyComplete(gen_id, text) at the end (->LISTENING). Cut drains playback +
cancels the LLM + bumps the gen so stale frames are dropped (spec section 11).
Mirrors controller.py:461-528 (steer injection, feed/synthesize/play loop,
cancellable wrapper)."""

import asyncio

from modes.director.bus import EventBus
from modes.director import events as E
from modes.director import commands as C


class GenerationWorker:
    def __init__(self, llm, tts, chunker_factory, playback, bus: EventBus):
        self._llm = llm
        self._tts = tts
        self._chunker_factory = chunker_factory
        self._playback = playback
        self._bus = bus
        self._task = None

    async def execute(self, command) -> None:
        if isinstance(command, C.StartGeneration):
            await self._start(command)
        elif isinstance(command, C.Cut):
            await self._cut(command)

    async def _start(self, cmd: C.StartGeneration) -> None:
        """Run one generation to completion. Caller (runtime) awaits this; the
        runtime keeps draining the bus, so emitted FirstTtsFrame/ReplyComplete
        are processed in order."""
        self._task = asyncio.current_task()
        gen_id = cmd.gen_id
        self._playback.set_gen(gen_id)
        messages = list(cmd.messages)
        if cmd.steer:                                    # one-shot steer (controller.py:467-469)
            messages = messages + [{"role": "system", "content": cmd.steer}]
        chunker = self._chunker_factory()
        full = []
        first_frame_sent = False
        try:
            async for token in self._llm.stream(messages):
                full.append(token)
                chunk = chunker.feed(token)
                if chunk:
                    first_frame_sent = await self._speak_chunk(chunk, gen_id,
                                                               first_frame_sent)
            remaining = chunker.flush()
            if remaining:
                first_frame_sent = await self._speak_chunk(remaining, gen_id,
                                                           first_frame_sent)
        except asyncio.CancelledError:
            self._llm.cancel()                           # controller.py:527
            raise
        await self._bus.emit(E.ReplyComplete(gen_id=gen_id,
                                             assistant_text="".join(full)))

    async def _speak_chunk(self, text: str, gen_id: int, first_frame_sent: bool) -> bool:
        audio = await self._tts.synthesize(text)
        if audio is None or len(audio) == 0:
            return first_frame_sent
        if not first_frame_sent:
            await self._bus.emit(E.FirstTtsFrame(gen_id=gen_id))
            first_frame_sent = True
        await self._playback.play(audio, gen_id)
        return first_frame_sent

    async def _cut(self, cmd: C.Cut) -> None:
        """Drain playback (bumps _play_gen) then cancel the in-flight LLM/task
        (controller.py:715-721). The arbiter client is never touched here (Plan 06)."""
        await self._playback.drain()
        self._llm.cancel()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_generation_worker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/workers/generation.py tests/director/test_generation_worker.py
git commit -m "feat(director): GenerationWorker — StartGeneration/Cut, FirstTtsFrame/ReplyComplete, gen_id drop"
```

---

## Task 6: IngestionWorker — mic → AEC → VAD → RMS → Smart Turn → ECAPA → events

**Files:**
- Create: `modes/director/workers/ingestion.py`
- Test: `tests/director/test_ingestion_worker.py`

**Interfaces:**
- Consumes: `MicrophoneStream.stream()` (blocking generator, via `run_in_executor`), `SileroVAD` (`process_chunk` → `SpeechSegment`s, `is_speaking`), `AecProcessor` (`process_frame`; may be `None`), `SmartTurnDetector`/`NullTurnDetector` (`endpoint_prob`, via executor), `EmbeddingExtractor` (`extract`, via executor), the `PlaybackWorker` (read-only `get_reference_frame` via its `Player`), and a read-only `state_getter()` (returns the Director's current `State`).
- Produces: emits `NearFieldOnset(rms, is_target)` on voiced onset during SPEAKING, `SegmentEndpointed(duration_ms, rms, is_target, endpoint_prob)` for segments in LISTENING, `InterjectionSegment(duration_ms, rms, is_target, speaker_score)` for segments in EVALUATING; stages segment audio into the SttWorker. `async run()` (the supervised loop), `stop()`.

> **Plan 02 stubs (Plan 05 replaces):** `is_target` is **hard-coded True** (pVAD lands in Plan 05); `speaker_score` is a synchronous ECAPA call via `run_in_executor` against the primary embedding (`cosine_similarity`, the same call as controller.py:683-686, but off the synchronous decision path). State-awareness (spec §3): the worker reads `state_getter()` to pick which event to emit — onset-duck only in SPEAKING (controller.py:839-849), segment routing by state (LISTENING vs EVALUATING). The AEC per-frame consume mirrors controller.py:819-832 and reads the reference ring the **Playback** worker fills (spec §10 invariant 3 — the ring is the only cross-worker hand-off).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_ingestion_worker.py
import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.director.workers.ingestion import IngestionWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director.state import State
from core.vad.silero_vad import SpeechSegment
from modes.director import events as E


class FakeMic:
    """Yields a fixed list of chunks then stops (mirrors MicrophoneStream.stream)."""
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self):
        for c in self._chunks:
            yield c


class FakeVad:
    """Returns queued segment-lists per chunk; is_speaking is settable."""
    def __init__(self, per_chunk_segments, is_speaking=False):
        self._per_chunk = list(per_chunk_segments)
        self.is_speaking = is_speaking

    def process_chunk(self, chunk):
        return self._per_chunk.pop(0) if self._per_chunk else []


class FakeTurn:
    def __init__(self, prob):
        self._prob = prob

    def endpoint_prob(self, audio, sample_rate):
        return self._prob


def _seg(duration_ms=900.0, level=0.5):
    n = int(duration_ms / 1000 * 16000)
    return SpeechSegment(audio=np.full(n, level, dtype=np.float32),
                         start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms)


def make_worker(mic, vad, state, turn_prob=0.9, embedder_score=0.9):
    bus = EventBus()
    stt = MagicMock()
    stt.set_pending_user_audio = MagicMock()
    stt.set_pending_interjection_audio = MagicMock()
    embedder = MagicMock()
    embedder.extract = MagicMock(return_value=np.ones(192, dtype=np.float32))
    playback = MagicMock()
    playback.get_reference_frame = MagicMock(return_value=None)
    w = IngestionWorker(
        mic=mic, vad=vad, aec=None,
        turn_detector=FakeTurn(turn_prob), embedder=embedder,
        primary_embedding=np.ones(192, dtype=np.float32),
        stt_worker=stt, playback=playback, bus=bus,
        cfg=DirectorConfig(), proximity_rms=0.02,
        state_getter=lambda: state,
        score_fn=lambda a, b: embedder_score,    # injected cosine (no real ECAPA)
    )
    return w, bus, stt


@pytest.mark.asyncio
async def test_listening_segment_emits_segment_endpointed_and_stages_audio():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.LISTENING, turn_prob=0.8)
    await w.run()
    evs = [await bus.get() for _ in range(bus.qsize())]
    seps = [e for e in evs if isinstance(e, E.SegmentEndpointed)]
    assert len(seps) == 1
    assert seps[0].is_target is True and seps[0].endpoint_prob == 0.8
    assert seps[0].duration_ms == 900.0 and seps[0].rms > 0.0
    stt.set_pending_user_audio.assert_called_once()


@pytest.mark.asyncio
async def test_evaluating_segment_emits_interjection_with_speaker_score():
    seg = _seg()
    w, bus, stt = make_worker(FakeMic([seg.audio]), FakeVad([[seg]]),
                              State.EVALUATING, embedder_score=0.77)
    await w.run()
    evs = [await bus.get() for _ in range(bus.qsize())]
    inter = [e for e in evs if isinstance(e, E.InterjectionSegment)]
    assert len(inter) == 1 and inter[0].speaker_score == 0.77
    stt.set_pending_interjection_audio.assert_called_once()


@pytest.mark.asyncio
async def test_speaking_onset_emits_near_field_onset_once():
    seg_audio = np.full(512, 0.5, dtype=np.float32)
    vad = FakeVad([[]], is_speaking=True)        # speaking, no endpointed segment yet
    w, bus, stt = make_worker(FakeMic([seg_audio]), vad, State.SPEAKING)
    await w.run()
    evs = [await bus.get() for _ in range(bus.qsize())]
    onsets = [e for e in evs if isinstance(e, E.NearFieldOnset)]
    assert len(onsets) == 1 and onsets[0].is_target is True and onsets[0].rms > 0.0


@pytest.mark.asyncio
async def test_far_onset_below_proximity_does_not_emit():
    quiet = np.full(512, 0.001, dtype=np.float32)
    vad = FakeVad([[]], is_speaking=True)
    w, bus, stt = make_worker(FakeMic([quiet]), vad, State.SPEAKING)
    await w.run()
    assert bus.qsize() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_ingestion_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.workers.ingestion'`

- [ ] **Step 3: Implement the IngestionWorker**

```python
# modes/director/workers/ingestion.py
"""IngestionWorker — the mic-side reflex pipeline.

Reads mic chunks via run_in_executor over MicrophoneStream.stream() (blocking
generator), runs AEC per-frame against the playback reference ring (read-only;
the ring is filled by the Playback worker under its lock — spec section 10
invariant 3), runs Silero VAD, computes RMS, and routes events by the Director's
current state (read-only, spec section 3):

  SPEAKING   -> NearFieldOnset on near-field voiced onset (controller.py:839-849)
  LISTENING  -> SegmentEndpointed (Smart Turn endpoint_prob via executor)
  EVALUATING -> InterjectionSegment (ECAPA speaker_score via executor)

Plan 02 stubs (Plan 05 replaces): is_target is hard-coded True; speaker_score is
a synchronous ECAPA cosine via run_in_executor (off the synchronous decision
path). The captured audio is staged into the SttWorker so a later
TranscribeUserTurn/TranscribeInterjection has the right buffer."""

import asyncio

import numpy as np

from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director.state import State
from modes.director import events as E


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


class IngestionWorker:
    def __init__(self, mic, vad, aec, turn_detector, embedder,
                 primary_embedding, stt_worker, playback, bus: EventBus,
                 cfg: DirectorConfig, proximity_rms: float, state_getter,
                 score_fn):
        self._mic = mic
        self._vad = vad
        self._aec = aec
        self._turn = turn_detector
        self._embedder = embedder
        self._primary = primary_embedding
        self._stt = stt_worker
        self._playback = playback
        self._bus = bus
        self._cfg = cfg
        self._proximity_rms = proximity_rms
        self._state_getter = state_getter
        self._score_fn = score_fn          # cosine(embedding, primary) -> float
        self._running = False
        self._ducked_onset = False         # one onset per speech run (controller.py:842)

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_event_loop()
        mic_iter = self._mic.stream()

        def _next_chunk():
            try:
                return next(mic_iter)
            except StopIteration:
                return None

        while self._running:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                break
            state = self._state_getter()
            chunk = self._apply_aec(chunk, state)
            segments = self._vad.process_chunk(chunk)
            self._maybe_onset(chunk, state)
            for seg in segments:
                await self._on_segment(seg, self._state_getter())

    def _apply_aec(self, chunk: np.ndarray, state: State) -> np.ndarray:
        """Per-frame AEC during playback (controller.py:819-832). Reads the
        reference ring the Playback worker fills; never records it here."""
        if state is not State.SPEAKING or self._aec is None:
            return chunk
        fs = self._aec.frame_samples
        cleaned = []
        for i in range(0, len(chunk), fs):
            frame = chunk[i:i + fs]
            if len(frame) < fs:
                break
            ref = self._playback.get_reference_frame(fs)
            if ref is not None:
                frame = self._aec.process_frame(frame, ref)
            cleaned.append(frame)
        return np.concatenate(cleaned) if cleaned else chunk

    def _maybe_onset(self, chunk: np.ndarray, state: State) -> None:
        """Duck-at-onset reflex (controller.py:839-849): emit NearFieldOnset on
        the first voiced, near-field chunk of a speech run during SPEAKING."""
        if state is not State.SPEAKING:
            self._ducked_onset = False
            return
        if self._ducked_onset or getattr(self._vad, "is_speaking", False) is not True:
            return
        rms = _rms(chunk)
        if rms >= self._proximity_rms:
            self._ducked_onset = True
            asyncio.ensure_future(
                self._bus.emit(E.NearFieldOnset(rms=rms, is_target=True))
            )

    async def _on_segment(self, seg, state: State) -> None:
        rms = _rms(seg.audio)
        if state is State.LISTENING:
            prob = await self._endpoint_prob(seg.audio)
            self._stt.set_pending_user_audio(seg.audio)
            await self._bus.emit(E.SegmentEndpointed(
                duration_ms=seg.duration_ms, rms=rms,
                is_target=True, endpoint_prob=prob,
            ))
        elif state is State.EVALUATING:
            score = await self._speaker_score(seg.audio)
            self._stt.set_pending_interjection_audio(seg.audio)
            await self._bus.emit(E.InterjectionSegment(
                duration_ms=seg.duration_ms, rms=rms,
                is_target=True, speaker_score=score,
            ))
        # SPEAKING/THINKING/IDLE: onset handled separately; segments are ignored.

    async def _endpoint_prob(self, audio: np.ndarray) -> float:
        loop = asyncio.get_event_loop()
        return float(await loop.run_in_executor(
            None, self._turn.endpoint_prob, audio, 16000))

    async def _speaker_score(self, audio: np.ndarray) -> float:
        """ECAPA speaker_score off the synchronous path (Plan 05 swaps for pVAD)."""
        if self._embedder is None or self._primary is None:
            return 1.0
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embedder.extract, audio)
        return float(self._score_fn(embedding, self._primary))
```

> **Note on `_maybe_onset`:** it uses `asyncio.ensure_future` rather than `await` so the onset emit never blocks the per-chunk loop on the rare contended bus; the integration test (Task 7) tolerates ordering by draining the bus fully. The endpointed/interjection emits are awaited because their order relative to the staged audio matters.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_ingestion_worker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modes/director/workers/ingestion.py tests/director/test_ingestion_worker.py
git commit -m "feat(director): IngestionWorker — AEC/VAD/RMS/SmartTurn/ECAPA -> Onset/Segment/Interjection events"
```

---

## Task 7: DirectorRuntime — one loop, event→dispatch→route, DirectorResult

**Files:**
- Create: `modes/director/runtime.py`
- Test: `tests/director/test_runtime.py`

**Interfaces:**
- Consumes: `Director` (Plan 01), `EventBus` (Task 1), `AsyncWatchdog` (Task 2), all four workers (Tasks 3-6), all Plan-01 commands.
- Produces: `DirectorRuntime(director, bus, watchdog, ingestion, stt_worker, generation, playback, clock)` with `async run_async() -> DirectorResult` (the `event = await bus.get(); cmds = director.dispatch(event); for c in cmds: await route(c)` loop) and `run() -> DirectorResult` (sync entry: `asyncio.new_event_loop()` + `run_until_complete`, mirroring controller.py:246-255). `async _route(command)` dispatches each command to its worker. On `EndSession`, runs teardown (drain playback, stop watchdog, close stream) and returns `DirectorResult`.

> **Loop structure mirrors controller.py:246-255 / 257-459.** `run()` spins a fresh loop and `run_until_complete(run_async())`, with a synchronous `playback.close()` backstop in `finally` (controller.py:254 — interrupt-safe). `run_async()` starts the Ingestion worker + watchdog as tasks, then drains the bus one event at a time, dispatching to the Director (the sole mutator) and routing commands. `StartGeneration`/`Cut`/`Restore`/`Duck`/`SpeakNudge` go to their workers; `EndSession` ends the loop. Routing `StartGeneration` runs the generation **as a task** (so the runtime keeps draining FirstTtsFrame/ReplyComplete the generation emits) and tracks it so teardown can cancel it.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_runtime.py
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.director.runtime import DirectorRuntime
from modes.director.director import Director
from modes.director.config import DirectorConfig
from modes.director.bus import EventBus
from modes.director.watchdog import AsyncWatchdog
from modes.director.workers.stt_worker import SttWorker
from modes.director.workers.generation import GenerationWorker
from modes.director.workers.playback import PlaybackWorker
from modes.director.state import State
from modes.talkback.conversation import ConversationManager
from modes.talkback.player import Player
from modes.talkback.chunker import SentenceChunker
from modes.director import events as E
from modes.director import commands as C


class FakeLlm:
    def __init__(self, tokens):
        self._tokens = tokens
        self.cancelled = False

    async def stream(self, messages):
        for t in self._tokens:
            await asyncio.sleep(0)
            yield t

    def cancel(self):
        self.cancelled = True


class FakeStt:
    async def transcribe_segment(self, audio):
        return "tell me a story"


def build_runtime(clock):
    bus = EventBus()
    conv = ConversationManager(system_prompt="s")
    director = Director(DirectorConfig(), conv, now=clock(), proximity_rms=0.02)
    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
    playback = PlaybackWorker(tts=tts, player=Player(16000, 2.0),
                              cfg=DirectorConfig(), bus=bus)
    playback._out_stream = MagicMock()
    playback._running = True
    stt_worker = SttWorker(FakeStt(), bus)
    generation = GenerationWorker(
        llm=FakeLlm(["Once upon a time. ", "The end."]), tts=tts,
        chunker_factory=lambda: SentenceChunker(max_chunk_chars=120),
        playback=playback, bus=bus)
    ingestion = MagicMock()           # driven manually in the test, not run
    ingestion.run = AsyncMock()
    ingestion.stop = MagicMock()
    watchdog = AsyncWatchdog(tick_s=1.0, clock=clock, bus=bus,
                             on_session_end=lambda r: None)
    rt = DirectorRuntime(director=director, bus=bus, watchdog=watchdog,
                         ingestion=ingestion, stt_worker=stt_worker,
                         generation=generation, playback=playback, clock=clock)
    return rt, bus, director, playback, generation


@pytest.mark.asyncio
async def test_full_synthetic_turn_round_trip_and_clean_teardown():
    t = [0.0]
    rt, bus, director, playback, generation = build_runtime(lambda: t[0])

    async def drive():
        # 1. user turn transcribed -> StartGeneration (THINKING)
        await bus.emit(E.UserTurnTranscribed(text="tell me a story", mean_word_prob=0.9))
        # 2. let the runtime process: it will route StartGeneration, which emits
        #    FirstTtsFrame (-> SPEAKING) and ReplyComplete (-> LISTENING).
        await asyncio.sleep(0.05)
        assert director.state in (State.SPEAKING, State.LISTENING)
        await asyncio.sleep(0.05)
        assert director.state is State.LISTENING
        # 3. assistant turn recorded by the reducer on ReplyComplete
        msgs = director.ctx.conversation.get_messages()
        assert {"role": "assistant", "content": "Once upon a time. The end."} in msgs
        # 4. advance the clock past silence_timeout while LISTENING -> EndSession.
        t[0] = 100.0
        await bus.emit(E.Tick(now=100.0))

    driver = asyncio.create_task(drive())
    result = await asyncio.wait_for(rt.run_async(), timeout=5.0)
    await driver
    assert result.reason == "silence_timeout"
    assert result.total_duration_s >= 0.0
    # clean teardown: stream closed, llm cancelled-or-done, no orphan task.
    assert playback._out_stream is None
    assert generation._task is None or generation._task.done()


@pytest.mark.asyncio
async def test_duck_command_routes_to_playback():
    t = [0.0]
    rt, bus, director, playback, generation = build_runtime(lambda: t[0])
    await rt._route(C.Duck(level=0.15))
    assert playback.gain == 0.15
    await rt._route(C.Restore())
    assert playback.gain == 1.0


@pytest.mark.asyncio
async def test_end_session_sets_result_and_stops():
    t = [0.0]
    rt, bus, director, playback, generation = build_runtime(lambda: t[0])

    async def drive():
        await asyncio.sleep(0.01)
        await bus.emit(E.Tick(now=400.0))     # past hard_timeout (300s)

    asyncio.create_task(drive())
    result = await asyncio.wait_for(rt.run_async(), timeout=5.0)
    assert result.reason == "hard_timeout"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.runtime'`

- [ ] **Step 3: Implement the DirectorRuntime**

```python
# modes/director/runtime.py
"""DirectorRuntime — owns ONE asyncio loop around the Plan-01 Director.

Mirrors TalkbackController.run/_run_async (controller.py:246-459): run() spins a
fresh loop (asyncio.new_event_loop + run_until_complete) with a synchronous
playback.close() backstop in finally (interrupt-safe, controller.py:254).
run_async() starts the Ingestion worker + watchdog, then drains the bus one event
at a time:  event = await bus.get(); cmds = director.dispatch(event); route each.
The Director is the SOLE mutator (spec section 3). Returns DirectorResult at
session end (spec Req 5: single owner of lifecycle + teardown)."""

import asyncio

from modes.director.result import DirectorResult
from modes.director import commands as C


class DirectorRuntime:
    def __init__(self, director, bus, watchdog, ingestion, stt_worker,
                 generation, playback, clock):
        self._director = director
        self._bus = bus
        self._watchdog = watchdog
        self._ingestion = ingestion
        self._stt = stt_worker
        self._generation = generation
        self._playback = playback
        self._clock = clock
        self._started_at = clock()
        self._result_reason = None
        self._gen_task = None

    def run(self) -> DirectorResult:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_async())
        finally:
            # Interrupt-safe backstop (controller.py:254): if KeyboardInterrupt
            # unwound the loop, make sure no playback thread is still writing the
            # about-to-be-torn-down audio device.
            self._playback.close()
            loop.close()

    async def run_async(self) -> DirectorResult:
        self._started_at = self._clock()
        ingestion_task = asyncio.create_task(self._ingestion.run())
        self._watchdog.start()
        try:
            while self._result_reason is None:
                event = await self._bus.get()
                commands = self._director.dispatch(event)
                for command in commands:
                    await self._route(command)
        finally:
            await self._teardown(ingestion_task)
        return DirectorResult(
            reason=self._result_reason or "stopped",
            turns=self._director.ctx.conversation.turn_count,
            total_duration_s=self._clock() - self._started_at,
        )

    async def _route(self, command) -> None:
        if isinstance(command, (C.Duck, C.Restore, C.SpeakNudge)):
            await self._playback.execute(command)
        elif isinstance(command, (C.TranscribeUserTurn, C.TranscribeInterjection)):
            await self._stt.execute(command)
        elif isinstance(command, C.StartGeneration):
            # Run the generation as a task so the runtime keeps draining the
            # FirstTtsFrame/ReplyComplete events it emits (controller.py:644-647).
            self._gen_task = asyncio.create_task(self._generation.execute(command))
        elif isinstance(command, C.Cut):
            await self._generation.execute(command)
        elif isinstance(command, C.EndSession):
            self._result_reason = command.reason
            self._watchdog.request_stop(command.reason)

    async def _teardown(self, ingestion_task) -> None:
        """Graceful, no-orphan teardown (spec section 11): cancel ingestion +
        generation, drain playback BEFORE close (controller.py:436-441), stop the
        watchdog, close the stream."""
        self._ingestion.stop()
        ingestion_task.cancel()
        try:
            await ingestion_task
        except (asyncio.CancelledError, Exception):
            pass
        if self._gen_task is not None and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except (asyncio.CancelledError, Exception):
                pass
        self._gen_task = None
        await self._playback.drain()        # await in-flight write BEFORE close
        await self._watchdog.stop()
        self._playback.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_runtime.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full director suite**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS (all Plan-01 + Plan-02 tests green)

- [ ] **Step 6: Commit**

```bash
git add modes/director/runtime.py tests/director/test_runtime.py
git commit -m "feat(director): DirectorRuntime — one loop, event->dispatch->route, DirectorResult + clean teardown"
```

---

## Task 8: Cut-during-playback regression (no segfault; drain-before-close ordering)

**Files:**
- Test: `tests/director/test_cut_teardown_regression.py`

**Interfaces:**
- Consumes: `PlaybackWorker` (Task 4), `GenerationWorker` (Task 5), `DirectorRuntime` (Task 7).
- Produces: regression coverage for spec §10/§12 — a cut while a frame write is in flight must (a) never call `sd.write` and `stream.close` concurrently, (b) `drain()` (await the future) strictly before `close()`, (c) leave `_play_future` set across the cut.

> This is the explicit Req-5 / §12 "concurrency/teardown regression": cut-during-playback (no PortAudio segfault), `_drain_playback` awaited before close. We simulate the cross-thread hazard with a write that records call ordering and a barrier; the assertions prove write and close are serialized by the one `_write_lock` and that drain precedes close.

- [ ] **Step 1: Write the failing/then-passing regression test**

```python
# tests/director/test_cut_teardown_regression.py
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.director.workers.playback import PlaybackWorker
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.talkback.player import Player


def make_playback():
    pw = PlaybackWorker(tts=MagicMock(), player=Player(16000, 2.0),
                        cfg=DirectorConfig(), bus=EventBus())
    pw._out_stream = MagicMock()
    pw._running = True
    return pw


@pytest.mark.asyncio
async def test_cut_during_playback_serializes_write_and_close():
    """Write and close must never overlap (concurrent PortAudio calls segfault).
    The shared _write_lock is the guard; assert no interleave by checking the
    lock is held during write and that close waits for it."""
    pw = make_playback()
    events = []
    write_entered = threading.Event()
    release_write = threading.Event()

    def blocking_write(frame):
        events.append("write_start")
        write_entered.set()
        release_write.wait(timeout=2.0)     # hold the lock briefly
        events.append("write_end")

    pw._out_stream.write.side_effect = blocking_write

    # Start a play in a background executor (it grabs _write_lock for the frame).
    play_task = asyncio.create_task(pw.play(np.ones(480, dtype=np.float32), gen_id=0))
    await asyncio.get_event_loop().run_in_executor(None, write_entered.wait, 2.0)

    # While the write holds the lock, a close from "another thread" must block
    # until the write releases — proving they never run concurrently.
    def do_close():
        events.append("close_attempt")
        pw.close()
        events.append("close_done")

    closer = asyncio.get_event_loop().run_in_executor(None, do_close)
    await asyncio.sleep(0.05)
    # close_attempt is recorded but close_done is NOT yet (blocked on the lock).
    assert "close_attempt" in events and "close_done" not in events
    release_write.set()
    await play_task
    await closer
    # write fully finished before close finished (lock serialized them).
    assert events.index("write_end") < events.index("close_done")


@pytest.mark.asyncio
async def test_drain_awaits_future_strictly_before_close():
    """drain() must await the in-flight write before close() (invariant 4); the
    future survives the cut (invariant 6)."""
    pw = make_playback()
    order = []
    pw._out_stream.write.side_effect = lambda f: order.append("write")
    await pw.play(np.ones(1920, dtype=np.float32), gen_id=0)   # 4 frames
    fut = pw._play_future
    await pw.drain()
    order.append("drain_done")
    assert pw._play_future is fut                  # invariant 6: not cleared
    pw.close()
    order.append("close")
    assert order.count("write") == 4
    assert order.index("drain_done") < order.index("close")


@pytest.mark.asyncio
async def test_stale_generation_frames_dropped_after_cut():
    """After a cut bumps _play_gen, a late play() for the OLD gen writes nothing
    (spec section 11 stale-gen drop)."""
    pw = make_playback()
    written = []
    pw._out_stream.write.side_effect = lambda f: written.append(f)
    await pw.drain()                       # bumps _play_gen 0 -> 1
    pw._play_audio(np.ones(480, dtype=np.float32), gen=0)    # stale gen 0
    assert written == []
```

- [ ] **Step 2: Run the regression test**

Run: `python3 -m pytest tests/director/test_cut_teardown_regression.py -v`
Expected: PASS (3 tests) — the invariants from Task 4 already satisfy these; this task pins them against future refactors.

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest tests/director/ -v`
Expected: PASS (all tests green)

- [ ] **Step 4: Commit**

```bash
git add tests/director/test_cut_teardown_regression.py
git commit -m "test(director): cut-during-playback regression — write/close serialized, drain-before-close, stale-gen drop"
```

---

## Self-Review

- **Spec coverage (this plan = spec §13.1 worker layer + §3 concurrency model, making the Plan-01 reducer runnable):**
  - EventBus (async, asyncio.Queue): Task 1. ✓
  - DirectorRuntime (one loop, `asyncio.new_event_loop` + `run_until_complete` mirroring controller.py:246-255; `event = await bus.get(); cmds = director.dispatch(event); for c: await route(c)`; returns `DirectorResult(reason, turns, total_duration_s)` on EndSession): Task 7. ✓
  - AsyncWatchdog emits `Tick(now)` on a timer and ends the session via the `EndSession`-equivalent (`request_stop(reason)` + `on_session_end`); the non-terminal nudge stays in the reducer (Plan 01) — scope documented in Task 2. ✓
  - IngestionWorker (mic via `run_in_executor`, per-frame AEC against the playback ref ring [read-only], Silero VAD, RMS, Smart Turn `endpoint_prob` via executor on LISTENING segments; emits `NearFieldOnset`/`SegmentEndpointed`/`InterjectionSegment`; `is_target` hard-coded True; `speaker_score` from synchronous ECAPA via executor; reads Director state read-only to route): Task 6. ✓
  - SttWorker (executes `TranscribeUserTurn`/`TranscribeInterjection`, emits `UserTurnTranscribed`/`InterjectionTranscribed`; `TranscriptResult` defined here + `wrap_transcript` shim wrapping today's bare-str return as `mean_word_prob=1.0` until Plan 04): Tasks 1 & 3. ✓
  - GenerationWorker (executes `StartGeneration` — streams LLM→`SentenceChunker`→`TtsEngine`→Player; emits `FirstTtsFrame` at first post-gain frame and `ReplyComplete` at end; executes `Cut` = drain + cancel LLM + bump gen; honors gen_id / drops stale): Task 5. ✓
  - PlaybackWorker/Player (executes `Duck`/`Restore`/`SpeakNudge` — direct TTS of "Are you still there?"): Task 4. ✓
  - **6 race-fixed teardown invariants (spec §10) verbatim with controller.py line citations:** `_write_lock` around every write AND stop/close (221-228, 237-244); `_play_gen` checked before each frame + inside the lock (216, 222); `record_reference` co-located with write under one lock (224-227), Ingestion only reads the ring; `drain()` awaits `asyncio.shield(_play_future)` before close (189-203, 439); two-path teardown (async drain then sync idempotent `close`, 231-244, 254); `_play_future` not cleared on barge-in (181-187 docstring). Implemented in Task 4, regression-tested in Tasks 4 & 8. ✓
  - DirectorRuntime integration test (fake mic via the runtime test's manual bus driving + fake STT/LLM/TTS, full synthetic turn, event→command→event round-trip, clean teardown, silence suspended while speaking then timeout): Task 7. ✓
  - Cut-during-playback no-segfault + drain-before-close + stale-gen-drop regression: Task 8. ✓
  - Testing with fakes/mocks mirroring `tests/kiosk/talkback/` patterns (`MagicMock` out_stream, `Player`/`SentenceChunker` real, fake mic/STT/LLM/VAD): every task. ✓
  - NOT in this plan (correctly deferred): WakeGate + double-manage deletion + `DirectorHandoff` wiring (Plan 03); STT re-backing to NeMo/openai-whisper with real `mean_word_prob` (Plan 04 — Plan 02 ships the `wrap_transcript` shim); pVAD `is_target` + ECAPA safety-net demotion + lockout + verify-before-serve (Plan 05 — `is_target` is hard-coded True, `speaker_score` is plain ECAPA here); bounded interrupted-stack + LLM-steered continuation + auto-resume + two-client arbiter lifecycle (Plan 06 — only the main LLM is cancelled on Cut here).
- **Placeholder scan:** no TBD/TODO/FIXME; every code step is complete and runnable (`grep -nE "TBD|TODO|FIXME"` clean — verified after authoring).
- **Type consistency:** `EventBus.emit/get` async throughout; every worker exposes `async execute(command)` except IngestionWorker (`async run()`); `DirectorRuntime._route` dispatches commands by `isinstance` against `modes/director/commands.py` types (`Duck`/`Restore`/`SpeakNudge`/`TranscribeUserTurn`/`TranscribeInterjection`/`StartGeneration`/`Cut`/`EndSession`); events constructed match Plan-01 `events.py` field names verbatim (`Tick.now`, `SegmentEndpointed(duration_ms, rms, is_target, endpoint_prob)`, `NearFieldOnset(rms, is_target)`, `InterjectionSegment(duration_ms, rms, is_target, speaker_score)`, `UserTurnTranscribed(text, mean_word_prob)`, `InterjectionTranscribed(text, mean_word_prob)`, `FirstTtsFrame(gen_id)`, `ReplyComplete(gen_id, assistant_text)`); `gen_id` is an `int` consistently aligned with `PlaybackWorker._play_gen`; `DirectorResult` fields match `TalkbackResult`. The `Director.dispatch(event) -> list[Command]` contract (Plan 01) is the sole mutator entry the runtime calls. ✓
- **Composition with sibling plans (BINDING CONTRACT honored):** Plan 03 consumes `DirectorRuntime.run(handoff)`-shaped entry (this plan provides `run()`/`run_async()`; Plan 03 wires `DirectorHandoff` → constructs the workers/runtime and the real `OutputStream` injected via `PlaybackWorker.open()`); Plan 04 swaps `StreamingStt` internals behind the unchanged `transcribe_segment` call — `wrap_transcript` already accepts a real `TranscriptResult`; Plan 05 replaces the IngestionWorker's hard-coded `is_target=True` and the `score_fn` ECAPA path with pVAD; Plan 06 adds the arbiter `LlmClient` (never cancelled on `Cut`) and the resume-stack consumption — the `GenerationWorker._cut` already only cancels the main LLM. ✓
