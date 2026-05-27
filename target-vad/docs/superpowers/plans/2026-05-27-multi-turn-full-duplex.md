# Multi-Turn Full-Duplex Conversation Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multi-turn voice conversation within a single wake-word session, with the mic staying hot during TTS playback and speaker-verified barge-in.

**Architecture:** A `_listen_loop` asyncio task continuously reads mic chunks via executor → AEC → VAD → segment queue. The main `_run_async` loop pulls segments and dispatches: LISTENING → transcribe + respond, SPEAKING → speaker-verify + barge-in. `_generate_response` runs as a cancellable task for barge-in support.

**Tech Stack:** asyncio, sounddevice, faster-whisper, numpy, Silero VAD, ECAPA-TDNN embedder, cosine similarity

---

### Task 1: Extend TalkbackHandoff with vad and embedder fields

**Files:**
- Modify: `modes/talkback/handoff.py`
- Modify: `tests/kiosk/talkback/test_handoff.py`

- [ ] **Step 1: Update test for new handoff fields**

In `tests/kiosk/talkback/test_handoff.py`, add a test that verifies the two new fields:

```python
def test_construction_with_vad_and_embedder(self):
    mic = MagicMock()
    emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
    seg = MagicMock()
    seg.duration_ms = 1000.0
    cfg = {"sample_rate_hz": 16000}
    vad = MagicMock()
    embedder = MagicMock()

    h = TalkbackHandoff(
        mic=mic,
        primary_embedding=emb,
        first_segment=seg,
        config=cfg,
        vad=vad,
        embedder=embedder,
    )
    assert h.vad is vad
    assert h.embedder is embedder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kiosk/talkback/test_handoff.py::TestTalkbackHandoff::test_construction_with_vad_and_embedder -v`
Expected: FAIL — `TypeError: __init__() got unexpected keyword argument 'vad'`

- [ ] **Step 3: Add vad and embedder fields to TalkbackHandoff**

In `modes/talkback/handoff.py`, add to the dataclass:

```python
@dataclass
class TalkbackHandoff:
    """Payload KioskPipeline passes to TalkbackController at session start."""
    mic: Any
    primary_embedding: np.ndarray
    first_segment: Any
    config: dict
    vad: Any              # pipeline's SileroVAD instance
    embedder: Any         # pipeline's EmbeddingExtractor
```

- [ ] **Step 4: Fix existing tests that construct TalkbackHandoff without the new fields**

The two existing tests in `test_handoff.py` construct TalkbackHandoff with only 4 args. Add `vad=MagicMock(), embedder=MagicMock()` to both.

Also check `tests/kiosk/talkback/test_controller.py` — any `TalkbackHandoff(...)` calls there need the same fix.

- [ ] **Step 5: Run all handoff and controller tests**

Run: `pytest tests/kiosk/talkback/test_handoff.py tests/kiosk/talkback/test_controller.py -v`
Expected: All PASS

- [ ] **Step 6: Update pipeline to pass vad and embedder in handoff**

In `modes/kiosk/pipeline.py`, in `_start_session_from_segment`, update the TalkbackHandoff construction:

```python
handoff = TalkbackHandoff(
    mic=self.mic,
    primary_embedding=embedding,
    first_segment=segment,
    config=self._talkback_config,
    vad=self.vad,
    embedder=self.embedder,
)
```

- [ ] **Step 7: Run full talkback test suite**

Run: `pytest tests/kiosk/talkback/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add modes/talkback/handoff.py modes/kiosk/pipeline.py tests/kiosk/talkback/test_handoff.py
git commit -m "feat(handoff): add vad and embedder fields for multi-turn"
```

---

### Task 2: Add _listen_loop to TalkbackController

**Files:**
- Modify: `modes/talkback/controller.py`
- Create: `tests/kiosk/talkback/test_listen_loop.py`

This task adds the async `_listen_loop` method that reads mic chunks, applies AEC during playback, feeds VAD, and puts completed speech segments into an asyncio.Queue. It does NOT yet wire it into `_run_async` — that happens in Task 4.

- [ ] **Step 1: Write tests for _listen_loop**

Create `tests/kiosk/talkback/test_listen_loop.py`:

```python
"""Tests for TalkbackController._listen_loop — mic → AEC → VAD → segment queue."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState


def make_controller(**overrides):
    defaults = dict(
        stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
        player=MagicMock(), logger=MagicMock(),
    )
    defaults.update(overrides)
    return TalkbackController(**defaults)


def fake_segment(duration_ms=500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestListenLoop:
    @pytest.mark.asyncio
    async def test_segments_land_in_queue(self):
        """VAD-detected segments are pushed to _segment_queue."""
        ctrl = make_controller()
        ctrl._segment_queue = asyncio.Queue()
        ctrl._running = True
        ctrl.state = TalkbackState.LISTENING

        seg = fake_segment()
        vad = MagicMock()
        vad.process_chunk = MagicMock(side_effect=[
            [seg],  # first chunk produces a segment
            [],     # second chunk produces nothing
        ])

        chunks = [np.zeros(480, dtype=np.float32)] * 2
        chunk_iter = iter(chunks)

        mic = MagicMock()
        mic.stream = MagicMock(return_value=chunk_iter)

        embedder = MagicMock()

        # Stop after 2 chunks
        call_count = 0
        original_process = vad.process_chunk
        def process_and_stop(chunk):
            nonlocal call_count
            result = original_process(chunk)
            call_count += 1
            if call_count >= 2:
                ctrl._running = False
            return result
        vad.process_chunk = process_and_stop

        await ctrl._listen_loop(mic, vad, embedder, np.zeros(192, dtype=np.float32))

        assert ctrl._segment_queue.qsize() == 1
        queued = await ctrl._segment_queue.get()
        assert queued.duration_ms == seg.duration_ms

    @pytest.mark.asyncio
    async def test_aec_applied_during_speaking(self):
        """When state is SPEAKING and AEC is available, mic frames are cleaned."""
        ctrl = make_controller()
        ctrl._segment_queue = asyncio.Queue()
        ctrl._running = True
        ctrl.state = TalkbackState.SPEAKING

        aec = MagicMock()
        cleaned = np.ones(160, dtype=np.float32) * 0.5
        aec.process_frame = MagicMock(return_value=cleaned)
        aec.frame_samples = 160
        ctrl._aec = aec

        player = MagicMock()
        player.get_reference_frame = MagicMock(
            return_value=np.zeros(160, dtype=np.float32)
        )
        ctrl._player = player

        vad = MagicMock()
        vad.process_chunk = MagicMock(return_value=[])

        chunk = np.zeros(480, dtype=np.float32)
        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([chunk]))

        def stop_on_process(c):
            ctrl._running = False
            return []
        vad.process_chunk = stop_on_process

        embedder = MagicMock()
        await ctrl._listen_loop(mic, vad, embedder, np.zeros(192, dtype=np.float32))

        # AEC should have been called 3 times (480 / 160 = 3 frames)
        assert aec.process_frame.call_count == 3

    @pytest.mark.asyncio
    async def test_no_aec_during_listening(self):
        """AEC is not applied when state is LISTENING."""
        ctrl = make_controller()
        ctrl._segment_queue = asyncio.Queue()
        ctrl._running = True
        ctrl.state = TalkbackState.LISTENING

        aec = MagicMock()
        aec.frame_samples = 160
        ctrl._aec = aec

        vad = MagicMock()

        chunk = np.zeros(480, dtype=np.float32)
        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([chunk]))

        def stop_on_process(c):
            ctrl._running = False
            return []
        vad.process_chunk = stop_on_process

        embedder = MagicMock()
        await ctrl._listen_loop(mic, vad, embedder, np.zeros(192, dtype=np.float32))

        aec.process_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_last_speech_at_updated_on_segment(self):
        """_last_speech_at is bumped when a speech segment is detected."""
        ctrl = make_controller()
        ctrl._segment_queue = asyncio.Queue()
        ctrl._running = True
        ctrl.state = TalkbackState.LISTENING
        ctrl._last_speech_at = 0.0

        seg = fake_segment()
        vad = MagicMock()

        chunk = np.zeros(480, dtype=np.float32)
        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([chunk]))

        def process_and_stop(c):
            ctrl._running = False
            return [seg]
        vad.process_chunk = process_and_stop

        embedder = MagicMock()
        await ctrl._listen_loop(mic, vad, embedder, np.zeros(192, dtype=np.float32))

        assert ctrl._last_speech_at > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/kiosk/talkback/test_listen_loop.py -v`
Expected: FAIL — `AttributeError: 'TalkbackController' object has no attribute '_listen_loop'`

- [ ] **Step 3: Implement _listen_loop**

In `modes/talkback/controller.py`, add these imports at the top:

```python
from modes.talkback.aec import AecProcessor
```

Add an `_aec` attribute in `__init__`:

```python
self._aec: Optional[AecProcessor] = None
self._segment_queue: asyncio.Queue = asyncio.Queue()
```

Add the `_listen_loop` method:

```python
async def _listen_loop(
    self,
    mic,
    vad,
    embedder,
    primary_embedding: np.ndarray,
) -> None:
    loop = asyncio.get_event_loop()
    mic_iter = mic.stream()

    def _next_chunk():
        try:
            return next(mic_iter)
        except StopIteration:
            return None

    while self._running:
        chunk = await loop.run_in_executor(None, _next_chunk)
        if chunk is None:
            break

        if self.state == TalkbackState.SPEAKING and self._aec is not None:
            frame_samples = self._aec.frame_samples
            cleaned_frames = []
            for i in range(0, len(chunk), frame_samples):
                frame = chunk[i:i + frame_samples]
                if len(frame) < frame_samples:
                    break
                ref = self._player.get_reference_frame(frame_samples)
                if ref is not None:
                    frame = self._aec.process_frame(frame, ref)
                cleaned_frames.append(frame)
            if cleaned_frames:
                chunk = np.concatenate(cleaned_frames)

        segments = vad.process_chunk(chunk)
        for seg in segments:
            self._last_speech_at = time.monotonic()
            await self._segment_queue.put(seg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/kiosk/talkback/test_listen_loop.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add modes/talkback/controller.py tests/kiosk/talkback/test_listen_loop.py
git commit -m "feat(controller): add _listen_loop for continuous mic → AEC → VAD"
```

---

### Task 3: Add speaker-verified barge-in segment handling

**Files:**
- Modify: `modes/talkback/controller.py`
- Create: `tests/kiosk/talkback/test_multi_turn.py`

This task adds `_handle_segment` which decides what to do with a segment based on state: LISTENING → transcribe, SPEAKING → speaker-verify → barge-in. Does NOT yet wire into `_run_async`.

- [ ] **Step 1: Write tests for segment handling**

Create `tests/kiosk/talkback/test_multi_turn.py`:

```python
"""Tests for multi-turn segment handling in TalkbackController."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.conversation import ConversationManager


def make_controller(**overrides):
    defaults = dict(
        stt=MagicMock(), llm=MagicMock(), tts=MagicMock(),
        player=MagicMock(), logger=MagicMock(),
    )
    defaults.update(overrides)
    return TalkbackController(**defaults)


def fake_segment(duration_ms=500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestHandleSegmentListening:
    @pytest.mark.asyncio
    async def test_listening_segment_transcribed_and_response_started(self):
        """In LISTENING, a segment is transcribed and a response task created."""
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="hello there")
        llm = MagicMock()
        llm.stream = MagicMock(return_value=AsyncIterator([]))

        ctrl = make_controller(stt=stt)
        ctrl.state = TalkbackState.LISTENING
        ctrl._running = True
        ctrl._conversation = ConversationManager(system_prompt="test")
        ctrl._primary_embedding = np.zeros(192, dtype=np.float32)
        ctrl._embedder = MagicMock()
        ctrl._talkback_config = {}

        seg = fake_segment()
        response_task = await ctrl._handle_segment(seg)

        stt.transcribe_segment.assert_awaited_once_with(seg.audio)
        assert "hello there" in [m["content"] for m in ctrl._conversation.get_messages()]
        assert ctrl.state == TalkbackState.SPEAKING

    @pytest.mark.asyncio
    async def test_listening_empty_transcript_ignored(self):
        """In LISTENING, an empty transcript doesn't start a response."""
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="")

        ctrl = make_controller(stt=stt)
        ctrl.state = TalkbackState.LISTENING
        ctrl._running = True
        ctrl._conversation = ConversationManager(system_prompt="test")
        ctrl._primary_embedding = np.zeros(192, dtype=np.float32)
        ctrl._embedder = MagicMock()
        ctrl._talkback_config = {}

        seg = fake_segment()
        response_task = await ctrl._handle_segment(seg)

        assert response_task is None
        assert ctrl.state == TalkbackState.LISTENING


class TestHandleSegmentSpeaking:
    @pytest.mark.asyncio
    async def test_speaking_primary_speaker_triggers_barge_in(self):
        """In SPEAKING, primary speaker segment triggers barge-in."""
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="stop")

        primary_emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        embedder = MagicMock()
        embedder.extract = MagicMock(return_value=primary_emb)

        ctrl = make_controller(stt=stt)
        ctrl.state = TalkbackState.SPEAKING
        ctrl._running = True
        ctrl._conversation = ConversationManager(system_prompt="test")
        ctrl._primary_embedding = primary_emb.copy()
        ctrl._embedder = embedder
        ctrl._talkback_config = {"barge_in": {"enabled": True, "require_speaker_match": True, "min_speech_ms": 100}}
        ctrl._llm.cancel = MagicMock()
        ctrl._player.flush = MagicMock()

        dummy_task = asyncio.create_task(asyncio.sleep(100))
        ctrl._response_task = dummy_task

        seg = fake_segment(duration_ms=500)
        response_task = await ctrl._handle_segment(seg)

        assert dummy_task.cancelled()
        assert ctrl.state == TalkbackState.SPEAKING  # transitions through BARGED_IN back to SPEAKING

    @pytest.mark.asyncio
    async def test_speaking_non_primary_ignored(self):
        """In SPEAKING, non-primary speaker segment is ignored."""
        primary_emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        other_emb = np.zeros(192, dtype=np.float32)
        other_emb[0] = 1.0

        embedder = MagicMock()
        embedder.extract = MagicMock(return_value=other_emb)

        ctrl = make_controller()
        ctrl.state = TalkbackState.SPEAKING
        ctrl._running = True
        ctrl._conversation = ConversationManager(system_prompt="test")
        ctrl._primary_embedding = primary_emb
        ctrl._embedder = embedder
        ctrl._talkback_config = {"barge_in": {"enabled": True, "require_speaker_match": True, "min_speech_ms": 100}}
        ctrl._response_task = asyncio.create_task(asyncio.sleep(100))

        seg = fake_segment(duration_ms=500)
        response_task = await ctrl._handle_segment(seg)

        assert response_task is None
        assert ctrl.state == TalkbackState.SPEAKING

    @pytest.mark.asyncio
    async def test_speaking_short_segment_ignored(self):
        """Segments shorter than min_speech_ms don't trigger barge-in."""
        ctrl = make_controller()
        ctrl.state = TalkbackState.SPEAKING
        ctrl._running = True
        ctrl._conversation = ConversationManager(system_prompt="test")
        ctrl._primary_embedding = np.ones(192, dtype=np.float32)
        ctrl._embedder = MagicMock()
        ctrl._talkback_config = {"barge_in": {"enabled": True, "require_speaker_match": True, "min_speech_ms": 300}}
        ctrl._response_task = asyncio.create_task(asyncio.sleep(100))

        seg = fake_segment(duration_ms=100)  # too short
        response_task = await ctrl._handle_segment(seg)

        assert response_task is None


class AsyncIterator:
    """Helper to create an async iterator from a list."""
    def __init__(self, items):
        self._items = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/kiosk/talkback/test_multi_turn.py -v`
Expected: FAIL — `AttributeError: 'TalkbackController' object has no attribute '_handle_segment'`

- [ ] **Step 3: Implement _handle_segment**

In `modes/talkback/controller.py`, add the import:

```python
from core.speaker.verifier import cosine_similarity
```

Add these attributes in `__init__`:

```python
self._primary_embedding: Optional[np.ndarray] = None
self._embedder = None
self._talkback_config: dict = {}
self._response_task: Optional[asyncio.Task] = None
```

Add the method:

```python
async def _handle_segment(self, segment) -> Optional[asyncio.Task]:
    if self.state == TalkbackState.LISTENING:
        text = await self._stt.transcribe_segment(segment.audio)
        if not text:
            return None
        turn = self._conversation.turn_count + 1
        self._emit("user_turn_complete", {"text": text, "turn_number": turn})
        self._conversation.add_user_turn(text)
        self._transition(TalkbackState.SPEAKING)
        self._emit("turn_started", {"turn_number": turn})
        task = asyncio.create_task(
            self._generate_and_speak(self._conversation, self._talkback_config)
        )
        self._response_task = task
        return task

    elif self.state == TalkbackState.SPEAKING:
        barge_cfg = self._talkback_config.get("barge_in", {})
        if not barge_cfg.get("enabled", True):
            return None
        if segment.duration_ms < barge_cfg.get("min_speech_ms", 120):
            return None

        if barge_cfg.get("require_speaker_match", True):
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self._embedder.extract, segment.audio
            )
            score = cosine_similarity(embedding, self._primary_embedding)
            threshold = self._talkback_config.get(
                "barge_in", {}
            ).get("speaker_threshold", 0.5)
            if score < threshold:
                self._emit("barge_in_rejected", {
                    "score": float(score), "threshold": threshold,
                })
                return None
        else:
            score = 1.0

        # Cancel current response
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        import sounddevice as sd
        sd.stop()
        self._handle_barge_in(primary_score=float(score), speech_ms=segment.duration_ms)

        # Transcribe barge-in speech and start new response
        text = await self._stt.transcribe_segment(segment.audio)
        if not text:
            self._transition(TalkbackState.LISTENING)
            return None

        turn = self._conversation.turn_count + 1
        self._emit("user_turn_complete", {"text": text, "turn_number": turn})
        self._conversation.add_user_turn(text)
        self._transition(TalkbackState.SPEAKING)
        self._emit("turn_started", {"turn_number": turn})
        task = asyncio.create_task(
            self._generate_and_speak(self._conversation, self._talkback_config)
        )
        self._response_task = task
        return task

    return None
```

- [ ] **Step 4: Add _generate_and_speak wrapper**

Rename the existing `_generate_response` to `_generate_and_speak` and make it handle CancelledError gracefully. The method should catch `asyncio.CancelledError` and re-raise after cleanup:

```python
async def _generate_and_speak(
    self, conversation: ConversationManager, config: dict
) -> str:
    try:
        return await self._generate_response(conversation, config)
    except asyncio.CancelledError:
        self._llm.cancel()
        raise
```

Keep `_generate_response` as-is (it already works). `_generate_and_speak` is the cancellable wrapper that tasks use.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/kiosk/talkback/test_multi_turn.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add modes/talkback/controller.py tests/kiosk/talkback/test_multi_turn.py
git commit -m "feat(controller): add _handle_segment for multi-turn and barge-in"
```

---

### Task 4: Rewrite _run_async for multi-turn main loop

**Files:**
- Modify: `modes/talkback/controller.py`
- Modify: `tests/kiosk/talkback/test_controller.py`

This task rewrites `_run_async` to use `_listen_loop` + segment queue + multi-turn loop instead of the single-turn-then-sleep pattern.

- [ ] **Step 1: Write integration test for multi-turn flow**

Add to `tests/kiosk/talkback/test_controller.py`:

```python
class TestMultiTurnRunAsync:
    @pytest.mark.asyncio
    async def test_two_turn_conversation(self):
        """Controller processes first segment, then a second from the mic."""
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(side_effect=["hello", "thanks"])

        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        llm.cancel = MagicMock()

        async def fake_stream(messages):
            yield "response"
        llm.stream = fake_stream

        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=np.zeros(1600, dtype=np.float32))

        player = MagicMock()
        player.enqueue = AsyncMock()
        player.flush = MagicMock()
        player.get_reference_frame = MagicMock(return_value=None)

        logger = MagicMock()
        logger.log = MagicMock()
        logger.start_session = MagicMock()

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=tts, player=player, logger=logger,
        )

        first_seg = make_segment(500)
        second_seg = make_segment(500)

        # Mic yields one chunk that produces second_seg, then stops
        vad = MagicMock()
        call_count = 0
        def vad_process(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [second_seg]
            return []
        vad.process_chunk = vad_process

        embedder = MagicMock()

        chunk = np.zeros(480, dtype=np.float32)
        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([chunk]))

        config = {
            "silence_timeout_s": 1.0,
            "hard_timeout_s": 300.0,
            "watchdog": {"tick_ms": 100},
            "llm": {"system_prompt": "test"},
            "chunker": {"max_chunk_chars": 120},
            "barge_in": {"enabled": False},
        }

        handoff = TalkbackHandoff(
            mic=mic,
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=first_seg,
            config=config,
            vad=vad,
            embedder=embedder,
        )

        # The mic iterator will exhaust (1 chunk), listen_loop will exit,
        # then silence_timeout will fire after 1s
        with patch("modes.talkback.controller.sd"):
            result = await ctrl._run_async(handoff)

        assert result.turns >= 2
        assert stt.transcribe_segment.await_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/kiosk/talkback/test_controller.py::TestMultiTurnRunAsync -v`
Expected: FAIL — current `_run_async` doesn't use the listen loop

- [ ] **Step 3: Rewrite _run_async**

Replace the body of `_run_async` in `modes/talkback/controller.py`:

```python
async def _run_async(self, handoff: TalkbackHandoff) -> TalkbackResult:
    self._started_at = time.monotonic()
    self._last_speech_at = self._started_at
    self._running = True
    self._transition(TalkbackState.LISTENING)
    self._segment_queue = asyncio.Queue()

    await self._llm.close()

    session_id = uuid.uuid4().hex[:12]
    self._logger.start_session(session_id)

    config = handoff.config
    self._talkback_config = config
    self._primary_embedding = handoff.primary_embedding
    self._embedder = handoff.embedder
    silence_timeout = config.get("silence_timeout_s", 10.0)
    hard_timeout = config.get("hard_timeout_s", 300.0)

    aec_cfg = config.get("aec", {})
    if aec_cfg.get("enabled", False):
        try:
            self._aec = AecProcessor(
                sample_rate=config.get("sample_rate_hz", 16000),
                frame_ms=config.get("frame_ms", 10),
            )
        except Exception:
            self._aec = None
    else:
        self._aec = None

    self._conversation = ConversationManager(
        system_prompt=config.get("llm", {}).get(
            "system_prompt", "You are a concise voice assistant.",
        )
    )
    conversation = self._conversation

    if not await self._check_llm_available():
        self._emit("session_ended", {
            "reason": "llm_unavailable", "turns": 0, "total_duration_ms": 0,
        })
        return TalkbackResult(reason="llm_unavailable", turns=0, total_duration_s=0.0)

    watchdog_tick = config.get("watchdog", {}).get("tick_ms", 500) / 1000.0
    watchdog = AsyncWatchdog(
        tick_s=watchdog_tick,
        on_timeout=self._handle_timeout,
        get_silence_duration=lambda: time.monotonic() - self._last_speech_at,
        get_session_duration=lambda: time.monotonic() - self._started_at,
        silence_timeout_s=silence_timeout,
        hard_timeout_s=hard_timeout,
    )

    self._emit("handoff_to_talkback", {
        "primary_embedding_norm": float(np.linalg.norm(handoff.primary_embedding)),
    })

    # Process the first speech segment synchronously
    first_text = await self._stt.transcribe_segment(handoff.first_segment.audio)
    response_task = None

    if first_text:
        self._last_speech_at = time.monotonic()
        self._emit("user_turn_complete", {"text": first_text, "turn_number": 1})
        conversation.add_user_turn(first_text)
        self._transition(TalkbackState.SPEAKING)
        self._emit("turn_started", {"turn_number": 1})
        response_task = asyncio.create_task(
            self._generate_and_speak(conversation, config)
        )
        self._response_task = response_task

    # Start continuous mic listener
    listen_task = asyncio.create_task(
        self._listen_loop(handoff.mic, handoff.vad, handoff.embedder, handoff.primary_embedding)
    )
    watchdog.start()

    try:
        while self._running:
            # Check if response task completed
            if response_task and response_task.done():
                try:
                    assistant_text = response_task.result()
                    if assistant_text:
                        conversation.add_assistant_turn(assistant_text)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                self._transition(TalkbackState.LISTENING)
                response_task = None
                self._response_task = None

            # Check for new speech segment
            try:
                segment = self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)
                continue

            new_task = await self._handle_segment(segment)
            if new_task is not None:
                response_task = new_task

            if self._run_result is not None:
                break
    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        if response_task and not response_task.done():
            response_task.cancel()
            try:
                await response_task
            except asyncio.CancelledError:
                pass
        await watchdog.stop()

    await self._llm.close()

    if self._run_result is None:
        self._run_result = TalkbackResult(
            reason="stopped",
            turns=conversation.turn_count,
            total_duration_s=time.monotonic() - self._started_at,
        )

    self._transition(TalkbackState.IDLE)
    self._emit("session_ended", {
        "reason": self._run_result.reason,
        "turns": self._run_result.turns,
        "total_duration_ms": self._run_result.total_duration_s * 1000,
    })

    return self._run_result
```

- [ ] **Step 4: Run the multi-turn test**

Run: `pytest tests/kiosk/talkback/test_controller.py::TestMultiTurnRunAsync -v`
Expected: PASS

- [ ] **Step 5: Run the full talkback test suite to check for regressions**

Run: `pytest tests/kiosk/talkback/ -v`
Expected: All existing tests PASS (some may need minor fixes for the new `_run_async` signature; fix any regressions)

- [ ] **Step 6: Commit**

```bash
git add modes/talkback/controller.py tests/kiosk/talkback/test_controller.py
git commit -m "feat(controller): multi-turn main loop with listen_loop + segment queue"
```

---

### Task 5: Wire AEC initialization from config

**Files:**
- Modify: `modes/talkback/controller.py`
- Create: `tests/kiosk/talkback/test_aec_wiring.py`

AEC creation in `_run_async` (Task 4) can fail if the shim isn't built. Verify it degrades gracefully.

- [ ] **Step 1: Write tests for AEC initialization**

Create `tests/kiosk/talkback/test_aec_wiring.py`:

```python
"""Tests for AEC initialization in the multi-turn controller."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController, TalkbackState
from modes.talkback.handoff import TalkbackHandoff


def make_segment(duration_ms=500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class TestAecWiring:
    @pytest.mark.asyncio
    async def test_aec_disabled_in_config(self):
        """When aec.enabled is false, _aec stays None."""
        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="hi")

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=MagicMock(), player=MagicMock(), logger=MagicMock(),
        )

        handoff = TalkbackHandoff(
            mic=MagicMock(stream=MagicMock(return_value=iter([]))),
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=make_segment(),
            config={"aec": {"enabled": False}, "silence_timeout_s": 0.1, "hard_timeout_s": 1.0},
            vad=MagicMock(process_chunk=MagicMock(return_value=[])),
            embedder=MagicMock(),
        )

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._run_async(handoff)

        assert ctrl._aec is None

    @pytest.mark.asyncio
    async def test_aec_enabled_but_init_fails_degrades(self):
        """When AEC init throws, _aec is None and controller continues."""
        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(return_value="hi")

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=MagicMock(), player=MagicMock(), logger=MagicMock(),
        )

        handoff = TalkbackHandoff(
            mic=MagicMock(stream=MagicMock(return_value=iter([]))),
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=make_segment(),
            config={"aec": {"enabled": True}, "silence_timeout_s": 0.1, "hard_timeout_s": 1.0},
            vad=MagicMock(process_chunk=MagicMock(return_value=[])),
            embedder=MagicMock(),
        )

        with patch("modes.talkback.controller.AecProcessor", side_effect=RuntimeError("no shim")):
            with patch("modes.talkback.controller.sd"):
                result = await ctrl._run_async(handoff)

        assert ctrl._aec is None
        assert result.reason in ("silence_timeout", "stopped")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/kiosk/talkback/test_aec_wiring.py -v`
Expected: PASS (the AEC init logic was already written in Task 4)

- [ ] **Step 3: Commit**

```bash
git add tests/kiosk/talkback/test_aec_wiring.py
git commit -m "test(controller): AEC wiring and graceful degradation"
```

---

### Task 6: Update existing tests for new handoff/controller shape

**Files:**
- Modify: `tests/kiosk/talkback/test_controller.py`
- Modify: `tests/kiosk/talkback/test_barge_in.py`

After Tasks 1-4, some existing tests may construct TalkbackHandoff or TalkbackController in ways that no longer match. This task fixes any regressions.

- [ ] **Step 1: Run the full talkback test suite**

Run: `pytest tests/kiosk/talkback/ -v`
Note all failures.

- [ ] **Step 2: Fix each failure**

Common fixes:
- `TalkbackHandoff(...)` calls missing `vad=` and `embedder=` → add `vad=MagicMock(), embedder=MagicMock()`
- Tests that called `_generate_response` directly may need updating if it was renamed/wrapped
- `_handle_barge_in` tests should still pass since that method is unchanged

- [ ] **Step 3: Run full suite again**

Run: `pytest tests/kiosk/talkback/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/kiosk/talkback/
git commit -m "fix(tests): update existing tests for multi-turn controller shape"
```

---

### Task 7: End-to-end multi-turn integration test

**Files:**
- Create: `tests/kiosk/talkback/test_e2e_multi_turn.py`

A self-contained test that exercises the full `_run_async` path with mocked backends: two user turns, two assistant responses, timeout-based exit.

- [ ] **Step 1: Write the end-to-end test**

Create `tests/kiosk/talkback/test_e2e_multi_turn.py`:

```python
"""End-to-end multi-turn test with mocked backends."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.controller import TalkbackController
from modes.talkback.handoff import TalkbackHandoff, TalkbackResult


def make_segment(duration_ms=500.0):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=np.random.randn(samples).astype(np.float32) * 0.1,
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


class AsyncTokenStream:
    def __init__(self, tokens):
        self._tokens = tokens
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        return self._tokens.pop(0)


class TestE2EMultiTurn:
    @pytest.mark.asyncio
    async def test_two_turns_then_silence_timeout(self):
        transcripts = iter(["What is Python?", "Thanks!"])
        stt = MagicMock()
        stt.transcribe_segment = AsyncMock(side_effect=lambda audio: next(transcripts))

        responses = [
            ["Python ", "is a ", "programming language."],
            ["You're ", "welcome!"],
        ]
        response_iter = iter(responses)
        llm = MagicMock()
        llm.ping = AsyncMock(return_value=True)
        llm.close = AsyncMock()
        llm.cancel = MagicMock()
        def make_stream(messages):
            return AsyncTokenStream(next(response_iter))
        llm.stream = make_stream

        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=np.zeros(1600, dtype=np.float32))

        player = MagicMock()
        player.enqueue = AsyncMock()
        player.flush = MagicMock()
        player.get_reference_frame = MagicMock(return_value=None)

        logger = MagicMock()
        logger.log = MagicMock()
        logger.start_session = MagicMock()

        ctrl = TalkbackController(
            stt=stt, llm=llm, tts=tts, player=player, logger=logger,
        )

        first_seg = make_segment(500)
        second_seg = make_segment(500)

        # VAD produces second_seg on first chunk, nothing after
        vad = MagicMock()
        vad_call = 0
        def vad_process(chunk):
            nonlocal vad_call
            vad_call += 1
            if vad_call == 1:
                return [second_seg]
            return []
        vad.process_chunk = vad_process

        mic = MagicMock()
        mic.stream = MagicMock(return_value=iter([np.zeros(480, dtype=np.float32)]))

        config = {
            "silence_timeout_s": 0.5,
            "hard_timeout_s": 60.0,
            "watchdog": {"tick_ms": 100},
            "llm": {"system_prompt": "You are helpful."},
            "chunker": {"max_chunk_chars": 200},
            "barge_in": {"enabled": False},
            "aec": {"enabled": False},
        }

        handoff = TalkbackHandoff(
            mic=mic,
            primary_embedding=np.zeros(192, dtype=np.float32),
            first_segment=first_seg,
            config=config,
            vad=vad,
            embedder=MagicMock(),
        )

        with patch("modes.talkback.controller.sd"):
            result = await ctrl._run_async(handoff)

        assert result.turns == 2
        assert result.reason == "silence_timeout"

        # Verify both transcripts were processed
        assert stt.transcribe_segment.await_count == 2

        # Verify conversation has 4 messages (2 user + 2 assistant)
        msgs = ctrl._conversation.get_messages()
        # msgs[0] is system prompt
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is Python?"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "Thanks!"
        assert msgs[4]["role"] == "assistant"

        # Verify events logged
        log_events = [call[0][0] for call in logger.log.call_args_list]
        assert "handoff_to_talkback" in log_events
        assert log_events.count("user_turn_complete") == 2
        assert "session_ended" in log_events
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/kiosk/talkback/test_e2e_multi_turn.py -v`
Expected: PASS

- [ ] **Step 3: Run full talkback test suite one final time**

Run: `pytest tests/kiosk/talkback/ tests/core/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/kiosk/talkback/test_e2e_multi_turn.py
git commit -m "test(controller): end-to-end multi-turn conversation integration test"
```

---

### Task 8: Final integration — run full suite and clean up

**Files:**
- Possibly modify: any file with minor issues found during final testing

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/kiosk/ tests/core/ -v`
Expected: All PASS (120+ tests)

- [ ] **Step 2: Verify no import issues in production path**

Run: `cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad && python3 -c "from modes.talkback.controller import TalkbackController; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Final commit if any cleanup was needed**

```bash
git add -u
git commit -m "chore: multi-turn cleanup and final fixes"
```
