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
        self._llm_loop_bound = False

    async def execute(self, command) -> None:
        if isinstance(command, C.StartGeneration):
            await self._start(command)
        elif isinstance(command, C.Cut):
            await self._cut(command)

    async def _rebind_llm_once(self) -> None:
        """Drop any aiohttp session bound to a PREVIOUS (now-closed) event loop —
        e.g. the startup `asyncio.run(llm.ping())` loop (kiosk.py) — so the first
        stream() binds a fresh session to THIS runtime's loop. Without this,
        LlmClient._ensure_session reuses the dead-loop session and stream() hangs.
        Mirrors TalkbackController._run_async (controller.py:264). Runs ONCE per
        GenerationWorker (a new one is built per session)."""
        if self._llm_loop_bound:
            return
        self._llm_loop_bound = True
        close = getattr(self._llm, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Close the LLM session at session teardown (it is bound to THIS loop
        now, so closing is clean) — avoids leaking one aiohttp session per
        session across the kiosk's many wake cycles."""
        close = getattr(self._llm, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    async def _start(self, cmd: C.StartGeneration) -> None:
        """Run one generation to completion. Caller (runtime) awaits this; the
        runtime keeps draining the bus, so emitted FirstTtsFrame/ReplyComplete
        are processed in order."""
        await self._rebind_llm_once()
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
