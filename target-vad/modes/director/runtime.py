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
        await self._generation.aclose()     # close the LLM session (no leak per session)
        await self._playback.drain()        # await in-flight write BEFORE close
        await self._watchdog.stop()
        self._playback.close()
