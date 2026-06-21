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
