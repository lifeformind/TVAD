"""Async watchdog — ticks independently of chunk arrival to fire timeouts."""

import asyncio
from typing import Callable


class AsyncWatchdog:
    """Periodically checks silence/hard timeouts for TalkbackController."""

    def __init__(
        self,
        tick_s: float,
        on_timeout: Callable[[str], None],
        get_silence_duration: Callable[[], float],
        get_session_duration: Callable[[], float],
        silence_timeout_s: float,
        hard_timeout_s: float,
    ):
        self._tick_s = tick_s
        self._on_timeout = on_timeout
        self._get_silence = get_silence_duration
        self._get_session = get_session_duration
        self._silence_timeout = silence_timeout_s
        self._hard_timeout = hard_timeout_s
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._tick_s)
            session_dur = self._get_session()
            if session_dur >= self._hard_timeout:
                self._on_timeout("hard_timeout")
                return
            silence_dur = self._get_silence()
            if silence_dur >= self._silence_timeout:
                self._on_timeout("silence_timeout")
                return
