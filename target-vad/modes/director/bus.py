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
