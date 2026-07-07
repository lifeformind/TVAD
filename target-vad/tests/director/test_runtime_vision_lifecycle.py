import asyncio
from modes.director.runtime import DirectorRuntime
from modes.director import commands as C


class _Spy:
    def __init__(self): self.started = None; self.stopped = False
    def start(self, loop): self.started = loop
    def stop(self): self.stopped = True


class _NoopIngestion:
    async def run(self):
        await asyncio.sleep(3600)
    def stop(self): pass


class _NoopPlayback:
    async def drain(self): pass
    def close(self): pass
    async def execute(self, cmd): pass


class _NoopGen:
    async def aclose(self): pass
    async def execute(self, cmd): pass


class _ImmediateWatchdog:
    def start(self): pass
    def request_stop(self, reason): pass
    async def stop(self): pass


class _Director:
    class _Ctx:
        class conversation:
            turn_count = 0
        conversation = conversation()
        proximity_rms = 0.0
        owner_bearing = None
    ctx = _Ctx()

    def dispatch(self, event):
        return []


def test_runtime_starts_and_stops_vision():
    vision = _Spy()
    bus_events = [C.EndSession("test")]   # one event then end

    class _Bus:
        async def get(self):
            return bus_events.pop(0)

    director = _Director()

    rt = DirectorRuntime(director=director, bus=_Bus(),
                         watchdog=_ImmediateWatchdog(), ingestion=_NoopIngestion(),
                         stt_worker=None, generation=_NoopGen(),
                         playback=_NoopPlayback(), clock=lambda: 0.0, vision=vision)

    # Make dispatch return the EndSession command so the loop exits
    rt._director.dispatch = lambda e: [e]

    async def _drive():
        await rt.run_async()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drive())
    finally:
        loop.close()
    assert vision.started is not None      # started with the running loop
    assert vision.stopped is True          # stopped in teardown
