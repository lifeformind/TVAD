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
