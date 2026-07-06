"""SafetyNetWorker — executes AccumulateSpeakerAudio (Director-09).

The reducer decides WHICH segments count (served/plausibly-owner only, spec
s3.2); this worker only buffers the last-staged audio and, when a verify window
fills, embeds it OFF the event loop (run_in_executor, ECAPA ~108ms p95) and
emits SpeakerWindowVerdict. No decisions here — the reducer owns the ladder.
An empty pending buffer is a no-op: the assembly factory's seeded first segment
(the enrollment utterance) is deliberately never staged, so window 1 is real
post-enrollment speech (spec s3.2 seed exclusion)."""

import asyncio

from modes.director.bus import EventBus
from modes.director import commands as C
from modes.director import events as E


class SafetyNetWorker:
    def __init__(self, safety_net, bus: EventBus):
        self._net = safety_net
        self._bus = bus
        self._pending = None

    def set_pending_audio(self, audio) -> None:
        self._pending = audio

    async def execute(self, command) -> None:
        if not isinstance(command, C.AccumulateSpeakerAudio):
            return
        audio, self._pending = self._pending, None
        if audio is None or len(audio) == 0:
            return
        loop = asyncio.get_event_loop()
        verdicts = await loop.run_in_executor(None, self._drain, audio)
        for v in verdicts:
            await self._bus.emit(E.SpeakerWindowVerdict(
                score=v.score, smoother_ok=v.smoother_ok,
                window_rms=v.window_rms))

    def _drain(self, audio):
        """Accumulate, then consume EVERY full window (a long turn can complete
        more than one). Runs in the executor: accumulate + embed off the loop."""
        self._net.accumulate(audio, is_target=True)
        out = []
        while True:
            v = self._net.maybe_verify()
            if v is None:
                return out
            out.append(v)
