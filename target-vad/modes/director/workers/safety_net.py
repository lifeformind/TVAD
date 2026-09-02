"""SafetyNetWorker — executes AccumulateSpeakerAudio (Director-09).

The reducer decides WHICH segments count (served/plausibly-owner only, spec
s3.2); this worker only buffers the last-staged audio and, when a verify window
fills, embeds it OFF the event loop (run_in_executor, ECAPA ~108ms p95) and
emits SpeakerWindowVerdict. No decisions here — the reducer owns the ladder.
An empty pending buffer is a no-op: the assembly factory's seeded first segment
(the enrollment utterance) is deliberately never staged, so window 1 is real
post-enrollment speech (spec s3.2 seed exclusion). Staging is seq-matched: the
command's seq (echoed from the segment event) must equal the staged seq, so a
stale command can never consume a LATER segment's audio — in particular a
D08-rejected bystander segment (staged but never commanded) can't be pulled
into the hijack buffer by an earlier accepted segment's command."""

import asyncio

from modes.director.bus import EventBus
from modes.director import commands as C
from modes.director import events as E


class SafetyNetWorker:
    def __init__(self, safety_net, bus: EventBus):
        self._net = safety_net
        self._bus = bus
        self._pending = None

    def set_pending_audio(self, audio, seq: int = 0) -> None:
        self._pending = (seq, audio)

    async def execute(self, command) -> None:
        if not isinstance(command, C.AccumulateSpeakerAudio):
            return
        if self._pending is None or self._pending[0] != command.seq:
            return                    # stale command; staged audio waits for ITS command
        (_, audio), self._pending = self._pending, None
        if audio is None or len(audio) == 0:
            return
        loop = asyncio.get_running_loop()
        verdicts = await loop.run_in_executor(None, self._drain, audio)
        for v in verdicts:
            await self._bus.emit(E.SpeakerWindowVerdict(
                score=v.score, smoother_ok=v.smoother_ok,
                window_rms=v.window_rms, norm_score=v.norm_score))

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
