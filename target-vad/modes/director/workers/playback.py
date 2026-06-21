"""PlaybackWorker — sole owner of the OutputStream and the AEC reference ring.

The 6 race-fixed teardown invariants from spec section 10 are copied VERBATIM
from modes/talkback/controller.py (cross-thread sd.write/stream.close segfault
PortAudio). The Ingestion worker only READS the reference ring via
get_reference_frame; record_reference + sd.write stay co-located here under ONE
_write_lock. Also executes Duck/Restore (gain) and SpeakNudge (direct TTS of
"Are you still there?", spec section 5 — no LLM round-trip)."""

import asyncio
import threading

import numpy as np

from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director import commands as C

# 30ms playback frames (controller.py:172): small enough for ~30ms duck latency.
PLAYBACK_FRAME_SAMPLES = 480

NUDGE_TEXT = "Are you still there?"


class PlaybackWorker:
    def __init__(self, tts, player, cfg: DirectorConfig, bus: EventBus):
        self._tts = tts
        self._player = player
        self._cfg = cfg
        self._bus = bus
        self._out_stream = None
        self._running = False
        self._gain = 1.0
        # Invariant 2: generation counter, aligned with Context.gen_id; checked
        # before each frame and inside the lock. Invariant 6: _play_future is the
        # running write job and is NOT cleared on barge-in.
        self._play_gen = 0
        self._play_future = None
        # Invariant 1: held around every write AND around stream close.
        self._write_lock = threading.Lock()

    @property
    def gain(self) -> float:
        return self._gain

    def open(self, out_stream) -> None:
        """Inject the already-started OutputStream (runtime/WakeGate owns device
        creation; tests inject a MagicMock)."""
        self._out_stream = out_stream
        self._running = True

    def set_gen(self, gen_id: int) -> None:
        """Align _play_gen with the Context gen_id at the start of a generation."""
        self._play_gen = gen_id

    async def execute(self, command) -> None:
        if isinstance(command, C.Duck):
            self._gain = command.level
        elif isinstance(command, C.Restore):
            self._gain = 1.0
        elif isinstance(command, C.SpeakNudge):
            await self._speak_nudge()

    async def _speak_nudge(self) -> None:
        audio = await self._tts.synthesize(NUDGE_TEXT)
        if audio is not None and len(audio) > 0:
            await self.play(audio, gen_id=self._play_gen)

    async def play(self, audio: np.ndarray, gen_id: int) -> None:
        """Play one utterance in an executor, tracking the job so a barge-in or
        shutdown can wait for it before the stream is touched again. _play_future
        is intentionally NOT cleared on cancellation (invariant 6): cancelling the
        await does not stop the executor thread (run_in_executor jobs can't be
        cancelled once running), so the reference must survive for drain()."""
        self._play_gen = gen_id
        self._play_future = asyncio.get_event_loop().run_in_executor(
            None, self._play_audio, audio, gen_id
        )
        await self._play_future

    async def drain(self) -> None:
        """Stop in-flight playback and wait for the write thread to exit
        (invariant 4). Bumping the generation makes _play_audio break at its next
        frame; awaiting the shielded future guarantees no sd.write is running when
        the stream is closed (concurrent PortAudio calls across threads segfault).
        _play_future is NOT cleared (invariant 6)."""
        self._play_gen += 1
        fut = self._play_future
        if fut is not None:
            try:
                await asyncio.shield(fut)
            except Exception:
                pass

    def _play_audio(self, audio: np.ndarray, gen: int) -> None:
        """Write one utterance frame-by-frame (blocking; runs in an executor).
        Applies the current gain (ducking), records each played (post-gain) frame
        as the AEC reference, bails if a barge-in superseded this generation or the
        session ended (invariants 2 & 3)."""
        if self._out_stream is None or len(audio) == 0:
            return
        frame = PLAYBACK_FRAME_SAMPLES
        for i in range(0, len(audio), frame):
            if not self._running or gen != self._play_gen:
                break
            gained = (audio[i:i + frame] * self._gain).astype(np.float32)
            # Invariant 1: the lock makes write and close mutually exclusive;
            # re-check the stream/gen inside it (teardown may have closed while we
            # waited). Invariant 3: record + write co-located under ONE lock.
            with self._write_lock:
                if self._out_stream is None or gen != self._play_gen:
                    break
                if self._player is not None:
                    self._player.record_reference(gained)
                try:
                    self._out_stream.write(gained)
                except Exception:
                    break

    def close(self) -> None:
        """Stop playback and close the output stream — synchronous, lock-guarded,
        idempotent (invariants 1 & 5). Safe even when a KeyboardInterrupt has
        killed the loop (the async drain can't run then)."""
        self._running = False
        self._play_gen += 1
        with self._write_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None
