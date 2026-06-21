"""IngestionWorker — the mic-side reflex pipeline.

Reads mic chunks via run_in_executor over MicrophoneStream.stream() (blocking
generator), runs AEC per-frame against the playback reference ring (read-only;
the ring is filled by the Playback worker under its lock — spec section 10
invariant 3), runs Silero VAD, computes RMS, and routes events by the Director's
current state (read-only, spec section 3):

  SPEAKING   -> NearFieldOnset on near-field voiced onset (controller.py:839-849)
  LISTENING  -> SegmentEndpointed (Smart Turn endpoint_prob via executor)
  EVALUATING -> InterjectionSegment (ECAPA speaker_score via executor)

Plan 02 stubs (Plan 05 replaces): is_target is hard-coded True; speaker_score is
a synchronous ECAPA cosine via run_in_executor (off the synchronous decision
path). The captured audio is staged into the SttWorker so a later
TranscribeUserTurn/TranscribeInterjection has the right buffer."""

import asyncio
import os
import sys

import numpy as np

from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director.state import State
from modes.director import events as E

_DIAG = bool(os.environ.get("TVAD_DIAG"))


def _diag(msg: str) -> None:
    if _DIAG:
        print(f"[DIAG ingest] {msg}", file=sys.stderr, flush=True)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


class IngestionWorker:
    def __init__(self, mic, vad, aec, turn_detector, embedder,
                 primary_embedding, stt_worker, playback, bus: EventBus,
                 cfg: DirectorConfig, proximity_rms: float, state_getter,
                 score_fn):
        self._mic = mic
        self._vad = vad
        self._aec = aec
        self._turn = turn_detector
        self._embedder = embedder
        self._primary = primary_embedding
        self._stt = stt_worker
        self._playback = playback
        self._bus = bus
        self._cfg = cfg
        self._proximity_rms = proximity_rms
        self._state_getter = state_getter
        self._score_fn = score_fn          # cosine(embedding, primary) -> float
        self._running = False
        self._ducked_onset = False         # one onset per speech run (controller.py:842)

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        self._running = True
        _diag("started")
        if _DIAG:
            async def _buf_probe():
                while self._running:
                    await asyncio.sleep(1.0)
                    buf = getattr(self._mic, "_buffer", None)
                    _diag(f"mic buffer_len={len(buf) if buf is not None else '?'} "
                          f"mic_running={getattr(self._mic, '_running', '?')}")
            asyncio.create_task(_buf_probe())

        # Non-blocking drain loop: pull all available chunks from the mic buffer,
        # process them, then yield. No run_in_executor over a blocking generator
        # (that Event-handshake producer/consumer race deadlocked under the
        # Director's concurrency — the buffer would peg at maxlen and starve).
        n = 0
        while self._running:
            chunks = self._mic.read_available()
            if not chunks:
                await asyncio.sleep(0.005)
                continue
            for chunk in chunks:
                n += 1
                state = self._state_getter()
                if _DIAG and n % 100 == 0:
                    _diag(f"{n} chunks read, state={state.name}, "
                          f"is_speaking={getattr(self._vad, 'is_speaking', '?')}")
                chunk = self._apply_aec(chunk, state)
                segments = self._vad.process_chunk(chunk)
                if segments:
                    _diag(f"VAD produced {len(segments)} segment(s) "
                          f"dur={[round(s.duration_ms) for s in segments]} state={state.name}")
                await self._maybe_onset(chunk, state)
                for seg in segments:
                    await self._on_segment(seg, self._state_getter())
            await asyncio.sleep(0)   # cooperative yield to the loop after a batch

    def _apply_aec(self, chunk: np.ndarray, state: State) -> np.ndarray:
        """Per-frame AEC during playback (controller.py:819-832). Reads the
        reference ring the Playback worker fills; never records it here."""
        if state is not State.SPEAKING or self._aec is None:
            return chunk
        fs = self._aec.frame_samples
        cleaned = []
        for i in range(0, len(chunk), fs):
            frame = chunk[i:i + fs]
            if len(frame) < fs:
                break
            ref = self._playback.get_reference_frame(fs)
            if ref is not None:
                frame = self._aec.process_frame(frame, ref)
            cleaned.append(frame)
        return np.concatenate(cleaned) if cleaned else chunk

    async def _maybe_onset(self, chunk: np.ndarray, state: State) -> None:
        """Duck-at-onset reflex (controller.py:839-849): emit NearFieldOnset on
        the first voiced, near-field chunk of a speech run during SPEAKING."""
        if state is not State.SPEAKING:
            self._ducked_onset = False
            return
        if self._ducked_onset or getattr(self._vad, "is_speaking", False) is not True:
            return
        rms = _rms(chunk)
        if rms >= self._proximity_rms:
            self._ducked_onset = True
            await self._bus.emit(E.NearFieldOnset(rms=rms, is_target=True))

    async def _on_segment(self, seg, state: State) -> None:
        rms = _rms(seg.audio)
        if state is State.LISTENING:
            prob = await self._endpoint_prob(seg.audio)
            self._stt.set_pending_user_audio(seg.audio)
            await self._bus.emit(E.SegmentEndpointed(
                duration_ms=seg.duration_ms, rms=rms,
                is_target=True, endpoint_prob=prob,
            ))
        elif state is State.EVALUATING:
            score = await self._speaker_score(seg.audio)
            self._stt.set_pending_interjection_audio(seg.audio)
            await self._bus.emit(E.InterjectionSegment(
                duration_ms=seg.duration_ms, rms=rms,
                is_target=True, speaker_score=score,
            ))
        # SPEAKING/THINKING/IDLE: onset handled separately; segments are ignored.

    async def _endpoint_prob(self, audio: np.ndarray) -> float:
        loop = asyncio.get_event_loop()
        return float(await loop.run_in_executor(
            None, self._turn.endpoint_prob, audio, 16000))

    async def _speaker_score(self, audio: np.ndarray) -> float:
        """ECAPA speaker_score off the synchronous path (Plan 05 swaps for pVAD)."""
        if self._embedder is None or self._primary is None:
            return 1.0
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embedder.extract, audio)
        return float(self._score_fn(embedding, self._primary))
