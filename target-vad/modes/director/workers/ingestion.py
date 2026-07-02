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
                 score_fn, pvad=None, safety_worker=None):
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
        self._pvad = pvad                  # Plan 05 PvadWorker; None -> is_target=True
        self._safety = safety_worker       # Director-09 SafetyNetWorker; None -> no staging
        self._chunk_frames = []            # most recent chunk's pVAD SpeakerFrames
        self._seg_frames = []              # pVAD frames over the current voiced run
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
        try:
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
                    self._run_pvad(chunk, n)
                    segments = self._vad.process_chunk(chunk)
                    if segments:
                        _diag(f"VAD produced {len(segments)} segment(s) "
                              f"dur={[round(s.duration_ms) for s in segments]} state={state.name}")
                    await self._maybe_onset(chunk, state)
                    for seg in segments:
                        _diag(f"on_segment ENTER state={self._state_getter().name} "
                              f"dur={round(seg.duration_ms)}")
                        await self._on_segment(seg, self._state_getter())
                        _diag("on_segment EXIT")
                await asyncio.sleep(0)   # cooperative yield to the loop after a batch
        except asyncio.CancelledError:
            raise
        except Exception as exc:        # a crash here silently stops the mic forever
            import traceback
            _diag(f"CRASHED at chunk {n}: {type(exc).__name__}: {exc}")
            print(f"[DIAG ingest] traceback:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

    def _run_pvad(self, chunk: np.ndarray, n: int) -> None:
        """Run the per-chunk pVAD (Plan 05) to keep the streaming state advancing
        and accumulate the voiced run's frames for the segment-level is_target.
        Must run on EVERY chunk (state is carried) but only voiced frames feed the
        segment decision, so inter-run silence doesn't dilute it."""
        if self._pvad is None:
            self._chunk_frames = []
            return
        self._chunk_frames = self._pvad.process(chunk, ts=float(n))
        if getattr(self._vad, "is_speaking", False):
            self._seg_frames.extend(self._chunk_frames)

    def _target_from(self, frames) -> tuple:
        """(is_target, speaker_score) from pVAD frames. No pVAD -> (True, None)
        so the caller keeps the legacy is_target=True / ECAPA score. With pVAD but
        no frames -> fail-open is_target (don't drop the user on no signal) with a
        0.0 score (a no-signal interjection won't clear the speaker gate)."""
        if self._pvad is None:
            return True, None
        if not frames:
            return True, 0.0
        n_target = sum(1 for f in frames if f.is_target)
        is_target = n_target * 2 >= len(frames)          # majority of the run
        score = max(f.confidence for f in frames)
        return is_target, score

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
            is_target, _ = self._target_from(self._chunk_frames)
            await self._bus.emit(E.NearFieldOnset(rms=rms, is_target=is_target))

    async def _on_segment(self, seg, state: State) -> None:
        rms = _rms(seg.audio)
        is_target, pvad_score = self._target_from(self._seg_frames)
        self._seg_frames = []                            # consume the run
        if state is State.LISTENING:
            prob = await self._endpoint_prob(seg.audio)
            if self._safety is not None:
                # Overwrite-last staging (same discipline as the STT pending buffer):
                # if two segments endpoint before the runtime drains the first event,
                # the later segment's audio can be consumed by the earlier segment's
                # AccumulateSpeakerAudio. Narrow window; the 1-of-3 smoother + 2s
                # windows + the eject rms check absorb a single mis-staged segment.
                # Structural fix (seq echo through event->command->worker) is a
                # recorded fast-follow.
                self._safety.set_pending_audio(seg.audio)
            self._stt.set_pending_user_audio(seg.audio)
            await self._bus.emit(E.SegmentEndpointed(
                duration_ms=seg.duration_ms, rms=rms,
                is_target=is_target, endpoint_prob=prob,
            ))
        elif state is State.EVALUATING:
            # pVAD confidence is the primary speaker_score; ECAPA (off the hot
            # path) becomes the SafetyNet's job. No pVAD -> legacy ECAPA score.
            score = pvad_score if pvad_score is not None \
                else await self._speaker_score(seg.audio)
            if self._safety is not None:
                self._safety.set_pending_audio(seg.audio)
            self._stt.set_pending_interjection_audio(seg.audio)
            await self._bus.emit(E.InterjectionSegment(
                duration_ms=seg.duration_ms, rms=rms,
                is_target=is_target, speaker_score=score,
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
