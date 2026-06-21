"""Director runnable-assembly seam (Plan 03, prompt CRITICAL note).

The WakeGate makes ONE blocking call `runtime.run(handoff)`, but Plan-02's
DirectorRuntime takes pre-built workers in __init__ and its run() takes NO
handoff. This module bridges the two: build_director_runtime(handoff, ...)
constructs the REAL heavy components and all four workers from a DirectorHandoff,
mirroring the original TalkbackController._run_async assembly (controller.py:
257-392) — AEC, the persistent OutputStream, proximity_rms auto-calibration from
the first enrollment segment, a fresh ConversationManager, and the silence/hard
timeout pair — and wires them into a Plan-02 DirectorRuntime.

It returns a DirectorRuntimeFactory whose .run(handoff) seeds the first segment
(captured by the WakeGate, so the conversation's opening turn enters the loop)
and then drives the real runtime to a DirectorResult. The first segment is staged
into the SttWorker and announced as ONE SegmentEndpointed event so the Director
answers it exactly like any later LISTENING turn (spec section 6).
"""

import asyncio
import time
from typing import Any, Callable, Optional

import numpy as np

from core.speaker.verifier import cosine_similarity
from modes.director.bus import EventBus
from modes.director.config import DirectorConfig
from modes.director.director import Director
from modes.director.runtime import DirectorRuntime
from modes.director.state import State
from modes.director.watchdog import AsyncWatchdog
from modes.director.workers.generation import GenerationWorker
from modes.director.workers.ingestion import IngestionWorker
from modes.director.workers.playback import PlaybackWorker
from modes.director.workers.stt_worker import SttWorker
from modes.director import events as E
from modes.talkback.chunker import SentenceChunker
from modes.talkback.conversation import ConversationManager
from modes.talkback.endpointing import NullTurnDetector

try:
    from modes.talkback.aec import AecProcessor
except Exception:  # pragma: no cover - AEC backend optional
    AecProcessor = None  # type: ignore[assignment,misc]

try:
    from modes.talkback.endpointing import SmartTurnDetector
except Exception:  # pragma: no cover - onnxruntime/pipecat optional
    SmartTurnDetector = None  # type: ignore[assignment,misc]


def _director_config_from(tb_cfg: dict) -> DirectorConfig:
    """Map the kiosk.talkback.* config onto the frozen DirectorConfig. Pulls the
    same keys the reducer's thresholds came from (spec section 5/6)."""
    barge = tb_cfg.get("barge_in", {})
    return DirectorConfig(
        silence_timeout_s=tb_cfg.get("silence_timeout_s", 30.0),
        hard_timeout_s=tb_cfg.get("hard_timeout_s", 300.0),
        endpoint_threshold=tb_cfg.get("turn_gate", {}).get("endpoint_threshold", 0.5),
        min_speech_ms=barge.get("min_speech_ms", 120.0),
        verify_window_ms=barge.get("verify_window_ms", 700.0),
        speaker_threshold=barge.get("speaker_threshold", 0.20),
        duck_level=barge.get("duck_level", 0.15),
    )


def _calibrate_proximity_rms(first_segment, tb_cfg: dict) -> float:
    """Auto-calibrate the proximity RMS floor from the primary enrollment segment
    (the user is AT the kiosk), unless an explicit threshold is configured —
    verbatim policy from controller.py:336-344."""
    prox = tb_cfg.get("barge_in", {}).get("proximity", {})
    thr = prox.get("rms_threshold")
    if thr is None:
        audio = getattr(first_segment, "audio", None)
        primary_rms = (
            float(np.sqrt(np.mean(np.square(audio))))
            if audio is not None and len(audio) else 0.0
        )
        thr = primary_rms * prox.get("rms_factor", 0.5)
    return thr


def _build_aec(tb_cfg: dict):
    aec_cfg = tb_cfg.get("aec", {})
    if not aec_cfg.get("enabled", False) or AecProcessor is None:
        return None
    try:
        return AecProcessor(
            sample_rate=tb_cfg.get("sample_rate_hz", 16000),
            frame_ms=tb_cfg.get("frame_ms", 10),
        )
    except Exception:
        return None


def _build_turn_detector():
    """SmartTurn when the ONNX backend is importable; NullTurnDetector otherwise
    (CI / no-onnxruntime). Null always reports turn-complete, so endpointed
    segments still answer — degraded but runnable."""
    if SmartTurnDetector is not None:
        try:
            return SmartTurnDetector()
        except Exception:
            pass
    return NullTurnDetector()


class DirectorRuntimeFactory:
    """Adapter exposing the WakeGate's expected .run(handoff) -> DirectorResult
    around a fully-wired Plan-02 DirectorRuntime. Holds the bus, stt_worker, and
    first segment so run() can seed the opening turn before draining."""

    def __init__(self, runtime: DirectorRuntime, bus: EventBus,
                 stt_worker: SttWorker, first_segment):
        self.runtime = runtime
        self._bus = bus
        self._stt = stt_worker
        self._first_segment = first_segment

    def run(self, handoff: Any) -> Any:
        # Seed the opening turn: the WakeGate already captured the first segment,
        # so stage its audio and announce ONE SegmentEndpointed. The runtime's
        # loop drains it first (FIFO bus), the Director answers it, and the
        # conversation proceeds normally. The watchdog owns every later timeout.
        seg = self._first_segment

        async def _run_with_seed():
            if seg is not None and getattr(seg, "audio", None) is not None \
                    and len(seg.audio):
                self._stt.set_pending_user_audio(seg.audio)
                rms = float(np.sqrt(np.mean(np.square(seg.audio))))
                await self._bus.emit(E.SegmentEndpointed(
                    duration_ms=seg.duration_ms, rms=rms,
                    is_target=True, endpoint_prob=1.0,
                ))
            return await self.runtime.run_async()

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run_with_seed())
        finally:
            self.runtime._playback.close()
            loop.close()


def build_director_runtime(
    handoff: Any,
    *,
    stt: Any,
    llm: Any,
    tts: Any,
    player: Any,
    logger: Any = None,
    clock: Callable[[], float] = time.monotonic,
    _out_stream: Optional[Any] = None,
    _watchdog_tick_s: float = 0.5,
) -> DirectorRuntimeFactory:
    """Construct a fully-wired Plan-02 DirectorRuntime from a DirectorHandoff.

    Args:
        handoff: the DirectorHandoff the WakeGate built (mic, vad, embedder,
                 primary/holdout embeddings, first_segment, talkback config).
        stt/llm/tts/player: the heavy backends, already warmed by the caller.
        logger: optional EventLogger (kept for parity; not on the hot path yet).
        clock: monotonic clock source (injected in tests).
        _out_stream: inject a fake OutputStream in tests; production opens a real
                     sounddevice OutputStream below.
        _watchdog_tick_s: watchdog cadence (small in tests for fast timeouts).
    """
    tb_cfg = handoff.config or {}
    cfg = _director_config_from(tb_cfg)

    bus = EventBus()
    conversation = ConversationManager(
        system_prompt=tb_cfg.get("llm", {}).get(
            "system_prompt", "You are a concise voice assistant."),
    )
    proximity_rms = _calibrate_proximity_rms(handoff.first_segment, tb_cfg)
    now = clock()
    director = Director(cfg, conversation, now=now, proximity_rms=proximity_rms)

    aec = _build_aec(tb_cfg)
    turn_detector = _build_turn_detector()

    playback = PlaybackWorker(tts=tts, player=player, cfg=cfg, bus=bus)
    out_stream = _out_stream if _out_stream is not None else _open_output_stream(tb_cfg)
    if out_stream is not None:
        playback.open(out_stream)

    stt_worker = SttWorker(stt, bus)

    chunker_cfg = tb_cfg.get("chunker", {})
    generation = GenerationWorker(
        llm=llm, tts=tts,
        chunker_factory=lambda: SentenceChunker(
            sentence_terminators=chunker_cfg.get("sentence_terminators"),
            max_chunk_chars=chunker_cfg.get("max_chunk_chars", 120),
        ),
        playback=playback, bus=bus,
    )

    ingestion = IngestionWorker(
        mic=handoff.mic, vad=handoff.vad, aec=aec, turn_detector=turn_detector,
        embedder=handoff.embedder, primary_embedding=handoff.primary_embedding,
        stt_worker=stt_worker, playback=playback, bus=bus, cfg=cfg,
        proximity_rms=proximity_rms, state_getter=lambda: director.state,
        score_fn=cosine_similarity,
    )

    watchdog = AsyncWatchdog(
        tick_s=_watchdog_tick_s, clock=clock, bus=bus,
        on_session_end=lambda reason: None,
    )

    runtime = DirectorRuntime(
        director=director, bus=bus, watchdog=watchdog, ingestion=ingestion,
        stt_worker=stt_worker, generation=generation, playback=playback,
        clock=clock,
    )
    return DirectorRuntimeFactory(runtime, bus, stt_worker, handoff.first_segment)


def _open_output_stream(tb_cfg: dict):  # pragma: no cover - needs real audio device
    """Open a persistent sounddevice OutputStream so playback frames can be
    written + recorded as the AEC reference (controller.py:316-323). Returns None
    if no device is available (degraded: no audible output, still runs)."""
    try:
        import sounddevice as sd
        stream = sd.OutputStream(
            samplerate=tb_cfg.get("sample_rate_hz", 16000), channels=1,
            dtype="float32", device=tb_cfg.get("output_device"),
        )
        stream.start()
        return stream
    except Exception:
        return None
