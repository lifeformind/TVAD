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
import os
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

_DIAG = bool(os.environ.get("TVAD_DIAG"))

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
from modes.director.safety_net import SafetyNet
from modes.director.workers.safety_net import SafetyNetWorker
from modes.director.workers.stt_worker import SttWorker
from modes.director import events as E
from modes.director.vision.opencv_backend import OpenCvBackend, cv2_available as _cv2_available
from modes.director.workers.vision import VisionWorker
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


def _build_pvad(primary_embedding, proximity_rms: float, tb_cfg: dict):
    """Build the Plan-05 crowd-focus PvadWorker, or None for the legacy
    single-speaker path (is_target=True). Returns None when there is no enrolled
    embedding to condition on, when crowd_focus is disabled, or when the ONNX
    model can't load (offline / uncached) — the kiosk still runs, just without
    bystander rejection. The PvadWorker's own RMS crash-fallback covers a model
    that loads but fails mid-session."""
    if primary_embedding is None:
        return None
    cf = tb_cfg.get("crowd_focus", {})
    if not cf.get("enabled", True):
        return None
    try:
        from modes.director.pvad.loader import load_pvad
        from modes.director.pvad.stream import VADStream
        from modes.director.pvad.worker import PvadWorker

        session = load_pvad()
        stream = VADStream(session, threshold=float(cf.get("threshold", 0.5)))

        def _emit(event, payload):
            print(f"[director] pvad {event}: {payload}", file=sys.stderr, flush=True)

        worker = PvadWorker(stream, proximity_rms=proximity_rms, emit=_emit)
        worker.update_speaker(primary_embedding)
        print("[director] pVAD crowd-focus ENABLED", file=sys.stderr, flush=True)
        return worker
    except Exception as exc:   # noqa: BLE001 — degrade to legacy is_target=True
        print(f"[director] pVAD unavailable -> FOCUS via is_target=True fallback: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return None


def _build_vision(tb_cfg: dict, *, bus):
    """Build the Director-07 camera floor-control VisionWorker, or None for the
    no-vision path (today's Director). None when vision is disabled or cv2 is
    unavailable — the no-regression guarantee. The worker self-enrolls the owner
    at session start and reports UNAVAILABLE if it can't (never falsely ABSENT)."""
    v = tb_cfg.get("vision", {})
    enabled = v.get("enabled", False)
    if enabled is not True:
        # Only a real boolean True enables vision. A non-bool value fails SAFE to
        # off rather than fail-open: the live `enabled: flase` typo parses as a
        # truthy STRING and would otherwise silently keep vision on. Warn when a
        # present value is malformed so the misconfig is visible, not silent.
        if "enabled" in v and not isinstance(enabled, bool):
            print(f"[director] vision.enabled is not a boolean (got {enabled!r}) -> "
                  "treating as disabled", file=sys.stderr, flush=True)
        return None
    if not _cv2_available():
        print("[director] vision enabled but cv2/FaceDetectorYN unavailable -> "
              "floor control via audio timeout only", file=sys.stderr, flush=True)
        return None
    backend = OpenCvBackend(
        camera_index=v.get("camera_index", 0), width=v.get("width", 640),
        height=v.get("height", 360), identity_threshold=v.get("identity_threshold", 0.40),
        min_area_frac=v.get("min_area_frac", 0.015))
    return VisionWorker(
        backend, bus, fps=v.get("fps", 3.0),
        present_after_s=v.get("present_after_s", 1.0),
        absent_after_s=v.get("absent_after_s", 2.0),
        enroll_frames=v.get("enroll_frames", 8))


def _build_safety_net(tb_cfg: dict, primary_embedding, embedder, bus):
    """Build the Director-09 accumulated-window ECAPA safety net, or None (no
    hijack detection — byte-for-byte Director-08 behavior). Strict-bool enable:
    only a real True builds it (the 'flase' lesson); warn on a present non-bool."""
    tg = tb_cfg.get("turn_gate", {})
    enabled = tg.get("require_speaker_match", False)
    if enabled is not True:
        if "require_speaker_match" in tg and not isinstance(enabled, bool):
            print(f"[director] turn_gate.require_speaker_match is not a boolean "
                  f"(got {enabled!r}) -> safety net disabled",
                  file=sys.stderr, flush=True)
        return None
    if embedder is None or primary_embedding is None:
        return None
    lock = tg.get("lockout", {})
    net = SafetyNet(
        embedder, primary_embedding,
        verify_window_ms=tg.get("verify_window_ms", 2000),
        threshold=tg.get("speaker_threshold", 0.30),
        window_size=lock.get("window_size", 3),
        min_matches=lock.get("min_matches", 1),
        sr=tb_cfg.get("sample_rate_hz", 16000),
    )
    print("[director] safety net ENABLED (accumulated-window ECAPA)",
          file=sys.stderr, flush=True)
    return SafetyNetWorker(net, bus)


def _director_config_from(tb_cfg: dict) -> DirectorConfig:
    """Map the kiosk.talkback.* config onto the frozen DirectorConfig. Pulls the
    same keys the reducer's thresholds came from (spec section 5/6)."""
    barge = tb_cfg.get("barge_in", {})
    vision = tb_cfg.get("vision", {})
    return DirectorConfig(
        silence_timeout_s=tb_cfg.get("silence_timeout_s", 30.0),
        hard_timeout_s=tb_cfg.get("hard_timeout_s", 300.0),
        nudge_lead_s=tb_cfg.get("nudge_lead_s", 5.0),
        endpoint_threshold=tb_cfg.get("turn_gate", {}).get("endpoint_threshold", 0.5),
        min_speech_ms=barge.get("min_speech_ms", 120.0),
        verify_window_ms=barge.get("verify_window_ms", 700.0),
        speaker_threshold=barge.get("speaker_threshold", 0.20),
        conf_floor=barge.get("conf_floor", 0.5),
        duck_level=barge.get("duck_level", 0.35),
        owner_absent_grace_s=vision.get("owner_absent_grace_s", 3.0),
        active_talk_guard_s=vision.get("active_talk_guard_s", 3.0),
        reject_bystanders=tb_cfg.get("turn_gate", {}).get("reject_bystanders", False) is True,
        lockout_enabled=tb_cfg.get("turn_gate", {}).get("lockout", {})
                              .get("enabled", False) is True,
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
        _seg_len = len(seg.audio) if (seg is not None and getattr(seg, "audio", None) is not None) else 0
        if _DIAG:
            print(f"[DIAG assembly] seed first_segment samples={_seg_len} "
                  f"dur_ms={getattr(seg, 'duration_ms', 0)}", file=sys.stderr, flush=True)

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

    pvad = _build_pvad(handoff.primary_embedding, proximity_rms, tb_cfg)
    safety = _build_safety_net(tb_cfg, handoff.primary_embedding,
                               handoff.embedder, bus)
    ingestion = IngestionWorker(
        mic=handoff.mic, vad=handoff.vad, aec=aec, turn_detector=turn_detector,
        embedder=handoff.embedder, primary_embedding=handoff.primary_embedding,
        stt_worker=stt_worker, playback=playback, bus=bus, cfg=cfg,
        proximity_rms=proximity_rms, state_getter=lambda: director.state,
        score_fn=cosine_similarity, pvad=pvad, safety_worker=safety,
    )

    vision = _build_vision(tb_cfg, bus=bus)

    watchdog = AsyncWatchdog(
        tick_s=_watchdog_tick_s, clock=clock, bus=bus,
        on_session_end=lambda reason: None,
    )

    runtime = DirectorRuntime(
        director=director, bus=bus, watchdog=watchdog, ingestion=ingestion,
        stt_worker=stt_worker, generation=generation, playback=playback,
        clock=clock, vision=vision, safety_worker=safety,
    )
    return DirectorRuntimeFactory(runtime, bus, stt_worker, handoff.first_segment)


def _pipewire_sinks():
    """Live PipeWire Audio/Sink nodes as [{'name', 'description'}], via
    pw-dump. Raises on a missing/broken pw-dump — the caller decides whether
    unverifiable routing is fatal (it is, wherever the array pin matters)."""
    import json
    import subprocess

    out = subprocess.run(["pw-dump"], capture_output=True, text=True,
                         timeout=5, check=True).stdout
    sinks = []
    for obj in json.loads(out):
        props = ((obj.get("info") or {}).get("props") or {})
        if props.get("media.class") == "Audio/Sink":
            sinks.append({"name": props.get("node.name", ""),
                          "description": props.get("node.description", "")})
    return sinks


def resolve_pipewire_sink(spec, sinks):
    """Resolve kiosk.talkback.output_device (a name substring) to a PipeWire
    sink node.name; no match -> RuntimeError. Fail loud is the point
    (Director-10): TTS must verifiably reach the array — its onboard AEC
    cancels the kiosk's own voice (Bug A); silently landing on another sink
    would resurrect the bug invisibly. The resolved name is passed via
    PIPEWIRE_NODE at stream open, which pins the stream to that sink
    regardless of default-sink elections — but a PIPEWIRE_NODE naming a
    NONEXISTENT node falls back to the default sink silently (measured
    2026-07-06), so the guarantee lives here, in resolving against live
    pw-dump state first."""
    needle = str(spec).lower()
    for s in sinks:
        if (needle in s.get("name", "").lower()
                or needle in s.get("description", "").lower()):
            return s["name"]
    raise RuntimeError(
        f"output_device {spec!r} matches no PipeWire audio sink. "
        f"Check the ReSpeaker's USB connection (lsusb should list 2886:0018) "
        f"and `wpctl status` sinks, or set kiosk.talkback.output_device to "
        f"null for the system default.")


def _open_output_stream(tb_cfg: dict):  # pragma: no cover - needs real audio device
    """Open a persistent sounddevice OutputStream so playback frames can be
    written + recorded as the AEC reference (controller.py:316-323).

    output_device set (Director-10 array pin): failures RAISE — never fall
    back to another device. A string spec resolves to a PipeWire sink and
    opens PortAudio's 'pipewire' device with PIPEWIRE_NODE set to the
    resolved node: opening the array's ALSA device directly races PipeWire's
    card reservation (EBUSY whenever the sink is active or within its
    suspend-timeout — live finding 2026-07-06). An int spec stays the raw
    PortAudio-index escape hatch. output_device null: legacy best-effort —
    system default, None (no audible output) on any failure."""
    if os.environ.get("TVAD_NO_OUTPUT"):
        # Isolation switch: skip opening the OutputStream entirely. If the mic
        # then works (turns transcribed), opening the output stream was resetting
        # the (USB) audio device and killing the mic.
        if _DIAG:
            print("[DIAG assembly] TVAD_NO_OUTPUT set: NOT opening OutputStream",
                  file=sys.stderr, flush=True)
        return None
    device_spec = tb_cfg.get("output_device")
    if device_spec is None:
        try:
            import sounddevice as sd
            stream = sd.OutputStream(
                samplerate=tb_cfg.get("sample_rate_hz", 16000), channels=1,
                dtype="float32", device=None,
            )
            stream.start()
            return stream
        except Exception:
            return None
    import sounddevice as sd
    if isinstance(device_spec, int):
        stream = sd.OutputStream(
            samplerate=tb_cfg.get("sample_rate_hz", 16000), channels=1,
            dtype="float32", device=device_spec,
        )
        stream.start()
        return stream
    node = resolve_pipewire_sink(device_spec, _pipewire_sinks())
    os.environ["PIPEWIRE_NODE"] = node
    try:
        stream = sd.OutputStream(
            samplerate=tb_cfg.get("sample_rate_hz", 16000), channels=1,
            dtype="float32", device="pipewire",
        )
        stream.start()
    finally:
        # The pin is read at stream creation; scope it so no other stream
        # (or a later legacy-null session) inherits it.
        os.environ.pop("PIPEWIRE_NODE", None)
    if _DIAG:
        print(f"[DIAG assembly] OutputStream pinned to PipeWire sink {node}",
              file=sys.stderr, flush=True)
    return stream
