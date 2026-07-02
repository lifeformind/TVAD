# tests/director/test_assembly.py
"""build_director_runtime — the runnable-assembly seam (prompt CRITICAL note).

Given a DirectorHandoff and the four heavy backends (stt/llm/tts/player) it must
construct the REAL Plan-02 DirectorRuntime: EventBus, Director (from a fresh
ConversationManager + proximity_rms auto-calibrated from the first segment, like
controller.py), AsyncWatchdog, and all four workers wired together. The returned
object exposes .run(handoff) so the WakeGate can call it. These tests use fakes
(no GPU/mic) and drive one synthetic turn end-to-end."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from core.vad.silero_vad import SpeechSegment
from modes.talkback.handoff import DirectorHandoff, DirectorResult


def _segment(duration_ms=1000.0, rms=0.1):
    samples = int(duration_ms / 1000 * 16000)
    return SpeechSegment(
        audio=(np.ones(samples, dtype=np.float32) * rms),
        start_ms=0.0, end_ms=duration_ms, duration_ms=duration_ms,
    )


def _talkback_config():
    return {
        "sample_rate_hz": 16000,
        "frame_ms": 10,
        "aec": {"enabled": False},
        "stt": {"model": "base"},
        "llm": {"system_prompt": "You are a concise voice assistant."},
        "barge_in": {"proximity": {"rms_threshold": None, "rms_factor": 0.5}},
        "silence_timeout_s": 30,
        "hard_timeout_s": 300,
    }


class _FakeLlm:
    def __init__(self, tokens):
        self._tokens = tokens
        self.cancelled = False

    async def stream(self, messages):
        for t in self._tokens:
            await asyncio.sleep(0)
            yield t

    def cancel(self):
        self.cancelled = True

    async def close(self):
        pass


class _FakeStt:
    async def transcribe_segment(self, audio):
        return "tell me a story"


def _handoff(mic, vad, embedder):
    emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
    return DirectorHandoff(
        mic=mic, primary_embedding=emb, holdout_embedding=emb,
        first_segment=_segment(), config=_talkback_config(),
        vad=vad, embedder=embedder,
    )


def test_build_director_runtime_constructs_real_runtime_and_calibrates_proximity():
    from modes.director.assembly import build_director_runtime
    from modes.director.runtime import DirectorRuntime

    mic = MagicMock()
    mic.stream = MagicMock(return_value=iter([]))   # no further chunks
    mic.read_available = MagicMock(return_value=[])
    vad = MagicMock(process_chunk=MagicMock(return_value=[]), reset=MagicMock())
    embedder = MagicMock(extract=MagicMock(
        return_value=np.ones(192, dtype=np.float32) / np.sqrt(192)))

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
    player = MagicMock()
    player.record_reference = MagicMock()
    player.get_reference_frame = MagicMock(return_value=None)
    logger = MagicMock()

    handoff = _handoff(mic, vad, embedder)
    factory = build_director_runtime(
        handoff, stt=_FakeStt(), llm=_FakeLlm(["Once upon a time. ", "The end."]),
        tts=tts, player=player, logger=logger, clock=lambda: 0.0,
    )
    rt = factory.runtime
    assert isinstance(rt, DirectorRuntime)
    # proximity_rms auto-calibrated from the first segment RMS * rms_factor (0.5)
    first_rms = float(np.sqrt(np.mean(np.square(handoff.first_segment.audio))))
    assert rt._director.ctx.proximity_rms == pytest.approx(first_rms * 0.5, rel=1e-3)


def test_run_drives_first_segment_to_a_full_turn_and_returns_result():
    """The first segment (captured by the WakeGate) must enter the conversation:
    run(handoff) stages it, the Director answers it, and a silence Tick ends the
    session. Uses an out-stream stub so playback never touches real audio."""
    from modes.director.assembly import build_director_runtime

    # Clock jumps far past silence_timeout after a few reads so the watchdog's
    # next Tick ends the session deterministically.
    ticks = {"n": 0}

    def clock_fn():
        ticks["n"] += 1
        return 0.0 if ticks["n"] < 4 else 1000.0

    # mic yields nothing further; the only turn is the staged first segment.
    mic = MagicMock()
    mic.stream = MagicMock(return_value=iter([]))
    mic.read_available = MagicMock(return_value=[])
    vad = MagicMock(process_chunk=MagicMock(return_value=[]), reset=MagicMock())
    embedder = MagicMock(extract=MagicMock(
        return_value=np.ones(192, dtype=np.float32) / np.sqrt(192)))

    tts = MagicMock()
    tts.synthesize = AsyncMock(return_value=np.ones(480, dtype=np.float32))
    player = MagicMock(record_reference=MagicMock(),
                       get_reference_frame=MagicMock(return_value=None))
    logger = MagicMock()

    handoff = _handoff(mic, vad, embedder)
    factory = build_director_runtime(
        handoff, stt=_FakeStt(), llm=_FakeLlm(["Once upon a time. ", "The end."]),
        tts=tts, player=player, logger=logger, clock=clock_fn,
        _out_stream=MagicMock(), _watchdog_tick_s=0.01,
    )

    result = factory.run(handoff)
    assert isinstance(result, DirectorResult)
    assert result.reason in ("silence_timeout", "hard_timeout", "stopped")


def _pvad_cached():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="FireRedTeam/FireRedChat-pvad",
                        filename="pvad.onnx", local_files_only=True)
        return True
    except Exception:
        return False


def test_build_pvad_returns_real_worker_not_silent_fallback():
    """Regression for the live NameError: _build_pvad called an undefined _diag(),
    which threw -> caught -> silent fallback to is_target=True (crowd focus OFF the
    whole session). The fake-pVAD ingestion tests never exercised _build_pvad, so
    this drives it directly and asserts a real worker comes back."""
    from modes.director.assembly import _build_pvad
    emb = np.random.RandomState(0).randn(192).astype(np.float32)
    w = _build_pvad(emb, proximity_rms=0.02, tb_cfg={"crowd_focus": {"enabled": True}})
    if not _pvad_cached():
        import pytest
        pytest.skip("pvad.onnx not cached")
    assert w is not None and type(w).__name__ == "PvadWorker"
    out = w.process(np.zeros(3200, dtype=np.float32), ts=0.0)
    assert out and all(hasattr(f, "is_target") for f in out)


def test_build_pvad_disabled_returns_none():
    from modes.director.assembly import _build_pvad
    emb = np.ones(192, dtype=np.float32)
    assert _build_pvad(emb, 0.02, {"crowd_focus": {"enabled": False}}) is None


def test_build_pvad_no_embedding_returns_none():
    from modes.director.assembly import _build_pvad
    assert _build_pvad(None, 0.02, {"crowd_focus": {"enabled": True}}) is None


def test_safety_net_requires_strict_bool_true():
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    emb = object()
    prim = np.ones(4, dtype=np.float32)
    bus = EventBus()
    on = {"turn_gate": {"require_speaker_match": True}}
    off = {"turn_gate": {"require_speaker_match": False}}
    truthy_string = {"turn_gate": {"require_speaker_match": "true"}}
    missing = {}
    assert _build_safety_net(on, prim, emb, bus) is not None
    assert _build_safety_net(off, prim, emb, bus) is None
    assert _build_safety_net(truthy_string, prim, emb, bus) is None
    assert _build_safety_net(missing, prim, emb, bus) is None


def test_safety_net_none_without_embedder_or_primary():
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    cfg = {"turn_gate": {"require_speaker_match": True}}
    assert _build_safety_net(cfg, None, object(), EventBus()) is None
    assert _build_safety_net(cfg, np.ones(4), None, EventBus()) is None


def test_lockout_enabled_strict_bool_mapping():
    from modes.director.assembly import _director_config_from
    assert _director_config_from(
        {"turn_gate": {"lockout": {"enabled": True}}}).lockout_enabled is True
    assert _director_config_from(
        {"turn_gate": {"lockout": {"enabled": "true"}}}).lockout_enabled is False
    assert _director_config_from({}).lockout_enabled is False
