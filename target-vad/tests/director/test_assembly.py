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
        mic=mic, primary_embedding=emb,
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


def _write_cohort(tmp_path, n=4, d=192):
    """A tiny unit-norm imposter cohort .npy for the score_norm mapping tests."""
    rng = np.random.RandomState(0)
    cohort = rng.randn(n, d).astype(np.float32)
    cohort /= np.linalg.norm(cohort, axis=1, keepdims=True)
    path = tmp_path / "cohort.npy"
    np.save(path, cohort)
    return str(path)


def test_score_norm_shadow_mode_sets_normalizer_raw_still_decides(tmp_path):
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    cohort_path = _write_cohort(tmp_path)
    emb = object()
    prim = np.ones(192, dtype=np.float32)
    bus = EventBus()
    cfg = {"turn_gate": {"require_speaker_match": True,
                         "score_norm": {"mode": "shadow", "cohort_path": cohort_path}}}
    worker = _build_safety_net(cfg, prim, emb, bus)
    assert worker is not None
    net = worker._net
    assert net._normalizer is not None
    assert net._norm_decides is False


def test_score_norm_on_mode_norm_decides_and_uses_norm_threshold(tmp_path):
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    cohort_path = _write_cohort(tmp_path)
    emb = object()
    prim = np.ones(192, dtype=np.float32)
    bus = EventBus()
    cfg = {"turn_gate": {"require_speaker_match": True,
                         "score_norm": {"mode": "on", "cohort_path": cohort_path,
                                        "speaker_threshold_norm": 1.23}}}
    worker = _build_safety_net(cfg, prim, emb, bus)
    net = worker._net
    assert net._normalizer is not None
    assert net._norm_decides is True
    assert net._smoother.threshold == pytest.approx(1.23)


def test_score_norm_off_mode_no_normalizer(tmp_path):
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    cohort_path = _write_cohort(tmp_path)
    emb = object()
    prim = np.ones(192, dtype=np.float32)
    bus = EventBus()
    cfg = {"turn_gate": {"require_speaker_match": True,
                         "score_norm": {"mode": "off", "cohort_path": cohort_path}}}
    worker = _build_safety_net(cfg, prim, emb, bus)
    assert worker._net._normalizer is None


def test_score_norm_missing_cohort_fails_open_and_warns(tmp_path, capsys):
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    emb = object()
    prim = np.ones(192, dtype=np.float32)
    bus = EventBus()
    cfg = {"turn_gate": {"require_speaker_match": True,
                         "score_norm": {"mode": "on",
                                        "cohort_path": str(tmp_path / "nope.npy")}}}
    worker = _build_safety_net(cfg, prim, emb, bus)
    assert worker._net._normalizer is None
    assert worker._net._norm_decides is False
    err = capsys.readouterr().err
    assert "score_norm" in err and "falling back to raw scores" in err


def test_score_norm_empty_cohort_fails_open_and_warns(tmp_path, capsys):
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    empty_path = tmp_path / "empty.npy"
    np.save(empty_path, np.zeros((0, 192), dtype=np.float32))
    emb = object()
    prim = np.ones(192, dtype=np.float32)
    bus = EventBus()
    cfg = {"turn_gate": {"require_speaker_match": True,
                         "score_norm": {"mode": "on", "cohort_path": str(empty_path)}}}
    worker = _build_safety_net(cfg, prim, emb, bus)
    assert worker._net._normalizer is None
    err = capsys.readouterr().err
    assert "score_norm" in err and "falling back to raw scores" in err


def test_score_norm_non_2d_cohort_fails_open_and_warns(tmp_path, capsys):
    from modes.director.assembly import _build_safety_net
    from modes.director.bus import EventBus
    flat_path = tmp_path / "flat.npy"
    np.save(flat_path, np.zeros(192, dtype=np.float32))  # 1-D, not (n, d)
    emb = object()
    prim = np.ones(192, dtype=np.float32)
    bus = EventBus()
    cfg = {"turn_gate": {"require_speaker_match": True,
                         "score_norm": {"mode": "on", "cohort_path": str(flat_path)}}}
    worker = _build_safety_net(cfg, prim, emb, bus)
    assert worker._net._normalizer is None
    assert worker._net._norm_decides is False
    err = capsys.readouterr().err
    assert "score_norm" in err and "falling back to raw scores" in err


def test_lockout_enabled_strict_bool_mapping():
    from modes.director.assembly import _director_config_from
    assert _director_config_from(
        {"turn_gate": {"lockout": {"enabled": True}}}).lockout_enabled is True
    assert _director_config_from(
        {"turn_gate": {"lockout": {"enabled": "true"}}}).lockout_enabled is False
    assert _director_config_from({}).lockout_enabled is False


def test_nudge_lead_and_conf_floor_are_mapped():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({"nudge_lead_s": 7.5,
                                 "barge_in": {"conf_floor": 0.65}})
    assert cfg.nudge_lead_s == 7.5
    assert cfg.conf_floor == 0.65


def test_onset_floor_speaking_is_mapped():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({"barge_in": {"onset_floor_speaking": 0.22}})
    assert cfg.onset_floor_speaking == 0.22
    assert _director_config_from({}).onset_floor_speaking == 0.0


def test_shipped_config_yaml_matches_live_readers():
    import yaml
    with open("config.yaml") as f:
        full = yaml.safe_load(f)
    tb = full["kiosk"]["talkback"]
    # keys this feature makes/keeps live
    assert tb["turn_gate"]["require_speaker_match"] is True
    assert tb["turn_gate"]["lockout"]["enabled"] is True
    assert tb["turn_gate"]["endpoint_threshold"] == 0.5
    assert tb["verify_before_serve_threshold"] == 0.80
    assert tb["lockout_idle_after_s"] == 5
    assert tb["nudge_lead_s"] == 5.0
    assert tb["barge_in"]["conf_floor"] == 0.5
    assert tb["barge_in"]["onset_floor_speaking"] == 0.15
    assert tb["watchdog"]["tick_ms"] == 500
    assert tb["turn_gate"]["score_norm"]["mode"] == "shadow"
    assert tb["turn_gate"]["score_norm"]["cohort_path"] == "./voiceprints/cohort.npy"
    assert tb["turn_gate"]["score_norm"]["top_k"] == 50
    assert tb["turn_gate"]["score_norm"]["speaker_threshold_norm"] == 0.0
    assert tb["barge_in"]["speaker_threshold_norm"] == 0.0
    assert tb["turn_gate"]["enrollment_update_alpha"] == 0.10
    assert tb["turn_gate"]["enrollment_update_margin"] == 0.10
    assert tb["stt"]["backend"] == "nemo"
    assert tb["stt"]["nemo_model"] == "nvidia/parakeet-tdt-0.6b-v2"
    # dead keys must be GONE
    assert "decision_smoother" not in full["kiosk"]
    assert "suppression_level" not in tb["aec"]
    assert "partials_every_ms" not in tb["stt"]
    assert "require_speaker_match" not in tb["barge_in"]
    assert "audio_safety_net" not in tb["vision"]
    assert tb["vision"]["preview"]["enabled"] is True
    assert tb["vision"]["preview"]["path"] == "/dev/shm/tvad-vision-preview.jpg"
    assert "resume" not in tb
    assert "include_partial_transcripts" not in tb["logging"]


def test_echo_guard_tail_is_mapped():
    from modes.director.assembly import _director_config_from
    cfg = _director_config_from({"barge_in": {"echo_guard_tail_s": 1.5}})
    assert cfg.echo_guard_tail_s == 1.5
    assert _director_config_from({}).echo_guard_tail_s == 0.0


def test_shipped_echo_guard_value():
    import yaml
    with open("config.yaml") as f:
        tb = yaml.safe_load(f)["kiosk"]["talkback"]
    assert tb["barge_in"]["echo_guard_tail_s"] == 2.0
