# tests/director/test_wakegate_doa_seed.py
"""Director-11 seed direction filter: waking while a podcast played let the
podcast become the enrollment seed — floor, voiceprint AND bearing enrolled
on the bystander (live 2026-07-07 19:04, session 2 chatted with the podcast).
The wake phrase is the one utterance guaranteed to be the owner: the WakeGate
measures the bearing over ITS window and skips candidate seeds from other
directions. Abstain (no tracker / thin evidence) enrolls — fail open."""

import numpy as np
import pytest

from tests.director.conftest import make_segment
from modes.director.wakegate import WakeGate


class _FakeDoa:
    """medians: one entry per median_between call (call 1 = ambient window,
    call 2 = wake window when ambient was None). angles_per_call: one entry
    per angles_between call (wake window first when ambient exists, then one
    per candidate segment)."""
    def __init__(self, medians=(), angles_per_call=()):
        self._medians = list(medians)
        self._angles = list(angles_per_call)

    def median_between(self, t0, t1):
        return self._medians.pop(0) if self._medians else None

    def angles_between(self, t0, t1):
        return self._angles.pop(0) if self._angles else ()


def _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
          fake_runtime, doa_tracker):
    return WakeGate(config=base_config, runtime=fake_runtime,
                    _mic=fake_mic, _vad=fake_vad, _embedder=fake_embedder,
                    _wake_detector=fake_wake, doa_tracker=doa_tracker)


def _wake(g, fake_wake):
    fake_wake.process.return_value = 0.9
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._state == "AWAIT_FIRST_SEGMENT"
    fake_wake.process.return_value = None


def test_bystander_direction_seed_skipped_then_owner_seed_enrolls(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    # Quiet ambient (median call 1 -> None) -> wake bearing = plain wake-window
    # median (call 2 -> 97).
    doa = _FakeDoa(medians=(None, 97.0),
                   angles_per_call=[(193.0,) * 6,        # podcast-direction seed
                                    (97.0, 95.0, 100.0, 99.0)])  # the owner
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))     # podcast seed -> skipped
    assert g._pending_handoff is None
    assert g._state == "AWAIT_FIRST_SEGMENT"
    g._handle_chunk(np.zeros(480, dtype=np.float32))     # owner seed -> enrolls
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing == 97.0


def test_thin_doa_evidence_enrolls_fail_open(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    doa = _FakeDoa(medians=(None, 97.0), angles_per_call=[(193.0,)])  # 1 sample
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None                # abstain = enroll


def test_no_wake_bearing_enrolls_fail_open(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    doa = _FakeDoa(medians=(None, None), angles_per_call=[(193.0,) * 6])
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing is None


def test_no_tracker_is_legacy_first_segment_wins(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa_tracker=None)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing is None


def test_reset_clears_the_wake_bearing(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    doa = _FakeDoa(medians=(None, 97.0), angles_per_call=[(97.0,) * 4])
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._wake_bearing == 97.0
    g._reset_to_idle()
    assert g._wake_bearing is None
    assert g._wake_bearing_known is False


# ---- novel-cluster wake bearing (live 2026-07-07 19:46: plain median over
# the wake window got OUTVOTED by the podcast — bearing landed on it, 90° from
# the user, while the array's LED tracked the user during their speech) ----

def test_ambient_speech_bearing_is_the_novel_cluster(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    # Ambient = podcast at 250°. Wake window: 5 podcast samples + 4 user
    # samples at ~162°. Median would say 250; the novel cluster says 162.
    wake_window = (250.0, 251.0, 162.0, 160.0, 250.0, 164.0, 249.0, 162.0, 250.0)
    doa = _FakeDoa(medians=(250.0,),
                   angles_per_call=[wake_window,
                                    (162.0, 160.0, 163.0, 161.0)])  # owner seed
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing == 162.0


def test_ambient_speech_podcast_seed_skipped_under_novel_bearing(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    wake_window = (250.0, 162.0, 160.0, 250.0, 164.0, 250.0)
    doa = _FakeDoa(medians=(250.0,),
                   angles_per_call=[wake_window,
                                    (250.0,) * 6,                   # podcast seed
                                    (162.0, 160.0, 163.0, 161.0)])  # owner seed
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))     # podcast seed skipped
    assert g._pending_handoff is None
    g._handle_chunk(np.zeros(480, dtype=np.float32))     # owner seed enrolls
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing == pytest.approx(162.0, abs=2.5)


def test_ambient_speech_no_novel_cluster_abstains(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    # Owner co-located with the source (or unseen): a wrong bearing locks the
    # owner out, so the bearing abstains and the first seed enrolls (D10-era
    # behavior; split-half + safety net remain the defenses).
    doa = _FakeDoa(medians=(250.0,),
                   angles_per_call=[(250.0, 251.0, 249.0, 250.0),
                                    (250.0,) * 4])
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing is None


def test_single_novel_sample_is_not_a_cluster(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    doa = _FakeDoa(medians=(250.0,),
                   angles_per_call=[(250.0, 162.0, 250.0, 251.0),   # 1 novel blip
                                    (250.0,) * 4])
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None
    assert g._pending_handoff.wake_bearing is None
