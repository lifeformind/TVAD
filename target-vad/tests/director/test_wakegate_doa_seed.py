# tests/director/test_wakegate_doa_seed.py
"""Director-11 seed direction filter: waking while a podcast played let the
podcast become the enrollment seed — floor, voiceprint AND bearing enrolled
on the bystander (live 2026-07-07 19:04, session 2 chatted with the podcast).
The wake phrase is the one utterance guaranteed to be the owner: the WakeGate
measures the bearing over ITS window and skips candidate seeds from other
directions. Abstain (no tracker / thin evidence) enrolls — fail open."""

import numpy as np

from tests.director.conftest import make_segment
from modes.director.wakegate import WakeGate


class _FakeDoa:
    def __init__(self, wake_bearing=97.0, angles_per_call=()):
        self._wake_bearing = wake_bearing
        self._angles = list(angles_per_call)     # one entry per angles_between call

    def median_between(self, t0, t1):
        return self._wake_bearing

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
    doa = _FakeDoa(wake_bearing=97.0,
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
    doa = _FakeDoa(wake_bearing=97.0, angles_per_call=[(193.0,)])  # 1 sample
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._pending_handoff is not None                # abstain = enroll


def test_no_wake_bearing_enrolls_fail_open(
        base_config, fake_mic, fake_vad, fake_embedder, fake_wake, fake_runtime):
    doa = _FakeDoa(wake_bearing=None, angles_per_call=[(193.0,) * 6])
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
    doa = _FakeDoa(wake_bearing=97.0, angles_per_call=[(97.0,) * 4])
    g = _gate(base_config, fake_mic, fake_vad, fake_embedder, fake_wake,
              fake_runtime, doa)
    _wake(g, fake_wake)
    fake_vad.process_chunk.return_value = [make_segment()]
    g._handle_chunk(np.zeros(480, dtype=np.float32))
    assert g._wake_bearing == 97.0
    g._reset_to_idle()
    assert g._wake_bearing is None
