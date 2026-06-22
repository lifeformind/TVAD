import os
import numpy as np
import pytest
from core.speaker.enrollment_store import EnrollmentStore


def test_holdout_captured_before_finalize_deletes(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    store.enroll("u1", e1)
    store.enroll("u1", e2)

    holdout = store.holdout_utterance_embedding("u1")    # BEFORE finalize
    assert np.allclose(holdout, e2)                       # last utterance row

    store.finalize_enrollment("u1")                       # deletes utterances file
    assert not os.path.exists(os.path.join(str(tmp_path), "u1_utterances.npy"))


def test_holdout_missing_raises(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.holdout_utterance_embedding("nobody")
