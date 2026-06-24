import numpy as np
from modes.director.vision.enroll import enroll_reference


def test_enroll_means_embeddings():
    frames = iter(["a", "b", "c"])
    embs = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    ref = enroll_reference(lambda: next(frames, None),
                           lambda f: embs[f], n_frames=3, max_attempts=10)
    assert ref is not None
    assert np.allclose(ref, [1.0, 0.0])


def test_enroll_skips_no_face_frames():
    seq = iter(["x", None, "x"])
    ref = enroll_reference(lambda: next(seq, None),
                           lambda f: np.array([0.0, 1.0]) if f else None,
                           n_frames=2, max_attempts=10)
    assert ref is not None
    assert np.allclose(ref / np.linalg.norm(ref), [0.0, 1.0])


def test_enroll_returns_none_when_no_face_ever():
    ref = enroll_reference(lambda: "frame", lambda f: None,
                           n_frames=3, max_attempts=5)
    assert ref is None
