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


def test_enroll_normalizes_to_unit_length():
    """Verify enroll_reference returns L2-normalized embedding on non-unit input."""
    frames = iter([1, 2])
    # embed_fn returns [3.0, 4.0] (norm = 5) for every frame
    ref = enroll_reference(lambda: next(frames, None),
                           lambda f: np.array([3.0, 4.0]),
                           n_frames=2, max_attempts=10)
    assert ref is not None
    # Should be unit length
    assert np.isclose(np.linalg.norm(ref), 1.0)
    # Should point in the right direction: [3,4]/5 = [0.6, 0.8]
    assert np.allclose(ref, [0.6, 0.8])
