import importlib


def test_module_imports_without_cv2():
    # Importing the backend must NOT import cv2 at module load (lazy inside methods).
    mod = importlib.import_module("modes.director.vision.opencv_backend")
    assert hasattr(mod, "OpenCvBackend")
    assert hasattr(mod, "cv2_available")


def test_cv2_available_is_bool():
    from modes.director.vision.opencv_backend import cv2_available
    assert isinstance(cv2_available(), bool)


def test_open_returns_false_on_bad_index(monkeypatch):
    # With no usable camera, open() must return False (never raise).
    from modes.director.vision.opencv_backend import OpenCvBackend, cv2_available
    if not cv2_available():
        import pytest
        pytest.skip("cv2 not installed in this environment")
    b = OpenCvBackend(camera_index=999, width=640, height=360,
                      identity_threshold=0.40, min_area_frac=0.015)
    assert b.open() is False
    b.close()
