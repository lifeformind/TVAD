import subprocess
import sys


def test_importing_module_does_not_load_cv2():
    # Watertight: in a FRESH interpreter, importing the backend must not pull cv2
    # into sys.modules (all cv2 use is lazy, inside methods).
    code = (
        "import sys; import modes.director.vision.opencv_backend as m; "
        "assert hasattr(m, 'OpenCvBackend') and hasattr(m, 'cv2_available'); "
        "assert 'cv2' not in sys.modules, 'cv2 was imported at module load!'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd="/home/ldrgx10/FullDuplexVoice/TVAD/target-vad")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


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
