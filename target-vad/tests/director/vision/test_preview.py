"""preview.py — annotated debug frames for the tuning console. Overlay draws the
face box + identity score + raw/debounced presence; the writer publishes JPEGs
atomically so a concurrent reader never sees a torn file."""
import subprocess
import sys

import numpy as np
import pytest

from modes.director.vision.preview import annotate, write_jpeg_atomic


def _frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_importing_module_does_not_load_cv2():
    # House rule (opencv_backend.py): cv2 use is lazy, inside functions.
    code = ("import sys; import modes.director.vision.preview as m; "
            "assert hasattr(m, 'annotate') and hasattr(m, 'write_jpeg_atomic'); "
            "assert 'cv2' not in sys.modules, 'cv2 was imported at module load!'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_annotate_draws_on_a_copy():
    pytest.importorskip("cv2")
    frame = _frame()
    out = annotate(frame, box=(20.0, 20.0, 40.0, 40.0), score=0.83,
                   raw_present=True, stable="PRESENT")
    assert out.shape == frame.shape
    assert out.any()                      # something was drawn
    assert not frame.any()                # original untouched


def test_annotate_handles_no_face():
    pytest.importorskip("cv2")
    out = annotate(_frame(), box=None, score=None,
                   raw_present=False, stable="ABSENT")
    assert out.shape == (120, 160, 3) and out.any()   # status text still drawn


def test_write_jpeg_atomic_produces_decodable_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "preview.jpg"
    assert write_jpeg_atomic(_frame(), str(path)) is True
    img = cv2.imread(str(path))
    assert img is not None and img.shape == (120, 160, 3)
    assert not list(tmp_path.glob("*.tmp*"))          # no droppings


def test_write_jpeg_atomic_false_on_unwritable_dir(tmp_path):
    pytest.importorskip("cv2")
    assert write_jpeg_atomic(_frame(), str(tmp_path / "nope" / "x.jpg")) is False


def test_annotate_raw_none_means_no_verdict():
    # Direct-grabber mode (kiosk stopped): no reference to score against.
    pytest.importorskip("cv2")
    out = annotate(_frame(), box=(20.0, 20.0, 40.0, 40.0), score=None,
                   raw_present=None, stable="kiosk stopped (direct)")
    assert out.shape == (120, 160, 3) and out.any()


def test_encode_jpeg_roundtrips():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from modes.director.vision.preview import encode_jpeg
    data = encode_jpeg(_frame())
    assert isinstance(data, bytes) and data[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (120, 160, 3)
