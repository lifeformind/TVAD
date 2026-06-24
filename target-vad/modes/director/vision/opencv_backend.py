"""Live OpenCV adapter for camera presence/identity (YuNet detect + SFace embed).
ALL cv2 use is lazy inside methods so importing this module never needs cv2 — a
missing/broken cv2 degrades to VisionWorker=None upstream. Models + logic are the
validated spike (bench/vision_presence_probe.py)."""
import pathlib
import sys
import urllib.request
from typing import Optional

import numpy as np

from modes.director.events import PresenceStatus
from modes.director.vision.classify import classify_presence

CACHE = pathlib.Path.home() / ".cache" / "target-vad" / "vision"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")


def cv2_available() -> bool:
    try:
        import cv2  # noqa: F401
        return hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
    except Exception:
        return False


def _ensure_model(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


class OpenCvBackend:
    def __init__(self, camera_index, width, height, identity_threshold, min_area_frac):
        self._index = camera_index
        self._w = width
        self._h = height
        self._thr = identity_threshold
        self._min_area = min_area_frac
        self._cap = None
        self._det = None
        self._rec = None

    def open(self) -> bool:
        self._det = self._rec = None
        try:
            import cv2
            self._det = cv2.FaceDetectorYN.create(
                str(_ensure_model(YUNET_URL, CACHE / "yunet.onnx")), "",
                (self._w, self._h), 0.7, 0.3, 50)
            self._rec = cv2.FaceRecognizerSF.create(
                str(_ensure_model(SFACE_URL, CACHE / "sface.onnx")), "")
            cap = cv2.VideoCapture(self._index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
            # NB: do NOT set CAP_PROP_FPS — it switches this UVC camera's mode.
            if not cap.isOpened():
                cap.release()
                return False
            ok, _ = cap.read()
            if not ok:
                cap.release()
                return False
            self._cap = cap
            return True
        except Exception as exc:   # noqa: BLE001
            print(f"[vision] OpenCvBackend.open failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return False

    def grab(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def _largest_face(self, frame):
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: float(f[2]) * float(f[3]))

    def embed(self, frame) -> Optional[np.ndarray]:
        f = self._largest_face(frame)
        if f is None:
            return None
        aligned = self._rec.alignCrop(frame, f)
        return self._rec.feature(aligned).ravel().copy()

    def make_classify_fn(self, reference):
        """Bind classify_presence to live detect+embed of the largest central face."""
        def classify_fn(frame) -> PresenceStatus:
            h, w = frame.shape[:2]
            f = self._largest_face(frame)
            if f is None:
                return classify_presence(None, None, w, h, reference,
                                         identity_threshold=self._thr,
                                         min_area_frac=self._min_area)
            box = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
            aligned = self._rec.alignCrop(frame, f)
            emb = self._rec.feature(aligned).ravel()
            return classify_presence(emb, box, w, h, reference,
                                     identity_threshold=self._thr,
                                     min_area_frac=self._min_area)
        return classify_fn

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._det = None
        self._rec = None
