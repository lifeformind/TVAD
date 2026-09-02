"""DirectGrabber — the tuning console's own camera path while the kiosk is
STOPPED (the kiosk's VisionWorker owns the camera when running; V4L2 access is
exclusive). Lazy open on first grab, annotated JPEG out (face box only — there
is no session reference to score identity against), released by the server
before the kiosk child starts and on server shutdown."""
from modes.director.vision.opencv_backend import OpenCvBackend
from modes.director.vision.preview import annotate, encode_jpeg


class DirectGrabber:
    def __init__(self, vision_cfg: dict, _backend=None):
        # _backend is a test seam (house rule: production code omits it).
        self._backend = _backend or OpenCvBackend(
            camera_index=vision_cfg.get("camera_index", 0),
            width=vision_cfg.get("width", 640),
            height=vision_cfg.get("height", 360),
            identity_threshold=vision_cfg.get("identity_threshold", 0.40),
            min_area_frac=vision_cfg.get("min_area_frac", 0.015))
        self._opened = False

    def grab_jpeg(self):
        """Annotated JPEG bytes of the current camera view, or None."""
        if not self._opened:
            if not self._backend.open():
                return None
            self._opened = True
        frame = self._backend.grab()
        if frame is None:
            return None
        out = annotate(frame, box=self._backend.detect_box(frame), score=None,
                       raw_present=None, stable="kiosk stopped (direct)")
        return encode_jpeg(out)

    def close(self) -> None:
        self._backend.close()
        self._opened = False
