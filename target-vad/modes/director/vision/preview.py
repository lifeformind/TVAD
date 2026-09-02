"""Annotated debug frames for the tuning console's camera panel.

annotate() draws what the presence pipeline saw (face box, SFace identity
score, raw + debounced verdict) so a false ABSENT explains itself on screen.
write_jpeg_atomic() publishes frames via temp-file + os.replace so a reader
polling the path never sees a torn JPEG. House rule: ALL cv2 use is lazy
inside functions — importing this module never needs cv2."""
import os
import tempfile
import time


def annotate(frame, *, box, score, raw_present, stable):
    """Return a COPY of frame with the debug overlay drawn. box is (x, y, w, h)
    or None (no face); score is the SFace cosine or None; raw_present True/False
    is the per-frame verdict, None = no verdict (direct-grabber mode, no session
    reference); stable is the debounced status name shown to the Director."""
    import cv2
    out = frame.copy()
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        color = ((0, 200, 200) if raw_present is None       # BGR: yellow = n/a
                 else (0, 200, 0) if raw_present else (0, 0, 220))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    raw = "n/a" if raw_present is None else ("PRESENT" if raw_present else "ABSENT")
    score_txt = f" cos={score:.2f}" if score is not None else ""
    lines = [f"raw={raw}{score_txt}", f"stable={stable}",
             time.strftime("%H:%M:%S")]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def encode_jpeg(frame):
    """JPEG bytes for frame, or None on encode failure."""
    import cv2
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def write_jpeg_atomic(frame, path) -> bool:
    """Encode frame as JPEG and publish it atomically at path. Returns False on
    any failure — preview must never take down the vision thread."""
    try:
        data = encode_jpeg(frame)
        if data is None:
            return False
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".preview-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return True
    except Exception:                     # noqa: BLE001 — best-effort publisher
        return False
