"""Pure presence classification + debounce (no cv2, no I/O). The live cv2 adapter
(opencv_backend.py) supplies the face embedding + box; this decides PRESENT/ABSENT.
Logic ported from the validated spike bench/vision_presence_probe.py."""
import numpy as np

from modes.director.events import PresenceStatus


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _box_in_zone(box, frame_w, frame_h, zone, min_area_frac) -> bool:
    x, y, w, h = box
    cx, cy = (x + w / 2) / frame_w, (y + h / 2) / frame_h
    zx, zy, zw, zh = zone
    in_zone = (zx <= cx <= zx + zw) and (zy <= cy <= zy + zh)
    big_enough = (w * h) / (frame_w * frame_h) >= min_area_frac
    return bool(in_zone and big_enough)


def classify_presence(face_embedding, box, frame_w, frame_h, reference, *,
                      identity_threshold, min_area_frac,
                      zone=(0.2, 0.0, 0.6, 1.0)) -> PresenceStatus:
    """PRESENT only when the central, large-enough face matches the owner reference
    (cosine >= identity_threshold). No face, off-center, too small, or a stranger
    (low cosine) all read ABSENT. UNAVAILABLE is the caller's job (no reference /
    detector error), NOT here."""
    if face_embedding is None or box is None:
        return PresenceStatus.ABSENT
    if not _box_in_zone(box, frame_w, frame_h, zone, min_area_frac):
        return PresenceStatus.ABSENT
    if cosine(face_embedding, reference) >= identity_threshold:
        return PresenceStatus.PRESENT
    return PresenceStatus.ABSENT


class PresenceDebouncer:
    """Hysteresis over raw per-frame detections. 'present' after present_after_s of
    continuous detection, 'absent' after absent_after_s of continuous non-detection.
    Starts 'absent'. (Ported verbatim from the spike.)"""

    def __init__(self, present_after_s=1.0, absent_after_s=2.0):
        self._pa = present_after_s
        self._aa = absent_after_s
        self._state = "absent"
        self._since = None

    def update(self, detected: bool, now: float) -> str:
        if self._since is None or self._since[0] != detected:
            self._since = (detected, now)
        run = now - self._since[1]
        if self._state == "absent" and detected and run >= self._pa:
            self._state = "present"
        elif self._state == "present" and not detected and run >= self._aa:
            self._state = "absent"
        return self._state
