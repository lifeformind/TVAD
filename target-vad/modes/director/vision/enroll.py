"""Owner face enrollment — capture a stable reference embedding at session start.
A failed enroll returns None; the VisionWorker then reports UNAVAILABLE (never
ABSENT), so the camera can't falsely free the kiosk when it simply can't see a face."""
from typing import Optional

import numpy as np


def enroll_reference(grab_fn, embed_fn, *, n_frames, max_attempts) -> Optional[np.ndarray]:
    embs = []
    attempts = 0
    while len(embs) < n_frames and attempts < max_attempts:
        attempts += 1
        frame = grab_fn()
        if frame is None:
            continue
        e = embed_fn(frame)
        if e is not None:
            embs.append(np.asarray(e, dtype=np.float64).ravel())
    if not embs:
        return None
    mean = np.mean(np.stack(embs), axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm else mean
