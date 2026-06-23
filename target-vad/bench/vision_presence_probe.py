"""Throwaway de-risk spike: cheap camera presence + enrolled identity on the GB10.

Pure helpers (model fetch, zone/size filter, IOU, presence debounce, cosine,
separation report) are unit-tested in tests/bench/. The live OpenCV/camera glue is
measured by a human run. See docs/superpowers/specs/2026-06-23-vision-presence-spike-design.md.
"""
import argparse
import pathlib
import sys
import time
import urllib.request

import numpy as np

CACHE = pathlib.Path.home() / ".cache" / "target-vad" / "vision"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")


def ensure_model(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download `url` to `dest` once; skip if a non-empty file already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def probe_deps() -> dict:
    """Report importability of the pieces the spike needs (no network, no camera)."""
    report = {}
    try:
        import cv2
        report["cv2"] = cv2.__version__
        report["FaceDetectorYN"] = hasattr(cv2, "FaceDetectorYN")
        report["FaceRecognizerSF"] = hasattr(cv2, "FaceRecognizerSF")
    except Exception as e:  # noqa: BLE001
        report["cv2"] = f"MISSING ({type(e).__name__})"
    for m in ("onnxruntime", "numpy", "psutil"):
        try:
            report[m] = __import__(m).__version__
        except Exception as e:  # noqa: BLE001
            report[m] = f"MISSING ({type(e).__name__})"
    return report


def cmd_capture(index: int, width: int, height: int, seconds: float) -> None:
    """Half 1: open the camera, grab frames, report resolution + achieved fps."""
    import cv2
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[capture] index {index}: FAILED to open", file=sys.stderr)
        return
    n, t0 = 0, time.monotonic()
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while time.monotonic() - t0 < seconds:
        ok, _ = cap.read()
        if not ok:
            print("[capture] read() returned False", file=sys.stderr)
            break
        n += 1
    dt = time.monotonic() - t0
    cap.release()
    print(f"[capture] index={index} actual_res={aw}x{ah} frames={n} "
          f"fps={n/dt:.1f} over {dt:.1f}s")


def iou(a, b) -> float:
    """IOU of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_in_zone(box, frame_w, frame_h, zone=(0.2, 0.0, 0.6, 1.0),
                min_area_frac=0.03) -> bool:
    """True if the box CENTER is inside the fractional `zone` (zx,zy,zw,zh) AND the
    box covers at least `min_area_frac` of the frame (close enough to be the user)."""
    x, y, w, h = box
    cx, cy = (x + w / 2) / frame_w, (y + h / 2) / frame_h
    zx, zy, zw, zh = zone
    in_zone = (zx <= cx <= zx + zw) and (zy <= cy <= zy + zh)
    big_enough = (w * h) / (frame_w * frame_h) >= min_area_frac
    return bool(in_zone and big_enough)


class PresenceDebouncer:
    """Hysteresis over raw per-frame detections. Flips to 'present' after
    `present_after_s` of continuous detection, to 'absent' after `absent_after_s`
    of continuous non-detection. Starts 'absent'."""

    def __init__(self, present_after_s=1.0, absent_after_s=2.0):
        self._pa = present_after_s
        self._aa = absent_after_s
        self._state = "absent"
        self._since = None      # (value, start_time) of the current run

    def update(self, detected: bool, now: float) -> str:
        if self._since is None or self._since[0] != detected:
            self._since = (detected, now)
        run = now - self._since[1]
        if self._state == "absent" and detected and run >= self._pa:
            self._state = "present"
        elif self._state == "present" and not detected and run >= self._aa:
            self._state = "absent"
        return self._state


def cmd_presence(index: int, width: int, height: int, seconds: float, fps: float) -> None:
    """Half 2: YuNet detect at `fps`, apply zone/size filter + debounce, and report
    detect latency, achieved fps, CPU%, and present/absent reliability."""
    import cv2
    import psutil
    model = ensure_model(YUNET_URL, CACHE / "yunet.onnx")
    det = cv2.FaceDetectorYN.create(str(model), "", (width, height),
                                    score_threshold=0.7, nms_threshold=0.3, top_k=50)
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    deb = PresenceDebouncer()
    proc = psutil.Process()
    proc.cpu_percent(None)            # prime
    period = 1.0 / fps
    lat, present_frames, total = [], 0, 0
    t0, next_t = time.monotonic(), time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        if now < next_t:
            continue
        next_t = now + period
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        ts = time.monotonic()
        _, faces = det.detect(frame)
        lat.append((time.monotonic() - ts) * 1000)
        detected = False
        if faces is not None:
            for f in faces:
                box = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
                if box_in_zone(box, w, h):
                    detected = True
                    break
        state = deb.update(detected, now)
        total += 1
        present_frames += 1 if state == "present" else 0
        print(f"[presence] t={now-t0:5.1f}s detected={int(detected)} state={state}")
    cap.release()
    cpu = proc.cpu_percent(None)
    arr = np.array(lat) if lat else np.array([0.0])
    print(f"\n[presence] samples={total} detect_ms p50={np.percentile(arr,50):.2f} "
          f"p95={np.percentile(arr,95):.2f} sampled_fps={total/seconds:.1f} "
          f"proc_cpu={cpu:.1f}% present_fraction={present_frames/max(total,1):.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deps")
    c = sub.add_parser("capture")
    c.add_argument("--index", type=int, default=0)
    c.add_argument("--width", type=int, default=320)
    c.add_argument("--height", type=int, default=240)
    c.add_argument("--seconds", type=float, default=10.0)
    pr = sub.add_parser("presence")
    for name, typ, default in [("--index", int, 0), ("--width", int, 320),
                               ("--height", int, 240), ("--seconds", float, 30.0),
                               ("--fps", float, 3.0)]:
        pr.add_argument(name, type=typ, default=default)
    args = p.parse_args()
    if args.cmd == "deps":
        for k, v in probe_deps().items():
            print(f"{k}: {v}")
    elif args.cmd == "capture":
        cmd_capture(args.index, args.width, args.height, args.seconds)
    elif args.cmd == "presence":
        cmd_presence(args.index, args.width, args.height, args.seconds, args.fps)


if __name__ == "__main__":
    main()
