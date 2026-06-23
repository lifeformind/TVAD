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


def cmd_presence(index: int, width: int, height: int, seconds: float, fps: float,
                 preview: bool = False) -> None:
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
    _gate("[Half 2] PRESENCE — stand in front of the kiosk at arm's length.\n"
          "You will be told to STEP OUT near the end so we can confirm 'absent'.")
    _countdown("presence")
    deb = PresenceDebouncer()
    proc = psutil.Process()
    proc.cpu_percent(None)            # prime
    period = 1.0 / fps
    lat, present_frames, total = [], 0, 0
    last_print = -1.0
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
        box = None
        if faces is not None:
            for f in faces:
                candidate = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
                if box_in_zone(candidate, w, h):
                    detected = True
                    box = candidate
                    break
        state = deb.update(detected, now)
        total += 1
        present_frames += 1 if state == "present" else 0
        if now - last_print >= 1.0:
            last_print = now
            remaining = seconds - (now - t0)
            print(f"[presence] elapsed={now-t0:5.1f}s remaining={remaining:4.1f}s "
                  f"detected={int(detected)} state={state}", flush=True)
            if 4.0 < remaining <= 5.0:
                print("\n>>> STEP OUT OF FRAME NOW — verifying 'absent' <<<\n", flush=True)
        if preview:
            remaining = seconds - (now - t0)
            lines = [f"HALF 2 PRESENCE  state={state}  detected={int(detected)}",
                     f"remaining={remaining:4.1f}s",
                     ("STEP OUT NOW" if remaining <= 5.0 else "stand still")]
            if not _show("vision spike", _draw_overlay(frame, lines,
                         box if detected else None)):
                preview = False     # no display -> fall back to terminal-only
    cap.release()
    cpu = proc.cpu_percent(None)
    arr = np.array(lat) if lat else np.array([0.0])
    print(f"\n[presence] samples={total} detect_ms p50={np.percentile(arr,50):.2f} "
          f"p95={np.percentile(arr,95):.2f} sampled_fps={total/seconds:.1f} "
          f"proc_cpu={cpu:.1f}% present_fraction={present_frames/max(total,1):.2f}")


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def separation_report(self_scores, cross_scores) -> dict:
    """Sweep candidate thresholds over the score range; pick the one maximizing
    min(accept-self, reject-stranger). `separated` is True only when BOTH reach 1.0
    on this data (the Half-3 GO bar)."""
    s = np.asarray(self_scores, dtype=np.float64)
    c = np.asarray(cross_scores, dtype=np.float64)
    cands = np.unique(np.concatenate([s, c]))
    best = {"threshold": 0.0, "self_accept_rate": 0.0, "cross_reject_rate": 0.0}
    best_min = -1.0
    for t in cands:
        sa = float(np.mean(s >= t)) if s.size else 0.0
        cr = float(np.mean(c < t)) if c.size else 0.0
        if min(sa, cr) > best_min:
            best_min = min(sa, cr)
            best = {"threshold": float(t), "self_accept_rate": sa, "cross_reject_rate": cr}
    best["self_min"] = float(s.min()) if s.size else 0.0
    best["cross_max"] = float(c.max()) if c.size else 0.0
    best["separated"] = bool(best["self_accept_rate"] == 1.0 and best["cross_reject_rate"] == 1.0)
    return best


def _largest_face(det, frame):
    """Return the YuNet face row (15-vec) with the largest box, or None."""
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: float(f[2]) * float(f[3]))


def _embed(det, rec, frame):
    """YuNet-detect the largest face, SFace-align+embed -> 128-vec (or None)."""
    f = _largest_face(det, frame)
    if f is None:
        return None
    aligned = rec.alignCrop(frame, f)
    return rec.feature(aligned).ravel().copy()


def cmd_identity(index, width, height, enroll_seconds, test_seconds,
                 preview: bool = False) -> None:
    """Half 3: enroll person A, then measure self-similarity (A) and cross-similarity
    (a DIFFERENT person B) against A's mean embedding. Prints the separation report."""
    import cv2
    ydet = cv2.FaceDetectorYN.create(str(ensure_model(YUNET_URL, CACHE / "yunet.onnx")),
                                     "", (width, height), 0.7, 0.3, 50)
    rec = cv2.FaceRecognizerSF.create(str(ensure_model(SFACE_URL, CACHE / "sface.onnx")), "")
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def collect(label, seconds, who, preview_flag):
        _gate(f"[Half 3] {who}")
        _countdown(label)
        embs, t0, last_print = [], time.monotonic(), -1.0
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            e = _embed(ydet, rec, frame)
            if e is not None:
                embs.append(e)
            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                print(f"[{label}] elapsed={now-t0:4.1f}/{seconds:.0f}s "
                      f"face_this_frame={'YES' if e is not None else 'no '} "
                      f"collected={len(embs)}", flush=True)
            if preview_flag:
                lines = [f"HALF 3 {label}", f"collected={len(embs)}",
                         "face: YES" if e is not None else "face: no"]
                if not _show("vision spike", _draw_overlay(frame, lines)):
                    preview_flag = False
            time.sleep(0.2)
        print(f"    collected {len(embs)} embeddings")
        return embs

    enroll = collect("ENROLL", enroll_seconds,
                     "PERSON A (the enrolled user): stand at the kiosk to ENROLL.", preview)
    if not enroll:
        print("[identity] no face during enroll — abort", file=sys.stderr)
        cap.release()
        return
    ref = np.mean(np.stack(enroll), axis=0)
    self_e = collect("SELF", test_seconds,
                     "PERSON A again: stay at the kiosk to TEST SELF.", preview)
    cross_e = collect("STRANGER", test_seconds,
                      "PERSON B (a DIFFERENT person): stand at the kiosk to TEST STRANGER.", preview)
    cap.release()
    try:
        import cv2 as _cv2
        _cv2.destroyAllWindows()
    except Exception:
        pass

    self_scores = [cosine(e, ref) for e in self_e]
    cross_scores = [cosine(e, ref) for e in cross_e]
    rep = separation_report(self_scores, cross_scores)
    print(f"\n[identity] self  n={len(self_scores)} "
          f"min={min(self_scores, default=0):.3f} mean={np.mean(self_scores or [0]):.3f}")
    print(f"[identity] cross n={len(cross_scores)} "
          f"max={max(cross_scores, default=0):.3f} mean={np.mean(cross_scores or [0]):.3f}")
    print(f"[identity] report: {rep}")
    print(f"[identity] HALF-3 {'GO' if rep['separated'] else 'NO-GO'} "
          f"(threshold={rep['threshold']:.3f}, "
          f"self_accept={rep['self_accept_rate']:.2f}, cross_reject={rep['cross_reject_rate']:.2f})")


def _supports_input() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _gate(message: str) -> None:
    """Pause until the operator presses Enter so the right person is in frame
    before a phase. Prints + continues (no block) when stdin isn't interactive."""
    print("\n" + "=" * 64 + f"\n{message}\n" + "=" * 64, flush=True)
    if _supports_input():
        try:
            input(">>> Press Enter to start... ")
        except EOFError:
            pass


def _countdown(label: str, secs: int = 3) -> None:
    for k in range(secs, 0, -1):
        print(f"[{label}] starting in {k}...", flush=True)
        time.sleep(1.0)


def _draw_overlay(frame, lines, box=None):
    import cv2
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    yy = 24
    for ln in lines:
        cv2.putText(frame, ln, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        yy += 26
    return frame


def _show(window: str, frame) -> bool:
    """Show a preview frame; return False if no display (caller disables preview)."""
    import cv2
    try:
        cv2.imshow(window, frame)
        cv2.waitKey(1)
        return True
    except Exception:
        return False


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
    pr.add_argument("--preview", action="store_true")
    idp = sub.add_parser("identity")
    for name, typ, default in [("--index", int, 0), ("--width", int, 320),
                               ("--height", int, 240), ("--enroll-seconds", float, 8.0),
                               ("--test-seconds", float, 8.0)]:
        idp.add_argument(name, type=typ, default=default)
    idp.add_argument("--preview", action="store_true")
    args = p.parse_args()
    if args.cmd == "deps":
        for k, v in probe_deps().items():
            print(f"{k}: {v}")
    elif args.cmd == "capture":
        cmd_capture(args.index, args.width, args.height, args.seconds)
    elif args.cmd == "presence":
        cmd_presence(args.index, args.width, args.height, args.seconds, args.fps,
                     preview=args.preview)
    elif args.cmd == "identity":
        cmd_identity(args.index, args.width, args.height,
                     args.enroll_seconds, args.test_seconds, preview=args.preview)


if __name__ == "__main__":
    main()
