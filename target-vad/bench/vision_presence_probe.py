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


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deps")
    c = sub.add_parser("capture")
    c.add_argument("--index", type=int, default=0)
    c.add_argument("--width", type=int, default=320)
    c.add_argument("--height", type=int, default=240)
    c.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    if args.cmd == "deps":
        for k, v in probe_deps().items():
            print(f"{k}: {v}")
    elif args.cmd == "capture":
        cmd_capture(args.index, args.width, args.height, args.seconds)


if __name__ == "__main__":
    main()
