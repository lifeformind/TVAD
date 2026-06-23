# Vision Presence + Identity De-Risk Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway harness that proves or kills cheap, CPU-only camera **presence** and **enrolled-identity** on the GB10, with an honest GO/NO-GO gate whose identity half is an explicit self-vs-stranger discrimination test.

**Architecture:** A single bench script `bench/vision_presence_probe.py` with pure, unit-tested helpers (model fetch, zone/size filter, IOU, presence debounce, cosine, separation report) plus live-measurement glue around OpenCV. Detection = YuNet via `cv2.FaceDetectorYN`. Identity = SFace via `cv2.FaceRecognizerSF` (both ONNX from the OpenCV Zoo, pure-cv2 — no insightface). The numbers-we-must-trust live in tested functions so a measurement bug can't produce a false verdict (the pVAD lesson).

**Tech Stack:** Python 3.12, OpenCV 4.13 (`cv2.FaceDetectorYN`, `cv2.FaceRecognizerSF`), onnxruntime-CPU 1.24, numpy, psutil. No GPU (reserved for the LLM/TTS/STT stack).

## Global Constraints

- **Hardware/runtime:** NVIDIA GB10, aarch64. Vision runs **CPU-only**; never touch the GPU.
- **No new heavy deps:** detection + identity use `cv2` (4.13, present) + onnxruntime (present). **Do NOT add `insightface`** (no aarch64 wheel — confirmed missing). If an embedder beyond SFace is ever tried, it must be a plain ONNX via onnxruntime.
- **Throwaway:** this is a spike. No Director integration, no FSM/events, no production structure. Code lives under `bench/` and `tests/bench/`.
- **Models cached, not committed:** fetch ONNX models to `~/.cache/target-vad/vision/` on first use (like whisper/Kokoro/Silero). Never commit the `.onnx` binaries.
- **Commit trailer:** every commit message MUST end with the line
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Camera:** present at `/dev/video0`, `/dev/video1`; the working index is a spike output, not an assumption.
- **Verdict is the deliverable:** the spike succeeds when `docs/notes/2026-06-23-vision-presence.md` records GO/NO-GO per half with measured numbers.

---

## File Structure

- **Create** `bench/vision_presence_probe.py` — the harness: pure helpers + CLI subcommands (`deps`, `capture`, `presence`, `identity`).
- **Create** `tests/bench/test_vision_presence_probe.py` — unit tests for the pure helpers (no camera needed; CI-safe).
- **Create** `docs/notes/2026-06-23-vision-presence.md` — the GO/NO-GO verdict (Task 4).
- **Modify** memory `pvad-conditioning-inert.md` / `MEMORY.md` — record the verdict pointer (Task 4).

The harness keeps pure logic (testable) separate from live OpenCV/camera glue (measured by a human run). Tests import only the pure helpers, so `pytest` stays green without a camera.

---

### Task 1: Harness scaffold — model fetch + dependency probe + camera capture (Half 1)

**Files:**
- Create: `bench/vision_presence_probe.py`
- Test: `tests/bench/test_vision_presence_probe.py`

**Interfaces:**
- Produces: `ensure_model(url: str, dest: pathlib.Path) -> pathlib.Path` (idempotent download); `probe_deps() -> dict` (importability report); `cmd_capture(index: int, width: int, height: int, seconds: float) -> None` (live).
- Consumes: nothing (first task).

- [ ] **Step 1: Write the failing test for `ensure_model` idempotency**

```python
# tests/bench/test_vision_presence_probe.py
import pathlib
import importlib.util

# Import the bench module by path (it lives outside any package).
_spec = importlib.util.spec_from_file_location(
    "vpp", pathlib.Path(__file__).resolve().parents[2] / "bench" / "vision_presence_probe.py")
vpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpp)


def test_ensure_model_skips_existing(tmp_path):
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"already here")
    # Should NOT attempt any network fetch when a non-empty file already exists.
    out = vpp.ensure_model("http://invalid.invalid/should-not-be-fetched.onnx", dest)
    assert out == dest
    assert dest.read_bytes() == b"already here"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py::test_ensure_model_skips_existing -v`
Expected: FAIL (`bench/vision_presence_probe.py` does not exist → import error).

- [ ] **Step 3: Create the harness with `ensure_model`, `probe_deps`, and a capture command**

```python
# bench/vision_presence_probe.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py::test_ensure_model_skips_existing -v`
Expected: PASS.

- [ ] **Step 5: Run the dependency probe + camera capture live (Half 1 measurement)**

Run:
```bash
PYTHONPATH=. python3 bench/vision_presence_probe.py deps
for i in 0 1; do PYTHONPATH=. python3 bench/vision_presence_probe.py capture --index $i --seconds 5; done
```
Expected: `deps` prints `cv2: 4.13.0`, `FaceDetectorYN: True`, `FaceRecognizerSF: True`. `capture` prints actual resolution + achieved fps for whichever index is the real camera.
**Half 1 GO criterion:** at least one index opens and yields frames at a stable fps. Record the working index + resolution for later tasks. If neither opens → Half 1 NO-GO (stop; report in verdict).

- [ ] **Step 6: Commit**

```bash
git add bench/vision_presence_probe.py tests/bench/test_vision_presence_probe.py
git commit -m "spike(vision): harness scaffold — model fetch + deps + camera capture (Half 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Presence — zone/size filter, IOU, debounce + live YuNet measurement (Half 2)

**Files:**
- Modify: `bench/vision_presence_probe.py` (add helpers + `cmd_presence`)
- Test: `tests/bench/test_vision_presence_probe.py`

**Interfaces:**
- Consumes: `ensure_model`, `YUNET_URL`, `CACHE` from Task 1.
- Produces: `iou(a, b) -> float`; `box_in_zone(box, frame_w, frame_h, zone=(0.2,0.0,0.6,1.0), min_area_frac=0.03) -> bool` (box = `(x, y, w, h)`); `PresenceDebouncer(present_after_s, absent_after_s)` with `update(detected: bool, now: float) -> str` returning `"present"`/`"absent"`; `cmd_presence(index, width, height, seconds, fps) -> None` (live).

- [ ] **Step 1: Write failing tests for `iou`, `box_in_zone`, `PresenceDebouncer`**

```python
# append to tests/bench/test_vision_presence_probe.py

def test_iou_overlap_and_disjoint():
    assert vpp.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert vpp.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    # Half-overlap on x: intersection 5x10=50, union 100+100-50=150 -> 1/3.
    assert abs(vpp.iou((0, 0, 10, 10), (5, 0, 10, 10)) - (50 / 150)) < 1e-6


def test_box_in_zone_center_and_size():
    # frame 320x240; central zone x in [0.2,0.8]. A big centered box passes.
    assert vpp.box_in_zone((130, 80, 60, 80), 320, 240) is True
    # Off to the left edge (center x ~ 0.06) -> outside zone.
    assert vpp.box_in_zone((0, 80, 40, 80), 320, 240) is False
    # Centered but tiny (far away) -> fails min_area_frac.
    assert vpp.box_in_zone((150, 110, 12, 16), 320, 240) is False


def test_presence_debouncer_hysteresis():
    d = vpp.PresenceDebouncer(present_after_s=1.0, absent_after_s=2.0)
    assert d.update(True, 0.0) == "absent"      # not yet continuous 1s
    assert d.update(True, 0.5) == "absent"
    assert d.update(True, 1.0) == "present"      # 1s continuous -> present
    assert d.update(False, 1.5) == "present"     # brief miss, < 2s -> still present
    assert d.update(False, 3.0) == "present"     # 1.5s of absence (<2s from first miss)
    assert d.update(False, 3.6) == "absent"      # >=2s continuous absence -> absent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py -k "iou or zone or debouncer" -v`
Expected: FAIL (`iou`, `box_in_zone`, `PresenceDebouncer` not defined).

- [ ] **Step 3: Implement the pure helpers**

```python
# add to bench/vision_presence_probe.py (above main())

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py -k "iou or zone or debouncer" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add `cmd_presence` (live YuNet detection + measurement)**

```python
# add to bench/vision_presence_probe.py

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
```

Wire it into `main()`:
```python
    pr = sub.add_parser("presence")
    for name, typ, default in [("--index", int, 0), ("--width", int, 320),
                               ("--height", int, 240), ("--seconds", float, 30.0),
                               ("--fps", float, 3.0)]:
        pr.add_argument(name, type=typ, default=default)
    # ... in the dispatch block:
    elif args.cmd == "presence":
        cmd_presence(args.index, args.width, args.height, args.seconds, args.fps)
```

- [ ] **Step 6: Run the presence probe live (Half 2 measurement)**

Run (use the working index from Task 1; stand at kiosk distance for ~30 s, then step out of frame for the last few seconds):
```bash
PYTHONPATH=. python3 bench/vision_presence_probe.py presence --index 0 --seconds 30 --fps 3
```
Expected: per-sample `detected`/`state` log, then a summary line.
**Half 2 GO criteria (record all):** standing → `present_fraction ≥ 0.95` while present; stepping out → flips to `absent`; `detect_ms p95` small (≪ frame period); `proc_cpu` modest (single-digit-to-low-double-digit % of one core at 3 fps). Empty-scene false-present ≤1 over the window. Numbers go in the verdict.

- [ ] **Step 7: Commit**

```bash
git add bench/vision_presence_probe.py tests/bench/test_vision_presence_probe.py
git commit -m "spike(vision): presence — zone/size filter, IOU, debounce + YuNet live probe (Half 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Identity — cosine + separation report + live SFace discrimination test (Half 3)

**Files:**
- Modify: `bench/vision_presence_probe.py` (add helpers + `cmd_identity`)
- Test: `tests/bench/test_vision_presence_probe.py`

**Interfaces:**
- Consumes: `ensure_model`, `YUNET_URL`, `SFACE_URL`, `CACHE`, `box_in_zone` from Tasks 1–2.
- Produces: `cosine(a, b) -> float`; `separation_report(self_scores, cross_scores) -> dict` with keys `threshold`, `self_accept_rate`, `cross_reject_rate`, `self_min`, `cross_max`, `separated`; `cmd_identity(index, width, height, enroll_seconds, test_seconds) -> None` (live).

- [ ] **Step 1: Write failing tests for `cosine` and `separation_report`**

```python
# append to tests/bench/test_vision_presence_probe.py
import numpy as np


def test_cosine_basic():
    assert abs(vpp.cosine(np.array([1.0, 0]), np.array([1.0, 0])) - 1.0) < 1e-6
    assert abs(vpp.cosine(np.array([1.0, 0]), np.array([0, 1.0]))) < 1e-6


def test_separation_report_clean_split():
    # self scores high, cross scores low, fully separable.
    rep = vpp.separation_report([0.7, 0.8, 0.75], [0.1, 0.2, 0.15])
    assert rep["separated"] is True
    assert rep["self_accept_rate"] == 1.0
    assert rep["cross_reject_rate"] == 1.0
    assert rep["cross_max"] < rep["threshold"] <= rep["self_min"]


def test_separation_report_overlap_not_separated():
    # overlapping distributions -> no threshold gives 100/100.
    rep = vpp.separation_report([0.4, 0.55, 0.3], [0.35, 0.5, 0.45])
    assert rep["separated"] is False
    assert min(rep["self_accept_rate"], rep["cross_reject_rate"]) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py -k "cosine or separation" -v`
Expected: FAIL (`cosine`, `separation_report` not defined).

- [ ] **Step 3: Implement the pure helpers**

```python
# add to bench/vision_presence_probe.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py -k "cosine or separation" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add `cmd_identity` (live SFace enroll + self/cross measurement)**

```python
# add to bench/vision_presence_probe.py

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


def cmd_identity(index, width, height, enroll_seconds, test_seconds) -> None:
    """Half 3: enroll person A, then measure self-similarity (A) and cross-similarity
    (a DIFFERENT person B) against A's mean embedding. Prints the separation report."""
    import cv2
    ydet = cv2.FaceDetectorYN.create(str(ensure_model(YUNET_URL, CACHE / "yunet.onnx")),
                                     "", (width, height), 0.7, 0.3, 50)
    rec = cv2.FaceRecognizerSF.create(str(ensure_model(SFACE_URL, CACHE / "sface.onnx")), "")
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def collect(label, seconds):
        embs, t0 = [], time.monotonic()
        print(f"\n>>> {label}: stand at the kiosk for {seconds:.0f}s ...")
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            e = _embed(ydet, rec, frame)
            if e is not None:
                embs.append(e)
            time.sleep(0.2)        # ~5 fps is plenty
        print(f"    collected {len(embs)} embeddings")
        return embs

    enroll = collect("ENROLL person A", enroll_seconds)
    if not enroll:
        print("[identity] no face during enroll — abort", file=sys.stderr)
        cap.release()
        return
    ref = np.mean(np.stack(enroll), axis=0)
    self_e = collect("TEST person A again (self)", test_seconds)
    cross_e = collect("TEST person B (stranger)", test_seconds)
    cap.release()

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
```

Wire into `main()`:
```python
    idp = sub.add_parser("identity")
    for name, typ, default in [("--index", int, 0), ("--width", int, 320),
                               ("--height", int, 240), ("--enroll-seconds", float, 8.0),
                               ("--test-seconds", float, 8.0)]:
        idp.add_argument(name, type=typ, default=default)
    # ... in the dispatch block:
    elif args.cmd == "identity":
        cmd_identity(args.index, args.width, args.height,
                     args.enroll_seconds, args.test_seconds)
```

- [ ] **Step 6: Run the identity probe live with TWO people (Half 3 measurement)**

Run (person A enrolls + re-tests; person B is a different person):
```bash
PYTHONPATH=. python3 bench/vision_presence_probe.py identity --index 0 --enroll-seconds 8 --test-seconds 8
```
Expected: self/cross cosine summaries + a `report` dict + a `HALF-3 GO/NO-GO` line.
**Half 3 GO criterion:** `separated == True` — i.e. a single threshold gives `self_accept_rate == 1.0` and `cross_reject_rate == 1.0` at kiosk distance/lighting. Record the self/cross distributions and threshold. If NO-GO → identity falls back to Tier-1 presence-only (note in verdict).

- [ ] **Step 7: Commit**

```bash
git add bench/vision_presence_probe.py tests/bench/test_vision_presence_probe.py
git commit -m "spike(vision): identity — cosine + separation report + SFace live discrimination (Half 3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Contention run + GO/NO-GO verdict note + memory

**Files:**
- Create: `docs/notes/2026-06-23-vision-presence.md`
- Modify: `/home/ldrgx10/.claude/projects/-home-ldrgx10-FullDuplexVoice-TVAD/memory/pvad-conditioning-inert.md` and `MEMORY.md`

**Interfaces:**
- Consumes: all three live commands from Tasks 1–3.
- Produces: the verdict note (the spike's actual deliverable).

- [ ] **Step 1: Measure contention with the conversation stack running**

In terminal A:
```bash
./kiosk-stack.sh start
```
In terminal B (while a wake session is active / the LLM is loaded):
```bash
PYTHONPATH=. python3 bench/vision_presence_probe.py presence --index 0 --seconds 20 --fps 3
```
Record `proc_cpu` and whether the audio reflex loop / playback shows any added latency or stutter. **Contention GO criterion:** presence runs at the same fps/CPU as standalone, with **no measurable degradation** to the audio stack. (This is the number that actually matters — it gates running vision continuously.)

- [ ] **Step 2: Run the full pytest suite to confirm nothing regressed**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py -v`
Expected: all helper tests PASS (camera-free, CI-safe).

- [ ] **Step 3: Write the verdict note**

Create `docs/notes/2026-06-23-vision-presence.md` mirroring `docs/notes/2026-06-22-pvad.md`. Fill every bracket with the measured numbers from Tasks 1–3 and Step 1:

```markdown
# Vision Presence + Identity — De-Risk Spike Verdict (GB10)

**Date:** 2026-06-23
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, CPU-only vision)
**Spec:** docs/superpowers/specs/2026-06-23-vision-presence-spike-design.md
**Method:** bench/vision_presence_probe.py — YuNet (cv2.FaceDetectorYN) detect +
SFace (cv2.FaceRecognizerSF) identity, both ONNX CPU. Live, two people.

## VERDICT: [GO / GO Tier-1 only / NO-GO]

## Half 1 — camera + capture: [PASS/FAIL]
- working index: [N], resolution [WxH], achieved fps [..]

## Half 2 — presence reliable + cheap: [PASS/FAIL]
- detect_ms p50/p95: [..]/[..]; sampled fps: [..]; proc_cpu: [..]%
- present_fraction while standing: [..]; flips to absent on exit: [yes/no]
- empty-scene false-present: [..]

## Half 3 — identity discriminates: [PASS/FAIL]
- self cosine min/mean: [..]/[..]; cross cosine max/mean: [..]/[..]
- chosen threshold: [..]; self_accept: [..]; cross_reject: [..]; separated: [..]

## Contention (stack running): [PASS/FAIL]
- proc_cpu with stack up: [..]%; audio-loop impact: [none/observed ...]

## Consequences for Sub-project 2
- [if GO] design the Director floor-control integration on these numbers:
  present-keep-alive, absent-timeout T=[..], owner-changed via identity threshold [..].
- [if Tier-1 only] presence-only floor control; swap-detection deferred.
- [if NO-GO] [reason] -> fall back to [audio-only floor control / revisit].
```

- [ ] **Step 4: Update memory with the verdict**

Append a "Vision spike verdict (2026-06-23)" line to `pvad-conditioning-inert.md` (one or two sentences: GO/NO-GO + the headline numbers + pointer to the note), and add an index line to `MEMORY.md`:
`- [Vision presence spike](vision-presence-spike.md) — ...` **only if** you create a dedicated memory file; otherwise just update the existing pointer. (Follow the memory format: frontmatter + body; link `[[pvad-conditioning-inert]]`.)

- [ ] **Step 5: Commit**

```bash
git add docs/notes/2026-06-23-vision-presence.md
git commit -m "spike(vision): GO/NO-GO verdict — cheap camera presence + identity on GB10

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Decide next step with the user**

Present the verdict. If GO (or Tier-1-only GO), the next action is to **brainstorm Sub-project 2** (Director floor-control integration) on the measured numbers — do NOT start integration without that design pass. If NO-GO, bring the fallback options back to the user.

---

## Self-Review

**1. Spec coverage:**
- Half 1 (camera+capture) → Task 1, Steps 5. ✓
- Half 2 (presence reliable + cheap, fps/CPU/latency/reliability) → Task 2, Step 6. ✓
- Half 3 (identity discrimination, self vs cross, single-threshold 100/100) → Task 3, Step 6 + `separation_report`. ✓
- Contention with stack running → Task 4, Step 1. ✓
- Dependency probe / aarch64 wheel risk → Task 1 `probe_deps` + SFace-not-insightface choice (Global Constraints). ✓
- Verdict note deliverable → Task 4, Step 3. ✓
- Memory update → Task 4, Step 4. ✓
- Non-goals (no integration) → honored; plan stops at verdict + a brainstorm handoff. ✓

**2. Placeholder scan:** Live-measurement steps intentionally leave numeric *results* to the run (that is the spike's output), but every step has concrete commands, code, and explicit GO criteria — no "TBD"/"add error handling"/"similar to". The verdict note has brackets because it is a fill-in-on-run artifact, which is correct. ✓

**3. Type consistency:** `box` is `(x,y,w,h)` in `box_in_zone`/`iou`/`cmd_presence`. `separation_report` keys (`threshold`, `self_accept_rate`, `cross_reject_rate`, `self_min`, `cross_max`, `separated`) match the tests and `cmd_identity` usage. `ensure_model(url, dest)`, `cosine(a,b)` signatures consistent across tasks. YuNet/SFace created via `cv2.FaceDetectorYN.create` / `cv2.FaceRecognizerSF.create` consistently. ✓

---

### Task 5: Guided operator prompts for the live runs (UX, added on request)

**Why:** the live runs (Half 2 step-out timing; Half 3 person A → person B swap) are easy to botch without clear cues. Add input-gated phase prompts, per-second on-screen status, an explicit "STEP OUT NOW" cue, and an optional `--preview` camera window with on-frame overlay. No new pure logic; the 7 existing unit tests must still pass and module import must stay cv2-free at load.

**Files:**
- Modify: `bench/vision_presence_probe.py` (add prompt helpers; weave into `cmd_presence` + `cmd_identity`; add `--preview` flag)
- Test: `tests/bench/test_vision_presence_probe.py` (one CI-safe test for `_supports_input`/`_gate` non-interactive no-op)

**Interfaces:**
- Consumes: existing `cmd_presence`, `cmd_identity`, `box_in_zone`, `_largest_face`, `_embed`.
- Produces: `_supports_input() -> bool`; `_gate(message: str) -> None` (pauses for Enter when interactive, else prints + continues); `_countdown(label: str, secs: int = 3) -> None`; `_draw_overlay(frame, lines, box=None)`; `_show(window, frame) -> bool`. New `--preview` store_true arg on `presence` and `identity`.

- [ ] **Step 1: Write a CI-safe failing test for the non-interactive gate**

```python
# append to tests/bench/test_vision_presence_probe.py
def test_gate_is_noop_when_non_interactive(monkeypatch, capsys):
    # Force non-interactive stdin; _gate must print the message and NOT block.
    monkeypatch.setattr(vpp.sys.stdin, "isatty", lambda: False, raising=False)
    vpp._gate("PERSON A: stand at the kiosk")     # must return immediately
    out = capsys.readouterr().out
    assert "PERSON A: stand at the kiosk" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py::test_gate_is_noop_when_non_interactive -v`
Expected: FAIL (`_gate`/`_supports_input` not defined).

- [ ] **Step 3: Add the prompt + preview helpers (above `main()`)**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py::test_gate_is_noop_when_non_interactive -v`
Expected: PASS.

- [ ] **Step 5: Weave guidance into `cmd_presence`**

Add a `preview: bool = False` parameter. Before the loop:
```python
    _gate("[Half 2] PRESENCE — stand in front of the kiosk at arm's length.\n"
          "You will be told to STEP OUT near the end so we can confirm 'absent'.")
    _countdown("presence")
```
Inside the loop, replace the bare per-sample print with a once-per-second status line that also fires the step-out cue and (optionally) a preview frame:
```python
        # ... after computing `state`, `detected`, frame `w,h`, and the matched `box` (or None):
        remaining = seconds - (now - t0)
        print(f"[presence] elapsed={now-t0:5.1f}s remaining={remaining:4.1f}s "
              f"detected={int(detected)} state={state}", flush=True)
        if 4.0 < remaining <= 5.0:
            print("\n>>> STEP OUT OF FRAME NOW — verifying 'absent' <<<\n", flush=True)
        if preview:
            lines = [f"HALF 2 PRESENCE  state={state}  detected={int(detected)}",
                     f"remaining={remaining:4.1f}s",
                     ("STEP OUT NOW" if remaining <= 5.0 else "stand still")]
            if not _show("vision spike", _draw_overlay(frame, lines,
                         box if detected else None)):
                preview = False     # no display -> fall back to terminal-only
```
(Keep the existing summary line.)

- [ ] **Step 6: Weave guidance into `cmd_identity`**

Add a `preview: bool = False` parameter. Replace the inner `collect(label, seconds)` so each phase is gated and shows live face feedback:
```python
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
```
Update the three call sites with explicit person instructions:
```python
    enroll = collect("ENROLL", enroll_seconds,
                     "PERSON A (the enrolled user): stand at the kiosk to ENROLL.", preview)
    # ... abort-if-empty unchanged ...
    self_e = collect("SELF", test_seconds,
                     "PERSON A again: stay at the kiosk to TEST SELF.", preview)
    cross_e = collect("STRANGER", test_seconds,
                      "PERSON B (a DIFFERENT person): stand at the kiosk to TEST STRANGER.", preview)
```
At the end of `cmd_identity`, if a preview window was opened, close it:
```python
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass
```

- [ ] **Step 7: Wire `--preview` into `main()` for both subparsers and pass it through**

```python
    pr.add_argument("--preview", action="store_true")
    idp.add_argument("--preview", action="store_true")
    # dispatch:
    elif args.cmd == "presence":
        cmd_presence(args.index, args.width, args.height, args.seconds, args.fps,
                     preview=args.preview)
    elif args.cmd == "identity":
        cmd_identity(args.index, args.width, args.height,
                     args.enroll_seconds, args.test_seconds, preview=args.preview)
```

- [ ] **Step 8: Run the full suite + confirm cv2-free import**

Run: `PYTHONPATH=. pytest tests/bench/test_vision_presence_probe.py -v`
Expected: 8 passed (7 prior + the new gate test). Import remains cv2-free at load (cv2 only inside `_draw_overlay`/`_show`/`cmd_*`).

- [ ] **Step 9: Commit**

```bash
git add bench/vision_presence_probe.py tests/bench/test_vision_presence_probe.py
git commit -m "spike(vision): guided operator prompts + optional --preview for live runs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
