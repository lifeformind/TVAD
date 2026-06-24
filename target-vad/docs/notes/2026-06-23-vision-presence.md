# Vision Presence + Identity Spike — GO/NO-GO Verdict (GB10)

**Date:** 2026-06-23
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128 GB unified)
**Spike:** Sub-project 1 of camera-driven floor control
**Spec:** `docs/superpowers/specs/2026-06-23-vision-presence-spike-design.md`
**Harness:** `bench/vision_presence_probe.py` (throwaway; kept for re-measurement)
**Method:** live on the GB10 with two people, mirroring the pVAD live test.

## VERDICT: **GO** on all three halves

Cheap, CPU-only camera **presence** (YuNet) and enrolled **identity** (SFace) both
work at kiosk distance/lighting on this hardware. Both run pure-OpenCV on CPU; the
GPU stays free for the conversation stack. Identity passes the *explicit
discrimination test* the pVAD gate skipped — a single threshold gives 100%
accept-self / 100% reject-stranger with a 0.73 cosine margin.

## Backend (dependency probe)

- **insightface: NOT used** — no aarch64 wheel (the risk the spike existed to resolve).
- **Detection:** `cv2.FaceDetectorYN` — YuNet `face_detection_yunet_2023mar.onnx`
  (OpenCV Zoo), CPU.
- **Identity:** `cv2.FaceRecognizerSF` — SFace `face_recognition_sface_2021dec.onnx`
  (OpenCV Zoo), 128-dim embedding, CPU. `rec.alignCrop` + `rec.feature`.
- Both are stock OpenCV — no extra ML runtime, no GPU.

## Half 1 — camera + capture: PASS

- Camera **index 0** streams via `cv2.VideoCapture` at **640×360 @ ~29 fps**.
- **Do NOT set `CAP_PROP_FPS`** on this UVC camera — it silently switches the capture
  mode and YuNet then detects nothing. CPU is kept low instead by `grab()`-ing
  (no decode) between samples.

## Half 2 — presence is reliable AND cheap (Tier 1): PASS

- YuNet detect at 640×360, sampled at ~3 fps, with a zone + min-box-size filter
  (`box_in_zone`, central zone, `min_area_frac=0.015`) and IOU/debounce hysteresis.
- **Detect latency ~12 ms p50.** Reliability: **present_fraction 0.93** over a stand
  (person at kiosk distance reliably detected); empty scene reads absent.
- `min_area_frac` tuned **0.03 → 0.015**: at 640×360 a face at arm's length covers
  ~0.024–0.029 of the frame, so 0.03 flickered; 0.015 is stable with margin.
- **CPU ~59% of one core standalone — but that is almost entirely the 30 fps `grab()`
  loop, not detection (~2%).** A production integration should use a dedicated
  low-rate capture (presence needs ~2–4 fps, not 30).
- **Contention (presence + full `kiosk-stack.sh` running): PASS.** Presence probe
  under the live LLM/TTS/STT stack vs standalone:

  | metric | standalone | under full stack |
  |--------|-----------|------------------|
  | sampled_fps | ~3.0 | **2.9** (held — not starved) |
  | detect_ms p50 / p95 | ~12 | **16.3 / 18.0** |
  | proc_cpu | ~59% | **58.9%** (unchanged) |
  | present_fraction | 0.93 | **0.93** (reliability unchanged) |

  fps held at target and present_fraction unchanged → the stack does not starve
  presence. detect latency rose only 12→16 ms (the stack competing for CPU); 18 ms p95
  is ~18× under the 333 ms sampling period at 3 fps. (Stack was up and serving during
  the run — the latency rise is the contention signature. A peak-load run during active
  TTS/LLM generation would tighten this further but is not needed for the gate.)

## Half 3 — identity actually discriminates (Tier 2): GO — the explicit test

Two live people at kiosk distance/lighting. Enroll Person A → test A (re-approach) →
test Person B. SFace embedding vs A's mean embedding (cosine).

Measurement-hygiene fix found live (kept in the harness): the camera **buffers frames
during the Enter-prompt + countdown**, so the first reads of each phase returned
several-seconds-stale frames — at the A→B swap these embedded the *previous* person
and poisoned the cross distribution (early runs showed exactly 3–4 high outliers
clustered at the phase start: `0.95 0.95 0.95 → 0.00 -0.02 -0.04 …`). Fixed by
draining the capture buffer (`grab()` ×10) before timing + a 1.0 s settle window.
With that, the clean run:

| | n | min | mean | max |
|--------|----|-------|-------|-------|
| self (A vs A) | 33 | **0.789** | 0.932 | 0.98 |
| cross (B vs A) | 33 | −0.04 | 0.010 | **0.057** |

- **Separation gap = 0.789 − 0.057 = 0.73.** Threshold sweep: **threshold 0.789,
  self_accept 1.00, cross_reject 1.00, separated=True.**
- Every genuine Person B frame is **orthogonal** (~0 cosine) to the owner.
- Self dipped to 0.79–0.92 in the re-approach tail (different angle) — still far above
  any reasonable threshold.
- **Recommended operating threshold ~0.40** for margin (well clear of both
  distributions; robust to a bad-angle self frame or a look-alike stranger).
- Embedding cadence in the test was ~5 fps (`time.sleep(0.2)`); production identity
  needs only ~1 fps.

## Consequences for Sub-project 2 (Director floor-control integration)

- **Presence is the floor-control authority** (owner present → keep serving through
  silence; owner absent → free the kiosk fast; owner changed → new customer). Audio
  becomes *content only* — the inert FireRedChat pVAD and ECAPA's ≥2 s window limit
  are no longer load-bearing for "who owns the floor."
- **Identity (Tier 2) is available**, not just presence-only: SFace cleanly separates
  owner vs stranger, so "a stranger stepped into the exact gap" swap-detection is
  feasible — the Tier-1-only fallback is NOT needed.
- Integration must track the **owner as a specific box** (central zone, largest) and
  never average across a person-swap — the stale-buffer/transition contamination seen
  here is a measurement artifact that disappears when the owner box is tracked, but it
  is the same hazard (two faces / handoff) to design around.
- The dormant `SafetyNet`/`Lockout`/verify-before-serve components are NOT wired by
  the rejected "rolling eject" policy; their reuse, if any, is a Sub-project 2 decision.

## Open items (carried into Sub-project 2 design — not gate blockers)

- **Peak-load contention** — the contention run above was under a serving stack; a run
  during *active* TTS/LLM generation would tighten the worst case. Not a gate blocker
  (18 ms p95 detect at a 333 ms period leaves ~18× headroom).
- **Dedicated low-rate capture** — the spike's ~59% one-core CPU is the 30 fps grab
  loop, not detection (~2%). Sub-project 2 should capture at ~2–4 fps so presence costs
  a small fraction of a core.

## Reproduce

```
PYTHONPATH=. python3 bench/vision_presence_probe.py deps
PYTHONPATH=. python3 bench/vision_presence_probe.py capture  --index 0 --width 640 --height 360
PYTHONPATH=. python3 bench/vision_presence_probe.py presence --index 0 --width 640 --height 360 --seconds 30 --fps 3
PYTHONPATH=. python3 bench/vision_presence_probe.py identity --index 0 --width 640 --height 360 --enroll-seconds 8 --test-seconds 8
```
