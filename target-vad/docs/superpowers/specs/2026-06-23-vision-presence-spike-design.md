# Vision Capability Spike — Camera Presence + Enrolled Identity (Design)

**Date:** 2026-06-23
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128 GB unified)
**Status:** Design approved; spec for a de-risk spike (Sub-project 1 of camera-driven floor control)
**Camera:** present on the box — `/dev/video0`, `/dev/video1` enumerate.

## Why this spike exists

The kiosk needs **crowd focus / floor control**: serve only the customer who started
the session, tolerate ambient barge-ins in a lively space, and hand the kiosk to a
new customer when the original leaves — without burdening the customer ("say
continue") and without the audio-identity gymnastics that ECAPA can't do reliably on
short, overlapping speech.

Brainstorming concluded that **presence is a physical fact, not an acoustic one**, and
a camera measures it directly. The plan: a vision subsystem becomes the floor-control
authority (owner present → keep serving through silence; owner absent → free the
kiosk fast; owner changed → new customer), while audio handles *content*. That makes
the hard audio problems (the inert FireRedChat pVAD; ECAPA's ≥2 s window limit) mostly
moot.

**But camera presence + identity on this hardware is currently an UNVALIDATED
capability assumption.** We just spent a detour because the Plan-05 pVAD passed a
GO-gate that checked only "it loads + it's fast" and never tested whether it actually
discriminated speakers (it didn't — see memory `pvad-conditioning-inert`). This spike
exists so we **do not repeat that mistake**: prove or kill cheap vision presence +
identity, with an explicit discrimination test, before designing any integration.

## Goal

On the actual GB10 + kiosk camera, answer with measured numbers:

1. Can we capture frames cheaply?
2. Is **presence** (is the customer standing there?) reliable and CPU-cheap enough to
   run continuously alongside the LLM/TTS/STT stack?
3. Does **identity** (is it the *enrolled* customer, vs a stranger?) actually
   discriminate at kiosk distance/lighting?

Throwaway harness. **No Director integration, no events, no session logic.**

## The GO/NO-GO gate — three halves

The pVAD gate had analogues of Halves 1–2 only. **Half 3 is the lesson learned** and
is mandatory for a GO on identity.

### Half 1 — Camera + capture (PASS/FAIL)
- A camera enumerates and streams via OpenCV (`cv2.VideoCapture`) on the GB10.
- Record: device index that works, usable resolution(s), native fps, pixel format
  quirks (e.g. MJPG vs YUYV).
- **NO-GO** if no camera streams. (Devices exist at `/dev/video0/1`; confirm one
  actually yields frames — USB cams sometimes enumerate but fail to open.)

### Half 2 — Presence is reliable AND cheap (Tier 1)
- Detector: **YuNet** (`cv2.FaceDetectorYN`) at low resolution (≈320×240), sampled at
  **2–4 fps** (presence does not need framerate).
- Apply a **zone + min-box-size filter**: a face is "the customer" only if it is in
  the central region and large enough (close to the kiosk), so passers-by in the
  background don't register.
- Continuity: a trivial **IOU/centroid tracker** to confirm "the same blob stayed."
- **Measure:**
  - per-frame detection latency (ms),
  - achievable fps at the sampling cadence,
  - **CPU% while the full conversation stack (llama.cpp + Kokoro + Whisper) is
    running** — contention is the real risk, not raw speed,
  - any GPU contention (there should be none; YuNet is CPU).
- **Reliability targets (measured live):**
  - person standing at kiosk distance → detected in **≥95%** of sampled frames over a
    30 s stand,
  - empty scene → **≤1 false "present"** over the same window.
- **Budget to beat (stated, not assumed):** detection cost ≪ one CPU core at the
  sampling cadence, and **no measurable added latency** to the audio reflex loop or
  GPU stack. Exact pass numbers recorded in the verdict.

### Half 3 — Identity actually discriminates (Tier 2) — THE explicit test
- Embedder: **InsightFace `buffalo_s`** (or `buffalo_sc`) if it imports on aarch64;
  else a plain **MobileFaceNet ONNX** via onnxruntime-CPU (already in the stack).
  Run at **~1 fps**.
- Capture an enrolled face embedding for **person A** at "wake."
- **Measure both directions** at kiosk distance/lighting, with two live people:
  - **self-similarity:** A vs A across frames and a re-approach (cosine),
  - **cross-similarity:** A vs **person B** (a different person standing at the
    kiosk).
- **GO requires a real separation:** a single threshold that yields **~100%
  accept-self / ~100% reject-stranger** on this data. Report the self and cross cosine
  distributions and the chosen threshold. (This is exactly the matching-vs-non-matching
  test the pVAD gate skipped.)
- Also measure embedder latency + CPU at 1 fps.
- **NO-GO (Tier 2 only):** if self ≈ stranger (no usable threshold). Fallback:
  **Tier 1 presence-only** — still fixes "customer present vs gone" and kiosk
  turnover; loses only the "stranger stepped into the exact gap" swap-detection, which
  audio near-field + the wake flow can partially cover.

## Method

- A single throwaway harness: `bench/vision_presence_probe.py`.
- Run **live on the GB10 with two people** (operator + a second person), mirroring the
  pVAD live test. The harness prints:
  - capture info (Half 1),
  - rolling fps / per-frame ms / CPU% and a present/absent log (Half 2),
  - a self-vs-stranger cosine table + suggested threshold (Half 3).
- Optionally run once **with the conversation stack up** (`kiosk-stack.sh`) to capture
  the contention numbers honestly.
- Dependency probe first: the harness checks importability of `cv2.FaceDetectorYN`
  and the chosen embedder and reports which backend it used, so an aarch64 wheel gap
  (we were burned by `faster-whisper`) surfaces as a logged fallback, not a crash.

## Deliverable

- **Verdict note:** `docs/notes/2026-06-23-vision-presence.md` (mirroring
  `docs/notes/2026-06-22-pvad.md`): GO/NO-GO per half, measured numbers, chosen
  models, dependency notes, and any kill reason + its fallback.
- **Throwaway probe:** `bench/vision_presence_probe.py` (kept for re-measurement;
  not production code).
- A memory update capturing the verdict (feasible/cheap or killed), so it is not
  re-derived.

## Non-goals (YAGNI)

- No Director events, FSM changes, or session/floor logic.
- No presence-debounce / absence-timeout tuning (that's integration).
- No two-faces tiebreak, no owner-changed policy, no privacy/retention design.
- No production code structure — the probe is a spike.

These belong to **Sub-project 2 (Director floor-control integration)**, which is
specced only **after** this spike returns GO, on the measured numbers.

## Defaults (confirmed)

- **Time-box:** small — a few hours, like the pVAD spike.
- **Runtime target:** onnxruntime-CPU for the embedder; OpenCV-CPU for detection.
  Keep the GPU for the conversation stack.

## Risks / open questions the spike resolves

- aarch64 wheel availability for the face embedder (InsightFace build vs plain ONNX).
- CPU contention with the live LLM/TTS/STT stack (the number that actually matters).
- Whether `buffalo_s`-class embeddings separate self/stranger at *kiosk distance and
  lighting* (not the clean benchmark conditions the models are reported on).
- Which `/dev/videoN` is the real kiosk camera and its usable format/fps.

## Downstream (after GO — not part of this spike)

Sub-project 2 will design how the vision signals drive the Director: `OwnerPresent` /
`OwnerAbsent(T)` / `OwnerChanged` events; presence as the primary keep-alive (audio
silence-timeout demoted to a backstop); graceful degradation when the camera glitches
while the user is actively talking; and the relaxed audio identity role (near-field +
light check to reject a bystander leaning in while the owner stands there). The
dormant `SafetyNet`/`Lockout`/verify-before-serve components are **not** wired by the
rejected "rolling eject" policy; their reuse, if any, is decided in Sub-project 2.

Related memory: `pvad-conditioning-inert`, `plan05-crowd-focus-resume`,
`ecapa-short-segment-unreliable`, `kiosk-architecture-decision`.
