# Director Floor Control — Camera Presence + Identity (Design)

**Date:** 2026-06-24
**Status:** Design approved; spec for Sub-project 2 (Director floor-control integration)
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128 GB unified), Python 3.12
**Supersedes:** Section 7 ("Speaker focus in a crowd") of
`docs/superpowers/specs/2026-06-21-director-architecture-design.md` — the acoustic
pVAD crowd-focus mechanism. See **§11 Relationship to the architecture**.
**Builds on:** the vision spike GO verdict `docs/notes/2026-06-23-vision-presence.md`
(all three halves GO + contention PASS) and throwaway harness
`bench/vision_presence_probe.py`.

---

## 1. Why this exists

The kiosk must serve one customer in a lively space: keep serving the person who
started the session, free the kiosk quickly when they leave, and hand it to a new
customer — without the audio-identity gymnastics ECAPA can't do on short, overlapping
speech. The Director's original FOCUS mechanism (architecture doc §7) did this
*acoustically* with the FireRedChat pVAD, conditioned on an ECAPA embedding. **That is
dead**: the pVAD's `spkemb` conditioning is inert with our embeddings — it degenerates
to a plain energy VAD and bystanders leak (memory `pvad-conditioning-inert`; shipped
disabled in `config.yaml`). Brainstorming concluded **presence is a physical fact, not
an acoustic one**, and a camera measures it directly. The vision spike then proved,
live on the GB10, that cheap CPU-only camera presence (YuNet) and enrolled identity
(SFace) both work at kiosk distance — identity with a 0.73 cosine separation margin.

This sub-project wires that validated capability into the Director as the
**floor-control authority**, while audio remains *content*.

## 2. Scope (the five decisions)

This design is the product of five explicit decisions (each a deliberate, conservative
choice over a more aggressive alternative):

1. **Keep-alive = presence as an ADD-ON, not a replacement.** The existing 30 s
   silence timeout and the 25 s "are you still there?" nudge are **unchanged**. We
   *add* one new end-condition: owner physically gone → free the kiosk fast. A
   present-but-silent owner still times out at 30 s as today (the nudge covers them).
2. **Floor authority is IDENTITY-aware (Tier 2), not presence-only.** "Present" means
   the *enrolled owner's face* is in frame. A stranger standing there reads as
   owner-absent → the session frees. This subsumes "owner changed": a swap looks like
   absence → end → the new person wakes normally.
3. **Audio speaker-verification is DEFERRED behind a seam.** Ship proximity-only for
   content (camera owner-present + near-field RMS). The accumulated-window ECAPA check
   that would catch "a bystander speaking right beside the present owner" is built as a
   flag-gated seam (default off), revisited after live data shows whether it's a real
   problem.
4. **Camera degradation FAILS SAFE.** Distinguish "camera confidently sees no owner"
   (valid absence → end after grace) from "camera can't tell" (glitch / dropped frames
   → ignore vision, fall back to the 30 s audio timeout). Plus an active-talk guard:
   never owner-absent-end within a few seconds of the owner speaking.
5. **Dormant scaffolding is REPURPOSED.** `SafetyNet` (accumulated-window ECAPA
   verifier) + `Lockout` (its action arm) become the deferred audio seam from (3), not
   the primary crowd filter. `verify_before_serve` (enrollment quality) is untouched.

## 3. Architecture

**Approach: a separate `VisionWorker` thread emitting presence events onto the existing
async EventBus, consumed by the reducer.** This fits the Director's core principle —
*parallelism in the workers, decision-making serialized; only the reducer mutates
state, one event at a time* (architecture doc §3, §11). Rejected alternatives: folding
camera reads into `IngestionWorker` (couples camera cadence to the 30 ms audio reflex
loop — wrong coupling); a separate process + IPC (isolation we don't need on one box;
noted only as a future escape hatch).

```
WAKE (session start)
  ├─ existing: capture first audio segment + ECAPA voiceprint (holdout, etc.)
  └─ NEW: grab a few camera frames → mean SFace embedding = owner face reference
        → DirectorHandoff.primary_face_embedding   (None if no camera/face)

ASSEMBLY
  └─ _build_vision(primary_face_embedding, vision_cfg)
        ├─ cv2/camera unavailable OR vision.enabled=false OR reference None → None
        │     → runtime identical to today (NO-REGRESSION GUARANTEE)
        └─ else → VisionWorker; runtime starts its capture thread on run, joins on teardown

RUNTIME (vision thread, ~3 fps)
  frame → PresenceClassifier(frame, reference)
            → PRESENT      (owner face in central zone, cosine ≥ identity_threshold)
            → ABSENT       (no face, or a face that is NOT the owner)
            → (raises)     → worker maps to UNAVAILABLE
        → PresenceDebouncer (present_after_s / absent_after_s hysteresis)
        → on DEBOUNCED status change only:
            bus.emit(OwnerPresenceEvent(status, now))     [thread→loop via run_coroutine_threadsafe]

REDUCER (sole mutator)
  _on_owner_presence(ev):  ctx.presence_status = ev.status; ctx.presence_since = ev.now   (NO transition)
  _on_tick(ev):  ...existing hard-cap + silence + nudge checks UNCHANGED...
     + owner-absent end-condition (see §6)
```

## 4. Components & boundaries

Each unit has one purpose, a defined interface, and is testable in isolation.

| Unit | File (new unless noted) | Purpose | Depends on |
|---|---|---|---|
| `VisionWorker` | `modes/director/workers/vision.py` | Owns the capture thread; at low fps pulls a frame, classifies, debounces, emits `OwnerPresenceEvent` on debounced edges; catches all exceptions → `UNAVAILABLE`. | injected frame-source + classifier + bus + clock |
| `PresenceClassifier` | `modes/director/vision/classifier.py` | frame + owner reference → `PRESENT`/`ABSENT` (raises on detector failure). Wraps YuNet detect + zone/size filter + SFace embed + cosine. | cv2 (live), reference embedding, thresholds |
| `PresenceDebouncer` | reuse spike logic | hysteresis over raw per-frame verdicts. | — (pure) |
| `OwnerPresenceEvent` | `modes/director/events.py` (+) | carries `status ∈ {PRESENT, ABSENT, UNAVAILABLE}` + `now`. | — |
| Reducer additions | `modes/director/reducer.py`, `state.py`/context | record presence in ctx; owner-absent end inside `_on_tick`. | — (pure) |
| `VisionConfig` | `modes/director/config.py` (+) | typed floor-control params from `kiosk.talkback.vision`. | — |
| Face enrollment | `modes/director/vision/enroll.py` + WakeGate/assembly seam | capture owner face embedding at wake → `handoff.primary_face_embedding`. | cv2 (live), SFace |
| Vision build | `modes/director/assembly.py` (+`_build_vision`) | build worker or `None`; wire into runtime start/stop. | all above |
| SafetyNet seam | `safety_net.py` + `lockout.py` (existing), wired flag-gated | deferred audio speaker-check (default off). | IngestionWorker audio, embedder |

**Live/pure split (the testability boundary):** all OpenCV/camera glue is confined to
`VisionWorker`'s thin frame-source adapter and `PresenceClassifier`'s live path, both
injected. The *decision* logic — debounce, the reducer transitions, the worker's
edge-emission and exception→UNAVAILABLE mapping — is pure and unit-tested with fakes,
exactly as the FSM reducer is tested today (architecture doc §12).

## 5. Events & state

- **`OwnerPresenceEvent(status, now)`** — the only new event. `status` is an enum
  `PresenceStatus { PRESENT, ABSENT, UNAVAILABLE }`. Emitted by `VisionWorker` **only
  on a debounced status change**, never per-frame.
- **Context additions:** `presence_status: PresenceStatus` (init `UNAVAILABLE` — vision
  says nothing until the worker reports), `presence_since: float` (monotonic time of
  the last status change).
- **`OwnerPresenceEvent` is pure state-recording — it NEVER itself causes a
  transition.** This is deliberate: a single camera blip must never directly kill a
  session. The *decision* always happens on a `Tick`, where the grace + guards apply.

## 6. Reducer logic (precise)

`_on_owner_presence(ctx, ev)`:
```
ctx.presence_status = ev.status
ctx.presence_since  = ev.now
return state, []          # no transition, no command
```

`_on_tick(state, ctx, ev)` — the existing handler, with ONE added end-condition after
the unchanged hard-cap / silence / nudge checks:
```
# ...unchanged: hard_timeout, silence_timeout, nudge...
if (ctx.presence_status == ABSENT
        and ev.now - ctx.presence_since >= cfg.owner_absent_grace_s   # sustained valid absence
        and ev.now - ctx.last_speech_at  >= cfg.active_talk_guard_s): # active-talk guard
    return IDLE, [EndSession("owner_absent")]
return state, []
```

Properties:
- **Owner-absent end requires all three** simultaneously: debounced `ABSENT`, sustained
  ≥ `owner_absent_grace_s`, and no owner speech within `active_talk_guard_s`.
- A **stranger in frame is `ABSENT`** (identity fail) → frees the kiosk on the same
  path (swap detection).
- **`UNAVAILABLE` ⇒ no new end-condition** — the unchanged 30 s silence timeout is the
  sole authority (fail-safe degradation).
- **`PRESENT` ⇒ no effect on the silence timeout** (decision 1, add-on only): a
  present-but-silent owner still times out at 30 s, nudged at 25 s, exactly as today.
- **`EndSession("owner_absent")`** reuses the existing teardown path verbatim — only a
  new reason string, no new teardown code.
- The watchdog remains the **sole timeout authority** (architecture doc §4a, §5): this
  adds a *condition inside* `_on_tick`, not a competing timer.

## 7. Wake-time face enrollment

The classifier needs an owner reference before the session runs. At session start
(parallel to the existing audio voiceprint capture in the WakeGate → handoff seam), a
small helper grabs `enroll_frames` camera frames, embeds the largest central face per
frame (SFace), and stores the **mean** as `DirectorHandoff.primary_face_embedding`.

- If it captures nothing (no camera, no face, cv2 missing) → `primary_face_embedding =
  None` → `_build_vision` returns `None` → **today's behavior, no session blocked on the
  camera.** Starting a session must never hard-depend on the camera.
- Reuses the spike's embed path (`rec.alignCrop` + `rec.feature`) and the
  zone/largest-face selection.

## 8. Config

New `kiosk.talkback.vision` block → `VisionConfig` (and the relevant fields onto
`DirectorConfig` for the reducer):

```yaml
kiosk.talkback.vision:
  enabled: true
  camera_index: 0
  width: 640                # spike-validated capture mode; do NOT set CAP_PROP_FPS
  height: 360
  fps: 3                    # low-rate dedicated capture (NOT 30; spike contention caveat)
  identity_threshold: 0.40  # spike Tier-2 GO: self ≥0.79 vs stranger ≤0.06, margin 0.73
  min_area_frac: 0.015      # spike-tuned zone/size gate at 640×360
  present_after_s: 1.0      # debounce: continuous detection before PRESENT
  absent_after_s: 2.0       # debounce: continuous non-detection before ABSENT
  owner_absent_grace_s: 3.0 # how fast to free the kiosk after the owner leaves
  active_talk_guard_s: 3.0  # never owner-absent-end within this of owner speech
  enroll_frames: 8          # face-reference frames captured at wake
  audio_safety_net:
    enabled: false          # §9 deferred seam (SafetyNet + Lockout), default OFF
```

## 9. Degradation, safety, and the deferred audio seam

**No-regression guarantee (the central safety property).** With `vision.enabled:
false`, no camera, no cv2, or no captured face reference, `_build_vision` returns
`None` and the runtime is byte-for-byte today's Director. This is an asserted test.

**Fail-safe degradation.** `PresenceClassifier` / capture exceptions → the worker emits
`UNAVAILABLE`; repeated `cap.read()` failures → `UNAVAILABLE`; recovery re-emits
`PRESENT`/`ABSENT`. The worker **never** propagates an exception into the session loop
(mirrors `PvadWorker`'s crash-fallback discipline). `UNAVAILABLE` ⇒ vision is ignored,
audio silence timeout governs. The active-talk guard and debounce protect an
actively-talking owner from a transient miss.

**Deferred audio seam (decision 3 + 5).** `SafetyNet.accumulate(audio, is_target)` +
`maybe_verify()` is exactly an accumulated-window ECAPA speaker check, and `Lockout` is
its action arm. Wired flag-gated behind `vision.audio_safety_net.enabled` (default
**off**): when on, `IngestionWorker` feeds endpointed segment audio to `SafetyNet`; a
reject verdict drives a `Lockout` action (e.g. `EndSession` / a duck-and-warn command).
When off, the path is inert. This addresses "a bystander speaking right beside the
present owner," which the camera (face present, not voice) cannot catch — but only
*after* live data shows it's a real problem in the space. `verify_before_serve` and the
enrollment holdout are a separate concern and are not touched here.

## 10. Testing

- **Reducer (pure, fake clock):** owner-absent end fires only after grace; active-talk
  guard suppresses it; `UNAVAILABLE` falls back to silence timeout; `PRESENT` does
  **not** extend silence (decision 1); stranger (`ABSENT`) → end (swap);
  `OwnerPresenceEvent` alone causes no transition.
- **`VisionWorker`** with injected fake frame-source + fake classifier: emits on
  debounced edges only; maps classifier exceptions → `UNAVAILABLE`; joins cleanly on
  stop; never raises into the loop.
- **`PresenceClassifier`** with stub embeddings: owner → PRESENT, stranger → ABSENT,
  zone/size filter, threshold boundary at `identity_threshold`.
- **Assembly no-regression:** vision disabled / no camera / no face reference →
  `_build_vision` is `None` and the assembled runtime equals today's (asserted).
- **SafetyNet seam:** flag-off path inert; flag-on feeds audio and acts on a verdict
  (reuses existing `test_safety_net.py` / `test_lockout.py`).
- **Live validation** reuses `bench/vision_presence_probe.py` numbers; no new live
  harness is built for V1.

## 11. Relationship to the Director architecture doc

This spec is the detailed design referenced by the master architecture doc, which is
revised to match (decision: full master-doc revision). Concretely:

- **Supersedes §7 ("Speaker focus in a crowd") V1.** The acoustic pVAD mechanism, the
  ECAPA-conditioned `is_target` gating, the "rolling-window ECAPA safety net" as
  *primary*, and Plan 05's crowd-focus are retired. Req 1 (FOCUS) is reinterpreted from
  an **acoustic** fact to a **physical** one measured by the camera.
- **Adds to §4 (state machine):** one event input (`OwnerPresenceEvent`) and one
  end-condition inside `_on_tick`. The 5-state FSM is unchanged.
- **Preserves §4a/§5 (sole timeout authority):** a condition, not a competing timer.
- **Repurposes §7's safety-net + lockout** as the deferred audio seam (§9).
- **Leaves untouched** §6 (turn-taking), §8 (resume), §9 (model stack/STT), §10
  (reuse), §11 (concurrency/teardown).
- **§13 build sequence:** step 5 ("Crowd focus / pVAD") is rewritten to "camera floor
  control"; this becomes a new plan (Director-07-vision-floor-control).
- **§14 risks:** removes "pVAD un-benchmarked on GB10 CPU" and "combined hot-path
  latency" from the critical path; **§7 V2** (bespoke trained TS-VAD on 2000h+ data)
  drops from "needed for FOCUS" to optional. Adds a new dependency risk: the camera and
  its mount/lighting at the real kiosk.

Net effect: this **narrows and de-risks** the remaining Director effort — it removes
the biggest unsolved audio problem and replaces it with a spike-validated path, at the
cost of a camera dependency and one new plan.

## 12. Non-goals (YAGNI)

- No two-faces tiebreak / multi-customer arbitration (lock to one owner by design).
- No privacy/retention design for face embeddings (session-scoped, in-memory; a real
  deployment policy is separate).
- No re-identification across sessions (the reference is captured fresh at each wake).
- No production camera-health telemetry/dashboards beyond the `UNAVAILABLE` signal.
- No live tuning of the audio SafetyNet seam (deferred until live data).
- No new live measurement harness (reuse the spike's).

## 13. Risks / open questions

- **Camera placement/lighting at the real kiosk** — the spike was at a desk; mount
  height, backlighting, and kiosk distance can shift the identity margin. Mitigation:
  `identity_threshold` is config; re-measure with `bench/vision_presence_probe.py`
  on-site; the 0.73 spike margin is generous headroom.
- **Two people at the kiosk** (owner + companion leaning in) — `PresenceClassifier`
  picks the largest central face; if a companion is larger/closer, it could read the
  owner as absent. Mitigation: zone/size gate + debounce; revisit if observed.
- **Wake-time enrollment cost** — grabbing `enroll_frames` adds a brief moment at
  session start. Mitigation: small frame count; runs parallel to audio capture; falls
  back to `None` (presence-off) rather than blocking.
- **Bystander-beside-owner content leak** — explicitly accepted for V1 (decision 3);
  the SafetyNet seam is the planned mitigation if live data warrants.

## 14. Downstream (after this ships)

The implementation plan (writing-plans) sequences: events + reducer (pure, tests
first) → config → `PresenceClassifier` → `VisionWorker` → wake-time enrollment →
assembly `_build_vision` + runtime start/stop → SafetyNet seam (flag-gated) →
no-regression assertion. Then the master architecture doc revision lands alongside.

Related: `docs/notes/2026-06-23-vision-presence.md`,
`docs/superpowers/specs/2026-06-23-vision-presence-spike-design.md`,
`docs/superpowers/specs/2026-06-21-director-architecture-design.md`; memories
`pvad-conditioning-inert`, `plan05-crowd-focus-resume`,
`ecapa-short-segment-unreliable`, `kiosk-architecture-decision`.
