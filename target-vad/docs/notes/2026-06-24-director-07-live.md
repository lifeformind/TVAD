# Director-07 Camera Floor Control — Live Validation Verdict (GB10)

**Date:** 2026-06-25 (checks run live on the kiosk)
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128 GB unified)
**Sub-project:** 2 of camera-driven floor control (the Director integration)
**Branch:** `feat/director-07-vision-floor-control`
**Spec:** `docs/superpowers/specs/2026-06-24-director-floor-control-design.md`
**Plan:** `docs/superpowers/plans/2026-06-24-director-07-vision-floor-control.md`
**Method:** live `TVAD_DIAG=1 ./kiosk-stack.sh start` runs, reading the DIAG event log.

## VERDICT: **GO — ready to merge**

All three live merge-gate checks pass. The camera adds owner-absent fast-end as a
pure ADD-ON: the silence/nudge/hard timeouts are unchanged, the watchdog stays the
sole timeout authority, and with vision off the runtime is byte-for-byte today's
Director. Fail-safe holds — an unavailable camera never falsely ends a session.

## The three checks

| # | Scenario | Config | Expected | Result |
|---|----------|--------|----------|--------|
| 1 | Owner steps away mid-session | vision on, camera on | end `owner_absent` fast | **PASS** — ended `reason=owner_absent`, freed ~5s after step-out (absent_after 2s + grace 3s); PRESENT emitted while serving, ABSENT on step-out |
| 2 | No-regression | vision **off** | no worker, no presence events, normal end | **PASS** — zero `OwnerPresenceEvent` lines, zero camera-open lines, ended `silence_timeout` |
| 3 | Fail-safe (camera unplugged) | vision on, **no camera** | UNAVAILABLE, never `owner_absent` | **PASS** — V4L2/FFMPEG "camera index out of range", enroll failed, exactly one `OwnerPresenceEvent` (UNAVAILABLE), full normal session, ended `silence_timeout` |

### Check 1 — owner-absent fires (the feature)
Session served the owner (PRESENT → THINKING/SPEAKING), then on step-out the worker
debounced to ABSENT and, after the 3s grace with no recent owner speech, the reducer
returned `EndSession("owner_absent")` from `_on_tick`. The session ended on camera
evidence, not the 30s audio silence timeout.

### Check 2 — no-regression (vision off)
With `vision.enabled: false`, `_build_vision` returns `None`, no `VisionWorker` is
constructed, the camera is never opened, and the event stream contains **no**
`OwnerPresenceEvent`. The reducer's presence branch stays inert (default
`UNAVAILABLE`), so the session runs and ends exactly as the pre-Director-07 kiosk did.
This matches the asserted unit test (`vision=None` → identical Director); the live run
is the belt-and-suspenders confirmation.

### Check 3 — fail-safe (camera unplugged)
With vision enabled but no camera present, the OpenCV backend's `open()` returned
False without raising; enrollment failed; the worker emitted a single `UNAVAILABLE`
(never `ABSENT`). The reducer's owner-absent end requires `ABSENT`, so it never fired,
and the session fell back to the normal 30s audio silence timeout. Degradation is
fail-safe: uncertainty extends, it never ends.

## Gotcha found during validation (config robustness)

The no-regression run first appeared to fail: `OwnerPresenceEvent` kept showing up
even after setting vision off. Root cause was a typo — `enabled: flase` in
`config.yaml`. YAML parses `flase` as a non-empty **string**, which is **truthy**, so
vision stayed enabled. Fixed to `enabled: false`, after which Check 2 was clean.

**Follow-up (post-merge, NOT blocking):** `_build_vision` reads `vision.enabled` as a
bare truthy value, so a malformed config silently **enables** a feature meant to be
off — fail-*open*. Harden it to treat only a real boolean `true` as enabled (any
string / malformed value → treated as off, optionally with a warning). Small change,
own TDD cycle.

## Known limitation (carried from the plan, accepted)

After a camera UNAVAILABLE → recovery **into an already-empty scene**, the monitor
emits no `ABSENT` (only the next `PRESENT`), so the reducer stays `UNAVAILABLE` and
falls back to the 30s silence timeout instead of fast-freeing. Still fail-safe (the
session ends), matches the plan's recovery test by design. Revisit only if live data
shows the glitch-into-absent case matters in practice.

## Status

Director-07 is validated and ready to merge to master (local). The whole-branch opus
review (`8668c9e..18414d3`) returned **Ready to merge: Yes** with no Critical/Important
findings; full suite 629 pass. This note closes Task 10 of the plan.
