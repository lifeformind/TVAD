# Bystander Gate v1 — New-Turn Reject-by-Default (hardware-independent)

**Status:** DESIGN — approved for build (decisions encoded below)
**Date:** 2026-06-25
**Scope:** Director-08, sub-piece of the bystander-rejection sub-project
**Depends on:** Director-07 camera floor control (shipped); the barge-in proximity gate
**Related:** memory `bystander-rejection-subproject`, `c10-dual-mono-no-stereo`,
`ecapa-short-segment-unreliable`

## 1. Problem

In a lively space the kiosk accepts NON-TARGET speech (bystanders / surrounding
conversation, sometimes loud) as new-turn input and answers it — degrading service for
the enrolled owner. Root cause: the Director's **new-turn path has zero speaker gating**.
`reducer.py:_on_user_segment` accepts a turn on `is_target` + endpoint only; `is_target`
defaults `True` (pVAD/crowd-focus disabled, inert). The proximity-RMS gate is applied to
**barge-in only**, never to new turns. So any new-turn segment is transcribed and served.

Product priority (decided): **"never answer a stranger"** — bias hard against
false-accept; reject-by-default.

## 2. Scope

**In scope (v1):** two reject-by-default gates on NEW turns, both from signals already
present in the reducer — **proximity (RMS)** and **camera presence** (Director-07). Pure
reducer change + one config flag + one DIAG line. No new worker, no new event, no hardware.

**Out of scope (named for continuity, NOT built here):**
- Accumulated-window ECAPA `SafetyNet`/`Lockout` (built but unwired; deferred — may be
  superseded by the array's DOA vote).
- The ReSpeaker **DOA-cone vote** — will plug into this exact gate sequence as one more
  reject check (`if bearing not in owner_cone: reject`) when the array lands.

**Known limitation (accepted):** v1 does NOT reject a bystander who is *co-located* with
the owner (same direction AND similar distance/loudness) — they pass proximity and the
owner is PRESENT. That is precisely the case the mic array is being bought to solve. v1
closes the common leaks: distant/quiet bystanders, any turn while the owner is not in
frame, and (via the clock change) bystander chatter holding the kiosk hostage.

## 3. Decision logic

Single change, single place: `_on_user_segment` in the pure reducer. Flag-gated so the
legacy branch is byte-for-byte today's behavior.

```
_on_user_segment(ctx, ev):                       # ev: SegmentEndpointed (has rms, is_target, endpoint_prob)
  if not ctx.cfg.reject_bystanders:              # ---- legacy path: unchanged ----
      ctx.last_speech_at = ctx.now
      if not ev.is_target:               return LISTENING, []
      if ev.endpoint_prob < cfg.endpoint_threshold: return LISTENING, []
      return LISTENING, [TranscribeUserTurn()]

  # ---- reject-by-default path ----
  if not ev.is_target:                   return LISTENING, []   # pVAD bystander (inert today)
  if ev.rms < ctx.proximity_rms:         return LISTENING, []   # too quiet/distant -> reject
  if ctx.presence_status is ABSENT:      return LISTENING, []   # owner not in frame -> reject
  ctx.last_speech_at = ctx.now                                  # ACCEPTED -> reset clocks
  if ev.endpoint_prob < cfg.endpoint_threshold: return LISTENING, []  # owner still talking
  return LISTENING, [TranscribeUserTurn()]
```

Note the reject branches return **without** updating `ctx.last_speech_at` — see §5.

## 4. Fail-safe / no-regression

Mirrors Director-07's fail-safe discipline (uncertainty never blocks the owner):

- **Presence gate fires ONLY on `ABSENT`.** `PRESENT` → allow; `UNAVAILABLE` → allow
  (camera can't judge → never block on it). With vision disabled / no camera,
  `presence_status` stays at its `UNAVAILABLE` default → the camera gate is inert.
- **`reject_bystanders` defaults OFF** → the legacy branch runs → asserted byte-for-byte
  identical to today's reducer. This is the no-regression guarantee (a test asserts the
  flag-off reducer behaves exactly as the pre-Director-08 reducer for every input class).

## 5. Clock semantics (silence / owner-absent)

Decided change: the silence / owner-absent clock (`ctx.last_speech_at`) resets **only on
accepted turns**. Today `_on_user_segment` sets it unconditionally on the first line, so
*any* voiced segment — including a bystander's — resets it, which (a) keeps the session
alive through bystander chatter and (b) blocks the Director-07 owner-absent fast-end
(which requires `now - last_speech_at >= active_talk_guard_s`).

Under v1 (flag ON): a genuinely-rejected bystander turn no longer advances
`last_speech_at`, so bystander chatter no longer holds the kiosk hostage and owner-absent
can fire amid it. Accepted owner turns — including still-accumulating ones
(`endpoint_prob < threshold`) — DO reset, because the reset happens after the gates but
before the endpoint check.

**Trade-off (accepted):** a too-quiet owner turn that fails the proximity gate won't
reset the clock either. In practice an owner who is PRESENT and at the kiosk passes
proximity; this mainly bites genuine bystanders.

This clock change is **inside the `reject_bystanders` branch only** — flag-off behavior
(unconditional reset) is unchanged.

## 6. Config

Add one field, default off:

- `DirectorConfig.reject_bystanders: bool = False` (`config.py`)
- Mapped in `_director_config_from` (`assembly.py:122`):
  `reject_bystanders = tb_cfg.get("turn_gate", {}).get("reject_bystanders", False)`
- New key in `config.yaml` under `kiosk.talkback.turn_gate`: `reject_bystanders: true`
  (the feature ships ON in config, OFF by default in code — same pattern as Director-07's
  `vision.enabled`). A short comment explains the two gates + fail-safe.

**Proximity threshold:** reuse the existing `ctx.proximity_rms` — already calibrated from
the enrollment segment (`_calibrate_proximity_rms`, enrollment-RMS × `barge_in.proximity.
rms_factor`, default 0.5) and already used by barge-in. No new calibration or key. A
separate new-turn factor is a possible later refinement, not in v1.

**Config robustness:** read `reject_bystanders` as a real bool (only boolean `True`
enables), echoing the Director-07 `enabled: flase` fail-open fix — a malformed value is
treated as off.

## 7. Observability (live validation)

The reducer stays pure (no I/O). A rejected new-turn currently shows in DIAG only as
`SegmentEndpointed -> state=LISTENING cmds=[]`, indistinguishable from an
incomplete-but-accepted turn. For live validation, add a `TVAD_DIAG`-gated line in the
**runtime** — the correct place because, unlike the ingestion worker, it holds both the
reducer `ctx` (so it can read `proximity_rms` and `presence_status`) and the event after
`reduce()`. When `reject_bystanders` is on and a `SegmentEndpointed` in LISTENING yields
no `TranscribeUserTurn`, log the gate inputs: `ev.rms` vs `ctx.proximity_rms`, the current
`presence_status`, and the inferred reject reason (`too_quiet` / `owner_absent_frame` /
`not_target` / `incomplete`). This watches the gate decide live without polluting the
pure reducer or the command stream.

## 8. Testing

Pure-reducer unit tests (no I/O, inject `ctx`/`ev`):

**Flag OFF (no-regression):** for non-target, quiet, owner-absent, present, and
incomplete inputs, behavior is identical to the current reducer; `last_speech_at` always
resets. An explicit assertion that flag-off == legacy for every input class.

**Flag ON:**
- quiet turn (`rms < proximity_rms`) → no `TranscribeUserTurn`; `last_speech_at` NOT reset.
- owner `ABSENT` → rejected; `last_speech_at` NOT reset.
- owner `PRESENT` + proximate + endpoint complete → `TranscribeUserTurn`; clock reset.
- owner `UNAVAILABLE` + proximate + endpoint complete → accepted (fail-safe).
- non-target → rejected; clock NOT reset.
- accepted-but-incomplete (`endpoint_prob < threshold`, proximate, present/unavailable)
  → no `TranscribeUserTurn` but clock IS reset.

**Clock-semantics integration:** with the flag on, simulate rejected bystander segments
during LISTENING + a Tick stream and assert the owner-absent end (Director-07) still
fires (rejected chatter no longer blocks it).

**Config mapping:** `turn_gate.reject_bystanders` maps onto `DirectorConfig`; absent key
→ `False`; malformed (non-bool) → `False`.

## 9. Build order (for the plan)

1. `DirectorConfig.reject_bystanders` field + assembly mapping + `config.yaml` key.
2. Reducer `_on_user_segment` flag-gated reject-by-default + clock change, with the
   flag-off no-regression assertion.
3. Runtime DIAG line (reads `ctx` + event) for live observability.
4. Live validation: with `reject_bystanders: true`, confirm a bystander turn is rejected
   (no `UserTurnTranscribed`) while the owner is served normally; confirm owner-absent
   still fires amid chatter; with the flag false, confirm no-regression.
