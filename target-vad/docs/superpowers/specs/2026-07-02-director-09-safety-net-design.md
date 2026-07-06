# Director-09 — ECAPA Safety Net, Lockout & Verify-Before-Serve

**Date:** 2026-07-02
**Status:** Designed (this doc). Depends on Director-08 (branch `feat/director-08-bystander-gate`);
stacked on that branch until D08 merges.
**Prior art:** `modes/director/safety_net.py`, `lockout.py`, `verify.py` — built and unit-tested
during Plan 05, never wired. This spec wires them (porting where noted) and realizes the
"audio speaker-check seam" reserved by the Director-07 spec (§9 there; `vision.audio_safety_net`).

## 1. Problem

The live Director path has **no speaker verification at all**. `classify_new_turn`
(Director-08) gates new turns on `is_target` (always True — pVAD disabled) + proximity RMS +
camera presence. A bystander who is **close and loud while the owner's camera presence holds**
(or with the camera UNAVAILABLE) is served indefinitely. Session-hijack — a second person
taking over an owner's session — is undetectable. Additionally, `verify_before_serve` is a
no-op: `wakegate.py` passes the primary embedding *as* the holdout, so the check trivially
passes at cosine == 1.0, and a garbage first-segment enrollment (cough, noise, TTS bleed)
becomes the session voiceprint unchallenged.

Product priority (standing): **"never answer a stranger"** — bias against false-accept, but
never permanently eject the real user (ECAPA is unreliable on short segments; see memory
`ecapa-short-segment-unreliable`).

## 2. Goals / non-goals

**Goals**
1. Wire the accumulated-window ECAPA verifier (`SafetyNet`) as the session-hijack detector,
   fed ONLY by audio the Director actually served.
2. Pure-reducer WARN→EJECT ladder (port of `lockout.py`), with EJECT = silent session end +
   WakeGate quiet-hold.
3. Real verify-before-serve: split-half self-similarity at session start + window-1
   semantics (first verdict window failing == bad enrollment, not hijack).
4. Full config truth pass: dead keys deleted, silently-defaulted keys added and mapped,
   `turn_gate.*`/`lockout*`/`verify_before_serve_threshold` become live.

**Non-goals**
- No change to Director-08 gate order or semantics (proximity/presence gates untouched).
- No pVAD revival, no per-turn (single-segment) ECAPA gating — measured dead end.
- No spoken warnings or spoken ejections (never answer a stranger; silence toward suspects).
- Spatial/DOA voting — blocked on ReSpeaker hardware; slots in later as another
  `classify_new_turn` branch, orthogonal to this work.
- Retiring `modes/talkback/controller.py` — separate task; this spec only deletes config
  keys, not the legacy engine.

## 3. Architecture (approach: worker computes, reducer decides)

```
ingestion stages segment audio (every LISTENING segment / EVALUATING interjection)
        │
        ▼
reducer verdict on the segment (existing D08 classify_new_turn / interjection ladder)
        │  ACCEPT or ACCUMULATE (turns), gate-passing interjections
        │  → emits AccumulateSpeakerAudio (new command)
        ▼
SafetyNetWorker: pending → rolling buffer; buffer ≥ verify_window_ms (2000ms)?
        │  → embed window in executor (ECAPA ~108ms p95, off the hot path)
        │  → DecisionSmoother.update(score)  (M-of-N: min_matches=1, window_size=3)
        │  → emits SpeakerWindowVerdict (new event)
        ▼
reducer (pure): windows_seen / miss_streak / rms_ok ladder
        → no-op | WARN (log-only) | EndSession("enroll_verify_failed" | "speaker_mismatch")
```

Decision authority stays entirely in the reducer (the Director constitution). The worker
does audio buffering + embedding only. `lockout.py`'s decision half is **ported into the
reducer** and the module deleted; its idle-quiet half is **ported into the WakeGate** (§6).

### 3.1 New event and command (frozen dataclasses)

```python
@dataclass(frozen=True)
class AccumulateSpeakerAudio:       # commands.py — reducer → SafetyNetWorker
    pass                            # audio travels via worker staging, same as STT pending

@dataclass(frozen=True)
class SpeakerWindowVerdict:         # events.py — SafetyNetWorker → reducer
    score: float                    # cosine(window embedding, primary)
    smoother_ok: bool               # M-of-N smoother output for this window
    window_rms: float               # RMS over the window's audio (for the eject rms check)
```

No clock field: the verdict ladder is not time-based (the quiet-hold clock lives in the
WakeGate, which owns time after session end).

### 3.2 Staging (mirrors the STT pending-audio pattern)

`IngestionWorker` already stages every LISTENING segment into `SttWorker.set_pending_user_audio`
and every EVALUATING segment into `set_pending_interjection_audio`. It additionally stages the
same audio into `SafetyNetWorker.set_pending_audio(audio)` (overwrite-last, same discipline).
When the reducer's verdict is plausibly-owner, it emits `AccumulateSpeakerAudio`; the runtime
routes it to the `SafetyNetWorker`, which moves pending → rolling buffer. Segments the
reducer REJECTS are never accumulated — **Director-08-rejected bystander chatter cannot
pollute the hijack buffer and eject the owner.**

Accumulation triggers (exactly these reducer paths):
- `_on_user_segment`: verdict ACCEPT **or** ACCUMULATE (both passed is_target + proximity +
  presence; ACCUMULATE is the same speech mid-turn). Emitted alongside/instead of
  `TranscribeUserTurn` as `[AccumulateSpeakerAudio()] (+ [TranscribeUserTurn()] on ACCEPT)`.
- `_on_interjection_segment`: the gate-passing branch only (emitted with
  `TranscribeInterjection`). Rejected interjections RESTORE and are not accumulated.
- With `reject_bystanders: false` (legacy mode), ACCEPT/non-target semantics follow the
  legacy verdicts unchanged; accumulation follows the same ACCEPT/ACCUMULATE rule.
- **Seed exclusion:** the assembly factory's seeded first segment (the enrollment
  utterance, announced as the opening `SegmentEndpointed`) is NOT staged into the safety
  net. If it were, window 1 would largely be the very audio the primary was derived from
  and would trivially pass, gutting the window-1 verify. The reducer still emits
  `AccumulateSpeakerAudio` for the seed (it is pure and cannot know); the worker's empty
  pending buffer makes it a no-op.

### 3.3 SafetyNetWorker

New `modes/director/workers/safety_net.py`, wrapping the existing `SafetyNet` class
(unchanged API: `accumulate` / `maybe_verify`):
- `set_pending_audio(np.ndarray)` — called by ingestion (thread: the ingestion loop).
- On `AccumulateSpeakerAudio`: `safety_net.accumulate(pending, is_target=True)` (the reducer
  already made the is_target call; the worker does not second-guess it), then if a window is
  ready, run `maybe_verify()` in `run_in_executor` and `bus.emit(SpeakerWindowVerdict(...))`.
- `window_rms` computed over the exact window audio consumed (SafetyNet returns it alongside
  the verdict — small extension to `SafetyVerdict`: add `window_rms: float`).
- Multiple windows can complete from one long turn: loop `maybe_verify()` until None,
  emitting one verdict event per window, in order.
- Built in `assembly.py` only when `turn_gate.require_speaker_match is True` (strict-bool,
  the `flase` lesson). Absent worker == no events == reducer ladder inert == byte-for-byte
  Director-08 behavior.

## 4. Reducer ladder (pure; new ctx fields `windows_seen: int = 0`, `miss_streak: int = 0`)

On `SpeakerWindowVerdict` (any state except after session end — the runtime stops
dispatching after `EndSession`, which already covers late-arriving verdicts):

1. `windows_seen += 1`.
2. `smoother_ok` → `miss_streak = 0`; return `state, []` (no-op).
3. Not ok and `windows_seen == 1` → **bad enrollment, not hijack**:
   `IDLE, [EndSession("enroll_verify_failed")]`. Silent. No WakeGate hold — it is
   probably the real user with a garbage voiceprint; let them re-wake immediately.
4. Not ok, later window → `miss_streak += 1`; `rms_ok = ev.window_rms >= ctx.proximity_rms`.
   - `miss_streak >= 2 and not rms_ok` → **EJECT**:
     `IDLE, [EndSession("speaker_mismatch")]`. Silent — never answer a stranger.
   - else → **WARN**: `state, []`. Log-only (runtime DIAG + event log, §7). No behavior
     change; a later passing window resets the streak.
5. **Shadow mode:** if `cfg.lockout_enabled is False`, steps 3-4 log their would-be action
   (via the DIAG helper, §7) but always return `state, []`. Verdicts + WARN visibility with
   zero eject authority — the graduated-rollout knob.

Timing at defaults (2s windows, 1-of-3 smoother): mid-session `smoother_ok` first goes false
after **3 consecutive** below-threshold windows (~6s of served non-matching speech) = WARN;
EJECT needs a **second** consecutive smoother-fail window (~8s) **and** the window quieter
than the proximity floor. Window 1 is the exception by construction: a single below-threshold
score in a deque of one fails the smoother — which is exactly the verify-before-serve
semantic (2s of real served speech disagreeing with the enrollment ends it immediately).

`DirectorConfig` addition: `lockout_enabled: bool = False` (strict-bool mapped).
New `DirectorResult` reasons: `enroll_verify_failed`, `speaker_mismatch`.

## 5. Verify-before-serve, made real (WakeGate, pre-session)

In `_start_session_from_segment`, before staging the handoff:
1. Split the first segment's audio in half; embed both halves (2 extra ECAPA extracts,
   ~200ms, pre-serve so off every hot path). **Only when the segment is ≥ 1.0s** (halves
   ≥ 0.5s): the VAD floor is 300ms and 150ms halves are too short for an honest 0.80
   comparison (would false-refuse real users). Shorter first segments skip the split-half
   check and rely on window 1. An embedder exception on a half is treated like the
   existing first-embed failure: reset to IDLE, no session.
2. `ok, score = verify_before_serve(emb_h1, emb_h2, threshold)` with
   `threshold = kiosk.talkback.verify_before_serve_threshold` (0.80). Same-utterance halves
   of one speaker are highly self-similar; noise/garbage/degenerate embeddings are not —
   this is the statistically-honest use of 0.80 (cross-utterance short-segment scores are
   too noisy per memory `ecapa-short-segment-unreliable`; that job belongs to window 1, §4.3).
3. Fail → no session: `_reset_to_idle()`, emit the `verify_refused` callback event with the
   score (observability), do NOT hand off. The user simply wakes again.
4. Pass → handoff as today, with the **placeholder holdout removed**: `DirectorHandoff`
   drops `holdout_embedding` (its only consumer was the placeholder comment).
   `verify.py`'s docstring is updated to the split-half role; the function is unchanged.

The primary embedding remains the full-segment embedding (unchanged).

## 6. Post-eject quiet-hold (WakeGate)

Ported from `lockout.py`'s idle half. When a session ends with reason `speaker_mismatch`
(the WakeGate's `run()` sees `DirectorResult.reason` — it already emits `session_ended`
from it), the WakeGate enters a **HOLD** sub-state of IDLE:
- While holding, wake detections are ignored (chunk still feeds the RMS check).
- Any chunk with RMS ≥ the enrollment-derived proximity floor resets the quiet clock
  (someone is still talking near the kiosk — likely the hijacker).
- After `lockout_idle_after_s` (5s) of continuous sub-floor RMS, HOLD clears; the next wake
  is accepted normally. Never a permanent lockout.
- `enroll_verify_failed` and every other end reason do NOT hold.

Note: the WakeGate's proximity floor comes from the ended session's calibrated
`proximity_rms`, passed back alongside the result (small plumbing addition); fallback when
absent is no hold (fail-open here is acceptable: hold is a nuisance-reduction, not a gate).

`lockout.py` is deleted once both halves are ported; `tests/director/test_lockout.py`
assertions migrate into the reducer-ladder and wakegate-hold test files.

## 7. Observability

Runtime DIAG (pattern of `gate_diag_reason` — a pure helper the runtime calls, so the
reducer stays print-free):
- Per verdict: `safety-net window=<n> score=<s> smoother_ok=<b> streak=<m> rms=<r>`
- On WARN: `safety-net WARN streak=<m>` (plus `shadow` marker when `lockout_enabled` false)
- On eject/refuse: `safety-net EJECT reason=speaker_mismatch` / `reason=enroll_verify_failed`
- WakeGate: `verify_refused score=<s>` callback event; `wake ignored (hold, quiet=<t>s)` DIAG.

New end reasons flow through the existing `session_ended` event/JSONL path unchanged.

## 8. Config truth pass

### Goes live (currently dead)
| Key | Live meaning |
|---|---|
| `turn_gate.require_speaker_match` | Master enable for the SafetyNet pipeline (worker built at all). Strict-bool; default off in code; **shipped `true`**. |
| `turn_gate.speaker_threshold: 0.30` | SafetyNet cosine threshold (tuned for ≥2s windows). |
| `turn_gate.verify_window_ms: 2000` | SafetyNet window length. |
| `turn_gate.lockout.enabled` | EJECT authority (`false` = shadow mode, §4.5). Strict-bool; **shipped `true`**. |
| `turn_gate.lockout.window_size: 3`, `min_matches: 1` | DecisionSmoother params. |
| `talkback.verify_before_serve_threshold: 0.80` | WakeGate split-half threshold (§5). |
| `talkback.lockout_idle_after_s: 5` | WakeGate post-eject quiet-hold (§6). |

### Added + mapped (code reads them today; config could not set them)
`talkback.nudge_lead_s: 5.0`, `talkback.barge_in.conf_floor: 0.5`,
`talkback.turn_gate.endpoint_threshold: 0.5` — all three mapped in
`assembly._director_config_from`; `talkback.watchdog.tick_ms: 500` — read by `kiosk.py`.
Comments state what each does and the default it previously took silently.

### Deleted (dead: controller-only or reserved-never-read)
`kiosk.decision_smoother.*`; `talkback.aec.suppression_level`; `talkback.stt.partials_every_ms`;
`talkback.barge_in.require_speaker_match`; `talkback.vision.audio_safety_net.*`
(this spec realizes that seam); `talkback.resume.enabled`;
`talkback.logging.include_partial_transcripts`.
Not deleted: the `talkback.turn_gate` block itself — its keys go live (§8 first table) and its
accumulated-window comment finally becomes true; the comment is updated, not removed.
`modes/director/config.py` stale line-number citations fixed in passing.
The legacy controller reads some deleted keys via `.get(...)` defaults only; its tests build
their own config dicts — deleting the YAML keys breaks nothing (verified in the plan's test
gate).

## 9. Edge cases

- **Verdict after end:** runtime stops dispatching post-`EndSession`; late executor
  completions are dropped with the bus. No reducer guard needed beyond existing behavior.
- **Session shorter than one window:** no verdict ever fires; proximity + presence gates
  still apply. Accepted residual risk (a <2s-total hijack was already covered by D08 gates).
- **Long single turn:** may complete >1 window; worker emits each in order (§3.3).
- **Owner voice drift mid-session (illness, distance):** WARN-only until the eject ladder's
  rms check also fails; a passing window resets. Tuning knob: `turn_gate.speaker_threshold`.
- **Camera interplay:** none — this pipeline is camera-independent by design (it is the
  fallback when presence can't discriminate); no presence reads in the ladder.
- **`reject_bystanders: false` + safety net on:** valid combo; accumulation follows legacy
  accept semantics (§3.2). Each flag degrades independently.
- **Both new flags strict-bool** (`is True`), warn on non-bool — the `flase` lesson.

## 10. Testing & merge gate

TDD throughout (project standard). New/changed test surface:
- Reducer ladder: window-1 fail → `enroll_verify_failed`; 3-miss WARN; WARN+rms-fail →
  eject at streak 2; passing window resets streak; shadow mode never ends; verdicts ignored
  in legacy-flag-off assembly (no worker → no events); clock untouched by verdicts.
- SafetyNetWorker: accumulate only on command; pending overwrite discipline; multi-window
  drain; executor embed off-loop; `SafetyVerdict.window_rms` extension.
- WakeGate: split-half pass/fail (fail → no handoff + `verify_refused`); hold engages only
  on `speaker_mismatch`; RMS activity resets hold; hold expiry accepts wake; no hold on
  `enroll_verify_failed`.
- Assembly: strict-bool for both flags; worker absent when off; new key mappings
  (`nudge_lead_s`, `conf_floor`, `endpoint_threshold`); deleted keys absent → no KeyError.
- Full suite green (baseline 648 pass / 2 skip + additions).

**Merge gate — live validation at the kiosk (all with `TVAD_DIAG=1`):**
1. Owner-normal session: zero WARN lines across a multi-turn conversation.
2. Hijack simulation: second speaker takes over mid-session → WARN lines then
   `EJECT reason=speaker_mismatch`; wake refused until ~5s of quiet; owner re-wakes fine.
3. Garbage enrollment: cough/noise as the first segment → `verify_refused` (or window-1
   `enroll_verify_failed` if it slips through), immediate re-wake works.
4. Shadow-mode spot check (`lockout.enabled: false`): hijack shows WARN/would-eject DIAG
   but session continues.
Then verdict note `docs/notes/<date>-director-09-live.md` and finishing-a-development-branch.

## 11. Known limitations (accepted)

- A hijacker **matching the owner's voice** (or ECAPA-confusable) is not caught — that is
  the ReSpeaker DOA vote's job (different bearing), which composes with this ladder later.
- A hijacker who stays **louder than the proximity floor** is WARNed but never ejected
  (rms check fails the eject condition by design — proximity says "someone is at the
  kiosk"). The D08 gates + camera absent-end remain the backstop; revisit with live data.
- Split-half verify does not catch "bystander spoke first" (both halves are the bystander) —
  window-1 + camera identity carry that case.
