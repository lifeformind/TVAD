# Director-08 + Director-09 Bystander Gate & ECAPA Safety Net — Live Validation Verdict (GB10)

**Date:** 2026-07-03 (Check 1) and 2026-07-06 (Checks 2–5), run live on the kiosk
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64); mic = ReSpeaker 4 Mic Array v2.0
(ch0 via PipeWire) from Check 2 onward (Check 1 ran pre-array), camera = Nuroum C10
**Sub-projects:** Director-08 (reject-by-default new-turn gate) + Director-09
(accumulated-window ECAPA safety net) — validated together; D09 is stacked on D08
**Branches:** `feat/director-08-bystander-gate` → `feat/director-09-safety-net`
**Specs:** `docs/superpowers/specs/2026-06-25-bystander-gate-v1-design.md`,
`docs/superpowers/specs/2026-07-02-director-09-safety-net-design.md`
**Method:** live `TVAD_DIAG=1 ./kiosk-stack.sh start` runs, reading the DIAG event
log; solo testing with a podcast on a phone speaker as the stand-in bystander/hijacker,
plus one unplanned real human visitor (the best data of the week).

## VERDICT: **GO — ready to merge (both branches)**

All five merge-gate checks closed. Two live-found bugs: one fixed on the branch
(teardown crash), one documented as a known issue with a structural fix planned for
the array workstream (TTS bleed). One threshold live-tuned (0.30 → 0.15) after the
offline-derived value produced two false ejects of the owner. The safety net's
discriminator is emphatically validated: across every session, intruder-dominant
2s windows scored **0.019–0.080** against the owner's **0.230–0.492** band.

## The five checks

| # | Scenario | Expected | Result |
|---|----------|----------|--------|
| 1 | Owner-normal session | zero WARN, barge-in + resume fine | **PASS** (2026-07-03) — windows 0.399/0.308/0.235 all green, clean `silence_timeout` end |
| 2 | Bystander (distant podcast; owner steps out of frame) | `too_quiet` rejects; chatter can't hold session | **PASS** — `REJECT=too_quiet` ×2, `presence=PRESENT` live, ended `owner_absent` mid-podcast |
| 3 | Hijack (podcast at the mic) | WARN streak; eject only if also quiet | **CLOSED with rationale** — detection proven (all intruder-dominant windows < 0.10); live WARN blocked by mixed-window scores; smoother flip later seen live in Check 5 |
| 4 | Garbage enrollment (cough at wake) | refused or window-1 eject; re-wake works | **PASS** — cough never enrolls (`No speech after wake` → clean re-wake); window-1 eject path seen live twice |
| 5 | Shadow mode (`lockout.enabled: false`) | `would_end` DIAG, session survives | **PASS** — `WARN (shadow) would_end=enroll_verify_failed` on window 1 (score 0.004), streaks 1→4, session kept serving |

## What the live data established

### The ECAPA discriminator works on real strangers
A second person entered the frame unplanned (2026-07-06) and was served (see Known
Limitations); their 2s window scored **0.069** against the owner's voiceprint. The
podcast scored 0.019–0.085 whenever it dominated a window. The owner's live band
across ~30 windows and both mics: 0.230–0.492. Clean separation — *when windows are
pure*.

### Mixed windows are the fundamental limit (threshold tuning can't fix it)
When the kiosk's own TTS answers a hijacker, its bleed lands in every accumulated 2s
window, and mixed windows score on a continuum (observed 0.133–0.207) that straddles
any threshold placed between the pure-intruder and owner bands. This is why the live
`WARN` streak required shadow mode to reproduce (window-1 eject otherwise cut the
session first, and TTS mixing otherwise broke consecutive misses). **Extends spec
§11:** a sustained hijack *with the kiosk talking back* evades the net; a hijacker
monopolizing a quiet mic flips the smoother in ~3 windows (~6s). The structural fix
is the per-turn DOA cone vote (Director-10) — direction is orthogonal to mixing.

### Threshold live-tuned: `turn_gate.speaker_threshold` 0.30 → 0.15
The offline-derived 0.30 sat *inside* the owner's live band and produced two
window-1 false ejects of the real owner (`enroll_verify_failed` — window 1 has no
smoother by design). 0.15 is the midpoint of the live gap (owner min 0.230, intruder
max 0.085); after tuning, the owner ran multi-turn story sessions with barge-ins and
zero false WARN/eject.

### D08 gate: validated on both mics, with a caveat on the array
`REJECT=too_quiet` fired correctly in every session (podcast at conversational
distance never transcribed); camera presence (`presence=PRESENT`, `owner_absent`
end) worked once the camera was reconnected. Caveat: on the far-field array the
proximity floor is compressed — rejects passed by margins as thin as 0.0002, and
the owner speaking over a podcast was too_quiet-rejected. Floor semantics need
recalibration (AGC off, rms_factor revisit) as part of the array migration.
`owner_absent_frame` REJECT was not exercised live (Director-07's absent-timeout
ended the session before an out-of-frame new turn arrived) — outcome correct,
that specific reject path remains unit-test-only.

## Bugs found live

### Bug B (FIXED on branch): teardown crash on mid-playback session end
`owner_absent` fired during active TTS → `_teardown` cancels the generation task →
task cancellation cancels the `_play_future` asyncio wrapper it was awaiting →
`drain()`'s `asyncio.shield(fut)` raised `CancelledError` — a `BaseException` that
escaped `except Exception` — and the kiosk process died instead of returning to the
wake loop. Pre-dates D09 (Director-07-era code); first live trigger. Fixed in
`playback.py drain()` (catch `(asyncio.CancelledError, Exception)`, matching the
`_teardown` idiom); stream safety unaffected (close()'s gen-bump + write lock).
Regression test: `test_drain_survives_cancelled_play_future`.

### Bug A (KNOWN ISSUE, deferred to Director-10): TTS bleed passes the interjection gate
AEC leakage of the kiosk's own reply was captured as a near-field interjection,
transcribed, and **served** — the kiosk answered its own voice ("How can I help you
today?"), and in one session the bleed formed safety window 1 and falsely ejected
the owner. Root cause is upstream of D09 (AEC/barge floor, known re-measure item);
consequence became sharper with D09. Structural fix planned with the array: route
TTS playback through the ReSpeaker so its onboard AEC cancels it against the ch5
reference, plus DOA-at-speaker-bearing filtering. Interim mitigation: threshold 0.15
keeps bleed windows (0.213–0.230 observed) from ejecting the owner.

## Known limitations (accepted, on record)

1. **Co-located bystander is served** (D08 v1 limitation, now witnessed live): a
   second person in frame with the owner, above the proximity floor, gets answered.
   The visitor session proved D09 *detects* them (0.069) but the de-risked config
   (1-of-3 smoother, quiet-eject-only) acts slowly and never ejected. This is
   Director-10's job (DOA cone).
2. **Loud hijacker is WARN-only** (spec §11, de-risked by design) — plus the mixing
   extension above.
3. **Session-relative proximity floor** scales with the enrollment seed's loudness;
   on the far-field array this makes both false-accepts and owner-rejects possible
   at the margins. Array migration work.

## Validation-driven config changes (this branch)

- `turn_gate.speaker_threshold: 0.15` (live-tuned; rationale in comment)
- `core.audio.device_index: "pipewire"` (explicit capture path; PortAudio cannot
  open the array directly, and `null` let PipeWire device elections silently swap
  the mic mid-validation)
- `lockout.enabled: true` (shipped on; Check 5 verified shadow mode works)

## Status

Director-08 and Director-09 are validated and ready to merge to master (local),
D08 first (D09 is stacked on it). Full suite **689 pass / 2 skip** including the
Bug B regression test. Fast-follows (ledger): seq-echo staging hardening,
wakegate sr→config, `get_running_loop`, conftest dedupe. Next feature: Director-10
DOA cone vote — array hardware GO'd same day (raw capsules real, DOAANGLE tracks,
owner/bystander discrimination clean; see memory + `bench/respeaker_doa.py`).
