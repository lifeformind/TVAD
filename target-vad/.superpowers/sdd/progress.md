# Vision Presence Spike — Progress Ledger

Plan: docs/superpowers/plans/2026-06-23-vision-presence-spike.md
Branch: feat/director-05-crowd-focus

Task 1: complete (commits c27979e..a459480, review clean)
  - Minor (deferred to final review): top-level `import numpy as np` unused in Task 1; used from Task 2 on.
Task 2: complete (commits a459480..fbae018, review clean)
  - Minor (live-run note): cmd_presence sampled_fps divides by requested `seconds`, not actual elapsed — read raw counts when interpreting the live run.
Task 3: complete (commits fbae018..b190b77, review clean; separation_report sweep hand-traced)
  - Live-path notes (verify during live run): SFace feature() L2-norm unknown (cosine normalizes both, so OK); time.sleep(0.2) collect overshoot (fine for 8s windows).
Tasks 1-3 (harness build) DONE + reviewed clean. Task 4 = live runs + verdict (BLOCKED on human at kiosk + 2nd person).
Task 5: complete (commits b190b77..68bb4bb, review clean; guided prompts + --preview, 8/8 tests, cv2-free import, graceful headless)
  - Minor (cosmetic, no fix needed): remaining recomputed in preview block; helper defs after cmd_* (fine in Python).
HARNESS COMPLETE (Tasks 1-3,5 reviewed clean). Final whole-branch review deferred until after live runs (Task 4) in case live paths need a fix.

--- Live runs (Task 4), 2026-06-23 ---
Half 1 (camera): PASS — index 0, 640x360 @ ~29fps (index 1 = V4L2 error, ignore).
Half 2 (presence): PASS — min_area_frac tuned 0.03 -> 0.015 (0.03 flickered at the
  user's face size ~0.024-0.028 of a 640x360 frame; 0.015 holds). detected=1 stable,
  state=present held, absent on step-out. detect ~12ms p50. CPU ~59%/core (mostly the
  30fps grab loop, detection ~2%) — acceptable pending contention test; real
  integration should use a dedicated low-rate capture.
  HARNESS EDITS (uncommitted): cmd_presence gained --debug, --min-area-frac (default
  0.015 in BOTH signature + argparse), grab()-idle loop. Do NOT set CAP_PROP_FPS — it
  switches this UVC camera's capture mode and YuNet then detects nothing (regression
  seen + reverted).
Half 3 (identity): GO (2026-06-24, 2nd person). self n=33 min 0.789 mean 0.932 vs
  cross n=33 max 0.057 mean 0.010 — gap 0.73; thr 0.789 -> self_accept 1.0 +
  cross_reject 1.0; recommend operating thr ~0.40. Identity (Tier 2) USABLE, not just
  presence-only. Live gotcha fixed (kept in harness): camera buffers frames during the
  Enter-prompt+countdown -> first reads stale -> at A->B swap embedded the previous
  person (3-4 high cross outliers clustered at phase start). Fix: drain buffer (grab
  x10) + 1.0s settle before collecting; per-frame score print + '*' contamination flag
  added. 8/8 bench tests still pass.
ALL THREE HALVES GO. Verdict written: docs/notes/2026-06-23-vision-presence.md.
Contention: PASS (2026-06-24, presence + full kiosk-stack). sampled_fps 2.9 (held),
  present_fraction 0.93 (unchanged), detect 12->16ms p50 (stack competing; ~18x under
  the 333ms period at 3fps), proc_cpu 58.9% (unchanged). Presence co-exists with the
  LLM/TTS/STT stack.
SPIKE COMPLETE — all 3 halves GO + contention. NEXT: commit GO checkpoint, final
  whole-branch review (opus), finish branch (finishing-a-development-branch).
