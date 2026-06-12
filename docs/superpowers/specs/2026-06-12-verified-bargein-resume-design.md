# Verified barge-in + interruption-resume (talkback)

**Date:** 2026-06-12
**Status:** approved, implementing
**Mode:** S2 kiosk talkback (`modes/talkback`)

## Problem

The talkback assistant is not truly full-duplex:

1. **The registered user cannot interrupt the AI.** Across all live sessions,
   barge-in succeeded **0 / 159** times; rejected scores during TTS playback
   cluster at ~0 (median 0.062, max 0.462) and never reach the 0.75 threshold.
2. **Root cause:** AEC is a no-op. `Player.get_next_frame()` is never called, so
   the AEC reference ring buffer is never written; `get_reference_frame()`
   returns `None` and `AecProcessor.process_frame` never runs. Playback goes
   straight through `sd.play(blocking=True)`, so the mic hears full TTS echo.
   The ECAPA embedding of a barge-in is therefore the Kokoro TTS voice, not the
   user → cosine-to-primary collapses to ~0 for everyone.
3. **No resume.** On barge-in the controller cancels the response task and
   discards the partial answer; nothing remembers the interrupted topic or
   offers to continue it.

## Goals

- The **registered user** (session primary) can interrupt mid-reply; **all
  other speakers and crowd noise are ignored** (kiosk may be in a crowd).
- After answering an interruption, the AI offers to resume the prior topic and,
  on confirmation, continues from where it left off — as in:
  > U: how do I get to the conference room? · AI: go from the cafe toward the
  > blue door and… · U (barge-in): which side is the cafe? · AI: the cafe's to
  > the right, away from the main door. Would you still like directions to the
  > conference room? · U: yes · AI: as I was saying, head toward the blue door…

## Decisions (from brainstorming)

- **Barge-in policy:** verify-then-cut (crowd-safe). Never yield to an
  unverified speaker.
- **Capture mechanism:** proximity pre-gate + duck. Ignore speech too quiet to
  be someone at the kiosk; for near-field speech, duck TTS to capture clean
  audio, verify, then cut (user) or restore (bystander/noise).
- **Resume mechanism:** hybrid — the controller owns the resume lifecycle/state;
  the LLM phrases the interjection answer, the resume offer, and the
  continuation via injected steering instructions.

## Design

### 1. Streaming, gain-controlled player (foundation)

Replace per-utterance `sd.play(blocking)` with a persistent `sd.OutputStream`
driven by a playback coroutine:

- TTS utterance audio is chunked into fixed frames and `enqueue`d.
- A playback task pulls each frame (`get_next_frame()`), multiplies by the
  current **gain**, records the *gained* frame as the AEC reference
  (`record_reference`), and writes it to the output stream.
- **Ducking:** `gain` drops to `duck_level` (~0.15) on near-field onset and
  ramps back to 1.0 on restore (`duck_ramp_ms`).
- **Cut:** `flush()` drops queued frames and the in-flight utterance.

Recording the gained frame makes the AEC reference equal what was actually
played, so AEC cancels the ducked echo and the mic residual is the user's voice.

`Player` changes: `get_next_frame()` no longer auto-records; add public
`record_reference(frame)`. Add `is_speaking` property to `SileroVAD`.

### 2. Barge-in flow during SPEAKING

Onset (chunk-level, in the listen loop) + verify (endpoint, segment handler):

1. **Voiced onset** over TTS → AEC-cleaned chunk **RMS** below
   `proximity.rms_threshold` (auto = primary segment RMS × `rms_factor`, when
   `rms_threshold` is null) → ignore, keep talking (`barge_in_ignored_far`).
   Above → **duck** (`barge_in_ducked`), capture.
2. At VAD **endpoint**, verify the interjection vs the primary. Shorter than
   `verify_window_ms` → restore + ignore (no cut on un-verifiable audio).
3. Score ≥ `speaker_threshold` → **cut**: flush, store interrupted state,
   handle interjection (`barge_in`).
4. Score < threshold → **restore** gain to 1.0, resume the utterance
   (`barge_in_rejected`).

**Threshold is measurement-gated.** Existing scores are AEC-off garbage; ship
provisional `0.30` and re-measure during ducked playback with
`bench/speaker_scores.py --source barge_in` before finalizing (checkpoint
between Phase 2 and Phase 3).

### 3. Interruption-resume (hybrid)

On a cut, store `interrupted = {query, partial}` (the question being answered +
assistant text spoken so far) and mark that partial assistant turn `[interrupted]`
in history. Then:

- **Answer interjection** with a one-shot steering note appended to that
  generation: *"You were interrupted while answering '{query}' (you'd said:
  '{partial}'). Briefly answer the new question, then ask if they'd like you to
  continue with {topic}."* → state `RESUME_PENDING`.
- **Next turn**: inject *"The user was asked whether to continue {topic};
  interpret their reply and either resume from '{partial}' (e.g. 'As I was
  saying…') or move on."* The LLM does the yes/no interpretation and the
  continuation; the controller just holds the `RESUME_PENDING` flag + stored
  fields and clears them after.

### 4. Config / events / testing

**Config (`kiosk.talkback`):**
```yaml
barge_in:
  enabled: true
  require_speaker_match: true
  speaker_threshold: 0.30      # provisional; re-measure during ducked playback
  verify_window_ms: 1200
  duck_level: 0.15
  duck_ramp_ms: 120
  proximity:
    enabled: true
    rms_threshold: null        # null = auto-calibrate from primary segment RMS
    rms_factor: 0.5
resume:
  enabled: true
```

**Events:** `barge_in_ducked`, `barge_in_ignored_far`, `barge_in_restored`,
`interruption_stored`, `resume_pending`, `resume_continued` (plus existing
`barge_in`, `barge_in_rejected`).

**Testing (TDD):**
- Player: frame chunking, gain application, ring-buffer records the gained
  frame, flush. Regression: AEC reference is non-zero after playback (locks the
  no-op bug shut).
- Barge-in SM: proximity-ignore (low RMS), duck on near-field, accept→cut,
  reject→restore, too-short→ignore.
- Resume: cut stores {query, partial}; steering message built; `RESUME_PENDING`
  set; affirmative continues; negative clears.

## Implementation sequencing

1. Player streaming + real AEC wiring (+ `SileroVAD.is_speaking`).
2. Barge-in state machine (proximity + duck + verify).
3. **Measure** barge-in threshold during ducked playback; set config.
4. Interruption-resume.

## Out of scope

- Cross-session memory (resume state is per-session only).
- Improving AEC algorithm itself beyond correct wiring.
- Multi-party conversation (only the single session primary is ever served).
