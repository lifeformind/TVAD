# TVAD Roadmap

The 5-pass offline pipeline (S1, S2, 2A, 2B, 4, 3) shipped 2026-05-14 → 2026-05-16. This document organizes the next set of work into batches by priority and dependency.

**Status as of 2026-05-17:** 237 tests passing on `master`. All 5 passes validated end-to-end on `Voice 001 short.wav`. README + INTEGRATION docs landed. No outstanding bugs in shipped passes.

Items below are independent unless a dependency is called out. Pick batches based on available time and what's blocked vs unblocked.

---

## Batch 1 — Quick wins / cross-cutting cleanup

Effort: **~1-2 days total**. Each item is 1-3 hours. Items 1a + 1b are DRY refactors that should ship first — subsequent items become easier because of them. Items 1c–1f are independent and can ship in any order.

### 1a. Extract `_atomic_write_json` into `core/io/atomic.py`

The atomic-write helper is now copy-pasted in 4 files: `transcribe.py`, `sentiment.py`, `metrics.py`, `prosody.py`. Same DRY opportunity as the audio-loader extraction (which dedupes 3 files via `core/audio/load.py`).

**Effort:** ~30 min.
**Acceptance:** new `target-vad/core/io/atomic.py` exporting `atomic_write_json(path: str, data: dict)`. All 4 CLIs import from it. Test count unchanged.
**Risk:** trivial — pure no-op refactor.

### 1b. Shared error-shape helper in `core/errors.py`

The four CLI passes each have ~30 lines of duplicated `console.print(f"[red]...[/]")` + exit-code-return error patterns. A `core/errors.py` with `print_bad_input(reason, hint=None) -> EXIT_BAD_INPUT` and `print_env_failure(reason, hint=None) -> EXIT_CONFIG_OR_IO` would dedupe and standardize message wording.

**Effort:** ~1 hour.
**Acceptance:** new `target-vad/core/errors.py`. All 4 CLI passes use the helpers for error paths. Console output formatting unchanged (verify via smoke).
**Risk:** trivial — confirm via running each CLI against a known-bad input and comparing the messages.

### 1c. Phase 3 metrics integration of Phase 4 prosody

`prosody_baselines` and per-segment prosody data sit in the JSON but never surface in the Markdown AAR report. Add a `Prosody (per speaker)` table showing median pitch + IQR + median energy + IQR per speaker. Optionally a new highlight kind `highest_pitch_range_window` for sessions that warrant it.

**Effort:** ~2 hours (renderer-only change, ~40 lines + golden-file update).
**Acceptance:**
- New `## Prosody (per speaker)` section in the Markdown report when `prosody_baselines` is present in the JSON
- Optional new `highest_pitch_range_window` highlight (skip if no prosody data)
- Section omitted entirely when prosody hasn't run (no `prosody_baselines` field)
- Golden-file test updated
- One new test for the omission rule + one for the new section's content

**Risk:** low — pure renderer addition, no aggregator changes.
**Why it matters:** makes Phase 4's data actually visible to facilitators who read the report. Currently the only consumer is a hypothetical Phase 2C engagement model.

### 1d. Real-pyin integration test

Phase 4's 8 analyzer tests all use synthetic sine waves; the orchestration tests stub the analyzer entirely. There's no test that exercises real `librosa.pyin` + real `librosa.feature.rms` end-to-end. Real-library version drift would silently break the pass.

**Effort:** ~1 hour (record + commit a tiny ~50 KB speech WAV; write one integration test).
**Acceptance:**
- New fixture `tests/prosody/fixtures/speech_sample.wav` (~1-2 seconds of clean speech, committed)
- New test in `test_analyzer.py` or `test_integration.py` that runs the real `analyze_segment` on the fixture and asserts pitch/energy are in plausible ranges (e.g., `80 < pitch_median < 250` for an adult voice)
- Test runs in under 5 seconds (pyin's numba JIT is the main cost)

**Risk:** low — the test asserts ranges rather than exact values, so it'll be robust to library version changes.

### 1e. Phase 3 untested highlight kinds

`busiest_window`, `quietest_window`, `solo_dominator`, `high_disgust_window` are exercised only by manual smoke on `Voice 001`. None has a dedicated unit test. The selection rules are simple but the tie-breaking and skip-conditions need pinning.

**Effort:** ~1 hour (4 new tests in `test_aggregator.py`).
**Acceptance:**
- One test per kind with a fixture that triggers it and asserts the highlight emitted with correct fields
- One test per kind for the skip condition (no positive segments → most_positive omitted, etc.)
- Tests run synchronously without I/O

**Risk:** none — pure unit tests.

### 1f. Small backlog cleanup

Six tiny items, can be done as one combined cleanup commit:

- **`id` parameter shadows the Python builtin** in 4 files (enroll.py, etc.). Rename to `user_id` or similar.
- **Pin `scipy>=1.11.0`** explicitly in `requirements.txt` (currently transitively pulled but not pinned).
- **Friendly error on missing `diarization:` config block** in `diarize.py` (currently throws `KeyError: diarization`; should be a friendly `[red]Config missing diarization: block[/]` + exit 3, matching the pattern other passes use).
- **`enroll_from_intros` short-slice warning** — warn if a manifest entry has duration < 800 ms (currently crashes inside ECAPA cryptically).
- **F6: structured JSONL logging for kiosk** — the on_event hook is wired only for dry-run console output. Add a JSONL writer for production audit trails.
- **F4: kiosk watchdog for silence/hard timeouts** — currently relies on chunk arrival to fire timeouts; latent risk on idle mics.

**Effort:** ~1-2 hours total.
**Acceptance:** each item has either a new test (for the cleanup ones) or a manual smoke (for F4/F6). Test count grows by ~5.
**Risk:** F4/F6 touch kiosk runtime; lower priority because S2 is single-user-mode and the gap is latent.

---

## Batch 2 — Validation (no code, awaits real audio)

These are gap-closures that need real recordings, not code changes. **The highest-value items in the whole roadmap** — they answer "does this thing actually work?" — but they're blocked on you having the right audio fixtures.

### 2a. Live no-intros S1 run when `HF_TOKEN` is set

Closes the forward-path loop on the recurring-unknown clusters work. The synthetic JSON in `c:/tmp/voice001-as-no-intros.diarization.json` stands in for this but doesn't replace a real run.

**Effort:** 5 minutes once `HF_TOKEN` is set in the shell.
**Acceptance:** `py -3.14 diarize.py "../Voice 001 short.wav"` (no `--introductions`) succeeds; the resulting JSON has `enrolled_users_matched: []` and `unknown_speakers_observed` with 2 entries (`SPEAKER_00` + `SPEAKER_01`).
**Risk:** none — verifies expected behavior.

### 2b. Multi-speaker / longer recording end-to-end

Voice 001 is single-recording, mostly two speakers, ~90 s. The activity-chart rendering path (`≥ 2 buckets`), the `busiest_window`/`quietest_window`/`solo_dominator` highlights, and the multi-segment-overlap timeline behaviors are exercised only by Voice 001 — and Voice 001 fits in one bucket, so those code paths aren't really tested.

**Effort:** 10-30 minutes once a multi-speaker recording exists (a real classroom recording or even a podcast clip with 3-4 speakers and ≥15 min duration).
**Acceptance:** full chain runs (diarize → transcribe → sentiment → prosody → metrics). The Markdown report has a multi-bucket activity chart visible; at least one of `busiest_window`/`quietest_window`/`solo_dominator` fires. No crashes.
**Risk:** medium — exposes any latent bug in the underexercised paths. Expected to uncover 1-2 small issues.

### 2c. Non-self false-positive test for S2 kiosk

The highest-value validation gap in the entire project. S2's 0.50 cosine threshold is tuned for "Siddharth vs Siddharth" because no other voice has ever been tested. We don't know what cosine the kiosk produces for an interloper voice. Without this, the kiosk could be either too permissive (accepts strangers) or too strict (rejects the enrolled user under varied conditions).

**Effort:** 30 minutes once a second human is available to test.
**Procedure:**
1. Enroll Siddharth (already done if voiceprints/siddharth.npy exists)
2. Run `kiosk.py --dry-run`
3. Wake the kiosk (say "hey jarvis")
4. Have a different person say a few sentences after the wake
5. Inspect `[SCORED ... → MATCH/no_match]` events — non-self segments should consistently score below the threshold
6. Repeat 5+ times to get a distribution

**Acceptance:** documented non-self cosine distribution (mean + range) committed to memory or a new spec. If non-self cosines overlap meaningfully with self cosines (0.4-0.7 range per current data), recommend a new threshold or note the C10's limits.

**Risk:** high signal — may reveal the kiosk needs a different mic or a smarter discriminator (e.g., requiring N consecutive matches before locking in the session primary speaker).

---

## Batch 3 — Topic segmentation (next non-LLM pass)

The last new analytical pass on the "without LLM" list. Adds a `topic_segments` field showing topic boundaries detected over the transcript.

### 3a. Topic segmentation pass

**What it does.** Embeds each segment's text with sentence-BERT (or similar small embedder), computes pairwise similarities along the time axis, and runs change-point detection to find boundaries where the conversation shifts topic. Output: `topic_segments: [{start, end, segment_indices}]`. **Unlabeled** — labeling each topic ("aircraft control", "radar systems") is a Phase 2C LLM job; boundary detection is fully non-LLM.

**Dependencies:**
- New embedder model: sentence-BERT (~80 MB) — likely `sentence-transformers/all-MiniLM-L6-v2` (which LAILAI also uses, per its README — useful for shared cache)
- New library: `sentence-transformers>=2.0`

**Effort:** ~2 days (full spec → plan → 6-task subagent execution, similar in size to Phase 4 prosody).
**Acceptance:** new `topics.py` CLI; per-spec brainstorming on the change-point algorithm choice (BERTopic vs custom sliding-window cosine vs HMM); ~15 new tests.
**Risk:** medium — sentence-BERT version compatibility with the rest of the stack (numpy / torch versions) may require adjustment.

### 3b. Phase 3 integration of topic data (after 3a ships)

Once `topic_segments` exist, Phase 3 metrics can surface "we discussed N topics; topic 1: minutes 0-12, topic 2: minutes 12-18" — even without labels. Adds a `## Topics` section to the Markdown report with timestamps and word-counts per topic.

**Effort:** ~1 hour (renderer-only).

**Order:** 3a then 3b. 3b is small enough to fold into the 3a plan.

---

## Batch 4 — Future work (blocked / deferred)

Items here are listed for visibility but not actionable until something else changes.

### 4a. Cross-session comparison

Longitudinal aggregation across multiple sessions. Needs a small session-store on disk (file structure TBD). Wait until you have 3+ real classroom recordings to compare — otherwise the design is hypothetical.

**Blocker:** need a real corpus to design against.

### 4b. Phase 2C engagement labels (LLM-driven)

The reserved `sentiment.engagement` slot in the JSON. Backend choice (local llama.cpp via LAILAI vs Anthropic API) is the blocking design decision. The TVAD-side spec needs brainstorming when this is prioritized.

**Blocker:** LAILAI integration progress + decision on LLM backend (local vs API).

**Recommended trigger:** when LAILAI starts consuming TVAD's enriched JSON in its meetings hub, the prompt to write Phase 2C becomes immediate. See [INTEGRATION.md Appendix A](INTEGRATION.md#appendix-a--lailai-integration-recipe).

### 4c. HTML / PDF report renderer

Markdown is the only Phase 3 output format. An HTML renderer (with embedded charts via Chart.js or similar) would make the AAR more visually rich. Deferred — Markdown is sufficient for facilitator review and renders well in VSCode / GitHub.

### 4d. Per-cluster identity for unknown speakers (S1 schema upgrade)

If multiple unenrolled speakers in a session don't pass the recurrence threshold, they currently all collapse to the literal `"unknown"` bucket. A future upgrade could preserve pyannote's cluster id for sub-threshold clusters too — at the cost of more "speakers" cluttering the Markdown report. Deferred until classroom recordings show this is a real problem.

### 4e. Speaker-relative z-scores per segment (prosody)

Phase 4 emits raw prosody values + per-speaker baselines. Consumers compute z-scores or IQR-offsets themselves. Adding pre-computed z-score fields to each segment would be convenient but doubles the per-segment field count. Defer until a real consumer requests it.

---

## Suggested sequencing

Given typical work-session cadence and the relative leverage of each batch, here's a recommended order:

1. **First** — Batch 1a + 1b + 1c (atomic.py extraction, error helper, prosody → metrics). Half-day total. Maximum DRY + maximum user-visible improvement.

2. **Second** — Batch 1d + 1e + 1f (real-pyin test, untested highlight kinds, small backlog). Closes test coverage gaps; ~half day.

3. **Third (whenever real audio is available)** — Batch 2a, then 2b, then 2c. The 2c gap is the most important but requires a second human; do it when one is available.

4. **Fourth** — Batch 3 (topic segmentation). Multi-day; do this when you want a meaty new feature rather than incremental polish.

5. **Eventually** — Batch 4 items as their blockers clear.

**Parallel-track suggestion:** Batch 2 is no-code, so it can happen alongside any code work. If you have a multi-speaker recording, run Batch 2b before any Batch 1 work — it might surface a bug worth fixing first.

---

## What's deliberately NOT on this list

- **Refactoring `target-vad/` into a proper installable package** (setup.py, pyproject.toml). Currently the project relies on CWD-based imports. Defer until someone wants to install TVAD via pip; the subprocess integration model in INTEGRATION.md doesn't require it.
- **GPU / NPU acceleration.** All passes are CPU-only by design. Acceleration is a future optimization, not a feature gap.
- **CI/CD setup.** No GitHub Actions workflow exists. Defer until external contributors or a release process needs it.
- **Streaming / online modes for Phase 1-4.** The pipeline is one-shot batch processing on completed recordings. S2 is the only real-time mode; extending the analysis passes to streaming would be a major architecture shift, not on the table.
- **Localization / multilingual support.** Phase 2A is English-only (`language: "en"`); Phase 2B uses English-only sentiment models. Adding language-conditional model selection would be a separate spec.

---

For per-feature design rationale, see [specs](superpowers/specs/). For integration with external consumers (LAILAI, etc.), see [INTEGRATION.md](INTEGRATION.md). For the project overview, see [../README.md](../README.md).
