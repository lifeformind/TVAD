# Shared Speaker Stack — Design

**Date:** 2026-05-14
**Status:** Draft (awaiting user review)
**Scope:** Shared infrastructure decisions that the Classroom Diarization (S1) and Kiosk Talkback (S2) specs both depend on.

## Purpose

The current `target-vad/` codebase is built around a single streaming use case: continuously listen, gate by speaker identity, fire a callback. Two new use cases — offline classroom diarization and a real-time wake-word kiosk — share the underlying speaker-embedding stack but want different orchestration. This doc establishes the shared layout, configuration shape, and naming so the two project specs can be implementation-ready without circular references or ad-hoc decisions during build.

## Project layout

**Current (flat):**

```
target-vad/
  audio/             mic streaming
  speaker/           ECAPA embedder, enrollment store, verifier
  vad/               Silero VAD wrapper
  pipeline/          target_vad_pipeline.py (legacy streaming demo)
  tests/
  compat.py
  config.yaml
  enroll.py          enrollment CLI
  main.py            legacy --demo entrypoint
  live_test.py       (existing)
```

**After refactor:**

```
target-vad/
  core/              shared primitives, no orchestration
    audio/           (moved)
    speaker/         (moved)
    vad/             (moved)
    compat.py        (moved)
  modes/
    kiosk/           S2 implementation lives here
    diarization/     S1 implementation lives here
  pipeline/          target_vad_pipeline.py stays (legacy demo)
  tests/             grows: tests/core/, tests/kiosk/, tests/diarization/
  config.yaml        re-namespaced (see below)
  enroll.py          stays at root; updated import paths
  main.py            stays at root (legacy demo entry); updated import paths
  diarize.py         new; S1 entry point
  kiosk.py           new; S2 entry point
```

Rationale: `core/` makes the shared primitives unambiguous; `modes/` makes mode-specific orchestration discoverable; entry points stay at the project root for short CLI invocations.

## Configuration

**Current `config.yaml` (flat):**

```yaml
vad: {...}
speaker: {...}
audio: {...}
paths: {...}
```

**After refactor:**

```yaml
core:
  vad: {...}            # unchanged contents
  speaker: {...}        # unchanged contents
  audio: {...}          # unchanged contents
  paths: {...}          # unchanged contents

kiosk:
  wake_phrase: "hey_jarvis"
  session_primary_threshold: 0.60
  session_silence_timeout_s: 10
  session_hard_timeout_s: 300
  decision_smoother:
    window_size: 3
    min_matches: 2
    threshold: 0.60

diarization:
  identification_threshold: 0.55
  default_output_format: "json"   # or "rttm"
  pyannote_pipeline: "pyannote/speaker-diarization-3.1"
```

Rationale: Each mode owns its tunables; `core:` holds knobs that apply to anything using the embedder/VAD. Existing code reading `config["speaker"]["..."]` becomes `config["core"]["speaker"]["..."]` — small, mechanical change.

## Reused components (no behavior changes)

These move into `core/` as-is and are consumed by both S1 and S2:

| Component | File | Used by |
|---|---|---|
| ECAPA embedder | `core/speaker/embedder.py` | S1 (cluster centroid → embedding), S2 (wake-word snapshot, segment matching) |
| Enrollment store | `core/speaker/enrollment_store.py` | S1 (load enrolled voiceprints for identification), S2 (optional auth gate, currently out of scope per Variant A) |
| Cosine similarity | `core/speaker/verifier.py` (`cosine_similarity` function) | Both |
| Silero VAD | `core/vad/silero_vad.py` | S2 (segmenting active session); S1 ignores (pyannote does its own VAD) |
| MicrophoneStream | `core/audio/mic_stream.py` | S2 only |
| compat.py | `core/compat.py` | Both (must be imported before speechbrain in any entry point) |

The existing `SpeakerVerifier` class is consumed by S2 (per-segment matching against the session snapshot — used inside the decision smoother). S1 uses only `cosine_similarity` directly because it operates on cluster centroids, not a continuous stream.

## Decision smoother

S2 needs a sliding-window decision smoother for "is this segment from the session-primary speaker?" The same primitive could be used elsewhere later (including S1's optional cluster-merging heuristic), so it lives in `core/`:

```
core/speaker/decision_smoother.py
  class DecisionSmoother:
    window_size: int
    min_matches: int
    threshold: float
    update(score: float) -> bool   # returns True when M-of-N hit
```

State is per-instance (a deque of recent scores). The smoother is dumb on purpose — it just counts threshold-crossings in its window. The kiosk pipeline owns the lifecycle (creates one per active session, discards on session end).

## Naming and conventions

- Project name unchanged: **Target VAD** / **TVAD**.
- Module imports use `core.*` and `modes.kiosk.*` / `modes.diarization.*` after refactor.
- `from core.speaker.embedder import EmbeddingExtractor` (replaces `from speaker.embedder import ...`).
- All entry points (`enroll.py`, `main.py`, `kiosk.py`, `diarize.py`) import `core.compat` first as the torchaudio/speechbrain shim.

## Logging

`rich`-formatted console output stays for human-readable runtime. **New:** structured event logs as JSON-lines, written to `./logs/<mode>-YYYYMMDD-HHMMSS.jsonl` when `--log` flag is passed. Schema:

```json
{"ts": "2026-05-14T10:23:01.234Z", "mode": "kiosk", "event": "wake_detected", "phrase": "hey_jarvis", "score": 0.87}
{"ts": "...", "mode": "kiosk", "event": "session_started", "snapshot_norm": 1.0}
{"ts": "...", "mode": "kiosk", "event": "segment_scored", "score": 0.71, "duration_ms": 1840, "decision": "match"}
```

Used for offline tuning and post-hoc debugging without re-recording. Both modes write the same way; `event` namespace differs.

## Migration tasks (not implementation, just inventory)

The shared refactor itself is a discrete task:

1. Create `core/` and move `audio/`, `speaker/`, `vad/`, `compat.py`. Update `__init__.py` files accordingly.
2. Rewrite imports across `enroll.py`, `main.py`, `pipeline/target_vad_pipeline.py`, `tests/*` from `from speaker.X` to `from core.speaker.X`, etc.
3. Re-namespace `config.yaml` under `core:` and update consumers (`config["speaker"]["..."]` → `config["core"]["speaker"]["..."]`). Affected files: `enroll.py`, `main.py`, `pipeline/target_vad_pipeline.py`.
4. Run `pytest` — all 23 existing tests should still pass with no behavior change.

This refactor is a prerequisite for both S1 and S2 implementations and should land as its own PR before either mode is built.

## Testing approach

- **Existing `tests/` suite continues to pass unchanged in behavior**, just with updated imports. This is the regression net for the refactor.
- **New tests live under `tests/core/`** for shared primitives (decision_smoother).
- **Mode-specific tests** under `tests/kiosk/` and `tests/diarization/`, scoped per spec.
- Audio-dependent tests use canned fixtures (synthetic speech, real recorded utterances) to keep CI hermetic; mic-dependent tests are manual-only and clearly marked.

## Out of scope for this doc

- Specific test cases for kiosk and diarization (those live in their respective spec docs).
- Downstream STT/LLM/TTS integration for S2 (entirely separate project).
- Phase-2 ASR for S1 (mentioned only as a hook point in the S1 spec).
- Any changes to ECAPA model selection, VAD model, or enrollment-time flow.
- Custom wake-word training (S2 uses bundled openwakeword models).

## Open questions

None known at the time of writing. All called-out decisions have been resolved by user input on 2026-05-14.
