# Classroom Diarization & Identification — Design (S1)

**Date:** 2026-05-14
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-14-shared-speaker-stack.md`](./2026-05-14-shared-speaker-stack.md)

## Purpose

Take a recording of a classroom session (single-channel, single-mic, up to 10 active speakers including the teacher) and produce a timestamped timeline of who-spoke-when, attributing each segment to a previously enrolled speaker by name or labeling it `"unknown"`. The output is consumed offline for post-session analysis (attendance, participation tracking, content review). Phase 2 layers Whisper ASR on top to produce a labeled transcript; Phase 1 (this spec) is timeline-only.

## Inputs and outputs

### Input

A WAV file, single channel (mono), 16 kHz preferred.

- If the recording is at a different sample rate / channel count, the entry point resamples / mixes down to 16 kHz mono in-memory before invoking pyannote. (pyannote can handle other formats, but standardizing simplifies the embedder hand-off.)
- No assumed maximum duration; pyannote runs over the whole file in one batch. Memory bounded by the file, not the pipeline.

### Output

JSON file (default), or RTTM if `--rttm` is passed.

**JSON schema:**

```json
{
  "audio_file": "path/to/session.wav",
  "duration_s": 2734.51,
  "diarized_at": "2026-05-14T10:23:01Z",
  "config": {
    "pyannote_pipeline": "pyannote/speaker-diarization-3.1",
    "identification_threshold": 0.55
  },
  "enrolled_users_matched": ["siddharth"],
  "segments": [
    {"start": 0.42, "end": 3.81, "speaker": "siddharth"},
    {"start": 3.81, "end": 5.10, "speaker": "unknown"},
    {"start": 5.20, "end": 7.45, "speaker": "siddharth"}
  ]
}
```

Notes:
- `speaker` is either an enrolled username or the literal string `"unknown"`. (Per user decision, no `unknown_1`/`unknown_2` differentiation.)
- Segments are sorted by `start`. Adjacent same-speaker segments are NOT merged at this layer; consumers can do that if they want.
- `enrolled_users_matched` is a deduped list of enrolled names that appeared at least once in the session, ordered by first appearance time. Useful for quick consumption (e.g., attendance check) without scanning all segments.

**RTTM** is the standard speaker-diarization community format (used by pyannote and others). The `--rttm` flag writes a `.rttm` file alongside (or instead of) the JSON. Useful for interop with diarization evaluation tools.

## Pipeline

```
[wav file]
    │
    ▼
[load + resample → 16 kHz mono float32]
    │
    ▼
[pyannote.audio "speaker-diarization-3.1" pipeline]
    │   ├─ does VAD internally
    │   ├─ does speaker change detection
    │   └─ does unsupervised clustering
    ▼
[pyannote.core.Annotation: list of (start, end, cluster_id)]
    │
    ▼
[For each cluster_id:
   - extract all audio segments belonging to this cluster
   - concat into a single waveform (or a sample of segments if very long)
   - pass through ECAPA → centroid embedding (192-dim, L2-normalized)
   - cosine match against all enrolled voiceprints
   - if best_score >= identification_threshold (0.55): label = enrolled name
   - else: label = "unknown"]
    │
    ▼
[Apply cluster→label mapping to all segments]
    │
    ▼
[Write JSON / RTTM]
```

### Why operate on cluster centroids, not per-segment

pyannote already groups same-speaker segments into clusters. A centroid embedding (averaged across all of a cluster's audio) is more stable than any single segment's embedding, especially with the C10's noisy spectral content. This was an explicit lesson from the streaming work in this same project — averaged embeddings are tighter than per-segment ones. Identification threshold is set lower (0.55) than streaming would use because centroid-vs-voiceprint cosines tend to land slightly lower than peak-segment-vs-voiceprint cosines, but with much less variance.

### Centroid sampling for very long clusters

If a cluster has more than ~30 s of audio, sample 30 s of segments rather than concatenating everything. Reduces compute without meaningfully reducing centroid quality (ECAPA centroids saturate well before 30 s). Implementation: pick segments evenly spaced through the cluster's timeline.

## Components

| File | Responsibility |
|---|---|
| `target-vad/diarize.py` | CLI entry point: arg parsing, file loading, orchestration, output writing |
| `target-vad/modes/diarization/diarizer.py` | `Diarizer` class: wraps pyannote pipeline, returns clusters |
| `target-vad/modes/diarization/identifier.py` | `ClusterIdentifier` class: takes clusters + enrolled voiceprints, returns label per cluster |
| `target-vad/modes/diarization/output.py` | JSON and RTTM writers |
| `target-vad/modes/diarization/__init__.py` | (empty) |
| `core/speaker/embedder.py` | (reused as-is) used by `ClusterIdentifier` to embed cluster audio |
| `core/speaker/enrollment_store.py` | (reused as-is) loads enrolled voiceprints |
| `core/speaker/verifier.py::cosine_similarity` | (reused as-is) used by `ClusterIdentifier` |

## CLI

```
py -3.14 diarize.py <input.wav> [--out output.json] [--rttm] [--config config.yaml] [--log]
```

- `<input.wav>`: required positional.
- `--out`: defaults to `<input>.diarization.json` next to the input file.
- `--rttm`: also writes `<input>.diarization.rttm`. Can be combined with custom `--out` (RTTM takes the same stem).
- `--config`: defaults to `./config.yaml`.
- `--log`: enables JSON-lines event log per the shared spec.

CLI prints a progress summary using `rich`: file loaded → diarization in progress (with spinner) → N clusters found → identifying each → output written. Exit code 0 on success, non-zero on failure with stderr message.

## Configuration

Per the shared spec, the `diarization:` block in `config.yaml`:

```yaml
diarization:
  identification_threshold: 0.55
  default_output_format: "json"
  pyannote_pipeline: "pyannote/speaker-diarization-3.1"
  hf_token_env_var: "HF_TOKEN"
  centroid_max_sample_seconds: 30
```

`hf_token_env_var` lets the user keep their HuggingFace token out of the config file; `diarize.py` reads `os.environ[hf_token_env_var]` at startup. If not set, prints an actionable error pointing to the pyannote setup docs.

## Error handling

| Failure | Behavior |
|---|---|
| Input file missing or unreadable | Exit 2 with error message |
| Sample rate / channel count not supported by `soundfile` (e.g., exotic codec) | Exit 2 with conversion hint |
| HF token not set in environment | Exit 3 with message: how to obtain a token, where to set it |
| pyannote model download fails (no internet, gated model) | Exit 3 with hint about HF model gating + token scopes |
| pyannote returns zero clusters (e.g., silent file) | Exit 0; write JSON with empty `segments`. Print warning to stderr |
| Embedder fails on a cluster (e.g., audio is all zeros after a clipping bug) | Skip that cluster: label as `"unknown"`, log warning, continue |
| No enrolled voiceprints | Exit 0; all clusters labeled `"unknown"`. Print info message |

## Testing approach

`tests/diarization/` contains:

- **Unit tests:**
  - `test_identifier.py`: `ClusterIdentifier` given mock pyannote output + a fake enrollment store → asserts correct label assignment for "should match" / "should be unknown" cases. No model loading.
  - `test_output.py`: JSON and RTTM serialization roundtrip tests.
  - `test_centroid_sampling.py`: long cluster sampling logic — fed 60 s of timestamps, asserts ≤ 30 s sampled, evenly spaced.
- **Integration test (manual, not in CI):**
  - `test_end_to_end.py.skip`: runs full pipeline against a small canned multi-speaker WAV. Marked skip by default because it downloads pyannote models; flag to enable with `--run-slow`.
- **Existing 23 tests must continue to pass.**

## Quality expectations and known limitations

These belong in the spec because they shape what "done" means:

- **Diarization Error Rate (DER) target: ≤ 30%** on classroom-condition audio (10 speakers, C10, moderate noise). For comparison: pyannote benchmarks on AMI/DIHARD report 18–22% DER; we degrade for the harder mic+room combo.
- **Overlapping speech:** when two speakers talk simultaneously, pyannote produces a single cluster with mixed audio, the centroid embedding is non-physical, and identification typically falls below threshold → labeled `"unknown"`. **This is expected behavior**; reliably attributing overlapping speech requires separation models that are out of scope.
- **Unknown speakers leaking into enrolled labels:** the 0.55 threshold is conservative-ish; some unknown speakers may still cross it (false positive). Tunable via config. Recommend evaluating against a labeled session before production use.
- **Up to 10 speakers:** at 7+ active speakers, pyannote's clustering quality degrades. Diarization may merge two distinct quiet speakers into one cluster (which then either matches an enrolled person incorrectly or is labeled unknown).

## Phase 2 ASR — hook point only

The output JSON schema reserves space for a future `text` field per segment:

```json
{"start": 0.42, "end": 3.81, "speaker": "siddharth", "text": "OK class let's begin"}
```

Phase 2 will add a `transcribe.py` post-processor that takes a diarization JSON + the original audio, runs `faster-whisper` per segment, and writes back the same JSON with `text` filled in. No changes to Phase 1's output shape required — the `text` field is just absent until transcribed. Spec for Phase 2 is deferred.

## Out of scope

- ASR (Phase 2, deferred).
- Speaker separation for overlapping speech.
- Real-time / streaming diarization (this is offline only).
- Per-segment confidence scores (could be added; not requested).
- Speaker identification across multiple sessions (each session is processed independently; matching is per-session).
- Dashboards, UI, or any consumption tool for the JSON output.

## Open questions

None at the time of writing. All decisions resolved 2026-05-14.
