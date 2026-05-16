# Prosody Pass (Phase 4) — Design

**Date:** 2026-05-16
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-15-transcription-pass-design.md`](./2026-05-15-transcription-pass-design.md) (Phase 2A, shipped)

## Purpose

Add `prosody.py`, the fourth analysis pass. Reads a post-2A diarization JSON + the audio WAV, computes per-segment prosodic features (pitch, energy, speaking rate) via DSP on the audio waveform plus word-timestamp arithmetic from 2A, and attaches a nested `prosody` block per segment. Also emits a top-level `prosody_baselines` summarizing each speaker's distribution.

The motivating use case is signal that text-only sentiment misses. The 2B emotion classifier struggles on short or ambiguous utterances ("Yeah." can be enthusiastic, dismissive, neutral, or stressed depending on prosody alone). Phase 2C engagement labels will need this signal. The Phase 3 AAR report doesn't surface prosody in v1 — it sits in the JSON for downstream consumers.

Local-first, no model downloads, no network. Pure DSP via librosa.

## Architecture

`prosody.py` is a stand-alone CLI that reads a post-2A diarization JSON, validates the prerequisites, loads the audio once, walks each segment to compute prosody features, aggregates per-speaker baselines, and writes back atomically.

```
session.diarization.json (with text + words per segment)
        +
session.wav
            │
            ▼
[load JSON, validate passes_run ⊇ {"transcription"}]
            │
            ▼
[load full audio once -> mono 16 kHz numpy array]
            │
            ▼
[for each segment:
   - if already has `prosody` and --rerun not set: skip
   - else: slice audio chunk, analyze, attach `prosody` block]
            │
            ▼
[aggregate per-speaker baselines from all classified segments]
            │
            ▼
[update top-level passes_run += "prosody"; emit prosody_baselines + prosody_config]
            │
            ▼
[atomic write back to original path (or --out)]
```

Pure CPU. Per-segment work is independent — could be parallelized later if needed; v1 is sequential for simplicity. Baseline aggregation is pure-Python over the final segment list.

## Output schema additions

**Per-segment field** — one new nested block, leaves existing fields untouched:

```json
{
  "start": 0.42, "end": 3.81,
  "speaker_id": "siddharth", "speaker": "Siddharth Jain",
  "text": "...", "words": [...], "sentiment": {...},
  "prosody": {
    "pitch_hz_median": 142.3,
    "pitch_hz_std": 22.1,
    "pitch_range_hz": 85.0,
    "energy_db_mean": -23.8,
    "energy_db_range": 9.2,
    "speech_rate_wps": 2.8,
    "pause_ratio": 0.15
  }
}
```

**Sentinel:** `prosody: null` when the segment has zero voiced frames AND empty `words` (no signal of any kind). Otherwise emit the block with individual fields possibly null:

| Field | Null when |
|---|---|
| `pitch_hz_median`, `pitch_hz_std`, `pitch_range_hz` | All frames unvoiced (pyin returns NaN throughout) |
| `energy_db_mean`, `energy_db_range` | Zero-length audio chunk (degenerate) |
| `speech_rate_wps`, `pause_ratio` | Segment's `words` is empty |

**Field semantics:**

| Field | Computation |
|---|---|
| `pitch_hz_median` | Median f0 across voiced frames (via `librosa.pyin`, default range 80–400 Hz) |
| `pitch_hz_std` | Standard deviation of voiced f0 |
| `pitch_range_hz` | 95th percentile minus 5th percentile of voiced f0 (clipped to avoid octave-error outliers) |
| `energy_db_mean` | Mean of `librosa.amplitude_to_db(rms, ref=1.0)` across all frames |
| `energy_db_range` | 95th percentile minus 5th percentile of frame-level dB |
| `speech_rate_wps` | `len(words) / segment_duration_s` |
| `pause_ratio` | `(segment_duration - sum(word_durations)) / segment_duration`, clamped to [0, 1] |

All numeric fields round to 2 decimals for stable JSON output.

**Top-level additions:**

```json
"passes_run": ["diarization", "transcription", "sentiment", "prosody"],
"prosody_config": {
  "pitch_min_hz": 80,
  "pitch_max_hz": 400,
  "frame_length_ms": 25,
  "hop_length_ms": 10,
  "analyzed_at": "2026-05-16T19:42:01Z"
},
"prosody_baselines": {
  "siddharth": {
    "pitch_hz_median": 138.2,
    "pitch_hz_iqr": 24.8,
    "energy_db_median": -25.4,
    "energy_db_iqr": 6.1,
    "segment_count": 8
  },
  "SPEAKER_00": {
    "pitch_hz_median": 175.5,
    "pitch_hz_iqr": 31.0,
    "energy_db_median": -22.1,
    "energy_db_iqr": 4.8,
    "segment_count": 4
  }
}
```

`prosody_baselines` keys are every speaker_id that has at least one segment with non-null prosody data. Includes enrolled, recurring unknowns (pyannote ids), and the literal `"unknown"` catchall if it has any prosody data. Median + IQR (interquartile range, 75th - 25th percentile) chosen over mean + std for robustness against outliers.

`passes_run` appends `"prosody"` deduped on rerun; `prosody_config` overwritten each run.

## Components

| File | Status | Responsibility |
|---|---|---|
| `target-vad/prosody.py` | create | CLI entry: arg parsing, audio load, JSON load/save, atomic write, orchestration |
| `target-vad/modes/prosody/__init__.py` | create | empty package marker |
| `target-vad/modes/prosody/analyzer.py` | create | `analyze_segment(audio_chunk, sample_rate, words, segment_duration, cfg) -> Dict` — pure function returning the 7-field prosody block |
| `target-vad/modes/prosody/baselines.py` | create | `compute_baselines(segments) -> Dict[speaker_id, baseline_dict]` — pure-Python aggregation |
| `target-vad/core/audio/load.py` | create | extract `load_audio_as_mono16k` from `diarize.py` into a shared module so `prosody.py` can import it |
| `target-vad/diarize.py` | modify | import `load_audio_as_mono16k` from the new shared module instead of defining inline |
| `target-vad/config.yaml` | modify | add `prosody:` block |
| `target-vad/requirements.txt` | modify | add `librosa>=0.10.0` |
| `target-vad/tests/prosody/__init__.py` | create | empty |
| `target-vad/tests/prosody/test_analyzer.py` | create | analyzer tests on synthetic audio |
| `target-vad/tests/prosody/test_baselines.py` | create | baseline aggregation tests |
| `target-vad/tests/prosody/test_orchestration.py` | create | CLI tests with a stub analyzer |

The audio loader extraction is the only existing-code refactor. The current `load_audio_as_mono16k` in `diarize.py` moves to `core/audio/load.py` unchanged; `diarize.py` switches to importing it. This tiny refactor pays for itself once `prosody.py` becomes the second caller.

## CLI

```
py -3.14 prosody.py <diarization.json> [--audio <wav>] [--out <path>] [--rerun] [--config <path>]
```

- Positional `<diarization.json>` — required, must have `passes_run` ⊇ `{"transcription"}`.
- `--audio` — path to the WAV. Default: infer from the JSON's `audio_file` field (absolute path written by `diarize.py`).
- `--out` — output JSON path. Defaults to in-place atomic write.
- `--rerun` — re-analyze segments that already have a `prosody` field. Default skips them.
- `--config` — path to `config.yaml`. Default `./config.yaml`.

**Pre-flight validation** (exit 2 with clear pointer):

- JSON missing or unreadable → exit 2
- JSON malformed → exit 2
- JSON missing `segments` field → exit 2
- `passes_run` missing `"transcription"` → exit 2 with pointer to `transcribe.py`
- Any segment missing `text` or `words` → exit 2 with segment index
- Audio file missing → exit 2
- Audio file unreadable / wrong format → exit 2
- `prosody:` config block missing → exit 3

**Console output on success:**

```
Prosody written: 9 analyzed, 0 skipped (already had prosody), 0 failed (no voiced frames).
  JSON -> /path/to/session.diarization.json
```

A `rich.progress` bar `Analyzing [bar] N/M` is shown while running.

## Configuration

```yaml
prosody:
  pitch_min_hz: 80           # pyin floor — typical human speech floor
  pitch_max_hz: 400          # pyin ceiling — typical adult speech ceiling
  frame_length_ms: 25        # standard speech-analysis window
  hop_length_ms: 10          # standard 60% overlap
```

All four knobs are config-only for v1; CLI overrides can be added non-breakingly later. All four surface through to `prosody_config` in the JSON for reproducibility.

## Analyzer behavior

`analyze_segment(audio_chunk, sample_rate, words, segment_duration, cfg) -> Dict`:

### Pitch

```python
f0, voiced_flag, voiced_prob = librosa.pyin(
    audio_chunk,
    fmin=cfg["pitch_min_hz"],
    fmax=cfg["pitch_max_hz"],
    sr=sample_rate,
    frame_length=int(sample_rate * cfg["frame_length_ms"] / 1000),
    hop_length=int(sample_rate * cfg["hop_length_ms"] / 1000),
)
voiced_f0 = f0[~np.isnan(f0)]
```

- If `len(voiced_f0) == 0`: all three pitch fields are null
- Else: `pitch_hz_median = np.median(voiced_f0)`, `pitch_hz_std = np.std(voiced_f0)`, `pitch_range_hz = np.percentile(voiced_f0, 95) - np.percentile(voiced_f0, 5)`

### Energy

```python
rms = librosa.feature.rms(
    y=audio_chunk,
    frame_length=int(sample_rate * cfg["frame_length_ms"] / 1000),
    hop_length=int(sample_rate * cfg["hop_length_ms"] / 1000),
)[0]
db = librosa.amplitude_to_db(rms, ref=1.0)
```

- If audio_chunk is zero-length: energy fields null
- Else: `energy_db_mean = np.mean(db)`, `energy_db_range = np.percentile(db, 95) - np.percentile(db, 5)`

### Rate

```python
if not words:
    speech_rate_wps = None
    pause_ratio = None
else:
    word_total_s = sum(w["end"] - w["start"] for w in words)
    speech_rate_wps = len(words) / segment_duration
    pause_ratio = max(0.0, min(1.0, (segment_duration - word_total_s) / segment_duration))
```

The `max(0.0, min(1.0, ...))` clamp protects against whisper's word-timestamp slop where summed durations can slightly exceed segment duration.

### Baseline aggregation

`compute_baselines(segments) -> Dict[speaker_id, dict]`:

For each speaker_id appearing in segments (excluding speakers with all-null prosody):

- Collect every segment's `pitch_hz_median` (non-null) → baseline `pitch_hz_median` = median of those values; baseline `pitch_hz_iqr` = `np.percentile(values, 75) - np.percentile(values, 25)`
- Same for `energy_db_mean` → baseline `energy_db_median`, baseline `energy_db_iqr`
- `segment_count` = number of segments contributing (at least one non-null prosody field)

Speakers with zero contributing segments are omitted from `prosody_baselines` entirely.

## Conflict resolution / edge cases

| Case | Behavior |
|---|---|
| Re-run on already-analyzed JSON, no `--rerun` | Segments with non-null `prosody` are skipped; reported in summary as `skipped N` |
| Re-run with `--rerun` | All segments re-analyzed from scratch |
| Segment with no voiced frames AND empty `words` | `prosody: null` (full sentinel) |
| Segment with no voiced frames but `words` present | `prosody: {pitch_hz_median: null, pitch_hz_std: null, pitch_range_hz: null, energy_db_mean: <num>, energy_db_range: <num>, speech_rate_wps: <num>, pause_ratio: <num>}` |
| Audio sample rate ≠ 16 kHz | Resampled via the shared `load_audio_as_mono16k` helper (`scipy.signal.resample_poly`) |
| Segment duration < frame_length (25 ms) | librosa returns empty arrays → pitch and energy null; rate fields computed normally if words exist |
| Stereo input WAV | Loader returns mono (mean across channels) |
| Per-segment analyzer crash (unexpected exception) | Print `[yellow]warning[/]`, set `prosody: null`, continue. Counted as "failed" in summary |
| Whole-pipeline crash mid-run | Atomic write means original JSON is unmodified. User reruns; incremental mode resumes |
| JSON missing `audio_file` field AND no `--audio` arg | Exit 2 with clear error |

## Error handling table

| Failure | Exit code |
|---|---|
| JSON missing / unreadable / malformed | 2 |
| JSON missing `segments` | 2 |
| `passes_run` missing `"transcription"` | 2 |
| Segment missing `text` or `words` | 2 |
| Audio file missing / unreadable | 2 |
| `prosody:` config block missing | 3 |
| librosa import failure | 3 |
| Per-segment analyzer crash | 0 (warning + null marker, continue) |
| Atomic write failure | 3 |

Same categorization as the prior passes: 2 for user-supplied bad input, 3 for environment/IO failures.

## Testing approach

`tests/prosody/test_analyzer.py` (~8 tests, synthetic audio):

- 200 Hz sine wave → `pitch_hz_median ≈ 200`, `pitch_hz_std` small, `pitch_range_hz` small
- Silence (zeros) → all pitch fields null; `energy_db_mean` very low (floor)
- Concatenated 200 Hz + 400 Hz sine waves → `pitch_range_hz` reflects the spread
- Empty `words` → `speech_rate_wps = None`, `pause_ratio = None`
- 3 words spanning 1.0 s over 2.0 s segment → `speech_rate_wps = 1.5`, `pause_ratio ≈ 0.5`
- Words summing longer than segment duration → `pause_ratio` clamped to 0.0
- Zero-length audio chunk → both pitch and energy fields null
- Configurable pitch range honored (50–500 Hz config) — pyin called with those bounds

`tests/prosody/test_baselines.py` (~4 tests, pure-Python dicts):

- Two speakers, three segments each → both in output with correct medians/IQRs and `segment_count: 3`
- One speaker with all-null prosody → omitted entirely
- Mixed null/non-null segments for a speaker → baseline from non-null only; correct `segment_count`
- IQR on a constant sequence → 0.0

`tests/prosody/test_orchestration.py` (~7 tests, stub analyzer + tiny synthetic WAV):

- Happy path: fresh analysis attaches `prosody` to each segment + baselines block; `passes_run` += `"prosody"`
- `--out` path: input JSON untouched, output JSON has the new fields
- `--audio` override path used when JSON's `audio_file` is wrong
- Idempotent rerun (no `--rerun`): segments with existing `prosody` skipped; baselines recomputed
- `--rerun` forces re-analysis of all segments
- Missing `passes_run` `"transcription"` → exit 2 with pointer to `transcribe.py`
- Audio file missing → exit 2

Stub analyzer returns deterministic prosody dicts so orchestration tests don't depend on librosa or real audio.

Manual smoke (not a test): real `prosody.py` against `Voice 001 short.wav.diarization.json` + the WAV → eyeball the JSON additions and confirm `prosody_baselines` looks sane.

Expected test count: 218 (current) → ~237 (~19 new across 3 test files).

## Dependencies

Add to `requirements.txt`:

```
librosa>=0.10.0
```

Transitive: `numba`, `audioread`, `pooch`. `scipy`, `soundfile`, `numpy` already present. Cold-install cost ~30 MB. No model downloads.

## Out of scope (deferred)

- **Phase 3 metrics integration** — Markdown AAR report doesn't surface prosody for v1. Easy follow-up once real prosody output exists
- **Voice quality (jitter, shimmer)** — Tier 4 features intentionally skipped; add later if Phase 2C or other consumers want them
- **Cross-session baseline tracking** — each session computes its own baselines; multi-session aggregation is the cross-session pass's job
- **VAD-based pause_ratio** — using word-timestamp gaps is approximate. A real Silero-VAD pass over segment audio would be more accurate. Defer; word-gap version is good enough for relative comparison
- **Per-frame contour output** — summary stats only, not full f0/energy time series. Consumers that want it rerun pyin from audio
- **Speaker-relative z-scores per segment** — consumers compute these from raw + baseline if needed
- **Parallel per-segment analysis** — sequential v1; parallelize later if profiling shows it's needed
- **CLI overrides for `pitch_min_hz` / `pitch_max_hz`** — config-only for v1; non-breaking to add later

## Migration path

Purely additive. Phase 3 metrics doesn't read `prosody` data, so existing metrics output is unchanged. Old JSONs without `prosody` continue to work everywhere downstream. The `core/audio/load.py` extraction is a no-op refactor: `diarize.py` keeps the same audio-loading semantics, just via a different import path.

## Open questions

None. All decisions resolved during brainstorming 2026-05-16.
