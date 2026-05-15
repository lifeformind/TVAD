# Transcription Pass (Phase 2A) — Design

**Date:** 2026-05-15
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-14-classroom-diarization-design.md`](./2026-05-14-classroom-diarization-design.md) (S1, shipped), [`2026-05-15-in-session-enrollment-design.md`](./2026-05-15-in-session-enrollment-design.md) (in-session enrollment, shipped)

## Purpose

Add `transcribe.py`, the first of a planned series of analysis passes that progressively enrich a diarization JSON with derived information. This pass adds `text` and word-level timestamps to each segment via faster-whisper, leaving room for future passes (sentiment, topics, references, contribution metrics) to layer in without breaking consumers.

The product context is a facilitated in-class structured discussion plus an After-Action Review (AAR) debrief on individual performance and contribution. The transcript is the foundation other analytics build on; this spec covers transcript generation only. Future passes get their own specs.

## Architecture

`transcribe.py` is a stand-alone CLI that reads a diarization JSON (produced by `diarize.py`), pairs it with its source WAV, runs faster-whisper per segment with a rolling-context prompt, and writes the enriched JSON back atomically. No framework abstraction yet — future passes follow the same "read JSON + audio → add fields → write JSON" contract, and the right abstraction will emerge after 2-3 of them exist.

```
session.diarization.json + session.wav
            │
            ▼
[load JSON, validate shape (must have segments, duration_s, audio_file)]
            │
            ▼
[resolve WAV path: --audio override else JSON's audio_file]
            │
            ▼
[load faster-whisper model once (config.transcription.model)]
            │
            ▼
[for each segment in segments:
   - skip if segment already has non-null text and --retranscribe is not set
   - slice WAV[int(start*sr):int(end*sr)]
   - call whisper with initial_prompt=rolling_context (last N chars)
   - extract (text, words) from whisper output
   - append text to rolling context, truncate to N chars
   - attach text + words to segment]
            │
            ▼
[update top-level passes_run + transcription_config]
            │
            ▼
[atomic write back to original path (or --out)]
```

faster-whisper handles model loading, device placement, and word-level timestamps natively. The pipeline does no audio resampling — diarize.py already wrote a 16 kHz mono WAV, and faster-whisper accepts 16 kHz directly.

## Output schema additions

**Per segment** — adds two fields, leaves existing ones untouched:

```json
{
  "start": 0.42,
  "end": 3.81,
  "speaker_id": "siddharth",
  "speaker": "Siddharth Jain",
  "text": "OK class, let's begin.",
  "words": [
    {"start": 0.42, "end": 0.56, "word": "OK", "probability": 0.98},
    {"start": 0.61, "end": 0.99, "word": "class,", "probability": 0.99}
  ]
}
```

- `text`: full segment transcription as a single string. Empty string `""` when whisper returns nothing (silent or sub-100 ms segments).
- `words`: list of `{start, end, word, probability}` objects, in order. Always present, may be empty when text is empty.

**Top-level additions:**

```json
{
  "audio_file": "...",
  "duration_s": 90.0,
  "diarized_at": "...",
  "config": {...},
  "enrolled_users_matched": [...],
  "segments": [...],
  "passes_run": ["diarization", "transcription"],
  "transcription_config": {
    "model": "small",
    "language": "en",
    "initial_prompt_chars": 200,
    "compute_type": "int8",
    "transcribed_at": "2026-05-15T14:33:01Z"
  }
}
```

`passes_run` is the canonical record of which analysis passes have completed. Future passes append themselves and add a sibling `<pass>_config` block. `diarize.py` will be updated to populate `passes_run: ["diarization"]` when it writes a fresh JSON.

## Components

| File | Status | Responsibility |
|---|---|---|
| `target-vad/transcribe.py` | create | CLI entry: arg parsing, JSON load/save, orchestration, rich progress bar |
| `target-vad/modes/transcription/__init__.py` | create | empty package marker |
| `target-vad/modes/transcription/whisper_runner.py` | create | `WhisperRunner` class: lazy-loads faster-whisper, transcribes a single audio slice with an optional initial_prompt, returns `(text, words)` |
| `target-vad/modes/transcription/rolling_context.py` | create | pure helper: maintains the last N chars of running transcript |
| `target-vad/diarize.py` | modify | populate `passes_run: ["diarization"]` in output JSON |
| `target-vad/config.yaml` | modify | add `transcription:` block |
| `target-vad/requirements.txt` | modify | add `faster-whisper>=1.0.0` |
| `target-vad/tests/transcription/__init__.py` | create | empty |
| `target-vad/tests/transcription/test_rolling_context.py` | create | pure unit tests for the rolling-prompt helper |
| `target-vad/tests/transcription/test_whisper_runner.py` | create | runner tests with a mocked faster-whisper Model |
| `target-vad/tests/transcription/test_orchestration.py` | create | tests for the JSON-read-modify-write flow with a stub WhisperRunner |

`WhisperRunner` itself is unit-tested with a mocked `faster_whisper.WhisperModel` (no real model download). The full pipeline is validated by a manual end-to-end smoke run against the existing `Voice 001 short.wav.diarization.json` from S1.

## CLI

```
py -3.14 transcribe.py <diarization.json> [--out <path>] [--audio <wav>]
                                          [--model small|large|<custom>] [--retranscribe]
                                          [--config config.yaml]
```

- Positional: `<diarization.json>` — required.
- `--out`: where to write the enriched JSON. Defaults to in-place atomic write back to `<diarization.json>`.
- `--audio`: override the WAV path. Defaults to the JSON's `audio_file` field.
- `--model`: one-off override of the config's `transcription.model`. Accepts the documented presets (`small`, `large-v3`) or any faster-whisper-compatible model name (HuggingFace path, local CTranslate2 dir).
- `--retranscribe`: re-run transcription on segments that already have a `text` field. Default is to skip those segments (incremental processing).
- `--config`: path to config.yaml. Default `./config.yaml`.

The CLI prints a rich progress bar showing `segment N/M` with the current speaker label. On success, prints a summary line: `Transcribed N segments in MM:SS (model=small, language=en)`. Exit code 0 on success, non-zero with documented messages on failure.

## Configuration

Add to `config.yaml`:

```yaml
transcription:
  model: "small"              # preset shortcut or any faster-whisper-compatible model identifier
  language: "en"              # ISO 639-1 code; set to null to let whisper auto-detect per call
  initial_prompt_chars: 200   # rolling-context window passed to whisper.transcribe(initial_prompt=...)
  device: "cpu"               # "cpu" on this hardware; "cuda" is reserved for future GPU work
  compute_type: "int8"        # int8 quantization is CPU-optimal; "float32" trades ~2x speed for ~1% WER
```

The two preset names `small` and `large` map to faster-whisper's bundled models (resolved internally — `small` → `small`, `large` → `large-v3`). Any other string is passed through to `faster_whisper.WhisperModel(...)` as-is so future models (e.g., `large-v4`, custom CTranslate2 checkpoints) drop in via config without code changes.

## Rolling-context behavior

For each segment after the first, the previous transcript is passed as faster-whisper's `initial_prompt`. Implementation: a pure helper `RollingContext` maintains a single string of at most `initial_prompt_chars` characters; after each segment transcribes, the new `text` is appended and the head is truncated to the limit.

- First segment: empty prompt.
- Subsequent segments: `initial_prompt` = the last N chars of all previously transcribed text, joined with single spaces.
- `--retranscribe` resets the rolling context to empty at the start of the run; partial reruns (incremental) start with the context built from already-transcribed segments earlier in the file.

This gives whisper enough discourse context to disambiguate disfluencies and proper nouns without paying the full price of whole-file transcription. The 200-char default is a tunable knob — too short loses context, too long eats whisper's prompt budget.

## Conflict resolution / edge cases

| Case | Behavior |
|---|---|
| Re-run on already-transcribed JSON, no flag | Segments with non-null `text` are skipped; rolling context still includes their text. Only segments with no `text` are transcribed. |
| Re-run with `--retranscribe` | Rolling context resets; every segment is transcribed from scratch. |
| Segment shorter than ~0.1 s | Whisper handles gracefully; commonly returns empty text. Empty `text: ""` and empty `words: []` attached. Not an error. |
| Segment audio is silent | Whisper may return empty text or hallucinated text (known whisper failure mode). Accept as-is; no silence detection in this pass. |
| `audio_file` in JSON doesn't exist | Exit 2 with message pointing at the missing path and suggesting `--audio <path>`. |
| WAV's duration disagrees with JSON's `duration_s` by > 1 s | Print a warning; proceed anyway (user may have re-encoded with imprecise framing). |
| JSON missing required fields (`segments`, `duration_s`, `audio_file`) | Exit 2 with a schema-version error. |
| Whisper model download fails (network, HuggingFace outage) | Exit 3 with a hint about HF cache and connectivity. |
| Whisper crashes on a single segment | Log a warning; set `text: null` and `words: []` for that segment; continue. Other segments still proceed. `text: null` is the "tried and failed" sentinel; `text: ""` is "ran successfully, nothing transcribed". |
| `"unknown"` speakers | Transcribed normally — text is independent of identification. |
| Whole transcription crashes mid-run | The atomic-write discipline means the original JSON is unmodified. User can re-run; incremental mode picks up where it left off. |

## Error handling table

| Failure | Exit code | Behavior |
|---|---|---|
| JSON not found or unreadable | 2 | message + abort |
| JSON malformed or schema mismatch | 2 | message includes the missing field |
| WAV not found (after `--audio` override) | 2 | message + abort |
| Whisper model download / load fails | 3 | message + abort |
| Per-segment whisper crash | 0 | warning + null marker, continue |
| Atomic write failure (disk full, permission) | 3 | message + abort, original JSON untouched |

## diarize.py change

`diarize.py` already writes the JSON top-level fields. Add a single line so future readers can introspect which passes have run:

```python
payload["passes_run"] = ["diarization"]
```

That's the only `diarize.py` change in this spec. Existing field order in `write_json` may need a small adjustment in `output.py` to slot `passes_run` in the right place; details in the implementation plan.

## Testing approach

- **`tests/transcription/test_rolling_context.py`** (4-6 tests): empty context returns "", append-and-truncate at boundary, multi-segment accumulation, reset.
- **`tests/transcription/test_whisper_runner.py`** (4-6 tests): with a `MagicMock` standing in for `faster_whisper.WhisperModel`, verify:
  - `WhisperRunner.transcribe(audio, initial_prompt="ctx")` calls the underlying model with the right kwargs
  - Returns `(text, [word_dicts])` from the mocked output
  - Empty result is handled
  - Lazy load — model isn't constructed until first transcribe call
- **`tests/transcription/test_orchestration.py`** (4-6 tests): with a stub WhisperRunner, verify:
  - JSON in → JSON out with text and words populated
  - Skip-already-transcribed behavior
  - `--retranscribe` retranscribes everything
  - `passes_run` and `transcription_config` correctly populated
  - Atomic write doesn't leave partial files on a stubbed crash

End-to-end smoke (manual, not CI): run `transcribe.py` against `Voice 001 short.wav.diarization.json` and verify the transcript is sensible. This is the only test path that downloads a real whisper model.

Expected test count: current 120 + ~15 new = ~135.

## Dependencies

Add to `requirements.txt`:

```
faster-whisper>=1.0.0
```

Transitive: `ctranslate2`, `tokenizers`, `huggingface_hub`. Model files (`small` ≈ 244 MB, `large-v3` ≈ 1.5 GB) auto-download to the HuggingFace cache on first use. CPU-only on Strix Halo; `compute_type: "int8"` uses CTranslate2's quantized kernels.

## Migration path

No migration needed. Phase 2A only adds fields; existing diarization JSONs from S1 + in-session enrollment work remain valid input to `transcribe.py`. The `passes_run: ["diarization"]` field is new; existing JSONs without it are treated as if they have it (defensible because the JSON itself proves diarization ran).

## Out of scope

Deferred to future phases per the user's "phase-wise development" direction:

- Sentiment analysis pass
- Topic / reference extraction pass
- Per-user contribution metrics (speaking time, question count, content density) derived from `text`
- Speaker diarization re-segmentation based on transcription gaps
- Multi-language detection within a single recording (auto-detect runs at most once; result applies to all segments in a run)
- Word-level alignment refinement beyond what faster-whisper provides natively
- GUI / playback UI consuming the enriched JSON
- A formal `AnalysisPass` framework — the convention is documented; the abstraction emerges later

## Open questions

None. All decisions resolved during brainstorming 2026-05-15.
