# Sentiment Pass (Phase 2B) — Design

**Date:** 2026-05-15
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-15-transcription-pass-design.md`](./2026-05-15-transcription-pass-design.md) (Phase 2A, shipped)

## Purpose

Add `sentiment.py`, the second analysis pass in the planned phase-wise series toward facilitated-classroom-discussion AAR debrief. Reads a post-transcription diarization JSON, classifies each segment's `text` with both a polarity model (3-class: positive/neutral/negative) and an emotion model (7-class: joy/sadness/anger/fear/surprise/disgust/neutral), and writes the per-segment `sentiment` block back atomically. Validates the analysis-pass pattern with a second pass and produces signals that feed directly into Phase 3 user-contribution metrics ("who got frustrated when?", "where was engagement highest?").

Two models in one pass is a deliberate choice: both are cheap text classifiers running on CPU, both feed the AAR, and bundling them avoids two separate model-download passes and two separate JSON read/write cycles. Phase C (engagement-oriented LLM labels) is reserved as a sibling `sentiment.engagement` field for a later phase.

## Architecture

`sentiment.py` is a stand-alone CLI that reads a diarization JSON (post-transcription required), runs HuggingFace transformer classifiers in batches on each segment's `text`, attaches a nested `sentiment.{polarity, emotion}` block per segment, updates top-level `passes_run` and `sentiment_config`, and writes the JSON back atomically via the same temp-file + atomic-rename pattern `transcribe.py` uses.

```
session.diarization.json (with text per segment)
            │
            ▼
[load JSON, validate it has segments with text fields]
            │
            ▼
[load polarity model + emotion model once (via transformers.pipeline)]
            │
            ▼
[for each segment in segments:
   - if text is None or empty: assign sentiment: null, skip inference
   - if segment already has sentiment and --rerun not set: skip
   - else: queue text for batch inference]
            │
            ▼
[batched inference (polarity model first, then emotion model)]
            │
            ▼
[attach sentiment block per queued segment]
            │
            ▼
[update top-level passes_run + sentiment_config]
            │
            ▼
[atomic write back to original path (or --out)]
```

Both models run on CPU and download to the HuggingFace cache on first use (~500 MB polarity + ~330 MB emotion = ~830 MB combined cold-cache cost). Subsequent runs use the cache. Inference is batched per the `sentiment.batch_size` config knob (default 16).

## Output schema additions

**Per segment** — adds one nested field, leaves existing ones untouched:

```json
{
  "start": 0.42, "end": 3.81,
  "speaker_id": "siddharth", "speaker": "Siddharth Jain",
  "text": "...", "words": [...],
  "sentiment": {
    "polarity": {
      "label": "neutral",
      "score": 0.72,
      "scores": {"positive": 0.18, "neutral": 0.72, "negative": 0.10}
    },
    "emotion": {
      "label": "surprise",
      "score": 0.51,
      "scores": {
        "joy": 0.12, "surprise": 0.51, "anger": 0.03, "fear": 0.04,
        "sadness": 0.05, "disgust": 0.02, "neutral": 0.23
      }
    }
  }
}
```

- `polarity.label` ∈ {`positive`, `neutral`, `negative`}. `polarity.scores` keys are exactly those three.
- `emotion.label` ∈ {`joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `neutral`}. `emotion.scores` keys are exactly those seven.
- `label` is always the argmax key of `scores`. `score` is `scores[label]`. Redundant but ergonomic for consumers that only want the top class.
- Each `scores` dict sums to ~1.0 (softmax output; small rounding error tolerated).

**Sentinel:** `sentiment: null` (not a missing key, an explicit null) when the segment's `text` is `null` (transcription failed) or `""` (silent segment). Consumers can distinguish "didn't run" (no `sentiment` key at all — possible if Phase 2B hasn't been applied yet) from "ran successfully and there was nothing to classify" (`sentiment: null`).

**Top-level additions:**

```json
"passes_run": ["diarization", "transcription", "sentiment"],
"sentiment_config": {
  "polarity_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
  "device": "cpu",
  "batch_size": 16,
  "analyzed_at": "2026-05-15T15:42:01Z"
}
```

`passes_run` is appended (not replaced); dedupe by membership check so a rerun doesn't grow the list. `sentiment_config` is overwritten on each run.

## Components

| File | Status | Responsibility |
|---|---|---|
| `target-vad/sentiment.py` | create | CLI entry: arg parsing, JSON load/save, eager classifier load, batched orchestration |
| `target-vad/modes/sentiment/__init__.py` | create | empty package marker |
| `target-vad/modes/sentiment/classifier.py` | create | `SentimentClassifier` — lazy-loads two HF pipelines, exposes `classify_batch(texts) -> List[Dict]` returning per-text nested polarity+emotion blocks |
| `target-vad/config.yaml` | modify | add `sentiment:` block |
| `target-vad/requirements.txt` | modify | add `transformers>=4.40.0` |
| `target-vad/tests/sentiment/__init__.py` | create | empty |
| `target-vad/tests/sentiment/test_classifier.py` | create | tests with mocked `transformers.pipeline` (no real downloads) |
| `target-vad/tests/sentiment/test_orchestration.py` | create | tests for `sentiment.py main()` with a stub classifier |

`SentimentClassifier` mirrors `WhisperRunner` (Phase 2A) in shape: lazy-load on first `classify_batch()` call; expose a public `load()` for eager loading so download failures surface early via `EXIT_MODEL_OR_IO` rather than getting masked by per-batch exception handlers.

## CLI

```
py -3.14 sentiment.py <diarization.json> [--out <path>] [--rerun] [--config <path>]
```

- Positional: `<diarization.json>` — required, must have `text` field per segment (transcription pass must have run).
- `--out`: where to write the enriched JSON. Defaults to in-place atomic write.
- `--rerun`: re-classify segments that already have a `sentiment` field. Default is to skip those (incremental processing).
- `--config`: path to config.yaml. Default `./config.yaml`.

No `--audio` (sentiment is text-only). No `--model` overrides at the CLI for now — model swapping is config-only. Adding CLI model overrides later is non-breaking.

**Pre-flight validation:** if any segment lacks a `text` field at all (not just `null` or `""`), `sentiment.py` exits 2 with a clear message pointing to `transcribe.py`. Mixed states (some segments have `text`, others don't) are also exit 2 — the JSON should be in a coherent post-transcription state.

The CLI prints a rich progress bar `Classifying [bar] N/M`. On success, prints a summary line: `Classified N segments, skipped M (null/empty text), reused R (already had sentiment) (polarity=..., emotion=...)`.

## Configuration

Add to `config.yaml`:

```yaml
sentiment:
  polarity_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  emotion_model: "j-hartmann/emotion-english-distilroberta-base"
  device: "cpu"
  batch_size: 16
```

Any string accepted by `transformers.pipeline()` is valid as a model name: HuggingFace model identifiers, local paths to a saved model dir, etc. Future model swaps (domain-tuned classifiers, distilled variants for speed, multilingual versions) happen via config alone — no code change.

## Model expectations

`SentimentClassifier` uses `transformers.pipeline("text-classification", model=..., return_all_scores=True, device=...)` for both models. This returns a list of `[{label, score}, ...]` per input. The classifier wraps that into the spec's nested shape:

- Sorts/maps the polarity labels to the canonical `positive`/`neutral`/`negative` keys (handles models that emit `LABEL_0`/`LABEL_1`/`LABEL_2` by reading the model's `id2label` mapping).
- Sorts/maps the emotion labels to the canonical 7-class keys.
- If a config model emits labels outside the canonical set (e.g., a future engagement-oriented model), the classifier raises a clear error at first inference: "polarity model emitted label 'X' which is not in the canonical set; use `engagement_model` config in Phase C instead." This prevents silent schema drift.

## Conflict resolution / edge cases

| Case | Behavior |
|---|---|
| Re-run on already-classified JSON, no `--rerun` | Segments with non-null `sentiment` are skipped; reported in summary as `reused R`. |
| Re-run with `--rerun` | All segments with `text` are re-classified from scratch. |
| Segment `text: null` (transcription failed) | `sentiment: null` (sentinel — distinguishes from "didn't run"). Not counted as "classified". |
| Segment `text: ""` (whisper returned empty) | `sentiment: null`. Same as above. |
| Segment text is suspicious shape (e.g., not a string) | Exit 2 with the offending segment index. |
| Model emits non-canonical labels | Exit 3 with the offending label and a suggested config fix. |
| Model download/load fails | Exit 3 via `SentimentClassifier.load()` (eager, before the loop). |
| Per-batch classifier crash | Print `[yellow]warning[/]`, set `sentiment: null` for affected segments, continue. |
| Whole-pipeline crash mid-run | Atomic write means original JSON is unmodified. User reruns; incremental mode resumes. |
| JSON missing `segments` field | Exit 2 with schema-version error. |

## Error handling table

| Failure | Exit code |
|---|---|
| JSON not found / unreadable | 2 |
| JSON malformed (decode error) | 2 |
| JSON missing required fields (`segments`) | 2 |
| Segments lack `text` field (transcription not run) | 2 |
| Sentiment config block missing | 3 |
| Model download / load fails | 3 |
| Per-batch classifier crash | 0 (warning + null marker, continue) |
| Atomic write failure | 3 |
| Model emits non-canonical labels | 3 |

Same categorization as `transcribe.py`: 2 for user-supplied bad input, 3 for environment/model/io failures.

## Testing approach

`tests/sentiment/test_classifier.py` (~5 tests, mocked transformers.pipeline):
- `classify_batch(texts)` returns one nested block per text in input order
- Lazy load — model isn't constructed in `__init__`
- `load()` triggers eager construction; second `load()` is a no-op
- Canonical label mapping: when the pipeline returns `LABEL_0`/`LABEL_1`/`LABEL_2`, the classifier maps via `id2label` to `positive`/`neutral`/`negative`
- Non-canonical label raises a clear error (specific message)

`tests/sentiment/test_orchestration.py` (~10 tests, stub classifier):
- Fresh classification attaches `sentiment` to all segments with non-empty text
- `text: null` segments get `sentiment: null` (sentinel)
- `text: ""` segments get `sentiment: null`
- `--rerun` re-classifies; default skips already-classified
- `passes_run` gains `"sentiment"` (dedupe on rerun)
- `sentiment_config` block has all 5 fields
- Missing `text` field on any segment → exit 2
- Atomic write — `--out` path doesn't modify the input
- Malformed JSON → exit 2
- Missing `segments` field → exit 2
- Model load failure → exit 3 (via eager `load()` call)

Manual smoke (not a separate task — run after merge):
- Real `sentiment.py` against `Voice 001 short.wav.diarization.json` (which already has text + words from Phase 2A)
- Verify the model downloads happen on cold cache, classification finishes reasonably fast, and the JSON gains coherent `sentiment` blocks

Expected test count: 151 (current) → ~166 (~15 new across 2 test files).

## Dependencies

Add to `requirements.txt`:

```
transformers>=4.40.0
```

Transitive: `tokenizers`, `huggingface_hub` (already present from faster-whisper). `torch` already present.

Combined first-run model download: ~830 MB. Subsequent runs use the HF cache.

## Forward-compatibility for Phase C (engagement labels)

The `sentiment.{polarity, emotion}` nested shape leaves room for `sentiment.engagement` as a sibling field — an LLM-based pass that adds engagement-oriented labels (engaged, curious, hesitant, frustrated, dismissive, etc.). Phase C will:

- Read the JSON (already has polarity + emotion from Phase 2B)
- For each segment with `text`, prompt an LLM (via the Anthropic SDK, with prompt caching) for an engagement label
- Add `sentiment.engagement: {label, score, scores}` per segment
- Append `"engagement"` to `passes_run`, add `engagement_config` top-level block

No changes to Phase 2B's output shape are required for Phase C to slot in cleanly.

## Out of scope (deferred)

- **Phase C engagement-oriented LLM labels** — sibling field, separate pass, separate spec
- **Audio-based affect** (prosody, energy, speaking rate) — Phase 5+ if text-based sentiment misses important signals
- **Per-speaker sentiment aggregation** (mean sentiment per user across the session) — Phase 3 user-contribution-metrics responsibility
- **Multilingual sentiment** — both default models are English-only; matches the transcription pass's `language: "en"` default. Future passes can add language-conditional model selection
- **Sliding-context inference** — each segment classified independently; cross-segment context is not used (the rolling-context pattern from Phase 2A doesn't apply because text classifiers don't use it productively)
- **`--polarity-model` / `--emotion-model` CLI overrides** — config-only for Phase 2B; CLI overrides can be added non-breakingly later
- **Confidence thresholding** (drop low-confidence labels) — out of scope; consumers can filter on `score` themselves
- **Discrete affect categorization beyond the 7 emotion classes** — out of scope; future passes can layer richer schemes

## Migration path

No migration needed. Phase 2B only adds the `sentiment` per-segment field and the top-level `sentiment_config` / `passes_run` update. Existing diarization JSONs from Phase 2A remain valid input. JSONs that haven't been transcribed yet are rejected with a clear exit-2 message pointing to `transcribe.py`.

## Open questions

None. All decisions resolved during brainstorming 2026-05-15.
