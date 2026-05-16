# Contribution Metrics Pass (Phase 3) — Design

**Date:** 2026-05-16
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-15-transcription-pass-design.md`](./2026-05-15-transcription-pass-design.md) (Phase 2A, shipped) and [`2026-05-15-sentiment-pass-design.md`](./2026-05-15-sentiment-pass-design.md) (Phase 2B, shipped)

## Purpose

Add `metrics.py`, the third analysis pass in the planned phase-wise series toward facilitated-classroom-discussion AAR debrief. Reads a post-2A+2B diarization JSON and produces per-speaker + session-level contribution aggregates, a 5-minute-bucketed activity timeline, and a small set of deterministic narrative highlights. Writes the aggregate block back to the JSON in-place atomically *and* renders a human-readable Markdown report next to it.

This is the first pass with a direct facilitator-readable artifact. It answers the AAR questions the earlier specs pointed at — "who talked how much?", "where was engagement highest?", "where were the disagreements?" — without any LLM dependency, using only the text, word timestamps, and sentiment fields already produced by 2A and 2B.

Phase 2C (LLM-driven engagement labels) was deferred ahead of this pass because the local-vs-API backend question was unresolved; Phase 3 closes the loop on existing local-only signal and can ship today.

## Architecture

`metrics.py` is a stand-alone CLI that reads a diarization JSON, validates that prior passes have run, computes pure-Python aggregates over the segment list, builds a top-level `contribution_metrics` block, writes the JSON back atomically via the same temp-file + atomic-rename pattern `transcribe.py` and `sentiment.py` use, then renders a Markdown report to a sibling file.

```
session.diarization.json (with text + sentiment per segment)
            │
            ▼
[load + validate: passes_run ⊇ {"transcription","sentiment"}]
            │
            ▼
[aggregators run in fixed order:
   participation → sentiment_aggregation → turn_taking →
   pairwise_followers → timeline_buckets → highlights]
            │
            ▼
[build top-level `contribution_metrics` block]
            │
            ▼
[atomic JSON write back to original path (or --out)]
            │
            ▼
[render Markdown report → <input>.metrics.md]
```

All aggregation is pure Python over the existing JSON. No new model downloads. No GPU. No network. Sub-second on session-length inputs.

## Output schema additions

**Top-level additions** — one config block and one metrics block, plus an entry in `passes_run`:

```json
"passes_run": ["diarization", "transcription", "sentiment", "metrics"],
"metrics_config": {
  "bucket_seconds": 300,
  "top_k_highlights": 5,
  "quote_max_chars": 100,
  "analyzed_at": "2026-05-16T18:42:01Z"
},
"contribution_metrics": {
  "session": {
    "duration_s": 90.0,
    "speech_duration_s": 78.4,
    "silence_duration_s": 11.6,
    "total_segments": 9,
    "total_words": 312,
    "unique_speakers": 2,
    "identified_speakers": 2,
    "unknown_segments": 0,
    "polarity_distribution": {"positive": 2, "neutral": 7, "negative": 0},
    "emotion_distribution": {"joy": 1, "neutral": 5, "surprise": 1, "disgust": 2,
                             "anger": 0, "fear": 0, "sadness": 0}
  },
  "speakers": [
    {
      "speaker_id": "session_speaker_a",
      "speaker": "Speaker A",
      "participation": {
        "talk_seconds": 52.1,
        "talk_percent": 66.5,
        "segment_count": 5,
        "word_count": 198,
        "words_per_minute": 228.0,
        "mean_segment_seconds": 10.42,
        "median_segment_seconds": 8.7,
        "max_segment_seconds": 42.57
      },
      "sentiment": {
        "polarity": {
          "counts": {"positive": 1, "neutral": 4, "negative": 0},
          "percent": {"positive": 20.0, "neutral": 80.0, "negative": 0.0},
          "mean_top_confidence": 0.81
        },
        "emotion": {
          "counts": {"joy": 0, "neutral": 3, "surprise": 1, "disgust": 1,
                     "anger": 0, "fear": 0, "sadness": 0},
          "percent": {"joy": 0.0, "neutral": 60.0, "surprise": 20.0,
                      "disgust": 20.0, "anger": 0.0, "fear": 0.0, "sadness": 0.0},
          "mean_top_confidence": 0.62
        }
      },
      "turn_taking": {
        "turn_count": 4,
        "mean_gap_before_seconds": 1.42,
        "interruption_count": 1
      }
    }
  ],
  "pairwise_followers": {
    "session_speaker_a": {"session_speaker_b": 3, "unknown": 0},
    "session_speaker_b": {"session_speaker_a": 4, "unknown": 0}
  },
  "timeline": [
    {
      "bucket_start_s": 0,
      "bucket_end_s": 300,
      "per_speaker_talk_s": {"session_speaker_a": 42.6, "session_speaker_b": 35.8},
      "per_speaker_polarity_mode": {"session_speaker_a": "neutral",
                                    "session_speaker_b": "neutral"},
      "per_speaker_emotion_mode": {"session_speaker_a": "disgust",
                                   "session_speaker_b": "neutral"}
    }
  ],
  "highlights": [
    {
      "kind": "longest_segment",
      "speaker_id": "session_speaker_a",
      "start": 0.13, "end": 42.57,
      "value_s": 42.44,
      "quote": "The whole idea is that both of them should be able to understand..."
    },
    {
      "kind": "most_positive",
      "speaker_id": "session_speaker_b",
      "start": 65.2, "end": 71.8,
      "polarity_score": 0.94,
      "quote": "Yes, that's a great point about the radar pairing."
    },
    {
      "kind": "high_disgust_window",
      "bucket_start_s": 0,
      "bucket_end_s": 300,
      "speaker_id": "session_speaker_a",
      "count": 2
    }
  ]
}
```

**Field-level notes:**

- `speakers` is a list ordered by first-appearance start time (mirrors `enrolled_users_matched` convention).
- `pairwise_followers[A][B] = N` means "speaker B started a segment immediately after speaker A's segment N times." Self-transitions are excluded (same-speaker consecutive segments are collapsed into one turn for turn-taking purposes anyway).
- `polarity_mode` / `emotion_mode` in timeline buckets = the most-frequent label in that window for that speaker. `null` if the speaker didn't talk in that window.
- All seconds are floats with up to 2 decimals. Percentages are 0–100 floats rounded to 1 decimal.
- Highlight `quote` text is truncated to ≤ `quote_max_chars` (default 100) with trailing `...` when truncation actually happens.
- Top-k highlights cap (default 5) applies across all kinds combined.

**Per-segment fields are not modified.** Phase 3 is additive at the top level only.

**`passes_run` dedupe:** rerun appends `"metrics"` via membership check (no duplicates). `metrics_config` and `contribution_metrics` are overwritten on each run.

## Markdown report shape

`<input-stem>.metrics.md` is written next to the JSON. Deterministic, no LLM. The renderer takes the `contribution_metrics` block and the session header fields and produces a single Markdown file designed to be read top-to-bottom by a facilitator. GitHub-flavored tables render correctly in VSCode preview, GitHub, IDE Markdown previews.

```markdown
# Session Metrics — Voice 001 short.wav

**Duration:** 90.0 s (1 min 30 s) · **Speech:** 78.4 s · **Silence:** 11.6 s
**Speakers:** 2 (2 identified, 0 unknown) · **Words:** 312 · **Segments:** 9
**Analyzed:** 2026-05-16T18:42:01Z

## Notable moments
- **Longest contribution** — Speaker A, 42.4 s at 00:00: *"The whole idea is that both of them should be able to understand..."*
- **Most positive** — Speaker B at 01:05 (score 0.94): *"Yes, that's a great point about the radar pairing."*
- **High disgust window** — 00:00–05:00, mostly Speaker A (2 segments)

## Participation

| Speaker    | Talk  | %     | Segs | Words | WPM   | Mean seg | Max seg |
|------------|------:|------:|-----:|------:|------:|---------:|--------:|
| Speaker A  | 52.1s | 66.5% | 5    | 198   | 228.0 | 10.4 s   | 42.6 s  |
| Speaker B  | 26.3s | 33.5% | 4    | 114   | 260.0 | 6.6 s    | 9.1 s   |

## Sentiment — polarity (per speaker)

| Speaker    | Positive | Neutral | Negative | Mean conf. |
|------------|---------:|--------:|---------:|-----------:|
| Speaker A  | 20%      | 80%     | 0%       | 0.81       |
| Speaker B  | 25%      | 75%     | 0%       | 0.79       |

## Sentiment — emotion (per speaker)

| Speaker    | Joy | Neutral | Surprise | Disgust* | Anger | Fear | Sadness | Mean conf. |
|------------|----:|--------:|---------:|---------:|------:|-----:|--------:|-----------:|
| Speaker A  |  0% |    60%  |     20%  |     20%  |   0%  |  0%  |    0%   | 0.62       |
| Speaker B  | 25% |    50%  |      0%  |     25%  |   0%  |  0%  |    0%   | 0.58       |

\* "Disgust" from the emotion model tends to fire on polite-disagreement phrasing — read as "registered disagreement" rather than visceral disgust.

## Turn-taking

| Speaker    | Turns | Mean gap before | Interruptions |
|------------|------:|----------------:|--------------:|
| Speaker A  | 4     | 1.42 s          | 1             |
| Speaker B  | 4     | 0.85 s          | 0             |

## Who follows whom

Rows = previous speaker, columns = next speaker. Cell = transition count.

|              | → Speaker A | → Speaker B | → unknown |
|--------------|------------:|------------:|----------:|
| Speaker A →  |       —     |      3      |     0     |
| Speaker B →  |       4     |      —      |     0     |

## Activity over time (5-min windows)

(monospace block-character chart, one row per speaker, one cell per bucket)

---
_Caveat: 'unknown' segments may represent multiple physical speakers; the diarization layer collapses all unenrolled clusters into one bucket._
```

**Renderer notes:**

- Timestamps shown as `MM:SS` for readability.
- The disgust caveat is hard-coded in the renderer — it's the most actionable known calibration note from 2B.
- Activity bar chart: one row per speaker, one cell per timeline bucket. Each cell is a single Unicode block-character whose density reflects that speaker's talk-seconds in that bucket. Specific glyph thresholds are a renderer implementation detail and are pinned by a golden-file test (a representative mapping: ` ` = 0 s, `░` = (0, ¼·bucket], `▒` = (¼, ½], `▓` = (½, ¾], `█` = (¾, bucket]).
- The output file is always written UTF-8, so block characters render correctly regardless of host shell encoding.
- Empty optional sections are omitted entirely:
  - No highlights → `Notable moments` section absent
  - Single speaker → `Who follows whom` section absent
  - Single timeline bucket → `Activity over time` section absent

## Components

| File | Status | Responsibility |
|---|---|---|
| `target-vad/metrics.py` | create | CLI entry: arg parsing, JSON load/validate, orchestrate aggregators, atomic JSON write, Markdown render |
| `target-vad/modes/metrics/__init__.py` | create | empty package marker |
| `target-vad/modes/metrics/aggregator.py` | create | Pure functions: `aggregate_participation`, `aggregate_sentiment`, `aggregate_turn_taking`, `aggregate_pairwise`, `aggregate_timeline`, `select_highlights`. Each takes the segment list (plus the session header where needed) and returns a plain dict. |
| `target-vad/modes/metrics/renderer.py` | create | `render_markdown(metrics_block, session_meta) -> str`. Deterministic, no LLM. |
| `target-vad/config.yaml` | modify | add `metrics:` block (bucket_seconds, top_k_highlights, quote_max_chars) |
| `target-vad/tests/metrics/__init__.py` | create | empty |
| `target-vad/tests/metrics/test_aggregator.py` | create | per-aggregator unit tests |
| `target-vad/tests/metrics/test_renderer.py` | create | golden-file tests for Markdown shape |
| `target-vad/tests/metrics/test_orchestration.py` | create | CLI behavior (atomic write, error paths, idempotence) |

The aggregator functions are pure: same input segment list → identical output dict. The renderer is pure: same input metrics block → identical output Markdown. This makes both trivial to test against golden fixtures and reproducible on rerun.

## CLI

```
py -3.14 metrics.py <diarization.json> [--out <json-path>] [--report <md-path>] [--config <path>]
```

- Positional `<diarization.json>` — required, must have `passes_run` ⊇ `{"transcription", "sentiment"}`.
- `--out`: where to write the enriched JSON. Defaults to in-place atomic write.
- `--report`: where to write the Markdown. Defaults to `<input-stem>.metrics.md` next to the JSON.
- `--config`: path to `config.yaml`. Default `./config.yaml`.

**No `--rerun` flag** — rerun always overwrites the metrics block + Markdown unconditionally. Aggregation is pure and cheap; there is no equivalent of 2B's "skip already-classified segments" optimization to bypass.

**No `--audio`** — Phase 3 is text/timing only.

**Console output on success:**

```
Metrics written: 2 speakers, 9 segments, 312 words, 3 highlights.
  JSON     → /path/to/session.diarization.json
  Markdown → /path/to/session.diarization.metrics.md
```

## Configuration

Add to `config.yaml`:

```yaml
metrics:
  bucket_seconds: 300         # 5-min activity buckets
  top_k_highlights: 5         # cap on Notable moments items
  quote_max_chars: 100        # quote truncation length in highlights
```

All three knobs are config-only for Phase 3. CLI overrides can be added non-breakingly later.

## Aggregator behavior

### participation

For each speaker:
- `talk_seconds` = sum of `(end - start)` over the speaker's segments
- `talk_percent` = 100 × `talk_seconds` / `session.speech_duration_s`
- `segment_count` = count of segments
- `word_count` = total length of `words` arrays across the speaker's segments (with the `words` field guaranteed by the 2A prerequisite)
- `words_per_minute` = 60 × `word_count` / `talk_seconds` (None if `talk_seconds == 0`)
- `mean_segment_seconds`, `median_segment_seconds`, `max_segment_seconds` = standard statistics over per-segment durations

### sentiment

For each speaker, build `polarity.counts` and `emotion.counts` by tallying `segment.sentiment.polarity.label` and `segment.sentiment.emotion.label`. Compute percentages as `100 × count / total_classified_segments_for_this_speaker`. `mean_top_confidence` = mean of `segment.sentiment.<class>.score` across the speaker's classified segments.

Segments with `sentiment: null` (transcription-failed or empty-text) are skipped — they count toward `segment_count` in participation but are excluded from sentiment denominators.

### turn_taking

A *turn* = a contiguous run of segments by the same speaker. Two adjacent segments by the same speaker collapse into one turn even with a small inter-segment gap. Turn boundary = previous segment's speaker differs from this one's.

- `turn_count` per speaker = number of turn-onsets
- `mean_gap_before_seconds` = mean of `(turn.start - previous_turn.end)` across this speaker's turns; the first turn in the session is excluded
- `interruption_count` per speaker = count of turn-onsets where `turn.start < previous_turn.end` (overlap with the prior turn's end)

### pairwise_followers

For each adjacent pair of turns `(prev, next)` where `prev.speaker_id != next.speaker_id`, increment `pairwise[prev.speaker_id][next.speaker_id]`. The full matrix is rectangular: every observed speaker is both a row and a column, including `unknown`. Diagonal entries are always 0.

### timeline

Buckets are aligned to absolute session time: `[0, bucket), [bucket, 2×bucket), ...`. For each bucket:
- `per_speaker_talk_s[id]` = sum of segment-overlap-with-bucket per speaker. A segment that straddles bucket boundaries contributes proportionally to each bucket it overlaps.
- `per_speaker_polarity_mode[id]` = most-frequent polarity label among the speaker's segments *whose start falls within the bucket* (segments that straddle bucket boundaries are counted once, in the bucket containing their start). `null` if no segments.
- `per_speaker_emotion_mode[id]` = same as polarity, for emotion.

Empty buckets (no talk by anyone) are still emitted to keep the timeline a complete grid. Empty per-speaker entries within a bucket are omitted from the per-speaker dicts.

### highlights — selection rules

Selected deterministically. Top-k cap applied across all kinds combined. Rules in priority order:

| Kind | Selection rule | Skip if |
|---|---|---|
| `longest_segment` | Segment with max `(end - start)` across all segments | none |
| `most_positive` | Segment with max `sentiment.polarity.scores.positive`, and label is `"positive"` | no positive segments anywhere |
| `most_negative` | Segment with max `sentiment.polarity.scores.negative`, and label is `"negative"` | no negative segments anywhere |
| `high_disgust_window` | Bucket with max disgust-segment count; reports the dominant speaker in that bucket | zero disgust segments in session |
| `quietest_window` | Bucket with min `total_talk_s`; ties broken by earlier window | only one bucket exists |
| `busiest_window` | Bucket with max `total_talk_s`; ties broken by earlier window | only one bucket exists |
| `solo_dominator` | Bucket where one speaker has ≥ 80% of talk time and total talk ≥ 60 s | no bucket meets the bar |

All ties broken by earliest start time then alphabetical `speaker_id`. Rerun reproduces the same highlights exactly.

## Unknown-speaker handling

`speaker_id: "unknown"` is treated as a single regular bucket in all aggregations (participation, sentiment, turn-taking, pairwise, timeline, highlights). The Markdown footer includes a caveat: "'unknown' may represent multiple physical speakers; the diarization layer collapses unenrolled clusters."

Preserving per-cluster identity for unknown speakers is an upstream S1 schema upgrade (the current diarization output schema collapses all unidentified clusters to one literal `"unknown"`), not Phase 3's concern.

## Conflict resolution / edge cases

| Case | Behavior |
|---|---|
| Rerun on already-metricized JSON | `contribution_metrics`, `metrics_config`, Markdown all overwritten. `passes_run` deduped. |
| Segment with `sentiment: null` | Counted in participation/turn-taking; skipped in sentiment aggregation. |
| Segment with empty `words` array but non-empty `text` | Counted in participation; `word_count` reflects whatever was emitted by 2A. |
| Segment crossing a timeline bucket boundary | Talk-time apportioned proportionally; emotion/polarity mode credited to the bucket containing the start. |
| Only one timeline bucket (session shorter than `bucket_seconds`) | `quietest_window`/`busiest_window`/`solo_dominator` highlights skip. Activity bar chart section omitted from Markdown. |
| Single speaker session | `pairwise_followers` is `{id: {}}`. "Who follows whom" Markdown section omitted. |
| All segments are `unknown` | Treated as a single-speaker session with id `"unknown"`. Identification ratios reflect this honestly. |
| Whole-pipeline crash mid-run | Atomic write means original JSON is unmodified. User reruns. |

## Error handling table

| Failure | Exit code |
|---|---|
| JSON not found / unreadable | 2 |
| JSON malformed (decode error) | 2 |
| JSON missing `segments` field | 2 |
| `passes_run` missing `"transcription"` | 2 (with pointer to `transcribe.py`) |
| `passes_run` missing `"sentiment"` | 2 (with pointer to `sentiment.py`) |
| Any segment missing `text`, `words`, or `sentiment` field (where `sentiment: null` is the explicit-null sentinel and is allowed) | 2 (with segment index) |
| `metrics` config block missing from config.yaml | 3 |
| Atomic JSON write failure | 3 |
| Markdown write failure | 3 |

Same categorization as `transcribe.py` and `sentiment.py`: 2 for user-supplied bad input, 3 for environment/IO failures.

## Testing approach

`tests/metrics/test_aggregator.py` (~12 tests, pure functions on synthetic segment lists):

- `aggregate_participation`: talk_seconds, talk_percent, segment_count, word_count, WPM, mean/median/max segment length on a 3-speaker fixture
- `aggregate_sentiment`: per-speaker polarity + emotion counts/percents, mean top confidence
- `aggregate_sentiment` skips `sentiment: null` segments (treats as no signal, not as zero)
- `aggregate_turn_taking`: consecutive same-speaker segments collapse into one turn; gap before turn; interruption when `start < previous.end`
- `aggregate_pairwise`: who-follows-whom matrix omits self-transitions, includes `unknown` row/column when present
- `aggregate_timeline`: bucket boundaries, segments straddling boundaries split proportionally for talk-time
- `aggregate_timeline`: empty bucket → speaker entry omitted (not zeroed)
- `select_highlights`: each kind triggers under its rule and is skipped under its skip condition
- `select_highlights`: respects `top_k_highlights` cap
- `select_highlights`: deterministic tie-breaking (earliest start, then alphabetical id)
- Quote truncation at `quote_max_chars` adds trailing `...` only when truncation occurred
- Unknown speaker handled as a regular bucket (talk_seconds counted, included in pairwise)

`tests/metrics/test_renderer.py` (~5 tests, golden-file Markdown):

- Full report given a 2-speaker metrics block matches a golden fixture
- Empty `highlights` array → `Notable moments` section omitted entirely
- Single-speaker session → `Who follows whom` section omitted
- Activity bar chart maps talk-seconds → block characters with the documented threshold
- Disgust footnote is always present when the emotion table renders

`tests/metrics/test_orchestration.py` (~8 tests, CLI):

- Happy path: JSON + Markdown both written, JSON gains `contribution_metrics` + `metrics_config` + `passes_run` += `"metrics"`
- `--out` path: input JSON untouched, output JSON has the block
- `--report` path: Markdown written to specified path
- Idempotent rerun: `contribution_metrics` overwritten, `passes_run` deduped
- Missing `passes_run` `"transcription"` → exit 2 with pointer to `transcribe.py`
- Missing `passes_run` `"sentiment"` → exit 2 with pointer to `sentiment.py`
- Segment missing `text`/`words`/`sentiment` (not the null sentinel) → exit 2 with segment index
- Missing `metrics` config block → exit 3

**Manual smoke** (not a separate test — run after merge): real `metrics.py` against `Voice 001 short.wav.diarization.json` (which already has all 2A+2B output). Eyeball the JSON `contribution_metrics` block and the rendered Markdown.

Expected test count: 172 (current) → ~197 (~25 new across 3 test files).

## Dependencies

None new. All aggregation is pure Python over the existing JSON. Renderer uses only stdlib `str.format`.

## Out of scope (deferred)

- **Phase 2C engagement-oriented LLM labels** — separate spec, deferred until local-vs-API backend decision is made
- **HTML or PDF report renderers** — Markdown is the only target for Phase 3
- **Cross-session comparison** — would require a session-store; not yet needed
- **Per-topic aggregation** — no topic-segmentation pass exists yet
- **LLM-generated narrative highlights** — Phase 2C territory
- **Streaming aggregation** — one-shot pass is fast enough for any session length
- **Preserving per-cluster identity for unknown segments** — upstream S1 schema upgrade
- **CLI overrides for `bucket_seconds` / `top_k_highlights`** — config-only for Phase 3; non-breaking to add later
- **Aggregation against arbitrary `--audio`** — Phase 3 is JSON-only

## Migration path

No migration needed. Phase 3 only adds top-level `contribution_metrics` + `metrics_config` blocks and one entry in `passes_run`. Existing diarization JSONs from Phase 2B remain valid input. JSONs that haven't been transcribed or sentiment-classified yet are rejected with a clear exit-2 message.

## Open questions

None. All decisions resolved during brainstorming 2026-05-16.
