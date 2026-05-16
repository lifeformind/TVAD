# Contribution Metrics Pass (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `metrics.py`, the third analysis pass that reads a post-2A+2B diarization JSON, computes per-speaker + session-level contribution aggregates, a 5-min-bucketed activity timeline, and deterministic narrative highlights, writes the result back into the JSON atomically, and renders a sibling Markdown report.

**Architecture:** Bottom-up TDD. Six pure-function aggregators (`participation`, `sentiment`, `turn_taking`, `pairwise`, `timeline`, `highlights`) live in `modes/metrics/aggregator.py`. A pure Markdown renderer lives in `modes/metrics/renderer.py`. `metrics.py` is the CLI orchestrator that wires them, validates prerequisites, performs an atomic write, and writes the report. No new dependencies — pure Python on top of the existing JSON.

**Tech Stack:** Python 3.14 (`py -3.14`, never `python` — Python 3.12 lacks the dep stack). No new dependencies. Reused existing deps: `pyyaml`, `rich`. Markdown is plain stdlib `str.format`.

**Spec:** [`docs/superpowers/specs/2026-05-16-contribution-metrics-design.md`](../specs/2026-05-16-contribution-metrics-design.md). Read once before starting Task 1.

**Working directory:** `c:\repos\TVAD\target-vad\` for python/pytest. Git commands run from `c:\repos\TVAD\`.

---

## File Structure

Files this plan creates or modifies (relative to `target-vad/`):

| Path | Status | Responsibility |
|---|---|---|
| `config.yaml` | modify | add top-level `metrics:` block |
| `modes/metrics/__init__.py` | create | empty package marker |
| `modes/metrics/aggregator.py` | create | Six pure aggregator functions over segment lists |
| `modes/metrics/renderer.py` | create | `render_markdown(data, metrics_block) -> str` |
| `metrics.py` | create | CLI entry: arg parsing, JSON load + validate, orchestrate aggregators, atomic write, Markdown render |
| `tests/metrics/__init__.py` | create | empty |
| `tests/metrics/test_aggregator.py` | create | per-aggregator unit tests |
| `tests/metrics/test_renderer.py` | create | golden-file Markdown tests |
| `tests/metrics/test_orchestration.py` | create | CLI behavior + error paths |
| `tests/metrics/fixtures/golden_report.md` | create | golden Markdown fixture (committed) |

No modifications to existing diarization, transcription, sentiment, kiosk, or enrollment code — Phase 3 is purely additive.

---

## Task 1: Add `metrics:` config block

**Files:**
- Modify: `target-vad/config.yaml`

- [ ] **Step 1: Inspect existing config.yaml**

Run from `c:\repos\TVAD\target-vad\`:

```bash
py -3.14 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(sorted(c.keys()))"
```

Expected output before this task: `['core', 'diarization', 'kiosk', 'sentiment', 'transcription']`.

- [ ] **Step 2: Append `metrics:` block**

Open `target-vad/config.yaml`. Append to the end (do not touch other blocks):

```yaml

metrics:
  bucket_seconds: 300         # 5-min activity buckets
  top_k_highlights: 5         # cap on Notable moments items
  quote_max_chars: 100        # quote truncation length in highlights
```

- [ ] **Step 3: Verify config now has six blocks**

```bash
py -3.14 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(sorted(c.keys())); print(c['metrics'])"
```

Expected: `['core', 'diarization', 'kiosk', 'metrics', 'sentiment', 'transcription']` and the three knobs printed.

- [ ] **Step 4: Run the existing test suite to confirm baseline**

```bash
py -3.14 -m pytest tests/ -q
```

Expected: 172 passed. If not, STOP — the baseline is broken, not Task 1's fault but must be fixed before adding tasks.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/config.yaml
git -C c:/repos/TVAD commit -m "feat(metrics): add metrics config block"
```

---

## Task 2: Package skeleton + first aggregator (`aggregate_participation`)

**Files:**
- Create: `target-vad/modes/metrics/__init__.py`
- Create: `target-vad/modes/metrics/aggregator.py`
- Create: `target-vad/tests/metrics/__init__.py`
- Create: `target-vad/tests/metrics/test_aggregator.py`

- [ ] **Step 1: Create empty package markers**

```bash
touch target-vad/modes/metrics/__init__.py
touch target-vad/tests/metrics/__init__.py
```

- [ ] **Step 2: Write failing test for `aggregate_participation`**

Create `target-vad/tests/metrics/test_aggregator.py`:

```python
"""Tests for the metrics aggregator functions."""

import pytest

from modes.metrics import aggregator


def _seg(start, end, sid, name=None, text="", words=None, sentiment=None):
    """Build a segment dict with sane defaults."""
    return {
        "start": start,
        "end": end,
        "speaker_id": sid,
        "speaker": name or sid,
        "text": text,
        "words": words if words is not None else [],
        "sentiment": sentiment,
    }


class TestAggregateParticipation:
    def test_two_speakers_basic(self):
        segments = [
            _seg(0.0, 10.0, "alice", text="hi there friend",
                 words=[{"start": 0, "end": 1, "word": "hi", "probability": 0.9},
                        {"start": 1, "end": 2, "word": "there", "probability": 0.9},
                        {"start": 2, "end": 3, "word": "friend", "probability": 0.9}]),
            _seg(10.0, 14.0, "bob", text="hey",
                 words=[{"start": 10, "end": 11, "word": "hey", "probability": 0.9}]),
            _seg(14.0, 20.0, "alice", text="ok cool",
                 words=[{"start": 14, "end": 15, "word": "ok", "probability": 0.9},
                        {"start": 15, "end": 16, "word": "cool", "probability": 0.9}]),
        ]

        result = aggregator.aggregate_participation(segments)

        assert set(result.keys()) == {"session", "per_speaker"}
        sess = result["session"]
        assert sess["speech_duration_s"] == pytest.approx(20.0)
        assert sess["total_segments"] == 3
        assert sess["total_words"] == 6
        assert sess["unique_speakers"] == 2
        assert sess["identified_speakers"] == 2
        assert sess["unknown_segments"] == 0

        alice = result["per_speaker"]["alice"]
        assert alice["talk_seconds"] == pytest.approx(16.0)
        assert alice["talk_percent"] == pytest.approx(80.0)
        assert alice["segment_count"] == 2
        assert alice["word_count"] == 5
        assert alice["words_per_minute"] == pytest.approx(60 * 5 / 16.0)
        assert alice["mean_segment_seconds"] == pytest.approx(8.0)
        assert alice["median_segment_seconds"] == pytest.approx(8.0)
        assert alice["max_segment_seconds"] == pytest.approx(10.0)

        bob = result["per_speaker"]["bob"]
        assert bob["talk_seconds"] == pytest.approx(4.0)
        assert bob["talk_percent"] == pytest.approx(20.0)
        assert bob["segment_count"] == 1
        assert bob["word_count"] == 1

    def test_unknown_counts_as_regular_bucket_but_separately_summarized(self):
        segments = [
            _seg(0.0, 5.0, "alice"),
            _seg(5.0, 10.0, "unknown"),
        ]
        result = aggregator.aggregate_participation(segments)
        assert result["session"]["unique_speakers"] == 2
        assert result["session"]["identified_speakers"] == 1
        assert result["session"]["unknown_segments"] == 1
        assert "unknown" in result["per_speaker"]
        assert result["per_speaker"]["unknown"]["talk_seconds"] == pytest.approx(5.0)

    def test_empty_segments(self):
        result = aggregator.aggregate_participation([])
        assert result["session"]["speech_duration_s"] == 0.0
        assert result["session"]["total_segments"] == 0
        assert result["session"]["total_words"] == 0
        assert result["session"]["unique_speakers"] == 0
        assert result["per_speaker"] == {}
```

- [ ] **Step 3: Run test, confirm import fails**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: ImportError on `from modes.metrics import aggregator`.

- [ ] **Step 4: Implement `aggregate_participation`**

Create `target-vad/modes/metrics/aggregator.py`:

```python
"""Pure aggregator functions over a list of diarization segments.

Each function takes a List[dict] of segments (the `segments` field of a Phase-2B
diarization JSON, with `text`, `words`, and `sentiment` fields present per the
plan's prerequisites) and returns a plain dict. No side effects, no model loads.
Determinism: identical input -> identical output.
"""

from statistics import mean, median
from typing import Dict, List


def aggregate_participation(segments: List[Dict]) -> Dict:
    """Compute participation stats (Tier 1).

    Returns:
        {
          "session": {speech_duration_s, total_segments, total_words,
                      unique_speakers, identified_speakers, unknown_segments},
          "per_speaker": {sid: {talk_seconds, talk_percent, segment_count,
                                 word_count, words_per_minute,
                                 mean_segment_seconds, median_segment_seconds,
                                 max_segment_seconds}}
        }

    Note: speech_duration_s is the raw sum of segment durations (overlap is
    double-counted). talk_percent for each speaker is computed against this
    sum, so per-speaker percentages always sum to 100%. silence is computed
    elsewhere via the merged interval union (so it can't go negative).
    """
    if not segments:
        return {
            "session": {
                "speech_duration_s": 0.0,
                "total_segments": 0,
                "total_words": 0,
                "unique_speakers": 0,
                "identified_speakers": 0,
                "unknown_segments": 0,
            },
            "per_speaker": {},
        }

    # Group segments by speaker_id, preserving first-appearance order.
    by_speaker: Dict[str, List[Dict]] = {}
    for seg in segments:
        sid = seg["speaker_id"]
        by_speaker.setdefault(sid, []).append(seg)

    speech_duration_s = sum(s["end"] - s["start"] for s in segments)
    total_words = sum(len(s.get("words") or []) for s in segments)
    unknown_segments = sum(1 for s in segments if s["speaker_id"] == "unknown")
    identified_speakers = sum(1 for sid in by_speaker if sid != "unknown")

    per_speaker: Dict[str, Dict] = {}
    for sid, segs in by_speaker.items():
        durations = [s["end"] - s["start"] for s in segs]
        talk = sum(durations)
        wc = sum(len(s.get("words") or []) for s in segs)
        per_speaker[sid] = {
            "talk_seconds": round(talk, 2),
            "talk_percent": round(100.0 * talk / speech_duration_s, 1) if speech_duration_s else 0.0,
            "segment_count": len(segs),
            "word_count": wc,
            "words_per_minute": round(60.0 * wc / talk, 1) if talk else None,
            "mean_segment_seconds": round(mean(durations), 2),
            "median_segment_seconds": round(median(durations), 2),
            "max_segment_seconds": round(max(durations), 2),
        }

    return {
        "session": {
            "speech_duration_s": round(speech_duration_s, 2),
            "total_segments": len(segments),
            "total_words": total_words,
            "unique_speakers": len(by_speaker),
            "identified_speakers": identified_speakers,
            "unknown_segments": unknown_segments,
        },
        "per_speaker": per_speaker,
    }
```

- [ ] **Step 5: Run test, confirm pass**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/__init__.py target-vad/modes/metrics/aggregator.py target-vad/tests/metrics/__init__.py target-vad/tests/metrics/test_aggregator.py
git -C c:/repos/TVAD commit -m "feat(metrics): add aggregate_participation"
```

---

## Task 3: `aggregate_sentiment`

**Files:**
- Modify: `target-vad/modes/metrics/aggregator.py` (append function)
- Modify: `target-vad/tests/metrics/test_aggregator.py` (append test class)

- [ ] **Step 1: Add failing tests**

Append to `target-vad/tests/metrics/test_aggregator.py`:

```python
def _sent(pol_label, emo_label, pol_score=0.8, emo_score=0.6):
    """Build a canonical sentiment dict for a segment."""
    polarities = {"positive", "neutral", "negative"}
    emotions = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
    pol_scores = {k: 0.05 for k in polarities}
    pol_scores[pol_label] = pol_score
    emo_scores = {k: 0.05 for k in emotions}
    emo_scores[emo_label] = emo_score
    return {
        "polarity": {"label": pol_label, "score": pol_score, "scores": pol_scores},
        "emotion":  {"label": emo_label, "score": emo_score, "scores": emo_scores},
    }


class TestAggregateSentiment:
    def test_per_speaker_counts_and_percents(self):
        segments = [
            _seg(0, 5, "alice", sentiment=_sent("positive", "joy")),
            _seg(5, 10, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(10, 15, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(15, 20, "alice", sentiment=_sent("neutral", "surprise")),
            _seg(20, 25, "bob", sentiment=_sent("positive", "joy")),
        ]
        result = aggregator.aggregate_sentiment(segments)

        assert set(result.keys()) == {"session", "per_speaker"}
        sess = result["session"]
        assert sess["polarity_distribution"] == {"positive": 2, "neutral": 3, "negative": 0}
        assert sess["emotion_distribution"]["joy"] == 2
        assert sess["emotion_distribution"]["neutral"] == 2
        assert sess["emotion_distribution"]["surprise"] == 1

        alice = result["per_speaker"]["alice"]
        assert alice["polarity"]["counts"] == {"positive": 1, "neutral": 3, "negative": 0}
        assert alice["polarity"]["percent"] == {"positive": 25.0, "neutral": 75.0, "negative": 0.0}
        assert alice["emotion"]["counts"]["joy"] == 1
        assert alice["emotion"]["counts"]["neutral"] == 2
        assert alice["emotion"]["counts"]["surprise"] == 1
        # Mean top confidence = avg of pol_score across alice's segments = 0.8
        assert alice["polarity"]["mean_top_confidence"] == pytest.approx(0.8)

    def test_null_sentiment_skipped_in_denominator(self):
        segments = [
            _seg(0, 5, "alice", sentiment=_sent("positive", "joy")),
            _seg(5, 10, "alice", sentiment=None),
            _seg(10, 15, "alice", sentiment=None),
        ]
        result = aggregator.aggregate_sentiment(segments)
        alice = result["per_speaker"]["alice"]
        # Only the 1 non-null segment counts. Percent uses denominator=1, not 3.
        assert alice["polarity"]["counts"] == {"positive": 1, "neutral": 0, "negative": 0}
        assert alice["polarity"]["percent"]["positive"] == 100.0

    def test_speaker_with_all_null_sentiment(self):
        segments = [
            _seg(0, 5, "alice", sentiment=None),
            _seg(5, 10, "alice", sentiment=None),
        ]
        result = aggregator.aggregate_sentiment(segments)
        alice = result["per_speaker"]["alice"]
        assert alice["polarity"]["counts"] == {"positive": 0, "neutral": 0, "negative": 0}
        assert alice["polarity"]["percent"] == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        assert alice["polarity"]["mean_top_confidence"] is None
```

- [ ] **Step 2: Run, confirm `aggregate_sentiment` AttributeError**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py::TestAggregateSentiment -v
```

Expected: 3 errors (AttributeError on `aggregator.aggregate_sentiment`).

- [ ] **Step 3: Implement `aggregate_sentiment`**

Append to `target-vad/modes/metrics/aggregator.py`:

```python
_POLARITY_LABELS = ("positive", "neutral", "negative")
_EMOTION_LABELS = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")


def aggregate_sentiment(segments: List[Dict]) -> Dict:
    """Compute polarity + emotion distributions, per speaker and session-wide.

    Segments with sentiment: null are skipped (no signal). Percentages use
    the speaker's classified-segment count as denominator. mean_top_confidence
    is None for speakers with zero classified segments.
    """
    by_speaker: Dict[str, List[Dict]] = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker_id"], []).append(seg)

    sess_pol = {k: 0 for k in _POLARITY_LABELS}
    sess_emo = {k: 0 for k in _EMOTION_LABELS}

    per_speaker: Dict[str, Dict] = {}
    for sid, segs in by_speaker.items():
        pol_counts = {k: 0 for k in _POLARITY_LABELS}
        emo_counts = {k: 0 for k in _EMOTION_LABELS}
        pol_top_scores: List[float] = []
        emo_top_scores: List[float] = []

        for s in segs:
            sent = s.get("sentiment")
            if sent is None:
                continue
            pol = sent["polarity"]
            emo = sent["emotion"]
            pol_counts[pol["label"]] += 1
            emo_counts[emo["label"]] += 1
            sess_pol[pol["label"]] += 1
            sess_emo[emo["label"]] += 1
            pol_top_scores.append(float(pol["score"]))
            emo_top_scores.append(float(emo["score"]))

        classified = sum(pol_counts.values())
        if classified:
            pol_percent = {k: round(100.0 * pol_counts[k] / classified, 1) for k in _POLARITY_LABELS}
            emo_total = sum(emo_counts.values())
            emo_percent = {k: round(100.0 * emo_counts[k] / emo_total, 1) for k in _EMOTION_LABELS}
            pol_mean = round(mean(pol_top_scores), 2)
            emo_mean = round(mean(emo_top_scores), 2)
        else:
            pol_percent = {k: 0.0 for k in _POLARITY_LABELS}
            emo_percent = {k: 0.0 for k in _EMOTION_LABELS}
            pol_mean = None
            emo_mean = None

        per_speaker[sid] = {
            "polarity": {
                "counts": pol_counts,
                "percent": pol_percent,
                "mean_top_confidence": pol_mean,
            },
            "emotion": {
                "counts": emo_counts,
                "percent": emo_percent,
                "mean_top_confidence": emo_mean,
            },
        }

    return {
        "session": {
            "polarity_distribution": sess_pol,
            "emotion_distribution": sess_emo,
        },
        "per_speaker": per_speaker,
    }
```

- [ ] **Step 4: Run, confirm pass**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/aggregator.py target-vad/tests/metrics/test_aggregator.py
git -C c:/repos/TVAD commit -m "feat(metrics): add aggregate_sentiment"
```

---

## Task 4: `aggregate_turn_taking`

**Files:**
- Modify: `target-vad/modes/metrics/aggregator.py`
- Modify: `target-vad/tests/metrics/test_aggregator.py`

- [ ] **Step 1: Add failing tests**

Append to `target-vad/tests/metrics/test_aggregator.py`:

```python
class TestAggregateTurnTaking:
    def test_consecutive_same_speaker_collapses_to_one_turn(self):
        # Alice: 2 adjacent same-speaker segments -> 1 turn; Bob: 1 turn between.
        segments = [
            _seg(0.0, 5.0, "alice"),
            _seg(5.0, 10.0, "alice"),     # collapses with prior turn
            _seg(11.0, 15.0, "bob"),       # gap = 1.0
            _seg(16.0, 20.0, "alice"),     # gap = 1.0
        ]
        result = aggregator.aggregate_turn_taking(segments)
        assert result["per_speaker"]["alice"]["turn_count"] == 2
        assert result["per_speaker"]["bob"]["turn_count"] == 1

    def test_mean_gap_before_excludes_first_turn(self):
        segments = [
            _seg(0.0, 5.0, "alice"),       # first turn — no gap
            _seg(7.0, 10.0, "bob"),         # gap = 2.0
            _seg(13.0, 15.0, "alice"),      # gap = 3.0
            _seg(20.0, 25.0, "alice"),      # same-speaker, same turn (no new gap)
        ]
        result = aggregator.aggregate_turn_taking(segments)
        # Alice has 2 turns; only the second turn (start=13) counts a gap (3.0).
        assert result["per_speaker"]["alice"]["mean_gap_before_seconds"] == pytest.approx(3.0)
        # Bob's first and only turn (start=7) counts a gap of 2.0.
        assert result["per_speaker"]["bob"]["mean_gap_before_seconds"] == pytest.approx(2.0)

    def test_interruption_counted_when_overlap(self):
        segments = [
            _seg(0.0, 10.0, "alice"),
            _seg(8.0, 12.0, "bob"),         # starts before alice ends → interruption
        ]
        result = aggregator.aggregate_turn_taking(segments)
        assert result["per_speaker"]["bob"]["interruption_count"] == 1
        assert result["per_speaker"]["alice"]["interruption_count"] == 0

    def test_single_turn_speaker_has_no_gap(self):
        segments = [
            _seg(0.0, 5.0, "alice"),
        ]
        result = aggregator.aggregate_turn_taking(segments)
        assert result["per_speaker"]["alice"]["turn_count"] == 1
        # No prior turn → no gap. Use None to signal "not applicable".
        assert result["per_speaker"]["alice"]["mean_gap_before_seconds"] is None
        assert result["per_speaker"]["alice"]["interruption_count"] == 0
```

- [ ] **Step 2: Run, confirm fail**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py::TestAggregateTurnTaking -v
```

Expected: 4 errors (AttributeError).

- [ ] **Step 3: Implement `aggregate_turn_taking`**

Append to `target-vad/modes/metrics/aggregator.py`:

```python
def _turns_from_segments(segments: List[Dict]) -> List[Dict]:
    """Collapse contiguous same-speaker segments into turns.

    A turn boundary occurs whenever the speaker_id changes. Returns a list of
    {speaker_id, start, end} dicts where start is the first segment's start and
    end is the last segment's end in that run.
    """
    turns: List[Dict] = []
    for seg in segments:
        if turns and turns[-1]["speaker_id"] == seg["speaker_id"]:
            turns[-1]["end"] = seg["end"]
        else:
            turns.append({
                "speaker_id": seg["speaker_id"],
                "start": seg["start"],
                "end": seg["end"],
            })
    return turns


def aggregate_turn_taking(segments: List[Dict]) -> Dict:
    """Compute per-speaker turn count, mean gap before turn, interruption count.

    A turn = a contiguous run of same-speaker segments. Gap and interruption
    are evaluated against the previous turn (any speaker). First turn in the
    session has no gap.
    """
    turns = _turns_from_segments(segments)

    per_speaker: Dict[str, Dict] = {}
    for i, turn in enumerate(turns):
        sid = turn["speaker_id"]
        per_speaker.setdefault(sid, {"_gaps": [], "_interruptions": 0, "_turn_count": 0})
        per_speaker[sid]["_turn_count"] += 1
        if i > 0:
            prev = turns[i - 1]
            gap = turn["start"] - prev["end"]
            per_speaker[sid]["_gaps"].append(gap)
            if turn["start"] < prev["end"]:
                per_speaker[sid]["_interruptions"] += 1

    result: Dict[str, Dict] = {}
    for sid, accum in per_speaker.items():
        gaps = accum["_gaps"]
        result[sid] = {
            "turn_count": accum["_turn_count"],
            "mean_gap_before_seconds": round(mean(gaps), 2) if gaps else None,
            "interruption_count": accum["_interruptions"],
        }

    return {"per_speaker": result}
```

- [ ] **Step 4: Run, confirm pass**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/aggregator.py target-vad/tests/metrics/test_aggregator.py
git -C c:/repos/TVAD commit -m "feat(metrics): add aggregate_turn_taking"
```

---

## Task 5: `aggregate_pairwise`

**Files:**
- Modify: `target-vad/modes/metrics/aggregator.py`
- Modify: `target-vad/tests/metrics/test_aggregator.py`

- [ ] **Step 1: Add failing tests**

```python
class TestAggregatePairwise:
    def test_who_follows_whom_with_collapsed_turns(self):
        segments = [
            _seg(0, 5, "alice"),
            _seg(5, 10, "alice"),     # same turn as previous
            _seg(10, 15, "bob"),       # alice → bob
            _seg(15, 20, "alice"),     # bob → alice
            _seg(20, 25, "alice"),     # same turn
            _seg(25, 30, "bob"),       # alice → bob
        ]
        result = aggregator.aggregate_pairwise(segments)
        # Two alice→bob transitions, one bob→alice. Self-loops excluded.
        assert result["alice"]["bob"] == 2
        assert result["bob"]["alice"] == 1
        assert result["alice"]["alice"] == 0
        assert result["bob"]["bob"] == 0

    def test_unknown_appears_as_row_and_column_when_present(self):
        segments = [
            _seg(0, 5, "alice"),
            _seg(5, 10, "unknown"),
            _seg(10, 15, "alice"),
        ]
        result = aggregator.aggregate_pairwise(segments)
        assert result["alice"]["unknown"] == 1
        assert result["unknown"]["alice"] == 1

    def test_single_speaker_matrix_empty_off_diagonal(self):
        segments = [
            _seg(0, 5, "alice"),
            _seg(5, 10, "alice"),
        ]
        result = aggregator.aggregate_pairwise(segments)
        assert result == {"alice": {"alice": 0}}
```

- [ ] **Step 2: Run, confirm fail**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py::TestAggregatePairwise -v
```

Expected: 3 errors.

- [ ] **Step 3: Implement `aggregate_pairwise`**

Append to `target-vad/modes/metrics/aggregator.py`:

```python
def aggregate_pairwise(segments: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compute the who-follows-whom transition matrix.

    Builds turns first (collapses contiguous same-speaker segments). For each
    adjacent pair of turns (prev, next), increments matrix[prev_sid][next_sid].
    The returned matrix is rectangular: every observed speaker is both a row
    and a column, with self-transitions always 0.
    """
    turns = _turns_from_segments(segments)
    speakers = []
    seen = set()
    for t in turns:
        if t["speaker_id"] not in seen:
            seen.add(t["speaker_id"])
            speakers.append(t["speaker_id"])

    matrix: Dict[str, Dict[str, int]] = {a: {b: 0 for b in speakers} for a in speakers}
    for i in range(1, len(turns)):
        prev_sid = turns[i - 1]["speaker_id"]
        next_sid = turns[i]["speaker_id"]
        if prev_sid != next_sid:
            matrix[prev_sid][next_sid] += 1
    return matrix
```

- [ ] **Step 4: Run, confirm pass**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/aggregator.py target-vad/tests/metrics/test_aggregator.py
git -C c:/repos/TVAD commit -m "feat(metrics): add aggregate_pairwise"
```

---

## Task 6: `aggregate_timeline`

**Files:**
- Modify: `target-vad/modes/metrics/aggregator.py`
- Modify: `target-vad/tests/metrics/test_aggregator.py`

- [ ] **Step 1: Add failing tests**

```python
class TestAggregateTimeline:
    def test_single_bucket_session_under_threshold(self):
        segments = [
            _seg(0, 50, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(50, 80, "bob", sentiment=_sent("positive", "joy")),
        ]
        result = aggregator.aggregate_timeline(segments, duration_s=90.0, bucket_seconds=300)
        assert len(result) == 1
        b = result[0]
        assert b["bucket_start_s"] == 0
        assert b["bucket_end_s"] == 300
        assert b["per_speaker_talk_s"]["alice"] == pytest.approx(50.0)
        assert b["per_speaker_talk_s"]["bob"] == pytest.approx(30.0)
        assert b["per_speaker_polarity_mode"]["alice"] == "neutral"
        assert b["per_speaker_emotion_mode"]["bob"] == "joy"

    def test_segment_straddling_boundary_split_proportionally(self):
        # Bucket 0-10, bucket 10-20. Segment 5-15 spans both.
        segments = [
            _seg(5.0, 15.0, "alice", sentiment=_sent("neutral", "neutral")),
        ]
        result = aggregator.aggregate_timeline(segments, duration_s=20.0, bucket_seconds=10)
        assert len(result) == 2
        # Talk split 5/5
        assert result[0]["per_speaker_talk_s"]["alice"] == pytest.approx(5.0)
        assert result[1]["per_speaker_talk_s"]["alice"] == pytest.approx(5.0)
        # Mode credited only to bucket containing the start (bucket 0).
        assert result[0]["per_speaker_polarity_mode"]["alice"] == "neutral"
        assert result[1]["per_speaker_polarity_mode"] == {}

    def test_empty_bucket_emitted_but_per_speaker_dicts_omit_silent_speakers(self):
        # 30s session with bucket=10. Alice in 0-10 only.
        segments = [
            _seg(0, 10, "alice", sentiment=_sent("neutral", "neutral")),
        ]
        result = aggregator.aggregate_timeline(segments, duration_s=30.0, bucket_seconds=10)
        assert len(result) == 3
        assert result[1]["per_speaker_talk_s"] == {}
        assert result[2]["per_speaker_talk_s"] == {}

    def test_mode_picks_argmax_of_label_frequency_in_bucket(self):
        # Three segments all in bucket 0-100. Alice: 2 neutral, 1 surprise → mode=neutral.
        segments = [
            _seg(0, 10, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(10, 20, "alice", sentiment=_sent("neutral", "neutral")),
            _seg(20, 30, "alice", sentiment=_sent("neutral", "surprise")),
        ]
        result = aggregator.aggregate_timeline(segments, duration_s=30.0, bucket_seconds=100)
        assert result[0]["per_speaker_emotion_mode"]["alice"] == "neutral"
```

- [ ] **Step 2: Run, confirm fail**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py::TestAggregateTimeline -v
```

Expected: 4 errors.

- [ ] **Step 3: Implement `aggregate_timeline`**

Append to `target-vad/modes/metrics/aggregator.py`:

```python
from collections import Counter
import math


def aggregate_timeline(
    segments: List[Dict],
    duration_s: float,
    bucket_seconds: int,
) -> List[Dict]:
    """Bucket segments into fixed time windows.

    For each bucket:
      - per_speaker_talk_s: segment-overlap with bucket, apportioned proportionally
        for segments straddling boundaries.
      - per_speaker_polarity_mode / _emotion_mode: most-frequent label among the
        speaker's segments whose START falls within the bucket. Omitted if no
        such segments. Tie-broken by alphabetical label (stable on rerun).
    """
    n_buckets = max(1, math.ceil(duration_s / bucket_seconds)) if duration_s > 0 else 1
    buckets: List[Dict] = []
    for i in range(n_buckets):
        buckets.append({
            "bucket_start_s": i * bucket_seconds,
            "bucket_end_s": (i + 1) * bucket_seconds,
            "per_speaker_talk_s": {},
            "_pol_counters": {},
            "_emo_counters": {},
        })

    for seg in segments:
        sid = seg["speaker_id"]
        # Apportion talk seconds across overlapping buckets.
        for b in buckets:
            lo = max(seg["start"], b["bucket_start_s"])
            hi = min(seg["end"], b["bucket_end_s"])
            if hi > lo:
                b["per_speaker_talk_s"][sid] = round(
                    b["per_speaker_talk_s"].get(sid, 0.0) + (hi - lo), 2
                )

        # Credit polarity/emotion mode to the bucket containing the start.
        start_bucket_idx = min(int(seg["start"] // bucket_seconds), n_buckets - 1)
        b = buckets[start_bucket_idx]
        sent = seg.get("sentiment")
        if sent is not None:
            b["_pol_counters"].setdefault(sid, Counter())[sent["polarity"]["label"]] += 1
            b["_emo_counters"].setdefault(sid, Counter())[sent["emotion"]["label"]] += 1

    # Materialize modes from counters (stable tie-break: alphabetical label).
    out: List[Dict] = []
    for b in buckets:
        pol_mode: Dict[str, str] = {}
        emo_mode: Dict[str, str] = {}
        for sid, ctr in b["_pol_counters"].items():
            top = max(ctr.items(), key=lambda kv: (kv[1], -ord(kv[0][0])))
            # max by count desc, label asc — use sorted for clarity:
            top_label = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            pol_mode[sid] = top_label
        for sid, ctr in b["_emo_counters"].items():
            top_label = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            emo_mode[sid] = top_label
        out.append({
            "bucket_start_s": b["bucket_start_s"],
            "bucket_end_s": b["bucket_end_s"],
            "per_speaker_talk_s": b["per_speaker_talk_s"],
            "per_speaker_polarity_mode": pol_mode,
            "per_speaker_emotion_mode": emo_mode,
        })
    return out
```

- [ ] **Step 4: Run, confirm pass**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/aggregator.py target-vad/tests/metrics/test_aggregator.py
git -C c:/repos/TVAD commit -m "feat(metrics): add aggregate_timeline"
```

---

## Task 7: `select_highlights`

**Files:**
- Modify: `target-vad/modes/metrics/aggregator.py`
- Modify: `target-vad/tests/metrics/test_aggregator.py`

- [ ] **Step 1: Add failing tests**

```python
class TestSelectHighlights:
    def test_longest_segment_always_present_when_segments_exist(self):
        segments = [
            _seg(0, 30, "alice", text="long alice", sentiment=_sent("neutral", "neutral")),
            _seg(30, 35, "bob", text="short bob", sentiment=_sent("neutral", "neutral")),
        ]
        timeline = aggregator.aggregate_timeline(segments, duration_s=35.0, bucket_seconds=300)
        h = aggregator.select_highlights(segments, timeline, top_k=5, quote_max_chars=100)
        kinds = [x["kind"] for x in h]
        assert "longest_segment" in kinds
        longest = next(x for x in h if x["kind"] == "longest_segment")
        assert longest["speaker_id"] == "alice"
        assert longest["value_s"] == pytest.approx(30.0)
        assert longest["quote"] == "long alice"

    def test_most_positive_skipped_when_no_positive_segments(self):
        segments = [
            _seg(0, 10, "alice", text="meh", sentiment=_sent("neutral", "neutral")),
        ]
        timeline = aggregator.aggregate_timeline(segments, duration_s=10.0, bucket_seconds=300)
        h = aggregator.select_highlights(segments, timeline, top_k=5, quote_max_chars=100)
        assert all(x["kind"] != "most_positive" for x in h)
        assert all(x["kind"] != "most_negative" for x in h)

    def test_top_k_caps_highlights(self):
        segments = [
            _seg(0, 30, "alice", text="long", sentiment=_sent("positive", "joy", pol_score=0.99)),
            _seg(30, 35, "bob", text="bad", sentiment=_sent("negative", "anger", pol_score=0.99)),
            _seg(35, 40, "alice", text="d1", sentiment=_sent("neutral", "disgust")),
            _seg(40, 45, "alice", text="d2", sentiment=_sent("neutral", "disgust")),
        ]
        timeline = aggregator.aggregate_timeline(segments, duration_s=45.0, bucket_seconds=300)
        h = aggregator.select_highlights(segments, timeline, top_k=2, quote_max_chars=100)
        assert len(h) == 2

    def test_quote_truncated_with_ellipsis(self):
        long_text = "x" * 200
        segments = [_seg(0, 30, "alice", text=long_text,
                         sentiment=_sent("positive", "joy", pol_score=0.99))]
        timeline = aggregator.aggregate_timeline(segments, duration_s=30.0, bucket_seconds=300)
        h = aggregator.select_highlights(segments, timeline, top_k=5, quote_max_chars=50)
        for item in h:
            if "quote" in item:
                assert len(item["quote"]) <= 53  # 50 + "..."
                if len(long_text) > 50:
                    assert item["quote"].endswith("...")

    def test_deterministic_tie_break_earliest_then_alphabetical_sid(self):
        # Two segments with exact same duration — earliest start wins for longest_segment.
        segments = [
            _seg(10, 20, "bob", text="bob"),
            _seg(0, 10, "alice", text="alice"),
            _seg(20, 30, "carol", text="carol"),
        ]
        timeline = aggregator.aggregate_timeline(segments, duration_s=30.0, bucket_seconds=300)
        h = aggregator.select_highlights(segments, timeline, top_k=5, quote_max_chars=100)
        longest = next(x for x in h if x["kind"] == "longest_segment")
        assert longest["speaker_id"] == "alice"  # earliest start of equal-duration tie
```

- [ ] **Step 2: Run, confirm fail**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py::TestSelectHighlights -v
```

Expected: 5 errors.

- [ ] **Step 3: Implement `select_highlights`**

Append to `target-vad/modes/metrics/aggregator.py`:

```python
def _truncate_quote(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def select_highlights(
    segments: List[Dict],
    timeline: List[Dict],
    top_k: int,
    quote_max_chars: int,
) -> List[Dict]:
    """Pick up to top_k highlights deterministically.

    Priority order: longest_segment, most_positive, most_negative,
    high_disgust_window, quietest_window, busiest_window, solo_dominator.
    Each kind's selection rule is documented in the spec. Ties are broken by
    earliest start time, then alphabetical speaker_id (stable on rerun).
    """
    highlights: List[Dict] = []

    def cap_reached() -> bool:
        return len(highlights) >= top_k

    # longest_segment
    if segments and not cap_reached():
        # Sort: longest duration desc, earliest start asc, alphabetical sid asc.
        winner = sorted(
            segments,
            key=lambda s: (-(s["end"] - s["start"]), s["start"], s["speaker_id"]),
        )[0]
        highlights.append({
            "kind": "longest_segment",
            "speaker_id": winner["speaker_id"],
            "start": winner["start"],
            "end": winner["end"],
            "value_s": round(winner["end"] - winner["start"], 2),
            "quote": _truncate_quote(winner.get("text", ""), quote_max_chars),
        })

    # most_positive — only consider segments whose label IS positive.
    pos_candidates = [s for s in segments
                      if (s.get("sentiment") is not None
                          and s["sentiment"]["polarity"]["label"] == "positive")]
    if pos_candidates and not cap_reached():
        winner = sorted(
            pos_candidates,
            key=lambda s: (-s["sentiment"]["polarity"]["scores"]["positive"],
                           s["start"], s["speaker_id"]),
        )[0]
        highlights.append({
            "kind": "most_positive",
            "speaker_id": winner["speaker_id"],
            "start": winner["start"],
            "end": winner["end"],
            "polarity_score": round(winner["sentiment"]["polarity"]["scores"]["positive"], 2),
            "quote": _truncate_quote(winner.get("text", ""), quote_max_chars),
        })

    # most_negative
    neg_candidates = [s for s in segments
                      if (s.get("sentiment") is not None
                          and s["sentiment"]["polarity"]["label"] == "negative")]
    if neg_candidates and not cap_reached():
        winner = sorted(
            neg_candidates,
            key=lambda s: (-s["sentiment"]["polarity"]["scores"]["negative"],
                           s["start"], s["speaker_id"]),
        )[0]
        highlights.append({
            "kind": "most_negative",
            "speaker_id": winner["speaker_id"],
            "start": winner["start"],
            "end": winner["end"],
            "polarity_score": round(winner["sentiment"]["polarity"]["scores"]["negative"], 2),
            "quote": _truncate_quote(winner.get("text", ""), quote_max_chars),
        })

    # high_disgust_window — bucket with max disgust-segment count.
    disgust_counts: List[Dict] = []
    for b in timeline:
        # Count disgust-labeled segment-starts in this bucket per speaker.
        total = 0
        per_sid: Dict[str, int] = {}
        for s in segments:
            if (b["bucket_start_s"] <= s["start"] < b["bucket_end_s"]
                    and s.get("sentiment") is not None
                    and s["sentiment"]["emotion"]["label"] == "disgust"):
                total += 1
                per_sid[s["speaker_id"]] = per_sid.get(s["speaker_id"], 0) + 1
        if total > 0:
            top_sid = sorted(per_sid.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            disgust_counts.append({"bucket": b, "count": total, "speaker_id": top_sid})
    if disgust_counts and not cap_reached():
        winner = sorted(disgust_counts, key=lambda d: (-d["count"], d["bucket"]["bucket_start_s"]))[0]
        highlights.append({
            "kind": "high_disgust_window",
            "bucket_start_s": winner["bucket"]["bucket_start_s"],
            "bucket_end_s": winner["bucket"]["bucket_end_s"],
            "speaker_id": winner["speaker_id"],
            "count": winner["count"],
        })

    # quietest_window / busiest_window / solo_dominator — only when ≥ 2 buckets.
    if len(timeline) >= 2:
        # Total talk per bucket = sum across all speakers.
        bucket_totals = [
            (b, round(sum(b["per_speaker_talk_s"].values()), 2)) for b in timeline
        ]

        if not cap_reached():
            winner = sorted(bucket_totals, key=lambda t: (t[1], t[0]["bucket_start_s"]))[0]
            highlights.append({
                "kind": "quietest_window",
                "bucket_start_s": winner[0]["bucket_start_s"],
                "bucket_end_s": winner[0]["bucket_end_s"],
                "total_talk_s": winner[1],
            })

        if not cap_reached():
            winner = sorted(bucket_totals, key=lambda t: (-t[1], t[0]["bucket_start_s"]))[0]
            highlights.append({
                "kind": "busiest_window",
                "bucket_start_s": winner[0]["bucket_start_s"],
                "bucket_end_s": winner[0]["bucket_end_s"],
                "total_talk_s": winner[1],
            })

        if not cap_reached():
            for b, total in bucket_totals:
                if total < 60.0:
                    continue
                # Find max-talk speaker in this bucket.
                top_sid, top_talk = max(
                    b["per_speaker_talk_s"].items(),
                    key=lambda kv: (kv[1], kv[0]),
                )
                if top_talk / total >= 0.8:
                    highlights.append({
                        "kind": "solo_dominator",
                        "bucket_start_s": b["bucket_start_s"],
                        "bucket_end_s": b["bucket_end_s"],
                        "speaker_id": top_sid,
                        "talk_s": round(top_talk, 2),
                        "total_talk_s": total,
                    })
                    break  # Only the first qualifying bucket.

    return highlights[:top_k]
```

- [ ] **Step 4: Run, confirm pass**

```bash
py -3.14 -m pytest tests/metrics/test_aggregator.py -v
```

Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/aggregator.py target-vad/tests/metrics/test_aggregator.py
git -C c:/repos/TVAD commit -m "feat(metrics): add select_highlights"
```

---

## Task 8: Markdown renderer + golden-file test

**Files:**
- Create: `target-vad/modes/metrics/renderer.py`
- Create: `target-vad/tests/metrics/test_renderer.py`
- Create: `target-vad/tests/metrics/fixtures/golden_report.md`

- [ ] **Step 1: Build the fixture metrics block + write a stable golden Markdown**

First, sketch the metrics block the renderer will consume — keep it small so the golden can be reviewed by eye. Create `target-vad/tests/metrics/fixtures/golden_report.md` with:

```markdown
# Session Metrics — fixture.wav

**Duration:** 90.0 s (1 min 30 s) · **Speech:** 78.4 s · **Silence:** 11.6 s
**Speakers:** 2 (2 identified, 0 unknown) · **Words:** 312 · **Segments:** 9
**Analyzed:** 2026-05-16T18:42:01Z

## Notable moments
- **Longest contribution** — Speaker A, 42.4 s at 00:00: *"The whole idea is that both of them should..."*
- **Most positive** — Speaker B at 01:05 (score 0.94): *"Yes, that's a great point about the radar pairing."*

## Participation

| Speaker   | Talk  | %     | Segs | Words | WPM   | Mean seg | Max seg |
|-----------|------:|------:|-----:|------:|------:|---------:|--------:|
| Speaker A | 52.1s | 66.5% | 5    | 198   | 228.0 | 10.4 s   | 42.6 s  |
| Speaker B | 26.3s | 33.5% | 4    | 114   | 260.0 | 6.6 s    | 9.1 s   |

## Sentiment — polarity (per speaker)

| Speaker   | Positive | Neutral | Negative | Mean conf. |
|-----------|---------:|--------:|---------:|-----------:|
| Speaker A | 20%      | 80%     | 0%       | 0.81       |
| Speaker B | 25%      | 75%     | 0%       | 0.79       |

## Sentiment — emotion (per speaker)

| Speaker   | Joy | Neutral | Surprise | Disgust* | Anger | Fear | Sadness | Mean conf. |
|-----------|----:|--------:|---------:|---------:|------:|-----:|--------:|-----------:|
| Speaker A | 0%  | 60%     | 20%      | 20%      | 0%    | 0%   | 0%      | 0.62       |
| Speaker B | 25% | 50%     | 0%       | 25%      | 0%    | 0%   | 0%      | 0.58       |

\* "Disgust" from the emotion model tends to fire on polite-disagreement phrasing — read as "registered disagreement" rather than visceral disgust.

## Turn-taking

| Speaker   | Turns | Mean gap before | Interruptions |
|-----------|------:|----------------:|--------------:|
| Speaker A | 4     | 1.42 s          | 1             |
| Speaker B | 4     | 0.85 s          | 0             |

---
_Caveat: 'unknown' segments may represent multiple physical speakers; the diarization layer collapses all unenrolled clusters into one bucket._
```

(This single-bucket fixture deliberately omits "Activity over time" and "Who follows whom" sections — they are tested separately.)

- [ ] **Step 2: Write a failing test that builds the matching input and compares to the golden**

Create `target-vad/tests/metrics/test_renderer.py`:

```python
"""Golden-file and unit tests for modes.metrics.renderer."""

import os
from pathlib import Path

from modes.metrics import renderer

FIXTURES = Path(__file__).parent / "fixtures"


def _full_metrics_block():
    """Build the metrics block that should render to golden_report.md."""
    return {
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
            "emotion_distribution": {"joy": 1, "neutral": 5, "surprise": 1,
                                     "disgust": 2, "anger": 0, "fear": 0, "sadness": 0},
        },
        "speakers": [
            {
                "speaker_id": "session_speaker_a", "speaker": "Speaker A",
                "participation": {
                    "talk_seconds": 52.1, "talk_percent": 66.5,
                    "segment_count": 5, "word_count": 198,
                    "words_per_minute": 228.0,
                    "mean_segment_seconds": 10.4, "median_segment_seconds": 8.7,
                    "max_segment_seconds": 42.6,
                },
                "sentiment": {
                    "polarity": {
                        "counts": {"positive": 1, "neutral": 4, "negative": 0},
                        "percent": {"positive": 20.0, "neutral": 80.0, "negative": 0.0},
                        "mean_top_confidence": 0.81,
                    },
                    "emotion": {
                        "counts": {"joy": 0, "neutral": 3, "surprise": 1, "disgust": 1,
                                   "anger": 0, "fear": 0, "sadness": 0},
                        "percent": {"joy": 0.0, "neutral": 60.0, "surprise": 20.0,
                                    "disgust": 20.0, "anger": 0.0, "fear": 0.0, "sadness": 0.0},
                        "mean_top_confidence": 0.62,
                    },
                },
                "turn_taking": {"turn_count": 4, "mean_gap_before_seconds": 1.42, "interruption_count": 1},
            },
            {
                "speaker_id": "session_speaker_b", "speaker": "Speaker B",
                "participation": {
                    "talk_seconds": 26.3, "talk_percent": 33.5,
                    "segment_count": 4, "word_count": 114,
                    "words_per_minute": 260.0,
                    "mean_segment_seconds": 6.6, "median_segment_seconds": 6.0,
                    "max_segment_seconds": 9.1,
                },
                "sentiment": {
                    "polarity": {
                        "counts": {"positive": 1, "neutral": 3, "negative": 0},
                        "percent": {"positive": 25.0, "neutral": 75.0, "negative": 0.0},
                        "mean_top_confidence": 0.79,
                    },
                    "emotion": {
                        "counts": {"joy": 1, "neutral": 2, "surprise": 0, "disgust": 1,
                                   "anger": 0, "fear": 0, "sadness": 0},
                        "percent": {"joy": 25.0, "neutral": 50.0, "surprise": 0.0,
                                    "disgust": 25.0, "anger": 0.0, "fear": 0.0, "sadness": 0.0},
                        "mean_top_confidence": 0.58,
                    },
                },
                "turn_taking": {"turn_count": 4, "mean_gap_before_seconds": 0.85, "interruption_count": 0},
            },
        ],
        "pairwise_followers": {"session_speaker_a": {"session_speaker_a": 0}},  # single-pair => omitted
        "timeline": [
            {"bucket_start_s": 0, "bucket_end_s": 300,
             "per_speaker_talk_s": {"session_speaker_a": 52.1, "session_speaker_b": 26.3},
             "per_speaker_polarity_mode": {"session_speaker_a": "neutral",
                                            "session_speaker_b": "neutral"},
             "per_speaker_emotion_mode": {"session_speaker_a": "neutral",
                                           "session_speaker_b": "neutral"}}
        ],
        "highlights": [
            {"kind": "longest_segment", "speaker_id": "session_speaker_a",
             "start": 0.0, "end": 42.4, "value_s": 42.4,
             "quote": "The whole idea is that both of them should..."},
            {"kind": "most_positive", "speaker_id": "session_speaker_b",
             "start": 65.0, "end": 71.0, "polarity_score": 0.94,
             "quote": "Yes, that's a great point about the radar pairing."},
        ],
    }


def _session_meta():
    return {
        "audio_file": "fixture.wav",
        "analyzed_at": "2026-05-16T18:42:01Z",
    }


def _name_lookup_from_speakers(speakers):
    return {s["speaker_id"]: s["speaker"] for s in speakers}


class TestRenderer:
    def test_full_report_matches_golden(self):
        metrics = _full_metrics_block()
        meta = _session_meta()
        out = renderer.render_markdown(metrics, meta)
        expected = (FIXTURES / "golden_report.md").read_text(encoding="utf-8")
        assert out == expected

    def test_empty_highlights_omits_notable_moments_section(self):
        metrics = _full_metrics_block()
        metrics["highlights"] = []
        out = renderer.render_markdown(metrics, _session_meta())
        assert "## Notable moments" not in out

    def test_single_speaker_omits_pairwise_section(self):
        metrics = _full_metrics_block()
        metrics["speakers"] = [metrics["speakers"][0]]
        out = renderer.render_markdown(metrics, _session_meta())
        assert "## Who follows whom" not in out

    def test_single_bucket_omits_activity_section(self):
        metrics = _full_metrics_block()
        # Already single-bucket. Just confirm.
        out = renderer.render_markdown(metrics, _session_meta())
        assert "## Activity over time" not in out

    def test_disgust_footnote_present_when_emotion_table_rendered(self):
        metrics = _full_metrics_block()
        out = renderer.render_markdown(metrics, _session_meta())
        assert "registered disagreement" in out
```

- [ ] **Step 3: Run, confirm import fails**

```bash
py -3.14 -m pytest tests/metrics/test_renderer.py -v
```

Expected: ImportError on `from modes.metrics import renderer`.

- [ ] **Step 4: Implement `render_markdown`**

Create `target-vad/modes/metrics/renderer.py`:

```python
"""Render the contribution_metrics block as a human-readable Markdown report.

Pure function: same metrics + session metadata -> identical Markdown. No LLM,
no I/O. The caller writes the returned string to disk in UTF-8.

Section omission rules (per spec):
  - `## Notable moments` omitted when highlights list is empty
  - `## Who follows whom` omitted when there is only one speaker
  - `## Activity over time` omitted when timeline has only one bucket
"""

import os
from typing import Dict, List


_BLOCK_LADDER = [(0.0, " "), (0.25, "░"), (0.5, "▒"), (0.75, "▓"), (1.01, "█")]


def _fmt_seconds_compact(s: float) -> str:
    if s >= 60:
        mins = int(s // 60)
        rem = s - mins * 60
        if rem < 0.05:
            return f"{mins} min"
        return f"{mins} min {rem:.1f} s"
    return f"{s:.1f} s"


def _fmt_mmss(s: float) -> str:
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


def _fmt_pct(p: float) -> str:
    # Render whole-number percentages with no decimal, fractional with one.
    if p == int(p):
        return f"{int(p)}%"
    return f"{p:.1f}%"


def _fmt_quote(q: str) -> str:
    return f'*"{q}"*' if q else ""


def render_markdown(metrics: Dict, session_meta: Dict) -> str:
    lines: List[str] = []

    audio = session_meta.get("audio_file", "(unknown source)")
    audio_name = os.path.basename(audio) if audio else "(unknown source)"
    lines.append(f"# Session Metrics — {audio_name}")
    lines.append("")

    sess = metrics["session"]
    duration_pretty = _fmt_seconds_compact(sess["duration_s"])
    lines.append(
        f"**Duration:** {sess['duration_s']} s ({duration_pretty}) · "
        f"**Speech:** {sess['speech_duration_s']} s · "
        f"**Silence:** {sess['silence_duration_s']} s"
    )
    lines.append(
        f"**Speakers:** {sess['unique_speakers']} "
        f"({sess['identified_speakers']} identified, {sess['unknown_segments']} unknown) · "
        f"**Words:** {sess['total_words']} · **Segments:** {sess['total_segments']}"
    )
    lines.append(f"**Analyzed:** {session_meta.get('analyzed_at', '')}")
    lines.append("")

    highlights = metrics.get("highlights") or []
    if highlights:
        lines.append("## Notable moments")
        speaker_names = {sp["speaker_id"]: sp["speaker"] for sp in metrics["speakers"]}
        for h in highlights:
            lines.append(_render_highlight(h, speaker_names))
        lines.append("")

    # Participation table.
    lines.append("## Participation")
    lines.append("")
    lines.append("| Speaker   | Talk  | %     | Segs | Words | WPM   | Mean seg | Max seg |")
    lines.append("|-----------|------:|------:|-----:|------:|------:|---------:|--------:|")
    for sp in metrics["speakers"]:
        p = sp["participation"]
        wpm = f"{p['words_per_minute']:.1f}" if p["words_per_minute"] is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {p['talk_seconds']}s | {_fmt_pct(p['talk_percent'])} | "
            f"{p['segment_count']}    | {p['word_count']}   | {wpm} | "
            f"{p['mean_segment_seconds']} s   | {p['max_segment_seconds']} s  |"
        )
    lines.append("")

    # Polarity table.
    lines.append("## Sentiment — polarity (per speaker)")
    lines.append("")
    lines.append("| Speaker   | Positive | Neutral | Negative | Mean conf. |")
    lines.append("|-----------|---------:|--------:|---------:|-----------:|")
    for sp in metrics["speakers"]:
        pol = sp["sentiment"]["polarity"]["percent"]
        conf = sp["sentiment"]["polarity"]["mean_top_confidence"]
        conf_s = f"{conf:.2f}" if conf is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {_fmt_pct(pol['positive'])}      | "
            f"{_fmt_pct(pol['neutral'])}     | {_fmt_pct(pol['negative'])}       | {conf_s}       |"
        )
    lines.append("")

    # Emotion table.
    lines.append("## Sentiment — emotion (per speaker)")
    lines.append("")
    lines.append("| Speaker   | Joy | Neutral | Surprise | Disgust* | Anger | Fear | Sadness | Mean conf. |")
    lines.append("|-----------|----:|--------:|---------:|---------:|------:|-----:|--------:|-----------:|")
    for sp in metrics["speakers"]:
        emo = sp["sentiment"]["emotion"]["percent"]
        conf = sp["sentiment"]["emotion"]["mean_top_confidence"]
        conf_s = f"{conf:.2f}" if conf is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {_fmt_pct(emo['joy'])}  | {_fmt_pct(emo['neutral'])}     | "
            f"{_fmt_pct(emo['surprise'])}      | {_fmt_pct(emo['disgust'])}      | "
            f"{_fmt_pct(emo['anger'])}    | {_fmt_pct(emo['fear'])}   | {_fmt_pct(emo['sadness'])}      | {conf_s}       |"
        )
    lines.append("")
    lines.append(
        '\\* "Disgust" from the emotion model tends to fire on polite-disagreement phrasing — '
        'read as "registered disagreement" rather than visceral disgust.'
    )
    lines.append("")

    # Turn-taking table.
    lines.append("## Turn-taking")
    lines.append("")
    lines.append("| Speaker   | Turns | Mean gap before | Interruptions |")
    lines.append("|-----------|------:|----------------:|--------------:|")
    for sp in metrics["speakers"]:
        tt = sp["turn_taking"]
        gap = f"{tt['mean_gap_before_seconds']} s" if tt["mean_gap_before_seconds"] is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {tt['turn_count']}     | {gap}          | {tt['interruption_count']}             |"
        )
    lines.append("")

    # Who follows whom — only when ≥ 2 speakers.
    if len(metrics["speakers"]) >= 2:
        lines.append("## Who follows whom")
        lines.append("")
        lines.append("Rows = previous speaker, columns = next speaker. Cell = transition count.")
        lines.append("")
        speaker_ids = [sp["speaker_id"] for sp in metrics["speakers"]]
        speaker_names = {sp["speaker_id"]: sp["speaker"] for sp in metrics["speakers"]}
        header_cells = [f"→ {speaker_names[sid]}" for sid in speaker_ids]
        lines.append("|              | " + " | ".join(header_cells) + " |")
        sep_cells = ["----:" for _ in speaker_ids]
        lines.append("|--------------|" + "|".join(f" {c} " for c in sep_cells) + "|")
        pairwise = metrics.get("pairwise_followers", {})
        for sid in speaker_ids:
            row_cells = []
            for other in speaker_ids:
                if other == sid:
                    row_cells.append("—")
                else:
                    row_cells.append(str(pairwise.get(sid, {}).get(other, 0)))
            lines.append(f"| {speaker_names[sid]} → | " + " | ".join(row_cells) + " |")
        lines.append("")

    # Activity over time — only when ≥ 2 buckets.
    if len(metrics.get("timeline", [])) >= 2:
        lines.append("## Activity over time (5-min windows)")
        lines.append("")
        bucket_seconds = metrics["timeline"][0]["bucket_end_s"] - metrics["timeline"][0]["bucket_start_s"]
        # Header row of bucket-start timestamps.
        speaker_names = {sp["speaker_id"]: sp["speaker"] for sp in metrics["speakers"]}
        lines.append("```")
        timestamps = "        " + "  ".join(_fmt_mmss(b["bucket_start_s"]) for b in metrics["timeline"])
        lines.append(timestamps)
        for sp in metrics["speakers"]:
            cells = []
            for b in metrics["timeline"]:
                talk = b["per_speaker_talk_s"].get(sp["speaker_id"], 0.0)
                ratio = (talk / bucket_seconds) if bucket_seconds else 0
                cells.append(_block_for_ratio(ratio))
            label = sp["speaker"][:6].ljust(6)
            lines.append(f"{label}  " + "  ".join(cells))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append(
        "_Caveat: 'unknown' segments may represent multiple physical speakers; "
        "the diarization layer collapses all unenrolled clusters into one bucket._"
    )
    return "\n".join(lines)


def _render_highlight(h: Dict, speaker_names: Dict[str, str]) -> str:
    kind = h["kind"]
    if kind == "longest_segment":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return f"- **Longest contribution** — {name}, {h['value_s']} s at {_fmt_mmss(h['start'])}: {_fmt_quote(h['quote'])}"
    if kind == "most_positive":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return f"- **Most positive** — {name} at {_fmt_mmss(h['start'])} (score {h['polarity_score']:.2f}): {_fmt_quote(h['quote'])}"
    if kind == "most_negative":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return f"- **Most negative** — {name} at {_fmt_mmss(h['start'])} (score {h['polarity_score']:.2f}): {_fmt_quote(h['quote'])}"
    if kind == "high_disgust_window":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return (f"- **High disgust window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])}, mostly {name} ({h['count']} segments)")
    if kind == "quietest_window":
        return (f"- **Quietest window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])} ({h['total_talk_s']} s talk)")
    if kind == "busiest_window":
        return (f"- **Busiest window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])} ({h['total_talk_s']} s talk)")
    if kind == "solo_dominator":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return (f"- **Solo dominator window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])}, {name} held {h['talk_s']} of {h['total_talk_s']} s")
    return f"- **{kind}** — {h}"


def _block_for_ratio(r: float) -> str:
    for threshold, glyph in _BLOCK_LADDER:
        if r < threshold:
            return glyph
    return "█"
```

- [ ] **Step 5: Run renderer test, iterate on whitespace/spacing until golden matches**

```bash
py -3.14 -m pytest tests/metrics/test_renderer.py -v
```

Expected: 5 passed. If the golden-file test fails due to whitespace mismatch, **don't change the golden** — adjust the renderer's f-strings until the output is byte-identical to the committed golden file. This is the whole point of pinning the renderer with a golden test.

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/renderer.py target-vad/tests/metrics/test_renderer.py target-vad/tests/metrics/fixtures/golden_report.md
git -C c:/repos/TVAD commit -m "feat(metrics): add Markdown renderer with golden-file test"
```

---

## Task 9: `metrics.py` CLI orchestration

**Files:**
- Create: `target-vad/metrics.py`
- Create: `target-vad/tests/metrics/test_orchestration.py`

- [ ] **Step 1: Write failing orchestration tests**

Create `target-vad/tests/metrics/test_orchestration.py`:

```python
"""Tests for metrics.py CLI orchestration."""

import json
import os
import shutil
import tempfile

import pytest

import metrics


def _seg(start, end, sid, name=None, text="", words_count=0, sentiment=None):
    return {
        "start": start, "end": end,
        "speaker_id": sid, "speaker": name or sid,
        "text": text,
        "words": [{"start": start + i * 0.1, "end": start + (i + 1) * 0.1,
                   "word": f"w{i}", "probability": 0.9}
                  for i in range(words_count)],
        "sentiment": sentiment,
    }


def _sent(pol_label, emo_label, pol_score=0.8, emo_score=0.6):
    polarities = {"positive", "neutral", "negative"}
    emotions = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
    pol_scores = {k: 0.05 for k in polarities}
    pol_scores[pol_label] = pol_score
    emo_scores = {k: 0.05 for k in emotions}
    emo_scores[emo_label] = emo_score
    return {
        "polarity": {"label": pol_label, "score": pol_score, "scores": pol_scores},
        "emotion":  {"label": emo_label, "score": emo_score, "scores": emo_scores},
    }


@pytest.fixture
def tmp_workspace():
    d = tempfile.mkdtemp()
    json_path = os.path.join(d, "session.diarization.json")
    config_path = os.path.join(d, "config.yaml")
    with open(config_path, "w") as f:
        f.write("metrics:\n  bucket_seconds: 300\n  top_k_highlights: 5\n  quote_max_chars: 100\n")
    data = {
        "audio_file": "irrelevant.wav",
        "duration_s": 30.0,
        "diarized_at": "2026-05-15T10:00:00Z",
        "config": {"pyannote_pipeline": "p"},
        "enrolled_users_matched": [{"id": "alice", "name": "Alice"}, {"id": "bob", "name": "Bob"}],
        "segments": [
            _seg(0, 10, "alice", "Alice", text="hello there friend", words_count=3,
                 sentiment=_sent("positive", "joy")),
            _seg(10, 20, "bob", "Bob", text="hi", words_count=1,
                 sentiment=_sent("neutral", "neutral")),
            _seg(20, 30, "alice", "Alice", text="ok", words_count=1,
                 sentiment=_sent("neutral", "neutral")),
        ],
        "passes_run": ["diarization", "transcription", "sentiment"],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)
    yield {"dir": d, "json": json_path, "config": config_path}
    shutil.rmtree(d, ignore_errors=True)


def _read(path):
    with open(path) as f:
        return json.load(f)


class TestMetricsOrchestration:
    def test_happy_path_writes_json_and_markdown(self, tmp_workspace):
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 0
        data = _read(tmp_workspace["json"])
        assert "contribution_metrics" in data
        assert "metrics_config" in data
        assert "metrics" in data["passes_run"]
        # Markdown is sibling file.
        md_path = os.path.join(tmp_workspace["dir"], "session.diarization.metrics.md")
        assert os.path.exists(md_path)
        with open(md_path, encoding="utf-8") as f:
            assert "# Session Metrics" in f.read()

    def test_out_path_leaves_input_unchanged(self, tmp_workspace):
        original = _read(tmp_workspace["json"])
        out_json = os.path.join(tmp_workspace["dir"], "out.json")
        rc = metrics.main([tmp_workspace["json"], "--out", out_json,
                           "--config", tmp_workspace["config"]])
        assert rc == 0
        assert _read(tmp_workspace["json"]) == original
        assert "contribution_metrics" in _read(out_json)

    def test_report_path_writes_markdown_to_specified_path(self, tmp_workspace):
        report_path = os.path.join(tmp_workspace["dir"], "custom-report.md")
        rc = metrics.main([tmp_workspace["json"], "--report", report_path,
                           "--config", tmp_workspace["config"]])
        assert rc == 0
        assert os.path.exists(report_path)

    def test_idempotent_rerun_overwrites_and_dedupes_passes_run(self, tmp_workspace):
        metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        data = _read(tmp_workspace["json"])
        assert data["passes_run"].count("metrics") == 1

    def test_missing_transcription_pass_exit_2(self, tmp_workspace):
        data = _read(tmp_workspace["json"])
        data["passes_run"] = ["diarization", "sentiment"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_missing_sentiment_pass_exit_2(self, tmp_workspace):
        data = _read(tmp_workspace["json"])
        data["passes_run"] = ["diarization", "transcription"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_segment_missing_text_exit_2(self, tmp_workspace):
        data = _read(tmp_workspace["json"])
        del data["segments"][1]["text"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_missing_metrics_config_block_exit_3(self, tmp_workspace):
        with open(tmp_workspace["config"], "w") as f:
            f.write("other:\n  key: value\n")
        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 3
```

- [ ] **Step 2: Run, confirm import fails**

```bash
py -3.14 -m pytest tests/metrics/test_orchestration.py -v
```

Expected: ImportError on `import metrics`.

- [ ] **Step 3: Implement `metrics.py`**

Create `target-vad/metrics.py`:

```python
"""Contribution metrics pass (Phase 3) — see docs/superpowers/specs/2026-05-16-contribution-metrics-design.md.

Reads a post-2A+2B diarization JSON, computes per-speaker + session-level
aggregates plus a bucketed activity timeline and deterministic narrative
highlights, writes the contribution_metrics block back atomically, and
renders a sibling Markdown report.
"""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from typing import Dict, List

import yaml
from rich.console import Console

from modes.metrics import aggregator, renderer

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CONFIG_OR_IO = 3


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _atomic_write_json(path: str, data: dict) -> None:
    dirname = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", suffix=".json", dir=dirname)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _merged_speech_seconds(segments: List[Dict]) -> float:
    """Total wall-clock seconds where at least one speaker is talking."""
    if not segments:
        return 0.0
    intervals = sorted((s["start"], s["end"]) for s in segments)
    merged: List[List[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


def _build_metrics_block(data: Dict, cfg: Dict) -> Dict:
    segments = data["segments"]
    duration_s = float(data.get("duration_s", 0.0))
    bucket_seconds = int(cfg["bucket_seconds"])

    participation = aggregator.aggregate_participation(segments)
    sentiment = aggregator.aggregate_sentiment(segments)
    turn_taking = aggregator.aggregate_turn_taking(segments)
    pairwise = aggregator.aggregate_pairwise(segments)
    timeline = aggregator.aggregate_timeline(segments, duration_s, bucket_seconds)
    highlights = aggregator.select_highlights(
        segments, timeline, int(cfg["top_k_highlights"]), int(cfg["quote_max_chars"])
    )

    merged_speech = _merged_speech_seconds(segments)
    silence_s = round(max(0.0, duration_s - merged_speech), 2)

    session_block = {
        "duration_s": duration_s,
        "speech_duration_s": participation["session"]["speech_duration_s"],
        "silence_duration_s": silence_s,
        "total_segments": participation["session"]["total_segments"],
        "total_words": participation["session"]["total_words"],
        "unique_speakers": participation["session"]["unique_speakers"],
        "identified_speakers": participation["session"]["identified_speakers"],
        "unknown_segments": participation["session"]["unknown_segments"],
        "polarity_distribution": sentiment["session"]["polarity_distribution"],
        "emotion_distribution": sentiment["session"]["emotion_distribution"],
    }

    # Speakers list — order = first-appearance (matches enrolled_users_matched convention).
    seen = set()
    speakers_ordered: List[str] = []
    for s in segments:
        sid = s["speaker_id"]
        if sid not in seen:
            seen.add(sid)
            speakers_ordered.append(sid)
    name_lookup = {s["speaker_id"]: s["speaker"] for s in segments}

    speakers: List[Dict] = []
    for sid in speakers_ordered:
        speakers.append({
            "speaker_id": sid,
            "speaker": name_lookup[sid],
            "participation": participation["per_speaker"][sid],
            "sentiment": sentiment["per_speaker"][sid],
            "turn_taking": turn_taking["per_speaker"][sid],
        })

    return {
        "session": session_block,
        "speakers": speakers,
        "pairwise_followers": pairwise,
        "timeline": timeline,
        "highlights": highlights,
    }


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD — Contribution Metrics Pass (Phase 3)")
    parser.add_argument("input", help="Path to a transcribed + sentiment-classified diarization JSON")
    parser.add_argument("--out", default=None, help="Output JSON path (default: in-place atomic write)")
    parser.add_argument("--report", default=None, help="Markdown report path (default: <input-stem>.metrics.md)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)

    # Load JSON.
    if not os.path.exists(args.input):
        console.print(f"[red]Diarization JSON not found:[/] {args.input}")
        return EXIT_BAD_INPUT
    try:
        with open(args.input) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Diarization JSON is malformed:[/] {exc.msg} at offset {exc.pos}")
        return EXIT_BAD_INPUT

    if "segments" not in data:
        console.print("[red]Diarization JSON is missing the [bold]segments[/] field.[/]")
        return EXIT_BAD_INPUT

    passes_run = data.get("passes_run", [])
    if "transcription" not in passes_run:
        console.print("[red]This JSON has not been transcribed yet.[/] "
                      "[dim]Run [bold]transcribe.py[/] first.[/]")
        return EXIT_BAD_INPUT
    if "sentiment" not in passes_run:
        console.print("[red]This JSON has not been sentiment-classified yet.[/] "
                      "[dim]Run [bold]sentiment.py[/] first.[/]")
        return EXIT_BAD_INPUT

    for i, seg in enumerate(data["segments"]):
        missing = [k for k in ("text", "words", "sentiment") if k not in seg]
        if missing:
            console.print(
                f"[red]Segment {i} is missing field(s) {missing!r}.[/] "
                "[dim]This JSON is in a partial/inconsistent state — rerun the prior pass.[/]"
            )
            return EXIT_BAD_INPUT

    # Load config.
    try:
        cfg_full = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/] {args.config}")
        return EXIT_CONFIG_OR_IO
    cfg = cfg_full.get("metrics")
    if not cfg:
        console.print(f"[red]Config is missing the [bold]metrics:[/] block.[/]")
        return EXIT_CONFIG_OR_IO

    # Build metrics + write.
    try:
        block = _build_metrics_block(data, cfg)
    except Exception as exc:
        console.print(f"[red]Failed to aggregate metrics:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    data["contribution_metrics"] = block
    data["metrics_config"] = {
        "bucket_seconds": int(cfg["bucket_seconds"]),
        "top_k_highlights": int(cfg["top_k_highlights"]),
        "quote_max_chars": int(cfg["quote_max_chars"]),
        "analyzed_at": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "passes_run" not in data:
        data["passes_run"] = []
    if "metrics" not in data["passes_run"]:
        data["passes_run"].append("metrics")

    out_json = args.out or args.input
    try:
        _atomic_write_json(out_json, data)
    except Exception as exc:
        console.print(f"[red]Failed to write metrics JSON:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    # Markdown render + write.
    session_meta = {
        "audio_file": data.get("audio_file", ""),
        "analyzed_at": data["metrics_config"]["analyzed_at"],
    }
    md = renderer.render_markdown(block, session_meta)

    if args.report:
        report_path = args.report
    else:
        stem, _ext = os.path.splitext(args.input)
        # Strip a trailing ".diarization" so a typical input becomes "session.metrics.md".
        if stem.endswith(".diarization"):
            stem = stem[:-len(".diarization")]
        report_path = stem + ".metrics.md"

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as exc:
        console.print(f"[red]Failed to write Markdown report:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    speakers_n = len(block["speakers"])
    segs_n = block["session"]["total_segments"]
    words_n = block["session"]["total_words"]
    hl_n = len(block["highlights"])
    console.print(
        f"[green]Metrics written:[/] {speakers_n} speakers, {segs_n} segments, "
        f"{words_n} words, {hl_n} highlights."
    )
    console.print(f"  JSON     → {out_json}")
    console.print(f"  Markdown → {report_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all metrics tests**

```bash
py -3.14 -m pytest tests/metrics/ -v
```

Expected: 22 (aggregator) + 5 (renderer) + 8 (orchestration) = **35 passed.**

- [ ] **Step 5: Run the entire suite**

```bash
py -3.14 -m pytest tests/ -q
```

Expected: 172 (baseline) + 35 (new) = **207 passed.** If less, investigate before committing.

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/metrics.py target-vad/tests/metrics/test_orchestration.py
git -C c:/repos/TVAD commit -m "feat(metrics): add metrics.py CLI orchestration"
```

---

## Task 10: Manual smoke against the real fixture

**Files:** none changed.

- [ ] **Step 1: Run `metrics.py` against the existing real JSON**

The repo already has `Voice 001 short.wav.diarization.json` (post 2A + 2B). Run from `c:\repos\TVAD\target-vad\`:

```bash
py -3.14 metrics.py "../Voice 001 short.wav.diarization.json"
```

Expected:
- Exit code 0
- Console summary line printed: speakers/segments/words/highlights
- The JSON gains `contribution_metrics`, `metrics_config`, and `metrics` in `passes_run`
- A new file `Voice 001 short.wav.metrics.md` appears next to the JSON

- [ ] **Step 2: Eyeball the JSON block**

```bash
py -3.14 -c "import json; d=json.load(open('../Voice 001 short.wav.diarization.json')); cm=d['contribution_metrics']; print(json.dumps(cm['session'], indent=2)); print('speakers:', [s['speaker'] for s in cm['speakers']]); print('highlights:', len(cm['highlights']))"
```

Expected: a coherent `session` dict, the 2 speakers from prior smoke tests, and 1+ highlights.

- [ ] **Step 3: Eyeball the Markdown**

Open `Voice 001 short.wav.metrics.md` in VSCode preview or any Markdown viewer. Check:

- Title shows audio filename
- Header line shows duration, speech, silence
- Notable moments section has at least the longest_segment entry
- Participation, polarity, emotion, turn-taking tables render correctly
- Disgust footnote present
- Caveat about 'unknown' present at bottom

- [ ] **Step 4: Idempotent rerun check**

```bash
py -3.14 metrics.py "../Voice 001 short.wav.diarization.json"
py -3.14 -c "import json; d=json.load(open('../Voice 001 short.wav.diarization.json')); print('passes_run:', d['passes_run']); assert d['passes_run'].count('metrics') == 1"
```

Expected: `passes_run: [..., 'metrics']` with `metrics` appearing exactly once.

- [ ] **Step 5: Update auto-memory project file**

Append the validated-2026-05-16 paragraph to `C:\Users\AI PC\.claude\projects\c--repos-TVAD\memory\project_tvad.md` documenting Phase 3 shipped. The memory should note:
- `metrics.py` exists as the third analysis pass
- Adds `contribution_metrics` top-level block + sibling `*.metrics.md` report
- Validated against `Voice 001 short.wav.diarization.json` on 2026-05-16
- Test count: ~207 passing
- Reserved Phase 2C engagement slot is still unfilled — the next pass after Phase 3.

- [ ] **Step 6: Stage and commit the smoke-test artifacts**

The Voice 001 fixtures may have changed (the JSON gained `contribution_metrics`, a new `.metrics.md` appeared). They are listed as untracked or modified in `git status`. Decide with the user whether to commit those artifacts (they're useful as future regression-test snapshots) or leave them ignored. Default: commit them so future contributors can diff against a known-good Markdown report.

```bash
git -C c:/repos/TVAD status
# If acceptable, stage and commit:
git -C c:/repos/TVAD add "Voice 001 short.wav.diarization.json" "Voice 001 short.wav.metrics.md"
git -C c:/repos/TVAD commit -m "chore(metrics): snapshot smoke-test outputs for Voice 001"
```

---

## Self-review checklist (after all tasks)

- [ ] All 8 spec sections (Purpose, Architecture, Output schema, Markdown shape, Components, CLI, Configuration, Aggregator behavior, Highlights rules, Error handling, Testing) have at least one implementing task
- [ ] No "TODO" / "TBD" / "fill in later" strings in any committed file
- [ ] Test count baseline (172) increases by approximately 35
- [ ] `metrics.py` follows the same atomic-write pattern as `transcribe.py` and `sentiment.py`
- [ ] No new dependencies added to `requirements.txt`
- [ ] All commits are scoped — one logical change per commit
