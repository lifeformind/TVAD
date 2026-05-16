# Recurring-Unknown Cluster Identity — Design

**Date:** 2026-05-16
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-14-classroom-diarization-design.md`](./2026-05-14-classroom-diarization-design.md) (S1, shipped) and [`2026-05-16-contribution-metrics-design.md`](./2026-05-16-contribution-metrics-design.md) (Phase 3, shipped)

## Purpose

S1 currently collapses every unenrolled diarization cluster to the literal string `"unknown"`. A session with three unenrolled speakers shows as one anonymous bucket — pyannote's clustering work (which already groups recurring voices into stable cluster ids like `SPEAKER_00`, `SPEAKER_01`) is discarded at the identifier boundary.

This costs real fidelity in downstream consumers. Phase 3's per-speaker metrics report "unknown" as if it were one person; a facilitator can't tell whether the unknown bucket represents one quiet participant or three unenrolled students. Phase 2C engagement labels (when they land) will face the same problem.

The fix is to preserve pyannote's cluster id for unmatched clusters that pass a substance threshold (recurring AND substantive), while keeping the existing `"unknown"` catchall for one-off / brief unmatched segments that are statistically noise. The threshold gates which clusters get tracked as named entities; pyannote's own clustering provides the recurring-voice identity for free.

## Architecture

Behavior change is scoped to one method (`ClusterIdentifier.label_clusters`) plus two output-helper updates and three small renderer/aggregator fixups downstream.

```
For each pyannote cluster:
    embedding = ECAPA(cluster_audio)
    best_id, best_score = cosine_match(embedding, voiceprints)
    if best_score >= identification_threshold:
        label = best_id                              # enrolled match — unchanged
    else:
        seg_count   = len(cluster.segments)
        total_secs  = sum(end - start for start, end in cluster.segments)
        if seg_count >= unknown_min_segments AND total_secs >= unknown_min_seconds:
            label = cluster_id                       # preserve pyannote id, e.g. "SPEAKER_00"
        else:
            label = "unknown"                        # catchall (existing behavior)
```

Two new threshold knobs in `config.yaml` under `diarization:` — `unknown_min_segments` (default 2) and `unknown_min_seconds` (default 10.0). Both required (logical AND). A cluster failing either gate keeps the existing `"unknown"` behavior; a cluster passing both gets its pyannote literal id preserved verbatim through to the JSON output, RTTM, and all downstream passes.

**Two existing code paths that fast-track to `"unknown"` need the threshold check added:**

1. Empty voiceprint store (`label_clusters` line 60 currently `return {cid: "unknown" for cid in clusters}`) — instead, walk each cluster and apply the threshold; substantive clusters get their pyannote id.
2. Embedding failure (`except` block at line 70 currently writes `labels[cluster_id] = "unknown"`) — also apply the threshold so a substantive cluster with a broken embedding still gets tracked.

## Output schema additions

**Per-segment fields** — shape unchanged; only the `speaker_id` / `speaker` values widen:

```json
{"start": 12.4, "end": 18.2, "speaker_id": "SPEAKER_00", "speaker": "SPEAKER_00", ...}
```

`speaker_id` and `speaker` are both the literal pyannote string for recurring-unknown segments. No prettification, no qualifier — the uppercase `SPEAKER_NN` naming itself signals "unenrolled" to a reader. Enrolled segments are unchanged (`speaker_id: "siddharth"`, `speaker: "Siddharth Jain"`). Below-threshold segments remain `speaker_id: "unknown"`, `speaker: "unknown"`.

**New top-level `unknown_speakers_observed` field** — sibling to `enrolled_users_matched`, lists only clusters that cleared the threshold:

```json
"unknown_speakers_observed": [
  {"id": "SPEAKER_00", "segment_count": 4, "talk_seconds": 32.18},
  {"id": "SPEAKER_02", "segment_count": 7, "talk_seconds": 48.91}
]
```

Always present (empty list when no clusters qualify). First-appearance order by start time of the earliest segment in the cluster. `segment_count` and `talk_seconds` are summary stats analogous to `name` for enrolled speakers — useful for any consumer that wants to surface "speakers seen in this session" without iterating segments.

**Top-level `config` block** — gains the two new threshold knobs for reproducibility:

```json
"config": {
  "pyannote_pipeline": "pyannote/speaker-diarization-3.1",
  "identification_threshold": 0.55,
  "intro_override_warn_threshold": 0.3,
  "unknown_min_segments": 2,
  "unknown_min_seconds": 10.0,
  ...
}
```

**`enrolled_users_matched` unchanged** — still only lists enrolled-and-matched speakers, but the helper that produces it must now filter both `"unknown"` AND pyannote-id segments (since those aren't enrolled).

**RTTM unchanged.** `write_rttm()` already writes `speaker_id` verbatim into column 8. Pyannote-id recurring-unknown segments will land as `SPEAKER_00` etc., which is conventional RTTM format.

## Components

| Path | Status | Responsibility |
|---|---|---|
| `target-vad/modes/diarization/identifier.py` | modify | `label_clusters()` adds threshold-based fallback for unmatched clusters; signature accepts `unknown_min_segments`, `unknown_min_seconds` via constructor or per-call |
| `target-vad/modes/diarization/output.py` | modify | `_enrolled_users_in_first_appearance_order()` gains `enrolled_ids: set[str]` parameter; new `_unknown_speakers_observed_in_first_appearance_order()` helper; `write_json()` emits `unknown_speakers_observed` field |
| `target-vad/diarize.py` | modify | Read new config knobs; pass them into `ClusterIdentifier`; pass enrolled-id set into `write_json` |
| `target-vad/config.yaml` | modify | Add `unknown_min_segments: 2` and `unknown_min_seconds: 10.0` under `diarization:` |
| `target-vad/metrics.py` | modify | Recompute `identified_speakers` from `enrolled_users_matched`; add `recurring_unknown_speakers` count to `session` block |
| `target-vad/modes/metrics/renderer.py` | modify | Header line gains 3-way breakdown (enrolled / recurring unknown / catchall) |
| `target-vad/tests/diarization/test_identifier.py` | modify | Tests for threshold preservation, threshold collapse, mixed enrollment scenarios |
| `target-vad/tests/diarization/test_output.py` | modify | Tests for `unknown_speakers_observed` emission and `enrolled_users_matched` filtering |
| `target-vad/tests/metrics/test_orchestration.py` | modify | One new test exercising a fixture with a recurring-unknown speaker |
| `target-vad/tests/metrics/test_renderer.py` | modify | Update golden fixture; add omission test for the catchall portion when zero unknowns |

`identifier.py` and `output.py` together carry the substantive logic. `diarize.py` is a pass-through that wires config into the new identifier kwargs. `metrics.py` and `renderer.py` see small downstream fixups; everything else (transcribe, sentiment, the other aggregators) is unaffected because they're agnostic to specific `speaker_id` values.

## Ripple to downstream

**Phase 2A (transcribe.py) and Phase 2B (sentiment.py):** unaffected. Both walk the segment list and write text-only fields per segment; neither inspects `speaker_id` semantics.

**Phase 3 (metrics.py) — three fixups:**

1. **`session.identified_speakers`** must be recomputed. Today it counts `unique speaker_ids != "unknown"`, which would incorrectly include recurring unknowns. New definition: `len(data["enrolled_users_matched"])`. The orchestrator already reads that field.
2. **New `session.recurring_unknown_speakers`** field. Value: `len(data["unknown_speakers_observed"])`. Adjacent to `identified_speakers` in the `contribution_metrics.session` block.
3. **Header line in the Markdown report** updates from:
   > **Speakers:** 5 (2 identified, 3 unknown)

   to:
   > **Speakers:** 5 (2 enrolled, 2 recurring unknown, 1 catchall) · ...

   "catchall" = 1 if any segment has `speaker_id == "unknown"` else 0. The per-speaker rows in Participation / Sentiment / Turn-taking tables work unchanged — they iterate by `speaker_id` and naturally produce `SPEAKER_00`, `SPEAKER_01`, `unknown` etc. as their own rows.

## Configuration

```yaml
diarization:
  pyannote_pipeline: "pyannote/speaker-diarization-3.1"
  identification_threshold: 0.55
  intro_override_warn_threshold: 0.3
  unknown_min_segments: 2            # new
  unknown_min_seconds: 10.0          # new
```

Both new knobs are required-with-defaults. A missing block falls back to the defaults via the same `cfg.get("unknown_min_segments", 2)` idiom the rest of the codebase uses.

## Edge cases / conflict resolution

| Case | Behavior |
|---|---|
| Empty voiceprint store + 3 clusters all clearing threshold | All 3 get pyannote ids; `enrolled_users_matched: []`; `unknown_speakers_observed` lists all 3 |
| Enrollment with `id == "SPEAKER_00"` (collision) | Reserved naming pattern — document that user ids matching `^SPEAKER_\d+$` are not supported. No defensive code; the case would be very rare and would produce confusing-but-not-corrupt output |
| Cluster with 2 segments totaling 0.8 s | `seg_count >= 2` passes but `total_secs >= 10.0` fails — collapses to `"unknown"` |
| Cluster with 1 segment of 30 s | `total_secs >= 10.0` passes but `seg_count >= 2` fails — collapses to `"unknown"` |
| Cluster embedding fails inside `ClusterIdentifier._extract_cluster_audio` | Falls through to the threshold check on the cluster's segments — failed embedding doesn't auto-collapse to `"unknown"` if the cluster is substantive |
| `--introductions` manifest run | Unaffected. Manifest-driven session enrollments are applied before identification; clusters matching manifest voiceprints get the manifest id; clusters that don't match go through the standard enrolled-then-threshold-then-unknown path |
| Re-running `metrics.py` on a pre-upgrade JSON (no `unknown_speakers_observed` field) | `metrics.py` reads `data.get("unknown_speakers_observed", [])` — treats as empty. `identified_speakers` still uses `len(enrolled_users_matched)`. Backward compatible read path |

## Testing approach

`tests/diarization/test_identifier.py` (~4 new tests):

- Cluster with 2 segments + 12 s total, embedding doesn't match any voiceprint → label = cluster id (`"SPEAKER_00"` in fixture)
- Cluster with 3 segments + 4 s total → label = `"unknown"` (duration gate fails)
- Cluster with 1 segment + 30 s total → label = `"unknown"` (segment-count gate fails)
- Empty voiceprint store + 2 clusters (one substantive, one brief) → substantive one gets pyannote id; brief one gets `"unknown"`

`tests/diarization/test_output.py` (~2 new tests):

- `write_json` emits `unknown_speakers_observed: [...]` when segments include pyannote-id speakers; with correct `segment_count` and `talk_seconds`
- `_enrolled_users_in_first_appearance_order` correctly skips both `"unknown"` and pyannote-id segments when computing the enrolled list

`tests/metrics/test_orchestration.py` (~1 new test):

- Fixture with 1 enrolled + 1 recurring-unknown + 1 catchall-unknown segment → `session.identified_speakers == 1`, `session.recurring_unknown_speakers == 1`

`tests/metrics/test_renderer.py`:

- Golden fixture updated for the new header line
- Existing 4 omission tests continue to pass without modification

Total: ~7 new tests + 1 golden update. Expected test count: 210 (current) → ~217.

## Migration / backward compatibility

**Breaking change for any consumer that hardcoded `speaker_id == "unknown"` as "all unmatched".** The only internal such consumer is Phase 3's renderer header (handled in this spec). The earlier in-session-enrollment design memory notes that S1 has no production consumers, so no external compatibility is owed.

JSONs written before this upgrade lack the `unknown_speakers_observed` field. The Phase 3 read path uses `data.get("unknown_speakers_observed", [])`, treating absence as "no recurring unknowns" — backward compatible.

Existing tests that assert `speaker_id == "unknown"` on unmatched clusters need to be reviewed: any test whose fixture has a substantive multi-segment unmatched cluster needs to either (a) be reduced below the threshold so the assertion stays valid, or (b) update its assertion to the pyannote literal. The S1 test suite is the main affected one.

## Out of scope

- **Cross-session unknown re-identification** — `SPEAKER_00` in session A is not the same entity as `SPEAKER_00` in session B (pyannote ids are per-run). A future "promote recurring unknown to enrollment" workflow could let a facilitator name an unknown after the fact, but that's its own design
- **Sub-threshold cluster preservation** — clusters that fail the threshold gate stay collapsed to `"unknown"`. No `unknown_speakers_observed_subthreshold` list. If a consumer wants per-cluster fidelity for noisy speakers, raise this in a follow-up
- **Per-cluster cosine reporting** — the JSON doesn't currently expose how close each unmatched cluster came to clearing the identification threshold. Useful for tuning but out of scope here
- **S2 kiosk talkback** — uses session-scoped snapshot matching, not pyannote diarization. Unaffected
- **Phase 2C engagement labels** — still deferred pending the local-vs-API backend decision

## Open questions

None. All decisions resolved during brainstorming 2026-05-16.
