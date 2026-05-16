# Recurring-Unknown Cluster Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve pyannote's stable cluster ids (e.g. `SPEAKER_00`) for unmatched diarization clusters that pass a recurrence-and-substance threshold, while keeping the existing `"unknown"` catchall for one-off / brief unmatched segments. Adds a top-level `unknown_speakers_observed` sibling to `enrolled_users_matched`, and gives Phase 3 metrics a 3-way speaker breakdown (enrolled / recurring unknown / catchall).

**Architecture:** Surgical change to `ClusterIdentifier.label_clusters()` adds a threshold-based fallback for unmatched clusters. `write_json()` and one helper in `output.py` are extended to emit the new top-level field. `diarize.py` wires two new config knobs through. Phase 3 (`metrics.py` + `renderer.py`) gets small fixups for the recomputed `identified_speakers` count and the Markdown report header.

**Tech Stack:** Python 3.14 (`py -3.14`, never `python` — 3.12 lacks the dep stack). No new dependencies. All changes are pure logic over existing data structures.

**Spec:** [`docs/superpowers/specs/2026-05-16-recurring-unknown-clusters-design.md`](../specs/2026-05-16-recurring-unknown-clusters-design.md). Read once before starting Task 1.

**Working directory:** `c:\repos\TVAD\target-vad\` for python/pytest. Git commands run from `c:\repos\TVAD\`.

---

## File Structure

Files this plan modifies (relative to `target-vad/`):

| Path | Status | Responsibility |
|---|---|---|
| `config.yaml` | modify | Add `unknown_min_segments` and `unknown_min_seconds` under `diarization:` |
| `modes/diarization/identifier.py` | modify | `ClusterIdentifier` constructor gains threshold kwargs; `label_clusters()` applies threshold to unmatched clusters across all three paths (matched, empty-store, embedding-fail) |
| `modes/diarization/output.py` | modify | `_enrolled_users_in_first_appearance_order()` gains `enrolled_ids` parameter; new `_unknown_speakers_observed_in_first_appearance_order()` helper; `write_json()` accepts `enrolled_ids` and emits `unknown_speakers_observed` |
| `diarize.py` | modify | Read new config knobs; pass to `ClusterIdentifier`; pass `enrolled_ids` set to `write_json`; emit new keys in the JSON config block |
| `metrics.py` | modify | Recompute `identified_speakers` from `enrolled_users_matched`; add `recurring_unknown_speakers` field |
| `modes/metrics/renderer.py` | modify | Header line: 3-way breakdown (enrolled / recurring unknown / catchall) |
| `tests/diarization/test_identifier.py` | modify | Add `TestClusterIdentifierUnknownThreshold` class (4 tests) |
| `tests/diarization/test_output.py` | modify | Add tests for `unknown_speakers_observed` emission and filtered `enrolled_users_matched` |
| `tests/metrics/test_orchestration.py` | modify | Add 1 test for the 3-way breakdown |
| `tests/metrics/test_renderer.py` | modify | Update existing golden fixture; add omission/transition tests |
| `tests/metrics/fixtures/golden_report.md` | modify | New header line |

No external dependencies. No new files. ~7 new tests.

---

## Task 1: Add threshold knobs to `config.yaml`

**Files:**
- Modify: `target-vad/config.yaml`

- [ ] **Step 1: Inspect current `diarization:` block**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['diarization'])"
```

Confirm existing keys: `pyannote_pipeline`, `identification_threshold`, `intro_override_warn_threshold`, `centroid_max_sample_seconds`, `hf_token_env_var` (or similar — record exact set before modifying).

- [ ] **Step 2: Add the two new knobs under `diarization:`**

Open `target-vad/config.yaml`. Inside the `diarization:` block (do NOT touch other top-level blocks), append at the end of that block:

```yaml
  unknown_min_segments: 2          # minimum segment count for a recurring-unknown cluster to keep its pyannote id
  unknown_min_seconds: 10.0        # minimum total talk seconds for a recurring-unknown cluster to keep its pyannote id
```

Use 2-space indentation matching the existing keys in the block.

- [ ] **Step 3: Verify YAML parses and the two new keys appear**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "import yaml; d=yaml.safe_load(open('config.yaml'))['diarization']; print('unknown_min_segments:', d['unknown_min_segments']); print('unknown_min_seconds:', d['unknown_min_seconds'])"
```

Expected:
```
unknown_min_segments: 2
unknown_min_seconds: 10.0
```

- [ ] **Step 4: Run the full test suite to confirm baseline**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 210 passed. (Phase 3 baseline.) STOP if not — the existing suite must be green before changing more code.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/config.yaml
git -C c:/repos/TVAD commit -m "feat(diarization): add unknown_min_segments and unknown_min_seconds config knobs"
```

---

## Task 2: `ClusterIdentifier` — apply threshold to unmatched clusters

**Files:**
- Modify: `target-vad/modes/diarization/identifier.py`
- Modify: `target-vad/tests/diarization/test_identifier.py`

- [ ] **Step 1: Add failing tests**

Append a new test class at the end of `target-vad/tests/diarization/test_identifier.py`. Re-use the existing `fake_embedder`, `fake_store`, `unit_vec`, `make_silence`, and `SR` fixtures defined at module scope:

```python
class TestClusterIdentifierUnknownThreshold:
    """Behaviour for the recurring-unknown threshold (spec 2026-05-16)."""

    def test_substantive_unmatched_cluster_gets_pyannote_id(self, fake_embedder, fake_store):
        """2 segments + 12s total + no enrollment match → keep cluster_id."""
        fake_store.get_all.return_value = {"alice": unit_vec(seed=1)}
        fake_embedder.extract.return_value = unit_vec(seed=99)  # far from alice
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
            unknown_min_segments=2, unknown_min_seconds=10.0,
        )
        audio = make_silence(20.0)
        clusters = {"SPEAKER_00": [(0.0, 6.0), (10.0, 16.0)]}  # 2 segs, 12s
        labels = identifier.label_clusters(audio, sample_rate=SR, clusters=clusters)
        assert labels == {"SPEAKER_00": "SPEAKER_00"}

    def test_brief_unmatched_cluster_collapses_to_unknown(self, fake_embedder, fake_store):
        """3 segments but 4s total → 'unknown' (duration gate fails)."""
        fake_store.get_all.return_value = {"alice": unit_vec(seed=1)}
        fake_embedder.extract.return_value = unit_vec(seed=99)
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
            unknown_min_segments=2, unknown_min_seconds=10.0,
        )
        audio = make_silence(10.0)
        clusters = {"SPEAKER_00": [(0.0, 1.5), (3.0, 4.5), (5.0, 6.0)]}  # 3 segs, 4s
        labels = identifier.label_clusters(audio, sample_rate=SR, clusters=clusters)
        assert labels == {"SPEAKER_00": "unknown"}

    def test_single_long_unmatched_cluster_collapses_to_unknown(self, fake_embedder, fake_store):
        """1 segment of 30s → 'unknown' (segment-count gate fails)."""
        fake_store.get_all.return_value = {"alice": unit_vec(seed=1)}
        fake_embedder.extract.return_value = unit_vec(seed=99)
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
            unknown_min_segments=2, unknown_min_seconds=10.0,
        )
        audio = make_silence(35.0)
        clusters = {"SPEAKER_00": [(0.0, 30.0)]}  # 1 seg, 30s
        labels = identifier.label_clusters(audio, sample_rate=SR, clusters=clusters)
        assert labels == {"SPEAKER_00": "unknown"}

    def test_empty_voiceprint_store_still_applies_threshold(self, fake_embedder, fake_store):
        """No voiceprints + 2 clusters (one substantive, one brief) → substantive keeps id, brief collapses."""
        fake_store.get_all.return_value = {}  # empty store
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
            unknown_min_segments=2, unknown_min_seconds=10.0,
        )
        audio = make_silence(20.0)
        clusters = {
            "SPEAKER_00": [(0.0, 6.0), (8.0, 14.0)],   # 2 segs, 12s — substantive
            "SPEAKER_01": [(15.0, 15.5)],               # 1 seg, 0.5s — brief
        }
        labels = identifier.label_clusters(audio, sample_rate=SR, clusters=clusters)
        assert labels == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "unknown"}
```

- [ ] **Step 2: Run, confirm 4 failures**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/diarization/test_identifier.py::TestClusterIdentifierUnknownThreshold -v
```

Expected: 4 failures with `TypeError: __init__() got an unexpected keyword argument 'unknown_min_segments'`.

- [ ] **Step 3: Modify `ClusterIdentifier` constructor + add threshold helper**

Open `target-vad/modes/diarization/identifier.py`. Update the constructor signature and add a private threshold helper. Replace the existing `__init__` (lines 29-39) with:

```python
    def __init__(
        self,
        embedder,
        enrollment_store,
        threshold: float = 0.55,
        max_sample_seconds: float = 30.0,
        unknown_min_segments: int = 2,
        unknown_min_seconds: float = 10.0,
    ):
        self.embedder = embedder
        self.enrollment_store = enrollment_store
        self.threshold = threshold
        self.max_sample_seconds = max_sample_seconds
        self.unknown_min_segments = unknown_min_segments
        self.unknown_min_seconds = unknown_min_seconds
```

Add a small helper method anywhere inside the class (a good spot is right before `_extract_cluster_audio`):

```python
    def _threshold_label(self, cluster_id: str, segments: list) -> str:
        """Return cluster_id if the cluster passes the recurrence+substance threshold, else 'unknown'."""
        if len(segments) < self.unknown_min_segments:
            return "unknown"
        total_seconds = sum(end - start for start, end in segments)
        if total_seconds < self.unknown_min_seconds:
            return "unknown"
        return cluster_id
```

- [ ] **Step 4: Apply the threshold in all three unmatched paths**

In `label_clusters()`, three sites currently emit `"unknown"` unconditionally for an unmatched cluster. Replace each with a call to `_threshold_label`.

**Site 1 — empty voiceprint store** (around line 60):

```python
        if not voiceprints:
            # Nothing to compare against — apply threshold per cluster.
            return {cid: self._threshold_label(cid, segs) for cid, segs in clusters.items()}
```

**Site 2 — embedding failure** (around line 70, inside the `except`):

```python
            except Exception as exc:  # pragma: no cover — exercised via mocks
                logger.warning("Embedding failed for cluster %s: %s — applying threshold", cluster_id, exc)
                labels[cluster_id] = self._threshold_label(cluster_id, segments)
```

**Site 3 — cosine below threshold** (the existing `_best_label` returns `"unknown"`). The cleanest fix is to wrap the `_best_label` call in `label_clusters` so the threshold applies *after* the cosine decision. Change the success branch of the try/except to:

```python
                cluster_audio = self._extract_cluster_audio(audio, sample_rate, segments)
                embedding = self.embedder.extract(cluster_audio, sample_rate=sample_rate)
                best = self._best_label(embedding, voiceprints)
                if best == "unknown":
                    labels[cluster_id] = self._threshold_label(cluster_id, segments)
                else:
                    labels[cluster_id] = best
```

(`_best_label` itself stays unchanged — it still returns `"unknown"` when no enrolled voiceprint clears the threshold. `label_clusters` is now responsible for the further "substantive enough to track?" decision.)

- [ ] **Step 5: Run the new tests, confirm all 4 pass**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/diarization/test_identifier.py -v
```

Expected: all `TestClusterIdentifier*` tests pass. The previously-existing tests in this file pass via the threshold helper's defaults (2 segments, 10.0 seconds), but verify nothing regresses.

If any pre-existing test fails because its fixture has a sub-threshold cluster that used to be `"unknown"` and now gets a pyannote id — STOP and report which test. The plan author needs to know.

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/diarization/identifier.py target-vad/tests/diarization/test_identifier.py
git -C c:/repos/TVAD commit -m "feat(diarization): preserve pyannote cluster id for substantive unmatched clusters"
```

---

## Task 3: `output.py` — emit `unknown_speakers_observed` and filter `enrolled_users_matched`

**Files:**
- Modify: `target-vad/modes/diarization/output.py`
- Modify: `target-vad/tests/diarization/test_output.py`

- [ ] **Step 1: Add failing tests**

Append two new tests to `target-vad/tests/diarization/test_output.py` inside the `TestWriteJson` class (or as a new class — either works; the plan uses additions to existing class for brevity).

```python
    def test_emits_unknown_speakers_observed(self, temp_dir):
        """Pyannote-id segments produce an unknown_speakers_observed list with counts."""
        segments = [
            DiarizationSegment(0.0, 5.0, "alice", "Alice"),
            DiarizationSegment(5.0, 10.0, "SPEAKER_00", "SPEAKER_00"),
            DiarizationSegment(10.0, 13.0, "SPEAKER_00", "SPEAKER_00"),
            DiarizationSegment(13.0, 15.0, "unknown", "unknown"),
            DiarizationSegment(15.0, 20.0, "SPEAKER_01", "SPEAKER_01"),
        ]
        out = os.path.join(temp_dir, "out.json")
        write_json(
            out, audio_file="s.wav", duration_s=20.0,
            diarized_at="2026-05-16T00:00:00Z",
            config={}, segments=segments,
            enrolled_ids={"alice"},
        )
        with open(out) as f:
            data = json.load(f)
        assert data["enrolled_users_matched"] == [{"id": "alice", "name": "Alice"}]
        assert data["unknown_speakers_observed"] == [
            {"id": "SPEAKER_00", "segment_count": 2, "talk_seconds": 8.0},
            {"id": "SPEAKER_01", "segment_count": 1, "talk_seconds": 5.0},
        ]

    def test_unknown_speakers_observed_empty_list_when_no_pyannote_ids(self, temp_dir):
        """No recurring unknowns → emit empty list, not omit the field."""
        segments = [
            DiarizationSegment(0.0, 5.0, "alice", "Alice"),
            DiarizationSegment(5.0, 7.0, "unknown", "unknown"),
        ]
        out = os.path.join(temp_dir, "out.json")
        write_json(
            out, audio_file="s.wav", duration_s=7.0,
            diarized_at="2026-05-16T00:00:00Z",
            config={}, segments=segments,
            enrolled_ids={"alice"},
        )
        with open(out) as f:
            data = json.load(f)
        assert data["unknown_speakers_observed"] == []
```

- [ ] **Step 2: Run, confirm failures**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/diarization/test_output.py::TestWriteJson::test_emits_unknown_speakers_observed tests/diarization/test_output.py::TestWriteJson::test_unknown_speakers_observed_empty_list_when_no_pyannote_ids -v
```

Expected: 2 failures with `TypeError: write_json() got an unexpected keyword argument 'enrolled_ids'`.

- [ ] **Step 3: Modify `_enrolled_users_in_first_appearance_order` to accept `enrolled_ids`**

Open `target-vad/modes/diarization/output.py`. Replace the existing helper (lines 28-38):

```python
def _enrolled_users_in_first_appearance_order(
    segments: List[DiarizationSegment],
    enrolled_ids: set,
) -> List[Dict[str, str]]:
    """Return list of {id, name} objects deduped by id, in first-appearance order.

    Only emits speakers whose id is in `enrolled_ids` — filters out both the
    literal 'unknown' sentinel and pyannote-generated ids like 'SPEAKER_00'
    that don't correspond to any enrolled voiceprint.
    """
    seen = set()
    result = []
    for s in segments:
        if s.speaker_id not in enrolled_ids:
            continue
        if s.speaker_id not in seen:
            seen.add(s.speaker_id)
            result.append({"id": s.speaker_id, "name": s.speaker})
    return result
```

- [ ] **Step 4: Add the new helper for recurring unknowns**

Add immediately after `_enrolled_users_in_first_appearance_order`:

```python
def _unknown_speakers_observed_in_first_appearance_order(
    segments: List[DiarizationSegment],
    enrolled_ids: set,
) -> List[Dict[str, Any]]:
    """Return list of {id, segment_count, talk_seconds} for recurring-unknown speakers.

    A 'recurring unknown' is a speaker_id that is neither the literal 'unknown'
    sentinel nor in the enrolled_ids set — i.e., a pyannote-generated id that
    was preserved by the threshold gate in ClusterIdentifier. Ordered by first
    appearance (earliest segment.start).
    """
    counts: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for s in segments:
        sid = s.speaker_id
        if sid == "unknown" or sid in enrolled_ids:
            continue
        if sid not in counts:
            counts[sid] = {"id": sid, "segment_count": 0, "talk_seconds": 0.0}
            order.append(sid)
        counts[sid]["segment_count"] += 1
        counts[sid]["talk_seconds"] += s.end - s.start
    # Round talk_seconds to 2 decimals for stable JSON output.
    for sid in counts:
        counts[sid]["talk_seconds"] = round(counts[sid]["talk_seconds"], 2)
    return [counts[sid] for sid in order]
```

- [ ] **Step 5: Update `write_json` to accept `enrolled_ids` and emit the new field**

Modify `write_json` signature to add `enrolled_ids` after `segments`. Update the call to the renamed helper and emit the new field. The body becomes:

```python
def write_json(
    path: str,
    *,
    audio_file: str,
    duration_s: float,
    diarized_at: str,
    config: Dict[str, Any],
    segments: List[DiarizationSegment],
    enrolled_ids: set,
    passes_run: Optional[List[str]] = None,
) -> None:
    """Write the diarization timeline as JSON. Schema per spec.

    `enrolled_ids` is the set of ids that correspond to enrolled voiceprints
    (persistent or session-scoped). Speakers with ids outside this set and not
    equal to 'unknown' are treated as recurring unknowns (preserved pyannote
    cluster ids per the threshold gate in ClusterIdentifier).

    If `passes_run` is provided, it is emitted as a top-level field.
    """
    payload = {
        "audio_file": audio_file,
        "duration_s": duration_s,
        "diarized_at": diarized_at,
        "config": config,
        "enrolled_users_matched": _enrolled_users_in_first_appearance_order(segments, enrolled_ids),
        "unknown_speakers_observed": _unknown_speakers_observed_in_first_appearance_order(segments, enrolled_ids),
        "segments": [
            {"start": s.start, "end": s.end, "speaker_id": s.speaker_id, "speaker": s.speaker}
            for s in segments
        ],
    }
    if passes_run is not None:
        payload["passes_run"] = list(passes_run)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 6: Update existing tests in `test_output.py` that call `write_json` without `enrolled_ids`**

The pre-existing tests in `test_output.py` (e.g. `test_roundtrip_basic`, `test_enrolled_users_matched_objects`) call `write_json` without the new `enrolled_ids` kwarg, so they'll fail. Find each call site and add an appropriate `enrolled_ids={...}` argument.

Run `grep -n "write_json(" target-vad/tests/diarization/test_output.py` to locate every call. For each, add the `enrolled_ids` keyword argument inferring the right value from the test's segments:

- `test_roundtrip_basic`: segments have `siddharth`/`unknown`/`siddharth` → `enrolled_ids={"siddharth"}`
- `test_enrolled_users_matched_objects`: segments have `alice_smith`/`bob`/`alice_smith`/`unknown`/`alice_jones` → `enrolled_ids={"alice_smith", "bob", "alice_jones"}`
- Any other write_json call: include all non-"unknown" ids in the segments.

The simplest mechanical fix is to set `enrolled_ids={s.speaker_id for s in segments if s.speaker_id != "unknown"}` at each call site for now. This preserves the prior behaviour where every non-"unknown" id appeared in `enrolled_users_matched`.

- [ ] **Step 7: Run, confirm all output tests pass**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/diarization/test_output.py -v
```

Expected: all `TestWriteJson` tests (including 2 new + existing) pass. STOP if any unrelated test fails — investigate.

- [ ] **Step 8: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/diarization/output.py target-vad/tests/diarization/test_output.py
git -C c:/repos/TVAD commit -m "feat(diarization): emit unknown_speakers_observed sibling field"
```

---

## Task 4: Wire new knobs through `diarize.py`

**Files:**
- Modify: `target-vad/diarize.py`

- [ ] **Step 1: Confirm current behavior (baseline)**

The full suite still must pass at this point:

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 214 passed (210 baseline + 4 new identifier tests in Task 2). If different, investigate before continuing.

- [ ] **Step 2: Update `diarize.py` — pass new knobs into `ClusterIdentifier`**

Open `target-vad/diarize.py`. Find the `ClusterIdentifier` construction (around line 182-187) and add two kwargs from `diar_cfg`:

```python
        identifier = ClusterIdentifier(
            embedder=embedder,
            enrollment_store=view,
            threshold=diar_cfg["identification_threshold"],
            max_sample_seconds=diar_cfg["centroid_max_sample_seconds"],
            unknown_min_segments=diar_cfg.get("unknown_min_segments", 2),
            unknown_min_seconds=diar_cfg.get("unknown_min_seconds", 10.0),
        )
```

Use `.get()` with defaults so older `config.yaml` files (or test fixtures with a minimal `diar_cfg`) keep working.

- [ ] **Step 3: Update `diarize.py` — pass `enrolled_ids` to `write_json`**

Compute the enrolled-id set from the view and pass it. Find the `write_json(...)` call (around line 199-212). Above that call, add:

```python
    enrolled_ids = set(view.get_all().keys())
```

(`SessionEnrollmentView.get_all()` returns the merged persistent + session-scoped voiceprints dict. The keys are the enrolled ids.)

Inside the `write_json(...)` call, add the new kwarg and emit the new config keys in the embedded `config` block:

```python
    write_json(
        out_path,
        audio_file=os.path.abspath(args.input),
        duration_s=duration_s,
        diarized_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config={
            "pyannote_pipeline": diar_cfg["pyannote_pipeline"],
            "identification_threshold": diar_cfg["identification_threshold"],
            "intro_override_warn_threshold": diar_cfg["intro_override_warn_threshold"],
            "unknown_min_segments": diar_cfg.get("unknown_min_segments", 2),
            "unknown_min_seconds": diar_cfg.get("unknown_min_seconds", 10.0),
            "introductions_manifest": args.introductions,
        },
        segments=segments,
        enrolled_ids=enrolled_ids,
        passes_run=["diarization"],
    )
```

- [ ] **Step 4: Update `flatten_clusters` in `diarize.py` to handle pyannote-id segments cleanly**

`flatten_clusters` (around line 53-77) currently special-cases `matched_id == "unknown"` and falls back via try/except for missing ids. The current fallback at lines 69-71 already handles pyannote ids correctly (`display = matched_id`), so no code change is strictly needed.

However: tidy up the success-print loop (around line 189-194) so pyannote ids print without the misleading `[unknown]` line. Replace those lines with:

```python
        for cid, matched_id in labels.items():
            if matched_id == "unknown":
                console.print(f"  [dim]{cid}[/] -> [bold]unknown[/]")
            elif matched_id == cid:
                # Threshold passed but no enrollment match — pyannote id preserved.
                console.print(f"  [dim]{cid}[/] -> [bold]{matched_id}[/] [dim](recurring unknown)[/]")
            else:
                display = view.get_name(matched_id)
                console.print(f"  [dim]{cid}[/] -> [bold]{matched_id}[/] ([dim]{display}[/])")
```

- [ ] **Step 5: Run full suite**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 216 passed (210 baseline + 4 identifier + 2 output). If `diarize.py` has any unit tests that exercise this flow, they may need an `enrolled_ids=set()` kwarg fixup; investigate any failures.

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/diarize.py
git -C c:/repos/TVAD commit -m "feat(diarization): wire unknown-threshold knobs through diarize.py"
```

---

## Task 5: Metrics orchestrator — 3-way speaker breakdown

**Files:**
- Modify: `target-vad/metrics.py`
- Modify: `target-vad/tests/metrics/test_orchestration.py`

- [ ] **Step 1: Add failing test**

Append to `target-vad/tests/metrics/test_orchestration.py` inside the `TestMetricsOrchestration` class:

```python
    def test_session_block_breaks_down_enrolled_vs_recurring_unknown(self, tmp_workspace):
        """JSON with enrolled + recurring-unknown + catchall segments → session block reports all three."""
        data = _read(tmp_workspace["json"])
        # Mutate the fixture: replace bob with a recurring-unknown pyannote id; add a catchall.
        data["enrolled_users_matched"] = [{"id": "alice", "name": "Alice"}]
        data["unknown_speakers_observed"] = [
            {"id": "SPEAKER_00", "segment_count": 1, "talk_seconds": 10.0}
        ]
        data["segments"][1]["speaker_id"] = "SPEAKER_00"
        data["segments"][1]["speaker"] = "SPEAKER_00"
        # Append a brief catchall segment.
        data["duration_s"] = 33.0
        data["segments"].append({
            "start": 30.0, "end": 33.0,
            "speaker_id": "unknown", "speaker": "unknown",
            "text": "uh", "words": [{"start": 30.0, "end": 30.1, "word": "uh", "probability": 0.9}],
            "sentiment": _sent("neutral", "neutral"),
        })
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)

        rc = metrics.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 0
        result = _read(tmp_workspace["json"])
        session = result["contribution_metrics"]["session"]
        assert session["identified_speakers"] == 1
        assert session["recurring_unknown_speakers"] == 1
        assert session["unknown_segments"] == 1
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/metrics/test_orchestration.py::TestMetricsOrchestration::test_session_block_breaks_down_enrolled_vs_recurring_unknown -v
```

Expected: `KeyError: 'recurring_unknown_speakers'` (or `assert ... == 1` failing because `identified_speakers` is still using the old `speaker_id != "unknown"` rule).

- [ ] **Step 3: Modify `_build_metrics_block` in `metrics.py`**

Open `target-vad/metrics.py`. Find `_build_metrics_block` (the function that composes the contribution_metrics block, around line 70-130).

Locate the `session_block = {...}` construction. The current `identified_speakers` and `unknown_segments` come from `participation["session"]`. Override these from the top-level JSON metadata.

After `session_block = { ... }` is built, add an override block:

```python
    # Override the 3-way speaker breakdown using authoritative top-level metadata.
    enrolled_list = data.get("enrolled_users_matched", []) or []
    recurring_list = data.get("unknown_speakers_observed", []) or []
    session_block["identified_speakers"] = len(enrolled_list)
    session_block["recurring_unknown_speakers"] = len(recurring_list)
    # unknown_segments stays as participation aggregator computed it (count of speaker_id == "unknown" segments).
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/metrics/test_orchestration.py -v
```

Expected: all 9 tests pass (8 prior + 1 new). If any prior test fails, the override is interfering with an assumed default — investigate.

Notably: the prior `test_happy_path_writes_json_and_markdown` test sets up its fixture with `enrolled_users_matched` already populated, so `identified_speakers` should still be 2 (matching the original assertion). Verify.

- [ ] **Step 5: Commit**

```bash
git -C c:/repos/TVAD add target-vad/metrics.py target-vad/tests/metrics/test_orchestration.py
git -C c:/repos/TVAD commit -m "feat(metrics): 3-way speaker breakdown (enrolled / recurring unknown / catchall)"
```

---

## Task 6: Renderer header line + golden update

**Files:**
- Modify: `target-vad/modes/metrics/renderer.py`
- Modify: `target-vad/tests/metrics/fixtures/golden_report.md`
- Modify: `target-vad/tests/metrics/test_renderer.py`

- [ ] **Step 1: Update the renderer header logic**

Open `target-vad/modes/metrics/renderer.py`. Find the header line (the second `**Speakers:**` line in `render_markdown`, after the `**Duration:**` line). Currently:

```python
    lines.append(
        f"**Speakers:** {sess['unique_speakers']} "
        f"({sess['identified_speakers']} identified, {sess['unknown_segments']} unknown) · "
        f"**Words:** {sess['total_words']} · **Segments:** {sess['total_segments']}"
    )
```

Replace with:

```python
    catchall = 1 if sess.get("unknown_segments", 0) > 0 else 0
    recurring_unknown = sess.get("recurring_unknown_speakers", 0)
    enrolled = sess.get("identified_speakers", 0)
    speakers_total = enrolled + recurring_unknown + catchall
    lines.append(
        f"**Speakers:** {speakers_total} "
        f"({enrolled} enrolled, {recurring_unknown} recurring unknown, {catchall} catchall) · "
        f"**Words:** {sess['total_words']} · **Segments:** {sess['total_segments']}"
    )
```

The old `unique_speakers` field is no longer used in the header — that's intentional, the 3-way breakdown is more informative.

- [ ] **Step 2: Update the golden fixture header line**

Open `target-vad/tests/metrics/fixtures/golden_report.md`. Find line 4 (the `**Speakers:**` line) and replace:

Old:
```
**Speakers:** 2 (2 identified, 0 unknown) · **Words:** 312 · **Segments:** 9
```

New:
```
**Speakers:** 2 (2 enrolled, 0 recurring unknown, 0 catchall) · **Words:** 312 · **Segments:** 9
```

- [ ] **Step 3: Update the renderer test fixture `_full_metrics_block` to include the new field**

Open `target-vad/tests/metrics/test_renderer.py`. Find `_full_metrics_block()` and inside the `"session"` dict, add a new key (any consistent location, but near the other speaker-count fields):

```python
            "recurring_unknown_speakers": 0,
```

- [ ] **Step 4: Add a new test for the 3-way header**

Append to `class TestRenderer` in `target-vad/tests/metrics/test_renderer.py`:

```python
    def test_header_3way_breakdown(self):
        """Header reports enrolled / recurring unknown / catchall counts."""
        metrics_block = _full_metrics_block()
        metrics_block["session"]["identified_speakers"] = 2
        metrics_block["session"]["recurring_unknown_speakers"] = 1
        metrics_block["session"]["unknown_segments"] = 3
        out = renderer.render_markdown(metrics_block, _session_meta())
        assert "(2 enrolled, 1 recurring unknown, 1 catchall)" in out
        assert "**Speakers:** 4 " in out  # 2 + 1 + 1
```

- [ ] **Step 5: Run renderer tests**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/metrics/test_renderer.py -v
```

Expected: 6 passed (5 prior + 1 new). The `test_full_report_matches_golden` test should pass because the renderer change + golden change are aligned. If it still fails (byte-mismatch), inspect the renderer output vs the golden — likely a whitespace divergence in the new line.

- [ ] **Step 6: Run the full suite**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 218 passed (210 baseline + 4 identifier + 2 output + 1 orchestration + 1 renderer). STOP if not.

- [ ] **Step 7: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/metrics/renderer.py target-vad/tests/metrics/fixtures/golden_report.md target-vad/tests/metrics/test_renderer.py
git -C c:/repos/TVAD commit -m "feat(metrics): 3-way speaker breakdown in Markdown header"
```

---

## Task 7: Manual smoke — recurring unknowns in real audio

**Files:** none changed.

This task verifies the full upgrade end-to-end against a real recording where pyannote will find unenrolled speakers. The existing `Voice 001 short.wav` is short and unenrolled by default (no persistent enrollment was set up). Without `--introductions`, S1 will produce two pyannote clusters, both unmatched, both substantive → both should keep their pyannote ids.

- [ ] **Step 1: Back up the current Voice 001 outputs**

The repo currently has `Voice 001 short.wav.diarization.json` (with session-scoped intro enrollment for `session_speaker_a` and `session_speaker_b`). Set this aside:

```bash
cd c:/repos/TVAD && cp "Voice 001 short.wav.diarization.json" "/tmp/voice001-with-intros.diarization.json"
cp "Voice 001 short.wav.diarization.metrics.md" "/tmp/voice001-with-intros.metrics.md"
```

(Use a tempfile path appropriate for your shell — on Windows Git Bash, `/tmp/` may not exist; use `c:/tmp/` instead.)

- [ ] **Step 2: Re-run S1 WITHOUT `--introductions`**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 diarize.py "../Voice 001 short.wav"
```

Expected:
- Exit 0
- Console prints `N cluster(s) found. Identifying...`
- For each cluster, prints something like `[dim]SPEAKER_00[/] -> [bold]SPEAKER_00[/] [dim](recurring unknown)[/]`
- Writes `../Voice 001 short.wav.diarization.json`

If pyannote finds clusters that are below threshold (<2 segments or <10s), they'll print as `-> unknown` instead — that's fine.

- [ ] **Step 3: Inspect the JSON**

```bash
cd c:/repos/TVAD && py -3.14 -c "import json; d=json.load(open('Voice 001 short.wav.diarization.json',encoding='utf-8')); print('enrolled:', d['enrolled_users_matched']); print('unknowns:', d['unknown_speakers_observed']); print('config:', d['config']); print('speakers:', sorted({s['speaker_id'] for s in d['segments']}))"
```

Expected:
- `enrolled: []` (no persistent enrollment; no `--introductions`)
- `unknowns: [{"id": "SPEAKER_00", ...}, {"id": "SPEAKER_01", ...}]` — both substantive
- `config:` includes the two new keys
- `speakers:` contains pyannote ids

- [ ] **Step 4: Run the full 2A → 2B → 3 chain on the new JSON**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 transcribe.py "../Voice 001 short.wav.diarization.json"
py -3.14 sentiment.py "../Voice 001 short.wav.diarization.json"
py -3.14 metrics.py "../Voice 001 short.wav.diarization.json"
```

Expected:
- All three exit 0
- `metrics.py` console summary reports `2 speakers, ..., 0+ highlights`
- A new `Voice 001 short.wav.diarization.metrics.md` is written

- [ ] **Step 5: Eyeball the Markdown header**

Open `Voice 001 short.wav.diarization.metrics.md`. The header line should read:

```
**Speakers:** 2 (0 enrolled, 2 recurring unknown, 0 catchall) · **Words:** ... · **Segments:** ...
```

The Participation / Sentiment / Turn-taking tables should show rows for `SPEAKER_00` and `SPEAKER_01` instead of `Speaker A` / `Speaker B`. Sanity-check the totals against the prior smoke output.

- [ ] **Step 6: Restore the original Voice 001 fixture state**

The repo's committed `Voice 001 short.wav.diarization.json` is the "with intros" version used as a regression-test snapshot for prior phases. Restore it:

```bash
cp "/tmp/voice001-with-intros.diarization.json" "Voice 001 short.wav.diarization.json"
cp "/tmp/voice001-with-intros.metrics.md" "Voice 001 short.wav.diarization.metrics.md"
```

This preserves the original session-enrolled fixture without committing the "no intros" smoke artifact (which would be confusing in the repo because the project's primary smoke flow uses intros).

- [ ] **Step 7: Re-run the chain on the restored fixture to refresh the metrics block**

The restored JSON is from before this upgrade (no `unknown_speakers_observed` field). Re-run metrics.py to refresh it under the new schema:

```bash
cd c:/repos/TVAD/target-vad && py -3.14 metrics.py "../Voice 001 short.wav.diarization.json"
```

Expected: exit 0; the JSON gains `unknown_speakers_observed: []` (because the fixture's segments are all enrolled session speakers) and `recurring_unknown_speakers: 0` in the `session` block. The Markdown header now reads `(2 enrolled, 0 recurring unknown, 0 catchall)`.

Note: the JSON's `enrolled_users_matched` was written by the OLD diarize.py and contains both `session_speaker_a` and `session_speaker_b`, which is correct — the upgrade is backward-compatible in the read direction.

- [ ] **Step 8: Update auto-memory**

Append a paragraph to `C:\Users\AI PC\.claude\projects\c--repos-TVAD\memory\project_tvad.md` documenting the recurring-unknown upgrade:

- `identifier.py` now preserves pyannote cluster ids (e.g. `SPEAKER_00`) for substantive unmatched clusters (default threshold: ≥2 segments AND ≥10s total)
- Below-threshold clusters keep the existing `"unknown"` catchall
- New top-level `unknown_speakers_observed` JSON field — sibling to `enrolled_users_matched`
- Phase 3 metrics gains `recurring_unknown_speakers` in the `session` block; Markdown header now 3-way breakdown
- Validated against `Voice 001 short.wav` (run without `--introductions`) — both pyannote clusters preserved with their ids
- Test count: 210 → ~218

- [ ] **Step 9: Commit the snapshot refresh**

```bash
git -C c:/repos/TVAD status
git -C c:/repos/TVAD add "Voice 001 short.wav.diarization.json" "Voice 001 short.wav.diarization.metrics.md"
git -C c:/repos/TVAD commit -m "chore(diarization): refresh Voice 001 fixture under new schema"
```

---

## Self-review checklist (after all tasks)

- [ ] Spec sections all implemented:
  - Identifier threshold logic → Task 2
  - JSON schema (new `unknown_speakers_observed` field) → Task 3
  - Config knobs → Tasks 1, 4
  - Metrics ripple (identified_speakers redefinition, recurring_unknown_speakers field) → Task 5
  - Markdown header 3-way breakdown → Task 6
  - Backward-compat read path → exercised in Task 7 Step 7
- [ ] Test count baseline (210) increased by approximately 7 (218 final)
- [ ] No new dependencies
- [ ] No commits to `target-vad/transcribe.py` or `target-vad/sentiment.py` — they are unaffected
- [ ] All commits scoped — one logical change per commit
