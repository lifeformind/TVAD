# In-Session Enrollment for Classroom Diarization — Design

**Date:** 2026-05-15
**Status:** Draft (awaiting user review)
**Depends on:** [`2026-05-14-classroom-diarization-design.md`](./2026-05-14-classroom-diarization-design.md) (S1, shipped 2026-05-15)

## Purpose

Extend the classroom diarization pipeline (S1) so a single recording can carry its own enrollment phase. The user provides a manifest mapping intro time-ranges to (id, name) pairs; `diarize.py` extracts voiceprints from those ranges at runtime, merges them with persistent enrollments, and runs the standard cluster-identification flow. Output now carries both a stable `speaker_id` (for downstream tooling) and a `speaker` display name (for humans).

This unblocks the "first minute is intros" workflow where some speakers in a session aren't pre-enrolled, while preserving the "always-on" pre-enrollment path for stable users (teacher, recurring TAs).

## Architecture

Two phases, both inside this spec:

- **Phase 1 — persistent enrollment id/name refactor.** Migrate `EnrollmentStore`, `SpeakerVerifier`, `enroll.py`, `main.py`, and the legacy `pipeline/target_vad_pipeline.py` from name-as-key to (id, name) where `id` is the storage key and `name` is the display label. Pre-existing voiceprints not listed in `users.json` are invisible to the new store and must be re-enrolled (legacy data is discardable per user direction 2026-05-15).
- **Phase 2 — in-session enrollment + diarize.py wiring.** Add a `--introductions` flag to `diarize.py`, a new `modes/diarization/intro_enrollment.py` module, and update `ClusterIdentifier` + the output schema to carry `speaker_id` alongside `speaker`.

The kiosk (`modes/kiosk/*`) does **not** consume `EnrollmentStore` or `SpeakerVerifier` in its runtime path (confirmed by grep of `modes/kiosk/`); no kiosk changes are required.

## Phase 1: persistent enrollment id/name refactor

### Storage layout

**Before:**
```
voiceprints/
  siddharth.npy
```

**After:**
```
voiceprints/
  users.json          # {"siddharth": "Siddharth Jain", "alice_smith": "Alice"}
  siddharth.npy
  alice_smith.npy
```

`users.json` is a single JSON object mapping `id` → display `name`. Voiceprint files are still `<id>.npy` (the filename stem IS the id).

### Existing voiceprints are discardable

There is no auto-migration. Pre-existing `<id>.npy` files without a `users.json` entry are invisible to the new `EnrollmentStore` — only ids registered in `users.json` are considered enrolled. This decision is deliberate (user confirmed 2026-05-15: "no active usage yet only tests, treat all legacy data as discardable") and keeps `EnrollmentStore` free of filesystem-scanning back-compat logic.

`users.json` is the canonical registry. The current single `siddharth.npy` in the user's `voiceprints/` directory will be invisible after the upgrade and should either be deleted or re-enrolled via `enroll.py siddharth`.

### `EnrollmentStore` API changes

| Method | Before | After |
|---|---|---|
| `enroll(id, embedding)` | (existed) | unchanged — `id` is positional name now |
| `finalize_enrollment(id)` | (existed) | now also writes a default `users.json` entry `id → id` if one doesn't exist |
| `register(id, name)` | new | upserts a `users.json` entry without touching `<id>.npy` |
| `get(id) → np.ndarray` | (existed) | unchanged |
| `get_name(id) → str` | new | returns display name from `users.json`; raises `KeyError` if id not registered |
| `get_all() → Dict[str, np.ndarray]` | (existed) | returns id → embedding **only for ids registered in `users.json`**; orphan `.npy` files are ignored |
| `list_users() → List[str]` | (existed) | returns sorted list of registered ids (from `users.json`) |
| `delete(id)` | (existed) | removes both the `<id>.npy` and the `users.json` entry |

`enroll.py` always calls `store.register(id, name)` after `finalize_enrollment` so display names land in `users.json` immediately. Single-arg `enroll.py siddharth` calls `register("siddharth", "siddharth")` — id and name match. The two-arg form `enroll.py alice_smith --name "Alice"` calls `register("alice_smith", "Alice")`.

### `SpeakerVerifier` and `VerificationResult` changes

`VerificationResult` field rename:

```python
@dataclass
class VerificationResult:
    is_registered: bool
    matched_id: Optional[str]       # renamed from matched_user
    matched_name: Optional[str]     # NEW: store.get_name(matched_id) when matched, else None
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)  # id → score
```

**No backward-compat shim.** All callers of `matched_user` get migrated in this phase. Affected files:

- `core/speaker/verifier.py` (definition)
- `enroll.py` line 187 (`f"... user={result.matched_user}"` → `"... id={result.matched_id} name={result.matched_name}"`)
- `main.py` line 27 (same pattern)
- `pipeline/target_vad_pipeline.py` lines 71-79 (`matched_user` → `matched_id`)
- `tests/test_verifier.py` lines 103, 142 (assertions)
- `tests/test_pipeline.py` lines 56, 91-96 (assertions)

### `enroll.py` CLI

**Before:** `py -3.14 enroll.py <username>`. The username is both id and display name.

**After:** `py -3.14 enroll.py <id> [--name "Display Name"]`. If `--name` is omitted, defaults to `<id>` — preserving existing single-arg invocations as a no-op for display purposes. The enrollment flow is otherwise unchanged (5 utterances, self-similarity gate, etc.).

After the final `finalize_enrollment` call, `enroll.py` calls `store.register(id, name or id)` to record the display name.

## Phase 2: in-session enrollment

### Manifest schema

User-authored JSON file (path passed via `--introductions`):

```json
[
  {"id": "siddharth", "name": "Siddharth Jain", "start": 0.0, "end": 28.5},
  {"id": "alice_smith", "name": "Alice", "start": 28.5, "end": 47.0},
  {"id": "alice_jones", "name": "Alice", "start": 47.0, "end": 65.0}
]
```

**Rules:**
- Top-level value is a JSON array of objects.
- Each object has exactly the four keys: `id` (string), `name` (string), `start` (number, seconds), `end` (number, seconds).
- `id` must be unique within the manifest (hard error otherwise).
- `name` is free-form, may collide across entries.
- `end > start`, both non-negative.

### New module: `modes/diarization/intro_enrollment.py`

```python
@dataclass
class IntroVoiceprint:
    id: str
    name: str
    embedding: np.ndarray  # 192-dim L2-normalized

def load_manifest(path: str) -> List[ManifestEntry]: ...

def enroll_from_intros(
    audio: np.ndarray,
    sample_rate: int,
    manifest: List[ManifestEntry],
    embedder,  # injected EmbeddingExtractor
) -> List[IntroVoiceprint]: ...
```

`enroll_from_intros` slices the audio per manifest entry (`audio[int(start*sr):int(end*sr)]`), runs the existing `EmbeddingExtractor.extract` once per slice, and returns the list of `IntroVoiceprint`s. No disk writes. ECAPA's existing `MIN_DURATION_SAMPLES` reflect-padding handles short intros.

### Session-scoped overlay store

A small adapter, `SessionEnrollmentView`, presents the merged view to `ClusterIdentifier`:

```python
class SessionEnrollmentView:
    def __init__(self, persistent: EnrollmentStore, intros: List[IntroVoiceprint]):
        ...
    def get_all(self) -> Dict[str, np.ndarray]: ...  # id → embedding, intros shadow persistent
    def get_name(self, id: str) -> str: ...           # intro names shadow persistent
```

`ClusterIdentifier` consumes this view via its existing `enrollment_store` parameter (duck-typed on `.get_all()`). The persistent store on disk is never modified.

### `ClusterIdentifier` changes

`_best_label` returns the matched `id` instead of the matched display name. `label_clusters` return type changes from `Dict[str, str]` (cluster_id → label) to `Dict[str, str]` still — but the value is now an `id` (or `"unknown"`).

The display name lookup happens later in `diarize.py::flatten_clusters` via `view.get_name(matched_id)`.

### `diarize.py` changes

New flag: `--introductions <path>`.

Orchestration order in `main()` after audio load:

1. If `--introductions` passed: load the manifest, validate, run `enroll_from_intros`. Apply conflict-resolution rules below. Build `SessionEnrollmentView`.
2. Else: `SessionEnrollmentView(persistent_store, [])` — empty intros, behaves identical to today's flow.
3. Construct `ClusterIdentifier` with the view as `enrollment_store`.
4. Run `Diarizer.diarize` and `ClusterIdentifier.label_clusters` as today.
5. In `flatten_clusters`, look up display name for each label id.
6. Write JSON/RTTM with new schema (below).

### Conflict resolution

| Case | Behavior |
|---|---|
| Manifest `id` matches persistent `id`, cosine ≥ 0.30 | Intro overrides persistent for this run. Log `[INTRO OVERRIDE] {id}` at info level. |
| Manifest `id` matches persistent `id`, cosine < 0.30 | Intro overrides anyway; **warn** `[INTRO SUSPICIOUS] {id} cosine={cos:.2f} — same id intended?`. The user's id assertion wins. |
| Same `id` appears twice in manifest | **Hard error**, exit 2. Message: `Duplicate id '{id}' in manifest at entries [{i}, {j}]`. |
| Same `name`, different `id` | Both kept; both compete in matching. Output preserves both via distinct `speaker_id`. The same display string appears in `speaker` — consumers disambiguate via `speaker_id`. |
| Manifest entry's intro audio < 800 ms | Reflect-padded per existing `EmbeddingExtractor._pad_if_short`. No special-case. |
| Manifest entry's `end` exceeds audio duration | Clamp `end` to `len(audio)/sr`, log a warning. |

The `0.30` threshold for the "suspicious" warning is intentionally well below `identification_threshold` (0.55) so the warning fires only on truly mismatched voiceprints, not on noisy-but-real same-person matches. Configurable via `diarization.intro_override_warn_threshold` (default 0.30).

## Output schema (Phase 2)

### JSON segments

```json
{"start": 0.42, "end": 3.81, "speaker_id": "alice_smith", "speaker": "Alice"}
```

- `speaker_id`: stable identifier. For unknown clusters: `"unknown"`.
- `speaker`: display name. For unknown clusters: `"unknown"` (same string).

### Top-level `enrolled_users_matched`

Changes from `["siddharth"]` (list of strings) to a list of objects:

```json
"enrolled_users_matched": [
  {"id": "siddharth", "name": "Siddharth Jain"},
  {"id": "alice_smith", "name": "Alice"}
]
```

Deduped by `id`. Ordered by first-appearance time in `segments`.

### Top-level `config`

Add the two new knobs the run actually used:

```json
"config": {
  "pyannote_pipeline": "pyannote/speaker-diarization-3.1",
  "identification_threshold": 0.55,
  "intro_override_warn_threshold": 0.30,
  "introductions_manifest": "intros.json"   // or null if --introductions not passed
}
```

### RTTM (Phase 2)

The speaker column (field 8 of `SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>`) uses **`speaker_id`** (machine-consumed). Display names do not appear in RTTM. This matches NIST convention — RTTM consumers expect short stable tokens, not free-form display strings.

## Configuration

New keys under the existing `diarization:` block:

```yaml
diarization:
  identification_threshold: 0.55
  pyannote_pipeline: "pyannote/speaker-diarization-3.1"
  hf_token_env_var: "HF_TOKEN"
  centroid_max_sample_seconds: 30
  intro_override_warn_threshold: 0.30   # NEW
```

## Error handling additions

| Failure | Exit code | Behavior |
|---|---|---|
| `--introductions` file missing or unreadable | 2 | Error message + abort |
| Manifest JSON malformed (decode error) | 2 | Error includes parse offset |
| Manifest entry missing required key | 2 | Error includes entry index and missing key |
| `start >= end` or negative `start` | 2 | Error includes entry index |
| Duplicate `id` in manifest | 2 | Error includes both entry indices |
| Manifest entry exceeds audio duration | 0 | Clamp + warn, continue |
| Manifest entry's audio is silent / zero variance | 0 | Embedder runs anyway (its existing fallback path); intro voiceprint is included; downstream cluster matching naturally produces low scores. No special-case. |

## Testing approach

New test files:

- `tests/diarization/test_intro_enrollment.py`:
  - `load_manifest` round-trip (valid JSON → list of entries)
  - Malformed manifest errors (missing keys, bad types, negative start, end ≤ start)
  - Duplicate-id detection
  - `enroll_from_intros` with a mock embedder — verifies correct audio slicing per entry and that the embedder is called once per entry
  - End-of-audio clamping behavior

- `tests/diarization/test_session_view.py`:
  - `SessionEnrollmentView` shadowing: intro id `siddharth` masks persistent id `siddharth` in `get_all()` and `get_name()`
  - Empty intros → behaves identically to passing the persistent store directly

- `tests/diarization/test_output.py` (extend existing):
  - JSON includes `speaker_id` and `speaker` fields
  - `enrolled_users_matched` is the new object shape, deduped by id, in first-appearance order
  - RTTM speaker column uses `speaker_id`, not name

- `tests/diarization/test_identifier.py` (update existing):
  - `label_clusters` returns id values (not display names)
  - All existing assertions updated to check ids

- `tests/test_verifier.py` (update existing):
  - `VerificationResult.matched_id` (not `matched_user`)
  - `register` / `get_name` round-trip
  - `get_name` on unregistered id raises `KeyError`
  - Orphan `<id>.npy` not in `users.json` is ignored by `get_all()` / `list_users()`
  - `enroll` + `finalize_enrollment` auto-creates a `users.json` entry with `name == id` (so existing single-arg `enroll.py` tests pass unchanged)

- `tests/test_pipeline.py` (update existing):
  - Same migration as `test_verifier.py`

Expected test count after Phase 1 + Phase 2: ~115 (current 90 + ~25 new and migrated).

End-to-end smoke test (manual, not CI): a recording with explicit intros + a known-mismatched persistent voiceprint to confirm the override warning fires.

## Migration path for the user

User confirmed 2026-05-15 that no real data is in flight — only the test `siddharth.npy` exists. Migration is therefore trivial:

1. Update code (pull Phase 1 + Phase 2 commits).
2. Delete the existing `voiceprints/siddharth.npy` (it'll be invisible to the new store anyway).
3. Re-enroll: `py -3.14 enroll.py siddharth` (or `... siddharth --name "Siddharth Jain"` to set a display name in one shot).
4. To use in-session enrollment, author an `intros.json` per recording and pass `--introductions intros.json`.

Old single-arg `enroll.py siddharth` invocations still work — id and name both become `siddharth` (existing behavior preserved exactly).

## Out of scope

- Auto-detecting the intro phase (no ASR; user always provides the manifest).
- Updating kiosk to consume `EnrollmentStore` (it doesn't today; no change needed).
- Cross-session voiceprint averaging or re-enrollment from intros (intros are session-scoped only; the persistent voiceprint on disk is untouched even when overridden in a session).
- A dedicated migration CLI for `users.json` — orphan `.npy` files are invisible to the new store; re-enroll any users you want preserved.
- Updating Phase-2 ASR (the deferred `transcribe.py` post-processor) to consume the new schema — that work happens when ASR lands and will be in scope of that phase's spec.
- Changes to the manifest schema beyond the four documented keys (no nested groups, no roles, no times-as-strings). Keep it boring.

## Open questions

None at time of writing. All decisions resolved during brainstorming 2026-05-15.
