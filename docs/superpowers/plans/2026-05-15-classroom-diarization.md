# S1 Classroom Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline classroom diarization CLI that takes a single-channel WAV file, runs pyannote speaker-diarization-3.1, identifies each cluster against enrolled voiceprints via ECAPA centroid matching, and writes a JSON (and optional RTTM) timeline of who-spoke-when.

**Architecture:** Pure offline pipeline. WAV → resample to 16 kHz mono → pyannote pipeline emits clusters (cluster_id → list of (start, end) segments) → per cluster, gather a ≤30 s evenly-spaced sample of the cluster's audio, embed via existing ECAPA `EmbeddingExtractor`, L2-normalize the centroid, cosine-match against all enrolled voiceprints, label as the best-scoring enrolled name iff score ≥ 0.55, else `"unknown"`. Flatten clusters into segments, attach labels, sort by start time, write JSON/RTTM. Per-cluster identification (not per-segment) is the key design choice — centroids are dramatically more stable than single-segment embeddings on the C10 hardware.

**Tech Stack:** pyannote.audio 3.x (HuggingFace-gated, requires `HF_TOKEN` env var), soundfile 0.13.x for WAV I/O, existing `core.speaker.embedder.EmbeddingExtractor` (ECAPA-TDNN via SpeechBrain), existing `core.speaker.enrollment_store.EnrollmentStore`, existing `core.speaker.verifier.cosine_similarity`, existing `core.compat` shim, rich 13.x for console output, pytest for tests. Always invoke as `py -3.14` on this Windows machine — `python` resolves to 3.12 without the dep stack.

**Spec:** [`docs/superpowers/specs/2026-05-14-classroom-diarization-design.md`](../specs/2026-05-14-classroom-diarization-design.md). Read before starting.

**Working directory for all paths below:** `c:\repos\TVAD\target-vad\` (the project root inside the repo). All relative paths in this plan are relative to that directory unless noted.

---

## File Structure

Files this plan creates or modifies (relative to `target-vad/`):

| Path | Status | Responsibility |
|---|---|---|
| `requirements.txt` | modify | add `pyannote.audio>=3.1.0` |
| `config.yaml` | modify | add top-level `diarization:` block |
| `modes/diarization/__init__.py` | already exists | docstring only, no code |
| `modes/diarization/sampling.py` | create | `sample_cluster_segments()` — pick evenly-spaced segments up to N seconds total |
| `modes/diarization/identifier.py` | create | `ClusterIdentifier` — extracts cluster audio, embeds centroid, cosine-matches |
| `modes/diarization/diarizer.py` | create | `Diarizer` — thin pyannote.audio wrapper, returns `{cluster_id: [(start, end), ...]}` |
| `modes/diarization/output.py` | create | `write_json()`, `write_rttm()` — serialization |
| `diarize.py` | create | CLI entry: arg parsing, audio load, orchestration |
| `tests/diarization/__init__.py` | create | empty |
| `tests/diarization/test_sampling.py` | create | unit tests for `sample_cluster_segments()` |
| `tests/diarization/test_identifier.py` | create | unit tests with mocked embedder + fake enrolled voiceprints |
| `tests/diarization/test_output.py` | create | JSON + RTTM serialization tests |

`core/speaker/embedder.py`, `core/speaker/enrollment_store.py`, `core/speaker/verifier.py`, and `core/compat.py` are reused as-is with **no behavior changes**. If a test or task seems to require modifying any of them, stop and re-read the spec — the design assumes they stay frozen.

---

## Task 1: Add `pyannote.audio` dependency and `diarization:` config namespace

**Files:**
- Modify: `target-vad/requirements.txt`
- Modify: `target-vad/config.yaml`

- [ ] **Step 1: Add pyannote.audio to requirements**

Open `target-vad/requirements.txt` and append a line. Final file content:

```
torch>=2.1.0
torchaudio>=2.1.0
speechbrain>=1.0.0
onnxruntime>=1.17.0
numpy>=1.24.0
scipy>=1.11.0
sounddevice>=0.4.6
pyyaml>=6.0
rich>=13.0.0
openwakeword>=0.6.0
pyannote.audio>=3.1.0
soundfile>=0.12.0
```

(`soundfile` is added explicitly because we now depend on it directly for WAV loading — it was already a transitive dep via speechbrain, but pinning it makes the intent clear.)

- [ ] **Step 2: Install the new deps**

Run from `c:\repos\TVAD\target-vad\`:

```bash
py -3.14 -m pip install "pyannote.audio>=3.1.0" "soundfile>=0.12.0"
```

Expected: `Successfully installed pyannote.audio-3.x.x ...`. If pip resolves a conflicting torch version, abort the install and report — do NOT downgrade the existing torch 2.9.1 without confirming with the user, because the kiosk live tests are bound to that version.

- [ ] **Step 3: Verify pyannote imports**

Run:

```bash
py -3.14 -c "from pyannote.audio import Pipeline; print('ok')"
```

Expected stdout: `ok` (and possibly some torch/torchaudio init warnings on stderr — those are fine).

- [ ] **Step 4: Add `diarization:` block to `config.yaml`**

Open `target-vad/config.yaml`. Append after the existing `kiosk:` block (preserve the `core:` and `kiosk:` blocks exactly as they are). Final appended content:

```yaml

diarization:
  identification_threshold: 0.55
  default_output_format: "json"   # or "rttm"
  pyannote_pipeline: "pyannote/speaker-diarization-3.1"
  hf_token_env_var: "HF_TOKEN"
  centroid_max_sample_seconds: 30
```

- [ ] **Step 5: Verify config still parses**

Run:

```bash
py -3.14 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(sorted(c.keys()))"
```

Expected: `['core', 'diarization', 'kiosk']`

- [ ] **Step 6: Run the existing test suite to confirm no regression**

Run:

```bash
py -3.14 -m pytest tests/ -q
```

Expected: all existing tests pass (67 per current memory). If anything fails, stop — the dep install broke something and must be fixed before continuing.

- [ ] **Step 7: Commit**

```bash
git add target-vad/requirements.txt target-vad/config.yaml
git commit -m "feat(diarization): add pyannote.audio dep and config namespace"
```

---

## Task 2: Centroid sampling helper

**Why first:** Pure-function, no model dependency — fastest TDD cycle and the result is reused by `ClusterIdentifier`.

**Files:**
- Create: `target-vad/modes/diarization/sampling.py`
- Create: `target-vad/tests/diarization/__init__.py`
- Create: `target-vad/tests/diarization/test_sampling.py`

- [ ] **Step 1: Create empty test package**

Create `target-vad/tests/diarization/__init__.py` with content:

```python
```

(Empty file — pytest needs the directory to be importable.)

- [ ] **Step 2: Write the failing test for short clusters (no sampling needed)**

Create `target-vad/tests/diarization/test_sampling.py` with content:

```python
"""Tests for the centroid-sampling helper."""

import pytest

from modes.diarization.sampling import sample_cluster_segments


class TestSampleClusterSegments:
    def test_short_cluster_returns_all_segments(self):
        """If total duration is <= max_seconds, return everything unchanged."""
        segments = [(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)]  # total 8s
        result = sample_cluster_segments(segments, max_seconds=30.0)
        assert result == segments

    def test_exact_max_returns_all(self):
        """Total duration exactly equal to max_seconds — return all."""
        segments = [(0.0, 15.0), (20.0, 35.0)]  # total 30s
        result = sample_cluster_segments(segments, max_seconds=30.0)
        assert result == segments

    def test_long_cluster_samples_evenly(self):
        """Total > max_seconds: pick evenly-spaced segments until cap reached."""
        # 12 segments of 5s each = 60s total, cap 30s → 6 segments expected
        segments = [(i * 5.0, i * 5.0 + 5.0) for i in range(12)]
        result = sample_cluster_segments(segments, max_seconds=30.0)
        # Total duration of result must be <= max_seconds
        total = sum(end - start for start, end in result)
        assert total <= 30.0
        # Should not be empty
        assert len(result) > 0
        # Result is a subset of the input, in original order
        for seg in result:
            assert seg in segments
        # Result is sorted by start time
        assert result == sorted(result, key=lambda s: s[0])

    def test_long_cluster_evenly_spaced(self):
        """Selected indices should be roughly evenly spaced across the cluster."""
        # 10 segments of 6s each = 60s total, cap 30s → 5 segments expected
        segments = [(i * 6.0, i * 6.0 + 6.0) for i in range(10)]
        result = sample_cluster_segments(segments, max_seconds=30.0)
        selected_indices = [segments.index(s) for s in result]
        # Indices should be spread across the range (not all clustered at the start)
        assert min(selected_indices) <= 2
        assert max(selected_indices) >= 7

    def test_empty_input(self):
        """Empty segment list returns empty."""
        assert sample_cluster_segments([], max_seconds=30.0) == []

    def test_single_segment_under_cap(self):
        assert sample_cluster_segments([(0.0, 10.0)], max_seconds=30.0) == [(0.0, 10.0)]

    def test_single_segment_over_cap(self):
        """A single segment longer than cap is still returned as-is (we don't slice within a segment)."""
        # Design choice: we sample at segment granularity, not sub-segment.
        # ECAPA can ingest any length; centroid sampling is about which segments to include.
        result = sample_cluster_segments([(0.0, 60.0)], max_seconds=30.0)
        assert result == [(0.0, 60.0)]
```

- [ ] **Step 3: Run the test to verify it fails**

Run from `c:\repos\TVAD\target-vad\`:

```bash
py -3.14 -m pytest tests/diarization/test_sampling.py -v
```

Expected: `ModuleNotFoundError: No module named 'modes.diarization.sampling'` or collection error.

- [ ] **Step 4: Write the minimal implementation**

Create `target-vad/modes/diarization/sampling.py` with content:

```python
"""Centroid sampling: pick evenly-spaced segments from a cluster up to a duration cap.

Rationale: ECAPA centroid quality saturates around 30 s of audio. For very long
clusters, sampling instead of concatenating everything reduces compute without
hurting embedding quality. We sample at segment granularity (no slicing within a
segment) so the caller can directly extract the chosen segments from the waveform.
"""

from typing import List, Tuple


def sample_cluster_segments(
    segments: List[Tuple[float, float]],
    max_seconds: float,
) -> List[Tuple[float, float]]:
    """Pick a subset of segments whose total duration is <= max_seconds.

    Selection is evenly-spaced across the input order. If a single segment is
    longer than max_seconds, it is still returned alone (segment granularity).

    Args:
        segments: list of (start_s, end_s) tuples, sorted by start.
        max_seconds: target maximum total duration of returned segments.

    Returns:
        Subset of input segments, in original order. Empty input → empty output.
    """
    if not segments:
        return []

    total = sum(end - start for start, end in segments)
    if total <= max_seconds:
        return list(segments)

    # Need to subsample. Walk evenly-spaced indices and accumulate until cap reached.
    n = len(segments)
    # Estimate target count by average segment length; clamp to [1, n].
    avg_len = total / n
    target_count = max(1, min(n, int(max_seconds / avg_len)))

    # Pick `target_count` evenly-spaced indices across [0, n-1].
    if target_count == 1:
        indices = [n // 2]
    else:
        step = (n - 1) / (target_count - 1)
        indices = [round(i * step) for i in range(target_count)]
        # Dedupe while preserving order in case rounding collapsed indices
        seen = set()
        indices = [i for i in indices if not (i in seen or seen.add(i))]

    # Greedily accept selected segments while total stays <= max_seconds.
    chosen: List[Tuple[float, float]] = []
    running = 0.0
    for idx in indices:
        start, end = segments[idx]
        dur = end - start
        if running + dur <= max_seconds or not chosen:
            chosen.append((start, end))
            running += dur
        if running >= max_seconds:
            break
    return chosen
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
py -3.14 -m pytest tests/diarization/test_sampling.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add target-vad/modes/diarization/sampling.py target-vad/tests/diarization/__init__.py target-vad/tests/diarization/test_sampling.py
git commit -m "feat(diarization): add cluster centroid sampling helper"
```

---

## Task 3: JSON and RTTM output writers

**Why next:** Pure I/O, no model dependency, easy to test against fixed fixtures. Establishes the output shape we'll target in the rest of the pipeline.

**Files:**
- Create: `target-vad/modes/diarization/output.py`
- Create: `target-vad/tests/diarization/test_output.py`

- [ ] **Step 1: Write the failing tests**

Create `target-vad/tests/diarization/test_output.py` with content:

```python
"""Tests for JSON and RTTM output writers."""

import json
import os
import tempfile

import pytest

from modes.diarization.output import (
    DiarizationSegment,
    write_json,
    write_rttm,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_segments():
    return [
        DiarizationSegment(start=0.42, end=3.81, speaker="siddharth"),
        DiarizationSegment(start=3.81, end=5.10, speaker="unknown"),
        DiarizationSegment(start=5.20, end=7.45, speaker="siddharth"),
    ]


class TestWriteJson:
    def test_roundtrip_basic(self, temp_dir, sample_segments):
        out = os.path.join(temp_dir, "out.json")
        write_json(
            out,
            audio_file="session.wav",
            duration_s=2734.51,
            diarized_at="2026-05-14T10:23:01Z",
            config={"pyannote_pipeline": "pyannote/speaker-diarization-3.1", "identification_threshold": 0.55},
            segments=sample_segments,
        )
        with open(out) as f:
            data = json.load(f)

        assert data["audio_file"] == "session.wav"
        assert data["duration_s"] == pytest.approx(2734.51)
        assert data["diarized_at"] == "2026-05-14T10:23:01Z"
        assert data["config"]["identification_threshold"] == 0.55
        assert data["segments"] == [
            {"start": 0.42, "end": 3.81, "speaker": "siddharth"},
            {"start": 3.81, "end": 5.10, "speaker": "unknown"},
            {"start": 5.20, "end": 7.45, "speaker": "siddharth"},
        ]

    def test_enrolled_users_matched_dedup_in_first_appearance_order(self, temp_dir):
        segments = [
            DiarizationSegment(0.0, 1.0, "alice"),
            DiarizationSegment(1.0, 2.0, "bob"),
            DiarizationSegment(2.0, 3.0, "alice"),
            DiarizationSegment(3.0, 4.0, "unknown"),
            DiarizationSegment(4.0, 5.0, "carol"),
        ]
        out = os.path.join(temp_dir, "out.json")
        write_json(out, audio_file="a.wav", duration_s=5.0, diarized_at="t", config={}, segments=segments)
        with open(out) as f:
            data = json.load(f)
        assert data["enrolled_users_matched"] == ["alice", "bob", "carol"]

    def test_empty_segments_writes_empty_list(self, temp_dir):
        out = os.path.join(temp_dir, "out.json")
        write_json(out, audio_file="a.wav", duration_s=10.0, diarized_at="t", config={}, segments=[])
        with open(out) as f:
            data = json.load(f)
        assert data["segments"] == []
        assert data["enrolled_users_matched"] == []

    def test_unknown_only_no_enrolled_users(self, temp_dir):
        segments = [DiarizationSegment(0.0, 1.0, "unknown"), DiarizationSegment(1.0, 2.0, "unknown")]
        out = os.path.join(temp_dir, "out.json")
        write_json(out, audio_file="a.wav", duration_s=2.0, diarized_at="t", config={}, segments=segments)
        with open(out) as f:
            data = json.load(f)
        assert data["enrolled_users_matched"] == []


class TestWriteRttm:
    def test_basic_rttm_format(self, temp_dir, sample_segments):
        out = os.path.join(temp_dir, "out.rttm")
        write_rttm(out, audio_file_id="session", segments=sample_segments)
        with open(out) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 3
        # RTTM line:
        # SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        parts0 = lines[0].split()
        assert parts0[0] == "SPEAKER"
        assert parts0[1] == "session"
        assert parts0[2] == "1"
        assert float(parts0[3]) == pytest.approx(0.42)
        assert float(parts0[4]) == pytest.approx(3.81 - 0.42)  # duration
        assert parts0[5] == "<NA>"
        assert parts0[6] == "<NA>"
        assert parts0[7] == "siddharth"
        assert parts0[8] == "<NA>"
        assert parts0[9] == "<NA>"

    def test_rttm_uses_unknown_label_verbatim(self, temp_dir):
        out = os.path.join(temp_dir, "out.rttm")
        segs = [DiarizationSegment(1.0, 2.5, "unknown")]
        write_rttm(out, audio_file_id="x", segments=segs)
        with open(out) as f:
            line = f.read().strip()
        parts = line.split()
        assert parts[7] == "unknown"
        assert float(parts[4]) == pytest.approx(1.5)

    def test_empty_segments_writes_empty_file(self, temp_dir):
        out = os.path.join(temp_dir, "out.rttm")
        write_rttm(out, audio_file_id="x", segments=[])
        with open(out) as f:
            assert f.read() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
py -3.14 -m pytest tests/diarization/test_output.py -v
```

Expected: `ModuleNotFoundError: No module named 'modes.diarization.output'`.

- [ ] **Step 3: Implement the output module**

Create `target-vad/modes/diarization/output.py` with content:

```python
"""JSON and RTTM serializers for diarization output.

JSON schema is documented in docs/superpowers/specs/2026-05-14-classroom-diarization-design.md.
RTTM is the standard NIST speaker-diarization format:
    SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DiarizationSegment:
    """One timestamped, labeled speech segment."""
    start: float
    end: float
    speaker: str  # enrolled name or the literal "unknown"


def _enrolled_users_in_first_appearance_order(segments: List[DiarizationSegment]) -> List[str]:
    seen = set()
    result = []
    for s in segments:
        if s.speaker == "unknown":
            continue
        if s.speaker not in seen:
            seen.add(s.speaker)
            result.append(s.speaker)
    return result


def write_json(
    path: str,
    *,
    audio_file: str,
    duration_s: float,
    diarized_at: str,
    config: Dict[str, Any],
    segments: List[DiarizationSegment],
) -> None:
    """Write the diarization timeline as JSON. Schema per spec."""
    payload = {
        "audio_file": audio_file,
        "duration_s": duration_s,
        "diarized_at": diarized_at,
        "config": config,
        "enrolled_users_matched": _enrolled_users_in_first_appearance_order(segments),
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_rttm(
    path: str,
    *,
    audio_file_id: str,
    segments: List[DiarizationSegment],
) -> None:
    """Write segments as RTTM. Empty segments → empty file."""
    lines = []
    for s in segments:
        duration = s.end - s.start
        lines.append(
            f"SPEAKER {audio_file_id} 1 {s.start:.3f} {duration:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
        )
    content = "\n".join(lines)
    if content:
        content += "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
py -3.14 -m pytest tests/diarization/test_output.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add target-vad/modes/diarization/output.py target-vad/tests/diarization/test_output.py
git commit -m "feat(diarization): add JSON and RTTM output writers"
```

---

## Task 4: ClusterIdentifier — cluster audio → centroid → enrolled-name label

**What it does:** Given the full audio waveform plus pyannote's cluster→segments mapping, extract each cluster's audio (subsampled via `sample_cluster_segments`), embed via the injected `EmbeddingExtractor`, average to a centroid, L2-normalize, cosine-match against all enrolled voiceprints, and return a `{cluster_id: label_str}` map. `label_str` is the enrolled name if best score ≥ threshold else `"unknown"`. If embedding fails for a cluster, label as `"unknown"` and log a warning.

**Why pass full waveform + cluster timings (not pre-sliced audio):** Keeps the slicing in one place, keeps the public API small, and matches how pyannote returns its output (timing-based annotations over the original waveform).

**Files:**
- Create: `target-vad/modes/diarization/identifier.py`
- Create: `target-vad/tests/diarization/test_identifier.py`

- [ ] **Step 1: Write the failing tests**

Create `target-vad/tests/diarization/test_identifier.py` with content:

```python
"""Tests for ClusterIdentifier — mocks the embedder, uses real cosine math + sampling."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from modes.diarization.identifier import ClusterIdentifier


SR = 16000


def unit_vec(seed: int, dim: int = 192) -> np.ndarray:
    """Deterministic random unit vector for tests."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


@pytest.fixture
def fake_embedder():
    """Embedder mock — returns whatever extract() is told to return.

    Default behavior is overridden per-test by setting .extract.side_effect.
    """
    m = MagicMock()
    return m


@pytest.fixture
def fake_store():
    """Enrollment-store mock with .get_all() returning a dict of name → voiceprint."""
    m = MagicMock()
    m.get_all.return_value = {}
    return m


class TestClusterIdentifierBasics:
    def test_matches_enrolled_speaker(self, fake_embedder, fake_store):
        """Embedder returns alice's voiceprint exactly → cluster labels as 'alice'."""
        alice_vp = unit_vec(seed=1)
        bob_vp = unit_vec(seed=2)
        fake_store.get_all.return_value = {"alice": alice_vp, "bob": bob_vp}
        fake_embedder.extract.return_value = alice_vp

        identifier = ClusterIdentifier(
            embedder=fake_embedder,
            enrollment_store=fake_store,
            threshold=0.55,
            max_sample_seconds=30,
        )
        audio = make_silence(10.0)
        clusters = {"SPEAKER_00": [(0.0, 5.0)]}
        labels = identifier.label_clusters(audio, sample_rate=SR, clusters=clusters)
        assert labels == {"SPEAKER_00": "alice"}

    def test_below_threshold_is_unknown(self, fake_embedder, fake_store):
        """Embedder returns a vector orthogonal to all voiceprints → 'unknown'."""
        alice_vp = np.zeros(192, dtype=np.float32); alice_vp[0] = 1.0
        bob_vp = np.zeros(192, dtype=np.float32); bob_vp[1] = 1.0
        fake_store.get_all.return_value = {"alice": alice_vp, "bob": bob_vp}
        orthogonal = np.zeros(192, dtype=np.float32); orthogonal[2] = 1.0
        fake_embedder.extract.return_value = orthogonal

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(make_silence(5.0), SR, {"SPEAKER_00": [(0.0, 2.0)]})
        assert labels == {"SPEAKER_00": "unknown"}

    def test_at_threshold_inclusive(self, fake_embedder, fake_store):
        """A cosine exactly at threshold counts as a match (>=)."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}
        # Construct an embedding whose cosine with alice_vp == 0.55 exactly:
        # take alice_vp scaled by 0.55 plus an orthogonal component scaled by sqrt(1-0.55^2).
        ortho = unit_vec(seed=99)
        ortho = ortho - np.dot(ortho, alice_vp) * alice_vp
        ortho = ortho / np.linalg.norm(ortho)
        target = 0.55 * alice_vp + np.sqrt(1 - 0.55 ** 2) * ortho
        target = target / np.linalg.norm(target)
        fake_embedder.extract.return_value = target

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(make_silence(5.0), SR, {"SPEAKER_00": [(0.0, 2.0)]})
        assert labels == {"SPEAKER_00": "alice"}

    def test_no_enrolled_users_all_unknown(self, fake_embedder, fake_store):
        """Empty enrollment store → every cluster labeled 'unknown'."""
        fake_store.get_all.return_value = {}
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(
            make_silence(10.0), SR,
            {"SPEAKER_00": [(0.0, 2.0)], "SPEAKER_01": [(3.0, 4.0)]},
        )
        assert labels == {"SPEAKER_00": "unknown", "SPEAKER_01": "unknown"}
        fake_embedder.extract.assert_not_called()  # nothing to compare against

    def test_picks_best_match_among_multiple(self, fake_embedder, fake_store):
        """Highest-cosine enrolled user wins."""
        alice_vp = unit_vec(seed=1)
        bob_vp = unit_vec(seed=2)
        fake_store.get_all.return_value = {"alice": alice_vp, "bob": bob_vp}
        # Embedding closer to bob than to alice — verify bob wins
        bob_like = 0.9 * bob_vp + 0.1 * alice_vp
        bob_like = bob_like / np.linalg.norm(bob_like)
        fake_embedder.extract.return_value = bob_like

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(make_silence(5.0), SR, {"S0": [(0.0, 2.0)]})
        assert labels == {"S0": "bob"}


class TestClusterIdentifierAudioExtraction:
    def test_passes_concatenated_audio_to_embedder(self, fake_embedder, fake_store):
        """Cluster audio = concat of all chosen segments from the waveform."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}
        fake_embedder.extract.return_value = alice_vp

        # Construct a waveform with distinct values in two regions so we can verify
        # which samples were passed to the embedder.
        audio = np.zeros(10 * SR, dtype=np.float32)
        audio[1 * SR:2 * SR] = 1.0      # segment A: 1.0–2.0 s
        audio[3 * SR:4 * SR] = 2.0      # segment B: 3.0–4.0 s

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        identifier.label_clusters(audio, SR, {"S0": [(1.0, 2.0), (3.0, 4.0)]})

        # The embedder should have been called exactly once with the concatenated
        # audio of the two segments (1s + 1s = 2s = 32000 samples).
        assert fake_embedder.extract.call_count == 1
        passed_audio = fake_embedder.extract.call_args.args[0]
        assert passed_audio.shape == (2 * SR,)
        # First half is all 1.0, second half is all 2.0
        assert np.all(passed_audio[:SR] == 1.0)
        assert np.all(passed_audio[SR:] == 2.0)

    def test_subsamples_long_cluster(self, fake_embedder, fake_store):
        """Cluster longer than max_sample_seconds should be subsampled."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}
        fake_embedder.extract.return_value = alice_vp

        audio = np.zeros(120 * SR, dtype=np.float32)
        # 12 segments of 5 s each = 60 s total; cap at 10 s
        segments = [(i * 10.0, i * 10.0 + 5.0) for i in range(12)]
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=10,
        )
        identifier.label_clusters(audio, SR, {"S0": segments})

        passed_audio = fake_embedder.extract.call_args.args[0]
        # Concatenated subsample must be <= max_sample_seconds * SR (a bit of slack for boundary)
        assert len(passed_audio) <= 10 * SR + SR  # allow up to one segment's slack


class TestClusterIdentifierErrors:
    def test_embedder_raising_labels_unknown(self, fake_embedder, fake_store):
        """If embedder.extract() raises, label that cluster as 'unknown' and continue."""
        alice_vp = unit_vec(seed=1)
        fake_store.get_all.return_value = {"alice": alice_vp}

        def raise_then_succeed(audio, sample_rate=SR):
            if raise_then_succeed.called:
                return alice_vp
            raise_then_succeed.called = True
            raise RuntimeError("simulated embedder failure")
        raise_then_succeed.called = False
        fake_embedder.extract.side_effect = raise_then_succeed

        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        labels = identifier.label_clusters(
            make_silence(20.0), SR,
            {"S0": [(0.0, 2.0)], "S1": [(5.0, 7.0)]},
        )
        # Cluster ordering in dict is insertion-order; S0 fails, S1 succeeds.
        assert labels["S0"] == "unknown"
        assert labels["S1"] == "alice"

    def test_empty_clusters_returns_empty_map(self, fake_embedder, fake_store):
        fake_store.get_all.return_value = {"alice": unit_vec(1)}
        identifier = ClusterIdentifier(
            embedder=fake_embedder, enrollment_store=fake_store,
            threshold=0.55, max_sample_seconds=30,
        )
        assert identifier.label_clusters(make_silence(5.0), SR, {}) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
py -3.14 -m pytest tests/diarization/test_identifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'modes.diarization.identifier'`.

- [ ] **Step 3: Implement `ClusterIdentifier`**

Create `target-vad/modes/diarization/identifier.py` with content:

```python
"""ClusterIdentifier — match pyannote clusters to enrolled speakers via centroid cosine.

For each cluster, the identifier:
  1. Picks an evenly-spaced ≤ N-second sample of the cluster's segments.
  2. Slices and concatenates that audio from the full waveform.
  3. Embeds via the injected ECAPA EmbeddingExtractor → 192-dim L2-normalized vector.
     (EmbeddingExtractor already L2-normalizes; concat-then-embed-once produces a
     single embedding that is itself a kind of centroid because ECAPA pools internally.)
  4. Cosine-matches against all enrolled voiceprints; assigns the best-scoring name
     if score >= threshold else the literal string "unknown".

If embedding fails for any cluster, that cluster is labeled "unknown" and processing
continues for the rest.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from core.speaker.verifier import cosine_similarity
from modes.diarization.sampling import sample_cluster_segments

logger = logging.getLogger(__name__)


class ClusterIdentifier:
    def __init__(
        self,
        embedder,
        enrollment_store,
        threshold: float = 0.55,
        max_sample_seconds: float = 30.0,
    ):
        self.embedder = embedder
        self.enrollment_store = enrollment_store
        self.threshold = threshold
        self.max_sample_seconds = max_sample_seconds

    def label_clusters(
        self,
        audio: np.ndarray,
        sample_rate: int,
        clusters: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, str]:
        """Return a {cluster_id: label} map. Label is enrolled name or 'unknown'.

        Args:
            audio: full waveform (float32 mono).
            sample_rate: must match what the embedder expects (16000).
            clusters: pyannote output as {cluster_id: [(start_s, end_s), ...]}.
        """
        if not clusters:
            return {}

        voiceprints = self.enrollment_store.get_all()
        if not voiceprints:
            # Nothing to compare against — every cluster is unknown without embedding.
            return {cid: "unknown" for cid in clusters}

        labels: Dict[str, str] = {}
        for cluster_id, segments in clusters.items():
            try:
                cluster_audio = self._extract_cluster_audio(audio, sample_rate, segments)
                embedding = self.embedder.extract(cluster_audio, sample_rate=sample_rate)
                labels[cluster_id] = self._best_label(embedding, voiceprints)
            except Exception as exc:  # pragma: no cover — exercised via mocks
                logger.warning("Embedding failed for cluster %s: %s — labeling unknown", cluster_id, exc)
                labels[cluster_id] = "unknown"

        return labels

    def _extract_cluster_audio(
        self, audio: np.ndarray, sample_rate: int, segments: List[Tuple[float, float]]
    ) -> np.ndarray:
        sampled = sample_cluster_segments(segments, max_seconds=self.max_sample_seconds)
        chunks = []
        for start_s, end_s in sampled:
            start_i = max(0, int(start_s * sample_rate))
            end_i = min(len(audio), int(end_s * sample_rate))
            if end_i > start_i:
                chunks.append(audio[start_i:end_i])
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def _best_label(self, embedding: np.ndarray, voiceprints: Dict[str, np.ndarray]) -> str:
        best_name = None
        best_score = -1.0
        for name, vp in voiceprints.items():
            score = cosine_similarity(embedding, vp)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is not None and best_score >= self.threshold:
            return best_name
        return "unknown"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
py -3.14 -m pytest tests/diarization/test_identifier.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run the full diarization test suite to confirm nothing else broke**

Run:

```bash
py -3.14 -m pytest tests/diarization/ -v
```

Expected: 16 passed (7 sampling + 7 output + 9 identifier — adjust if you added more).

- [ ] **Step 6: Commit**

```bash
git add target-vad/modes/diarization/identifier.py target-vad/tests/diarization/test_identifier.py
git commit -m "feat(diarization): add ClusterIdentifier for centroid cosine matching"
```

---

## Task 5: Diarizer — pyannote.audio pipeline wrapper

**Why no unit tests:** pyannote requires a real gated model download (HF token, ~1 GB) and produces deterministic output only with seed control we don't want to wire up. We make the class trivially thin — load pipeline once, run it, convert `Annotation` to our dict format — and verify it via the manual smoke test in Task 7. The integration-test approach matches the spec's `test_end_to_end.py.skip` design intent (skip by default).

**Files:**
- Create: `target-vad/modes/diarization/diarizer.py`

- [ ] **Step 1: Implement the Diarizer wrapper**

Create `target-vad/modes/diarization/diarizer.py` with content:

```python
"""Diarizer — thin wrapper around pyannote.audio's speaker-diarization-3.1 pipeline.

The pipeline is loaded lazily (first .diarize() call) because instantiation downloads
the model the first time and is slow even when cached. Subsequent calls reuse the
loaded pipeline.

Output is normalized from pyannote's Annotation object into a plain dict:
    {cluster_id: [(start_s, end_s), ...]}
sorted by start time within each cluster. Cluster IDs are pyannote's own strings
(e.g. "SPEAKER_00"), preserved as-is.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Diarizer:
    def __init__(self, pipeline_name: str, hf_token: str):
        if not hf_token:
            raise ValueError("Diarizer requires a non-empty HuggingFace token")
        self.pipeline_name = pipeline_name
        self.hf_token = hf_token
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline
        logger.info("Loading pyannote pipeline %s (first call may download model)", self.pipeline_name)
        self._pipeline = Pipeline.from_pretrained(self.pipeline_name, use_auth_token=self.hf_token)

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Run diarization on a mono float32 waveform.

        Returns a dict mapping cluster_id → list of (start_s, end_s) tuples sorted by start.
        Empty dict if pyannote finds no speech.
        """
        self._ensure_pipeline()

        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).unsqueeze(0).float()
        else:
            waveform = torch.from_numpy(audio).float()

        annotation = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        clusters: Dict[str, List[Tuple[float, float]]] = {}
        for segment, _, label in annotation.itertracks(yield_label=True):
            clusters.setdefault(label, []).append((float(segment.start), float(segment.end)))

        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda s: s[0])

        return clusters
```

- [ ] **Step 2: Smoke-import the module**

Run from `target-vad/`:

```bash
py -3.14 -c "from modes.diarization.diarizer import Diarizer; print('ok')"
```

Expected: `ok`. (No model load here — `__init__` only validates the token.)

- [ ] **Step 3: Verify the rejection-on-empty-token branch**

Run:

```bash
py -3.14 -c "from modes.diarization.diarizer import Diarizer; Diarizer('p', '')"
```

Expected: `ValueError: Diarizer requires a non-empty HuggingFace token`.

- [ ] **Step 4: Run the existing diarization tests to confirm nothing else broke**

Run:

```bash
py -3.14 -m pytest tests/diarization/ -q
```

Expected: same count as Task 4 step 5 — still passing.

- [ ] **Step 5: Commit**

```bash
git add target-vad/modes/diarization/diarizer.py
git commit -m "feat(diarization): add pyannote pipeline wrapper"
```

---

## Task 6: `diarize.py` CLI entry point

**What it does:** Parses args, loads config + HF token, loads + resamples audio, runs `Diarizer`, runs `ClusterIdentifier`, flattens clusters to a sorted segment list, writes JSON (and optional RTTM). All errors map to the exit codes documented in the spec's error-handling table.

**Files:**
- Create: `target-vad/diarize.py`

- [ ] **Step 1: Implement the CLI**

Create `target-vad/diarize.py` with content:

```python
"""Classroom diarization entry point — see docs/superpowers/specs/2026-05-14-classroom-diarization-design.md."""

from core import compat  # noqa: F401 — torchaudio/speechbrain shim, must precede speechbrain imports

import argparse
import datetime as dt
import os
import sys
from typing import List

import numpy as np
import soundfile as sf
import yaml
from rich.console import Console

from core.speaker.embedder import EmbeddingExtractor
from core.speaker.enrollment_store import EnrollmentStore
from modes.diarization.diarizer import Diarizer
from modes.diarization.identifier import ClusterIdentifier
from modes.diarization.output import DiarizationSegment, write_json, write_rttm

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CONFIG_OR_MODEL = 3


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_audio_as_mono16k(path: str) -> np.ndarray:
    """Read a WAV file and return mono float32 at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mix down
    if sr != 16000:
        from scipy.signal import resample_poly
        # Use rational resampling for exact rate change
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def flatten_clusters(
    clusters: dict, cluster_labels: dict
) -> List[DiarizationSegment]:
    """Convert {cluster_id: [(start,end),...]} + {cluster_id: label} → sorted segment list."""
    segments: List[DiarizationSegment] = []
    for cid, time_ranges in clusters.items():
        label = cluster_labels.get(cid, "unknown")
        for start, end in time_ranges:
            segments.append(DiarizationSegment(start=start, end=end, speaker=label))
    segments.sort(key=lambda s: s.start)
    return segments


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD — Classroom Diarization (S1)")
    parser.add_argument("input", help="Path to a WAV file")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input>.diarization.json)")
    parser.add_argument("--rttm", action="store_true", help="Also write an RTTM file alongside the JSON")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", action="store_true", help="Reserved — JSON-lines event log (not yet wired)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        console.print(f"[red]Input file not found:[/] {args.input}")
        return EXIT_BAD_INPUT

    config = load_config(args.config)
    diar_cfg = config["diarization"]
    hf_token_var = diar_cfg["hf_token_env_var"]
    hf_token = os.environ.get(hf_token_var, "")
    if not hf_token:
        console.print(
            f"[red]HuggingFace token not set.[/] Get one at https://hf.co/settings/tokens, "
            f"accept the gated model at https://hf.co/{diar_cfg['pyannote_pipeline']}, "
            f"then set [bold]{hf_token_var}[/] in your environment."
        )
        return EXIT_CONFIG_OR_MODEL

    # Load audio
    try:
        console.print(f"[dim]Loading[/] {args.input}")
        audio = load_audio_as_mono16k(args.input)
    except Exception as exc:
        console.print(f"[red]Failed to read audio file:[/] {exc}")
        console.print("[dim]If this is an unusual codec, try converting to PCM WAV first (e.g. with ffmpeg).[/]")
        return EXIT_BAD_INPUT

    duration_s = float(len(audio) / 16000)
    console.print(f"[dim]Loaded[/] {duration_s:.1f}s of audio @ 16 kHz mono")

    # Diarize
    diarizer = Diarizer(pipeline_name=diar_cfg["pyannote_pipeline"], hf_token=hf_token)
    try:
        with console.status("[bold]Diarizing...[/]", spinner="dots"):
            clusters = diarizer.diarize(audio, sample_rate=16000)
    except Exception as exc:
        console.print(f"[red]Diarization failed:[/] {exc}")
        console.print(
            "[dim]If this looks like a model download issue, check your HF token has access "
            f"to {diar_cfg['pyannote_pipeline']} (the model is gated and requires accepting its license).[/]"
        )
        return EXIT_CONFIG_OR_MODEL

    if not clusters:
        console.print("[yellow]No speech detected — writing empty timeline.[/]")
        labels: dict = {}
    else:
        console.print(f"[green]{len(clusters)} cluster(s) found.[/] Identifying...")

        # Identify clusters
        embedder = EmbeddingExtractor()
        store = EnrollmentStore(config["core"]["paths"]["voiceprints_dir"])
        if not store.list_users():
            console.print("[yellow]No enrolled voiceprints — all clusters will be labeled 'unknown'.[/]")
        identifier = ClusterIdentifier(
            embedder=embedder,
            enrollment_store=store,
            threshold=diar_cfg["identification_threshold"],
            max_sample_seconds=diar_cfg["centroid_max_sample_seconds"],
        )
        labels = identifier.label_clusters(audio, sample_rate=16000, clusters=clusters)
        for cid, label in labels.items():
            console.print(f"  [dim]{cid}[/] → [bold]{label}[/]")

    # Build output
    segments = flatten_clusters(clusters, labels)
    out_path = args.out or (args.input + ".diarization.json")
    write_json(
        out_path,
        audio_file=os.path.abspath(args.input),
        duration_s=duration_s,
        diarized_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config={
            "pyannote_pipeline": diar_cfg["pyannote_pipeline"],
            "identification_threshold": diar_cfg["identification_threshold"],
        },
        segments=segments,
    )
    console.print(f"[green]Wrote[/] {out_path}")

    if args.rttm:
        rttm_path = os.path.splitext(out_path)[0] + ".rttm"
        audio_id = os.path.splitext(os.path.basename(args.input))[0]
        write_rttm(rttm_path, audio_file_id=audio_id, segments=segments)
        console.print(f"[green]Wrote[/] {rttm_path}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test `--help`**

Run from `target-vad/`:

```bash
py -3.14 diarize.py --help
```

Expected: usage block printed with all flags (`input`, `--out`, `--rttm`, `--config`, `--log`), exit 0.

- [ ] **Step 3: Smoke-test missing-input error path**

Run:

```bash
py -3.14 diarize.py /tmp/does-not-exist.wav
```

Expected: `Input file not found: /tmp/does-not-exist.wav`, exit code 2.

Verify the exit code on bash:

```bash
py -3.14 diarize.py /tmp/does-not-exist.wav; echo "exit=$?"
```

Expected: `exit=2`.

- [ ] **Step 4: Smoke-test missing-HF-token error path**

Make sure `HF_TOKEN` is unset:

```bash
HF_TOKEN= py -3.14 diarize.py target-vad/config.yaml
```

(Using config.yaml just as a file-that-exists; the HF check happens before audio loading. Even if it's not a valid WAV, the HF check fires first.)

Expected: HF token error message + exit code 3.

Actually — check order: in the CLI, file existence is verified before the HF token. So with a real existing non-WAV file, the order matters. Pick a path that exists but isn't a WAV (config.yaml works), and the test passes because file-existence check passes, then HF check fires. Good.

```bash
HF_TOKEN= py -3.14 diarize.py config.yaml; echo "exit=$?"
```

Expected: `exit=3` with the HF token guidance message.

- [ ] **Step 5: Commit**

```bash
git add target-vad/diarize.py
git commit -m "feat(diarization): add diarize.py CLI entry point"
```

---

## Task 7: Manual end-to-end validation

**Why manual:** Running pyannote requires a real HF token and a real WAV file. We don't bake this into pytest (matches the spec's `test_end_to_end.py.skip` intent). Instead, we exercise the full pipeline interactively and capture results in the session.

**Files:** none modified or created.

- [ ] **Step 1: Confirm prerequisites with the user**

Before running, ask:

1. Does the user have an HF token with access to `pyannote/speaker-diarization-3.1`? (Model is gated — license must be accepted on its HF page.)
2. Is there a short multi-speaker WAV available (mono or stereo, any sample rate — `load_audio_as_mono16k` will convert)?
3. Are there enrolled voiceprints in `target-vad/voiceprints/` from prior `enroll.py` runs? (At least one is needed to test the match path.)

If any answer is no, stop and report what's missing instead of attempting a partial run.

- [ ] **Step 2: Run the pipeline against the user's recording**

With `HF_TOKEN` exported in the shell, from `target-vad/`:

```bash
py -3.14 diarize.py <path-to-test.wav>
```

Expected console output:
- "Loading <path>" → "Loaded N.Ns of audio @ 16 kHz mono"
- A spinner "Diarizing..." (may take 30 s – several minutes depending on file length and whether the model needs to download)
- "K cluster(s) found. Identifying..."
- One line per cluster: `SPEAKER_NN → <name or unknown>`
- "Wrote <path>.diarization.json"
- Exit code 0

- [ ] **Step 3: Inspect the JSON output**

```bash
cat <path-to-test.wav>.diarization.json
```

Verify:
- `audio_file`, `duration_s`, `diarized_at`, `config` populated
- `enrolled_users_matched` non-empty if any cluster matched an enrolled name
- `segments` sorted by `start`, each with `start`, `end`, `speaker`
- No segment has a `speaker` value that is neither `"unknown"` nor a name returned by `list_users()`

- [ ] **Step 4: Re-run with `--rttm` and verify the RTTM file**

```bash
py -3.14 diarize.py <path-to-test.wav> --rttm
cat <path-to-test.wav>.diarization.rttm
```

Verify each line matches: `SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>`.

- [ ] **Step 5: Run the full pytest suite a final time**

```bash
py -3.14 -m pytest tests/ -q
```

Expected: same count as before plus the diarization unit tests, all passing.

- [ ] **Step 6: Report results to the user**

Summarize for the user:
- Was a real HF download required (cold cache) or was the model cached?
- Cluster count vs. expected (e.g., "3 speakers in the recording, pyannote produced 3 clusters" or "produced 5 clusters — over-clustering").
- Identification outcomes: which enrolled names matched, with cosine scores if available.
- DER cannot be computed here without a labeled reference; flag this if the user needs a quantitative quality measure.

- [ ] **Step 7: If everything works, no extra commit is needed for this task** — the prior commits already cover the implementation. Move to optional follow-ups below.

---

## Optional follow-ups (do NOT block completion)

These are flagged in the spec but explicitly deferred unless the user requests them:

- **JSON-lines event log:** the `--log` flag is currently a no-op. Wire it the same way the kiosk's `on_event` hook works once the user wants offline tuning data.
- **DER measurement harness:** if the user wants to track diarization quality over recordings, add a `tools/eval_der.py` that consumes a reference RTTM and the produced RTTM. Out of scope for this plan.
- **Phase 2 ASR:** spec already reserves the JSON schema for a future `text` field per segment — add `transcribe.py` post-processor when Phase 2 is greenlit.

---

## Self-review checklist (already applied by the plan author)

- **Spec coverage:** All 12 sections of the spec map to tasks above: inputs/outputs → Tasks 3/6; pipeline → Tasks 4/5/6; centroid sampling → Task 2; components table → Tasks 2–6; CLI → Task 6; configuration → Task 1; error handling → Task 6 (all rows covered: missing input → exit 2; unreadable audio → exit 2; missing HF token → exit 3; pyannote model failure → exit 3; zero clusters → exit 0 with empty list; embedder per-cluster failure → unknown label; no enrolled → unknown label).
- **Placeholders:** none.
- **Type consistency:** `DiarizationSegment` defined in Task 3 and consumed in Tasks 4/6; `Diarizer.diarize` returns `dict[str, list[tuple[float, float]]]` in Task 5 and consumed identically in Tasks 4/6; `ClusterIdentifier.label_clusters` returns `dict[str, str]` in Task 4 and consumed in `flatten_clusters` in Task 6. Names match across tasks.
- **`enrolled_users_matched` rule:** spec says "deduped list of enrolled names that appeared at least once, ordered by first appearance time." Task 3 helper `_enrolled_users_in_first_appearance_order` walks segments in input order; `flatten_clusters` (Task 6) sorts segments by start time before passing them in, so first-appearance-by-time is preserved through the pipeline.
