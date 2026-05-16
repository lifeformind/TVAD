# Prosody Pass (Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `prosody.py`, the fourth analysis pass. Reads a post-2A diarization JSON + the audio WAV, computes per-segment pitch / energy / speaking-rate features via librosa DSP, attaches a 7-field `prosody` block per segment, and emits top-level `prosody_baselines` summarizing each speaker's median + IQR distribution.

**Architecture:** Bottom-up TDD. Tiny refactor first to extract the duplicated `load_audio_as_mono16k` from `diarize.py`/`transcribe.py` into a shared `core/audio/load.py`. Then a pure `analyze_segment(audio_chunk, sample_rate, words, segment_duration, cfg)` function in `modes/prosody/analyzer.py` (testable on synthetic sine waves). Then a pure `compute_baselines(segments)` in `modes/prosody/baselines.py`. Then `prosody.py` orchestrates: load audio once, walk segments, atomic-write back. Finally a smoke against the existing Voice 001 fixture.

**Tech Stack:** Python 3.14 (`py -3.14`, never `python` — 3.12 lacks the dep stack). New dep: `librosa>=0.10.0` (transitively pulls `numba`, `audioread`, `pooch`; `scipy`/`soundfile`/`numpy` already present). No model downloads; pure DSP. Reused existing deps: `rich`, `pyyaml`, `numpy`.

**Spec:** [`docs/superpowers/specs/2026-05-16-prosody-pass-design.md`](../specs/2026-05-16-prosody-pass-design.md). Read once before starting Task 1.

**Working directory:** `c:\repos\TVAD\target-vad\` for python/pytest. Git commands run from `c:\repos\TVAD\`.

---

## File Structure

Files this plan creates or modifies (relative to `target-vad/`):

| Path | Status | Responsibility |
|---|---|---|
| `core/audio/load.py` | create | Shared `load_audio_as_mono16k(path)` helper |
| `diarize.py` | modify | Replace inline `load_audio_as_mono16k` with import from `core.audio.load` |
| `transcribe.py` | modify | Same — replace inline copy with shared import |
| `requirements.txt` | modify | add explicit `librosa>=0.10.0` pin |
| `config.yaml` | modify | add top-level `prosody:` block |
| `modes/prosody/__init__.py` | create | empty package marker |
| `modes/prosody/analyzer.py` | create | `analyze_segment(...)` — pure function returning 7-field prosody dict |
| `modes/prosody/baselines.py` | create | `compute_baselines(segments)` — pure aggregation over the segment list |
| `prosody.py` | create | CLI entry: arg parsing, audio load, JSON load/save, orchestration, atomic write |
| `tests/prosody/__init__.py` | create | empty |
| `tests/prosody/test_analyzer.py` | create | analyzer tests on synthetic audio |
| `tests/prosody/test_baselines.py` | create | baseline aggregation tests on synthetic prosody dicts |
| `tests/prosody/test_orchestration.py` | create | CLI tests with a stub analyzer + tiny synthetic WAV |

No modifications to existing kiosk, enrollment, sentiment, or metrics code — Phase 4 is purely additive after the small shared-loader refactor.

---

## Task 1: Extract `load_audio_as_mono16k` into a shared module

**Files:**
- Create: `target-vad/core/audio/load.py`
- Modify: `target-vad/diarize.py` (replace inline function with import)
- Modify: `target-vad/transcribe.py` (replace inline function with import)

- [ ] **Step 1: Inspect current call sites**

```bash
grep -rn "load_audio_as_mono16k" c:/repos/TVAD/target-vad/ --include="*.py"
```

Expected hits: `diarize.py:40` (definition), `diarize.py:134` (use), `transcribe.py:42` (definition), `transcribe.py:115` (use). Confirm before changing.

- [ ] **Step 2: Run baseline tests**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 218 passed. STOP if not.

- [ ] **Step 3: Create the shared module**

Create `target-vad/core/audio/load.py`:

```python
"""Shared audio loader — read a WAV file and return mono float32 at 16 kHz.

Used by diarize.py (S1), transcribe.py (Phase 2A), and prosody.py (Phase 4).
The function was duplicated in those entry points pre-Phase-4; this module
deduplicates it so all consumers get identical resampling and channel-mixing
semantics.
"""

from math import gcd

import numpy as np
import soundfile as sf


def load_audio_as_mono16k(path: str) -> np.ndarray:
    """Read a WAV file and return mono float32 at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = gcd(sr, 16000)
        audio = resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
    return audio.astype(np.float32, copy=False)
```

- [ ] **Step 4: Update `diarize.py` to import the shared helper**

Open `target-vad/diarize.py`. Add to the imports block (near the other `from core.*` imports, around line 16-20):

```python
from core.audio.load import load_audio_as_mono16k
```

Delete the inline definition (`def load_audio_as_mono16k(path: str) -> np.ndarray:` and its body, lines ~40-50). The two top-level `import` lines for `soundfile as sf` and `numpy as np` at the top of `diarize.py` are still needed for other code in the file — leave them.

- [ ] **Step 5: Update `transcribe.py` to import the shared helper**

Open `target-vad/transcribe.py`. Add to the imports block near the other `from core.*` imports:

```python
from core.audio.load import load_audio_as_mono16k
```

Delete the inline definition (def block around line 42-52). Keep any other imports the file still needs.

- [ ] **Step 6: Run the full suite — must remain green**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 218 passed. Same as before — this is a no-op refactor. If anything fails, the import paths are wrong or the inline-deletion didn't remove the right lines.

- [ ] **Step 7: Quick smoke import check**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "from core.audio.load import load_audio_as_mono16k; print(load_audio_as_mono16k.__doc__)"
```

Expected: `Read a WAV file and return mono float32 at 16 kHz.`

- [ ] **Step 8: Commit**

```bash
git -C c:/repos/TVAD add target-vad/core/audio/load.py target-vad/diarize.py target-vad/transcribe.py
git -C c:/repos/TVAD commit -m "refactor(core): extract load_audio_as_mono16k into shared core/audio/load.py"
```

---

## Task 2: Pin `librosa` and add `prosody:` config block

**Files:**
- Modify: `target-vad/requirements.txt`
- Modify: `target-vad/config.yaml`

- [ ] **Step 1: Append `librosa` to requirements.txt**

Open `target-vad/requirements.txt`. Append one line at the end:

```
librosa>=0.10.0
```

- [ ] **Step 2: Verify librosa is installable (if not already cached)**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "import librosa; print('librosa', librosa.__version__)"
```

If this prints a version number → already installed (transitively or directly), proceed.

If it raises `ModuleNotFoundError`, install it:

```bash
py -3.14 -m pip install "librosa>=0.10.0"
```

**If pip wants to downgrade `numpy`, `scipy`, `torch`, or `numba` significantly:** STOP and report BLOCKED — we don't downgrade the project's pinned stack without authorization.

- [ ] **Step 3: Confirm librosa exposes pyin and rms**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "import librosa; print(hasattr(librosa, 'pyin')); print(hasattr(librosa.feature, 'rms')); print(hasattr(librosa, 'amplitude_to_db'))"
```

Expected: three `True` lines.

- [ ] **Step 4: Add `prosody:` config block**

Open `target-vad/config.yaml`. Append after the existing top-level blocks (do not touch them):

```yaml

prosody:
  pitch_min_hz: 80           # pyin floor — typical human speech floor
  pitch_max_hz: 400          # pyin ceiling — typical adult speech ceiling
  frame_length_ms: 25        # standard speech-analysis window
  hop_length_ms: 10          # standard 60% overlap
```

Use 2-space indentation matching the existing blocks.

- [ ] **Step 5: Verify config parses with seven top-level blocks**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(sorted(c.keys())); print(c['prosody'])"
```

Expected first line: `['core', 'diarization', 'kiosk', 'metrics', 'prosody', 'sentiment', 'transcription']`.
Expected second line: dict with the four `prosody:` knobs.

- [ ] **Step 6: Run the full test suite to confirm baseline**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 218 passed.

- [ ] **Step 7: Commit**

```bash
git -C c:/repos/TVAD add target-vad/requirements.txt target-vad/config.yaml
git -C c:/repos/TVAD commit -m "feat(prosody): pin librosa and add prosody config block"
```

---

## Task 3: Package skeleton + `analyze_segment` analyzer

**Files:**
- Create: `target-vad/modes/prosody/__init__.py`
- Create: `target-vad/modes/prosody/analyzer.py`
- Create: `target-vad/tests/prosody/__init__.py`
- Create: `target-vad/tests/prosody/test_analyzer.py`

- [ ] **Step 1: Create empty package markers**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -c "open(r'modes/prosody/__init__.py','w').close(); open(r'tests/prosody/__init__.py','w').close()"
```

- [ ] **Step 2: Write failing analyzer tests**

Create `target-vad/tests/prosody/test_analyzer.py`:

```python
"""Tests for the prosody analyzer pure function."""

import numpy as np
import pytest

from modes.prosody import analyzer


SR = 16000
DEFAULT_CFG = {
    "pitch_min_hz": 80,
    "pitch_max_hz": 400,
    "frame_length_ms": 25,
    "hop_length_ms": 10,
}


def _sine(freq_hz: float, duration_s: float, sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    """Generate a pure sine tone at given frequency."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _word(start: float, end: float, w: str = "x") -> dict:
    return {"start": start, "end": end, "word": w, "probability": 0.9}


class TestAnalyzeSegment:
    def test_pure_200hz_sine_centered_pitch(self):
        audio = _sine(200.0, 1.5)
        result = analyzer.analyze_segment(audio, SR, words=[_word(0, 1.5)], segment_duration=1.5, cfg=DEFAULT_CFG)
        assert result["pitch_hz_median"] == pytest.approx(200.0, abs=5.0)
        assert result["pitch_hz_std"] < 5.0
        assert result["pitch_range_hz"] < 5.0

    def test_silence_gives_null_pitch_and_low_energy(self):
        audio = np.zeros(int(SR * 1.0), dtype=np.float32)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=1.0, cfg=DEFAULT_CFG)
        assert result["pitch_hz_median"] is None
        assert result["pitch_hz_std"] is None
        assert result["pitch_range_hz"] is None
        # librosa.amplitude_to_db on zero RMS returns very low dB (clipped near -80 to -100)
        assert result["energy_db_mean"] is not None
        assert result["energy_db_mean"] < -50.0

    def test_concatenated_sines_reflect_range(self):
        # 1s of 200Hz then 1s of 400Hz — pitch range should reflect the spread.
        audio = np.concatenate([_sine(200.0, 1.0), _sine(400.0, 1.0)])
        result = analyzer.analyze_segment(audio, SR, words=[_word(0, 2.0)], segment_duration=2.0, cfg=DEFAULT_CFG)
        # 5th-95th percentile spans most of the range; allow generous tolerance for pyin noise.
        assert result["pitch_range_hz"] > 100.0

    def test_empty_words_gives_null_rate_fields(self):
        audio = _sine(200.0, 1.0)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=1.0, cfg=DEFAULT_CFG)
        assert result["speech_rate_wps"] is None
        assert result["pause_ratio"] is None

    def test_three_words_over_two_seconds_rate_and_pause(self):
        # 3 words spanning 0–0.3, 0.4–0.7, 0.8–1.1 (total word duration = 0.9s, segment = 2.0s)
        words = [_word(0.0, 0.3), _word(0.4, 0.7), _word(0.8, 1.1)]
        audio = _sine(200.0, 2.0)
        result = analyzer.analyze_segment(audio, SR, words=words, segment_duration=2.0, cfg=DEFAULT_CFG)
        assert result["speech_rate_wps"] == pytest.approx(1.5)  # 3 words / 2.0s
        assert result["pause_ratio"] == pytest.approx((2.0 - 0.9) / 2.0)  # 0.55

    def test_words_exceeding_segment_duration_clamps_pause_to_zero(self):
        # Total word duration > segment duration → pause_ratio clamped to 0.0
        words = [_word(0.0, 1.5), _word(1.0, 2.5)]  # 1.5 + 1.5 = 3.0s across 2.0s segment
        audio = _sine(200.0, 2.0)
        result = analyzer.analyze_segment(audio, SR, words=words, segment_duration=2.0, cfg=DEFAULT_CFG)
        assert result["pause_ratio"] == 0.0

    def test_zero_length_audio_chunk(self):
        audio = np.array([], dtype=np.float32)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=0.0, cfg=DEFAULT_CFG)
        assert result["pitch_hz_median"] is None
        assert result["pitch_hz_std"] is None
        assert result["pitch_range_hz"] is None
        assert result["energy_db_mean"] is None
        assert result["energy_db_range"] is None
        assert result["speech_rate_wps"] is None
        assert result["pause_ratio"] is None

    def test_pitch_config_range_honored(self):
        # Configured 50-100 Hz range; 200 Hz sine should produce no voiced frames (out of range).
        cfg = {"pitch_min_hz": 50, "pitch_max_hz": 100, "frame_length_ms": 25, "hop_length_ms": 10}
        audio = _sine(200.0, 1.0)
        result = analyzer.analyze_segment(audio, SR, words=[], segment_duration=1.0, cfg=cfg)
        # pyin should not find voiced frames at 200 Hz when fmax=100; pitch fields null.
        assert result["pitch_hz_median"] is None
```

- [ ] **Step 3: Run, confirm ImportError**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/prosody/test_analyzer.py -v
```

Expected: `ImportError` on `from modes.prosody import analyzer`.

- [ ] **Step 4: Implement `analyze_segment`**

Create `target-vad/modes/prosody/analyzer.py`:

```python
"""Per-segment prosody analyzer — pure function over an audio chunk + word timestamps.

Computes pitch (median / std / range), energy (mean / range in dB), and
rate (words-per-second / pause ratio). All seven fields may be null in
degenerate cases (no voiced frames, zero-length audio, empty word list).
No I/O, no model loads, no global state.
"""

from typing import Dict, List, Optional

import librosa
import numpy as np


def _round_or_none(x: Optional[float], decimals: int = 2) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), decimals)


def analyze_segment(
    audio_chunk: np.ndarray,
    sample_rate: int,
    words: List[Dict],
    segment_duration: float,
    cfg: Dict,
) -> Dict:
    """Compute prosody features for a single segment.

    Args:
        audio_chunk: mono float32 audio for this segment.
        sample_rate: e.g. 16000.
        words: list of {start, end, word, probability} from 2A. May be empty.
        segment_duration: duration_s from the segment's start/end. Must be >= 0.
        cfg: dict with keys pitch_min_hz, pitch_max_hz, frame_length_ms, hop_length_ms.

    Returns:
        Dict with seven keys (pitch_hz_median, pitch_hz_std, pitch_range_hz,
        energy_db_mean, energy_db_range, speech_rate_wps, pause_ratio). Each
        value is a float or None.
    """
    pitch_min = cfg["pitch_min_hz"]
    pitch_max = cfg["pitch_max_hz"]
    frame_length = int(sample_rate * cfg["frame_length_ms"] / 1000)
    hop_length = int(sample_rate * cfg["hop_length_ms"] / 1000)

    # Pitch via pyin.
    if len(audio_chunk) >= frame_length:
        try:
            f0, _voiced_flag, _voiced_prob = librosa.pyin(
                audio_chunk,
                fmin=pitch_min,
                fmax=pitch_max,
                sr=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:
            voiced_f0 = np.array([])
    else:
        voiced_f0 = np.array([])

    if len(voiced_f0) > 0:
        pitch_median = float(np.median(voiced_f0))
        pitch_std = float(np.std(voiced_f0))
        pitch_range = float(np.percentile(voiced_f0, 95) - np.percentile(voiced_f0, 5))
    else:
        pitch_median = None
        pitch_std = None
        pitch_range = None

    # Energy via RMS.
    if len(audio_chunk) > 0:
        try:
            rms = librosa.feature.rms(
                y=audio_chunk,
                frame_length=frame_length,
                hop_length=hop_length,
            )[0]
            db = librosa.amplitude_to_db(rms, ref=1.0)
            energy_db_mean = float(np.mean(db))
            energy_db_range = float(np.percentile(db, 95) - np.percentile(db, 5))
        except Exception:
            energy_db_mean = None
            energy_db_range = None
    else:
        energy_db_mean = None
        energy_db_range = None

    # Rate from word timestamps.
    if words and segment_duration > 0:
        word_total = sum(w["end"] - w["start"] for w in words)
        speech_rate_wps = len(words) / segment_duration
        pause_ratio = max(0.0, min(1.0, (segment_duration - word_total) / segment_duration))
    else:
        speech_rate_wps = None
        pause_ratio = None

    return {
        "pitch_hz_median": _round_or_none(pitch_median),
        "pitch_hz_std": _round_or_none(pitch_std),
        "pitch_range_hz": _round_or_none(pitch_range),
        "energy_db_mean": _round_or_none(energy_db_mean),
        "energy_db_range": _round_or_none(energy_db_range),
        "speech_rate_wps": _round_or_none(speech_rate_wps),
        "pause_ratio": _round_or_none(pause_ratio),
    }
```

- [ ] **Step 5: Run, confirm all 8 tests pass**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/prosody/test_analyzer.py -v
```

Expected: 8 passed. Pyin can be slow on first run (numba JIT compile) — allow a few seconds.

If `test_concatenated_sines_reflect_range` fails because pyin's tolerance is tighter than expected, **lower the assertion bound** (e.g. `> 50.0` instead of `> 100.0`). Don't tighten the algorithm to chase the test — pyin's noise on synthetic sines is real.

- [ ] **Step 6: Run full suite — no regressions**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 226 passed (218 baseline + 8 analyzer).

- [ ] **Step 7: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/prosody/__init__.py target-vad/modes/prosody/analyzer.py target-vad/tests/prosody/__init__.py target-vad/tests/prosody/test_analyzer.py
git -C c:/repos/TVAD commit -m "feat(prosody): add analyze_segment with pitch / energy / rate features"
```

---

## Task 4: `compute_baselines` per-speaker aggregation

**Files:**
- Create: `target-vad/modes/prosody/baselines.py`
- Create: `target-vad/tests/prosody/test_baselines.py`

- [ ] **Step 1: Write failing tests**

Create `target-vad/tests/prosody/test_baselines.py`:

```python
"""Tests for compute_baselines — pure aggregation over a segment list."""

import pytest

from modes.prosody import baselines


def _seg(speaker_id: str, prosody=None) -> dict:
    return {"speaker_id": speaker_id, "speaker": speaker_id, "prosody": prosody}


def _p(pitch_median=None, pitch_std=None, pitch_range=None,
       energy_db_mean=None, energy_db_range=None,
       speech_rate_wps=None, pause_ratio=None) -> dict:
    """Build a 7-field prosody dict — defaults to all-null."""
    return {
        "pitch_hz_median": pitch_median, "pitch_hz_std": pitch_std,
        "pitch_range_hz": pitch_range, "energy_db_mean": energy_db_mean,
        "energy_db_range": energy_db_range,
        "speech_rate_wps": speech_rate_wps, "pause_ratio": pause_ratio,
    }


class TestComputeBaselines:
    def test_two_speakers_three_segments_each(self):
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", _p(pitch_median=145, energy_db_mean=-23)),
            _seg("alice", _p(pitch_median=150, energy_db_mean=-21)),
            _seg("bob", _p(pitch_median=170, energy_db_mean=-22)),
            _seg("bob", _p(pitch_median=175, energy_db_mean=-20)),
            _seg("bob", _p(pitch_median=180, energy_db_mean=-18)),
        ]
        result = baselines.compute_baselines(segments)
        assert set(result.keys()) == {"alice", "bob"}
        # Alice: pitch median = 145; IQR = p75 - p25 = 147.5 - 142.5 = 5.0
        assert result["alice"]["pitch_hz_median"] == pytest.approx(145.0)
        assert result["alice"]["pitch_hz_iqr"] == pytest.approx(5.0)
        assert result["alice"]["energy_db_median"] == pytest.approx(-23.0)
        assert result["alice"]["segment_count"] == 3
        # Bob: pitch median = 175
        assert result["bob"]["pitch_hz_median"] == pytest.approx(175.0)
        assert result["bob"]["segment_count"] == 3

    def test_all_null_prosody_speaker_omitted(self):
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("bob", None),
            _seg("bob", _p()),  # all-null dict — bob has no non-null fields
        ]
        result = baselines.compute_baselines(segments)
        assert "alice" in result
        assert "bob" not in result

    def test_mixed_null_segments_partial_baseline(self):
        # Alice has 2 segments with prosody, 1 with prosody: null.
        # Baseline computed from the 2 valid ones; segment_count = 2.
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", None),
            _seg("alice", _p(pitch_median=160, energy_db_mean=-21)),
        ]
        result = baselines.compute_baselines(segments)
        assert result["alice"]["pitch_hz_median"] == pytest.approx(150.0)  # median of [140, 160]
        assert result["alice"]["segment_count"] == 2

    def test_iqr_constant_sequence_zero(self):
        segments = [
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
            _seg("alice", _p(pitch_median=140, energy_db_mean=-25)),
        ]
        result = baselines.compute_baselines(segments)
        assert result["alice"]["pitch_hz_iqr"] == 0.0
        assert result["alice"]["energy_db_iqr"] == 0.0
```

- [ ] **Step 2: Run, confirm ImportError**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/prosody/test_baselines.py -v
```

Expected: ImportError on `from modes.prosody import baselines`.

- [ ] **Step 3: Implement `compute_baselines`**

Create `target-vad/modes/prosody/baselines.py`:

```python
"""Pure per-speaker baseline aggregation over the prosody-enriched segment list.

For each speaker_id appearing in segments, computes:
  - pitch_hz_median:  median across the speaker's non-null pitch_hz_median values
  - pitch_hz_iqr:     75th - 25th percentile of those values (interquartile range)
  - energy_db_median: median across the speaker's non-null energy_db_mean values
  - energy_db_iqr:    75th - 25th percentile of those values
  - segment_count:    number of segments contributing at least one non-null
                      prosody field

Speakers whose every segment has prosody: null or all-null fields are omitted
from the result entirely (not emitted with null baselines). Robust to outliers
because median + IQR aren't pulled around by single shouts or whispers.
"""

from typing import Dict, List

import numpy as np


def compute_baselines(segments: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-speaker prosody baselines from a list of enriched segments."""
    by_speaker: Dict[str, Dict[str, List[float]]] = {}
    seg_counts: Dict[str, int] = {}

    for seg in segments:
        sid = seg["speaker_id"]
        prosody = seg.get("prosody")
        if prosody is None:
            continue

        pitch = prosody.get("pitch_hz_median")
        energy = prosody.get("energy_db_mean")
        # Count as a contributing segment if it has any non-null prosody field.
        any_non_null = any(prosody.get(k) is not None for k in (
            "pitch_hz_median", "pitch_hz_std", "pitch_range_hz",
            "energy_db_mean", "energy_db_range",
            "speech_rate_wps", "pause_ratio",
        ))
        if not any_non_null:
            continue

        bucket = by_speaker.setdefault(sid, {"pitch": [], "energy": []})
        seg_counts[sid] = seg_counts.get(sid, 0) + 1
        if pitch is not None:
            bucket["pitch"].append(float(pitch))
        if energy is not None:
            bucket["energy"].append(float(energy))

    result: Dict[str, Dict] = {}
    for sid, bucket in by_speaker.items():
        pitch_vals = bucket["pitch"]
        energy_vals = bucket["energy"]
        entry: Dict = {}
        if pitch_vals:
            entry["pitch_hz_median"] = round(float(np.median(pitch_vals)), 2)
            entry["pitch_hz_iqr"] = round(
                float(np.percentile(pitch_vals, 75) - np.percentile(pitch_vals, 25)), 2
            )
        else:
            entry["pitch_hz_median"] = None
            entry["pitch_hz_iqr"] = None
        if energy_vals:
            entry["energy_db_median"] = round(float(np.median(energy_vals)), 2)
            entry["energy_db_iqr"] = round(
                float(np.percentile(energy_vals, 75) - np.percentile(energy_vals, 25)), 2
            )
        else:
            entry["energy_db_median"] = None
            entry["energy_db_iqr"] = None
        entry["segment_count"] = seg_counts[sid]
        result[sid] = entry
    return result
```

- [ ] **Step 4: Run, confirm 4 baseline tests pass**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/prosody/ -v
```

Expected: 12 passed (8 analyzer + 4 baselines).

- [ ] **Step 5: Run full suite**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 230 passed (218 baseline + 8 analyzer + 4 baselines).

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/modes/prosody/baselines.py target-vad/tests/prosody/test_baselines.py
git -C c:/repos/TVAD commit -m "feat(prosody): add compute_baselines per-speaker aggregator"
```

---

## Task 5: `prosody.py` CLI orchestration

**Files:**
- Create: `target-vad/prosody.py`
- Create: `target-vad/tests/prosody/test_orchestration.py`

- [ ] **Step 1: Write failing orchestration tests**

Create `target-vad/tests/prosody/test_orchestration.py`:

```python
"""Tests for prosody.py CLI orchestration with a stubbed analyzer + tiny synthetic WAV."""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import soundfile as sf

import prosody


def _word(start: float, end: float, w: str = "x") -> dict:
    return {"start": start, "end": end, "word": w, "probability": 0.9}


@pytest.fixture
def tmp_workspace():
    """A tmp dir with a tiny WAV + post-2A diarization JSON pointing at it."""
    d = tempfile.mkdtemp()
    wav_path = os.path.join(d, "session.wav")
    json_path = os.path.join(d, "session.diarization.json")
    config_path = os.path.join(d, "config.yaml")

    # 5 seconds of silence — analyzer is stubbed, so audio content doesn't matter.
    sf.write(wav_path, np.zeros(int(16000 * 5.0), dtype=np.float32), 16000)

    with open(config_path, "w") as f:
        f.write(
            "prosody:\n"
            "  pitch_min_hz: 80\n"
            "  pitch_max_hz: 400\n"
            "  frame_length_ms: 25\n"
            "  hop_length_ms: 10\n"
        )

    data = {
        "audio_file": wav_path,
        "duration_s": 5.0,
        "diarized_at": "2026-05-16T00:00:00Z",
        "config": {},
        "enrolled_users_matched": [{"id": "alice", "name": "Alice"}, {"id": "bob", "name": "Bob"}],
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker_id": "alice", "speaker": "Alice",
             "text": "hello", "words": [_word(0, 0.5)]},
            {"start": 2.0, "end": 4.0, "speaker_id": "bob", "speaker": "Bob",
             "text": "hi", "words": [_word(2.0, 2.3)]},
            {"start": 4.0, "end": 5.0, "speaker_id": "alice", "speaker": "Alice",
             "text": "ok", "words": [_word(4.0, 4.2)]},
        ],
        "passes_run": ["diarization", "transcription"],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)

    yield {"dir": d, "json": json_path, "wav": wav_path, "config": config_path}
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stub_analyzer(monkeypatch):
    """Replace analyzer.analyze_segment with a deterministic stub.

    Returns prosody dicts that vary by index so we can verify per-segment write.
    """
    calls = []

    def fake_analyze(audio_chunk, sample_rate, words, segment_duration, cfg):
        calls.append({"len_audio": len(audio_chunk), "n_words": len(words), "duration": segment_duration})
        idx = len(calls)
        return {
            "pitch_hz_median": 140.0 + idx * 10.0,
            "pitch_hz_std": 5.0,
            "pitch_range_hz": 20.0,
            "energy_db_mean": -25.0 + idx,
            "energy_db_range": 8.0,
            "speech_rate_wps": 2.0,
            "pause_ratio": 0.2,
        }

    monkeypatch.setattr(prosody, "analyze_segment", fake_analyze)
    return calls


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class TestProsodyOrchestration:
    def test_happy_path_attaches_prosody_per_segment(self, tmp_workspace, stub_analyzer):
        rc = prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 0
        data = _read(tmp_workspace["json"])
        assert all("prosody" in s for s in data["segments"])
        assert data["segments"][0]["prosody"]["pitch_hz_median"] == 150.0  # idx 1
        assert "prosody_baselines" in data
        assert "prosody_config" in data
        assert "prosody" in data["passes_run"]
        # Stub was called once per segment.
        assert len(stub_analyzer) == 3

    def test_out_path_leaves_input_unchanged(self, tmp_workspace, stub_analyzer):
        original = _read(tmp_workspace["json"])
        out_json = os.path.join(tmp_workspace["dir"], "out.json")
        rc = prosody.main([tmp_workspace["json"], "--out", out_json, "--config", tmp_workspace["config"]])
        assert rc == 0
        assert _read(tmp_workspace["json"]) == original
        assert "prosody_baselines" in _read(out_json)

    def test_audio_override_used_when_json_path_wrong(self, tmp_workspace, stub_analyzer):
        # Mutate JSON to have a wrong audio_file path; pass --audio explicitly.
        data = _read(tmp_workspace["json"])
        data["audio_file"] = "/does/not/exist.wav"
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = prosody.main(
            [tmp_workspace["json"], "--audio", tmp_workspace["wav"], "--config", tmp_workspace["config"]]
        )
        assert rc == 0

    def test_idempotent_rerun_skips_already_analyzed(self, tmp_workspace, stub_analyzer):
        prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        # After first run, stub_analyzer has 3 calls.
        prior_calls = len(stub_analyzer)
        prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        # Second run skips all 3 because prosody is already populated.
        assert len(stub_analyzer) == prior_calls

    def test_rerun_flag_forces_full_reanalysis(self, tmp_workspace, stub_analyzer):
        prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        prior_calls = len(stub_analyzer)
        prosody.main([tmp_workspace["json"], "--rerun", "--config", tmp_workspace["config"]])
        # --rerun re-analyzes all 3 segments.
        assert len(stub_analyzer) == prior_calls + 3

    def test_missing_transcription_pass_exit_2(self, tmp_workspace, stub_analyzer):
        data = _read(tmp_workspace["json"])
        data["passes_run"] = ["diarization"]
        with open(tmp_workspace["json"], "w") as f:
            json.dump(data, f)
        rc = prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2

    def test_audio_file_missing_exit_2(self, tmp_workspace, stub_analyzer):
        os.remove(tmp_workspace["wav"])
        rc = prosody.main([tmp_workspace["json"], "--config", tmp_workspace["config"]])
        assert rc == 2
```

- [ ] **Step 2: Run, confirm import fails**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/prosody/test_orchestration.py -v
```

Expected: ImportError on `import prosody`.

- [ ] **Step 3: Implement `prosody.py`**

Create `target-vad/prosody.py`:

```python
"""Prosody pass (Phase 4) — see docs/superpowers/specs/2026-05-16-prosody-pass-design.md.

Reads a post-2A diarization JSON + the audio WAV, computes per-segment pitch /
energy / rate features via librosa, attaches a 7-field `prosody` block per
segment, and emits a top-level prosody_baselines summary keyed by speaker_id.
Idempotent — rerunning skips segments that already have prosody; --rerun
forces full re-analysis.
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
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

from core.audio.load import load_audio_as_mono16k
from modes.prosody.analyzer import analyze_segment
from modes.prosody.baselines import compute_baselines

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_CONFIG_OR_IO = 3

SR = 16000


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


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Target VAD - Prosody Pass (Phase 4)")
    parser.add_argument("input", help="Path to a transcribed diarization JSON")
    parser.add_argument("--audio", default=None, help="Path to the WAV (default: from JSON's audio_file field)")
    parser.add_argument("--out", default=None, help="Output JSON path (default: in-place atomic write)")
    parser.add_argument("--rerun", action="store_true", help="Re-analyze segments that already have prosody")
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
        console.print(
            "[red]This JSON has not been transcribed yet.[/] "
            "[dim]Run [bold]transcribe.py[/] first.[/]"
        )
        return EXIT_BAD_INPUT

    for i, seg in enumerate(data["segments"]):
        for k in ("text", "words"):
            if k not in seg:
                console.print(
                    f"[red]Segment {i} is missing field [bold]{k}[/].[/] "
                    "[dim]This JSON is in a partial/inconsistent state - rerun transcribe.py.[/]"
                )
                return EXIT_BAD_INPUT

    # Resolve audio path.
    audio_path = args.audio or data.get("audio_file", "")
    if not audio_path or not os.path.exists(audio_path):
        console.print(f"[red]Audio file not found:[/] {audio_path or '(no path given)'}")
        console.print("[dim]Pass [bold]--audio[/] explicitly if the JSON's audio_file is wrong.[/]")
        return EXIT_BAD_INPUT

    # Load config.
    try:
        cfg_full = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/] {args.config}")
        return EXIT_CONFIG_OR_IO
    cfg = cfg_full.get("prosody")
    if not cfg:
        console.print("[red]Config is missing the [bold]prosody:[/] block.[/]")
        return EXIT_CONFIG_OR_IO

    # Load audio once.
    try:
        audio = load_audio_as_mono16k(audio_path)
    except Exception as exc:
        console.print(f"[red]Failed to read audio file:[/] {exc}")
        return EXIT_BAD_INPUT

    # Walk segments.
    segments = data["segments"]
    analyzed = 0
    skipped = 0
    failed = 0
    with Progress(
        TextColumn("[bold]Analyzing[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("prosody", total=len(segments))
        for seg in segments:
            if not args.rerun and seg.get("prosody") is not None:
                skipped += 1
                progress.advance(task)
                continue

            start_i = max(0, int(seg["start"] * SR))
            end_i = min(len(audio), int(seg["end"] * SR))
            chunk = audio[start_i:end_i]
            segment_duration = max(0.0, seg["end"] - seg["start"])
            words = seg.get("words") or []
            try:
                block = analyze_segment(chunk, SR, words, segment_duration, cfg)
            except Exception as exc:
                console.print(f"[yellow]warning:[/] analyzer crashed on segment {seg['start']:.2f}s: {exc}")
                seg["prosody"] = None
                failed += 1
                progress.advance(task)
                continue

            # Sentinel: prosody: null when ALL seven fields are None
            if all(v is None for v in block.values()):
                seg["prosody"] = None
            else:
                seg["prosody"] = block
            analyzed += 1
            progress.advance(task)

    # Baselines + top-level fields.
    data["prosody_baselines"] = compute_baselines(segments)
    data["prosody_config"] = {
        "pitch_min_hz": int(cfg["pitch_min_hz"]),
        "pitch_max_hz": int(cfg["pitch_max_hz"]),
        "frame_length_ms": int(cfg["frame_length_ms"]),
        "hop_length_ms": int(cfg["hop_length_ms"]),
        "analyzed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if "passes_run" not in data:
        data["passes_run"] = []
    if "prosody" not in data["passes_run"]:
        data["passes_run"].append("prosody")

    out_path = args.out or args.input
    try:
        _atomic_write_json(out_path, data)
    except Exception as exc:
        console.print(f"[red]Failed to write prosody JSON:[/] {exc}")
        return EXIT_CONFIG_OR_IO

    console.print(
        f"[green]Prosody written:[/] {analyzed} analyzed, {skipped} skipped (already had prosody), "
        f"{failed} failed (analyzer crash)."
    )
    console.print(f"  JSON -> {out_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run orchestration tests**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/prosody/test_orchestration.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run full suite**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 -m pytest tests/ -q
```

Expected: 237 passed (218 baseline + 8 analyzer + 4 baselines + 7 orchestration).

- [ ] **Step 6: Commit**

```bash
git -C c:/repos/TVAD add target-vad/prosody.py target-vad/tests/prosody/test_orchestration.py
git -C c:/repos/TVAD commit -m "feat(prosody): add prosody.py CLI orchestration"
```

---

## Task 6: Manual smoke against Voice 001

**Files:** none changed (unless smoke surfaces fixes).

- [ ] **Step 1: Run `prosody.py` against the real Voice 001 fixture**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 prosody.py "../Voice 001 short.wav.diarization.json"
```

Expected:
- Exit 0
- Console progress bar `Analyzing` 9/9
- Success summary: `Prosody written: 9 analyzed, 0 skipped, 0 failed`
- The JSON gains `prosody` per segment + `prosody_baselines` + `prosody_config`; `passes_run` += `"prosody"`

First-run cold-start: librosa.pyin triggers numba JIT compile and may take 10-30s on the first segment. Subsequent segments are fast.

If the run fails with a librosa import error, install the missing dep:

```bash
py -3.14 -m pip install "librosa>=0.10.0"
```

- [ ] **Step 2: Eyeball the JSON additions**

```bash
cd c:/repos/TVAD && py -3.14 -c "
import json
d = json.load(open('Voice 001 short.wav.diarization.json', encoding='utf-8'))
print('passes_run:', d['passes_run'])
print('prosody_config:', d['prosody_config'])
print('prosody_baselines:')
import json as j
print(j.dumps(d['prosody_baselines'], indent=2))
print('first segment prosody:', d['segments'][0].get('prosody'))
"
```

Expected:
- `passes_run` includes `"prosody"`
- `prosody_config` has 5 fields (4 knobs + `analyzed_at`)
- `prosody_baselines` has 2 entries (for `session_speaker_a`, `session_speaker_b`), each with `pitch_hz_median`, `pitch_hz_iqr`, `energy_db_median`, `energy_db_iqr`, `segment_count`
- First segment's `prosody` block has 7 fields with plausible values (pitch ~80-300 Hz for human speech, energy_db negative)

If the values look implausible (e.g., pitch_hz_median = 0 or NaN-stringified), the analyzer's pyin call may be misconfigured — inspect and fix.

- [ ] **Step 3: Idempotent rerun check**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 prosody.py "../Voice 001 short.wav.diarization.json"
```

Expected:
- Exit 0
- Summary: `Prosody written: 0 analyzed, 9 skipped, 0 failed`
- `passes_run` still has `"prosody"` exactly once

Verify:

```bash
cd c:/repos/TVAD && py -3.14 -c "
import json
d = json.load(open('Voice 001 short.wav.diarization.json', encoding='utf-8'))
print('passes_run:', d['passes_run'])
assert d['passes_run'].count('prosody') == 1, 'prosody duplicated'
print('OK')
"
```

- [ ] **Step 4: `--rerun` check**

```bash
cd c:/repos/TVAD/target-vad && py -3.14 prosody.py "../Voice 001 short.wav.diarization.json" --rerun
```

Expected: summary `9 analyzed, 0 skipped, 0 failed`. The prosody values should be identical (deterministic) — pyin is deterministic given identical input.

- [ ] **Step 5: Update auto-memory**

Append a paragraph to `C:\Users\AI PC\.claude\projects\c--repos-TVAD\memory\project_tvad.md` documenting the prosody pass. Key facts:

- `prosody.py` exists as the fourth analysis pass
- Reads diarization JSON + WAV; emits `prosody` block per segment + top-level `prosody_baselines`
- Hard-requires transcription (`passes_run` ⊇ `{"transcription"}`)
- Idempotent rerun (skip segments with existing prosody); `--rerun` forces re-analysis
- New dep: librosa>=0.10.0
- Validated against `Voice 001 short.wav.diarization.json` on 2026-05-16
- 237 unit tests passing
- Phase 3 metrics doesn't consume prosody for v1 (deferred)
- Refactor: `load_audio_as_mono16k` extracted to `core/audio/load.py`, shared with diarize.py and transcribe.py

- [ ] **Step 6: Commit refreshed fixture**

The Voice 001 fixture is now enriched with prosody. Commit it as a regression-test snapshot:

```bash
git -C c:/repos/TVAD status
git -C c:/repos/TVAD add "Voice 001 short.wav.diarization.json"
git -C c:/repos/TVAD commit -m "chore(prosody): snapshot Voice 001 prosody-enriched fixture"
```

---

## Self-review checklist (after all tasks)

- [ ] Spec sections all implemented:
  - Pitch / energy / rate analyzer → Task 3
  - Per-speaker median + IQR baselines → Task 4
  - CLI with `--audio`, `--out`, `--rerun`, `--config` → Task 5
  - Atomic in-place write → Task 5
  - Pre-flight exit codes (2 for user-bad-input, 3 for env/IO) → Task 5
  - `prosody_config` block with 5 fields → Task 5
  - `passes_run` appends `"prosody"` deduped → Task 5
  - Audio loader shared between callers → Task 1
- [ ] Test count baseline (218) increases by approximately 19 (237 final)
- [ ] `prosody.py` follows the same atomic-write / exit-code pattern as `transcribe.py` and `sentiment.py`
- [ ] One new dependency added (`librosa>=0.10.0`); no other version changes
- [ ] All commits scoped — one logical change per commit
