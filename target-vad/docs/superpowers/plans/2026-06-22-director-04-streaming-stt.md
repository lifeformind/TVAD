# Director Plan 04 — Streaming STT Re-backing (SPIKE-GATED) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-back `modes/talkback/stt.py`'s `StreamingStt` off faster-whisper (which has **no aarch64 CUDA wheel** on this GB10) onto a backend that actually runs on GB10 CUDA, and extend its `transcribe_segment` to return `TranscriptResult(text, mean_word_prob)` so the Director can RESTORE on empty/low-confidence STT.

**Architecture:** The backend choice is **not yet decided** — the spec (Section 9, Section 14) flags STT as version-fragile and unproven-on-this-box. So **Task 1 is a real, runnable backend-selection spike** that, on this GB10, attempts to load + run (a) openai-whisper (torch CUDA) and (b) NeMo streaming ASR, measures per-call latency and per-word confidence availability on the real ~3s clip, and records the result in a notes file whose verdict picks the primary. **Tasks 2–6 are written against the spec's RECOMMENDED PRIMARY = openai-whisper (torch CUDA)** — the only CUDA STT *proven* on this box (whisper-tiny 38.8ms p95, `docs/notes/2026-06-21-gb10-contention.md`). The class name and async interface of `StreamingStt` are preserved (callers in Plan 02's SttWorker are unchanged); only the internals are swapped and the return type widened. Task 7 explicitly bounds what changes if the Task-1 spike selects NeMo instead.

**Tech Stack:** Python 3.12, `python3`; openai-whisper (torch-native, `device="cuda"`, `word_timestamps=True`); torch 2.12.0+cu130 (already installed, CUDA confirmed); numpy; pytest + pytest-asyncio; NeMo (probed only, install gated behind the spike). Target: NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128GB unified memory).

## Global Constraints

- **Platform:** NVIDIA GB10 / DGX Spark, aarch64, single GPU, ~128GB unified memory, Python 3.12.3 (`/usr/bin/python3`). Use `python3`.
- **faster-whisper is BANNED as the runtime backend:** no aarch64 CUDA wheel exists; CPU fallback is ~270ms p50. The class `StreamingStt` keeps its name/interface but its faster-whisper internals are removed (spec Section 9, Section 14; reuse-map "re-backed (NOT as-is)").
- **Binding contract (do not change):** the SttWorker (Plan 02) calls `await StreamingStt.transcribe_segment(audio) -> TranscriptResult(text: str, mean_word_prob: float)`. The method stays `async`, takes a single `np.ndarray` (float32, 16kHz, mono), and returns that exact dataclass. `mean_word_prob` is a float in `[0.0, 1.0]`.
- **English-only:** keep `language="en"` (spec Section 2, matches `_transcribe_sync` today).
- **`mean_word_prob` semantics:** the mean of per-word `probability` over all words in the transcript; `0.0` when the transcript is empty (no words). The Director maps empty text → BACKCHANNEL and RESTOREs when `mean_word_prob < conf_floor` (`DirectorConfig.conf_floor = 0.5`, Plan 01 `modes/director/config.py:142`). This guard lives in the reducer (Plan 01), not in `StreamingStt`; this plan only DELIVERS the confidence number.
- **CI guard:** every test that loads a real model or touches CUDA MUST be guarded to **skip** (not fail) when CUDA or the model is unavailable, so CI on x86/CPU-only boxes stays green. Pure-logic tests (averaging, empty handling) use a fake model object and always run.
- **Spike-gate honesty:** Task 1 is a **decision gate**, not fictional integration code. Its acceptance is a written verdict in `docs/notes/2026-06-22-stt-backend.md`. The "if NeMo wins" branch is documented in Task 7 — the backend is gated, not pretended-decided.

---

## File Structure

| File | Responsibility |
|---|---|
| `bench/stt_backend_probe.py` | **Task 1 spike (create).** Standalone runnable probe: tries to load + run openai-whisper (`base.en`, `tiny`) and NeMo (`parakeet`/`nemotron-speech-streaming`) on `self.wav`, measuring per-call latency and whether each emits per-word confidence. Self-bootstraps `sys.path` like `bench/reflex_contention.py`. Prints a table; the human records the verdict. |
| `docs/notes/2026-06-22-stt-backend.md` | **Task 1 acceptance (create).** Records which backends loaded, their latency, confidence availability, and the RESULT that picks the primary. |
| `modes/talkback/stt.py` | **Re-backed (modify).** Remove faster-whisper internals; add the `TranscriptResult` dataclass; swap to openai-whisper (`whisper.load_model`, `device="cuda"`, `word_timestamps=True`); compute `mean_word_prob` from per-word `probability`; keep class name + async `transcribe_segment` signature, widened return type. |
| `tests/kiosk/talkback/test_stt.py` | **Rewritten (modify).** Pure-logic tests (fake model) for the `TranscriptResult` return, `mean_word_prob` averaging, empty handling — always run. |
| `tests/kiosk/talkback/test_stt_cuda.py` | **New CUDA integration test (create).** Loads real `tiny` on CUDA, transcribes `self.wav`, asserts `text` non-empty and `mean_word_prob` in `[0,1]`. Skips if CUDA/model unavailable. |
| `config.yaml` | **Modify** `kiosk.talkback.stt` (lines 59–64): set `backend`, `model`, `device` for the openai-whisper re-backing; document the `conf_floor` linkage. |

**Test fixture:** `self.wav` (repo root, `/home/ldrgx10/FullDuplexVoice/TVAD/target-vad/self.wav`) — real speech, 16kHz mono, 18.4s. Tasks slice the **first ~3s** (`48000` samples) for a realistic short-segment clip.

---

## Task 1: BACKEND-SELECTION SPIKE (decision gate)

This is the gate the whole plan hangs on. The spec says the STT backend is unproven on this box (Section 14, highest risk). **Task 1 produces a real, runnable probe and a written verdict** — it does NOT integrate anything. The probe attempts each candidate, measures latency + confidence availability on the real clip, and the human records which backend the rest of the plan should use. The plan body (Tasks 2–6) is pre-written against the spec's recommended primary (openai-whisper); Task 7 documents the bounded swap if NeMo wins.

**Per spec Section 9 install notes (do NOT violate):** NeMo installs from source on the PyTorch **25.10 (2.9)** container — *not* 2.10/25.12 (breaks NeMo/Lhotse); pin `lhotse>=1.32.2`; **NIM is x86-only — do NOT use it.** The probe does **not** install NeMo destructively into the running environment; it only *attempts* `import nemo` and reports `not installed` if absent (so the box's working torch 2.12 stack is never broken by this probe). Installing NeMo is a deliberate, separate, human-gated step recorded in the notes file if the verdict chooses NeMo.

**Files:**
- Create: `bench/stt_backend_probe.py`
- Create: `docs/notes/2026-06-22-stt-backend.md`

**Interfaces:**
- Consumes: `self.wav` (repo-root fixture); torch 2.12.0+cu130 (installed); optionally `whisper` (openai-whisper) and `nemo` (probed, not required).
- Produces: a printed results table and the written verdict in `docs/notes/2026-06-22-stt-backend.md`. The verdict's "PRIMARY" line is what Tasks 2–7 assume.

- [ ] **Step 1: Write the probe script**

```python
# bench/stt_backend_probe.py
"""bench/stt_backend_probe.py — STT backend-selection spike for GB10 (DGX Spark).

DECISION GATE for Director Plan 04. Attempts to load + run each candidate STT
backend on THIS box's real ~3s speech clip, measuring:
  * per-call latency (p50/p95 over N runs, warmups discarded)
  * whether the backend emits PER-WORD CONFIDENCE (needed for mean_word_prob)

Candidates
----------
  A. openai-whisper (torch CUDA) — base.en and tiny. The ONLY CUDA STT proven
     on this box (gb10-contention.md: whisper-tiny 38.8ms p95). word_timestamps
     gives per-word .probability.
  B. NeMo streaming (parakeet / nemotron-speech-streaming) — spec's PRIMARY if
     it loads. PROBED ONLY: we `import nemo` and report "not installed" if
     absent; we do NOT pip-install it here (installing NeMo is a separate,
     human-gated step per spec Section 9 — PyTorch 2.9/container 25.10,
     lhotse>=1.32.2, NIM is x86-only/forbidden).

This script INSTALLS NOTHING and MUTATES NOTHING. It only loads what is already
present and prints a table. The human records the verdict in
docs/notes/2026-06-22-stt-backend.md and that verdict picks the Plan-04 primary.

Usage
-----
    cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
    python3 bench/stt_backend_probe.py

Options (env-vars)
------------------
    STT_PROBE_N      timed iterations per cell (default 8; first 2 are warmup)
    STT_PROBE_WAV    path to the test clip (default: ./self.wav)
    STT_PROBE_SECS   seconds of the clip to use (default: 3.0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import traceback

import numpy as np

N_TOTAL = int(os.environ.get("STT_PROBE_N", 8))
N_WARMUP = 2
WAV = os.environ.get("STT_PROBE_WAV", "self.wav")
SECS = float(os.environ.get("STT_PROBE_SECS", 3.0))
SR = 16_000


def p50_p95(times):
    if not times:
        return float("nan"), float("nan")
    arr = np.array(times, dtype=float)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def load_clip():
    """Load the first SECS seconds of WAV as float32 mono @ 16kHz."""
    import soundfile as sf

    data, sr = sf.read(WAV, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f"{WAV} is {sr}Hz, expected {SR}Hz")
    n = int(SECS * SR)
    return data[:n]


def probe_openai_whisper(model_name, clip):
    """Returns dict with latency + confidence-availability, or {'error': ...}."""
    try:
        import torch
        import whisper
    except Exception as e:  # noqa: BLE001
        return {"error": f"import failed: {e}"}

    if not torch.cuda.is_available():
        return {"error": "torch.cuda not available"}

    try:
        model = whisper.load_model(model_name, device="cuda")
    except Exception as e:  # noqa: BLE001
        return {"error": f"load_model({model_name}) failed: {e}"}

    times = []
    text = ""
    has_word_conf = False
    sample_probs = []
    for i in range(N_TOTAL):
        t0 = time.perf_counter()
        result = model.transcribe(
            clip, language="en", word_timestamps=True, fp16=True
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= N_WARMUP:
            times.append(dt)
        text = result.get("text", "").strip()
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                if "probability" in w:
                    has_word_conf = True
                    if len(sample_probs) < 8:
                        sample_probs.append(round(float(w["probability"]), 3))
    p50, p95 = p50_p95(times)
    return {
        "loaded": True,
        "p50_ms": p50,
        "p95_ms": p95,
        "text": text[:80],
        "has_word_conf": has_word_conf,
        "sample_probs": sample_probs,
    }


def probe_nemo(model_name, clip):
    """Probe-only: import nemo if present; do NOT install. Report availability."""
    try:
        import nemo  # noqa: F401
        import nemo.collections.asr as nemo_asr
    except Exception as e:  # noqa: BLE001
        return {"error": f"nemo not installed / import failed: {e}"}

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        return {"error": f"from_pretrained({model_name}) failed: {e}"}

    # Write the clip to a temp wav (NeMo transcribe takes file paths).
    import tempfile

    import soundfile as sf

    times = []
    text = ""
    has_word_conf = False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        sf.write(tf.name, clip, SR)
        for i in range(N_TOTAL):
            t0 = time.perf_counter()
            out = model.transcribe([tf.name])
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= N_WARMUP:
                times.append(dt)
            # NeMo returns Hypothesis objects (or strings on older versions).
            hyp = out[0] if out else None
            if hasattr(hyp, "text"):
                text = (hyp.text or "").strip()
                # Per-word confidence lives on hyp.word_confidence when emitted.
                wc = getattr(hyp, "word_confidence", None)
                has_word_conf = bool(wc)
            elif isinstance(hyp, str):
                text = hyp.strip()
    p50, p95 = p50_p95(times)
    return {
        "loaded": True,
        "p50_ms": p50,
        "p95_ms": p95,
        "text": text[:80],
        "has_word_conf": has_word_conf,
    }


def fmt_row(name, r):
    if "error" in r:
        return f"  {name:<34} FAILED: {r['error'][:60]}"
    conf = "YES" if r["has_word_conf"] else "NO"
    return (
        f"  {name:<34} p50={r['p50_ms']:6.1f}ms  p95={r['p95_ms']:6.1f}ms  "
        f"word_conf={conf:<3}  text={r['text']!r}"
    )


def main():
    print("=" * 78)
    print("  STT BACKEND-SELECTION SPIKE — GB10 (DGX Spark)  [Director Plan 04]")
    print("=" * 78)
    try:
        import torch

        print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f"  device={torch.cuda.get_device_name(0)}")
        else:
            print()
    except Exception as e:  # noqa: BLE001
        print(f"  torch import failed: {e}")

    print(f"  clip={WAV}  secs={SECS}  N={N_TOTAL} (warmup {N_WARMUP})")
    print("-" * 78)

    try:
        clip = load_clip()
    except Exception as e:  # noqa: BLE001
        print(f"  CANNOT LOAD CLIP: {e}")
        traceback.print_exc()
        sys.exit(1)

    results = {}
    for name in ("base.en", "tiny"):
        print(f"  probing openai-whisper {name} ...", flush=True)
        results[f"openai-whisper/{name}"] = probe_openai_whisper(name, clip)
    for name in ("nvidia/parakeet-tdnn-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v2"):
        print(f"  probing NeMo {name} ...", flush=True)
        results[f"nemo/{name}"] = probe_nemo(name, clip)

    print()
    print("  RESULTS")
    print("  " + "-" * 74)
    for name, r in results.items():
        print(fmt_row(name, r))

    print()
    print("  VERDICT GUIDANCE")
    print("  " + "-" * 74)
    ow = [k for k, r in results.items()
          if k.startswith("openai-whisper") and r.get("loaded") and r.get("has_word_conf")]
    nemo = [k for k, r in results.items()
            if k.startswith("nemo") and r.get("loaded") and r.get("has_word_conf")]
    if nemo:
        print("  NeMo loaded WITH per-word confidence — candidate PRIMARY (spec preference).")
        print("  -> If its p95 is acceptable, record NeMo as PRIMARY; see Plan 04 Task 7.")
    if ow:
        print("  openai-whisper loaded WITH per-word confidence — PROVEN fallback / primary.")
        print("  -> Default per spec: openai-whisper is the PRIMARY unless NeMo clearly wins.")
    if not ow and not nemo:
        print("  NO backend loaded with per-word confidence. STOP — escalate before Task 2.")
    print()
    print("  ACTION: record the chosen PRIMARY + these numbers in")
    print("          docs/notes/2026-06-22-stt-backend.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe on GB10**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 bench/stt_backend_probe.py
```

Expected output (shape — exact ms vary; openai-whisper is the proven path, NeMo likely reports `not installed`):
```
==============================================================================
  STT BACKEND-SELECTION SPIKE — GB10 (DGX Spark)  [Director Plan 04]
==============================================================================
  torch 2.12.0+cu130  cuda=True  device=NVIDIA GB10
  clip=self.wav  secs=3.0  N=8 (warmup 2)
------------------------------------------------------------------------------
  probing openai-whisper base.en ...
  probing openai-whisper tiny ...
  probing NeMo nvidia/parakeet-tdnn-0.6b-v2 ...
  probing NeMo nvidia/parakeet-tdt-0.6b-v2 ...

  RESULTS
  --------------------------------------------------------------------------
  openai-whisper/base.en             p50=  ... ms  p95=  ... ms  word_conf=YES  text='...'
  openai-whisper/tiny                p50=  ... ms  p95=  ... ms  word_conf=YES  text='...'
  nemo/nvidia/parakeet-tdnn-0.6b-v2  FAILED: nemo not installed / import failed: ...
  nemo/nvidia/parakeet-tdt-0.6b-v2   FAILED: nemo not installed / import failed: ...

  VERDICT GUIDANCE
  --------------------------------------------------------------------------
  openai-whisper loaded WITH per-word confidence — PROVEN fallback / primary.
  -> Default per spec: openai-whisper is the PRIMARY unless NeMo clearly wins.
  ...
```

If openai-whisper is not importable, install it (it is pure-Python over the already-installed torch):
```bash
python3 -m pip install -U openai-whisper
```
(NeMo is **not** installed by this step — installing NeMo is a separate human-gated decision per spec Section 9, recorded in the notes file only if the verdict chooses it.)

- [ ] **Step 3: Record the verdict in the notes file**

Create `docs/notes/2026-06-22-stt-backend.md` with the actual measured numbers from Step 2:

```markdown
# STT Backend Selection — GB10 (DGX Spark)

**Date:** 2026-06-22
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128GB unified)
**Spike:** `bench/stt_backend_probe.py` (Director Plan 04, Task 1)
**Clip:** `self.wav` first 3.0s (real speech, 16kHz mono)

## Candidates probed

| Backend | Loaded? | p50 (ms) | p95 (ms) | Per-word confidence | Notes |
|---|---|---|---|---|---|
| openai-whisper `base.en` (CUDA) | <fill> | <fill> | <fill> | <fill> | torch-native; word_timestamps→.probability |
| openai-whisper `tiny` (CUDA)    | <fill> | <fill> | <fill> | <fill> | fastest; proven 38.8ms p95 in gb10-contention |
| NeMo `parakeet-*` (CUDA)        | <fill> | <fill> | <fill> | <fill> | NOT installed by probe; import-only; see spec §9 |

## Decision

**PRIMARY = <openai-whisper | NeMo> (model: <name>, device: cuda).**

Rationale: <one or two lines — confidence availability + latency + load success.>

## Consequences for Plan 04

- If PRIMARY = openai-whisper: Tasks 2–6 ship as written.
- If PRIMARY = NeMo: apply the bounded swap in Task 7 (different `_ensure_model`
  and `_transcribe_sync` bodies; `TranscriptResult` contract and tests unchanged).
- faster-whisper is removed regardless (no aarch64 CUDA wheel).
```

Fill every `<fill>` from the Step-2 output. The PRIMARY line is binding for Tasks 2–7.

- [ ] **Step 4: Commit the spike + verdict**

```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
git add bench/stt_backend_probe.py docs/notes/2026-06-22-stt-backend.md
git commit -m "spike(stt): GB10 backend-selection probe + verdict (openai-whisper vs NeMo)"
```

**GATE:** Do not start Task 2 until `docs/notes/2026-06-22-stt-backend.md` records a PRIMARY with per-word confidence available. If neither backend loads with per-word confidence, STOP and escalate — the empty/low-confidence RESTORE guard (spec Section 6) cannot ship without it.

---

## Task 2: `TranscriptResult` dataclass + failing pure-logic test

Re-export the CANONICAL `TranscriptResult` (owned by Plan 02 in `modes/director/transcript.py`) from `stt.py` and lock its shape with a model-free test. There is exactly ONE `TranscriptResult` class across the codebase — defining a second one in `stt.py` would break Plan 02's `wrap_transcript()` `isinstance` check. Pre-written against PRIMARY = openai-whisper.

**Dependency:** requires Plan 02's `modes/director/transcript.py` (the `TranscriptResult` dataclass). Plans run in order (02 before 04), so it exists.

**Files:**
- Modify: `modes/talkback/stt.py` (re-export the canonical type at top)
- Modify: `tests/kiosk/talkback/test_stt.py` (replace the faster-whisper-shaped tests)

**Interfaces:**
- Consumes: `TranscriptResult(text, mean_word_prob)` from `modes/director/transcript.py` (Plan 02).
- Produces: `modes.talkback.stt.TranscriptResult` re-exported as the SAME object as `modes.director.transcript.TranscriptResult` — the exact type the SttWorker (Plan 02) consumes and Tasks 3–7 return.

- [ ] **Step 1: Write the failing test**

Replace the entire contents of `tests/kiosk/talkback/test_stt.py` with:

```python
"""Tests for the re-backed StreamingStt (openai-whisper / torch CUDA).

Pure-logic tests use a fake model object and always run. The real-CUDA
integration test lives in test_stt_cuda.py and is skipped without a GPU.
"""

import numpy as np
import pytest

from modes.talkback.stt import StreamingStt, TranscriptResult


def test_transcript_result_shape():
    r = TranscriptResult(text="hello world", mean_word_prob=0.87)
    assert r.text == "hello world"
    assert r.mean_word_prob == 0.87


def test_transcript_result_is_frozen():
    r = TranscriptResult(text="x", mean_word_prob=0.5)
    with pytest.raises(Exception):
        r.text = "y"  # frozen dataclass -> FrozenInstanceError


def test_transcript_result_is_the_canonical_director_type():
    # Single source of truth: stt.py RE-EXPORTS Plan 02's type, not a copy,
    # so Plan 02's wrap_transcript() isinstance check stays valid.
    from modes.director.transcript import TranscriptResult as Canonical
    assert TranscriptResult is Canonical
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py::test_transcript_result_shape -v
```
Expected: FAIL with `ImportError: cannot import name 'TranscriptResult' from 'modes.talkback.stt'`.

- [ ] **Step 3: Re-export the canonical TranscriptResult**

Edit `modes/talkback/stt.py`. Replace the module docstring and the imports block (lines 1–9) with the following — note we IMPORT the canonical `TranscriptResult` from Plan 02's `modes/director/transcript.py` rather than redefining it (single source of truth; `mean_word_prob` semantics — mean per-word probability in [0,1], 0.0 when empty; Director RESTOREs below `DirectorConfig.conf_floor`, default 0.5 — are documented there):

```python
"""Streaming STT wrapper, re-backed onto openai-whisper (torch, CUDA).

faster-whisper / CTranslate2 has NO aarch64 CUDA wheel on this GB10 (DGX Spark)
and falls back to ~270ms CPU. This module keeps the StreamingStt class name and
async transcribe_segment interface (callers unchanged) but swaps the internals to
openai-whisper, and returns the canonical TranscriptResult(text, mean_word_prob)
(owned by modes/director/transcript.py, Plan 02) so the Director can RESTORE on
empty / low-confidence transcripts (spec Section 6).

See docs/notes/2026-06-22-stt-backend.md for the backend-selection verdict.
"""

import asyncio

import numpy as np

from modes.director.transcript import TranscriptResult  # canonical type (Plan 02)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py -v
```
Expected: PASS (both `test_transcript_result_shape` and `test_transcript_result_is_frozen`).

- [ ] **Step 5: Commit**

```bash
git add modes/talkback/stt.py tests/kiosk/talkback/test_stt.py
git commit -m "feat(stt): add TranscriptResult dataclass for confidence-aware STT"
```

---

## Task 3: Re-back `StreamingStt` internals onto openai-whisper + `mean_word_prob` averaging

Swap faster-whisper for openai-whisper, keep the class name and async `transcribe_segment`, widen the return to `TranscriptResult`, and compute `mean_word_prob` from per-word `probability`. The averaging logic is tested with a **fake model** so it runs everywhere; CUDA loading is tested separately in Task 4.

**Files:**
- Modify: `modes/talkback/stt.py` (class body)
- Modify: `tests/kiosk/talkback/test_stt.py` (add fake-model averaging tests)

**Interfaces:**
- Consumes: `TranscriptResult` (Task 2).
- Produces:
  - `StreamingStt(model: str = "base.en", device: str = "cuda")` — constructor (note: openai-whisper has **no** `compute_type`; fp16 is implied on CUDA).
  - `async transcribe_segment(audio: np.ndarray) -> TranscriptResult` — the binding contract.
  - `_transcribe_sync(audio) -> TranscriptResult` — the executor body (testable with a fake `self._model`).
  - `_mean_word_prob(result_dict) -> float` — static helper averaging per-word `probability` from a whisper result dict; `0.0` if no words.

- [ ] **Step 1: Write the failing tests (fake model, always run)**

Append to `tests/kiosk/talkback/test_stt.py`:

```python
class _FakeWhisperModel:
    """Mimics whisper.load_model(...).transcribe() output shape."""

    def __init__(self, result):
        self._result = result

    def transcribe(self, audio, **kwargs):
        return self._result


def _make_stt_with_model(result):
    stt = StreamingStt.__new__(StreamingStt)
    stt._model = _FakeWhisperModel(result)
    return stt


def test_mean_word_prob_averages_word_probabilities():
    result = {
        "text": " hello world ",
        "segments": [
            {"words": [
                {"word": "hello", "probability": 0.9},
                {"word": "world", "probability": 0.7},
            ]}
        ],
    }
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert isinstance(out, TranscriptResult)
    assert out.text == "hello world"
    assert out.mean_word_prob == pytest.approx(0.8)


def test_mean_word_prob_empty_text_is_zero():
    result = {"text": "  ", "segments": []}
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert out.text == ""
    assert out.mean_word_prob == 0.0


def test_mean_word_prob_multi_segment_concatenates_and_averages():
    result = {
        "text": " first second third ",
        "segments": [
            {"words": [
                {"word": "first", "probability": 1.0},
                {"word": "second", "probability": 0.5},
            ]},
            {"words": [
                {"word": "third", "probability": 0.6},
            ]},
        ],
    }
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert out.text == "first second third"
    assert out.mean_word_prob == pytest.approx((1.0 + 0.5 + 0.6) / 3.0)


def test_mean_word_prob_clamped_to_unit_interval():
    # whisper probabilities are already in [0,1]; defend against odd inputs.
    result = {
        "text": "x",
        "segments": [{"words": [{"word": "x", "probability": 1.4}]}],
    }
    stt = _make_stt_with_model(result)
    out = stt._transcribe_sync(np.zeros(48000, dtype=np.float32))
    assert 0.0 <= out.mean_word_prob <= 1.0


@pytest.mark.asyncio
async def test_transcribe_segment_returns_transcript_result():
    result = {
        "text": " hi ",
        "segments": [{"words": [{"word": "hi", "probability": 0.95}]}],
    }
    stt = _make_stt_with_model(result)
    out = await stt.transcribe_segment(np.zeros(48000, dtype=np.float32))
    assert isinstance(out, TranscriptResult)
    assert out.text == "hi"
    assert out.mean_word_prob == pytest.approx(0.95)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py -v
```
Expected: FAIL — `_transcribe_sync` now returns a `str` (or the old faster-whisper body errors); `AttributeError`/`AssertionError` on `out.text`, and `transcribe_segment` returns `str` not `TranscriptResult`.

- [ ] **Step 3: Replace the class body (minimal openai-whisper implementation)**

In `modes/talkback/stt.py`, replace the entire `class StreamingStt` body (the old faster-whisper `__init__`, `_ensure_model`, `transcribe_segment`, `_transcribe_sync`) with:

```python
class StreamingStt:
    """Segment-level STT over openai-whisper (torch, CUDA on GB10).

    Keeps the original async transcribe_segment interface so Plan 02's SttWorker
    is unchanged, but returns TranscriptResult(text, mean_word_prob) instead of a
    bare str. faster-whisper is gone (no aarch64 CUDA wheel on this box).
    """

    def __init__(
        self,
        model: str = "base.en",
        device: str = "cuda",
    ):
        self._model_name = model
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import whisper  # openai-whisper (torch-native), NOT faster_whisper

        self._model = whisper.load_model(self._model_name, device=self._device)

    @staticmethod
    def _mean_word_prob(result: dict) -> float:
        """Mean per-word probability over the transcript; 0.0 if no words."""
        probs = []
        for seg in result.get("segments", []) or []:
            for w in seg.get("words", []) or []:
                p = w.get("probability")
                if p is not None:
                    probs.append(float(p))
        if not probs:
            return 0.0
        mean = sum(probs) / len(probs)
        return max(0.0, min(1.0, mean))

    async def transcribe_segment(self, audio: np.ndarray) -> "TranscriptResult":
        self._ensure_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> "TranscriptResult":
        result = self._model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            fp16=(self._device == "cuda"),
        )
        text = (result.get("text", "") or "").strip()
        mean_word_prob = self._mean_word_prob(result)
        return TranscriptResult(text=text, mean_word_prob=mean_word_prob)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py -v
```
Expected: PASS (all pure-logic + fake-model tests).

- [ ] **Step 5: Confirm no faster-whisper references remain**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
grep -n "faster_whisper\|compute_type\|large-v3\|WhisperModel" modes/talkback/stt.py
```
Expected: **no output** (all faster-whisper internals removed). If anything prints, delete it.

- [ ] **Step 6: Commit**

```bash
git add modes/talkback/stt.py tests/kiosk/talkback/test_stt.py
git commit -m "feat(stt): re-back StreamingStt onto openai-whisper; return TranscriptResult"
```

---

## Task 4: Real-CUDA integration test on `self.wav` (GB10-gated)

Prove the re-backed `StreamingStt` actually loads on CUDA and produces a non-empty transcript with a valid `mean_word_prob` on the real clip. Skips cleanly off-GPU so CI stays green.

**Files:**
- Create: `tests/kiosk/talkback/test_stt_cuda.py`

**Interfaces:**
- Consumes: `StreamingStt`, `TranscriptResult` (Task 3); `self.wav` fixture.
- Produces: a GB10-gated regression that the binding contract holds end-to-end on real hardware.

- [ ] **Step 1: Write the CUDA integration test (skip-guarded)**

```python
# tests/kiosk/talkback/test_stt_cuda.py
"""Real-CUDA integration test for the re-backed StreamingStt.

SKIPS (does not fail) when torch/CUDA or openai-whisper is unavailable, so CI on
CPU-only / x86 boxes stays green. Run on GB10 to verify the binding contract:
transcribe_segment(audio) -> TranscriptResult(text, mean_word_prob in [0,1]).
"""

import os

import numpy as np
import pytest

from modes.talkback.stt import StreamingStt, TranscriptResult

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_WAV = os.path.join(_REPO_ROOT, "self.wav")


def _cuda_and_whisper_available():
    try:
        import torch
        import whisper  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _cuda_and_whisper_available(),
    reason="CUDA + openai-whisper required (GB10 only)",
)


def _load_clip(secs=3.0, sr=16000):
    import soundfile as sf

    data, file_sr = sf.read(_WAV, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert file_sr == sr, f"{_WAV} is {file_sr}Hz"
    return data[: int(secs * sr)]


@requires_gpu
@pytest.mark.asyncio
async def test_transcribe_real_clip_returns_valid_result():
    if not os.path.exists(_WAV):
        pytest.skip(f"fixture {_WAV} missing")
    stt = StreamingStt(model="tiny", device="cuda")
    clip = _load_clip(secs=3.0)
    out = await stt.transcribe_segment(clip)
    assert isinstance(out, TranscriptResult)
    assert isinstance(out.text, str)
    assert out.text != ""  # 3s of real speech -> non-empty transcript
    assert 0.0 <= out.mean_word_prob <= 1.0
    assert out.mean_word_prob > 0.0  # real speech -> some word confidence


@requires_gpu
@pytest.mark.asyncio
async def test_transcribe_silence_low_or_empty():
    stt = StreamingStt(model="tiny", device="cuda")
    silence = np.zeros(48000, dtype=np.float32)
    out = await stt.transcribe_segment(silence)
    assert isinstance(out, TranscriptResult)
    # Silence either transcribes to empty (mean_word_prob == 0.0) or to
    # low-confidence garbage; either way the Director's empty/low-conf guard
    # (conf_floor=0.5) must be able to RESTORE on it.
    assert out.text == "" or out.mean_word_prob < 0.5
```

- [ ] **Step 2: Run on GB10 to verify it passes**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt_cuda.py -v
```
Expected on GB10: 2 PASS (model loads on CUDA, real clip transcribes non-empty with `mean_word_prob` in `(0,1]`).

- [ ] **Step 3: Verify the skip path off-GPU**

Run (simulating CI by forcing CUDA-unavailable via a guaranteed-absent device is not portable; instead confirm the skip logic compiles and the file collects):
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt_cuda.py --collect-only -q
```
Expected: both tests collected; on a box without CUDA/whisper they report `SKIPPED` instead of failing.

- [ ] **Step 4: Commit**

```bash
git add tests/kiosk/talkback/test_stt_cuda.py
git commit -m "test(stt): GB10-gated CUDA integration test on self.wav (skips off-GPU)"
```

---

## Task 5: Update `config.yaml` `kiosk.talkback.stt` for the re-backing

Point the config at the openai-whisper backend and document the `conf_floor` linkage. The faster-whisper-only keys (`compute_type`, `device: cpu`) are corrected.

**Files:**
- Modify: `config.yaml` lines 59–64 (`kiosk.talkback.stt`)

**Interfaces:**
- Consumes: the re-backed `StreamingStt(model, device)` constructor (Task 3).
- Produces: config keys `kiosk.talkback.stt.backend`, `.model`, `.device` that the SttWorker (Plan 02) reads when constructing `StreamingStt`.

- [ ] **Step 1: Read the current block**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
sed -n '59,64p' config.yaml
```
Expected:
```
    stt:
      model: "base"
      compute_type: "int8"
      device: "cpu"
      partials_every_ms: 300
      end_of_utterance_tail_ms: 400
```

- [ ] **Step 2: Replace the block**

Replace lines 59–64 of `config.yaml` with:

```yaml
    stt:
      # Re-backed off faster-whisper (no aarch64 CUDA wheel on GB10) onto
      # openai-whisper (torch, CUDA). See docs/notes/2026-06-22-stt-backend.md.
      backend: "openai-whisper"   # "openai-whisper" (proven on GB10) | "nemo" (Plan 04 Task 7)
      model: "base.en"            # "tiny" for the lowest-latency reflex; "base.en" default
      device: "cuda"              # GB10 CUDA; falls back is NOT supported (faster-whisper banned)
      partials_every_ms: 300      # reserved (no streaming-partial path in V1; full-segment STT)
      end_of_utterance_tail_ms: 400
      # mean_word_prob from StreamingStt feeds the Director's empty/low-confidence
      # RESTORE guard: transcripts with mean_word_prob < barge_in.conf_floor (0.5)
      # are treated as non-content and RESTORE rather than CUT (spec Section 6).
```

- [ ] **Step 3: Verify the YAML parses**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); s=c['kiosk']['talkback']['stt']; print(s['backend'], s['model'], s['device'])"
```
Expected:
```
openai-whisper base.en cuda
```

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "config(stt): point talkback STT at openai-whisper/CUDA (faster-whisper removed)"
```

---

## Task 6: Full re-backed test suite + faster-whisper-removal regression

Run the whole talkback STT suite and assert the codebase no longer references the banned backend anywhere on the STT path.

**Files:**
- Modify: `tests/kiosk/talkback/test_stt.py` (add the no-faster-whisper guard test)

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: a regression that fails if anyone re-introduces faster-whisper into `stt.py`.

- [ ] **Step 1: Add the regression test**

Append to `tests/kiosk/talkback/test_stt.py`:

```python
def test_stt_module_does_not_import_faster_whisper():
    """faster-whisper has no aarch64 CUDA wheel; it must stay out of stt.py."""
    import inspect

    import modes.talkback.stt as stt_mod

    src = inspect.getsource(stt_mod)
    assert "faster_whisper" not in src
    assert "WhisperModel" not in src  # the faster-whisper class name
    assert "import whisper" in src    # openai-whisper is the backend
```

- [ ] **Step 2: Run the regression**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py::test_stt_module_does_not_import_faster_whisper -v
```
Expected: PASS.

- [ ] **Step 3: Run the full talkback STT suite (pure-logic + integration collect)**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py tests/kiosk/talkback/test_stt_cuda.py -v
```
Expected: all `test_stt.py` tests PASS; `test_stt_cuda.py` tests PASS on GB10 (or SKIP off-GPU). No failures.

- [ ] **Step 4: Check `test_integration_stt.py` still collects (it may reference the old `str` return)**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_integration_stt.py --collect-only -q 2>&1 | tail -20
```
If it asserts on a bare-`str` return from `transcribe_segment`, update those assertions to read `.text` / `.mean_word_prob` off the `TranscriptResult` (the contract changed by design). If it only checks loading, leave it. Re-run the file:
```bash
python3 -m pytest tests/kiosk/talkback/test_integration_stt.py -v
```
Expected: PASS or SKIP (GPU-gated), no failures.

- [ ] **Step 5: Commit**

```bash
git add tests/kiosk/talkback/test_stt.py tests/kiosk/talkback/test_integration_stt.py
git commit -m "test(stt): guard against faster-whisper re-introduction; align integration test"
```

---

## Task 7: (CONDITIONAL) NeMo swap — apply ONLY if Task-1 verdict picks NeMo

**Do nothing in this task if `docs/notes/2026-06-22-stt-backend.md` records PRIMARY = openai-whisper.** This task exists to keep the plan honest about the spike gate: the backend is *gated*, not pretended-decided. If — and only if — the Task-1 probe shows NeMo loads on this box **with per-word confidence** and a better/acceptable p95, swap the backend. The swap is **clearly bounded**: it touches only `_ensure_model` and `_transcribe_sync` inside `StreamingStt`; the `TranscriptResult` contract, the async `transcribe_segment` signature, the SttWorker (Plan 02), the empty/low-conf guard (Plan 01), and every test in Task 2 stay **unchanged** (they assert on the contract, not the engine).

**Per spec Section 9 (binding if this branch runs):** install NeMo **from source on the PyTorch 25.10 / 2.9 container — NOT 2.10/25.12** (breaks NeMo/Lhotse); pin `lhotse>=1.32.2`; **NIM is x86-only — do NOT use it.** A `word_confidence` length-mismatch bug exists in some parakeet variants (spec Section 14) — verify alignment before trusting `mean_word_prob`.

**Files (only if NeMo wins):**
- Modify: `modes/talkback/stt.py` (`_ensure_model`, `_transcribe_sync`, imports)
- Modify: `config.yaml` (`kiosk.talkback.stt.backend: "nemo"`, `.model: "<nemo-model-id>"`)

**Interfaces:**
- Consumes: `TranscriptResult` (Task 2, unchanged).
- Produces: the **same** `async transcribe_segment(audio) -> TranscriptResult` contract on a NeMo engine.

- [ ] **Step 1: Confirm the gate**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
grep -n "PRIMARY" docs/notes/2026-06-22-stt-backend.md
```
If the PRIMARY line says **openai-whisper**, STOP — this task is a no-op; the plan is complete after Task 6. Proceed only if PRIMARY says **NeMo**.

- [ ] **Step 2: Swap `_ensure_model` to load the NeMo model**

Replace `_ensure_model` in `modes/talkback/stt.py` with:

```python
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
        self._model.eval()
        if self._device == "cuda":
            self._model = self._model.cuda()
```

- [ ] **Step 3: Swap `_transcribe_sync` to NeMo + per-word confidence averaging**

Replace `_transcribe_sync` with a NeMo body that writes the clip to a temp wav (NeMo's `transcribe` takes file paths), pulls `hyp.text` and `hyp.word_confidence`, and averages into `mean_word_prob`:

```python
    def _transcribe_sync(self, audio: np.ndarray) -> "TranscriptResult":
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            sf.write(tf.name, audio, 16000)
            out = self._model.transcribe([tf.name])
        hyp = out[0] if out else None
        if hyp is None:
            return TranscriptResult(text="", mean_word_prob=0.0)
        text = (getattr(hyp, "text", "") or (hyp if isinstance(hyp, str) else "")).strip()
        word_conf = getattr(hyp, "word_confidence", None) or []
        # Guard the known parakeet length-mismatch bug (spec Section 14): only
        # average confidences when present; fall back to 0.0 (-> RESTORE) if absent.
        probs = [float(c) for c in word_conf if c is not None]
        mean_word_prob = (
            max(0.0, min(1.0, sum(probs) / len(probs))) if probs else 0.0
        )
        return TranscriptResult(text=text, mean_word_prob=mean_word_prob)
```

(Note: NeMo `word_confidence` is in `[0,1]` when its `confidence` decoding is enabled; if a probed model returns no confidence, set the model's decoding config to emit it, or — if unavailable — fall back to openai-whisper, since the empty/low-conf guard requires a real number.)

- [ ] **Step 4: Update config + the no-faster-whisper test still passes**

Edit `config.yaml` `kiosk.talkback.stt`: set `backend: "nemo"` and `model: "<nemo-model-id from the verdict>"`. The Task-6 regression (`import whisper`) must be relaxed for NeMo — change its last assertion:

```python
    # backend is one of the two approved engines; faster-whisper stays banned
    assert "import whisper" in src or "nemo" in src
```

- [ ] **Step 5: Run the contract tests against the NeMo backend**

Run:
```bash
cd /home/ldrgx10/FullDuplexVoice/TVAD/target-vad
python3 -m pytest tests/kiosk/talkback/test_stt.py -v
python3 -m pytest tests/kiosk/talkback/test_stt_cuda.py -v
```
Expected: pure-logic tests PASS unchanged (they use the fake model and the unchanged `TranscriptResult`); the CUDA integration test PASSES on GB10 with the NeMo model loaded (non-empty `text`, `mean_word_prob` in `(0,1]`). The `_FakeWhisperModel` in `test_stt.py` mimics the **whisper** result-dict shape; if NeMo wins, also add a `_FakeNemoModel` fake whose `transcribe([...])` returns an object with `.text` and `.word_confidence`, and point the averaging tests at `_transcribe_sync` via that fake (the contract assertions are identical).

- [ ] **Step 6: Commit**

```bash
git add modes/talkback/stt.py config.yaml tests/kiosk/talkback/test_stt.py
git commit -m "feat(stt): swap StreamingStt backend to NeMo per backend-selection verdict"
```

---

## Self-Review

**1. Spec coverage (Section 9 "Streaming STT", Section 6 "Empty/low-confidence guard", Section 14 risks):**

- *faster-whisper has no aarch64 CUDA wheel → re-back both paths* (Section 9): Task 3 removes faster-whisper internals; Task 6 regresses against re-introduction. ✓
- *Keep `StreamingStt` class name/interface; swap internals* (Section 9, reuse-map): Task 3 keeps the class + async `transcribe_segment`; Task 2/4 assert the contract. ✓
- *Extend `transcribe_segment` → `TranscriptResult(text, mean_word_prob)`* (Section 6 binding contract): Tasks 2–3 deliver exactly this; field names/types match Plan 01 `UserTurnTranscribed`/`InterjectionTranscribed` (`text`, `mean_word_prob`) and `DirectorConfig.conf_floor`. ✓
- *Empty/low-confidence RESTORE path* (Section 6): `mean_word_prob == 0.0` on empty, averaged otherwise; Task 5 documents the `conf_floor` linkage; the RESTORE decision itself lives in Plan 01's reducer (line 630, 977) and is *fed* by this number — not duplicated here. ✓
- *`language="en"`* (Section 2): preserved in `_transcribe_sync` (Task 3). ✓
- *openai-whisper recommended primary; NeMo upgrade path* (Section 9): Tasks 2–6 ship openai-whisper; Task 7 is the bounded NeMo swap, explicitly gated on the Task-1 verdict. ✓
- *NeMo install constraints — PyTorch 2.9/container 25.10, lhotse>=1.32.2, NIM x86-only/forbidden* (Section 9, 14): stated in Task 1 and Task 7; the probe does not install NeMo destructively. ✓
- *Task 1 = real runnable backend-selection spike with exact commands + expected outputs, result picks primary, recorded in `docs/notes/2026-06-22-stt-backend.md`* (prompt): `bench/stt_backend_probe.py` self-bootstraps `sys.path` (matching `bench/reflex_contention.py`), measures latency + per-word confidence on `self.wav`'s first 3s, and the notes file records the verdict; Step 4 GATEs Task 2 on a recorded PRIMARY. ✓
- *CI skip-guard for CUDA/model-unavailable* (prompt): `test_stt_cuda.py` uses `requires_gpu` skipif; pure-logic tests use a fake model and always run. ✓
- *Tests assert float in [0,1] for `mean_word_prob` on a tiny fixed clip* (prompt): `test_transcribe_real_clip_returns_valid_result` (real `self.wav`) and the fake-model averaging tests (fixed dicts). ✓
- *Update `config.yaml` `kiosk.talkback.stt` model/backend keys* (prompt): Task 5 sets `backend`/`model`/`device`. ✓
- *Mark which tasks change if NeMo wins* (prompt): Task 7, bounded to `_ensure_model` + `_transcribe_sync` + config; contract/tests unchanged. ✓

**2. Placeholder scan:** No "TBD/TODO/FIXME/implement later". The `<fill>` markers in the Task-1 notes template are **data-entry slots for real measured numbers**, by design (the human records actual probe output) — not code placeholders. The spike-gate language in Task 1 / Task 7 is a real decision gate with runnable commands and expected outputs, not a stub. Every code step shows complete code. ✓

**3. Type consistency:** `TranscriptResult(text: str, mean_word_prob: float)` is defined once (Task 2) and used identically in Tasks 3, 4, 6, 7. `StreamingStt(model, device)` constructor (Task 3) — note the **removal** of the old `compute_type` param (openai-whisper has none); Task 5's config drops it to match. `_transcribe_sync`, `_ensure_model`, `_mean_word_prob` names are consistent across Tasks 3 and 7. `transcribe_segment` stays `async` and returns `TranscriptResult` everywhere — matching the SttWorker (Plan 02) and the Plan 01 events `UserTurnTranscribed`/`InterjectionTranscribed`. ✓
