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
