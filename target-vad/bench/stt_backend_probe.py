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


def probe_nemo(model_name, clip, enable_confidence=False, verbose=False):
    """Live GO/NO-GO spike (Task 9): loads a Parakeet checkpoint from
    NGC/HF via from_pretrained (downloads + caches the .nemo on first use,
    ~2.4GB for the 0.6b models), runs it on CUDA, and measures:
      * mean transcribe() latency over N_TOTAL-N_WARMUP timed runs
      * whether per-word confidence is populated on the returned Hypothesis
        objects (transcribe(..., return_hypotheses=True) is required for
        Hypothesis objects at all; the plain-string API never carries them)

    If enable_confidence=True, first calls model.change_decoding_strategy()
    with a decoding config that turns on preserve_word_confidence (TDT/RNNT
    confidence is OFF by default in NeMo — this is the documented way to
    turn it on; see docs/notes for the exact invocation this discovered).
    """
    try:
        import nemo  # noqa: F401
        import nemo.collections.asr as nemo_asr
    except Exception as e:  # noqa: BLE001
        return {"error": f"nemo not installed / import failed: {e}"}

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch import failed: {e}"}

    if not torch.cuda.is_available():
        return {"error": "torch.cuda not available"}

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        return {"error": f"from_pretrained({model_name}) failed: {e}"}

    model = model.to("cuda").eval()

    confidence_cfg_applied = None
    if enable_confidence:
        try:
            from omegaconf import open_dict

            from nemo.collections.asr.parts.utils.asr_confidence_utils import (
                ConfidenceConfig,
            )

            confidence_cfg = ConfidenceConfig(
                preserve_word_confidence=True,
                preserve_token_confidence=True,
                preserve_frame_confidence=False,
            )
            decoding_cfg = model.cfg.decoding
            # decoding_cfg is a struct DictConfig — confidence_cfg is not an
            # existing key on the TDT/RNNT decoding schema, so a plain
            # attribute-set raises "Key 'confidence_cfg' is not in struct".
            # open_dict() temporarily disables struct-mode to allow adding it.
            with open_dict(decoding_cfg):
                decoding_cfg.confidence_cfg = confidence_cfg
            model.change_decoding_strategy(decoding_cfg)
            confidence_cfg_applied = (
                "from omegaconf import open_dict; "
                "decoding_cfg = model.cfg.decoding; "
                "with open_dict(decoding_cfg): "
                "decoding_cfg.confidence_cfg = ConfidenceConfig("
                "preserve_word_confidence=True, preserve_token_confidence=True, "
                "preserve_frame_confidence=False); "
                "model.change_decoding_strategy(decoding_cfg)"
            )
        except Exception as e:  # noqa: BLE001
            confidence_cfg_applied = f"FAILED to enable confidence: {e}"

    # transcribe() takes the raw float32 array directly on recent NeMo
    # (audio=[np.ndarray, ...]); fall back to a temp wav path for older
    # versions that only accept file paths.
    times = []
    text = ""
    has_word_conf = False
    conf_attrs = []
    sample_word_confidences = []
    device_seen = None
    hyp = None

    def _run_transcribe():
        try:
            return model.transcribe(audio=[clip], return_hypotheses=True, verbose=False)
        except TypeError:
            # Older NeMo: no `audio=` kwarg / no `verbose=` kwarg.
            try:
                return model.transcribe([clip], return_hypotheses=True)
            except Exception:
                import tempfile

                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                    sf.write(tf.name, clip, SR)
                    return model.transcribe([tf.name], return_hypotheses=True)

    # Warm-up (excluded from timing; also primes CUDA kernels/cuDNN autotune).
    try:
        _run_transcribe()
    except Exception as e:  # noqa: BLE001
        return {"error": f"transcribe() warm-up failed: {e}"}

    try:
        for p in model.parameters():
            device_seen = str(p.device)
            break
    except Exception:  # noqa: BLE001
        pass

    for i in range(N_TOTAL):
        t0 = time.perf_counter()
        out = _run_transcribe()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= N_WARMUP:
            times.append(dt)
        # NeMo returns Hypothesis objects (or strings on older versions).
        h = out[0] if out else None
        # Some NeMo versions nest single-output batches in a list-of-lists.
        if isinstance(h, list) and h:
            h = h[0]
        hyp = h
        if hasattr(hyp, "text"):
            text = (hyp.text or "").strip()
            conf_attrs = [a for a in dir(hyp) if "conf" in a.lower()]
            wc = getattr(hyp, "word_confidence", None)
            has_word_conf = bool(wc)
            if wc:
                sample_word_confidences = [round(float(x), 3) for x in list(wc)[:8]]
        elif isinstance(hyp, str):
            text = hyp.strip()

    p50, p95 = p50_p95(times)
    mean_ms = float(np.mean(times)) if times else float("nan")
    if verbose:
        print(f"    [nemo] device={device_seen} confidence_cfg={confidence_cfg_applied}")
        print(f"    [nemo] hypothesis attrs with 'conf': {conf_attrs}")
        print(f"    [nemo] sample word_confidence: {sample_word_confidences}")
    return {
        "loaded": True,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean_ms,
        "text": text[:80],
        "has_word_conf": has_word_conf,
        "conf_attrs": conf_attrs,
        "sample_word_confidences": sample_word_confidences,
        "device": device_seen,
        "confidence_cfg_applied": confidence_cfg_applied,
    }


def fmt_row(name, r):
    if "error" in r:
        return f"  {name:<34} FAILED: {r['error'][:60]}"
    conf = "YES" if r["has_word_conf"] else "NO"
    return (
        f"  {name:<34} p50={r['p50_ms']:6.1f}ms  p95={r['p95_ms']:6.1f}ms  "
        f"word_conf={conf:<3}  text={r['text']!r}"
    )


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "STT backend-selection spike. With no args, runs the full "
            "openai-whisper + NeMo matrix (original Plan-04 behavior). "
            "Pass --nemo-model for a fast, single-model Task-9 GO/NO-GO run "
            "(skips whisper, which is already proven on this box)."
        )
    )
    p.add_argument(
        "--nemo-model",
        action="append",
        default=None,
        metavar="MODEL",
        help=(
            "NeMo/HF pretrained model name to probe (repeatable), e.g. "
            "nvidia/parakeet-tdt-0.6b-v2. When given, only these NeMo "
            "models are probed (whisper is skipped) for a fast spike."
        ),
    )
    p.add_argument(
        "--nemo-confidence",
        action="store_true",
        help=(
            "Enable per-word/token confidence via "
            "model.change_decoding_strategy(...ConfidenceConfig(...)) "
            "before timing/transcribing."
        ),
    )
    p.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip the openai-whisper rows even in full-matrix mode.",
    )
    return p.parse_args()


def main():
    args = parse_args()
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

    if args.nemo_model:
        # Fast, targeted Task-9 spike: only the requested NeMo model(s).
        for name in args.nemo_model:
            print(f"  probing NeMo {name} (confidence={args.nemo_confidence}) ...", flush=True)
            results[f"nemo/{name}"] = probe_nemo(
                name, clip, enable_confidence=args.nemo_confidence, verbose=True
            )
    else:
        if not args.skip_whisper:
            for name in ("base.en", "tiny"):
                print(f"  probing openai-whisper {name} ...", flush=True)
                results[f"openai-whisper/{name}"] = probe_openai_whisper(name, clip)
        for name in ("nvidia/parakeet-tdt-0.6b-v2",):
            print(f"  probing NeMo {name} ...", flush=True)
            results[f"nemo/{name}"] = probe_nemo(
                name, clip, enable_confidence=args.nemo_confidence, verbose=True
            )

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
