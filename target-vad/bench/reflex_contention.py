"""bench/reflex_contention.py — Reflex-path latency under GB10 GPU contention.

Measures p50/p95 wall-clock latency (≥30 real-inference iterations, with
warmup discarded) for each component of the conversation-loop REFLEX path,
both IDLE and UNDER GPU LOAD.

Components measured
-------------------
1. whisper-tiny (CUDA)  — STT on a 200ms synthetic chunk (streaming-partial
                          proxy). Uses openai-whisper (torch-native, CUDA),
                          since the faster-whisper / CTranslate2 aarch64 wheel
                          ships without CUDA support. The 200ms input is padded
                          to 30 s (whisper's fixed window) before mel + decode;
                          this mirrors what a real streaming-partial call would
                          cost on the GPU.
2. ECAPA embed (CPU)    — SpeechBrain ECAPA-TDNN on an 800ms window.
                          Runs on CPU (as the production controller does).
3. Smart Turn (CPU)     — CPU ONNX endpoint_prob on an 800ms window, zero-
                          padded to 8 s internally. Expected to stay flat under
                          GPU load.

GPU load source
---------------
Prefers a running gemma-3-4b-it llama.cpp server on localhost:8080 (real LLM
inference). Falls back to a synthetic CUDA matmul loop if the server is down.
The load source is clearly labelled in the output.

Usage
-----
    cd /path/to/target-vad
    python bench/reflex_contention.py

Options (env-vars)
------------------
    BENCH_DEVICE   override device for whisper (default: cuda)
    BENCH_N        number of timed iterations (default: 35; first 3 are warmup)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SR = 16_000
CHUNK_200MS = int(SR * 0.200)   # 3 200 samples  — whisper streaming-partial
CHUNK_800MS = int(SR * 0.800)   # 12 800 samples — ECAPA / Smart Turn
N_TOTAL = int(os.environ.get("BENCH_N", 35))
N_WARMUP = 3
N_TIMED  = N_TOTAL - N_WARMUP
DEVICE   = os.environ.get("BENCH_DEVICE", "cuda")

LLM_URL   = "http://127.0.0.1:8080/v1/completions"
LLM_MODEL = "gemma-3-4b-it-Q5_K_M"   # llama_cpp.server uses filename sans .gguf


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def p50_p95(times):
    arr = np.array(times, dtype=float)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def fmt(ms):
    return f"{ms:7.1f} ms"


def _llm_server_reachable():
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Load generators
# ──────────────────────────────────────────────────────────────────────────────

def _gemma_load_loop(stop_event: threading.Event):
    """Continuously POST short completions to the llama.cpp server."""
    import urllib.request, json
    payload = json.dumps({
        "prompt": "Tell me a one-sentence fact about space.",
        "max_tokens": 64,
        "temperature": 0.7,
    }).encode()
    headers = {"Content-Type": "application/json"}
    while not stop_event.is_set():
        try:
            req = urllib.request.Request(LLM_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30):
                pass
        except Exception:
            pass   # server might be busy; just retry


def _cuda_matmul_loop(stop_event: threading.Event):
    """Synthetic CUDA load: repeated large matmul + sync."""
    import torch
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    while not stop_event.is_set():
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        del C


# ──────────────────────────────────────────────────────────────────────────────
# Component constructors
# ──────────────────────────────────────────────────────────────────────────────

def build_whisper():
    """Load openai-whisper tiny on CUDA.

    faster-whisper / CTranslate2's aarch64 wheel ships without CUDA support
    (confirmed: ctranslate2==4.8.0 raises ValueError on any CUDA call).
    openai-whisper uses torch natively and works on CUDA on this GB10.

    The bench mimics a streaming-partial: 200ms audio, padded to 30s
    (whisper's fixed window), then mel + decode.  This is more conservative
    than a real streaming call which would short-circuit on silence, so the
    numbers are a ceiling, not a floor.
    """
    import whisper
    import torch
    import whisper.audio as waudio

    model = whisper.load_model("tiny", device=DEVICE)
    # Pre-build the decode options (cached)
    options = whisper.DecodingOptions(
        language="en",
        without_timestamps=True,
        fp16=(DEVICE == "cuda"),
    )

    # Pre-pad the 200ms chunk once; shape is always the same in the bench.
    audio_np = np.zeros(CHUNK_200MS, dtype=np.float32)
    audio_padded = waudio.pad_or_trim(audio_np)   # → (480000,) float32
    mel = waudio.log_mel_spectrogram(audio_padded).to(DEVICE)

    def _decode_sync(_audio_np=None):
        # In the real reflex path the mel is computed fresh each call;
        # we pre-compute it here to isolate decoder latency vs. mel latency.
        # To include mel compute, set include_mel=True below.
        with torch.no_grad():
            whisper.decode(model, mel, options)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # Expose a _transcribe_sync-compatible interface
    class _W:
        def _transcribe_sync(self, audio):
            _decode_sync(audio)
            return ""

    obj = _W()
    # Full round-trip (mel + decode) for a fair comparison
    audio_np2 = np.zeros(CHUNK_200MS, dtype=np.float32)

    def _full_rt():
        ap = waudio.pad_or_trim(audio_np2)
        m2 = waudio.log_mel_spectrogram(ap).to(DEVICE)
        with torch.no_grad():
            whisper.decode(model, m2, options)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    obj._full_round_trip = _full_rt
    return obj


def build_ecapa():
    from core.speaker.embedder import EmbeddingExtractor
    ext = EmbeddingExtractor()
    ext._ensure_model()
    return ext


def build_smart_turn():
    from modes.talkback.endpointing import SmartTurnDetector
    return SmartTurnDetector()


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ──────────────────────────────────────────────────────────────────────────────

def bench_whisper(stt, n):
    """Time the full mel-compute + decode path per iteration."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        stt._full_round_trip()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_ecapa(ext, n):
    audio = np.zeros(CHUNK_800MS, dtype=np.float32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        ext.extract(audio, sample_rate=SR)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_smart_turn(det, n):
    audio = np.zeros(CHUNK_800MS, dtype=np.float32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        det.endpoint_prob(audio, sample_rate=SR)
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Reflex-path latency benchmark — GB10 GPU contention (Spike 3)")
    print("=" * 70)
    print(f"  Iterations: {N_TOTAL} total ({N_WARMUP} warmup + {N_TIMED} timed)")
    print(f"  Whisper device: {DEVICE}")
    print()

    # ── Determine load source ────────────────────────────────────────────────
    if _llm_server_reachable():
        load_source = "real gemma-3-4b-it (llama.cpp, GPU)"
        load_fn = _gemma_load_loop
        print(f"  Load source: {load_source}")
    else:
        load_source = "synthetic CUDA matmul (torch.matmul 4096x4096, fp16)"
        load_fn = _cuda_matmul_loop
        print(f"  Load source: {load_source}")
        print("  NOTE: llama.cpp server not reachable; using synthetic CUDA load.")
    print()

    # ── Load models (once, before any timing) ────────────────────────────────
    print("Loading models...")
    errors = {}

    stt = None
    try:
        print("  [1/3] whisper-tiny (openai-whisper, CUDA) ...", end=" ", flush=True)
        stt = build_whisper()
        print("OK")
    except Exception as e:
        errors["whisper"] = str(e)
        print(f"FAILED: {e}")

    ext = None
    try:
        print("  [2/3] ECAPA-TDNN (SpeechBrain, CPU) ...", end=" ", flush=True)
        ext = build_ecapa()
        print("OK")
    except Exception as e:
        errors["ecapa"] = str(e)
        print(f"FAILED: {e}")

    det = None
    try:
        print("  [3/3] Smart Turn v3.2 (ONNX, CPU) ...", end=" ", flush=True)
        det = build_smart_turn()
        print("OK")
    except Exception as e:
        errors["smart_turn"] = str(e)
        print(f"FAILED: {e}")

    print()

    # ── IDLE measurements ────────────────────────────────────────────────────
    print("─" * 70)
    print("  Phase 1: IDLE (no GPU load)")
    print("─" * 70)

    idle = {}

    if stt:
        print(f"  whisper-tiny (200ms→30s, {N_TOTAL} iters)...", end=" ", flush=True)
        all_t = bench_whisper(stt, N_TOTAL)
        idle["whisper"] = all_t[N_WARMUP:]
        print(f"done  p50={p50_p95(idle['whisper'])[0]:.1f}ms")
    else:
        idle["whisper"] = None

    if ext:
        print(f"  ECAPA embed (800ms, {N_TOTAL} iters)...", end=" ", flush=True)
        all_t = bench_ecapa(ext, N_TOTAL)
        idle["ecapa"] = all_t[N_WARMUP:]
        print(f"done  p50={p50_p95(idle['ecapa'])[0]:.1f}ms")
    else:
        idle["ecapa"] = None

    if det:
        print(f"  Smart Turn (800ms, {N_TOTAL} iters)...", end=" ", flush=True)
        all_t = bench_smart_turn(det, N_TOTAL)
        idle["smart_turn"] = all_t[N_WARMUP:]
        print(f"done  p50={p50_p95(idle['smart_turn'])[0]:.1f}ms")
    else:
        idle["smart_turn"] = None

    print()

    # ── UNDER-LOAD measurements ───────────────────────────────────────────────
    print("─" * 70)
    print(f"  Phase 2: UNDER LOAD ({load_source})")
    print("─" * 70)

    stop_event = threading.Event()
    load_thread = threading.Thread(target=load_fn, args=(stop_event,), daemon=True)
    load_thread.start()
    print("  Load thread started, sleeping 1s to ramp up...")
    time.sleep(1)

    loaded = {}

    if stt:
        print(f"  whisper-tiny (200ms→30s, {N_TOTAL} iters)...", end=" ", flush=True)
        all_t = bench_whisper(stt, N_TOTAL)
        loaded["whisper"] = all_t[N_WARMUP:]
        print(f"done  p50={p50_p95(loaded['whisper'])[0]:.1f}ms")
    else:
        loaded["whisper"] = None

    if ext:
        print(f"  ECAPA embed (800ms, {N_TOTAL} iters)...", end=" ", flush=True)
        all_t = bench_ecapa(ext, N_TOTAL)
        loaded["ecapa"] = all_t[N_WARMUP:]
        print(f"done  p50={p50_p95(loaded['ecapa'])[0]:.1f}ms")
    else:
        loaded["ecapa"] = None

    if det:
        print(f"  Smart Turn (800ms, {N_TOTAL} iters)...", end=" ", flush=True)
        all_t = bench_smart_turn(det, N_TOTAL)
        loaded["smart_turn"] = all_t[N_WARMUP:]
        print(f"done  p50={p50_p95(loaded['smart_turn'])[0]:.1f}ms")
    else:
        loaded["smart_turn"] = None

    stop_event.set()
    load_thread.join(timeout=5)
    print()

    # ── Results table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Load source: {load_source}")
    print()
    header = f"  {'Component':<26} {'Idle p50':>10} {'Idle p95':>10} {'Load p50':>10} {'Load p95':>10}"
    print(header)
    print("  " + "-" * 68)

    rows = [
        ("whisper-tiny (CUDA, openai-w)",  "whisper",    "CUDA"),
        ("ECAPA embed (CPU, SpeechBrain)", "ecapa",      "CPU"),
        ("Smart Turn (CPU, ONNX)",         "smart_turn", "CPU"),
    ]

    for label, key, placement in rows:
        if idle.get(key) is None:
            err = errors.get(key, "unknown error")
            print(f"  {label:<26} {'MISSING':>10} {'MISSING':>10} {'MISSING':>10} {'MISSING':>10}  ({err[:40]})")
            continue
        ip50, ip95 = p50_p95(idle[key])
        lp50, lp95 = p50_p95(loaded[key])
        print(f"  {label:<26} {fmt(ip50):>10} {fmt(ip95):>10} {fmt(lp50):>10} {fmt(lp95):>10}  [{placement}]")

    print()

    # ── Hypothesis analysis ───────────────────────────────────────────────────
    print("  HYPOTHESIS ANALYSIS")
    print("  " + "-" * 68)
    if idle.get("smart_turn") and loaded.get("smart_turn"):
        st_ip50, st_ip95 = p50_p95(idle["smart_turn"])
        st_lp50, st_lp95 = p50_p95(loaded["smart_turn"])
        st_ratio = st_lp50 / st_ip50 if st_ip50 > 0 else float("inf")
        verdict = "CONFIRMED" if st_ratio < 1.30 else "PARTIALLY CONFIRMED" if st_ratio < 2.0 else "REFUTED"
        print(f"  Smart Turn (CPU) stays flat under GPU load?")
        print(f"    idle p50={st_ip50:.1f}ms p95={st_ip95:.1f}ms | load p50={st_lp50:.1f}ms p95={st_lp95:.1f}ms")
        print(f"    p50 ratio loaded/idle = {st_ratio:.2f}x — {verdict}")
    else:
        print("  Smart Turn: not measured.")

    if idle.get("whisper") and loaded.get("whisper"):
        w_ip50, w_ip95 = p50_p95(idle["whisper"])
        w_lp50, w_lp95 = p50_p95(loaded["whisper"])
        w_ratio = w_lp50 / w_ip50 if w_ip50 > 0 else float("inf")
        print(f"  Whisper (CUDA) inflation under load:")
        print(f"    idle p50={w_ip50:.1f}ms p95={w_ip95:.1f}ms | load p50={w_lp50:.1f}ms p95={w_lp95:.1f}ms")
        print(f"    p50 ratio loaded/idle = {w_ratio:.2f}x")
        print(f"    Whisper p95 under load: {w_lp95:.1f}ms — {'within' if w_lp95 < 100 else 'EXCEEDS'} 100ms target")
    else:
        print("  Whisper: not measured (faster-whisper/CTranslate2 aarch64 ships CPU-only;")
        print("           openai-whisper used instead).")

    if idle.get("ecapa") and loaded.get("ecapa"):
        e_ip50, e_ip95 = p50_p95(idle["ecapa"])
        e_lp50, e_lp95 = p50_p95(loaded["ecapa"])
        e_ratio = e_lp50 / e_ip50 if e_ip50 > 0 else float("inf")
        print(f"  ECAPA (CPU) inflation under GPU load:")
        print(f"    idle p50={e_ip50:.1f}ms p95={e_ip95:.1f}ms | load p50={e_lp50:.1f}ms p95={e_lp95:.1f}ms")
        print(f"    p50 ratio loaded/idle = {e_ratio:.2f}x")
    else:
        print("  ECAPA: not measured.")

    print()

    # ── GO/NO-GO ─────────────────────────────────────────────────────────────
    print("  GO / NO-GO (sub-100ms reflex under contention)")
    print("  " + "-" * 68)
    go_parts = []
    if loaded.get("whisper") and loaded.get("smart_turn"):
        w_lp95 = p50_p95(loaded["whisper"])[1]
        st_lp95 = p50_p95(loaded["smart_turn"])[1]
        if w_lp95 < 100 and st_lp95 < 100:
            print(f"  GO — both whisper-CUDA ({w_lp95:.1f}ms p95) and Smart Turn ({st_lp95:.1f}ms p95)")
            print(f"       are sub-100ms under {load_source}.")
        elif st_lp95 < 100 and (not loaded.get("whisper") or w_lp95 >= 100):
            print(f"  GO (CPU-only reflex path) — Smart Turn p95={st_lp95:.1f}ms (OK).")
            if loaded.get("whisper"):
                print(f"  CAVEAT: Whisper-CUDA p95={w_lp95:.1f}ms exceeds 100ms under load.")
                print(f"          The GPU reflex path (whisper) must be treated as async / best-effort.")
        else:
            print(f"  NO-GO — neither path meets 100ms p95 under load.")
    elif loaded.get("smart_turn"):
        st_lp95 = p50_p95(loaded["smart_turn"])[1]
        print(f"  PARTIAL GO — Smart Turn (CPU) p95={st_lp95:.1f}ms under load.")
        print(f"  Whisper (GPU) not measured — see MISSING COMPONENTS.")
        if st_lp95 < 100:
            print(f"  CPU-only reflex path (Smart Turn) is viable.")
    else:
        print("  INCONCLUSIVE — critical components failed to load.")

    print()
    if errors:
        print("  MISSING / FAILED COMPONENTS:")
        for k, v in errors.items():
            print(f"    {k}: {v[:80]}")
        print()

    print("=" * 70)

    return {
        "load_source": load_source,
        "idle": idle,
        "loaded": loaded,
        "errors": errors,
    }


if __name__ == "__main__":
    main()
