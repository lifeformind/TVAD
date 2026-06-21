# GB10 Reflex-Path Latency Under GPU Contention — Spike 3 Benchmark

**Date:** 2026-06-21
**Hardware:** NVIDIA GB10 (DGX Spark, aarch64, single GPU, ~128 GB unified memory)
**Branch:** feat/conversation-derisk-spike
**Task:** Spike 3 — Measure reflex-path latency under GPU contention to validate
the "own the conversation loop" decision.

---

## Load Source

**Real gemma-3-4b-it LLM inference (llama.cpp, full GPU offload)**

The llama_cpp.server was started with `n_gpu_layers=-1` (full GPU offload of
gemma-3-4b-it-Q5_K_M.gguf, ~3.4 GB VRAM). A background thread continuously
POSTed 64-token completions to `http://127.0.0.1:8080/v1/completions`, creating
steady GPU memory pressure concurrent with reflex-path inference calls.

This is the realistic production contention scenario: the LLM generating a
response while the VAD/reflex path processes incoming user audio.

---

## Methodology

- **35 iterations total, first 3 discarded as warmup** → 32 timed samples per cell
- `time.perf_counter()` wall-clock timing (includes Python overhead)
- Load thread started 1 s before the under-load phase begins
- Audio inputs are zero-filled synthetic arrays of the correct shape/duration
- Whisper note: faster-whisper / CTranslate2 4.8.0 on aarch64 ships without CUDA
  support (PyPI wheel is CPU-only). `openai-whisper` (torch-native) was used
  instead. The benchmark times the full mel-spectrogram-compute + decode path
  on a 200ms chunk padded to the 30s whisper window — this is a ceiling
  (real streaming-partial would short-circuit on silence).

---

## Full Latency Table

| Component                         | Idle p50 | Idle p95 | Load p50 | Load p95 | Placement |
|-----------------------------------|----------|----------|----------|----------|-----------|
| whisper-tiny (CUDA, openai-whisper) | 13.3 ms | 14.7 ms  | 24.0 ms  | 38.8 ms  | GPU/CUDA  |
| ECAPA embed (CPU, SpeechBrain)    | 33.0 ms  | 43.0 ms  | 82.8 ms  | 108.4 ms | CPU       |
| Smart Turn (CPU, ONNX)            | 26.8 ms  | 29.1 ms  | 38.9 ms  | 55.4 ms  | CPU       |

---

## Hypothesis Analysis

### H1: Smart Turn (CPU ONNX) stays flat under GPU load

**PARTIALLY CONFIRMED.**

- Idle p50 = 26.8 ms → Load p50 = 38.9 ms (1.45× inflation)
- Idle p95 = 29.1 ms → Load p95 = 55.4 ms (1.90× inflation)

The detector did not stay perfectly flat — it inflated ~1.45–1.9× under real
gemma load. This is consistent with the unified-memory architecture of the GB10:
even though the ONNX session uses only CPUExecutionProvider, the CPU inference
cores share the memory bus with the GPU. Under LLM prefill/decode traffic the
memory bandwidth contention adds ~12 ms at p50 and ~26 ms at p95.

Despite the inflation, Smart Turn's **p95 under load (55 ms) is comfortably
within the 100 ms reflex budget**. The hypothesis that "CPU specialists are
insulated from GPU load" holds at the budget level even if not perfectly flat.

### H2: Whisper (GPU) inflates under GPU contention

**CONFIRMED — but stays within budget.**

- Idle p50 = 13.3 ms → Load p50 = 24.0 ms (1.80× inflation)
- Idle p95 = 14.7 ms → Load p95 = 38.8 ms (2.64× inflation)

Whisper-tiny on CUDA inflates ~1.8× at p50 under real LLM load. The absolute
numbers remain well within budget: p95 under load is 38.8 ms. This is
surprisingly fast for a 30s padded whisper window — the silent chunk is decoded
very quickly (no real speech to transcribe).

**Important caveat:** the idle measurement (13.3 ms p50) was taken immediately
after model load before the gemma server had been doing any inference. In a
production scenario whisper would share the GPU with an actively decoding LLM
(including KV-cache residency). Under heavier or sustained LLM load (long
sequences, multi-turn context), whisper latency could push higher. The 38.8 ms
p95 here is an optimistic floor.

### H3: ECAPA embed (CPU) under GPU load

**NOTABLE INFLATION** — p50 increases 2.51× (33 ms → 83 ms), and p95 **just
barely exceeds** the 100 ms target at **108.4 ms**.

ECAPA uses SpeechBrain with CPU inference. Under unified-memory contention it
shows the most inflation of the CPU components. This is expected: ECAPA runs
heavier convolution ops than Smart Turn, so it consumes more memory bandwidth
for longer periods per call.

For the keep-vs-escalate reflex path, ECAPA is used to gate speaker identity
on an 800ms window. A p95 of 108 ms (just over budget) means ECAPA **should
not sit on the synchronous reflex hot path** under GPU load. Instead it should
run concurrently with or after the Smart Turn decision.

---

## GO / NO-GO Conclusion

**GO — sub-100ms reflex is achievable on this GB10, with specific component
placement.**

### Recommended placement

| Reflex step              | Component         | Latency under load p95 | Verdict   |
|--------------------------|-------------------|------------------------|-----------|
| Turn-end detection       | Smart Turn (CPU)  | 55 ms                  | SAFE      |
| Streaming-partial STT    | whisper-tiny (GPU)| 39 ms                  | SAFE      |
| Speaker gating           | ECAPA (CPU)       | 108 ms                 | TOO SLOW for synchronous hot path |

### Design implications

1. **Smart Turn on CPU is the lowest-latency reliable reflex trigger.** At 55 ms
   p95 under real LLM contention, it can gate the cut/keep decision within the
   reflex budget. Keep it on CPU.

2. **Whisper-tiny on GPU is fast** (39 ms p95) and provides the transcribed
   partial for intent classification. However, faster-whisper/CTranslate2 lacks
   a CUDA wheel for aarch64 — the production pipeline must either (a) use
   openai-whisper (torch-native) as a drop-in replacement for CUDA, or (b) build
   CTranslate2 from source with CUDA support. The current `StreamingStt` class
   (which wraps faster-whisper) will fall back to CPU (~270 ms p50) on this
   hardware.

3. **ECAPA must be moved off the synchronous reflex hot path.** Its p95 under
   load (108 ms) just exceeds the 100 ms budget. The rolling-window speaker gate
   (from the existing `feat/` architecture) should run on a side-channel thread
   that produces a gate decision slightly after the reflex fires, not as a
   prerequisite to it. This is already consistent with the 2-3s window
   requirement documented in `ecapa-short-segment-unreliable.md`.

4. **Unified memory contention is real but manageable.** The GB10's single-bus
   architecture means even CPU components see memory-bus inflation (~1.4–2.5×)
   under heavy GPU load. At p95, everything except ECAPA still clears 100 ms.
   Thermal throttling under sustained load was not measured — add that to the
   next spike if long-session performance degrades.

---

## Caveats

- **Whisper STT proxy limitation:** the bench times a 30s-padded zero-array
  decode. Real streaming-partial inference on a 200ms speech chunk with VAD
  might be slower (more non-zero tokens) or faster (early exit on a short
  utterance). Treat 38.8 ms p95 as an optimistic floor; budget 60–80 ms for
  real speech.
- **CTranslate2 aarch64 / CUDA gap:** the production `StreamingStt` class uses
  faster-whisper which cannot run on CUDA on this hardware without a custom
  CTranslate2 build. This must be resolved before the reflex path goes into
  production (see implications above).
- **Single GPU, no concurrency isolation:** gemma and whisper compete for the
  same GPU. The numbers here reflect a single concurrent completion; if the LLM
  batch size increases or the context grows, whisper GPU latency could inflate
  further.
- **Warmup and server state:** the LLM server was freshly started for this
  benchmark. Production steady-state (warm KV cache, thermal equilibrium) may
  differ slightly.
- **ECAPA on short chunks:** as documented in `ecapa-short-segment-unreliable.md`,
  the 800ms window used here is at the minimum reliable duration; real deployment
  uses 2–3s windows, which will take ~2–4× longer to run, further confirming
  that ECAPA must be kept off the synchronous hot path.
