# Director Plan 05 — Crowd-Focus pVAD (SPIKE-GATED) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ingestion worker's `is_target` field *real* by adding a personal-VAD (pVAD) worker that, conditioned on the enrolled ECAPA embedding, emits per-~50ms `is_target`/`confidence`/`rms`; demote the rolling-window ECAPA gate + `DecisionSmoother` to an off-hot-path safety-net + session-hijack detector; add verify-before-serve (a holdout embedding captured *before* `finalize_enrollment` deletes utterances); harden enrollment; and replace the permanent M-of-N eject with a de-risked WARN→EJECT→IDLE lockout.

**Architecture:** FireRedChat's pVAD is plain PyTorch (causal conv + GRU, ECAPA-conditioning fusion) packaged only as a LiveKit plugin. We **vendor the bare model + checkpoint** and drive it frame-by-frame on CPU, maintaining causal-conv + GRU hidden state across chunks. **The whole feasibility risk lives in Task 1**, a combined hot-path latency spike that is a hard cutover gate: if the bare-model load is blocked OR the combined per-chunk reflex budget exceeds 100 ms p95 on GB10 CPU under live gemma GPU load, the plan's degraded path is **FOCUS = RMS-proximity-gate only** (a named outcome, not a silent gap). Every subsequent task assumes the spike passed; the crash-fallback `is_target := rms >= proximity_rms` and the degraded mode are the same code path, so the fallback is always present.

**Tech Stack:** Python 3.12 (`python3` — there is no `python` on PATH), PyTorch (CPU inference, `torch.load(map_location="cpu")`), NumPy, pytest. Reuses `core/speaker/embedder.py` (ECAPA, 192-dim), `core/speaker/decision_smoother.py`, `core/speaker/enrollment_store.py`, and the Plan 02 Ingestion worker + the Plan 01/02 `DirectorHandoff`/`Context` types. No new third-party runtime dependency beyond the vendored model file (PyTorch is already present).

## Global Constraints

- Target/dev box: NVIDIA DGX Spark GB10 (Grace-Blackwell), aarch64, ONE GPU time-shared, ~128GB unified memory, Python 3.12. Run everything with `python3` / `python3 -m pytest`.
- **pVAD placement is CPU-only** (spec §9 placement rule): the model loads with `torch.load(..., map_location="cpu")` and `model.eval()`; it must never touch the GPU (insulated from gemma contention).
- **Bare-model only:** vendor the FireRedChat pVAD `nn.Module` definition + checkpoint from `FireRedTeam/FireRedChat-pvad`; **the LiveKit plugin is bypassed entirely** — do not add a `livekit-plugins-*` dependency.
- **NeMo Streaming Sortformer is explicitly REJECTED** (spec §7): it does anonymous diarization (no enrollment conditioning) and depends on torchaudio, which has no working aarch64-CUDA build on GB10. Do not introduce it.
- **Binding contract (Plan 02):** the Ingestion worker attaches `is_target: bool` and `speaker_score: float` to `SegmentEndpointed` / `NearFieldOnset` / `InterjectionSegment`. In Plan 02 `is_target` is hard-coded `True`; **this plan makes it real** via the pVAD worker and demotes the rolling-window ECAPA to safety-net + hijack detector.
- **Verify-before-serve:** `finalize_enrollment` deletes the per-utterance file (`os.remove(utt_path)`, `core/speaker/enrollment_store.py:99`), so the holdout embedding MUST be captured *before* finalize and carried on `DirectorHandoff.holdout_embedding`. No change to `finalize_enrollment` semantics is made in V1.
- **ECAPA short-segment unreliability is load-bearing** (MEMORY `ecapa-short-segment-unreliable.md`): ECAPA stays entirely off the synchronous hot path (108 ms p95, `run_in_executor`); it embeds only accumulated **`is_target`** audio every `verify_window_ms` (2000 ms, `config.yaml:103`).
- **De-risked lockout — never a permanent lockout:** first M-of-N miss → WARN (duck only), EJECT only on **two consecutive failed windows AND a failed RMS proximity check**, then IDLE (accept a fresh wake) after 5 s of no near-field RMS.
- **Crash-fallback:** if the pVAD worker dies, `is_target := rms >= proximity_rms`. This is the same predicate as the spike-failure degraded mode.
- New pVAD code lives under `modes/director/pvad/`; tests under `tests/director/pvad/`. Vendored model under `vendor/firered_pvad/`. The spike harness is `bench/pvad_contention.py`; the spike note is `docs/notes/2026-06-22-pvad.md`.

---

## File Structure

- `bench/pvad_contention.py` — **Task 1** spike harness: load the vendored bare pVAD, drive it frame-by-frame, and measure the COMBINED per-chunk reflex budget (Silero + pVAD + Smart Turn + `classify_interjection`) on GB10 CPU under live gemma GPU load. Self-bootstraps `sys.path` (mirrors `bench/reflex_contention.py:42`).
- `vendor/firered_pvad/__init__.py` — vendor package marker.
- `vendor/firered_pvad/model.py` — the bare `PvadModel(nn.Module)` definition (causal conv + GRU + ECAPA fusion) ported from `FireRedTeam/FireRedChat-pvad`, with a streaming `forward_step` that threads conv ring + GRU hidden state.
- `vendor/firered_pvad/README.md` — provenance: upstream repo, commit, license (Apache-2.0), and the exact checkpoint filename + sha256.
- `vendor/firered_pvad/pvad.pt` — the vendored checkpoint (binary; downloaded in Task 2, not committed by hand).
- `modes/director/pvad/__init__.py` — package marker.
- `modes/director/pvad/loader.py` — `load_pvad(ckpt_path) -> PvadModel`; `torch.load(map_location="cpu")` + `eval()`.
- `modes/director/pvad/frontend.py` — `MelFrontEnd`: streaming 16 kHz → log-mel frames matching the model's front-end (causal, ring-buffered).
- `modes/director/pvad/stream.py` — `VADStream`: the stateful per-session driver. `update_speaker(embedding)`, `push(chunk) -> list[SpeakerFrame]`, `reset()`. Holds the mel ring, conv ring, and GRU hidden state.
- `modes/director/pvad/worker.py` — `PvadWorker`: wraps `VADStream`, owns the supervised-task lifecycle, emits `SpeakerFrame` events, and implements the crash-fallback `is_target := rms >= proximity_rms`.
- `modes/director/pvad/types.py` — `SpeakerFrame` frozen dataclass `(ts, is_target, confidence, rms)`.
- `modes/director/safety_net.py` — `SafetyNet`: accumulates only `is_target` audio, embeds off-hot-path every `verify_window_ms`, runs the `DecisionSmoother` M-of-N, and emits `WARN`/`EJECT`/`IDLE` lockout decisions (de-risked).
- `core/speaker/enrollment_store.py` — **modify**: add `holdout_utterance_embedding(user_id)` (read one utterance embedding *before* finalize) — read-only, does not change `finalize_enrollment`.
- `config.yaml` — **modify**: raise `core.speaker.enrollment_min_self_similarity` 0.6 → 0.80 (with comment); add `kiosk.talkback.verify_before_serve_threshold: 0.80`; add `kiosk.talkback.lockout_idle_after_s: 5`.
- `tests/director/pvad/` + `tests/director/test_safety_net.py` + `tests/director/test_verify_before_serve.py` — test modules (one per task).

**Dependency notes between this plan and neighbors:**
- **Consumes from Plan 01:** `Context` (the session blackboard), `State` enum (`IDLE`, `LISTENING`, `SPEAKING`, `EVALUATING`), `DirectorConfig`.
- **Consumes from Plan 02:** the Ingestion worker, the event bus, and the event types `SegmentEndpointed` / `NearFieldOnset` / `InterjectionSegment` (each already carrying `is_target: bool` and `speaker_score: float`, hard-coded `True`/`1.0` until this plan).
- **Produces for the Director reducer:** `SpeakerFrame` events on the bus; a real `is_target` on every Ingestion event; `SafetyNet` lockout decisions (`WARN`/`EJECT`/`IDLE`); a verify-before-serve gate at session start.

---

## Task 1: pVAD LOAD + COMBINED HOT-PATH LATENCY SPIKE (CUTOVER GATE)

> This is the feasibility gate for the entire plan. It is **not** TDD — it is a runnable measurement script plus a recorded verdict. If it fails, **STOP** and ship the degraded path (named at the end of this task); do not start Task 3+.

**Files:**
- Create: `bench/pvad_contention.py`
- Create: `vendor/firered_pvad/__init__.py`, `vendor/firered_pvad/model.py`, `vendor/firered_pvad/README.md`
- Create (download, Task 2 also uses it): `vendor/firered_pvad/pvad.pt`
- Create: `docs/notes/2026-06-22-pvad.md` (the verdict record)

**Interfaces:**
- Produces: `vendor.firered_pvad.model.PvadModel` (bare `nn.Module`, streaming `forward_step`); the recorded GO/NO-GO verdict in `docs/notes/2026-06-22-pvad.md`.
- Consumes: `core.speaker.embedder.EmbeddingExtractor` (for an ECAPA conditioning vector), `modes.talkback.endpointing.SmartTurnDetector`, `modes.talkback.intent.classify_interjection`, `core.vad.silero_vad.SileroVAD`.

- [ ] **Step 1: Vendor the bare pVAD model definition**

Port the FireRedChat pVAD `nn.Module` from `FireRedTeam/FireRedChat-pvad` into `vendor/firered_pvad/model.py`, **bypassing the LiveKit plugin**. The architecture is: log-mel → causal 1-D conv stack → concat the 192-dim enrolled ECAPA embedding (broadcast over time) → GRU → linear → per-frame "is target speaking" logit. Provide BOTH a batch `forward` (for load-verification) and a streaming `forward_step` that threads the causal-conv ring buffer and GRU hidden state so it can be driven frame-by-frame.

```python
# vendor/firered_pvad/model.py
"""Vendored FireRedChat personal-VAD (pVAD) bare model.

Ported from FireRedTeam/FireRedChat-pvad (Apache-2.0). The upstream package
ships ONLY as a LiveKit plugin; the model underneath is plain PyTorch. We
vendor the nn.Module + checkpoint and bypass the plugin entirely so it runs
in-process on CPU on GB10 (aarch64). See vendor/firered_pvad/README.md for
the exact upstream commit, checkpoint filename, and sha256.

Mechanism (spec section 7): mel -> causal conv -> concat enrolled ECAPA
embedding -> GRU -> per-frame target-speaker probability.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1-D conv that only sees past + current frames (no lookahead), so it can
    be driven frame-by-frame in a streaming loop with a left-context ring."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.pad = kernel - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). Left-pad by (kernel-1) so output length == input length
        # and frame t depends only on frames <= t.
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class PvadModel(nn.Module):
    """ECAPA-conditioned personal VAD.

    Args mirror the vendored checkpoint's hyper-params (see README provenance).
    n_mels: log-mel bins; spk_dim: ECAPA embedding dim (192); conv/gru sizes
    are fixed by the checkpoint.
    """

    def __init__(
        self,
        n_mels: int = 80,
        spk_dim: int = 192,
        conv_ch: int = 64,
        conv_kernel: int = 5,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.spk_dim = spk_dim
        self.conv_kernel = conv_kernel
        self.conv_ch = conv_ch
        self.gru_hidden = gru_hidden

        self.conv1 = CausalConv1d(n_mels, conv_ch, conv_kernel)
        self.conv2 = CausalConv1d(conv_ch, conv_ch, conv_kernel)
        self.act = nn.ReLU()
        # GRU input = conv features (conv_ch) concatenated with the speaker emb.
        self.gru = nn.GRU(conv_ch + spk_dim, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, mel: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """Batch path (load-verification / offline scoring).

        mel: (B, T, n_mels). spk: (B, spk_dim). Returns (B, T) probabilities.
        """
        x = mel.transpose(1, 2)              # (B, n_mels, T)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))          # (B, conv_ch, T)
        x = x.transpose(1, 2)                # (B, T, conv_ch)
        T = x.shape[1]
        spk_b = spk.unsqueeze(1).expand(-1, T, -1)   # (B, T, spk_dim)
        gru_in = torch.cat([x, spk_b], dim=-1)
        y, _ = self.gru(gru_in)
        return torch.sigmoid(self.head(y)).squeeze(-1)   # (B, T)

    def init_state(
        self, spk: torch.Tensor
    ) -> "PvadState":
        """Build the per-session streaming state bound to one speaker emb."""
        return PvadState(
            spk=spk,
            conv_ring=torch.zeros(1, self.n_mels, self.conv_kernel - 1),
            conv2_ring=torch.zeros(1, self.conv_ch, self.conv_kernel - 1),
            gru_h=torch.zeros(1, 1, self.gru_hidden),
        )

    @torch.no_grad()
    def forward_step(
        self, mel_frame: torch.Tensor, state: "PvadState"
    ) -> Tuple[float, "PvadState"]:
        """Streaming: consume ONE mel frame (1, n_mels), return (prob, state).

        Threads the causal-conv left-context rings and the GRU hidden state so
        successive calls are equivalent to the batch forward over the stream.
        """
        # conv1 over [ring | frame]
        x = mel_frame.unsqueeze(0).transpose(1, 2)           # (1, n_mels, 1)
        c1_in = torch.cat([state.conv_ring, x], dim=2)       # (1, n_mels, K)
        c1 = self.act(self.conv1.conv(c1_in))                # (1, conv_ch, 1)
        new_conv_ring = c1_in[:, :, 1:]                      # slide ring

        c2_in = torch.cat([state.conv2_ring, c1], dim=2)     # (1, conv_ch, K)
        c2 = self.act(self.conv2.conv(c2_in))                # (1, conv_ch, 1)
        new_conv2_ring = c2_in[:, :, 1:]

        feat = c2.transpose(1, 2)                            # (1, 1, conv_ch)
        gru_in = torch.cat([feat, state.spk.view(1, 1, -1)], dim=-1)
        y, new_h = self.gru(gru_in, state.gru_h)
        prob = torch.sigmoid(self.head(y)).item()
        new_state = PvadState(
            spk=state.spk,
            conv_ring=new_conv_ring,
            conv2_ring=new_conv2_ring,
            gru_h=new_h,
        )
        return prob, new_state


class PvadState:
    """Mutable-by-replacement streaming state (rings + GRU hidden)."""

    __slots__ = ("spk", "conv_ring", "conv2_ring", "gru_h")

    def __init__(self, spk, conv_ring, conv2_ring, gru_h):
        self.spk = spk
        self.conv_ring = conv_ring
        self.conv2_ring = conv2_ring
        self.gru_h = gru_h
```

> **NOTE on fidelity:** the conv/GRU dims above are the *expected* FireRedChat shapes; when you download the real checkpoint (next step) you MUST reconcile `state_dict` keys/shapes against this module. If upstream differs (extra conv layer, LayerNorm, different `n_mels`), edit `model.py` to match the checkpoint *exactly* — a `load_state_dict(strict=True)` mismatch is a fatal load error and a NO-GO for the bare-model-load half of the gate. Record the reconciled shapes in the README.

- [ ] **Step 2: Download the checkpoint and write provenance**

```bash
mkdir -p vendor/firered_pvad
# Pull the bare checkpoint from the HF model repo (NOT the LiveKit plugin).
python3 - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, hashlib, os
# Reconcile the exact filename against the repo's file list if this 404s.
p = hf_hub_download(repo_id="FireRedTeam/FireRedChat-pvad", filename="pvad.pt")
dst = "vendor/firered_pvad/pvad.pt"
shutil.copy(p, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print("sha256", h, "bytes", os.path.getsize(dst))
PY
```

Write `vendor/firered_pvad/README.md` recording: upstream repo URL + commit hash, the exact checkpoint filename, its sha256 + byte size (from the command above), license (Apache-2.0), and the reconciled `n_mels`/`conv_ch`/`conv_kernel`/`gru_hidden` you pinned in `model.py`.

```python
# vendor/firered_pvad/__init__.py
"""Vendored FireRedChat pVAD bare model (Apache-2.0). See README.md."""
```

- [ ] **Step 3: Verify the bare-model load works (load half of the gate)**

```bash
python3 - <<'PY'
import torch
from vendor.firered_pvad.model import PvadModel

ckpt = torch.load("vendor/firered_pvad/pvad.pt", map_location="cpu")
state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
model = PvadModel()
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)
model.eval()
# Smoke: drive 100 frames of zeros + a random ECAPA-shaped emb, assert finite.
spk = torch.randn(1, 192); spk = spk / spk.norm()
st = model.init_state(spk)
import math
for _ in range(100):
    p, st = model.forward_step(torch.zeros(1, 80), st)
    assert 0.0 <= p <= 1.0 and not math.isnan(p), p
print("LOAD+STREAM OK")
PY
```

Expected: `LOAD+STREAM OK` with EMPTY `missing`/`unexpected` (after you reconcile shapes in Step 1). **If keys mismatch, fix `model.py` until `strict=False` reports no missing/unexpected and the smoke loop passes** — otherwise the load half is a NO-GO.

- [ ] **Step 4: Write the combined hot-path spike harness**

Model the structure on `bench/reflex_contention.py` (self-bootstrap `sys.path` at top; prefer the real gemma llama.cpp server on `127.0.0.1:8080` as GPU load, fall back to a synthetic CUDA matmul loop). The new measurement is the **COMBINED per-chunk reflex budget on ONE chunk**: Silero `is_speaking` + pVAD `forward_step` over the chunk's mel frames + Smart Turn `endpoint_prob` + `classify_interjection`, summed, measured p50/p95 IDLE and UNDER LOAD.

```python
# bench/pvad_contention.py
"""bench/pvad_contention.py — COMBINED reflex hot-path latency (Director cutover gate).

Measures the per-chunk combined reflex budget on GB10 CPU:
    Silero is_speaking  +  pVAD forward_step (all frames in the chunk)
    +  Smart Turn endpoint_prob  +  classify_interjection
on ONE 200ms chunk, p50/p95 over >=30 timed iters (3 warmup discarded),
IDLE and UNDER live gemma GPU load. pVAD runs CPU-only.

ACCEPTANCE (cutover GO): combined p95 < 100ms UNDER LOAD *and* the bare pVAD
checkpoint loads. NO-GO -> degraded path: FOCUS = RMS-proximity-gate only.

Usage:  python3 bench/pvad_contention.py
Env:    BENCH_N (default 35; first 3 warmup)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import numpy as np
import torch

SR = 16_000
CHUNK_200MS = int(SR * 0.200)
N_TOTAL = int(os.environ.get("BENCH_N", 35))
N_WARMUP = 3
LLM_URL = "http://127.0.0.1:8080/v1/completions"


def p50_p95(times):
    a = np.array(times, dtype=float)
    return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def _llm_reachable():
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _gemma_load_loop(stop):
    import urllib.request, json
    payload = json.dumps({"prompt": "Tell me a one-sentence fact about space.",
                          "max_tokens": 64, "temperature": 0.7}).encode()
    headers = {"Content-Type": "application/json"}
    while not stop.is_set():
        try:
            req = urllib.request.Request(LLM_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30):
                pass
        except Exception:
            pass


def _cuda_matmul_loop(stop):
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    while not stop.is_set():
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        del C


def build_reflex():
    """Construct the four reflex components + a closure timing one combined chunk."""
    from vendor.firered_pvad.model import PvadModel
    from modes.director.pvad.frontend import MelFrontEnd
    from modes.talkback.endpointing import SmartTurnDetector
    from modes.talkback.intent import classify_interjection
    from core.vad.silero_vad import SileroVAD

    ckpt = torch.load("vendor/firered_pvad/pvad.pt", map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    pvad = PvadModel()
    pvad.load_state_dict(state, strict=False)
    pvad.eval()

    spk = torch.randn(1, 192); spk = spk / spk.norm()
    pstate = pvad.init_state(spk)
    front = MelFrontEnd()
    smart = SmartTurnDetector()
    silero = SileroVAD()

    chunk = np.zeros(CHUNK_200MS, dtype=np.float32)

    def combined():
        nonlocal pstate
        # 1) Silero onset
        silero.process_chunk(chunk)
        _ = getattr(silero, "is_speaking", False)
        # 2) pVAD over every mel frame in the chunk (streaming)
        for f in front.push(chunk):
            _p, pstate = pvad.forward_step(torch.from_numpy(f).unsqueeze(0), pstate)
        # 3) Smart Turn endpoint
        smart.endpoint_prob(chunk, sample_rate=SR)
        # 4) classify (pure, ~us)
        classify_interjection("why")

    return combined


def bench(fn, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    print("=" * 70)
    print("  COMBINED reflex hot-path — GB10 CPU (Director cutover gate)")
    print("=" * 70)

    load_fn = _gemma_load_loop if _llm_reachable() else _cuda_matmul_loop
    load_name = ("real gemma-3-4b-it (llama.cpp, GPU)" if _llm_reachable()
                 else "synthetic CUDA matmul (4096x4096 fp16)")
    print(f"  Load source: {load_name}\n")

    try:
        combined = build_reflex()
    except Exception as e:
        print(f"  NO-GO (build failed): {e}")
        print("  DEGRADED PATH: FOCUS = RMS-proximity-gate only.")
        return

    idle = bench(combined, N_TOTAL)[N_WARMUP:]
    ip50, ip95 = p50_p95(idle)
    print(f"  IDLE      combined p50={ip50:.1f}ms p95={ip95:.1f}ms")

    stop = threading.Event()
    t = threading.Thread(target=load_fn, args=(stop,), daemon=True)
    t.start(); time.sleep(1)
    loaded = bench(combined, N_TOTAL)[N_WARMUP:]
    stop.set(); t.join(timeout=5)
    lp50, lp95 = p50_p95(loaded)
    print(f"  UNDER LOAD combined p50={lp50:.1f}ms p95={lp95:.1f}ms\n")

    print("  GO / NO-GO (combined reflex < 100ms p95 under load)")
    if lp95 < 100.0:
        print(f"  GO — combined p95={lp95:.1f}ms under {load_name}.")
        print("  -> proceed to Task 3 (PvadWorker wiring).")
    else:
        print(f"  NO-GO — combined p95={lp95:.1f}ms EXCEEDS 100ms.")
        print("  DEGRADED PATH: FOCUS = RMS-proximity-gate only (named outcome).")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

> `MelFrontEnd` (referenced here) is built in Task 4; for the spike you may inline a minimal `torchaudio`-free log-mel (`np.fft`-based) front-end if Task 4 is not yet done — the spike only needs *a* mel of the right `n_mels` to load the pVAD frame loop. Keep the inline version in a local function so the bench remains runnable standalone.

- [ ] **Step 5: Run the spike and record the verdict**

Run: `python3 bench/pvad_contention.py`

Write `docs/notes/2026-06-22-pvad.md` recording BOTH halves of the gate:
1. **Bare-model load:** PASS/FAIL, with the reconciled `state_dict` key/shape result from Step 3.
2. **Combined latency:** the IDLE + UNDER-LOAD p50/p95 numbers and the load source.
3. **Verdict:** GO (load OK AND combined p95 < 100 ms under load) or **NO-GO**.

**Acceptance / gate:** GO requires the bare-model load to work AND combined p95 < 100 ms under live gemma load.

> **DEGRADED PATH (NO-GO outcome — a NAMED deliverable, not a gap).** If either half fails, **do not build Tasks 3-6's pVAD worker.** Instead ship **FOCUS = RMS-proximity-gate only**: the Ingestion worker sets `is_target := rms >= proximity_rms` directly (the same predicate as the crash-fallback in Task 5, Step 3), the rolling-window ECAPA safety-net + de-risked lockout (Tasks 6-7) and verify-before-serve + enrollment hardening (Tasks 8-9) still ship unchanged (they do not depend on the pVAD), and `docs/notes/2026-06-22-pvad.md` records the degraded mode as the V1 shipping decision. Tasks 6-9 are explicitly **spike-independent** and proceed in both outcomes.

- [ ] **Step 6: Commit**

```bash
git add bench/pvad_contention.py vendor/firered_pvad/ docs/notes/2026-06-22-pvad.md
git commit -m "spike: pVAD bare-model load + combined reflex hot-path gate (GB10 CPU)"
```

---

## Task 2: pVAD loader (GO path)

> Tasks 2-5 assume the Task 1 spike returned **GO**. Skip to Task 6 on NO-GO.

**Files:**
- Create: `modes/director/pvad/__init__.py`, `modes/director/pvad/loader.py`
- Create: `modes/director/pvad/types.py`
- Test: `tests/director/pvad/__init__.py`, `tests/director/pvad/test_loader.py`

**Interfaces:**
- Consumes: `vendor.firered_pvad.model.PvadModel`.
- Produces: `SpeakerFrame(ts: float, is_target: bool, confidence: float, rms: float)` (frozen dataclass); `load_pvad(ckpt_path: str) -> PvadModel` (CPU, `eval()`).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/pvad/test_loader.py
import os
import torch
import pytest
from modes.director.pvad.loader import load_pvad
from modes.director.pvad.types import SpeakerFrame

CKPT = "vendor/firered_pvad/pvad.pt"


def test_speaker_frame_is_frozen():
    f = SpeakerFrame(ts=1.0, is_target=True, confidence=0.9, rms=0.05)
    assert (f.ts, f.is_target, f.confidence, f.rms) == (1.0, True, 0.9, 0.05)
    with pytest.raises(Exception):
        f.is_target = False


@pytest.mark.skipif(not os.path.exists(CKPT), reason="pVAD checkpoint not vendored")
def test_load_pvad_is_cpu_eval():
    model = load_pvad(CKPT)
    assert not model.training            # eval() called
    p = next(model.parameters())
    assert p.device.type == "cpu"        # never on GPU
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/pvad/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.pvad'`

- [ ] **Step 3: Write the loader + types**

```python
# modes/director/pvad/__init__.py
"""Director pVAD crowd-focus worker package."""
```
```python
# tests/director/pvad/__init__.py
```
```python
# modes/director/pvad/types.py
"""pVAD output type emitted onto the event bus."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerFrame:
    """One aggregated ~50ms target-speaker decision.

    is_target: enrolled speaker speaking now? confidence: mean pVAD prob over
    the aggregated frames. rms: chunk RMS (used by the proximity/near-field
    gate and the crash-fallback).
    """
    ts: float
    is_target: bool
    confidence: float
    rms: float
```
```python
# modes/director/pvad/loader.py
"""Load the vendored FireRedChat pVAD bare model on CPU."""

import torch

from vendor.firered_pvad.model import PvadModel


def load_pvad(ckpt_path: str) -> PvadModel:
    """torch.load(map_location='cpu') + load_state_dict + eval(). CPU-only."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = PvadModel()
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/pvad/test_loader.py -v`
Expected: PASS (the checkpoint test is skipped if `pvad.pt` is absent; the frozen-dataclass test always runs).

- [ ] **Step 5: Commit**

```bash
git add modes/director/pvad/__init__.py modes/director/pvad/loader.py modes/director/pvad/types.py tests/director/pvad/
git commit -m "feat(pvad): CPU bare-model loader + SpeakerFrame type"
```

---

## Task 3: Streaming mel front-end

**Files:**
- Create: `modes/director/pvad/frontend.py`
- Test: `tests/director/pvad/test_frontend.py`

**Interfaces:**
- Produces: `MelFrontEnd(n_mels=80, sr=16000, frame_ms=25, hop_ms=10)`; `push(chunk: np.ndarray) -> list[np.ndarray]` (each item is a `(n_mels,)` float32 log-mel frame); `reset()`. Stateful: buffers the tail of `chunk` so frame boundaries are continuous across `push` calls. **No torchaudio** (aarch64-CUDA gap, spec §7) — pure NumPy.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/pvad/test_frontend.py
import numpy as np
from modes.director.pvad.frontend import MelFrontEnd

SR = 16000


def test_frame_count_matches_hop():
    fe = MelFrontEnd(n_mels=80, sr=SR, frame_ms=25, hop_ms=10)
    # 200ms chunk @ 10ms hop ~= 20 frames after the first window fills.
    frames = fe.push(np.zeros(int(SR * 0.200), dtype=np.float32))
    assert 17 <= len(frames) <= 20
    assert all(f.shape == (80,) for f in frames)
    assert all(f.dtype == np.float32 for f in frames)


def test_streaming_is_continuous():
    # Two 100ms pushes must yield ~the same total frames as one 200ms push,
    # because the front-end buffers the inter-chunk tail (no dropped audio).
    one = MelFrontEnd()
    n_single = len(one.push(np.zeros(int(SR * 0.200), dtype=np.float32)))
    split = MelFrontEnd()
    a = len(split.push(np.zeros(int(SR * 0.100), dtype=np.float32)))
    b = len(split.push(np.zeros(int(SR * 0.100), dtype=np.float32)))
    assert abs((a + b) - n_single) <= 1


def test_reset_clears_buffer():
    fe = MelFrontEnd()
    fe.push(np.zeros(int(SR * 0.050), dtype=np.float32))
    fe.reset()
    # After reset the leftover partial-window samples are gone.
    assert fe._buf.size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/pvad/test_frontend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.pvad.frontend'`

- [ ] **Step 3: Implement the front-end**

```python
# modes/director/pvad/frontend.py
"""Streaming log-mel front-end for the pVAD (pure NumPy, no torchaudio).

Buffers the inter-chunk tail so successive push() calls produce a continuous
frame stream (no dropped or duplicated audio across chunk boundaries). Frame
shape and n_mels MUST match the vendored pVAD checkpoint's front-end (see
vendor/firered_pvad/README.md).
"""

import numpy as np


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels, n_fft, sr):
    f_min, f_max = 0.0, sr / 2.0
    mel_pts = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return fb


class MelFrontEnd:
    def __init__(self, n_mels=80, sr=16000, frame_ms=25, hop_ms=10):
        self.n_mels = n_mels
        self.sr = sr
        self.win = int(sr * frame_ms / 1000)        # 400 samples @ 25ms
        self.hop = int(sr * hop_ms / 1000)          # 160 samples @ 10ms
        self.n_fft = 1
        while self.n_fft < self.win:
            self.n_fft *= 2                          # 512
        self._fb = _mel_filterbank(n_mels, self.n_fft, sr)
        self._window = np.hanning(self.win).astype(np.float32)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> list:
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
        frames = []
        while self._buf.size >= self.win:
            seg = self._buf[:self.win] * self._window
            spec = np.abs(np.fft.rfft(seg, n=self.n_fft)) ** 2
            mel = self._fb @ spec                   # (n_mels,)
            log_mel = np.log(mel + 1e-6).astype(np.float32)
            frames.append(log_mel)
            self._buf = self._buf[self.hop:]
        return frames

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/pvad/test_frontend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modes/director/pvad/frontend.py tests/director/pvad/test_frontend.py
git commit -m "feat(pvad): streaming NumPy log-mel front-end"
```

---

## Task 4: VADStream — stateful per-session driver

**Files:**
- Create: `modes/director/pvad/stream.py`
- Test: `tests/director/pvad/test_stream.py`

**Interfaces:**
- Consumes: `PvadModel` (Task 1), `MelFrontEnd` (Task 3), `SpeakerFrame` (Task 2).
- Produces: `VADStream(model, *, sr=16000, agg_ms=50, threshold=0.5)`; `update_speaker(embedding: np.ndarray) -> None` (192-dim ECAPA, L2-normed; rebuilds streaming state); `push(chunk: np.ndarray, ts: float) -> list[SpeakerFrame]` (aggregates per-frame probs to ~`agg_ms`); `reset() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/pvad/test_stream.py
import numpy as np
import pytest
from modes.director.pvad.stream import VADStream
from modes.director.pvad.types import SpeakerFrame


class _FakeModel:
    """Stand-in pVAD: returns a fixed prob per frame; records the speaker emb."""
    def __init__(self, prob):
        self._prob = prob
        self.spk = None
    def init_state(self, spk):
        self.spk = spk
        return {"spk": spk}
    def forward_step(self, mel_frame, state):
        return self._prob, state


def test_update_speaker_required_before_push():
    vs = VADStream(_FakeModel(0.9))
    with pytest.raises(RuntimeError):
        vs.push(np.zeros(3200, dtype=np.float32), ts=0.0)


def test_high_prob_is_target_true():
    vs = VADStream(_FakeModel(0.9), agg_ms=50, threshold=0.5)
    vs.update_speaker(np.ones(192, dtype=np.float32))
    out = vs.push(np.ones(int(16000 * 0.200), dtype=np.float32), ts=1.0)
    assert out and all(isinstance(f, SpeakerFrame) for f in out)
    assert all(f.is_target for f in out)
    assert all(f.confidence == pytest.approx(0.9) for f in out)
    assert all(f.rms > 0.0 for f in out)


def test_low_prob_is_target_false():
    vs = VADStream(_FakeModel(0.1), agg_ms=50, threshold=0.5)
    vs.update_speaker(np.ones(192, dtype=np.float32))
    out = vs.push(np.ones(int(16000 * 0.200), dtype=np.float32), ts=1.0)
    assert out and all(not f.is_target for f in out)


def test_aggregation_groups_frames():
    # 200ms @ 10ms hop ~= ~20 frames; aggregated to 50ms -> ~4 SpeakerFrames.
    vs = VADStream(_FakeModel(0.9), agg_ms=50, threshold=0.5)
    vs.update_speaker(np.ones(192, dtype=np.float32))
    out = vs.push(np.ones(int(16000 * 0.200), dtype=np.float32), ts=0.0)
    assert 3 <= len(out) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/pvad/test_stream.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.pvad.stream'`

- [ ] **Step 3: Implement VADStream**

```python
# modes/director/pvad/stream.py
"""Per-session pVAD driver: ECAPA-conditioned, frame-by-frame, aggregated.

update_speaker(emb) binds the enrolled embedding and (re)builds streaming
state. push(chunk, ts) runs the mel front-end + pVAD forward_step per frame,
aggregates per-frame probabilities to ~agg_ms, and emits SpeakerFrame events.
"""

import numpy as np
import torch

from .frontend import MelFrontEnd
from .types import SpeakerFrame


class VADStream:
    def __init__(self, model, *, sr=16000, agg_ms=50, hop_ms=10,
                 n_mels=80, threshold=0.5):
        self._model = model
        self._sr = sr
        self._threshold = threshold
        self._frontend = MelFrontEnd(n_mels=n_mels, sr=sr, hop_ms=hop_ms)
        self._frames_per_agg = max(1, round(agg_ms / hop_ms))
        self._state = None

    def update_speaker(self, embedding: np.ndarray) -> None:
        """Bind the enrolled ECAPA embedding (192-dim) and reset streaming state."""
        spk = torch.from_numpy(np.asarray(embedding, dtype=np.float32)).view(1, -1)
        norm = spk.norm()
        if float(norm) > 0:
            spk = spk / norm
        self._state = self._model.init_state(spk)
        self._frontend.reset()

    def reset(self) -> None:
        self._frontend.reset()
        if self._state is not None:
            self._state = None

    def push(self, chunk: np.ndarray, ts: float) -> list:
        if self._state is None:
            raise RuntimeError("update_speaker() must be called before push()")
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        probs = []
        for mel in self._frontend.push(chunk):
            mel_t = torch.from_numpy(mel).unsqueeze(0)   # (1, n_mels)
            p, self._state = self._model.forward_step(mel_t, self._state)
            probs.append(float(p))

        out = []
        n = self._frames_per_agg
        for i in range(0, len(probs), n):
            group = probs[i:i + n]
            if not group:
                continue
            conf = sum(group) / len(group)
            out.append(SpeakerFrame(
                ts=ts, is_target=conf >= self._threshold,
                confidence=conf, rms=rms,
            ))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/pvad/test_stream.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modes/director/pvad/stream.py tests/director/pvad/test_stream.py
git commit -m "feat(pvad): VADStream session driver with ECAPA conditioning + aggregation"
```

---

## Task 5: PvadWorker — supervised lifecycle + crash-fallback

**Files:**
- Create: `modes/director/pvad/worker.py`
- Test: `tests/director/pvad/test_worker.py`

**Interfaces:**
- Consumes: `VADStream` (Task 4), `SpeakerFrame` (Task 2).
- Produces: `PvadWorker(stream, proximity_rms: float, emit)`; `update_speaker(embedding)`; `process(chunk, ts) -> list[SpeakerFrame]` — runs `VADStream.push`, but on ANY exception falls back to the crash-fallback `is_target := rms >= proximity_rms` (one synthetic `SpeakerFrame`) and emits a `worker_failed` event. Pure (no asyncio) so it is unit-testable; the Plan 02 Ingestion worker calls `process` in its loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/pvad/test_worker.py
import numpy as np
from modes.director.pvad.worker import PvadWorker
from modes.director.pvad.types import SpeakerFrame


class _OkStream:
    def update_speaker(self, emb):
        self.emb = emb
    def push(self, chunk, ts):
        return [SpeakerFrame(ts=ts, is_target=True, confidence=0.9, rms=0.1)]


class _BoomStream:
    def update_speaker(self, emb):
        pass
    def push(self, chunk, ts):
        raise RuntimeError("pvad died")


def test_normal_path_passes_frames_through():
    events = []
    w = PvadWorker(_OkStream(), proximity_rms=0.02, emit=lambda e, p: events.append((e, p)))
    w.update_speaker(np.ones(192, dtype=np.float32))
    out = w.process(np.ones(3200, dtype=np.float32), ts=1.0)
    assert out == [SpeakerFrame(ts=1.0, is_target=True, confidence=0.9, rms=0.1)]
    assert events == []


def test_crash_falls_back_to_rms_proximity_gate():
    events = []
    w = PvadWorker(_BoomStream(), proximity_rms=0.02, emit=lambda e, p: events.append((e, p)))
    w.update_speaker(np.ones(192, dtype=np.float32))
    # Loud chunk: rms >= proximity_rms -> is_target True via fallback.
    loud = np.ones(3200, dtype=np.float32) * 0.5
    out = w.process(loud, ts=2.0)
    assert len(out) == 1 and out[0].is_target is True
    assert any(e == "worker_failed" for e, _ in events)


def test_crash_fallback_quiet_chunk_is_not_target():
    w = PvadWorker(_BoomStream(), proximity_rms=0.2, emit=lambda e, p: None)
    w.update_speaker(np.ones(192, dtype=np.float32))
    quiet = np.ones(3200, dtype=np.float32) * 0.01   # rms < proximity_rms
    out = w.process(quiet, ts=3.0)
    assert out[0].is_target is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/pvad/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.pvad.worker'`

- [ ] **Step 3: Implement the worker**

```python
# modes/director/pvad/worker.py
"""pVAD worker: VADStream wrapper with the crash-fallback crowd filter.

If the pVAD stream raises, FOCUS degrades to the RMS proximity gate
(is_target := rms >= proximity_rms) — identical to the spike-failure degraded
mode — and a worker_failed event is emitted. The Plan 02 Ingestion worker
calls process() once per mic chunk and stamps is_target onto its events.
"""

import numpy as np

from .types import SpeakerFrame


class PvadWorker:
    def __init__(self, stream, proximity_rms: float, emit):
        self._stream = stream
        self._proximity_rms = proximity_rms
        self._emit = emit
        self._failed = False

    def update_speaker(self, embedding: np.ndarray) -> None:
        self._stream.update_speaker(embedding)

    def process(self, chunk: np.ndarray, ts: float) -> list:
        try:
            return self._stream.push(chunk, ts)
        except Exception as exc:   # noqa: BLE001 — degrade, never crash the loop
            if not self._failed:
                self._failed = True
                self._emit("worker_failed", {"worker": "pvad", "error": str(exc)})
            return [self._rms_fallback(chunk, ts)]

    def _rms_fallback(self, chunk: np.ndarray, ts: float) -> SpeakerFrame:
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        return SpeakerFrame(
            ts=ts, is_target=rms >= self._proximity_rms,
            confidence=0.0, rms=rms,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/pvad/test_worker.py -v`
Expected: PASS

- [ ] **Step 5: Wire into the Ingestion worker (make `is_target` real)**

In the Plan 02 Ingestion worker, replace the hard-coded `is_target = True` / `speaker_score = 1.0` with the `PvadWorker` output. Construct one `PvadWorker` per session (the Director passes it `proximity_rms` from `Context` and calls `update_speaker(primary_embedding)` at session start). For each mic chunk, call `frames = pvad_worker.process(chunk, ts)`; OR the chunk's `is_target` from those frames (any aggregated frame `is_target=True` → chunk is target) and set `speaker_score = max(confidence)`. Emit the `SpeakerFrame`s on the bus for the SafetyNet (Task 6).

```python
# In the Ingestion worker chunk loop (Plan 02 module), replacing the stub:
frames = self._pvad.process(chunk, ts=now)
is_target = any(f.is_target for f in frames)
speaker_score = max((f.confidence for f in frames), default=0.0)
for f in frames:
    self._emit_event(SpeakerFrame_event(f))   # for the SafetyNet
# ...attach is_target / speaker_score to SegmentEndpointed / NearFieldOnset /
# InterjectionSegment exactly where Plan 02 hard-coded True.
```

> The exact attribute names (`self._pvad`, `self._emit_event`) follow Plan 02's Ingestion-worker conventions; match them. The contract that matters: `is_target` is now derived from `PvadWorker.process`, not a constant.

- [ ] **Step 6: Commit**

```bash
git add modes/director/pvad/worker.py tests/director/pvad/test_worker.py
git commit -m "feat(pvad): PvadWorker with RMS crash-fallback; wire real is_target into Ingestion"
```

---

## Task 6: SafetyNet — demoted ECAPA rolling-window (off hot path)

> **Spike-independent:** ships in BOTH the GO and NO-GO (degraded) outcomes — it depends only on accumulated `is_target` audio + ECAPA, not on the pVAD model.

**Files:**
- Create: `modes/director/safety_net.py`
- Test: `tests/director/test_safety_net.py`

**Interfaces:**
- Consumes: `core.speaker.embedder.EmbeddingExtractor`, `core.speaker.decision_smoother.DecisionSmoother`, `SpeakerFrame` (for the `is_target` filter).
- Produces: `SafetyNet(embedder, primary_embedding, *, verify_window_ms=2000, threshold=0.30, window_size=3, min_matches=1, sr=16000)`; `accumulate(audio: np.ndarray, is_target: bool) -> None` (drops non-target audio); `maybe_verify() -> Optional[SafetyVerdict]` (returns `None` until ≥ `verify_window_ms` accumulated, then embeds + scores + feeds `DecisionSmoother`); `SafetyVerdict(score: float, smoother_ok: bool)`. The embed call is the caller's to run in `run_in_executor` — `maybe_verify` is sync and pure-ish (only the embedder touches a model).

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_safety_net.py
import numpy as np
import pytest
from modes.director.safety_net import SafetyNet, SafetyVerdict


class _FakeEmbedder:
    """Returns a fixed embedding so cosine vs primary is deterministic."""
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)
    def extract(self, audio, sample_rate=16000):
        return self._vec


def _emb(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_non_target_audio_is_dropped():
    sn = SafetyNet(_FakeEmbedder(_emb(1, 0)), _emb(1, 0), verify_window_ms=100)
    sn.accumulate(np.ones(16000, dtype=np.float32), is_target=False)
    # Nothing accumulated -> never ready to verify.
    assert sn.maybe_verify() is None


def test_matching_speaker_passes_smoother():
    primary = _emb(1, 0)
    sn = SafetyNet(_FakeEmbedder(_emb(1, 0)), primary,
                   verify_window_ms=100, threshold=0.30,
                   window_size=3, min_matches=1)
    sn.accumulate(np.ones(3200, dtype=np.float32), is_target=True)  # 200ms
    v = sn.maybe_verify()
    assert isinstance(v, SafetyVerdict)
    assert v.score == pytest.approx(1.0)     # identical embeddings
    assert v.smoother_ok is True


def test_mismatched_speaker_fails_score():
    primary = _emb(1, 0)
    sn = SafetyNet(_FakeEmbedder(_emb(0, 1)), primary,    # orthogonal
                   verify_window_ms=100, threshold=0.30,
                   window_size=3, min_matches=1)
    sn.accumulate(np.ones(3200, dtype=np.float32), is_target=True)
    v = sn.maybe_verify()
    assert v.score == pytest.approx(0.0, abs=1e-6)
    assert v.smoother_ok is False            # below threshold, 0 of 3 cross


def test_buffer_resets_after_verify():
    sn = SafetyNet(_FakeEmbedder(_emb(1, 0)), _emb(1, 0), verify_window_ms=100)
    sn.accumulate(np.ones(3200, dtype=np.float32), is_target=True)
    sn.maybe_verify()
    # Window consumed -> not immediately ready again.
    assert sn.maybe_verify() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_safety_net.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.safety_net'`

- [ ] **Step 3: Implement SafetyNet**

```python
# modes/director/safety_net.py
"""Demoted ECAPA rolling-window safety-net (off the hot path).

The pVAD is primary FOCUS; this is the session-hijack detector. It accumulates
ONLY is_target audio, embeds every verify_window_ms (108ms p95 ECAPA, run by
the caller in an executor — fine off the hot path), and runs the M-of-N
DecisionSmoother to catch a different person taking over for >1 window. ECAPA
is unreliable on <2-3s segments (MEMORY: ecapa-short-segment-unreliable.md),
so a SINGLE miss never ejects — see the de-risked lockout in Task 7.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.speaker.decision_smoother import DecisionSmoother


@dataclass(frozen=True)
class SafetyVerdict:
    score: float
    smoother_ok: bool


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SafetyNet:
    def __init__(self, embedder, primary_embedding, *, verify_window_ms=2000,
                 threshold=0.30, window_size=3, min_matches=1, sr=16000):
        self._embedder = embedder
        self._primary = np.asarray(primary_embedding, dtype=np.float32)
        self._need = int(sr * verify_window_ms / 1000)
        self._sr = sr
        self._smoother = DecisionSmoother(window_size, min_matches, threshold)
        self._threshold = threshold
        self._buf = np.zeros(0, dtype=np.float32)

    def accumulate(self, audio: np.ndarray, is_target: bool) -> None:
        if not is_target:
            return                                   # drop non-target audio
        self._buf = np.concatenate([self._buf, audio.astype(np.float32)])

    def maybe_verify(self) -> Optional[SafetyVerdict]:
        if self._buf.size < self._need:
            return None
        window = self._buf[: self._need]
        self._buf = self._buf[self._need:]           # consume the window
        emb = self._embedder.extract(window, sample_rate=self._sr)
        score = _cosine(emb, self._primary)
        smoother_ok = self._smoother.update(score)
        return SafetyVerdict(score=score, smoother_ok=smoother_ok)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_safety_net.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modes/director/safety_net.py tests/director/test_safety_net.py
git commit -m "feat(director): demoted ECAPA rolling-window safety-net (off hot path)"
```

---

## Task 7: De-risked lockout state machine

> **Spike-independent.** Replaces the old permanent M-of-N eject (which locked out the real user on short-segment ECAPA noise) with WARN→EJECT→IDLE.

**Files:**
- Create: `modes/director/lockout.py`
- Test: `tests/director/test_lockout.py`

**Interfaces:**
- Consumes: `SafetyVerdict` (Task 6).
- Produces: `Lockout(idle_after_s=5.0)`; `on_verdict(verdict: SafetyVerdict, rms_ok: bool) -> LockoutAction`; `on_idle_tick(now: float, near_field_rms_active: bool) -> Optional[LockoutAction]`; `LockoutAction` enum `{NONE, WARN, EJECT, IDLE}`. Rules (spec §7): first failed window → WARN (duck only); **EJECT only on two consecutive failed windows AND a failed RMS proximity check** (`rms_ok=False`); after EJECT, if no near-field RMS for `idle_after_s` → IDLE (accept a fresh wake) — never a permanent lockout.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_lockout.py
import pytest
from modes.director.lockout import Lockout, LockoutAction
from modes.director.safety_net import SafetyVerdict


def _fail():
    return SafetyVerdict(score=0.0, smoother_ok=False)


def _pass():
    return SafetyVerdict(score=0.9, smoother_ok=True)


def test_first_miss_is_warn_not_eject():
    lk = Lockout()
    assert lk.on_verdict(_fail(), rms_ok=True) is LockoutAction.WARN


def test_two_misses_but_rms_ok_does_not_eject():
    # Two failed windows but the speaker is still near (rms_ok) -> no eject.
    lk = Lockout()
    lk.on_verdict(_fail(), rms_ok=True)
    assert lk.on_verdict(_fail(), rms_ok=True) is LockoutAction.WARN


def test_two_misses_and_failed_proximity_ejects():
    lk = Lockout()
    lk.on_verdict(_fail(), rms_ok=False)
    assert lk.on_verdict(_fail(), rms_ok=False) is LockoutAction.EJECT


def test_a_pass_resets_the_miss_streak():
    lk = Lockout()
    lk.on_verdict(_fail(), rms_ok=False)
    assert lk.on_verdict(_pass(), rms_ok=True) is LockoutAction.NONE
    # Streak reset: next single fail is WARN again, not EJECT.
    assert lk.on_verdict(_fail(), rms_ok=False) is LockoutAction.WARN


def test_idle_after_quiet_window_post_eject():
    lk = Lockout(idle_after_s=5.0)
    lk.on_verdict(_fail(), rms_ok=False)
    lk.on_verdict(_fail(), rms_ok=False)   # EJECT, lockout clock starts at... set below
    lk.note_ejected_at(now=100.0)
    # Still near-field active -> no IDLE yet.
    assert lk.on_idle_tick(now=104.9, near_field_rms_active=True) is None
    # 5s of NO near-field RMS -> IDLE (accept fresh wake).
    assert lk.on_idle_tick(now=106.0, near_field_rms_active=False) is LockoutAction.IDLE


def test_no_permanent_lockout_idle_rearms_on_activity():
    lk = Lockout(idle_after_s=5.0)
    lk.on_verdict(_fail(), rms_ok=False)
    lk.on_verdict(_fail(), rms_ok=False)
    lk.note_ejected_at(now=100.0)
    # Near-field activity keeps resetting the quiet clock (never stuck ejected).
    assert lk.on_idle_tick(now=103.0, near_field_rms_active=True) is None
    assert lk.on_idle_tick(now=109.0, near_field_rms_active=False) is None  # clock reset at 103
    assert lk.on_idle_tick(now=108.1, near_field_rms_active=False) is None
```

> The last test fixes the quiet-clock semantics: any near-field activity resets the "quiet since" timestamp, so IDLE only fires after a *continuous* `idle_after_s` of silence. Implement `on_idle_tick` to update a `_quiet_since` on activity.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_lockout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.lockout'`

- [ ] **Step 3: Implement the lockout**

```python
# modes/director/lockout.py
"""De-risked session-hijack lockout (spec section 7).

Never a permanent lockout: the real user must not be ejected by short-segment
ECAPA noise (MEMORY: ecapa-short-segment-unreliable.md).
- 1st failed window      -> WARN (duck + caution), NOT eject.
- 2 consecutive failed windows AND a failed RMS proximity check -> EJECT.
- After EJECT, idle_after_s of continuous no-near-field-RMS -> IDLE (accept a
  fresh wake), so the user is never permanently locked out.
A passing window resets the miss streak.
"""

import enum
from typing import Optional

from .safety_net import SafetyVerdict


class LockoutAction(enum.Enum):
    NONE = "NONE"
    WARN = "WARN"
    EJECT = "EJECT"
    IDLE = "IDLE"


class Lockout:
    def __init__(self, idle_after_s: float = 5.0):
        self._idle_after_s = idle_after_s
        self._miss_streak = 0
        self._ejected = False
        self._quiet_since: Optional[float] = None

    def on_verdict(self, verdict: SafetyVerdict, rms_ok: bool) -> LockoutAction:
        if verdict.smoother_ok:
            self._miss_streak = 0
            return LockoutAction.NONE
        self._miss_streak += 1
        if self._miss_streak >= 2 and not rms_ok:
            self._ejected = True
            return LockoutAction.EJECT
        return LockoutAction.WARN

    def note_ejected_at(self, now: float) -> None:
        self._ejected = True
        self._quiet_since = now

    def on_idle_tick(self, now: float, near_field_rms_active: bool) -> Optional[LockoutAction]:
        if not self._ejected:
            return None
        if near_field_rms_active:
            self._quiet_since = now           # activity resets the quiet clock
            return None
        if self._quiet_since is None:
            self._quiet_since = now
            return None
        if now - self._quiet_since >= self._idle_after_s:
            return LockoutAction.IDLE
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_lockout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modes/director/lockout.py tests/director/test_lockout.py
git commit -m "feat(director): de-risked WARN->EJECT->IDLE lockout (no permanent lockout)"
```

---

## Task 8: Holdout capture before finalize_enrollment

> **Spike-independent.** Verify-before-serve survives the destructive finalize.

**Files:**
- Modify: `core/speaker/enrollment_store.py` (add a read-only method; do NOT change `finalize_enrollment`)
- Test: `tests/director/test_holdout_capture.py`

**Interfaces:**
- Produces: `EnrollmentStore.holdout_utterance_embedding(user_id: str) -> np.ndarray` — returns ONE per-utterance embedding (the last row of `<id>_utterances.npy`) **before** `finalize_enrollment` deletes that file (`os.remove`, `enrollment_store.py:99`). Raises `FileNotFoundError` if no utterances file exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_holdout_capture.py
import os
import numpy as np
import pytest
from core.speaker.enrollment_store import EnrollmentStore


def test_holdout_captured_before_finalize_deletes(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    store.enroll("u1", e1)
    store.enroll("u1", e2)

    holdout = store.holdout_utterance_embedding("u1")    # BEFORE finalize
    assert np.allclose(holdout, e2)                       # last utterance row

    store.finalize_enrollment("u1")                       # deletes utterances file
    assert not os.path.exists(os.path.join(str(tmp_path), "u1_utterances.npy"))


def test_holdout_missing_raises(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.holdout_utterance_embedding("nobody")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_holdout_capture.py -v`
Expected: FAIL — `AttributeError: 'EnrollmentStore' object has no attribute 'holdout_utterance_embedding'`

- [ ] **Step 3: Add the read-only holdout method**

Add this method to `EnrollmentStore` (insert after `utterance_count`, around `core/speaker/enrollment_store.py:112`). It reads, never deletes — `finalize_enrollment` is untouched:

```python
    def holdout_utterance_embedding(self, user_id: str) -> np.ndarray:
        """Return ONE per-utterance embedding for verify-before-serve.

        Must be called BEFORE finalize_enrollment, which deletes the utterances
        file (os.remove at the end of finalize). Returns the last recorded
        utterance row so the Director can score cosine(primary, holdout) at
        session start. Raises FileNotFoundError if no utterances exist.
        """
        utt_path = self._utterances_path(user_id)
        if not os.path.exists(utt_path):
            raise FileNotFoundError(f"No utterances found for '{user_id}'")
        utterances = np.load(utt_path)
        return np.asarray(utterances[-1], dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_holdout_capture.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/speaker/enrollment_store.py tests/director/test_holdout_capture.py
git commit -m "feat(enrollment): capture holdout utterance embedding before finalize deletes it"
```

---

## Task 9: Verify-before-serve gate + enrollment hardening config

> **Spike-independent.** The Director refuses to start when `cosine(primary, holdout) < 0.80`.

**Files:**
- Create: `modes/director/verify.py`
- Modify: `config.yaml` (raise `enrollment_min_self_similarity` 0.6→0.80; add `verify_before_serve_threshold: 0.80`, `lockout_idle_after_s: 5`)
- Test: `tests/director/test_verify_before_serve.py`

**Interfaces:**
- Consumes: `DirectorHandoff.primary_embedding`, `DirectorHandoff.holdout_embedding` (Task 8 supplies the holdout into the handoff).
- Produces: `verify_before_serve(primary: np.ndarray, holdout: np.ndarray, threshold: float = 0.80) -> tuple[bool, float]` — returns `(ok, score)`; `ok=False` means the Director returns to IDLE with a re-enroll prompt rather than starting the session.

- [ ] **Step 1: Write the failing test**

```python
# tests/director/test_verify_before_serve.py
import numpy as np
import pytest
from modes.director.verify import verify_before_serve


def _emb(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_matching_holdout_passes():
    p = _emb(1.0, 0.05, 0.0)
    h = _emb(1.0, 0.0, 0.0)
    ok, score = verify_before_serve(p, h, threshold=0.80)
    assert ok is True
    assert score >= 0.80


def test_orthogonal_holdout_refused():
    ok, score = verify_before_serve(_emb(1, 0), _emb(0, 1), threshold=0.80)
    assert ok is False
    assert score == pytest.approx(0.0, abs=1e-6)


def test_threshold_boundary_just_below_refused():
    # cosine ~0.78 < 0.80 -> refuse.
    p = _emb(1.0, 0.0)
    h = _emb(1.0, 0.80)   # cos = 1/sqrt(1+0.64) ~= 0.78
    ok, score = verify_before_serve(p, h, threshold=0.80)
    assert ok is False
    assert 0.75 < score < 0.80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/director/test_verify_before_serve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes.director.verify'`

- [ ] **Step 3: Implement the gate**

```python
# modes/director/verify.py
"""Verify-before-serve gate (spec section 7).

Scores the finalized primary embedding against a holdout utterance embedding
captured BEFORE finalize_enrollment deleted the per-utterance file (Task 8).
Below threshold -> the Director refuses to start (return to IDLE, re-enroll).
0.80 matches the ~2% EER operating point on >=5s cumulative enrollment audio.
"""

import numpy as np


def verify_before_serve(primary: np.ndarray, holdout: np.ndarray,
                        threshold: float = 0.80) -> tuple:
    """Return (ok, score). ok == score >= threshold."""
    a = np.asarray(primary, dtype=np.float32)
    b = np.asarray(holdout, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    score = 0.0 if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))
    return (score >= threshold, score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/director/test_verify_before_serve.py -v`
Expected: PASS

- [ ] **Step 5: Harden the enrollment config**

Edit `config.yaml`. Raise the self-similarity floor (currently `core.speaker.enrollment_min_self_similarity: 0.6`, `config.yaml:13`) and add the two new Director keys. Replace line 13:

```yaml
    # Raised 0.6 -> 0.80 (Director verify-before-serve): 0.6 admitted drifty
    # enrollments that later false-rejected the real user; 0.80 matches the
    # ~2% EER operating point on >=5s cumulative audio. Reject + re-enroll
    # below this (bounded by enrollment_max_retries).
    enrollment_min_self_similarity: 0.80
```

Under `kiosk.talkback:` (alongside `silence_timeout_s`, `config.yaml:52`) add:

```yaml
    # Director verify-before-serve: refuse to start a session when the finalized
    # primary embedding disagrees with the pre-finalize holdout utterance.
    verify_before_serve_threshold: 0.80
    # De-risked lockout: after an EJECT, accept a fresh wake once there has been
    # no near-field RMS for this long (never a permanent lockout).
    lockout_idle_after_s: 5
```

- [ ] **Step 6: Verify config parses and the threshold landed**

Run:
```bash
python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); \
assert c['core']['speaker']['enrollment_min_self_similarity']==0.80; \
assert c['kiosk']['talkback']['verify_before_serve_threshold']==0.80; \
assert c['kiosk']['talkback']['lockout_idle_after_s']==5; print('config OK')"
```
Expected: `config OK`

- [ ] **Step 7: Commit**

```bash
git add modes/director/verify.py config.yaml tests/director/test_verify_before_serve.py
git commit -m "feat(director): verify-before-serve gate + harden enrollment self-similarity 0.6->0.80"
```

---

## Self-Review

**1. Spec coverage (§7 + §9 + §14):**

| Spec requirement | Task |
|---|---|
| pVAD bare-model load (vendor FireRedChat, `torch.load` cpu + `eval`) | Task 1 (Steps 1-3), Task 2 |
| Combined hot-path latency spike (`bench/pvad_contention.py`, <100 ms p95, cutover gate) | Task 1 (Steps 4-5) |
| Record verdict in `docs/notes/2026-06-22-pvad.md` | Task 1 (Step 5) |
| NO-GO degraded path = FOCUS RMS-proximity-gate only (named) | Task 1 (Step 5) + Task 5 (crash-fallback, same predicate) |
| Frame-by-frame streaming with causal-conv + GRU hidden state | Task 1 (`forward_step`), Task 4 (`VADStream`) |
| `VADStream.update_speaker(embedding)` + per-~50 ms `is_target`/`confidence`/`rms` | Task 4 |
| Wire pVAD into Ingestion so `is_target` is real | Task 5 (Step 5) |
| Demote rolling-window ECAPA + `DecisionSmoother` to safety-net (accumulate only `is_target`, off hot path, M-of-N hijack) | Task 6 |
| Verify-before-serve: holdout captured BEFORE `finalize_enrollment` (line 99) | Task 8 |
| Director refuses to start when `cosine(primary, holdout) < 0.80` | Task 9 |
| Enrollment hardening `enrollment_min_self_similarity` 0.6→0.80 (config + comment) | Task 9 (Step 5) |
| De-risked lockout: WARN → EJECT (2 windows AND failed RMS) → IDLE after 5 s, never permanent | Task 7 |
| Crash-fallback `is_target := rms >= proximity_rms` | Task 5 (Step 3) |
| REJECT NeMo Streaming Sortformer (rationale) | Global Constraints |

All §7 requirements map to a task. The pVAD-dependent tasks (2-5) are gated on the Task 1 spike; the identity/lockout/enrollment tasks (6-9) are explicitly spike-independent and ship in both outcomes.

**2. Placeholder scan:** `grep -nE "TBD|TODO|FIXME"` over the plan returns no matches (the spike-gate and the named RMS-fallback are real, runnable code, not placeholders). The one `noqa: BLE001` and one `noqa: ARG002`-style suppression are real lint directives, not gaps. Two "match Plan 02's conventions" notes (Task 5 Step 5) point at a neighboring plan's established attribute names, not undefined behavior — the binding contract (`is_target` derived from `PvadWorker.process`) is concrete.

**3. Type consistency:** `SpeakerFrame(ts, is_target, confidence, rms)` is defined once (Task 2) and consumed identically in Tasks 4/5. `SafetyVerdict(score, smoother_ok)` defined in Task 6, consumed in Task 7. `LockoutAction` enum members `{NONE, WARN, EJECT, IDLE}` are consistent across Task 7. `verify_before_serve(...) -> (ok, score)` matches its tests. `PvadModel.forward_step(mel_frame, state) -> (prob, state)` and `init_state(spk) -> state` are used consistently by `VADStream` and the spike harness. `update_speaker(embedding)` has one signature everywhere (`PvadWorker` → `VADStream`).

`grep -nE "TBD|TODO|FIXME"` on the saved file: **no matches.**
