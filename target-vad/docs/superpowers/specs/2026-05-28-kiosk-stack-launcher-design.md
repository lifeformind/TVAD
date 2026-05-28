# Kiosk Stack Launcher — Design Spec

**Date:** 2026-05-28
**Status:** Approved

## Problem

Bringing up the full-duplex talkback kiosk currently requires manual, undocumented
steps that are easy to get wrong:

- The LLM backend (`llama_cpp.server` on `127.0.0.1:8080`) must be started separately
  and by hand. There is no launch script anywhere in the repo.
- The installed `llama-cpp-python` (0.3.23) is **CPU-only**
  (`llama_cpp.llama_supports_gpu_offload()` returns `False`), so even on the GB10
  Blackwell GPU the LLM runs on CPU — slow enough to hurt the conversational feel.
- `config.yaml` names the model `qwen2.5-7b-instruct-q5_k_m`, but only the
  lower-quality **`q3_k_m`** quant is actually downloaded. With GPU offload and the
  GB10's unified memory there is ample room for the better q5 quant the config
  already references.
- Running the kiosk requires non-obvious env (`LD_LIBRARY_PATH=$HOME/.local/lib`
  for PortAudio).

We want a single script to start the whole stack and stop it, a one-time CUDA
rebuild so the LLM uses the GPU, and a one-time download of the q5 quant.

## Stack Shape

The talkback stack is exactly **two processes**:

1. **`llama_cpp.server`** — external LLM backend, OpenAI-compatible, on `127.0.0.1:8080`.
   The only thing that must be started separately.
2. **`kiosk.py --talkback`** — loads STT (faster-whisper) and TTS (Kokoro) **in-process**,
   owns the microphone, streams a live console, and exits on Ctrl-C. `kiosk.py:189`
   pings the LLM at startup and exits with code 3 if it is unreachable.

No other daemons are involved.

## Environment (verified 2026-05-28)

- GPU: **NVIDIA GB10**, compute capability **12.1 (sm_121, Blackwell)**, driver 580.
- CUDA toolkit **13.0** at `/usr/local/cuda` (`nvcc` 13.0.88).
- Build tools: cmake 3.28.3, gcc 13.3.0; aarch64; 20 cores.
- A **vLLM engine is already running on the GPU** (~15.5 GB). It is **not** on port 8080.
  GB10 unified memory leaves ample room, but the launcher must not assume port 8080 is free.

## Design — Approach A: one script, all subcommands

A single `kiosk-stack.sh` at the repo root (`TVAD/target-vad/`) with subcommands
`start | stop | status | build-llm | download-model`. Daily use is `start` / `stop`;
the CUDA rebuild (`build-llm`) and the q5 download (`download-model`) are one-time
setup steps captured in the script so they are reproducible in-repo.

### Config block (top of script, easy to edit)

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_REPO` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | HF repo for the GGUF |
| `MODEL_GLOB` | `*q5_k_m*.gguf` | quant to download / resolve |
| `HF_CACHE` | `$HOME/.cache/models` | same cache dir the existing q3 uses |
| `MODEL` | resolved glob `$HF_CACHE/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/*/*q5_k_m*.gguf` | resolve at runtime, not a hardcoded snapshot hash; if the quant is split, pick the `*-00001-of-*` shard |
| `N_GPU_LAYERS` | `-1` | full GPU offload (requires the CUDA rebuild) |
| `N_CTX` | `4096` | |
| `HOST` | `127.0.0.1` | |
| `PORT` | `8080` | |
| `CHAT_FORMAT` | `chatml` | Qwen 2.5 |
| `READY_TIMEOUT_S` | `120` | GPU model load headroom |

Runtime env: prepend `LD_LIBRARY_PATH=$HOME/.local/lib` for the kiosk (PortAudio).

### `download-model` (one-time)

```
hf download "$MODEL_REPO" --include "$MODEL_GLOB" --cache-dir "$HF_CACHE"
```

`hf download` is idempotent (cached), so re-running is a no-op once present. The
`--include` glob grabs the q5 file(s) whether the quant ships as a single `.gguf`
or as split shards. After download, resolve and report the `MODEL` path.

### `build-llm` (one-time)

```
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=121" \
PIP_BREAK_SYSTEM_PACKAGES=1 \
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

Then assert `python3 -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload()"`.
Targets sm_121 against CUDA 13. On assert failure, report and exit non-zero.

### `start`

0. **Model check.** Resolve `MODEL`; if no q5 file is present, abort with a hint to
   run `download-model`. (Does not auto-download — keeps `start` fast and predictable.)
1. **Port check.** If `PORT` is in use:
   - If our own server (live PID in `.llm.pid` matching a `llama_cpp.server` process) → reuse it, skip launch.
   - If a **foreign** process (e.g. the running vLLM) → abort with a clear message. **Never kill it.**
2. **Launch LLM** in background:
   `python3 -m llama_cpp.server --model "$MODEL" --host "$HOST" --port "$PORT" --n_ctx "$N_CTX" --n_gpu_layers "$N_GPU_LAYERS" --chat_format "$CHAT_FORMAT"`
   → stdout/stderr to `logs/llm.log`, PID to `.llm.pid`.
3. **Wait for ready:** poll `curl -sf "$HOST:$PORT/v1/models"` until success or `READY_TIMEOUT_S`.
   On timeout: tail `logs/llm.log`, kill the partial server, exit non-zero.
4. **Trap** EXIT/INT/TERM → kill the LLM (only if we started it this invocation).
5. **Run kiosk in foreground:**
   `LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH python3 kiosk.py --talkback`
6. Ctrl-C → kiosk exits → trap tears down the LLM.

### `stop`

Read `.llm.pid`; if the process is alive, send `SIGTERM`, wait briefly, then `SIGKILL`
if still alive; remove the pid file. Handle stale/missing pid gracefully (report, exit 0).

### `status`

Report: LLM pid (from `.llm.pid`) and liveness, whether `/v1/models` responds, and
`llama_cpp.llama_supports_gpu_offload()`.

### Config

No change needed — `config.yaml` already names `qwen2.5-7b-instruct-q5_k_m`, which is
exactly the quant `download-model` fetches. (`llama_cpp.server` serves the loaded model
regardless of the request's `model` field, but keeping the name accurate avoids confusion.)

## Files Changed

| File | Change |
|------|--------|
| `kiosk-stack.sh` | new — `start`/`stop`/`status`/`build-llm`/`download-model` |
| `.gitignore` | add `.llm.pid` and `logs/` (currently untracked, not yet ignored) |

## Error Handling

- Missing `MODEL` file on `start` → clear error hinting `download-model`, exit non-zero.
- Port owned by a foreign process → explain it is not ours, do not kill, exit non-zero.
- LLM never becomes ready → tail `logs/llm.log`, clean up, exit non-zero.
- Stale/missing `.llm.pid` on stop → report, exit 0.
- `build-llm` GPU offload assert fails → report, exit non-zero.
- `download-model` finds no matching file in the repo → report, exit non-zero.

## Testing

Shell script — verification is a manual smoke sequence (no bats harness; YAGNI):

1. `download-model` → q5 gguf present under `$HF_CACHE`; `MODEL` resolves to it.
2. `build-llm` → assert `llama_supports_gpu_offload()` is `True`.
3. `start` → `/v1/models` responds; `nvidia-smi` shows a second GPU process; kiosk
   prints "LLM server reachable ✓" and enters the wake-word listen state.
4. `stop` → process gone, port free, pid file removed.
5. `status` accuracy in both up and down states.
6. Foreign-port guard: with vLLM running, confirm `start` (if pointed at vLLM's port)
   refuses rather than killing it.

## Out of Scope

- Even higher precision (fp16/bf16) weights. `MODEL_GLOB` is a variable; a later upgrade.
- systemd service management.
- Backgrounding the kiosk (it is interactive by design).
