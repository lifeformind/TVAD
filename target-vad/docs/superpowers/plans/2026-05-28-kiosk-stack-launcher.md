# Kiosk Stack Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single `kiosk-stack.sh` script that brings up the full-duplex talkback stack (GPU LLM backend + interactive kiosk) and tears it down, plus one-time `build-llm` (CUDA rebuild) and `download-model` (q5 quant) setup steps.

**Architecture:** One bash script at `TVAD/target-vad/kiosk-stack.sh` with subcommands `start | stop | status | build-llm | download-model`. `start` launches `python3 -m llama_cpp.server` in the background (GPU offload, PID file, log), waits until `/v1/models` responds, installs a cleanup trap, then runs `kiosk.py --talkback` in the foreground (it owns the mic / live console). `stop` kills the LLM via the PID file. Setup subcommands rebuild llama-cpp-python for sm_121 and download the q5 GGUF into the existing `~/.cache/models` HF cache.

**Tech Stack:** bash, llama-cpp-python (CUDA 13 / Blackwell sm_121), huggingface_hub `hf` CLI, curl, ss.

**Spec:** `docs/superpowers/specs/2026-05-28-kiosk-stack-launcher-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `kiosk-stack.sh` (new) | All stack lifecycle: config block, helpers, and the 5 subcommands |
| `.gitignore` (modify) | Ignore `.llm.pid` and `logs/` so runtime artifacts aren't committed |

The script is small and cohesive (~130 lines); it stays in one file. Helpers are defined once (DRY) and reused by the subcommands.

**Notes for the implementer:**
- The repo root for this work is `~/FullDuplexVoice/TVAD/target-vad`. The script lives there alongside `kiosk.py` and `config.yaml`.
- Two env requirements on this machine: PortAudio needs `LD_LIBRARY_PATH=$HOME/.local/lib` (for the kiosk), and pip needs `PIP_BREAK_SYSTEM_PACKAGES=1`. The script exports both at the top.
- `set -euo pipefail` is on, so any pipeline that may legitimately produce no output (e.g. `ls ... | grep ...`) MUST end with `|| true`, and "not found" checks belong inside `if` conditions (where `set -e` is suppressed).
- Tasks 3 and 4 actually run real setup (a ~5 GB download and a multi-minute CUDA rebuild). They are environment-modifying — see the caution note in each.

---

### Task 1: Script skeleton — config block, helpers, dispatch

**Files:**
- Create: `kiosk-stack.sh`

- [ ] **Step 1: Write the full skeleton** (subcommands are stubs for now)

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- location ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- config (edit here) ----
MODEL_REPO="Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_GLOB="*q5_k_m*.gguf"
HF_CACHE="$HOME/.cache/models"
N_GPU_LAYERS=-1          # full GPU offload; needs the CUDA build (build-llm)
N_CTX=4096
HOST=127.0.0.1
PORT=8080
CHAT_FORMAT=chatml       # Qwen 2.5
READY_TIMEOUT_S=120

PID_FILE="$SCRIPT_DIR/.llm.pid"
LOG_DIR="$SCRIPT_DIR/logs"
LLM_LOG="$LOG_DIR/llm.log"

export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
export PIP_BREAK_SYSTEM_PACKAGES=1

# ---- helpers ----
log() { printf '\033[1m[stack]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[stack]\033[0m %s\n' "$*" >&2; }

# Echo the resolved q5 model path, preferring a split's first shard. Empty if absent.
resolve_model() {
  local dir="$HF_CACHE/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots"
  local shard
  shard="$(ls -1 "$dir"/*/$MODEL_GLOB 2>/dev/null | grep -E '00001-of-' | head -n1 || true)"
  if [[ -n "$shard" ]]; then echo "$shard"; return 0; fi
  ls -1 "$dir"/*/$MODEL_GLOB 2>/dev/null | head -n1 || true
}

llm_pid() { [[ -f "$PID_FILE" ]] && cat "$PID_FILE" || true; }

llm_alive() {
  local pid; pid="$(llm_pid)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

llm_reachable() { curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; }

# True if anything is listening on PORT (an OpenAI server OR a foreign listener).
port_in_use() {
  llm_reachable && return 0
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"
}

gpu_offload() {
  python3 -c "import llama_cpp,sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null
}

# ---- subcommands (stubs; filled in later tasks) ----
cmd_download() { err "download-model not implemented yet"; exit 1; }
cmd_build()    { err "build-llm not implemented yet"; exit 1; }
cmd_status()   { err "status not implemented yet"; exit 1; }
cmd_stop()     { err "stop not implemented yet"; exit 1; }
cmd_start()    { err "start not implemented yet"; exit 1; }

usage() {
  cat <<EOF
Usage: $0 {start|stop|status|build-llm|download-model}
  start          bring up the LLM (GPU) then run the kiosk in the foreground
  stop           stop the LLM server started by this script
  status         show LLM / model / GPU status
  build-llm      one-time: rebuild llama-cpp-python with CUDA (sm_121)
  download-model one-time: download the q5 GGUF into the HF cache
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start)          cmd_start ;;
    stop)           cmd_stop ;;
    status)         cmd_status ;;
    build-llm)      cmd_build ;;
    download-model) cmd_download ;;
    "")             usage; exit 1 ;;
    *)              err "Unknown command: $cmd"; usage; exit 2 ;;
  esac
}

# Dispatch only when executed directly, so the script can be sourced for testing.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
```

- [ ] **Step 2: Make executable and verify dispatch**

Run: `chmod +x kiosk-stack.sh && ./kiosk-stack.sh && echo "rc=$?"`
Expected: prints the usage block, `rc=1`.

Run: `./kiosk-stack.sh bogus; echo "rc=$?"`
Expected: `Unknown command: bogus`, usage, `rc=2`.

Run: `bash -n kiosk-stack.sh && echo "syntax ok"`
Expected: `syntax ok` (no syntax errors).

- [ ] **Step 3: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): kiosk-stack.sh skeleton with config + helpers"
```

---

### Task 2: Ignore runtime artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check current .gitignore**

Run: `cat .gitignore 2>/dev/null || echo "(no .gitignore)"`
Note whether `logs/` and `.llm.pid` are already present (they are not, per spec).

- [ ] **Step 2: Append ignore rules**

Append these lines to `.gitignore` (create the file if absent):

```
# kiosk stack runtime artifacts
.llm.pid
logs/
```

- [ ] **Step 3: Verify**

Run: `git status --short`
Expected: `.llm.pid` and anything under `logs/` no longer appear as untracked. `.gitignore` shows as modified/new.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore .llm.pid and logs/"
```

---

### Task 3: `download-model` (downloads the q5 GGUF for real)

> **Caution — real network action.** This downloads ~5 GB into `$HF_CACHE`. `hf download` is cached/idempotent, so re-running is a no-op once present. Requires network access.

**Files:**
- Modify: `kiosk-stack.sh` (replace the `cmd_download` stub)

- [ ] **Step 1: Replace the `cmd_download` stub**

```bash
cmd_download() {
  log "Downloading $MODEL_GLOB from $MODEL_REPO into $HF_CACHE ..."
  hf download "$MODEL_REPO" --include "$MODEL_GLOB" --cache-dir "$HF_CACHE"
  local model; model="$(resolve_model)"
  if [[ -z "$model" ]]; then
    err "Download finished but no file matching $MODEL_GLOB was found under $HF_CACHE."
    exit 1
  fi
  log "Model ready: $model"
}
```

- [ ] **Step 2: Run the download**

Run: `./kiosk-stack.sh download-model`
Expected: `hf` shows download progress (or "already cached"), then `Model ready: /home/ldrgx10/.cache/models/.../qwen2.5-7b-instruct-q5_k_m.gguf`.

- [ ] **Step 3: Verify the model resolves**

Run: `ls -la ~/.cache/models/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/*/*q5_k_m*.gguf`
Expected: one or more q5 `.gguf` files (a single file, or `-00001-of-...`/`-00002-of-...` shards) of multi-GB size.

- [ ] **Step 4: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): download-model fetches the q5 GGUF"
```

---

### Task 4: `build-llm` (rebuilds llama-cpp-python with CUDA for real)

> **Caution — environment-modifying.** This force-reinstalls `llama-cpp-python`, replacing the current CPU-only 0.3.23 build with a CUDA build. The compile takes several minutes (20 cores). If the build fails, the package may be left broken — recovery is to reinstall a known-good version: `PIP_BREAK_SYSTEM_PACKAGES=1 pip install --force-reinstall --no-cache-dir llama-cpp-python` (CPU) — so confirm the verify step passes before committing.

**Files:**
- Modify: `kiosk-stack.sh` (replace the `cmd_build` stub)

- [ ] **Step 1: Replace the `cmd_build` stub**

```bash
cmd_build() {
  log "Rebuilding llama-cpp-python with CUDA (sm_121). This takes several minutes..."
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=121" \
    pip install --force-reinstall --no-cache-dir llama-cpp-python
  if gpu_offload; then
    log "GPU offload supported. \xe2\x9c\x93"
  else
    err "Build completed but llama_cpp.llama_supports_gpu_offload() is False."
    err "The CUDA build did not take effect. Check CUDA toolkit / arch and retry."
    exit 1
  fi
}
```

- [ ] **Step 2: Run the rebuild**

Run: `./kiosk-stack.sh build-llm`
Expected: a long cmake/nvcc compile, then `GPU offload supported. ✓`.

- [ ] **Step 3: Independently verify GPU offload**

Run: `python3 -c "import llama_cpp; print('gpu_offload:', llama_cpp.llama_supports_gpu_offload())"`
Expected: `gpu_offload: True`.

- [ ] **Step 4: Commit** (only after Step 3 prints True)

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): build-llm rebuilds llama-cpp-python with CUDA (sm_121)"
```

---

### Task 5: `status`

**Files:**
- Modify: `kiosk-stack.sh` (replace the `cmd_status` stub)

- [ ] **Step 1: Replace the `cmd_status` stub**

```bash
cmd_status() {
  local pid; pid="$(llm_pid)"
  if llm_alive; then log "LLM process: running (pid $pid)"; else log "LLM process: not running"; fi
  if llm_reachable; then
    log "LLM endpoint: reachable at http://$HOST:$PORT/v1"
  else
    log "LLM endpoint: unreachable"
  fi
  if gpu_offload; then log "llama_cpp GPU offload: available"; else log "llama_cpp GPU offload: NOT available (CPU build)"; fi
  local model; model="$(resolve_model)"
  if [[ -n "$model" ]]; then log "Model: $model"; else log "Model: not downloaded (run: $0 download-model)"; fi
}
```

- [ ] **Step 2: Verify with LLM down**

Run: `./kiosk-stack.sh status`
Expected (no server running yet): `LLM process: not running`, `LLM endpoint: unreachable`, `llama_cpp GPU offload: available` (after Task 4), `Model: <q5 path>` (after Task 3).

- [ ] **Step 3: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): status reports LLM/model/GPU state"
```

---

### Task 6: `stop`

> Implemented before `start` because `start`'s cleanup trap calls `cmd_stop`.

**Files:**
- Modify: `kiosk-stack.sh` (replace the `cmd_stop` stub)

- [ ] **Step 1: Replace the `cmd_stop` stub**

```bash
cmd_stop() {
  local pid; pid="$(llm_pid)"
  if [[ -z "$pid" ]]; then log "No .llm.pid found; nothing to stop."; return 0; fi
  if kill -0 "$pid" 2>/dev/null; then
    log "Stopping LLM (pid $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then log "Still alive; forcing kill."; kill -KILL "$pid" 2>/dev/null || true; fi
  else
    log "LLM pid $pid is not running (stale pid file)."
  fi
  rm -f "$PID_FILE"
}
```

- [ ] **Step 2: Verify graceful no-op when nothing is running**

Run: `rm -f .llm.pid; ./kiosk-stack.sh stop; echo "rc=$?"`
Expected: `No .llm.pid found; nothing to stop.`, `rc=0`.

- [ ] **Step 3: Verify stale-pid handling**

Run: `echo 999999 > .llm.pid; ./kiosk-stack.sh stop; echo "rc=$?"; test ! -f .llm.pid && echo "pid removed"`
Expected: `LLM pid 999999 is not running (stale pid file).`, `rc=0`, `pid removed`.

- [ ] **Step 4: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): stop tears down the LLM via pid file"
```

---

### Task 7: `start` — bring up LLM, then run kiosk in the foreground

**Files:**
- Modify: `kiosk-stack.sh` (add `start_llm_bg` + `wait_for_llm` helpers; replace the `cmd_start` stub)

- [ ] **Step 1: Add the two LLM-launch helpers** (place them just above the `cmd_start` definition)

```bash
start_llm_bg() {
  mkdir -p "$LOG_DIR"
  log "Starting llama_cpp.server on $HOST:$PORT (n_gpu_layers=$N_GPU_LAYERS)..."
  nohup python3 -m llama_cpp.server \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --n_ctx "$N_CTX" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --chat_format "$CHAT_FORMAT" \
    >"$LLM_LOG" 2>&1 &
  echo $! > "$PID_FILE"
  WE_STARTED_LLM=1
}

wait_for_llm() {
  log "Waiting for LLM to load (timeout ${READY_TIMEOUT_S}s)..."
  local waited=0
  while (( waited < READY_TIMEOUT_S )); do
    if ! llm_alive; then
      err "LLM process exited during startup. Last log lines:"; tail -n 30 "$LLM_LOG" >&2
      rm -f "$PID_FILE"; exit 1
    fi
    if llm_reachable; then log "LLM ready."; return 0; fi
    sleep 2; waited=$((waited + 2))
  done
  err "LLM not ready after ${READY_TIMEOUT_S}s. Last log lines:"; tail -n 30 "$LLM_LOG" >&2
  cmd_stop; exit 1
}
```

- [ ] **Step 2: Replace the `cmd_start` stub**

```bash
cmd_start() {
  MODEL="$(resolve_model)"
  if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
    err "No q5 model found under $HF_CACHE. Run: $0 download-model"
    exit 1
  fi

  if llm_reachable; then
    if llm_alive; then
      log "Reusing the LLM already running from this script (pid $(llm_pid))."
    else
      err "Port $PORT is already serving an OpenAI endpoint we did not start (no live .llm.pid)."
      err "Refusing to touch it. Stop it manually or change PORT."
      exit 1
    fi
  elif port_in_use; then
    err "Port $PORT is in use by another (non-OpenAI) process. Refusing to start."
    err "Free the port or change PORT in $0."
    exit 1
  else
    start_llm_bg
    wait_for_llm
  fi

  if [[ "${WE_STARTED_LLM:-0}" == "1" ]]; then
    trap 'cmd_stop' EXIT
  fi

  log "Launching kiosk (foreground). Ctrl-C to end the session and stop the LLM."
  python3 kiosk.py --talkback
}
```

- [ ] **Step 3: Syntax check**

Run: `bash -n kiosk-stack.sh && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 4: Verify the LLM bring-up path non-interactively**

The main guard from Task 1 makes the script sourceable, so its functions can be driven
directly — exercising everything in `start` except the foreground kiosk. Run this in a
subshell so the script's `set -e`/`exit` stay contained:

```bash
( source ./kiosk-stack.sh
  MODEL="$(resolve_model)"
  start_llm_bg
  wait_for_llm
  echo "--- /v1/models ---"
  curl -s "http://$HOST:$PORT/v1/models" | head -c 300; echo
  cmd_stop
)
```

Expected: `Starting llama_cpp.server...`, `LLM ready.`, a JSON blob listing the model,
then `Stopping LLM...`. While it is up (before `cmd_stop`), a second shell can confirm
`curl -s 127.0.0.1:8080/v1/models` returns JSON and `nvidia-smi` shows a second GPU process.

- [ ] **Step 5: Confirm teardown left things clean**

Run: `./kiosk-stack.sh status`
Expected: `LLM process: not running`, `LLM endpoint: unreachable` (the trap stopped it), pid file gone.

- [ ] **Step 6: Commit**

```bash
git add kiosk-stack.sh
git commit -m "feat(stack): start brings up GPU LLM then runs kiosk in foreground"
```

---

### Task 8: End-to-end smoke + live handoff

**Files:** none (verification only)

- [ ] **Step 1: Full status snapshot**

Run: `./kiosk-stack.sh status`
Expected: model present, GPU offload available, LLM not running.

- [ ] **Step 2: Idempotency checks**

Run: `./kiosk-stack.sh download-model`
Expected: no re-download (cached), `Model ready: ...`.

Run: `./kiosk-stack.sh stop`
Expected: graceful `No .llm.pid found; nothing to stop.` (or stops a leftover), `rc=0`.

- [ ] **Step 3: Hand off the live interactive test to the user**

`./kiosk-stack.sh start` launches the real mic kiosk and blocks on the foreground —
this requires a human at the microphone and cannot be auto-verified. Report that the
script is ready and that the live test is:

```
./kiosk-stack.sh start
# say the wake phrase, hold a multi-turn conversation, try barging in
# Ctrl-C to end — the LLM is torn down automatically
```

Confirm in a second shell during the session: `curl -s 127.0.0.1:8080/v1/models` and `nvidia-smi`.

- [ ] **Step 4: No commit** (verification only). If any prior task left uncommitted tweaks, commit them now.

---

## Self-Review

**Spec coverage:**
- `start` (bg LLM + foreground kiosk + trap) → Task 7. ✓
- `stop` → Task 6. ✓
- `status` → Task 5. ✓
- `build-llm` (CUDA sm_121) → Task 4. ✓
- `download-model` (q5 into `$HF_CACHE`) → Task 3. ✓
- Config block / variables → Task 1. ✓
- Foreign-port guard (don't kill vLLM) → Task 7 `cmd_start` (`llm_reachable`/`port_in_use` branches). ✓
- Model-missing guard with download hint → Task 7 Step 2. ✓
- Ready-timeout with log tail → Task 7 `wait_for_llm`. ✓
- `.gitignore` for `.llm.pid` + `logs/` → Task 2. ✓
- No config.yaml change needed (already q5) → confirmed, no task. ✓

**Placeholder scan:** No TBD/TODO; every code step contains full code. ✓

**Type/name consistency:** Helper names (`resolve_model`, `llm_pid`, `llm_alive`, `llm_reachable`, `port_in_use`, `gpu_offload`) and subcommand functions (`cmd_start`/`cmd_stop`/`cmd_status`/`cmd_build`/`cmd_download`) defined in Task 1 are used consistently in Tasks 3-7. `start_llm_bg`/`wait_for_llm`/`WE_STARTED_LLM`/`MODEL` introduced and used within Task 7. Config vars (`PORT`, `HOST`, `N_GPU_LAYERS`, `N_CTX`, `CHAT_FORMAT`, `READY_TIMEOUT_S`, `PID_FILE`, `LOG_DIR`, `LLM_LOG`, `HF_CACHE`, `MODEL_GLOB`, `MODEL_REPO`) all defined in Task 1. ✓
