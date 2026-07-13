#!/usr/bin/env bash
set -euo pipefail

# ---- location ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- config (edit here) ----
MODEL_REPO="unsloth/gemma-3-4b-it-GGUF"
MODEL_GLOB="gemma-3-4b-it-Q5_K_M.gguf"
HF_CACHE="$HOME/.cache/models"
N_GPU_LAYERS=-1          # full GPU offload; needs the CUDA build (build-llm)
N_CTX=4096
HOST=127.0.0.1
PORT=8080
# Empty -> use the GGUF's embedded chat template. Required for Gemma 3: the
# built-in 'gemma'/'chatml' handlers drop the system prompt, so replies ignore
# the "1-3 sentences, no markdown" instruction and run to the token cap.
CHAT_FORMAT=""
READY_TIMEOUT_S=120

PID_FILE="$SCRIPT_DIR/.llm.pid"
LOG_DIR="$SCRIPT_DIR/logs"
LLM_LOG="$LOG_DIR/llm.log"

export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
export PIP_BREAK_SYSTEM_PACKAGES=1

# ---- helpers ----
log() { printf '\033[1m[stack]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[stack]\033[0m %s\n' "$*" >&2; }

# Echo the resolved model path, preferring a split's first shard. Empty if absent.
# Derives the HF cache dir from MODEL_REPO (org/name -> models--org--name).
resolve_model() {
  local dir="$HF_CACHE/models--${MODEL_REPO//\//--}/snapshots"
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
cmd_build() {
  log "Rebuilding llama-cpp-python with CUDA (sm_121). This takes several minutes..."
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=121" \
    pip install --force-reinstall --no-cache-dir llama-cpp-python
  if gpu_offload; then
    log "GPU offload supported. ✓"
  else
    err "Build completed but llama_cpp.llama_supports_gpu_offload() is False."
    err "The CUDA build did not take effect. Check CUDA toolkit / arch and retry."
    exit 1
  fi
}
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
# TERM a pid, wait up to ~5s, then KILL. No-op if already gone.
term_then_kill() {
  local pid="$1"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
  log "PID $pid still alive; forcing kill."
  kill -KILL "$pid" 2>/dev/null || true
}

stop_llm() {
  local pid; pid="$(llm_pid)"
  if [[ -z "$pid" ]]; then log "LLM: no .llm.pid; not started by this script."; return 0; fi
  if kill -0 "$pid" 2>/dev/null; then
    log "Stopping LLM (pid $pid)..."
    term_then_kill "$pid"
  else
    log "LLM pid $pid is not running (stale pid file)."
  fi
  rm -f "$PID_FILE"
}

# Stop any running talkback kiosk, however it was launched (foreground or detached).
# Match only the actual interpreter process: pgrep -f matches on the whole command
# line, so an unrelated shell/grep/editor that merely mentions "kiosk.py --talkback"
# would otherwise be caught and killed. Filter by /proc/<pid>/comm == python*.
stop_kiosk() {
  local pid comm found=0
  for pid in $(pgrep -f 'kiosk\.py --talkback' 2>/dev/null || true); do
    comm="$(cat "/proc/$pid/comm" 2>/dev/null || true)"
    [[ "$comm" == python* ]] || continue
    log "Stopping kiosk (pid $pid)..."
    term_then_kill "$pid"
    found=1
  done
  [[ "$found" == 0 ]] && log "Kiosk: no running process."
  return 0
}

# Full shutdown: kiosk first (so it stops calling the LLM), then the LLM server.
cmd_stop() {
  stop_kiosk
  stop_llm
}
start_llm_bg() {
  mkdir -p "$LOG_DIR"
  log "Starting llama_cpp.server on $HOST:$PORT (n_gpu_layers=$N_GPU_LAYERS)..."
  local extra=()
  [[ -n "$CHAT_FORMAT" ]] && extra+=(--chat_format "$CHAT_FORMAT")
  nohup python3 -m llama_cpp.server \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --n_ctx "$N_CTX" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    "${extra[@]}" \
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

# Bring the LLM up (or adopt one we already started) and arm the EXIT trap.
# Shared by cmd_start (kiosk in the terminal) and cmd_tune (kiosk under the
# tuning console).
ensure_llm() {
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
}

cmd_start() {
  ensure_llm

  log "Launching kiosk (foreground). Ctrl-C to end the session and stop the LLM."
  # Tee stdout+stderr to a log so a crash/segfault is captured (the kiosk runs
  # in the foreground, so otherwise the traceback only hits the terminal).
  # PYTHONFAULTHANDLER dumps a C-level traceback on segfault (e.g. in the AEC
  # ctypes shim). stdbuf keeps the tee'd stream unbuffered so nothing is lost.
  local kiosk_log="$LOG_DIR/kiosk.err.log"
  PYTHONFAULTHANDLER=1 stdbuf -oL -eL python3 kiosk.py --talkback 2>&1 | tee "$kiosk_log"
}

# One command for a tuning evening: LLM up, then the tuning console in the
# foreground with the kiosk auto-started under it (DIAG on). The kiosk's
# stdout goes to the console's browser log pane, not this terminal; Ctrl-C
# stops console -> kiosk -> (via the EXIT trap) the LLM.
cmd_tune() {
  ensure_llm
  log "Launching tuning console (foreground). Ctrl-C stops the kiosk and the LLM."
  PYTHONFAULTHANDLER=1 python3 -m tune --start-kiosk "$@"
}

usage() {
  cat <<EOF
Usage: $0 {start|tune|stop|status|build-llm|download-model}
  start          bring up the LLM (GPU) then run the kiosk in the foreground
  tune           bring up the LLM, then the tuning console with the kiosk
                 auto-started under it (extra args pass through, e.g.
                 $0 tune --host 0.0.0.0)
  stop           full shutdown: stop the kiosk (any launch mode) and the LLM server
  status         show LLM / model / GPU status
  build-llm      one-time: rebuild llama-cpp-python with CUDA (sm_121)
  download-model one-time: download the q5 GGUF into the HF cache
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start)          cmd_start ;;
    tune)           shift; cmd_tune "$@" ;;
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
