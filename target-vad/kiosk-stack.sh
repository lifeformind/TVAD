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
