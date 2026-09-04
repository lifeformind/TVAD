#!/usr/bin/env bash
# Train a custom openwakeword wake-word model (ONNX-only) in the training
# environment at ~/.local/opt/oww-train (override with OWW_TRAIN_DIR).
#
# That environment is built once, outside this repo (it is heavy and shares
# nothing with the kiosk runtime): openWakeWord + dscripka/piper-sample-generator
# clones, a --system-site-packages venv, the piper libritts_r checkpoint, MIT
# RIRs, an AudioSet shard + 1h FMA for background noise, and the precomputed
# ACAV100M negative features (~16 GB). See docs/notes/2026-09-04-hey-kusu-wakeword.md
# for the full setup recipe and the local patches applied.
#
# Usage: bench/train_wakeword.sh [config.yml]      # default: hey_kusu.yml
# Output: <output_dir from config>/<model_name>.onnx — copy into models/wake/
# and point kiosk.wake_phrase at the path. We never pass --convert_to_tflite;
# the kiosk detector runs ONNX.
set -euo pipefail

TRAIN_DIR="${OWW_TRAIN_DIR:-$HOME/.local/opt/oww-train}"
CFG="${1:-hey_kusu.yml}"

# run_train.py applies the torchaudio->soundfile shim (no torchcodec/FFmpeg
# on this box), then executes openWakeWord/openwakeword/train.py as __main__.
cd "$TRAIN_DIR"
exec ./venv/bin/python run_train.py \
  --training_config "$CFG" --generate_clips --augment_clips --train_model
