#!/usr/bin/env bash
# Shared setup, sourced by the numbered scripts; do not run this file directly.
# Training and evaluation options stay in those scripts.

# Stop the calling script on errors or missing variables.
set -euo pipefail

# Run from the repository root and make its Python modules importable.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Pretrained model IDs for the model configurations shipped with CRU.
case "$MODEL" in
  Phi-3.5-mini-instruct) MODEL_ID="microsoft/$MODEL" ;;
  zephyr-7b-beta) MODEL_ID="HuggingFaceH4/$MODEL" ;;
  *) MODEL_ID="meta-llama/$MODEL" ;;
esac

# Explicit local paths supplied by the caller take precedence.
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MODEL_ID}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_ID}

# Each single-node launch uses a free TCP port rather than shared port 29500.
# Set MAIN_PROCESS_PORT explicitly when the launch environment requires it.
launch_distributed() {
  local port=${MAIN_PROCESS_PORT:-}
  if [[ -z $port ]]; then
    port=$(python - <<'PYPORT'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PYPORT
    )
  fi
  if [[ ! $port =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Invalid MAIN_PROCESS_PORT: $port" >&2
    return 2
  fi
  printf 'DISTRIBUTED_LAUNCH port=%s job=%s\n' "$port" "${SLURM_JOB_ID:-local}" >&2
  accelerate launch --main_process_port "$port" "$@"
}

# Original shell evaluation inputs; preserve full/forget05/retain95 training datasets.
setup_eval_protocol() {
  if [[ "$FORGET_SPLIT" != forget05 || "$RETAIN_SPLIT" != retain95 || "$HOLDOUT_SPLIT" != holdout05 ]]; then
    echo "The original shell evaluation protocol requires forget05/retain95/holdout05." >&2
    return 2
  fi
  export CRU_EVAL_DATA_ROOT=${CRU_EVAL_DATA_ROOT:-$ROOT/datasets/legacy_sh}
  EVAL_ROOT="$OUTPUT_ROOT/legacy_sh"
}
