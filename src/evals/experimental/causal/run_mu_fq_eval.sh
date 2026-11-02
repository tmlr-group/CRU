#!/bin/bash
set -euo pipefail
_CRU_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_CRU_ROOT="$(cd -- "$_CRU_SCRIPT_DIR/../../../.." && pwd)"
cd "$_CRU_ROOT" || exit 1
export PYTHONPATH="$_CRU_ROOT/src:$_CRU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Wrapper: run open-unlearning standard MU/FQ eval on one checkpoint, return summary path.
# Usage: bash run_mu_fq_eval.sh <ckpt_path> <model_name> <output_dir> [forget_split] [holdout_split] [retain_split]
cd "$_CRU_ROOT"
PYTHON="${PYTHON:-python}"

CKPT="$1"
MODEL="$2"           # e.g. Llama-3.2-1B-Instruct
OUTDIR="$3"
FORGET_SPLIT="${4:-forget05}"
HOLDOUT_SPLIT="${5:-holdout05}"
RETAIN_SPLIT="${6:-retain95}"

REF="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
mkdir -p "$OUTDIR"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

$PYTHON src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=$FORGET_SPLIT \
    holdout_split=$HOLDOUT_SPLIT \
    retain_split=$RETAIN_SPLIT \
    forget_perturb_split=${FORGET_SPLIT}_perturbed \
    model=$MODEL \
    task_name=mu_fq_eval \
    model.model_args.pretrained_model_name_or_path="$CKPT" \
    paths.output_dir="$OUTDIR" \
    retain_logs_path="$REF" \
    h=1.0
echo "[eval] result at $OUTDIR/TOFU_SUMMARY.json"
