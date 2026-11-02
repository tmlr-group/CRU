#!/usr/bin/env bash
# Stage 5: evaluate cumulative Merge and plot every measured k.

# Match the model and method used for unlearning.
MODEL=${MODEL:-Llama-3.2-1B-Instruct}
METHOD=${METHOD:-NPO}

# Data splits.
FORGET_SPLIT=${FORGET_SPLIT:-forget05}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain95}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout05}

# Output directory, source checkpoints and optional weight saving.
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$OUTPUT_ROOT}
SAVE_MODELS=${SAVE_MODELS:-false}  # true saves a complete model for every k.

source "$(dirname "$0")/common.sh"
setup_eval_protocol

# Read Full FT and the retain-reference scores.
TASK="tofu_${MODEL}_${FORGET_SPLIT}_${METHOD}"

FULL_FT="$SOURCE_OUTPUT_ROOT/finetune/tofu_${MODEL}_full"
RETAIN_LOGS="$SOURCE_OUTPUT_ROOT/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

# Add layers in ranked order; reuse completed points when resuming.
python src/merge_sweep.py \
  experiment=eval/tofu/default \
  model="$MODEL" \
  task_name="$TASK" \
  forget_split=forget01 \
  retain_split=retain99 \
  retain_perturb_split=retain_perturbed_80 \
  holdout_split=holdout01 \
  forget_perturb_split=forget01_perturbed \
  model.model_args.pretrained_model_name_or_path="$FULL_FT" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  unlearn_model="$SOURCE_OUTPUT_ROOT/unlearn/$TASK" \
  retain_logs_path="$RETAIN_LOGS" \
  ranking="$EVAL_ROOT/rankings/$TASK.json" \
  paths.output_dir="$EVAL_ROOT/merge_sweep/$TASK" \
  save_models="$SAVE_MODELS"

# Plot all measured MU/FQ points; no threshold or best-k selection.
python visualization/curves/plot_merge.py \
  --results-dir "$EVAL_ROOT/merge_sweep/$TASK"
