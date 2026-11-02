#!/usr/bin/env bash
# Stage 3: evaluate one CRU replacement candidate for every transformer layer.

# Match the model and method used for unlearning.
MODEL=${MODEL:-Llama-3.2-1B-Instruct}
METHOD=${METHOD:-NPO}

# Data splits.
FORGET_SPLIT=${FORGET_SPLIT:-forget05}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain95}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout05}

# Output directory and source checkpoints.
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$OUTPUT_ROOT}

source "$(dirname "$0")/common.sh"
setup_eval_protocol
# Also support starting from existing checkpoints without rerunning stage 1.
python src/select/prepare_tofu_data.py --output-dir "$CRU_EVAL_DATA_ROOT"

# Read Full FT, Unlearn and the retain-reference scores.
TASK="tofu_${MODEL}_${FORGET_SPLIT}_${METHOD}"
FULL_FT="$SOURCE_OUTPUT_ROOT/finetune/tofu_${MODEL}_full"
UNLEARNED="$SOURCE_OUTPUT_ROOT/unlearn/$TASK"
RETAIN_LOGS="$SOURCE_OUTPUT_ROOT/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

# Test each layer independently, restoring Full FT between candidates.
python src/replace_sweep.py \
  experiment=eval/tofu/default \
  model="$MODEL" \
  task_name="$TASK" \
  forget_split=forget04 \
  retain_split=holdout04 \
  retain_perturb_split=retain_perturbed_320 \
  holdout_split=holdout04 \
  model.model_args.pretrained_model_name_or_path="$FULL_FT" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  unlearn_model="$UNLEARNED" \
  retain_logs_path="$RETAIN_LOGS" \
  paths.output_dir="$EVAL_ROOT/replace/$TASK" \
  forget_perturb_split=forget04_perturbed
