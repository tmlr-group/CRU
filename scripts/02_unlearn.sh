#!/usr/bin/env bash
# Stage 2: unlearn from Full FT using the selected method.

# Model and unlearning method.
MODEL=${MODEL:-Llama-3.2-1B-Instruct}
METHOD=${METHOD:-NPO}

# Data splits.
FORGET_SPLIT=${FORGET_SPLIT:-forget05}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain95}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout05}

# Training and output settings.
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
BATCH_SIZE=${BATCH_SIZE:-4}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}

source "$(dirname "$0")/common.sh"
setup_eval_protocol

# Read Full FT and retain-reference scores produced by stage 1.
TASK="tofu_${MODEL}_${FORGET_SPLIT}_${METHOD}"
FULL_FT="$OUTPUT_ROOT/finetune/tofu_${MODEL}_full"
RETAIN_LOGS="$OUTPUT_ROOT/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

# Train the selected unlearning method and save its checkpoint.
launch_distributed \
  --config_file "$ACCELERATE_CONFIG" \
  src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer="$METHOD" \
  model="$MODEL" \
  task_name="$TASK" \
  forget_split="$FORGET_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  holdout_split="$HOLDOUT_SPLIT" \
  model.model_args.pretrained_model_name_or_path="$FULL_FT" \
  retain_logs_path="$RETAIN_LOGS" \
  paths.output_dir="$OUTPUT_ROOT/unlearn/$TASK" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  trainer.args.per_device_train_batch_size="$BATCH_SIZE" \
  "$@"

# Final baseline evaluation uses held-out data; training above is unchanged.
python src/eval.py experiment=eval/tofu/default model="$MODEL" task_name="$TASK" \
  forget_split=forget01 forget_perturb_split=forget01_perturbed \
  retain_split=retain99 retain_perturb_split=retain_perturbed_80 holdout_split=holdout01 \
  model.model_args.pretrained_model_name_or_path="$OUTPUT_ROOT/unlearn/$TASK" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  retain_logs_path="$OUTPUT_ROOT/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
  paths.output_dir="$EVAL_ROOT/eval/$TASK/test"
