#!/usr/bin/env bash
# Stage 1: train Full FT and the separate retain-only reference.

# Pretrained model.
MODEL=${MODEL:-Llama-3.2-1B-Instruct}

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
python src/select/prepare_tofu_data.py --output-dir "$CRU_EVAL_DATA_ROOT"

# Retain checkpoint produced below and used for the reference evaluation.
RETAIN_MODEL="$OUTPUT_ROOT/finetune/tofu_${MODEL}_${RETAIN_SPLIT}"

# 1. Full-data fine-tuning starts from the pretrained model.
launch_distributed \
  --config_file "$ACCELERATE_CONFIG" \
  src/train.py \
  experiment=finetune/tofu/default \
  forget_split="$FORGET_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  holdout_split="$HOLDOUT_SPLIT" \
  model="$MODEL" \
  task_name="tofu_${MODEL}_full" \
  data/datasets@data.train=TOFU_QA_full \
  data.train.TOFU_QA_full.args.hf_args.name=full \
  model.model_args.pretrained_model_name_or_path="$BASE_MODEL_PATH" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  paths.output_dir="$OUTPUT_ROOT/finetune/tofu_${MODEL}_full" \
  trainer.args.per_device_train_batch_size="$BATCH_SIZE" \
  "$@"

# 2. Retain-only fine-tuning also starts from the pretrained model.
launch_distributed \
  --config_file "$ACCELERATE_CONFIG" \
  src/train.py \
  experiment=finetune/tofu/default \
  forget_split="$FORGET_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  holdout_split="$HOLDOUT_SPLIT" \
  model="$MODEL" \
  task_name="tofu_${MODEL}_${RETAIN_SPLIT}" \
  data/datasets@data.train=TOFU_QA_retain \
  data.train.TOFU_QA_retain.args.hf_args.name="$RETAIN_SPLIT" \
  model.model_args.pretrained_model_name_or_path="$BASE_MODEL_PATH" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  paths.output_dir="$RETAIN_MODEL" \
  trainer.args.per_device_train_batch_size="$BATCH_SIZE" \
  "$@"

# 3. Evaluate the retain-only model to obtain the FQ reference logs.
python src/eval.py \
  experiment=eval/tofu/default \
  model="$MODEL" \
  task_name="tofu_${MODEL}_${RETAIN_SPLIT}" \
  forget_split="$FORGET_SPLIT" \
  holdout_split="$HOLDOUT_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  forget_perturb_split="${FORGET_SPLIT}_perturbed" \
  model.model_args.pretrained_model_name_or_path="$RETAIN_MODEL" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
  paths.output_dir="$OUTPUT_ROOT/eval/tofu_${MODEL}_${RETAIN_SPLIT}"

# Original retrain test: retain95 weights, forget01 and retain_perturbed_80.
# retain_split=retain99 is an evaluator option, not the shell checkpoint variable.
for BASELINE in "tofu_${MODEL}_${RETAIN_SPLIT}" "tofu_${MODEL}_full"; do
  python src/eval.py experiment=eval/tofu/default model="$MODEL" task_name="$BASELINE" \
    forget_split=forget01 forget_perturb_split=forget01_perturbed \
    retain_split=retain99 retain_perturb_split=retain_perturbed_80 holdout_split=holdout01 \
    model.model_args.pretrained_model_name_or_path="$OUTPUT_ROOT/finetune/$BASELINE" \
    model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_PATH" \
    retain_logs_path="$OUTPUT_ROOT/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
    paths.output_dir="$EVAL_ROOT/eval/$BASELINE/test"
done
