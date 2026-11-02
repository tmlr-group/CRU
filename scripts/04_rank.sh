#!/usr/bin/env bash
# Stage 4: freeze the complete MU/FQ layer ranking.

# Match the model, method and forget split used in stage 3.
MODEL=${MODEL:-Llama-3.2-1B-Instruct}
METHOD=${METHOD:-NPO}
FORGET_SPLIT=${FORGET_SPLIT:-forget05}

# Read layer scores here; checkpoints may come from a separate run.
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:-$OUTPUT_ROOT}

source "$(dirname "$0")/common.sh"

TASK="tofu_${MODEL}_${FORGET_SPLIT}_${METHOD}"

# Rank every layer and write the complete order for stage 5.
python src/rank.py rank \
  --results-dir "$OUTPUT_ROOT/legacy_sh/replace/$TASK" \
  --base-model "$SOURCE_OUTPUT_ROOT/finetune/tofu_${MODEL}_full" \
  --output "$OUTPUT_ROOT/legacy_sh/rankings/$TASK.json"
