#!/usr/bin/env bash
# Rank Qwen3-4B GQA head groups by the RL gradient norm and build a mask.

set -euo pipefail

: "${RANK_DATA:?Set RANK_DATA to the ranking JSONL or Parquet dataset}"
RANK_OUTPUT_DIR="${RANK_OUTPUT_DIR:-${HEAD_ORDER_DIR:-}}"
: "${RANK_OUTPUT_DIR:?Set HEAD_ORDER_DIR to the head-ranking output directory}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the base Qwen3-4B directory}"
: "${MEGATRON_PATH:?Set MEGATRON_PATH to the Megatron-LM checkout}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${RANK_OUTPUT_DIR}"

for output in head-grad-importance.jsonl head-order.jsonl head-mask.json; do
  if [[ -e "${RANK_OUTPUT_DIR}/${output}" ]]; then
    echo "Refusing to append to existing ranking output: ${RANK_OUTPUT_DIR}/${output}" >&2
    exit 1
  fi
done

export TRAIN_DATA="${RANK_DATA}"
export SAVE_DIR="${RANK_OUTPUT_DIR}/probe-checkpoint"
export HEAD_MASK="${SCRIPT_DIR}/masks/qwen3-4b-all-heads-trainable.json"
export HEAD_GRAD_IMPORTANCE_OUTPUT="${RANK_OUTPUT_DIR}/head-grad-importance.jsonl"
export NUM_ROLLOUT=1
export ROLLOUT_BATCH_SIZE="${RANK_BATCH_SIZE:-1000}"
export SAMPLES_PER_PROMPT="${RANK_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))
export LEARNING_RATE="${RANK_LEARNING_RATE:-1e-8}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-999999}"

bash "${SCRIPT_DIR}/train_qwen3_4b.sh"

python "${REPO_ROOT}/miles/utils/build_head_order_from_grad_importance.py" \
  --grad-importance "${HEAD_GRAD_IMPORTANCE_OUTPUT}" \
  --output "${RANK_OUTPUT_DIR}/head-order.jsonl" \
  --q-per-group 4

python "${REPO_ROOT}/miles/utils/build_head_freeze_manifest.py" \
  --order "${RANK_OUTPUT_DIR}/head-order.jsonl" \
  --output "${RANK_OUTPUT_DIR}/head-mask.json" \
  --freeze-ratio "${FREEZE_RATIO:-0.5}" \
  --freeze-side suffix \
  --head-dim 128 \
  --num-layers 36 \
  --num-attention-heads 32 \
  --num-query-groups 8
