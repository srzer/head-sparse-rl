#!/usr/bin/env bash
# Evaluate a base model or a Megatron checkpoint produced by train_qwen3_4b.sh.

set -euo pipefail

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the base Qwen3-4B directory}"
: "${EVAL_DATA:?Set EVAL_DATA to a JSONL or Parquet dataset}"
: "${MEGATRON_PATH:?Set MEGATRON_PATH to the Megatron-LM checkout}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"

LOAD_ARGS=()
if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
  LOAD_ARGS=(--load "${LOAD_CHECKPOINT}" --finetune)
  if [[ -n "${CKPT_STEP:-}" ]]; then
    LOAD_ARGS+=(--ckpt-step "${CKPT_STEP}")
  fi
fi

if ! ray job list --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
  ray start --head \
    --node-ip-address="${MASTER_ADDR:-127.0.0.1}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=127.0.0.1 \
    --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"
fi

ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_PATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
  --colocate \
  --calculate-per-token-loss \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --megatron-to-hf-mode bridge \
  "${LOAD_ARGS[@]}" \
  --prompt-data "${EVAL_DATA}" \
  --input-key "${INPUT_KEY:-prompt}" \
  --label-key "${LABEL_KEY:-label}" \
  --apply-chat-template \
  --apply-chat-template-kwargs '{"enable_thinking":false}' \
  --rm-type "${RM_TYPE:-math}" \
  --num-rollout 0 \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}" \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len "${MAX_RESPONSE_LEN:-4096}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-16}" \
  --eval-interval 1 \
  --eval-prompt-data evaluation "${EVAL_DATA}" \
  --n-samples-per-eval-prompt "${SAMPLES_PER_PROMPT:-1}" \
  --eval-max-response-len "${MAX_RESPONSE_LEN:-4096}" \
  --eval-top-k 1 \
  --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --sequence-parallel \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --advantage-estimator grpo \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.5}" \
  --attention-backend flash
