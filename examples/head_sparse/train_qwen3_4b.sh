#!/usr/bin/env bash
# Full-weight sparse-head RL for Qwen3-4B.

set -euo pipefail

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the base Qwen3-4B directory}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a JSONL or Parquet dataset}"
: "${SAVE_DIR:?Set SAVE_DIR to the output checkpoint directory}"
: "${MEGATRON_PATH:?Set MEGATRON_PATH to the Megatron-LM checkout}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${HEAD_ORDER_FILE:-}" && -z "${HEAD_MASK:-}" ]]; then
  [[ -f "${HEAD_ORDER_FILE}" ]] || { echo "Missing head order: ${HEAD_ORDER_FILE}" >&2; exit 1; }
  HEAD_MASK="${SAVE_DIR}/head-mask.json"
  python "${REPO_ROOT}/miles/utils/build_head_freeze_manifest.py" \
    --order "${HEAD_ORDER_FILE}" \
    --output "${HEAD_MASK}" \
    --freeze-ratio "${FREEZE_RATIO:-0.5}" \
    --freeze-side suffix \
    --head-dim 128 \
    --num-layers 36 \
    --num-attention-heads 32 \
    --num-query-groups 8
elif [[ -n "${HEAD_ORDER_DIR:-}" && -z "${HEAD_MASK:-}" ]]; then
  HEAD_MASK="${HEAD_ORDER_DIR}/head-mask.json"
fi
HEAD_MASK="${HEAD_MASK:-${SCRIPT_DIR}/masks/qwen3-4b-dapo-math-top50.json}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

[[ -f "${HEAD_MASK}" ]] || { echo "Missing head mask: ${HEAD_MASK}" >&2; exit 1; }
source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"

EVAL_ARGS=()
if [[ -n "${EVAL_DATA:-}" ]]; then
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-20}"
    --eval-prompt-data validation "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-4096}"
    --eval-top-k 1
  )
fi

IMPORTANCE_ARGS=()
if [[ -n "${HEAD_GRAD_IMPORTANCE_OUTPUT:-}" ]]; then
  IMPORTANCE_ARGS=(--head-grad-importance-output "${HEAD_GRAD_IMPORTANCE_OUTPUT}")
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
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL:-200}" \
  --no-save-optim \
  --full-weight-head-freeze-mask-path "${HEAD_MASK}" \
  --full-weight-head-freeze-include-o-proj \
  --zero-frozen-grads-before-clip \
  "${IMPORTANCE_ARGS[@]}" \
  --prompt-data "${TRAIN_DATA}" \
  --input-key "${INPUT_KEY:-prompt}" \
  --label-key "${LABEL_KEY:-label}" \
  --apply-chat-template \
  --apply-chat-template-kwargs '{"enable_thinking":false}' \
  --rollout-shuffle \
  --rollout-seed "${ROLLOUT_SEED:-42}" \
  --rm-type "${RM_TYPE:-math}" \
  --num-rollout "${NUM_ROLLOUT:-600}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}" \
  --n-samples-per-prompt "${SAMPLES_PER_PROMPT:-4}" \
  --rollout-max-response-len "${MAX_RESPONSE_LEN:-3072}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-64}" \
  --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --sequence-parallel \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --advantage-estimator grpo \
  --optimizer adam \
  --lr "${LEARNING_RATE:-3e-6}" \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.5}" \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --attention-backend flash \
  "${EVAL_ARGS[@]}"
