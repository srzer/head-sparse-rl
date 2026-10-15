#!/usr/bin/env bash
# Evaluate a base model or checkpoint on an additional benchmark.
set -euo pipefail

: "${BENCHMARK:?Set BENCHMARK to math500, ifbench, mbppplus, or apps-intro}"
: "${RESULT_DIR:?Set RESULT_DIR to a fresh output directory}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/root/datasets/head-sparse-eval}"
export EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/${BENCHMARK}.jsonl}"
export EVAL_NAME="${BENCHMARK}"
export SAMPLES_PER_PROMPT=1
export GENERATION_ONLY=0
case "${BENCHMARK}" in
  math500) export RM_TYPE=math ;;
  ifbench) export RM_TYPE=ifbench ;;
  mbppplus|apps-intro) export RM_TYPE=math GENERATION_ONLY=1 ;;
  *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 1 ;;
esac
[[ -f "${EVAL_DATA}" ]] || { echo "Missing evaluation data: ${EVAL_DATA}" >&2; exit 1; }
[[ ! -e "${RESULT_DIR}" ]] || { echo "Refusing to reuse RESULT_DIR: ${RESULT_DIR}" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"
export EVAL_DUMP_PATH="${RESULT_DIR}/{rollout_id}.pt"
bash "${SCRIPT_DIR}/eval_qwen3_4b.sh" 2>&1 | tee "${RESULT_DIR}/generation.log"
python "${SCRIPT_DIR}/score_benchmarks.py" \
  --benchmark "${BENCHMARK}" --dump "${RESULT_DIR}/eval_0.pt" \
  --data "${EVAL_DATA}" --output-dir "${RESULT_DIR}"
