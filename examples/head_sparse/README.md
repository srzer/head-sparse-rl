# Sparse-head scripts

Run from the repository root. See the [README](../../README.md) for setup.

| Script | Purpose |
|---|---|
| `rank_qwen3_4b.sh` | Generate a head order. |
| `train_qwen3_4b.sh` | Train selected heads. |
| `eval_qwen3_4b.sh` | Evaluate a checkpoint. |
| `test_qwen3_4b.sh` | Test on MATH-500, IFBench, MBPP+, or APPS-intro. |
| `prepare_eval_data.py` | Download and prepare the test datasets. |

Required: `HF_CHECKPOINT`, `MEGATRON_PATH`.
Defaults: four GPUs, tensor parallel size 2, math reward.

## Ranking

Set `RANK_DATA`, `RANK_BATCH_SIZE` (the number of prompts), and
`HEAD_ORDER_DIR` (a fresh output directory). The head order is saved to
`$HEAD_ORDER_DIR/head-order.jsonl`.

| Option | Default | Meaning |
|---|---|---|
| `RANK_SAMPLES_PER_PROMPT` | `4` | Responses sampled per ranking prompt. |
| `RANK_LEARNING_RATE` | `1e-8` | Learning rate for the single probe update. |

Training uses the base model at `HF_CHECKPOINT` and the generated head order.

## Training

Set `TRAIN_DATA`, `SAVE_DIR`, and `HEAD_ORDER_FILE`.
The order file lists Qwen3-4B heads from most to least important.
Training evaluates on `TRAIN_DATA` by default. Set `EVAL_DATA` to override it.

```bash
HEAD_ORDER_FILE=/path/to/head-order.jsonl FREEZE_RATIO=0.5 \
  bash examples/head_sparse/train_qwen3_4b.sh
```

| Option | Default | Meaning |
|---|---|---|
| `TRAIN_MODE` | `sparse` | `sparse`, `full-attn` (all attention projection weights), or `full` (all model parameters). |
| `HEAD_SELECTION` | `top` | Train the `top` or `bottom` part of the ranking. |
| `FREEZE_RATIO` | `0.5` | Fraction of heads to freeze, so `0.7` trains the selected top or bottom 30%. |
| `NUM_ROLLOUT` | `600` | Number of training rollouts. |
| `ROLLOUT_BATCH_SIZE` | `16` | Prompts per rollout. |
| `SAMPLES_PER_PROMPT` | `4` | Responses per training prompt. |
| `GLOBAL_BATCH_SIZE` | `64` | Training samples per optimizer step. |
| `LEARNING_RATE` | `3e-6` | Training learning rate. |
| `SAVE_INTERVAL` | `200` | Checkpoint save interval. |
| `GPUS_PER_NODE` | `4` | Number of GPUs. |
| `TENSOR_PARALLEL_SIZE` | `2` | Tensor parallel size. |
| `RM_TYPE` | `math` | Reward function that matches the dataset. |

To train the bottom 50% instead:

```bash
HEAD_ORDER_FILE=/path/to/head-order.jsonl HEAD_SELECTION=bottom FREEZE_RATIO=0.5 \
  SAVE_DIR=/root/checkpoints/qwen3-4b-bottom50 \
  bash examples/head_sparse/train_qwen3_4b.sh
```

### Full-attention and full-parameter training

These modes do not require a ranking and ignore head-selection settings:

```bash
TRAIN_MODE=full-attn SAVE_DIR=/root/checkpoints/qwen3-4b-full-attn \
  bash examples/head_sparse/train_qwen3_4b.sh

TRAIN_MODE=full SAVE_DIR=/root/checkpoints/qwen3-4b-full \
  bash examples/head_sparse/train_qwen3_4b.sh
```

### Round-robin

Generate a ranking for each task, then allocate heads by taking turns. Each task
receives its highest-ranked heads that have not already been assigned to another task.
For two tasks with 50% each:

```bash
python examples/head_sparse/round_robin.py \
  --task math:/path/to/math-ranking.jsonl:/root/head-selections/math.json \
  --task ifeval:/path/to/ifeval-ranking.jsonl:/root/head-selections/ifeval.json \
  --active-ratio 0.5
```

Tasks take turns in the order listed. Use the corresponding selection when training
each task, with its own dataset, reward, and output directory:

```bash
HEAD_MASK=/root/head-selections/math.json TRAIN_DATA=/path/to/math-train.jsonl \
  RM_TYPE=math \
  SAVE_DIR=/root/checkpoints/qwen3-4b-math-rr50 \
  bash examples/head_sparse/train_qwen3_4b.sh
```

`HEAD_MASK` takes precedence over `HEAD_ORDER_FILE`, `HEAD_SELECTION`, and
`FREEZE_RATIO`. To allocate among three tasks, add another `--task` and use
`--active-ratio 0.3333333333`. All rankings must refer to the same model.

## Head orders

| Model | Math | MBPP | IFEval |
|---|---|---|---|
| Qwen3-1.7B | [Order](orders/qwen3-1.7b-math.jsonl) | [Order](orders/qwen3-1.7b-mbpp.jsonl) | [Order](orders/qwen3-1.7b-ifeval.jsonl) |
| Qwen3-4B | [Order](orders/qwen3-4b-math.jsonl) | [Order](orders/qwen3-4b-mbpp.jsonl) | [Order](orders/qwen3-4b-ifeval.jsonl) |
| Mistral-7B | — | — | [Order](orders/mistral-7b-ifeval.jsonl) |

For example, to train with the Qwen3-4B Math order:

```bash
HEAD_ORDER_FILE=examples/head_sparse/orders/qwen3-4b-math.jsonl \
  bash examples/head_sparse/train_qwen3_4b.sh
```

## Evaluation

Use `EVAL_DATA` for held-out data, `LOAD_CHECKPOINT` for the checkpoint
directory, and `CKPT_STEP` for the saved iteration.
Omit `LOAD_CHECKPOINT` to evaluate the base model.

For held-out DAPO-Math:

```bash
EVAL_DATA=/path/to/dapo-math-heldout-1000.jsonl RM_TYPE=math \
  LOAD_CHECKPOINT="$SAVE_DIR" CKPT_STEP=399 \
  bash examples/head_sparse/eval_qwen3_4b.sh
```

Set `CKPT_STEP` to the iteration you want to evaluate.

### Benchmark tests

Prepare the datasets once:

```bash
pip install -r examples/head_sparse/requirements_eval.txt
export DATA_DIR=/root/datasets/head-sparse-eval
python examples/head_sparse/prepare_eval_data.py --output-dir "$DATA_DIR"
```

Run any of the four benchmarks with the same interface:

```bash
BENCHMARK=math500 RESULT_DIR=/root/results/qwen3-4b-math500 \
  LOAD_CHECKPOINT="$SAVE_DIR" CKPT_STEP=599 \
  bash examples/head_sparse/test_qwen3_4b.sh
```

| `BENCHMARK` | Scoring |
|---|---|
| `math500` | Math answer verification on all 500 test problems. |
| `ifbench` | Official IFBench strict instruction-following checks. |
| `mbppplus` | EvalPlus base and extended-test pass@1 on 378 problems. |
| `apps-intro` | Pass@1 on 1,000 introductory APPS test problems, requiring all tests to pass. |

Change `BENCHMARK` and use a fresh `RESULT_DIR` for each test. Omit
`LOAD_CHECKPOINT` to test the base model. Results are written to
`$RESULT_DIR/summary.json`, with generations and logs in the same directory.
The default response limit is 4096 tokens, set by `MAX_RESPONSE_LEN`.

Use `--benchmark` when preparing a single dataset. If APPS `test.jsonl` is already
downloaded, pass `--apps-source /path/to/test.jsonl`. `EVAL_DATA` can override a
benchmark's prepared data file.

APPS-intro uses the same stdin/function-call runner as our local evaluations,
with a three-second timeout per test and stripped-output comparison for stdin
programs. MBPP+ is scored by EvalPlus after generation, not by the MBPP training
reward. Code benchmarks execute generated Python, so run them in a disposable
container without credentials or host mounts.
