# Sparse-head scripts

Run from the repository root. See the [README](../../README.md) for setup.

| Script | Purpose |
|---|---|
| `rank_qwen3_4b.sh` | Generate a head order. |
| `train_qwen3_4b.sh` | Train selected heads. |
| `eval_qwen3_4b.sh` | Evaluate a checkpoint. |

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
Set `EVAL_DATA` to evaluate during training.

```bash
HEAD_ORDER_FILE=/path/to/head-order.jsonl FREEZE_RATIO=0.5 \
  bash examples/head_sparse/train_qwen3_4b.sh
```

| Option | Default | Meaning |
|---|---|---|
| `FREEZE_RATIO` | `0.5` | Fraction of lowest-ranked heads to freeze; `0.7` trains the top 30%. |
| `NUM_ROLLOUT` | `600` | Number of training rollouts. |
| `ROLLOUT_BATCH_SIZE` | `16` | Prompts per rollout. |
| `SAMPLES_PER_PROMPT` | `4` | Responses per training prompt. |
| `GLOBAL_BATCH_SIZE` | `64` | Training samples per optimizer step. |
| `LEARNING_RATE` | `3e-6` | Training learning rate. |
| `SAVE_INTERVAL` | `200` | Checkpoint save interval. |
| `GPUS_PER_NODE` | `4` | Number of GPUs. |
| `TENSOR_PARALLEL_SIZE` | `2` | Tensor parallel size. |
| `RM_TYPE` | `math` | Reward function; must match the dataset. |

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
