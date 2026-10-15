# Ranking and Partitioning of Attention Heads for Sparse and Continual Reinforcement Learning Fine-tuning

Code for the paper above, based on [Miles](https://github.com/radixark/miles).

Rank attention heads by gradient importance, then train the top-ranked heads.

- [Training and evaluation curves](https://wandb.ai/ruizshi-uw/head-sparse-rl?nw=0c3q4nyqt95)
- [Models, checkpoints, and datasets](RESOURCES.md)
- [Head orders](examples/head_sparse/README.md#head-orders)

## Setup

The Qwen3-4B scripts use four GPUs. Start the container on a host with
NVIDIA Container Toolkit installed:

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network=host \
  -it radixark/miles:sglang-miles-v0.5.12 /bin/bash
```

Inside the container:

```bash
git clone https://github.com/srzer/head-sparse-rl.git /root/head-sparse-rl
cd /root/head-sparse-rl
pip install -e .

hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B

export MEGATRON_PATH=/root/Megatron-LM
export HF_CHECKPOINT=/root/models/Qwen3-4B
export TRAIN_DATA=/path/to/train.jsonl
export EVAL_DATA=/path/to/heldout.jsonl
export SAVE_DIR=/root/checkpoints/qwen3-4b-top50
```

`TRAIN_DATA` and `EVAL_DATA` are JSONL files with `prompt` and `label` fields.
The DAPO-Math runs use `dapo-math-eval-1000.jsonl` for training and
`dapo-math-heldout-1000.jsonl` for evaluation. See [datasets](RESOURCES.md#training-datasets)
for sources. These scripts default to the math reward.

## Rank heads

```bash
export HEAD_ORDER_DIR=/root/head-orders/qwen3-4b-math
RANK_DATA="$TRAIN_DATA" RANK_BATCH_SIZE=1000 \
  bash examples/head_sparse/rank_qwen3_4b.sh
```

Set `RANK_BATCH_SIZE` to the number of prompts in the ranking file. The head
order is saved to `$HEAD_ORDER_DIR/head-order.jsonl`.

We fixed an issue with gradient aggregation after reduce-scatter and verified
that recomputed rankings remain largely the same as the provided rankings,
although they may differ slightly.

## Train

Train the top 50%:

```bash
export HEAD_ORDER_FILE="$HEAD_ORDER_DIR/head-order.jsonl"
bash examples/head_sparse/train_qwen3_4b.sh
```

To use a different order and train the top 30%:

```bash
HEAD_ORDER_FILE=/path/to/another-head-order.jsonl FREEZE_RATIO=0.7 \
  SAVE_DIR=/root/checkpoints/qwen3-4b-other-top30 \
  bash examples/head_sparse/train_qwen3_4b.sh
```

`FREEZE_RATIO` is the fraction of heads to freeze, so `0.5` trains the top 50%
and `0.7` trains the top 30%.

## Evaluate

```bash
LOAD_CHECKPOINT="$SAVE_DIR" CKPT_STEP=599 \
  bash examples/head_sparse/eval_qwen3_4b.sh
```

Set `CKPT_STEP` to the saved iteration (599 for the default run).
Omit `LOAD_CHECKPOINT` to evaluate the base model.

[Script options](examples/head_sparse/README.md).

## License

[Apache License 2.0](LICENSE).
