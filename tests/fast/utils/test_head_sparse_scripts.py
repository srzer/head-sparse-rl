"""Check the actual launch arguments without starting Ray or allocating GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-fast")

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "examples/head_sparse"


@pytest.fixture
def launch(tmp_path):
    executable = tmp_path / "bin/ray"
    executable.parent.mkdir()
    executable.write_text(f"#!{sys.executable}\n" + """import json, os, pathlib, sys
args = sys.argv[1:]
if args[:2] == ["job", "list"]:
    raise SystemExit(0)
assert args[:2] == ["job", "submit"], args
pathlib.Path(os.environ["CAPTURE_RAY_ARGS"]).write_text(json.dumps(args))
if "--head-grad-importance-output" in args:
    output = pathlib.Path(args[args.index("--head-grad-importance-output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for layer in range(36):
            for group in range(8):
                handle.write(json.dumps(dict(layer=layer, group=group,
                    grad_norm=float(288 - layer * 8 - group))) + "\\n")
""")
    executable.chmod(0o755)

    def run(script, **overrides):
        env = dict(os.environ)
        for key in (
            "HEAD_MASK",
            "HEAD_ORDER_FILE",
            "HEAD_ORDER_DIR",
            "RANK_OUTPUT_DIR",
            "HEAD_GRAD_IMPORTANCE_OUTPUT",
            "NUM_ROLLOUT",
            "ROLLOUT_BATCH_SIZE",
            "SAMPLES_PER_PROMPT",
            "GLOBAL_BATCH_SIZE",
            "LEARNING_RATE",
            "EVAL_INTERVAL",
            "SKIP_EVAL_BEFORE_TRAIN",
            "GPUS_PER_NODE",
            "TENSOR_PARALLEL_SIZE",
            "ROLLOUT_SEED",
        ):
            env.pop(key, None)
        capture = tmp_path / "args.json"
        env.update(
            PATH=str(executable.parent) + os.pathsep + env["PATH"],
            CAPTURE_RAY_ARGS=str(capture),
            HF_CHECKPOINT="/models/Qwen3-4B",
            MEGATRON_PATH="/megatron",
            TRAIN_DATA="/data/train.jsonl",
            EVAL_DATA="/data/eval.jsonl",
            SAVE_DIR=str(tmp_path / "checkpoint"),
            **overrides,
        )
        subprocess.run(["bash", str(SCRIPTS / script)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        args = json.loads(capture.read_text())
        assert str(ROOT / "train.py") in args
        runtime = json.loads(next(a.split("=", 1)[1] for a in args if a.startswith("--runtime-env-json=")))
        assert runtime["env_vars"] == {
            "PYTHONPATH": f"{ROOT}:/megatron",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        return args

    return run


def assert_options(args, expected):
    for name, value in expected.items():
        assert args[args.index("--" + name) + 1] == str(value), name


def assert_formal_settings(args):
    assert_options(
        args,
        {
            "seed": 1234,
            "rollout-seed": 42,
            "rollout-temperature": 1,
            "tensor-model-parallel-size": 2,
            "pipeline-model-parallel-size": 1,
            "context-parallel-size": 1,
            "expert-model-parallel-size": 1,
            "expert-tensor-parallel-size": 1,
            "max-tokens-per-gpu": 4096,
            "advantage-estimator": "grpo",
            "eps-clip": 0.2,
            "eps-clip-high": 0.28,
            "kl-loss-coef": 0,
            "kl-coef": 0,
            "entropy-coef": 0,
            "optimizer": "adam",
            "lr-decay-style": "constant",
            "weight-decay": 0.1,
            "adam-beta1": 0.9,
            "adam-beta2": 0.98,
            "rollout-num-gpus-per-engine": 1,
            "sglang-mem-fraction-static": 0.5,
            "sglang-attention-backend": "flashinfer",
            "attention-backend": "flash",
            "attention-dropout": 0,
            "hidden-dropout": 0,
        },
    )
    for flag in (
        "--calculate-per-token-loss",
        "--sequence-parallel",
        "--use-dynamic-batch-size",
        "--deterministic-mode",
        "--sglang-enable-deterministic-inference",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
    ):
        assert flag in args


def test_training_matches_formal_qwen4b_settings(launch):
    args = launch("train_qwen3_4b.sh")
    assert_formal_settings(args)
    assert_options(
        args,
        {
            "num-rollout": 600,
            "rollout-batch-size": 16,
            "n-samples-per-prompt": 4,
            "global-batch-size": 64,
            "rollout-max-response-len": 3072,
            "lr": "3e-6",
            "eval-interval": 10,
            "eval-max-response-len": 4096,
            "eval-top-k": 1,
            "recompute-granularity": "full",
            "recompute-method": "uniform",
            "recompute-num-layers": 1,
            "kl-loss-type": "low_var_kl",
        },
    )


def test_ranking_inherits_formal_settings_without_evaluation(launch, tmp_path):
    output = tmp_path / "ranking"
    args = launch("rank_qwen3_4b.sh", RANK_DATA="/data/rank.jsonl", HEAD_ORDER_DIR=str(output))
    assert_formal_settings(args)
    assert_options(
        args,
        {
            "num-rollout": 1,
            "rollout-batch-size": 1000,
            "n-samples-per-prompt": 4,
            "global-batch-size": 4000,
            "lr": "1e-8",
            "prompt-data": "/data/rank.jsonl",
            "recompute-granularity": "full",
            "recompute-method": "uniform",
            "recompute-num-layers": 1,
        },
    )
    assert "--skip-eval-before-train" in args
    assert "--eval-prompt-data" not in args
    assert len((output / "head-order.jsonl").read_text().splitlines()) == 1728
    assert len(json.loads((output / "head-mask.json").read_text())["frozen_heads"]) == 864


def test_eval_matches_formal_settings_and_has_nonzero_schedule(launch):
    args = launch("eval_qwen3_4b.sh", LOAD_CHECKPOINT="/checkpoint", CKPT_STEP="399")
    assert_formal_settings(args)
    assert_options(
        args,
        {
            "num-rollout": 0,
            "load": "/checkpoint",
            "ckpt-step": 399,
            "lr-decay-iters": 1,
            "lr": "1e-6",
            "eval-top-k": 1,
        },
    )
    assert "--finetune" in args
