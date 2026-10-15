"""Check the actual launch arguments without starting Ray or allocating GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

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
    python_shim = executable.parent / "python"
    python_shim.write_text(f"#!{sys.executable}\n" + """import json, os, pathlib, sys
if sys.argv[1].endswith('/score_benchmarks.py'):
    pathlib.Path(os.environ['CAPTURE_SCORE_ARGS']).write_text(json.dumps(sys.argv[1:]))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
""")
    python_shim.chmod(0o755)

    def run(script, **overrides):
        env = dict(os.environ)
        for key in (
            "HEAD_MASK",
            "TRAIN_MODE",
            "GENERATION_ONLY",
            "EVAL_DUMP_PATH",
            "EVAL_NAME",
            "HEAD_ORDER_FILE",
            "HEAD_ORDER_DIR",
            "HEAD_SELECTION",
            "FREEZE_RATIO",
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
            CAPTURE_SCORE_ARGS=str(tmp_path / "score-args.json"),
            HF_CHECKPOINT="/models/Qwen3-4B",
            MEGATRON_PATH="/megatron",
            TRAIN_DATA="/data/train.jsonl",
            EVAL_DATA="/data/eval.jsonl",
            SAVE_DIR=str(tmp_path / "checkpoint"),
        )
        env.update(overrides)
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
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


def test_training_defaults_to_evaluating_training_data(launch):
    args = launch("train_qwen3_4b.sh", EVAL_DATA=None)
    index = args.index("--eval-prompt-data")
    assert args[index + 1 : index + 3] == ["validation", "/data/train.jsonl"]
    assert_options(args, {"eval-interval": 10})


def test_full_training_does_not_apply_any_head_freeze(launch):
    args = launch("train_qwen3_4b.sh", TRAIN_MODE="full", HEAD_ORDER_FILE="/missing/order.jsonl")
    assert_formal_settings(args)
    assert "--full-weight-head-freeze-mask-path" not in args
    assert "--zero-frozen-grads-before-clip" not in args


def test_full_attention_needs_no_ranking(launch):
    args = launch("train_qwen3_4b.sh", TRAIN_MODE="full-attn", HEAD_ORDER_FILE="/missing/order.jsonl")
    path = args[args.index("--full-weight-head-freeze-mask-path") + 1]
    assert json.loads(Path(path).read_text())["frozen_heads"] == []
    assert "--zero-frozen-grads-before-clip" in args


@pytest.mark.parametrize(
    "benchmark,rm,generation_only",
    [
        ("math500", "math", False),
        ("ifbench", "ifbench", False),
        ("mbppplus", "math", True),
        ("apps-intro", "math", True),
    ],
)
def test_benchmark_wrapper_dispatches_generation_and_scoring(launch, tmp_path, benchmark, rm, generation_only):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"Test","label":"0"}\n')
    result_dir = tmp_path / "results"
    args = launch("test_qwen3_4b.sh", BENCHMARK=benchmark, EVAL_DATA=str(data), RESULT_DIR=str(result_dir))
    assert_options(
        args,
        {
            "rm-type": rm,
            "n-samples-per-eval-prompt": 1,
            "save-debug-rollout-data": str(result_dir / "{rollout_id}.pt"),
        },
    )
    assert ("--custom-rm-path" in args) == generation_only
    assert args[args.index("--eval-prompt-data") + 1] == benchmark
    score_args = json.loads((tmp_path / "score-args.json").read_text())
    assert_options(
        score_args, {"benchmark": benchmark, "data": data, "dump": result_dir / "eval_0.pt", "output-dir": result_dir}
    )


def test_eval_supports_generation_only_dumps(launch):
    args = launch(
        "eval_qwen3_4b.sh", EVAL_DUMP_PATH="/results/{rollout_id}.pt", EVAL_NAME="apps-intro", GENERATION_ONLY="1"
    )
    assert_options(
        args,
        {
            "save-debug-rollout-data": "/results/{rollout_id}.pt",
            "custom-rm-path": "examples.head_sparse.score_benchmarks.generation_only_reward",
        },
    )
    index = args.index("--eval-prompt-data")
    assert args[index + 1] == "apps-intro"


def test_training_accepts_explicit_evaluation_data(launch):
    args = launch("train_qwen3_4b.sh", EVAL_DATA="/data/other.jsonl")
    index = args.index("--eval-prompt-data")
    assert args[index + 1 : index + 3] == ["validation", "/data/other.jsonl"]


def test_training_can_disable_evaluation_explicitly(launch):
    args = launch("train_qwen3_4b.sh", EVAL_DATA="")
    assert "--eval-prompt-data" not in args


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


def test_top_and_bottom_select_complementary_heads(launch, tmp_path):
    order = str(SCRIPTS / "orders/qwen3-4b-math.jsonl")
    selections = []
    for side in ("top", "bottom"):
        save = tmp_path / side
        launch("train_qwen3_4b.sh", HEAD_ORDER_FILE=order, HEAD_SELECTION=side, SAVE_DIR=str(save))
        mask = json.loads((save / "head-mask.json").read_text())
        selections.append({(h["layer"], h["projection"], h["head"]) for h in mask["frozen_heads"]})
    assert len(selections[0]) == len(selections[1]) == 864
    assert selections[0].isdisjoint(selections[1])


def test_bottom_selection_uses_ranking_directory(launch, tmp_path):
    ranking = tmp_path / "ranking"
    ranking.mkdir()
    (ranking / "head-order.jsonl").write_bytes((SCRIPTS / "orders/qwen3-4b-math.jsonl").read_bytes())
    launch("train_qwen3_4b.sh", HEAD_ORDER_DIR=str(ranking), HEAD_SELECTION="bottom")
    mask = json.loads((tmp_path / "checkpoint/head-mask.json").read_text())
    assert mask["metadata"]["freeze_side"] == "prefix"


def test_round_robin_selections_work_with_training(launch, tmp_path):
    outputs = [tmp_path / f"{task}.json" for task in ("math", "ifeval")]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "round_robin.py"),
            "--task",
            f"math:{SCRIPTS / 'orders/qwen3-4b-math.jsonl'}:{outputs[0]}",
            "--task",
            f"ifeval:{SCRIPTS / 'orders/qwen3-4b-ifeval.jsonl'}:{outputs[1]}",
            "--active-ratio",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen = []
    for output in outputs:
        mask = json.loads(output.read_text())
        frozen.append({(h["layer"], h["projection"], h["head"]) for h in mask["frozen_heads"]})
        args = launch("train_qwen3_4b.sh", HEAD_MASK=str(output))
        assert_options(args, {"full-weight-head-freeze-mask-path": output})
    assert len(frozen[0]) == len(frozen[1]) == 864
    assert frozen[0].isdisjoint(frozen[1])
