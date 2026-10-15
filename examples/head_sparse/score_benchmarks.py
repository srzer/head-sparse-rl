"""Score saved benchmark generations. Only load dumps produced by this checkout."""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from miles.rollout.rm_hub.mbpp import extract_python_code


def extract_code(response):
    text = str(response or "").replace("<|im_end|>", "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        compile(text, "<candidate>", "exec")
        return text
    except SyntaxError:
        return extract_python_code(response)


async def generation_only_reward(args, sample, **kwargs):
    """Leave code scoring to the post-generation benchmark evaluator."""
    return [0.0] * len(sample) if isinstance(sample, list) else 0.0


def validate_samples(samples, references):
    expected = [r["metadata"]["task_id"] for r in references]
    actual = [r["metadata"]["task_id"] for r in samples]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual):
        raise ValueError("Expected exactly one sample per distinct task ID")
    if set(actual) != set(expected):
        raise ValueError("Generation task IDs do not match the evaluation dataset")


def score_mbppplus(samples, output_dir, workers):
    output = output_dir / "samples.jsonl"
    with output.open("x") as handle:
        for sample in samples:
            handle.write(
                json.dumps(dict(task_id=sample["metadata"]["task_id"], completion=extract_code(sample["response"])))
                + "\n"
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            "--dataset",
            "mbpp",
            "--samples",
            str(output),
            "--parallel",
            str(workers),
        ],
        check=True,
    )
    result = json.loads((output_dir / "samples_eval_results.json").read_text())["eval"]
    if set(result) != {s["metadata"]["task_id"] for s in samples}:
        raise ValueError("EvalPlus results do not cover exactly the generated tasks")
    records = [rows[0] for rows in result.values()]
    return dict(
        benchmark="mbppplus",
        n=len(records),
        base_pass_at_1=sum(r["base_status"] == "pass" for r in records) / len(records),
        plus_pass_at_1=sum(r["base_status"] == r["plus_status"] == "pass" for r in records) / len(records),
    )


def normalize_output(value):
    return value.strip().replace("\r\n", "\n")


def run_stdin_program(code_path, input_text, timeout):
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(code_path)],
            input=input_text,
            text=True,
            cwd=code_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def run_function_program(code, function_name, args, timeout):
    runner = (
        "import json, sys, typing\n"
        f"code = {code!r}\n"
        "namespace = dict(vars(typing))\n"
        "exec(code, namespace)\n"
        f"fn = getattr(namespace['Solution'](), {function_name!r}) if 'Solution' in namespace else namespace[{function_name!r}]\n"
        "print(json.dumps(fn(*json.loads(sys.stdin.read()))))\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="apps_function_") as tmp:
            result = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                input=json.dumps(args),
                text=True,
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        return (True, json.loads(result.stdout)) if result.returncode == 0 else (False, None)
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return False, None


def score_apps_problem(sample, spec, timeout=3.0):
    code = extract_code(sample["response"])
    inputs, outputs = spec.get("inputs", []), spec.get("outputs", [])
    if not code or not inputs or len(inputs) != len(outputs):
        return False
    if spec.get("fn_name"):
        for args, expected in zip(inputs, outputs, strict=True):
            ok, actual = run_function_program(code, spec["fn_name"], args, timeout)
            if not ok or actual != expected:
                return False
        return True
    with tempfile.TemporaryDirectory(prefix="apps_intro_") as tmp:
        code_path = Path(tmp) / "solution.py"
        code_path.write_text(code)
        for input_text, expected in zip(inputs, outputs, strict=True):
            actual = run_stdin_program(code_path, input_text, timeout)
            if actual is None or normalize_output(actual) != normalize_output(expected):
                return False
    return True


def score_apps_intro(samples, references, workers, timeout):
    specs = {r["metadata"]["task_id"]: r["label"] for r in references}

    def score(sample):
        return score_apps_problem(sample, specs[sample["metadata"]["task_id"]], timeout)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        passed = list(pool.map(score, samples))
    return dict(benchmark="apps-intro", n=len(passed), passed=sum(passed), pass_at_1=sum(passed) / len(passed))


def score_rewards(benchmark, samples):
    rewards = [float(sample["reward"]) for sample in samples]
    if not rewards or not all(math.isfinite(v) and 0 <= v <= 1 for v in rewards):
        raise ValueError("Missing or invalid benchmark rewards")
    return dict(benchmark=benchmark, n=len(rewards), accuracy=sum(rewards) / len(rewards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("math500", "ifbench", "mbppplus", "apps-intro"), required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    if args.workers <= 0 or args.timeout <= 0:
        parser.error("--workers and --timeout must be positive")
    import torch

    samples = torch.load(args.dump, map_location="cpu", weights_only=False)["samples"]
    with args.data.open() as handle:
        references = [json.loads(line) for line in handle if line.strip()]
    if not samples or len(samples) != len(references):
        raise ValueError("Expected one generated response for every evaluation prompt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "summary.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    if args.benchmark in ("mbppplus", "apps-intro"):
        validate_samples(samples, references)
        if args.benchmark == "mbppplus":
            result = score_mbppplus(samples, args.output_dir, args.workers)
        else:
            result = score_apps_intro(samples, references, args.workers, args.timeout)
    else:
        result = score_rewards(args.benchmark, samples)
    with result_path.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
