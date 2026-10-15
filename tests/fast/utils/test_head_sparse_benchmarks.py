"""Dataset conversion and post-generation benchmark scoring."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from examples.head_sparse.prepare_eval_data import apps_records, math_records, mbpp_records
from examples.head_sparse.score_benchmarks import (
    extract_code,
    generation_only_reward,
)
from examples.head_sparse.score_benchmarks import main as score_main
from examples.head_sparse.score_benchmarks import (
    score_apps_intro,
    score_apps_problem,
    score_mbppplus,
    score_rewards,
    validate_samples,
)
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-fast")


def test_math_and_mbpp_data_preserve_labels_and_task_ids():
    rows = math_records([dict(problem="2+2?", answer="4", unique_id="x", subject="Algebra", level=1)])
    assert rows[0]["label"] == "4"
    assert "2+2?" in rows[0]["prompt"][0]["content"]
    rows = mbpp_records({"Mbpp/1": dict(prompt="Add two numbers", entry_point="add")})
    assert rows[0]["metadata"]["task_id"] == "Mbpp/1"
    assert rows[0]["label"] == {}


def test_apps_conversion_keeps_real_tests():
    spec = dict(inputs=["1\n"], outputs=["1\n"])
    problem = dict(
        id=1, difficulty="introductory", question="Echo input.", starter_code="", input_output=json.dumps(spec)
    )
    rows = apps_records([problem, {**problem, "id": 2, "difficulty": "competition"}])
    assert len(rows) == 1
    assert rows[0]["label"] == spec
    with pytest.raises(ValueError, match="mismatched"):
        apps_records([{**problem, "input_output": "{}"}])


def test_code_scoring_rejects_missing_or_duplicate_tasks():
    references = [dict(metadata=dict(task_id=1)), dict(metadata=dict(task_id=2))]
    validate_samples(list(reversed(references)), references)
    with pytest.raises(ValueError, match="match"):
        validate_samples(references[:1], references)
    with pytest.raises(ValueError, match="distinct"):
        validate_samples([references[0], references[0]], references)


def test_code_extraction_preserves_top_level_statements():
    code = "offset = 1\ndef f(x): return x + offset\nprint(f(1))"
    assert extract_code(code) == code
    assert extract_code(f"```python\n{code}\n```") == code


def test_apps_stdio_pass_fail_timeout_and_missing_tests():
    spec = dict(inputs=["2 3\n", "4 6\n"], outputs=["5\n", "10\n"])
    good = dict(response="```python\na,b=map(int,input().split())\nprint(a+b)\n```")
    assert score_apps_problem(good, spec)
    assert not score_apps_problem(dict(response="print(0)"), spec)
    assert not score_apps_problem(dict(response="while True: pass"), spec, timeout=0.1)
    assert not score_apps_problem(good, {})
    assert not score_apps_problem(good, dict(inputs=["a"], outputs=[]))


def test_apps_function_calls_and_summary():
    spec = dict(fn_name="add", inputs=[[2, 3], [4, 6]], outputs=[5, 10])
    good = dict(response="class Solution:\n def add(self,a,b): return a+b", metadata=dict(task_id=1))
    assert score_apps_problem(good, spec)
    assert not score_apps_problem(dict(response="raise ValueError()"), spec)
    result = score_apps_intro([good], [dict(metadata=dict(task_id=1), label=spec)], 1, 3.0)
    assert result["pass_at_1"] == 1


def test_math_ifbench_metrics_are_actual_rewards():
    assert score_rewards("math500", [dict(reward=1), dict(reward=0)])["accuracy"] == 0.5
    with pytest.raises(ValueError, match="invalid"):
        score_rewards("ifbench", [dict(reward=float("nan"))])


def test_generation_only_reward_does_not_execute_code():
    assert asyncio.run(generation_only_reward(None, dict(response="raise RuntimeError()"))) == 0
    assert asyncio.run(generation_only_reward(None, [{}, {}])) == [0, 0]


def test_mbppplus_uses_evalplus_extended_results(tmp_path, monkeypatch):
    def evaluate(command, **kwargs):
        assert command[1:5] == ["-m", "evalplus.evaluate", "--dataset", "mbpp"]
        result = {"eval": {"Mbpp/1": [dict(base_status="pass", plus_status="fail")]}}
        (tmp_path / "samples_eval_results.json").write_text(json.dumps(result))

    monkeypatch.setattr(subprocess, "run", evaluate)
    result = score_mbppplus([dict(metadata=dict(task_id="Mbpp/1"), response="def f(): return 1")], tmp_path, 1)
    assert result["base_pass_at_1"] == 1
    assert result["plus_pass_at_1"] == 0


@pytest.mark.parametrize("benchmark", ["math500", "ifbench", "apps-intro"])
def test_saved_generation_to_summary(benchmark, tmp_path, monkeypatch):
    import torch

    reference = dict(metadata=dict(task_id=1), label=dict(inputs=["2\n"], outputs=["2\n"]))
    sample = dict(metadata=dict(task_id=1), response="print(input())", reward=1.0)
    data, dump, output = tmp_path / "data.jsonl", tmp_path / "eval_0.pt", tmp_path / "result"
    data.write_text(json.dumps(reference) + "\n")
    torch.save(dict(samples=[sample]), dump)
    monkeypatch.setattr(
        sys,
        "argv",
        ["score", "--benchmark", benchmark, "--dump", str(dump), "--data", str(data), "--output-dir", str(output)],
    )
    score_main()
    result = json.loads((output / "summary.json").read_text())
    assert result["benchmark"] == benchmark
    assert result.get("accuracy", result.get("pass_at_1")) == 1.0
