from __future__ import annotations

import importlib.util
from pathlib import Path

from miles.rollout.rm_hub.mbpp import compute_mbpp_reward


def _load_builder_module():
    path = Path(__file__).resolve().parents[4] / "scripts" / "build_same_source_heldout_jsonl.py"
    spec = importlib.util.spec_from_file_location("build_same_source_heldout_jsonl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_unseen_records_excludes_used_prompts_deterministically() -> None:
    builder = _load_builder_module()
    candidates = [{"prompt": [{"role": "user", "content": str(i)}], "label": str(i)} for i in range(6)]
    used = [candidates[1], candidates[4]]

    first = builder.select_unseen_records(candidates, used, size=3, seed=7)
    second = builder.select_unseen_records(candidates, used, size=3, seed=7)

    assert first == second
    assert not ({builder.prompt_key(record) for record in first} & {builder.prompt_key(record) for record in used})


def test_mbpp_challenge_builder_uses_only_ignored_challenge_tests() -> None:
    builder = _load_builder_module()
    row = {
        "task_id": 1,
        "text": "Return the sum of two values.",
        "code": "def add(a, b): return a + b",
        "test_list": ["assert add(1, 2) == 3"],
        "challenge_test_list": ["assert add(-5, 2) == -3"],
        "test_setup_code": "",
    }
    records = builder.build_mbpp_challenge({"test": [row]})
    record = records[0]

    assert record["label"]["tests"] == row["challenge_test_list"]
    assert record["metadata"]["heldout_component"] == "challenge_test_list"
    assert compute_mbpp_reward("def add(a, b): return a + b", record["label"], record["metadata"]) == 1.0
