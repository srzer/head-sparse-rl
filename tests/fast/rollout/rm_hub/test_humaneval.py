from miles.rollout.rm_hub.humaneval import compute_humaneval_reward


LABEL = {
    "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(-3) == -6\n",
    "entry_point": "double",
    "timeout": 2,
}


def test_humaneval_accepts_correct_fenced_solution():
    response = "```python\ndef double(value):\n    return value * 2\n```"
    assert compute_humaneval_reward(response, LABEL) == 1.0


def test_humaneval_rejects_incorrect_solution():
    assert compute_humaneval_reward("def double(value):\n    return value", LABEL) == 0.0


def test_humaneval_rejects_missing_spec():
    assert compute_humaneval_reward("def double(value): return value * 2", None) == 0.0
