from __future__ import annotations

from typing import Any

from .mbpp import DEFAULT_TIMEOUT_SECONDS, extract_python_code, run_mbpp_tests


def _coerce_humaneval_spec(label: Any, metadata: dict[str, Any] | None) -> dict[str, Any]:
    spec = dict(label) if isinstance(label, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    for key in ("test", "entry_point", "timeout"):
        if key not in spec and key in metadata:
            spec[key] = metadata[key]
    return spec


def compute_humaneval_reward(
    response: str,
    label: Any,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Run an extracted solution against the official HumanEval test function."""

    spec = _coerce_humaneval_spec(label, metadata)
    test = spec.get("test")
    entry_point = spec.get("entry_point")
    if not isinstance(test, str) or not test.strip() or not isinstance(entry_point, str):
        return 0.0

    try:
        timeout = float(spec.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS

    code = extract_python_code(response)
    tests = [test, f"check({entry_point})"]
    return 1.0 if run_mbpp_tests(code, tests, timeout=timeout) else 0.0
