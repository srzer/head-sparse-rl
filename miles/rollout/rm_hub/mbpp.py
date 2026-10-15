from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 5.0


def _strip_chat_artifacts(text: str) -> str:
    text = re.sub(r"<\|im_start\|>\s*assistant\s*", "", text)
    text = re.sub(r"<\|im_start\|>\s*user\s*.*?<\|im_end\|>", "", text, flags=re.DOTALL)
    text = text.replace("<|im_end|>", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _trim_to_code_start(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if re.match(r"(from\s+\S+\s+import\s+|import\s+|def\s+\w+\s*\(|class\s+\w+\s*[:(]|@)", stripped):
            return "\n".join(lines[index:]).strip()
    return text.strip()


def extract_python_code(response: str | None) -> str:
    text = _strip_chat_artifacts(str(response or "").strip())
    if not text:
        return ""

    fenced_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        return fenced_blocks[0].strip()

    generic_blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[0].strip()

    return _trim_to_code_start(text)


def _loads_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_string_list(value: Any) -> list[str]:
    value = _loads_json_maybe(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _coerce_mbpp_spec(label: Any, metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    label = _loads_json_maybe(label)
    spec: dict[str, Any] = {}
    if isinstance(label, dict):
        spec.update(label)
    elif isinstance(label, (list, tuple, str)):
        spec["tests"] = label

    for key in ("tests", "test_list", "test_imports", "timeout"):
        if key not in spec and key in metadata:
            spec[key] = metadata[key]
    return spec


def _build_runner_source(code: str, test_imports: list[str], tests: list[str]) -> str:
    return "\n".join(
        [
            "import json",
            "import traceback",
            "",
            "candidate_code = " + repr(code),
            "test_imports = " + repr(test_imports),
            "tests = " + repr(tests),
            "namespace = {}",
            "try:",
            "    for statement in test_imports:",
            "        exec(statement, namespace)",
            "    exec(candidate_code, namespace)",
            "    for statement in tests:",
            "        exec(statement, namespace)",
            "except BaseException as exc:",
            "    print(json.dumps({'passed': False, 'error': type(exc).__name__, 'message': str(exc)}))",
            "    raise SystemExit(1)",
            "print(json.dumps({'passed': True}))",
            "",
        ]
    )


def run_mbpp_tests(
    code: str,
    tests: list[str],
    *,
    test_imports: list[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    if not code.strip() or not tests:
        return False

    test_imports = test_imports or []
    runner_source = _build_runner_source(code, test_imports, tests)
    with tempfile.TemporaryDirectory(prefix="miles_mbpp_") as tmpdir:
        runner_path = Path(tmpdir) / "run_mbpp_candidate.py"
        runner_path.write_text(runner_source, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(runner_path)],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def compute_mbpp_reward(response: str, label: Any, metadata: dict[str, Any] | None = None) -> float:
    spec = _coerce_mbpp_spec(label, metadata)
    tests = _normalize_string_list(spec.get("tests") or spec.get("test_list"))
    test_imports = _normalize_string_list(spec.get("test_imports"))
    try:
        timeout = float(spec.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS

    code = extract_python_code(response)
    return 1.0 if run_mbpp_tests(code, tests, test_imports=test_imports, timeout=timeout) else 0.0
