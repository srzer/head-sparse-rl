"""Prepare the additional evaluation datasets in Miles JSONL format."""

import argparse
import json
from pathlib import Path

BENCHMARKS = ("math500", "ifbench", "mbppplus", "apps-intro")
MATH_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""


def math_records(problems):
    return [
        dict(
            prompt=[dict(role="user", content=MATH_PROMPT.format(problem=p["problem"]))],
            label=p["answer"],
            metadata={k: p[k] for k in ("unique_id", "subject", "level")},
        )
        for p in problems
    ]


def mbpp_records(problems):
    return [
        dict(
            prompt=[
                dict(
                    role="user",
                    content=(
                        "Write a Python solution for the following programming problem.\n\n"
                        f"{p['prompt'].strip()}\n\n"
                        f"Define the function exactly as `{p['entry_point']}`.\n\n"
                        "Return only Python code. Do not include explanations or markdown fences."
                    ),
                )
            ],
            label={},
            metadata=dict(benchmark="mbppplus", task_id=task_id),
        )
        for task_id, p in problems.items()
    ]


def apps_records(problems):
    records = []
    for p in problems:
        if p["difficulty"] != "introductory":
            continue
        spec = json.loads(p["input_output"]) if isinstance(p["input_output"], str) else p["input_output"]
        if not spec.get("inputs") or len(spec["inputs"]) != len(spec["outputs"]):
            raise ValueError(f"APPS problem {p['id']} has missing or mismatched tests")
        starter = p.get("starter_code") or ""
        prompt = (
            "Solve the following programming problem. Write a complete Python 3 program "
            "that reads from standard input and writes to standard output.\n\n"
            f"{p['question'].strip()}\n"
        )
        if starter.strip():
            prompt += f"\nStarter code, if useful:\n{starter.strip()}\n"
        prompt += "\nReturn only Python code. Do not include explanations or markdown fences."
        records.append(
            dict(
                prompt=[dict(role="user", content=prompt)],
                label=spec,
                metadata=dict(benchmark="apps_intro", task_id=p["id"]),
            )
        )
    return records


def prepare(benchmark, apps_source=None):
    if benchmark == "math500":
        from datasets import load_dataset

        return math_records(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if benchmark == "mbppplus":
        from evalplus.data import get_mbpp_plus

        return mbpp_records(get_mbpp_plus())
    if benchmark == "apps-intro":
        if apps_source:
            with apps_source.open() as handle:
                problems = [json.loads(line) for line in handle if line.strip()]
        else:
            from huggingface_hub import hf_hub_download

            source = hf_hub_download("codeparrot/apps", "test.jsonl", repo_type="dataset")
            with open(source) as handle:
                problems = [json.loads(line) for line in handle if line.strip()]
        return apps_records(problems)
    from huggingface_hub import hf_hub_download

    source = hf_hub_download("zyzshishui0627/IFBench", "IFBench_eval.jsonl", repo_type="dataset")
    with open(source) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not all(r.get("metadata", {}).get("instruction_id_list") for r in records):
        raise ValueError("IFBench records must include instruction metadata")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("all", *BENCHMARKS), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apps-source", type=Path, help="Existing APPS test.jsonl, instead of downloading it.")
    args = parser.parse_args()
    tasks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        output = args.output_dir / f"{task}.jsonl"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        records = prepare(task, args.apps_source)
        expected = {"math500": 500, "mbppplus": 378, "apps-intro": 1000}.get(task)
        if not records or (expected is not None and len(records) != expected):
            raise ValueError(f"{task}: expected {expected} records, got {len(records)}")
        with output.open("x") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{task}: wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
