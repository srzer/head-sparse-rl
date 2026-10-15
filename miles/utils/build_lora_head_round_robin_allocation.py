"""Build task-aware round-robin Q/K/V LoRA head allocation manifests."""

import os
import sys

if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import json
from pathlib import Path


HeadId = tuple[int, str, int]
PROJECTION_ORDER = {"q": 0, "k": 1, "v": 2}


def _head_sort_key(head: HeadId) -> tuple[int, int, int]:
    layer, projection, index = head
    return layer, PROJECTION_ORDER[projection], index


def _load_order(path: str | Path) -> list[HeadId]:
    seen: set[HeadId] = set()
    order: list[HeadId] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            layer = event.get("layer")
            projection = event.get("projection")
            head = event.get("head")
            if not isinstance(layer, int) or projection not in PROJECTION_ORDER or not isinstance(head, int):
                raise ValueError(f"Invalid head event at {path}:{line_no}: {event!r}")
            head_id = (layer, projection, head)
            if head_id in seen:
                continue
            seen.add(head_id)
            order.append(head_id)
    if not order:
        raise ValueError(f"No activation events found in {path}")
    return order


def _all_qkv_heads(num_layers: int, num_attention_heads: int, num_query_groups: int) -> set[HeadId]:
    heads: set[HeadId] = set()
    for layer in range(num_layers):
        for head in range(num_attention_heads):
            heads.add((layer, "q", head))
        for projection in ("k", "v"):
            for head in range(num_query_groups):
                heads.add((layer, projection, head))
    return heads


def _complete_order(order: list[HeadId], universe: set[HeadId]) -> list[HeadId]:
    invalid = set(order) - universe
    if invalid:
        raise ValueError(f"Activation order contains heads outside the configured universe: {sorted(invalid)[:10]}")
    ranked = set(order)
    return order + [head for head in sorted(universe - ranked, key=_head_sort_key)]


def _round_robin_allocate(
    first_order: list[HeadId],
    second_order: list[HeadId],
    *,
    per_task_budget: int,
) -> tuple[set[HeadId], set[HeadId]]:
    allocations = [set(), set()]
    orders = [first_order, second_order]
    cursors = [0, 0]
    used: set[HeadId] = set()
    turn = 0
    total_budget = per_task_budget * 2

    while len(used) < total_budget:
        task = turn % 2
        if len(allocations[task]) >= per_task_budget:
            task = 1 - task
        if len(allocations[task]) >= per_task_budget:
            break

        order = orders[task]
        cursor = cursors[task]
        while cursor < len(order) and order[cursor] in used:
            cursor += 1
        cursors[task] = cursor
        if cursor >= len(order):
            other = 1 - task
            if len(allocations[other]) >= per_task_budget:
                break
            other_cursor = cursors[other]
            while other_cursor < len(orders[other]) and orders[other][other_cursor] in used:
                other_cursor += 1
            cursors[other] = other_cursor
            if other_cursor >= len(orders[other]):
                break
            task = other
            cursor = other_cursor
            order = orders[task]

        head = order[cursor]
        allocations[task].add(head)
        used.add(head)
        cursors[task] = cursor + 1
        turn += 1

    if len(allocations[0]) != per_task_budget or len(allocations[1]) != per_task_budget:
        raise ValueError(
            "Unable to allocate requested per-task budget="
            f"{per_task_budget}; allocated={len(allocations[0])}/{len(allocations[1])}"
        )
    return allocations[0], allocations[1]


def _build_manifest(frozen_heads: set[HeadId], *, head_dim: int, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1,
        "head_dim": head_dim,
        "frozen_heads": [
            {"layer": layer, "projection": projection, "head": head}
            for layer, projection, head in sorted(frozen_heads, key=_head_sort_key)
        ],
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifbench-order", required=True)
    parser.add_argument("--math-order", required=True)
    parser.add_argument("--ifbench-output", required=True)
    parser.add_argument("--math-output", required=True)
    parser.add_argument("--active-ratio", type=float, default=0.5)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.active_ratio <= 1.0:
        raise ValueError("--active-ratio must be in (0, 1]")

    universe = _all_qkv_heads(args.num_layers, args.num_attention_heads, args.num_query_groups)
    per_task_budget = round(len(universe) * args.active_ratio)
    if per_task_budget * 2 > len(universe):
        raise ValueError(
            "--active-ratio is per task and disjoint allocation requires active_ratio <= 0.5; "
            f"got {args.active_ratio}"
        )
    ifbench_order = _complete_order(_load_order(args.ifbench_order), universe)
    math_order = _complete_order(_load_order(args.math_order), universe)
    ifbench_heads, math_heads = _round_robin_allocate(
        ifbench_order,
        math_order,
        per_task_budget=per_task_budget,
    )

    shared = ifbench_heads & math_heads
    if shared:
        raise AssertionError(f"Round-robin allocation produced overlapping heads: {sorted(shared)[:10]}")

    metadata = {
        "strategy": "round-robin-task-aware",
        "ifbench_order": str(args.ifbench_order),
        "math_order": str(args.math_order),
        "per_task_active_ratio": args.active_ratio,
        "total_heads": len(universe),
        "per_task_active_budget": per_task_budget,
        "union_active_budget": len(ifbench_heads | math_heads),
        "ifbench_train_heads": len(ifbench_heads),
        "math_train_heads": len(math_heads),
        "ifbench_frozen_heads": len(universe - ifbench_heads),
        "math_frozen_heads": len(universe - math_heads),
    }

    outputs = [
        (Path(args.ifbench_output), universe - ifbench_heads, {**metadata, "task": "ifbench"}),
        (Path(args.math_output), universe - math_heads, {**metadata, "task": "dapo_math"}),
    ]
    for output_path, frozen_heads, output_metadata in outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(_build_manifest(frozen_heads, head_dim=args.head_dim, metadata=output_metadata), f, indent=2)
            f.write("\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
