"""Build task-aware N-way round-robin Q/K/V LoRA head allocation manifests.

Generalizes ``build_lora_head_round_robin_allocation.py`` from two tasks to an
arbitrary number of tasks. Given one activation-order JSONL per task, this
assigns each task a disjoint slice of the Q/K/V head universe by cycling through
the tasks in round-robin order and letting each task claim its most-preferred
head that has not yet been taken. For each task it then writes a freeze manifest
listing the heads NOT assigned to that task (i.e. the heads it must keep frozen
during training).

Example (3-way, each task owns 1/3 of the heads):

    python3 build_lora_head_round_robin_allocation_nway.py \\
        --task dapo_math:math_order.jsonl:math_freeze.json \\
        --task mbpp:mbpp_order.jsonl:mbpp_freeze.json \\
        --task rlvr_ifeval:ifeval_order.jsonl:ifeval_freeze.json \\
        --active-ratio 0.3333333333
"""

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


def _round_robin_allocate(orders: list[list[HeadId]], *, per_task_budget: int) -> list[set[HeadId]]:
    """Cycle through tasks; each claims its most-preferred not-yet-used head."""
    num_tasks = len(orders)
    allocations: list[set[HeadId]] = [set() for _ in range(num_tasks)]
    cursors = [0] * num_tasks
    used: set[HeadId] = set()
    total_budget = per_task_budget * num_tasks

    turn = 0
    # Each iteration either assigns one head or finds the current task is full.
    # A guard bounds the loop in case some task can never reach its budget.
    stalled = 0
    while len(used) < total_budget and stalled < num_tasks:
        task = turn % num_tasks
        turn += 1
        if len(allocations[task]) >= per_task_budget:
            stalled += 1
            continue

        order = orders[task]
        cursor = cursors[task]
        while cursor < len(order) and order[cursor] in used:
            cursor += 1
        cursors[task] = cursor
        if cursor >= len(order):
            # This task has exhausted its (universe-covering) order without
            # reaching its budget; nothing more it can claim.
            stalled += 1
            continue

        head = order[cursor]
        allocations[task].add(head)
        used.add(head)
        cursors[task] = cursor + 1
        stalled = 0

    short = [i for i, a in enumerate(allocations) if len(a) != per_task_budget]
    if short:
        got = {i: len(allocations[i]) for i in short}
        raise ValueError(
            f"Unable to allocate requested per-task budget={per_task_budget}; "
            f"under-filled tasks (index->count)={got}"
        )
    return allocations


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        required=True,
        metavar="NAME:ORDER_PATH:OUTPUT_PATH",
        help="A task spec: task name, its activation-order JSONL, and its freeze-manifest output path, "
        "separated by ':'. Repeat for each task. Round-robin priority follows the order given.",
    )
    parser.add_argument("--active-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--share-remainder",
        action="store_true",
        default=False,
        help="Heads left unclaimed by the round robin (universe - num_tasks * active_ratio) are made "
        "trainable for EVERY task instead of frozen for everyone. With --active-ratio 0.25 and 3 tasks "
        "this gives each task 25%% dedicated + 25%% shared = 50%% trainable.",
    )
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    return parser.parse_args()


def _parse_task_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--task must be NAME:ORDER_PATH:OUTPUT_PATH, got {spec!r}")
    name, order_path, output_path = (p.strip() for p in parts)
    if not name or not order_path or not output_path:
        raise ValueError(f"--task fields must be non-empty, got {spec!r}")
    return name, order_path, output_path


def main() -> None:
    args = parse_args()
    if not 0.0 < args.active_ratio <= 1.0:
        raise ValueError("--active-ratio must be in (0, 1]")

    specs = [_parse_task_spec(s) for s in args.tasks]
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Task names must be unique, got {names}")

    universe = _all_qkv_heads(args.num_layers, args.num_attention_heads, args.num_query_groups)
    per_task_budget = round(len(universe) * args.active_ratio)
    if per_task_budget * len(specs) > len(universe):
        raise ValueError(
            "--active-ratio is per task and disjoint allocation requires "
            f"active_ratio <= 1/num_tasks; got {args.active_ratio} for {len(specs)} tasks "
            f"(per-task budget {per_task_budget} x {len(specs)} > {len(universe)})"
        )

    orders = [_complete_order(_load_order(order_path), universe) for _, order_path, _ in specs]
    allocations = _round_robin_allocate(orders, per_task_budget=per_task_budget)

    # Sanity: allocations must be pairwise disjoint.
    for i in range(len(allocations)):
        for j in range(i + 1, len(allocations)):
            shared = allocations[i] & allocations[j]
            if shared:
                raise AssertionError(
                    f"Round-robin allocation produced overlapping heads between "
                    f"{names[i]} and {names[j]}: {sorted(shared)[:10]}"
                )

    shared: set[HeadId] = set()
    if args.share_remainder:
        shared = universe - set().union(*allocations)

    base_metadata = {
        "strategy": "round-robin-task-aware-nway" + ("-shared-remainder" if args.share_remainder else ""),
        "num_tasks": len(specs),
        "tasks": names,
        "orders": {name: order_path for name, order_path, _ in specs},
        "per_task_active_ratio": args.active_ratio,
        "total_heads": len(universe),
        "per_task_active_budget": per_task_budget,
        "union_active_budget": len(set().union(*allocations)),
        "share_remainder": bool(args.share_remainder),
        "shared_budget": len(shared),
    }

    for (name, order_path, output_path), dedicated in zip(specs, allocations):
        train_heads = dedicated | shared
        frozen = universe - train_heads
        metadata = {
            **base_metadata,
            "task": name,
            "order": order_path,
            "dedicated_heads": len(dedicated),
            "shared_heads": len(shared),
            "train_heads": len(train_heads),
            "frozen_heads": len(frozen),
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(_build_manifest(frozen, head_dim=args.head_dim, metadata=metadata), f, indent=2)
            f.write("\n")

    print(json.dumps(base_metadata, indent=2))


if __name__ == "__main__":
    main()
