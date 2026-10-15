"""Build task-aware N-way round-robin MLP-neuron allocation manifests.

Neuron-level twin of ``build_lora_head_round_robin_allocation_nway.py``. The
universe here is every FFN intermediate channel of the model -- one group per
``(layer, neuron)``, where a neuron binds ``gate_proj`` row, ``up_proj`` row and
``down_proj`` column (see :mod:`miles.utils.mlp_neuron_grad_importance`). Given
one importance-order JSONL per task, tasks take turns claiming their
most-preferred still-unclaimed neuron, so each ends up with a disjoint slice.
For each task it writes a manifest listing the neurons NOT assigned to it, i.e.
the ones it must hold frozen while training.

This is a SEPARATE allocation from the attention-head one: run both builders and
give the training job both manifests.

Example (3-way, each task owns 1/3 of the neurons):

    python3 build_mlp_neuron_round_robin_allocation_nway.py \\
        --task dapo_math:math_neurons.jsonl:math_neuron_freeze.json \\
        --task mbpp:mbpp_neurons.jsonl:mbpp_neuron_freeze.json \\
        --task rlvr_ifeval:ifeval_neurons.jsonl:ifeval_neuron_freeze.json \\
        --active-ratio 0.3333333333

The manifest stores frozen neurons grouped by layer (a flat list would be tens
of MB at 36 x 9728):

    {"version": 1, "ffn_hidden_size": 9728, "num_layers": 36,
     "frozen_neurons": {"0": [1, 4, 5, ...], "1": [...]}, "metadata": {...}}
"""

import os
import sys

if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import json
from pathlib import Path


NeuronId = tuple[int, int]


def _load_order(path: str | Path, *, num_layers: int, ffn_hidden_size: int) -> list[NeuronId]:
    """Read a neuron importance order (most important first), de-duplicated."""
    seen: set[NeuronId] = set()
    order: list[NeuronId] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            layer = event.get("layer")
            neuron = event.get("neuron")
            if not isinstance(layer, int) or not isinstance(neuron, int):
                raise ValueError(f"Invalid neuron event at {path}:{line_no}: {event!r}")
            if not (0 <= layer < num_layers) or not (0 <= neuron < ffn_hidden_size):
                raise ValueError(
                    f"Neuron event out of range at {path}:{line_no}: layer={layer} neuron={neuron} "
                    f"(num_layers={num_layers}, ffn_hidden_size={ffn_hidden_size})"
                )
            neuron_id = (layer, neuron)
            if neuron_id in seen:
                continue
            seen.add(neuron_id)
            order.append(neuron_id)
    if not order:
        raise ValueError(f"No neuron events found in {path}")
    return order


def _complete_order(order: list[NeuronId], *, num_layers: int, ffn_hidden_size: int) -> list[NeuronId]:
    """Append any neuron the order does not rank, so every task can reach its budget."""
    ranked = set(order)
    missing = [
        (layer, neuron)
        for layer in range(num_layers)
        for neuron in range(ffn_hidden_size)
        if (layer, neuron) not in ranked
    ]
    return order + missing


def _round_robin_allocate(orders: list[list[NeuronId]], *, per_task_budget: int) -> list[set[NeuronId]]:
    """Cycle through tasks; each claims its most-preferred not-yet-used neuron."""
    num_tasks = len(orders)
    allocations: list[set[NeuronId]] = [set() for _ in range(num_tasks)]
    cursors = [0] * num_tasks
    used: set[NeuronId] = set()
    total_budget = per_task_budget * num_tasks

    turn = 0
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
            stalled += 1
            continue

        neuron = order[cursor]
        allocations[task].add(neuron)
        used.add(neuron)
        cursors[task] = cursor + 1
        stalled = 0

    short = {i: len(a) for i, a in enumerate(allocations) if len(a) != per_task_budget}
    if short:
        raise ValueError(
            f"Unable to allocate requested per-task budget={per_task_budget}; "
            f"under-filled tasks (index->count)={short}"
        )
    return allocations


def _build_manifest(
    frozen: set[NeuronId],
    *,
    num_layers: int,
    ffn_hidden_size: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    by_layer: dict[str, list[int]] = {}
    for layer, neuron in sorted(frozen):
        by_layer.setdefault(str(layer), []).append(neuron)
    return {
        "version": 1,
        "num_layers": num_layers,
        "ffn_hidden_size": ffn_hidden_size,
        "frozen_neurons": by_layer,
        "metadata": metadata,
    }


def _parse_task_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--task must be NAME:ORDER_PATH:OUTPUT_PATH, got {spec!r}")
    name, order_path, output_path = (p.strip() for p in parts)
    if not name or not order_path or not output_path:
        raise ValueError(f"--task fields must be non-empty, got {spec!r}")
    return name, order_path, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        required=True,
        metavar="NAME:ORDER_PATH:OUTPUT_PATH",
        help="A task spec: task name, its neuron importance-order JSONL, and its freeze-manifest "
        "output path, separated by ':'. Repeat for each task. Round-robin priority follows the "
        "order given.",
    )
    parser.add_argument("--active-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--ffn-hidden-size", type=int, default=9728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.active_ratio <= 1.0:
        raise ValueError("--active-ratio must be in (0, 1]")

    specs = [_parse_task_spec(s) for s in args.tasks]
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Task names must be unique, got {names}")

    universe_size = args.num_layers * args.ffn_hidden_size
    per_task_budget = round(universe_size * args.active_ratio)
    if per_task_budget * len(specs) > universe_size:
        raise ValueError(
            "--active-ratio is per task and disjoint allocation requires "
            f"active_ratio <= 1/num_tasks; got {args.active_ratio} for {len(specs)} tasks "
            f"(per-task budget {per_task_budget} x {len(specs)} > {universe_size})"
        )

    orders = [
        _complete_order(
            _load_order(order_path, num_layers=args.num_layers, ffn_hidden_size=args.ffn_hidden_size),
            num_layers=args.num_layers,
            ffn_hidden_size=args.ffn_hidden_size,
        )
        for _, order_path, _ in specs
    ]
    allocations = _round_robin_allocate(orders, per_task_budget=per_task_budget)

    for i in range(len(allocations)):
        for j in range(i + 1, len(allocations)):
            shared = allocations[i] & allocations[j]
            if shared:
                raise AssertionError(
                    f"Round-robin allocation produced overlapping neurons between "
                    f"{names[i]} and {names[j]}: {sorted(shared)[:10]}"
                )

    base_metadata = {
        "strategy": "round-robin-task-aware-nway-mlp-neuron",
        "num_tasks": len(specs),
        "tasks": names,
        "orders": {name: order_path for name, order_path, _ in specs},
        "per_task_active_ratio": args.active_ratio,
        "total_neurons": universe_size,
        "per_task_active_budget": per_task_budget,
        "union_active_budget": len(set().union(*allocations)),
    }

    for (name, order_path, output_path), train_neurons in zip(specs, allocations):
        frozen = {
            (layer, neuron)
            for layer in range(args.num_layers)
            for neuron in range(args.ffn_hidden_size)
        } - train_neurons
        metadata = {
            **base_metadata,
            "task": name,
            "order": order_path,
            "train_neurons": len(train_neurons),
            "frozen_neurons": len(frozen),
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(
                _build_manifest(
                    frozen,
                    num_layers=args.num_layers,
                    ffn_hidden_size=args.ffn_hidden_size,
                    metadata=metadata,
                ),
                f,
            )
            f.write("\n")

    print(json.dumps(base_metadata, indent=2))


if __name__ == "__main__":
    main()
