"""Build a Q/K/V head-freeze manifest from a ranked-head JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECTION_ORDER = {"q": 0, "k": 1, "v": 2}
HeadId = tuple[int, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", required=True, help="Ranked-head JSONL file.")
    parser.add_argument("--output", required=True, help="Output freeze-manifest JSON path.")
    parser.add_argument(
        "--freeze-ratio",
        type=float,
        default=0.5,
        help="Fraction of the completed order to freeze.",
    )
    parser.add_argument(
        "--freeze-side",
        choices=("prefix", "suffix"),
        default="prefix",
        help="Freeze the task-preferred prefix or the less-preferred suffix of the completed order.",
    )
    parser.add_argument(
        "--train-window-start",
        type=float,
        default=None,
        help=(
            "Start fraction (in [0,1]) of the train window over the completed order. "
            "When set together with --train-window-end, heads whose rank position falls "
            "in [start, end) train and everything else is frozen; this overrides "
            "--freeze-ratio / --freeze-side."
        ),
    )
    parser.add_argument(
        "--train-window-end",
        type=float,
        default=None,
        help="End fraction (in [0,1]) of the train window over the completed order.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--expected-frozen-heads", type=int)
    return parser.parse_args()


def head_sort_key(head: HeadId) -> tuple[int, int, int]:
    layer, projection, index = head
    return layer, PROJECTION_ORDER[projection], index


def all_qkv_heads(num_layers: int, num_attention_heads: int, num_query_groups: int) -> set[HeadId]:
    heads: set[HeadId] = set()
    for layer in range(num_layers):
        for head in range(num_attention_heads):
            heads.add((layer, "q", head))
        for projection in ("k", "v"):
            for head in range(num_query_groups):
                heads.add((layer, projection, head))
    return heads


def load_order(path: str | Path) -> list[HeadId]:
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


def complete_order(order: list[HeadId], universe: set[HeadId]) -> list[HeadId]:
    invalid = set(order) - universe
    if invalid:
        raise ValueError(f"Activation order contains heads outside configured universe: {sorted(invalid)[:10]}")
    ranked = set(order)
    return order + [head for head in sorted(universe - ranked, key=head_sort_key)]


def build_manifest(frozen_heads: set[HeadId], *, head_dim: int, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1,
        "head_dim": head_dim,
        "frozen_heads": [
            {"layer": layer, "projection": projection, "head": head}
            for layer, projection, head in sorted(frozen_heads, key=head_sort_key)
        ],
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.freeze_ratio <= 1.0:
        raise ValueError("--freeze-ratio must be in [0, 1]")

    universe = all_qkv_heads(args.num_layers, args.num_attention_heads, args.num_query_groups)
    order = complete_order(load_order(args.order), universe)

    use_window = args.train_window_start is not None or args.train_window_end is not None
    if use_window:
        if args.train_window_start is None or args.train_window_end is None:
            raise ValueError("--train-window-start and --train-window-end must be set together")
        if not 0.0 <= args.train_window_start <= args.train_window_end <= 1.0:
            raise ValueError("Require 0 <= --train-window-start <= --train-window-end <= 1")
        n = len(order)
        train_start = round(n * args.train_window_start)
        train_end = round(n * args.train_window_end)
        train_heads = set(order[train_start:train_end])
        frozen_heads = set(order) - train_heads
        strategy = "activation_order_window"
        train_note = (
            f"Train window over the completed order is rank [{train_start}, {train_end}) "
            f"(fractions [{args.train_window_start}, {args.train_window_end})); "
            "every head outside the window is frozen at its initial value."
        )
    else:
        freeze_count = round(len(order) * args.freeze_ratio)
        if args.freeze_side == "prefix":
            frozen_heads = set(order[:freeze_count])
            train_note = "The frozen prefix is the task-preferred first part of the order; training is left on the suffix."
        else:
            frozen_heads = set(order[len(order) - freeze_count :])
            train_note = "The frozen suffix is the less-preferred last part of the order; training is left on the prefix."
        strategy = f"activation_order_{args.freeze_side}"
    if args.expected_frozen_heads is not None and len(frozen_heads) != args.expected_frozen_heads:
        raise ValueError(f"Expected {args.expected_frozen_heads} frozen heads, got {len(frozen_heads)}")

    metadata = {
        "strategy": strategy,
        "order": str(args.order),
        "freeze_ratio": args.freeze_ratio,
        "freeze_side": args.freeze_side,
        "train_window_start": args.train_window_start,
        "train_window_end": args.train_window_end,
        "total_heads": len(order),
        "frozen_heads": len(frozen_heads),
        "train_heads": len(order) - len(frozen_heads),
        "note": train_note,
    }
    manifest = build_manifest(frozen_heads, head_dim=args.head_dim, metadata=metadata)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
