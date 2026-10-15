#!/usr/bin/env python3
"""Build a deterministic per-layer random Q/K/V head-freeze manifest.

Unlike ``build_lora_head_freeze_manifest.py``, this utility does not require an
adapter checkpoint to enumerate the head universe. By default it samples the
requested fraction from the whole Q/K/V head-block pool of each layer, preserving
the same per-layer total budget as a top-k head experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-ratio", type=float, default=0.5)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-attention-heads", type=int, required=True)
    parser.add_argument("--num-query-groups", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--sampling-unit",
        choices=("layer", "layer-projection"),
        default="layer",
        help="Sample from all Q/K/V blocks in each layer (default), or per layer/projection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.freeze_ratio <= 1.0:
        raise ValueError("--freeze-ratio must be in [0, 1]")

    rng = random.Random(args.seed)
    frozen_heads: list[dict[str, int | str]] = []
    per_group_counts: dict[str, int] = {}

    for layer in range(args.num_layers):
        groups = (
            ("q", args.num_attention_heads),
            ("k", args.num_query_groups),
            ("v", args.num_query_groups),
        )
        if args.sampling_unit == "layer":
            candidates = [(projection, head) for projection, num_heads in groups for head in range(num_heads)]
            freeze_count = round(len(candidates) * args.freeze_ratio)
            selected = set(rng.sample(candidates, freeze_count))
            for projection, _ in groups:
                selected_heads = sorted(head for proj, head in selected if proj == projection)
                per_group_counts[f"{layer}:{projection}"] = len(selected_heads)
                frozen_heads.extend(
                    {"layer": layer, "projection": projection, "head": head}
                    for head in selected_heads
                )
        else:
            for projection, num_heads in groups:
                freeze_count = round(num_heads * args.freeze_ratio)
                selected = sorted(rng.sample(range(num_heads), freeze_count))
                per_group_counts[f"{layer}:{projection}"] = freeze_count
                frozen_heads.extend(
                    {"layer": layer, "projection": projection, "head": head}
                    for head in selected
                )

    total_heads = args.num_layers * (args.num_attention_heads + 2 * args.num_query_groups)
    metadata = {
        "strategy": f"uniform-random-per-{args.sampling_unit}",
        "sampling_unit": args.sampling_unit,
        "seed": args.seed,
        "freeze_ratio": args.freeze_ratio,
        "num_layers": args.num_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_query_groups": args.num_query_groups,
        "total_heads": total_heads,
        "frozen_heads": len(frozen_heads),
        "train_heads": total_heads - len(frozen_heads),
        "per_group_frozen_counts": per_group_counts,
    }
    manifest = {
        "version": 1,
        "head_dim": args.head_dim,
        "frozen_heads": frozen_heads,
        "metadata": metadata,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
