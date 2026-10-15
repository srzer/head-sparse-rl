"""Build semantic Q/K/V LoRA head-freeze manifests from adapter checkpoints."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch


HeadId = tuple[int, str, int]
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_PROJECTION_MARKERS = {"q": ".q_proj.", "k": ".k_proj.", "v": ".v_proj."}


def load_adapter_state(adapter_path: str | Path) -> dict[str, torch.Tensor]:
    adapter_path = Path(adapter_path)
    checkpoint_path = adapter_path if adapter_path.is_file() else adapter_path / "adapter_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"HF LoRA adapter_model.bin not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a tensor state dict in {checkpoint_path}")
    return state


def extract_qkv_head_activity(
    state: dict[str, torch.Tensor],
    *,
    head_dim: int,
    threshold: float,
) -> tuple[set[HeadId], set[HeadId]]:
    """Return all and active semantic Q/K/V LoRA B head IDs."""
    all_heads: set[HeadId] = set()
    active_heads: set[HeadId] = set()

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 or "lora_B" not in name:
            continue
        projection = next(
            (projection for projection, marker in _PROJECTION_MARKERS.items() if marker in name),
            None,
        )
        if projection is None:
            continue
        layer_match = _LAYER_RE.search(name)
        if layer_match is None:
            raise ValueError(f"Cannot extract layer index from HF LoRA B tensor name: {name}")
        if tensor.shape[0] % head_dim != 0:
            raise ValueError(
                f"LoRA B rows for {name} are not divisible by head_dim={head_dim}: shape={tuple(tensor.shape)}"
            )

        layer = int(layer_match.group(1))
        norms = tensor.view(-1, head_dim, tensor.shape[1]).float().pow(2).sum(dim=(1, 2)).sqrt()
        for head, norm in enumerate(norms.tolist()):
            head_id = (layer, projection, head)
            if head_id in all_heads:
                raise ValueError(f"Duplicate semantic LoRA head tensor for {head_id}: {name}")
            all_heads.add(head_id)
            if norm > threshold:
                active_heads.add(head_id)

    if not all_heads:
        raise ValueError("No Q/K/V lora_B tensors were found in the adapter state")
    return all_heads, active_heads


def select_frozen_heads(
    primary_all: set[HeadId],
    primary_active: set[HeadId],
    *,
    strategy: str,
    secondary_all: set[HeadId] | None = None,
    secondary_active: set[HeadId] | None = None,
    random_freeze_ratio: float = 0.5,
    seed: int = 1234,
) -> set[HeadId]:
    """Select frozen heads using adapter-derived or random policies."""
    if secondary_all is not None and primary_all != secondary_all:
        only_primary = sorted(primary_all - secondary_all)[:10]
        only_secondary = sorted(secondary_all - primary_all)[:10]
        raise ValueError(
            "Adapter head universes do not match; "
            f"only_primary={only_primary}, only_secondary={only_secondary}"
        )

    if strategy == "primary-active":
        return set(primary_active)
    if strategy == "primary-only":
        if secondary_active is None:
            raise ValueError("primary-only requires --secondary-adapter")
        return primary_active - secondary_active
    if strategy == "secondary-inactive":
        if secondary_active is None:
            raise ValueError("secondary-inactive requires --secondary-adapter")
        return primary_all - secondary_active
    if strategy == "random":
        if not 0.0 <= random_freeze_ratio <= 1.0:
            raise ValueError("--random-freeze-ratio must be in [0, 1]")
        grouped: dict[tuple[int, str], list[HeadId]] = defaultdict(list)
        for head_id in primary_all:
            grouped[head_id[:2]].append(head_id)
        rng = random.Random(seed)
        frozen: set[HeadId] = set()
        for group in sorted(grouped):
            candidates = sorted(grouped[group])
            frozen.update(rng.sample(candidates, round(len(candidates) * random_freeze_ratio)))
        return frozen
    raise ValueError(f"Unsupported freeze strategy: {strategy}")


def build_manifest(
    frozen_heads: set[HeadId],
    *,
    head_dim: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    projection_order = {"q": 0, "k": 1, "v": 2}
    entries = [
        {"layer": layer, "projection": projection, "head": head}
        for layer, projection, head in sorted(
            frozen_heads,
            key=lambda item: (item[0], projection_order[item[1]], item[2]),
        )
    ]
    return {
        "version": 1,
        "head_dim": head_dim,
        "frozen_heads": entries,
        "metadata": metadata or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-adapter", required=True)
    parser.add_argument("--secondary-adapter")
    parser.add_argument(
        "--strategy",
        choices=["primary-active", "primary-only", "secondary-inactive", "random"],
        default="primary-only",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--random-freeze-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--expected-frozen-heads", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_all, primary_active = extract_qkv_head_activity(
        load_adapter_state(args.primary_adapter),
        head_dim=args.head_dim,
        threshold=args.threshold,
    )
    secondary_all = None
    secondary_active = None
    if args.secondary_adapter is not None:
        secondary_all, secondary_active = extract_qkv_head_activity(
            load_adapter_state(args.secondary_adapter),
            head_dim=args.head_dim,
            threshold=args.threshold,
        )

    frozen_heads = select_frozen_heads(
        primary_all,
        primary_active,
        strategy=args.strategy,
        secondary_all=secondary_all,
        secondary_active=secondary_active,
        random_freeze_ratio=args.random_freeze_ratio,
        seed=args.seed,
    )
    if args.expected_frozen_heads is not None and len(frozen_heads) != args.expected_frozen_heads:
        raise ValueError(
            f"Freeze selection count mismatch: expected {args.expected_frozen_heads}, got {len(frozen_heads)}"
        )
    metadata = {
        "strategy": args.strategy,
        "primary_adapter": str(args.primary_adapter),
        "secondary_adapter": str(args.secondary_adapter) if args.secondary_adapter else None,
        "threshold": args.threshold,
        "total_heads": len(primary_all),
        "primary_active_heads": len(primary_active),
        "secondary_active_heads": len(secondary_active) if secondary_active is not None else None,
        "frozen_heads": len(frozen_heads),
        "train_heads": len(primary_all - frozen_heads),
        "seed": args.seed if args.strategy == "random" else None,
    }
    manifest = build_manifest(frozen_heads, head_dim=args.head_dim, metadata=metadata)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
