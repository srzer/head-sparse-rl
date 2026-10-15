"""Merge two Q/K/V LoRA adapters using task-specific head activation orders.

For each Q/K/V head, this script checks whether the head is in each task's
top-ratio prefix of the activation order.  The default rule is:

  - both tasks put the head in the same partition: average the two updates;
  - only Math puts it in the top partition: use the Math update;
  - only IFBench puts it in the top partition: use the IFBench update.

For GQA models, pass ``--coupling gqa_group`` to make every query group use a
single merge decision shared by its Q heads and the corresponding K/V heads.
The merged LoRA effective updates are materialized into the full HF model.
"""

from __future__ import annotations

import os
import sys

# Running this file as ``python miles/utils/this_script.py`` puts
# ``miles/utils`` on sys.path, where ``types.py`` shadows stdlib ``types``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _THIS_DIR:
    sys.path.pop(0)

import argparse
import json
import re
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


QKV_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
PROJECTION_ORDER = {"q": 0, "k": 1, "v": 2}
PROJECTION_FROM_WEIGHT = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
HeadId = tuple[int, str, int]


@dataclass(frozen=True)
class Adapter:
    path: Path
    state: dict[str, "torch.Tensor"]
    scale: float


def require_torch():
    import torch

    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="/root/models/Qwen3-4B")
    parser.add_argument(
        "--math-adapter",
        default="/root/Qwen3-4B-math-qkv-rank128/iter_0000399",
        help="Math LoRA adapter directory, or checkpoint directory containing adapter/.",
    )
    parser.add_argument(
        "--ifbench-adapter",
        default="/root/Qwen3-4B-ifbench-qkv-rank128/iter_0000599",
        help="IFBench LoRA adapter directory, or checkpoint directory containing adapter/.",
    )
    parser.add_argument(
        "--math-order",
        default="/root/Qwen3-4B-head-order/dapo-math-head-activation-order.jsonl",
    )
    parser.add_argument(
        "--ifbench-order",
        default="/root/Qwen3-4B-head-order/ifbench-head-activation-order.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/root/Qwen3-4B-qkv-rank128-head-order-merge-ifbench599-math399",
    )
    parser.add_argument("--top-ratio", type=float, default=0.5)
    parser.add_argument(
        "--merge-rule",
        choices=["hard_partition", "smooth_rank"],
        default="hard_partition",
        help=(
            "hard_partition: same top/bottom partition averages, otherwise the earlier-order task dominates. "
            "smooth_rank: use a sigmoid of normalized rank differences to smoothly interpolate task weights."
        ),
    )
    parser.add_argument("--dominant-weight", type=float, default=1.0)
    parser.add_argument("--secondary-weight", type=float, default=0.0)
    parser.add_argument(
        "--coupling",
        choices=["independent", "gqa_group"],
        default="independent",
        help=(
            "independent: merge each Q/K/V head separately. "
            "gqa_group: use one task weight decision for each query group, shared by "
            "its Q heads and matching K/V heads."
        ),
    )
    parser.add_argument(
        "--smooth-temperature",
        type=float,
        default=0.15,
        help="Temperature for --merge-rule smooth_rank; smaller values make choices sharper.",
    )
    parser.add_argument(
        "--allow-identical-partitions",
        action="store_true",
        help="Allow running even when Math and IFBench top partitions are identical.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="dtype used to load/save the HF model.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for loading and merging the HF model. Use cpu for safer offline merge.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def dtype_from_arg(name: str):
    if name == "auto":
        return "auto"
    torch = require_torch()
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def normalize_adapter_path(adapter_path: str | Path) -> Path:
    adapter_path = Path(adapter_path)
    if (adapter_path / "adapter_model.bin").is_file():
        return adapter_path
    nested = adapter_path / "adapter"
    if (nested / "adapter_model.bin").is_file():
        return nested
    return adapter_path


def load_adapter_state(adapter_path: str | Path) -> dict[str, "torch.Tensor"]:
    torch = require_torch()
    adapter_path = normalize_adapter_path(adapter_path)
    checkpoint_path = adapter_path / "adapter_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"HF LoRA adapter_model.bin not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a tensor state dict in {checkpoint_path}")
    return state


def load_adapter_config(adapter_path: str | Path) -> dict[str, object]:
    adapter_path = normalize_adapter_path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HF LoRA adapter_config.json not found: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def load_adapter(adapter_path: str | Path) -> Adapter:
    adapter_path = normalize_adapter_path(adapter_path)
    config = load_adapter_config(adapter_path)
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    return Adapter(
        path=adapter_path,
        state=load_adapter_state(adapter_path),
        scale=alpha / rank,
    )


def hf_weight_name_from_lora_a(lora_a_name: str) -> str:
    name = lora_a_name
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    if ".lora_A." not in name:
        raise ValueError(f"Not a LoRA A tensor name: {lora_a_name}")
    name = name.split(".lora_A.", 1)[0] + ".weight"
    return name


def find_lora_pair(
    state: dict[str, "torch.Tensor"],
    weight_name: str,
) -> tuple[str, "torch.Tensor", "torch.Tensor"] | None:
    candidates: list[tuple[str, "torch.Tensor", "torch.Tensor"]] = []
    for name, tensor in state.items():
        if ".lora_A." not in name:
            continue
        if hf_weight_name_from_lora_a(name) != weight_name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        if b_name not in state:
            raise ValueError(f"Missing LoRA B tensor for {name}: expected {b_name}")
        candidates.append((name, tensor, state[b_name]))
    if len(candidates) > 1:
        raise ValueError(f"Found multiple LoRA pairs for {weight_name}: {[c[0] for c in candidates]}")
    return candidates[0] if candidates else None


def is_qkv_weight(name: str) -> bool:
    return any(f".self_attn.{projection}.weight" in name for projection in QKV_PROJECTIONS)


def projection_from_weight_name(name: str) -> str:
    for projection, short_name in PROJECTION_FROM_WEIGHT.items():
        if f".self_attn.{projection}.weight" in name:
            return short_name
    raise ValueError(f"Cannot infer Q/K/V projection from {name}")


def layer_from_weight_name(name: str) -> int:
    match = LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot infer layer index from {name}")
    return int(match.group(1))


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
        raise ValueError(f"Activation order contains heads outside the configured universe: {sorted(invalid)[:10]}")
    ranked = set(order)
    return order + [head for head in sorted(universe - ranked, key=head_sort_key)]


def build_top_heads(order_path: str | Path, universe: set[HeadId], top_ratio: float) -> set[HeadId]:
    order = complete_order(load_order(order_path), universe)
    top_budget = round(len(universe) * top_ratio)
    return set(order[:top_budget])


def build_rank_map(order_path: str | Path, universe: set[HeadId]) -> dict[HeadId, float]:
    order = complete_order(load_order(order_path), universe)
    denom = max(len(order) - 1, 1)
    return {head: index / denom for index, head in enumerate(order)}


def smooth_task_weights(math_rank: float, ifbench_rank: float, temperature: float) -> tuple[float, float]:
    if temperature <= 0.0:
        raise ValueError("--smooth-temperature must be > 0")
    # Lower activation-order rank means the task prefers that head earlier.
    logit = (ifbench_rank - math_rank) / temperature
    math_weight = 1.0 / (1.0 + math.exp(-logit))
    return math_weight, 1.0 - math_weight


def gqa_group_heads(layer: int, group: int, q_per_group: int) -> list[HeadId]:
    return (
        [(layer, "q", group * q_per_group + offset) for offset in range(q_per_group)]
        + [(layer, "k", group), (layer, "v", group)]
    )


def head_merge_unit(
    *,
    head_id: HeadId,
    coupling: str,
    num_attention_heads: int,
    num_query_groups: int,
) -> tuple[HeadId, ...]:
    if coupling == "independent":
        return (head_id,)
    if num_attention_heads % num_query_groups != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} must be divisible by "
            f"num_query_groups={num_query_groups} for --coupling gqa_group"
        )
    layer, projection, head = head_id
    q_per_group = num_attention_heads // num_query_groups
    if projection == "q":
        if head >= num_attention_heads:
            raise ValueError(f"Q head index out of range for GQA coupling: {head_id}")
        group = head // q_per_group
    else:
        if head >= num_query_groups:
            raise ValueError(f"{projection.upper()} head index out of range for GQA coupling: {head_id}")
        group = head
    return tuple(gqa_group_heads(layer, group, q_per_group))


def merge_unit_task_weights(
    *,
    unit_heads: tuple[HeadId, ...],
    math_top_heads: set[HeadId],
    ifbench_top_heads: set[HeadId],
    math_rank_map: dict[HeadId, float],
    ifbench_rank_map: dict[HeadId, float],
    merge_rule: str,
    top_ratio: float,
    dominant_weight: float,
    secondary_weight: float,
    smooth_temperature: float,
) -> tuple[float, float, str]:
    math_rank = sum(math_rank_map[head] for head in unit_heads) / len(unit_heads)
    ifbench_rank = sum(ifbench_rank_map[head] for head in unit_heads) / len(unit_heads)

    if len(unit_heads) == 1:
        math_is_top = unit_heads[0] in math_top_heads
        ifbench_is_top = unit_heads[0] in ifbench_top_heads
    else:
        math_is_top = math_rank < top_ratio
        ifbench_is_top = ifbench_rank < top_ratio

    if merge_rule == "smooth_rank":
        math_weight, ifbench_weight = smooth_task_weights(math_rank, ifbench_rank, smooth_temperature)
        stat_key = "math_dominates" if math_weight >= ifbench_weight else "ifbench_dominates"
    elif math_is_top == ifbench_is_top:
        math_weight = 0.5
        ifbench_weight = 0.5
        stat_key = "same_top" if math_is_top else "same_bottom"
    elif math_is_top:
        math_weight = dominant_weight
        ifbench_weight = secondary_weight
        stat_key = "math_dominates"
    else:
        math_weight = secondary_weight
        ifbench_weight = dominant_weight
        stat_key = "ifbench_dominates"
    return math_weight, ifbench_weight, stat_key


def merge_qkv_weight_in_place(
    weight: "torch.Tensor",
    *,
    weight_name: str,
    math_pair: tuple[str, "torch.Tensor", "torch.Tensor"],
    ifbench_pair: tuple[str, "torch.Tensor", "torch.Tensor"],
    math_scale: float,
    ifbench_scale: float,
    math_top_heads: set[HeadId],
    ifbench_top_heads: set[HeadId],
    math_rank_map: dict[HeadId, float],
    ifbench_rank_map: dict[HeadId, float],
    merge_rule: str,
    coupling: str,
    top_ratio: float,
    dominant_weight: float,
    secondary_weight: float,
    smooth_temperature: float,
    head_dim: int,
    num_attention_heads: int,
    num_query_groups: int,
) -> dict[str, int | float]:
    torch = require_torch()
    _math_name, math_a, math_b = math_pair
    _ifbench_name, ifbench_a, ifbench_b = ifbench_pair
    if math_a.shape != ifbench_a.shape:
        raise ValueError(f"LoRA A shape mismatch for {weight_name}: {math_a.shape} vs {ifbench_a.shape}")
    if math_b.shape != ifbench_b.shape:
        raise ValueError(f"LoRA B shape mismatch for {weight_name}: {math_b.shape} vs {ifbench_b.shape}")
    if weight.shape != (math_b.shape[0], math_a.shape[1]):
        raise ValueError(
            f"Weight/LoRA shape mismatch for {weight_name}: "
            f"weight={tuple(weight.shape)}, B={tuple(math_b.shape)}, A={tuple(math_a.shape)}"
        )
    if weight.shape[0] % head_dim != 0:
        raise ValueError(f"{weight_name} rows={weight.shape[0]} not divisible by head_dim={head_dim}")

    layer = layer_from_weight_name(weight_name)
    projection = projection_from_weight_name(weight_name)
    num_heads = weight.shape[0] // head_dim
    stats: dict[str, int | float] = {
        "same_top": 0,
        "same_bottom": 0,
        "math_dominates": 0,
        "ifbench_dominates": 0,
        "smooth_math_weight_sum": 0.0,
        "smooth_ifbench_weight_sum": 0.0,
        "update_norm": 0.0,
    }

    math_a_f = math_a.float()
    ifbench_a_f = ifbench_a.float()
    weight_dtype = weight.dtype
    weight_device = weight.device

    with torch.no_grad():
        for head in range(num_heads):
            head_id = (layer, projection, head)
            start = head * head_dim
            end = start + head_dim
            unit_heads = head_merge_unit(
                head_id=head_id,
                coupling=coupling,
                num_attention_heads=num_attention_heads,
                num_query_groups=num_query_groups,
            )
            math_weight, ifbench_weight, stat_key = merge_unit_task_weights(
                unit_heads=unit_heads,
                math_top_heads=math_top_heads,
                ifbench_top_heads=ifbench_top_heads,
                math_rank_map=math_rank_map,
                ifbench_rank_map=ifbench_rank_map,
                merge_rule=merge_rule,
                top_ratio=top_ratio,
                dominant_weight=dominant_weight,
                secondary_weight=secondary_weight,
                smooth_temperature=smooth_temperature,
            )
            stats[stat_key] += 1

            stats["smooth_math_weight_sum"] += float(math_weight)
            stats["smooth_ifbench_weight_sum"] += float(ifbench_weight)
            update = torch.zeros((head_dim, weight.shape[1]), dtype=torch.float32)
            update.add_(math_b[start:end].float() @ math_a_f, alpha=math_weight * math_scale)
            update.add_(ifbench_b[start:end].float() @ ifbench_a_f, alpha=ifbench_weight * ifbench_scale)
            stats["update_norm"] += float(update.norm().item())
            weight[start:end].add_(update.to(device=weight_device, dtype=weight_dtype))

    print(
        f"merged layer={layer:02d} proj={projection} coupling={coupling}: "
        f"same_top={stats['same_top']} same_bottom={stats['same_bottom']} "
        f"math_dominates={stats['math_dominates']} ifbench_dominates={stats['ifbench_dominates']}"
    )
    return stats


def format_stats(stats: dict[str, int | float]) -> str:
    total = sum(int(stats[key]) for key in ("same_top", "same_bottom", "math_dominates", "ifbench_dominates"))
    parts = []
    for key in ("same_top", "same_bottom", "math_dominates", "ifbench_dominates"):
        value = int(stats[key])
        ratio = 0.0 if total == 0 else value / total * 100.0
        parts.append(f"{key}={value} ({ratio:.2f}%)")
    parts.append(f"total_heads={total}")
    parts.append(f"modules={int(stats['modules'])}")
    parts.append(f"total_update_norm={float(stats['total_update_norm']):.6g}")
    if float(stats.get("smooth_math_weight_sum", 0.0)) or float(stats.get("smooth_ifbench_weight_sum", 0.0)):
        parts.append(f"math_weight_sum={float(stats['smooth_math_weight_sum']):.6g}")
        parts.append(f"ifbench_weight_sum={float(stats['smooth_ifbench_weight_sum']):.6g}")
    return ", ".join(parts)


def copy_metadata(
    output: Path,
    *,
    args: argparse.Namespace,
    math_adapter: Adapter,
    ifbench_adapter: Adapter,
    stats: dict[str, int | float],
    math_top_heads: set[HeadId],
    ifbench_top_heads: set[HeadId],
    universe: set[HeadId],
) -> None:
    shared_top = math_top_heads & ifbench_top_heads
    metadata = {
        "merge_type": "qkv_head_order_effective_update_to_full_weight",
        "math_adapter": str(math_adapter.path),
        "ifbench_adapter": str(ifbench_adapter.path),
        "math_order": str(args.math_order),
        "ifbench_order": str(args.ifbench_order),
        "rule": (
            "hard_partition: same partition -> average; only Math top -> Math dominates; "
            "only IFBench top -> IFBench dominates. smooth_rank: sigmoid-normalized rank interpolation."
        ),
        "merge_rule": args.merge_rule,
        "coupling": args.coupling,
        "top_ratio": args.top_ratio,
        "dominant_weight": args.dominant_weight,
        "secondary_weight": args.secondary_weight,
        "smooth_temperature": args.smooth_temperature,
        "math_scale": math_adapter.scale,
        "ifbench_scale": ifbench_adapter.scale,
        "total_heads": len(universe),
        "math_top_heads": len(math_top_heads),
        "ifbench_top_heads": len(ifbench_top_heads),
        "shared_top_heads": len(shared_top),
        "math_only_top_heads": len(math_top_heads - ifbench_top_heads),
        "ifbench_only_top_heads": len(ifbench_top_heads - math_top_heads),
        "neither_top_heads": len(universe - (math_top_heads | ifbench_top_heads)),
        "merge_stats": stats,
    }
    with (output / "head_order_effective_merge_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_ratio < 1.0:
        raise ValueError("--top-ratio must be in (0, 1)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_path = Path(args.base_model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    universe = all_qkv_heads(args.num_layers, args.num_attention_heads, args.num_query_groups)
    math_top_heads = build_top_heads(args.math_order, universe, args.top_ratio)
    ifbench_top_heads = build_top_heads(args.ifbench_order, universe, args.top_ratio)
    math_rank_map = build_rank_map(args.math_order, universe)
    ifbench_rank_map = build_rank_map(args.ifbench_order, universe)
    if args.merge_rule == "hard_partition" and math_top_heads == ifbench_top_heads and not args.allow_identical_partitions:
        raise ValueError(
            "Math and IFBench top head partitions are identical. This would reduce the merge to "
            "plain 0.5/0.5 averaging for every head, with no task-aware dominance. Check whether "
            "the activation-order files contain real task-specific order information, or pass "
            "--allow-identical-partitions to run this baseline intentionally."
        )
    math_adapter = load_adapter(args.math_adapter)
    ifbench_adapter = load_adapter(args.ifbench_adapter)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype_from_arg(args.torch_dtype),
        device_map=None,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()

    total_stats: dict[str, int | float] = {
        "same_top": 0,
        "same_bottom": 0,
        "math_dominates": 0,
        "ifbench_dominates": 0,
        "modules": 0,
        "total_update_norm": 0.0,
        "smooth_math_weight_sum": 0.0,
        "smooth_ifbench_weight_sum": 0.0,
    }
    state = dict(model.named_parameters())
    qkv_weights = sorted(name for name in state if is_qkv_weight(name))
    if not qkv_weights:
        raise ValueError("No HF Q/K/V projection weights found in base model")

    for weight_name in qkv_weights:
        math_pair = find_lora_pair(math_adapter.state, weight_name)
        ifbench_pair = find_lora_pair(ifbench_adapter.state, weight_name)
        if math_pair is None and ifbench_pair is None:
            continue
        if math_pair is None or ifbench_pair is None:
            raise ValueError(
                f"Adapter coverage mismatch for {weight_name}: "
                f"math_pair={math_pair is not None}, ifbench_pair={ifbench_pair is not None}"
            )

        stats = merge_qkv_weight_in_place(
            state[weight_name].data,
            weight_name=weight_name,
            math_pair=math_pair,
            ifbench_pair=ifbench_pair,
            math_scale=math_adapter.scale,
            ifbench_scale=ifbench_adapter.scale,
            math_top_heads=math_top_heads,
            ifbench_top_heads=ifbench_top_heads,
            math_rank_map=math_rank_map,
            ifbench_rank_map=ifbench_rank_map,
            merge_rule=args.merge_rule,
            coupling=args.coupling,
            top_ratio=args.top_ratio,
            dominant_weight=args.dominant_weight,
            secondary_weight=args.secondary_weight,
            smooth_temperature=args.smooth_temperature,
            head_dim=args.head_dim,
            num_attention_heads=args.num_attention_heads,
            num_query_groups=args.num_query_groups,
        )
        total_stats["modules"] += 1
        total_stats["total_update_norm"] += float(stats["update_norm"])
        for key in ("same_top", "same_bottom", "math_dominates", "ifbench_dominates"):
            total_stats[key] += int(stats[key])
        for key in ("smooth_math_weight_sum", "smooth_ifbench_weight_sum"):
            total_stats[key] += float(stats[key])

    model.save_pretrained(output_path, safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=args.trust_remote_code)
        tokenizer.save_pretrained(output_path)
    except Exception as exc:  # pragma: no cover - tokenizer is a convenience.
        print(f"warning: failed to save tokenizer from {base_model_path}: {exc}")

    for label, adapter_path in (("math", math_adapter.path), ("ifbench", ifbench_adapter.path)):
        config_path = adapter_path / "adapter_config.json"
        if config_path.is_file():
            shutil.copyfile(config_path, output_path / f"{label}_adapter_config.json")
    copy_metadata(
        output_path,
        args=args,
        math_adapter=math_adapter,
        ifbench_adapter=ifbench_adapter,
        stats=total_stats,
        math_top_heads=math_top_heads,
        ifbench_top_heads=ifbench_top_heads,
        universe=universe,
    )
    print(f"saved head-order merged full model to {output_path}")
    print(f"total merge stats: {format_stats(total_stats)}")


if __name__ == "__main__":
    main()
