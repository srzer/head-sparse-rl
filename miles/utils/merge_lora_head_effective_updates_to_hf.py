"""Merge two Q/K/V LoRA adapters into a full HuggingFace model by head.

This is an offline experimental merge.  It materializes the exact LoRA effective
updates into the base model weights instead of writing a new LoRA adapter.

Default policy for each Q/K/V head:
  - if only adapter A is nonzero, use adapter A;
  - if only adapter B is nonzero, use adapter B;
  - if both are nonzero, average their effective updates;
  - if both are zero, add nothing.
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
import shutil
from dataclasses import dataclass
from pathlib import Path


QKV_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


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
        default="/root/Qwen3-4B-math-qkv-sparse/iter_0000249/adapter",
    )
    parser.add_argument(
        "--ifbench-adapter",
        default="/root/Qwen3-4B-ifbench-qkv-sparse/iter_0000499/adapter",
    )
    parser.add_argument(
        "--output",
        default="/root/Qwen3-4B-qkv-rank128-head-effective-merge-math249-ifbench499",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--zero-threshold", type=float, default=1e-6)
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


def load_adapter_state(adapter_path: str | Path) -> dict[str, "torch.Tensor"]:
    torch = require_torch()
    adapter_path = Path(adapter_path)
    checkpoint_path = adapter_path if adapter_path.is_file() else adapter_path / "adapter_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"HF LoRA adapter_model.bin not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a tensor state dict in {checkpoint_path}")
    return state


def load_adapter_config(adapter_path: str | Path) -> dict[str, object]:
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HF LoRA adapter_config.json not found: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def load_adapter(adapter_path: str | Path) -> Adapter:
    adapter_path = Path(adapter_path)
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
    for projection in QKV_PROJECTIONS:
        if f".self_attn.{projection}.weight" in name:
            return projection
    raise ValueError(f"Cannot infer Q/K/V projection from {name}")


def layer_from_weight_name(name: str) -> int:
    match = LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot infer layer index from {name}")
    return int(match.group(1))


def head_is_nonzero(b: "torch.Tensor", head: int, head_dim: int, threshold: float) -> bool:
    start = head * head_dim
    end = start + head_dim
    return bool(b[start:end].float().norm().item() > threshold)


def merge_qkv_weight_in_place(
    weight: "torch.Tensor",
    *,
    weight_name: str,
    math_pair: tuple[str, "torch.Tensor", "torch.Tensor"],
    ifbench_pair: tuple[str, "torch.Tensor", "torch.Tensor"],
    math_scale: float,
    ifbench_scale: float,
    head_dim: int,
    threshold: float,
) -> dict[str, int]:
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

    num_heads = weight.shape[0] // head_dim
    stats = {"math_only": 0, "ifbench_only": 0, "both": 0, "neither": 0}

    math_a_f = math_a.float()
    ifbench_a_f = ifbench_a.float()
    weight_dtype = weight.dtype
    weight_device = weight.device

    with torch.no_grad():
        for head in range(num_heads):
            start = head * head_dim
            end = start + head_dim
            math_active = head_is_nonzero(math_b, head, head_dim, threshold)
            ifbench_active = head_is_nonzero(ifbench_b, head, head_dim, threshold)

            if math_active and ifbench_active:
                math_coef = 0.5
                ifbench_coef = 0.5
                stats["both"] += 1
            elif math_active:
                math_coef = 1.0
                ifbench_coef = 0.0
                stats["math_only"] += 1
            elif ifbench_active:
                math_coef = 0.0
                ifbench_coef = 1.0
                stats["ifbench_only"] += 1
            else:
                stats["neither"] += 1
                continue

            update = torch.zeros((head_dim, weight.shape[1]), dtype=torch.float32)
            if math_coef:
                update.add_(math_b[start:end].float() @ math_a_f, alpha=math_coef * math_scale)
            if ifbench_coef:
                update.add_(ifbench_b[start:end].float() @ ifbench_a_f, alpha=ifbench_coef * ifbench_scale)
            weight[start:end].add_(update.to(device=weight_device, dtype=weight_dtype))

    layer = layer_from_weight_name(weight_name)
    projection = projection_from_weight_name(weight_name).replace("_proj", "")
    print(
        f"merged layer={layer:02d} proj={projection}: "
        f"math_only={stats['math_only']} ifbench_only={stats['ifbench_only']} "
        f"both={stats['both']} neither={stats['neither']}"
    )
    return stats


def merge_stats_ratios(merge_stats: dict[str, int]) -> dict[str, float]:
    total = sum(merge_stats[key] for key in ("math_only", "ifbench_only", "both", "neither"))
    if total == 0:
        return {key: 0.0 for key in ("math_only", "ifbench_only", "both", "neither")}
    return {key: merge_stats[key] / total for key in ("math_only", "ifbench_only", "both", "neither")}


def format_merge_stats(merge_stats: dict[str, int]) -> str:
    ratios = merge_stats_ratios(merge_stats)
    parts = []
    total = sum(merge_stats[key] for key in ("math_only", "ifbench_only", "both", "neither"))
    for key in ("math_only", "ifbench_only", "both", "neither"):
        parts.append(f"{key}={merge_stats[key]} ({ratios[key] * 100:.2f}%)")
    parts.append(f"total_heads={total}")
    parts.append(f"modules={merge_stats.get('modules', 0)}")
    return ", ".join(parts)


def copy_adapter_metadata(output: Path, math_adapter: Path, ifbench_adapter: Path, merge_stats: dict[str, int]) -> None:
    metadata = {
        "merge_type": "qkv_head_effective_update_to_full_weight",
        "math_adapter": str(math_adapter),
        "ifbench_adapter": str(ifbench_adapter),
        "rule": "math_only->math, ifbench_only->ifbench, both->average, neither->zero",
        "merge_stats": merge_stats,
        "merge_ratios": merge_stats_ratios(merge_stats),
    }
    with (output / "head_effective_merge_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_path = Path(args.base_model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

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

    total_stats = {"math_only": 0, "ifbench_only": 0, "both": 0, "neither": 0, "modules": 0}
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
            head_dim=args.head_dim,
            threshold=args.zero_threshold,
        )
        total_stats["modules"] += 1
        for key in ("math_only", "ifbench_only", "both", "neither"):
            total_stats[key] += stats[key]

    model.save_pretrained(output_path, safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=args.trust_remote_code)
        tokenizer.save_pretrained(output_path)
    except Exception as exc:  # pragma: no cover - tokenizer is a convenience.
        print(f"warning: failed to save tokenizer from {base_model_path}: {exc}")

    # Preserve adapter configs for auditing.
    for label, adapter_path in (("math", math_adapter.path), ("ifbench", ifbench_adapter.path)):
        config_path = adapter_path / "adapter_config.json"
        if config_path.is_file():
            shutil.copyfile(config_path, output_path / f"{label}_adapter_config.json")
    copy_adapter_metadata(output_path, math_adapter.path, ifbench_adapter.path, total_stats)
    print(f"saved merged full model to {output_path}")
    print(f"total merge stats: {format_merge_stats(total_stats)}")


if __name__ == "__main__":
    main()
