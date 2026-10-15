"""Rank GQA groups by the norm of their globally reduced QKV gradients.

Distributed optimizer shards are disjoint pieces of the reduced gradient.
Only optimizer-owned slices are valid; their squared norms are summed across
ranks before taking the square root.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass
class QKVGradientShard:
    """Only the reduced parameter slice owned by this optimizer rank."""

    values: torch.Tensor
    shape: tuple[int, int]
    start: int


def snapshot_qkv_head_gradients(model: Sequence[torch.nn.Module], optimizer) -> dict[str, QKVGradientShard]:
    """Capture valid reduced gradient shards before optimizer.step clears them.

    DistributedOptimizer's parameter ranges describe the reduce-scatter output
    owned by this rank, including boundaries that cut through a head. Reading
    any other part of main_grad after reduce-scatter is invalid.
    """
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    sharded = [opt for opt in optimizers if hasattr(opt, "model_param_gbuf_map")]
    snapshot: dict[str, QKVGradientShard] = {}
    with torch.no_grad():
        for chunk in model:
            for name, param in chunk.named_parameters():
                if not name.endswith("self_attention.linear_qkv.weight") or param.ndim != 2:
                    continue
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                if grad is None:
                    continue

                start, end = 0, param.numel()
                if sharded:
                    owners = [opt for opt in sharded if param in opt.model_param_gbuf_map]
                    if len(owners) > 1:
                        raise ValueError(f"Multiple optimizers own the same QKV parameter: {name}")
                    if owners:
                        owned = owners[0]._get_model_param_range_map(param)["param"]
                        start, end = owned.start, owned.end
                    else:
                        start = end = 0
                if not 0 <= start <= end <= param.numel():
                    raise ValueError(f"Invalid optimizer shard range for {name}: [{start}, {end})")
                snapshot[name] = QKVGradientShard(
                    values=grad.detach().reshape(-1)[start:end].float().clone(),
                    shape=tuple(param.shape),
                    start=start,
                )
    return snapshot


def merge_qkv_head_grad_parts(
    gathered: Sequence[Sequence[dict[str, float | int]]],
) -> list[dict[str, float | int]]:
    """Sum squared norms of disjoint reduced slices, never norms of DP replicas."""
    merged: dict[tuple[int, int], dict[str, float | int]] = {}
    for parts in gathered:
        for part in parts:
            key = (int(part["layer"]), int(part["group"]))
            if key not in merged:
                merged[key] = dict(part)
            else:
                total = merged[key]
                if total["expected_elements"] != part["expected_elements"]:
                    raise ValueError(f"Inconsistent head dimensions for {key}")
                for field in ("q_sq", "k_sq", "v_sq", "elements"):
                    total[field] += part[field]

    events = []
    for key, part in sorted(merged.items()):
        if part["elements"] != part["expected_elements"]:
            raise ValueError(
                f"Head {key} has {part['elements']} gradient elements; expected {part['expected_elements']}. "
                "Capture optimizer-owned shards; missing shards and replicated full gradients are not valid."
            )
        q_sq, k_sq, v_sq = (float(part[field]) for field in ("q_sq", "k_sq", "v_sq"))
        if not all(math.isfinite(value) and value >= 0 for value in (q_sq, k_sq, v_sq)):
            raise ValueError(f"Non-finite gradient importance for head {key}")
        events.append(
            {
                "layer": key[0],
                "group": key[1],
                "tp_rank": int(part["tp_rank"]),
                "q_norm": math.sqrt(q_sq),
                "k_norm": math.sqrt(k_sq),
                "v_norm": math.sqrt(v_sq),
                "grad_norm": math.sqrt(q_sq + k_sq + v_sq),
            }
        )
    return events


def collect_qkv_head_grad_norms(
    model: Sequence[torch.nn.Module],
    head_dim: int,
    num_attention_heads: int,
    num_query_groups: int,
    *,
    tp_rank: int,
    tp_size: int,
    layer_offset: int = 0,
    grad_snapshot: dict[str, torch.Tensor | QKVGradientShard] | None = None,
) -> list[dict[str, float | int]]:
    """Return one event per global GQA group with the L2 norm of its accumulated
    ``linear_qkv`` gradient (and the q/k/v sub-norms), merged across TP ranks.

    ``grad_snapshot`` must contain optimizer-owned reduced slices when using a
    distributed optimizer. Full tensors are supported only when they collectively
    cover each group exactly once (e.g. DP=1).
    """
    if head_dim <= 0:
        return []
    if num_attention_heads % num_query_groups != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"num_query_groups ({num_query_groups})"
        )
    q_per_group = num_attention_heads // num_query_groups
    blocks_per_group = q_per_group + 2  # 4 Q + 1 K + 1 V for Qwen3-4B
    if num_query_groups % tp_size != 0:
        raise ValueError(f"num_query_groups ({num_query_groups}) must be divisible by tp_size ({tp_size})")
    kv_per_rank = num_query_groups // tp_size

    local_events: list[dict[str, float | int]] = []
    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if not name.endswith("self_attention.linear_qkv.weight") or param.ndim != 2:
                    continue

                if grad_snapshot is not None:
                    grad = grad_snapshot.get(name)
                else:
                    grad = getattr(param, "main_grad", None)
                    if grad is None:
                        grad = param.grad
                if grad is None:
                    continue

                layer_match = _LAYER_INDEX_RE.search(name)
                if layer_match is None:
                    raise ValueError(f"Cannot extract layer index from QKV parameter name: {name}")
                layer = layer_offset + int(layer_match.group(1))

                if isinstance(grad, QKVGradientShard):
                    shape, shard_start = grad.shape, grad.start
                    values = grad.values
                else:
                    shape, shard_start = tuple(grad.shape), 0
                    values = grad.detach().reshape(-1).float()
                out_rows = shape[0]
                expected = kv_per_rank * blocks_per_group * head_dim
                if out_rows != expected:
                    raise ValueError(
                        f"{name}: fused-qkv grad rows {out_rows} != expected "
                        f"{expected} (kv_per_rank={kv_per_rank}, blocks_per_group={blocks_per_group}, "
                        f"head_dim={head_dim})"
                    )
                block_elements = head_dim * shape[1]
                group_elements = blocks_per_group * block_elements
                shard_end = shard_start + values.numel()

                def squared_norm(start: int, end: int) -> float:
                    left, right = max(start, shard_start), min(end, shard_end)
                    if right <= left:
                        return 0.0
                    return float(values[left - shard_start : right - shard_start].double().square().sum().item())

                for local_group in range(kv_per_rank):
                    global_group = tp_rank * kv_per_rank + local_group
                    start = local_group * group_elements
                    q_end = start + q_per_group * block_elements
                    k_end = q_end + block_elements
                    end = start + group_elements
                    local_events.append(
                        {
                            "layer": int(layer),
                            "group": int(global_group),
                            "q_sq": squared_norm(start, q_end),
                            "k_sq": squared_norm(q_end, k_end),
                            "v_sq": squared_norm(k_end, end),
                            "elements": max(0, min(end, shard_end) - max(start, shard_start)),
                            "expected_elements": group_elements,
                            "tp_rank": int(tp_rank),
                        }
                    )

    # Merge optimizer-owned slices across DP and disjoint groups across TP.
    gathered: list[list[dict[str, float | int]]] = [local_events]
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_events)

    return merge_qkv_head_grad_parts(gathered)
