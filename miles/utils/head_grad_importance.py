"""Per-head (GQA-group) gradient-importance from the accumulated gradient of the
fused ``linear_qkv`` weight.

Self-contained criterion for ranking full-weight attention heads:
run the RL objective over a dataset accumulating the gradient (typically one big
global batch so Megatron sums it into ``param.main_grad``) WITHOUT letting it move
the weights meaningfully, then score each attention "head" -- taken as a full GQA
group of ``(q_per_group Q + 1 K + 1 V)`` blocks that share one KV -- by the L2
norm of its accumulated gradient. Larger norm = the group the RL loss most wants
to move = more "important".

The fused Megatron ``linear_qkv`` weight is laid out per query group as
``[q_0 .. q_(g-1), k, v]`` (each ``head_dim`` rows), i.e. one contiguous
``(q_per_group + 2) * head_dim`` block per group -- exactly the desired unit.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora_utils import _LAYER_INDEX_RE


def collect_qkv_head_grad_norms(
    model: Sequence[torch.nn.Module],
    head_dim: int,
    num_attention_heads: int,
    num_query_groups: int,
    *,
    tp_rank: int,
    tp_size: int,
    layer_offset: int = 0,
    grad_snapshot: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, float | int]]:
    """Return one event per global GQA group with the L2 norm of its accumulated
    ``linear_qkv`` gradient (and the q/k/v sub-norms), merged across TP ranks.

    ``grad_snapshot`` maps parameter name -> accumulated gradient tensor, captured
    just BEFORE ``optimizer.step`` zeroes it (the live ``param.main_grad`` is zero
    once the step has run). When ``grad_snapshot`` is None, reads the live
    ``param.main_grad`` (falling back to ``param.grad``) -- only valid pre-step.
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
        raise ValueError(
            f"num_query_groups ({num_query_groups}) must be divisible by tp_size ({tp_size})"
        )
    kv_per_rank = num_query_groups // tp_size

    local_events: list[dict[str, float | int]] = []
    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                # Base fused QKV weight only -- skip LoRA adapters / non-weight tensors.
                if ".linear_qkv." not in name or not name.endswith(".weight"):
                    continue
                if any(marker in name for marker in (".lora_", ".adapter.")):
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

                g = grad.detach().to(torch.float32)  # [local_out, in]
                out_rows = g.shape[0]
                expected = kv_per_rank * blocks_per_group * head_dim
                if out_rows != expected:
                    raise ValueError(
                        f"{name}: fused-qkv grad rows {out_rows} != expected "
                        f"{expected} (kv_per_rank={kv_per_rank}, blocks_per_group={blocks_per_group}, "
                        f"head_dim={head_dim})"
                    )
                # [kv_per_rank, blocks_per_group, head_dim, in]
                gb = g.view(kv_per_rank, blocks_per_group, head_dim, g.shape[-1])
                for local_group in range(kv_per_rank):
                    block = gb[local_group]  # [blocks_per_group, head_dim, in]
                    global_group = tp_rank * kv_per_rank + local_group
                    q_norm = torch.linalg.vector_norm(block[:q_per_group]).item()
                    k_norm = torch.linalg.vector_norm(block[q_per_group]).item()
                    v_norm = torch.linalg.vector_norm(block[q_per_group + 1]).item()
                    total_norm = torch.linalg.vector_norm(block).item()
                    local_events.append(
                        {
                            "layer": int(layer),
                            "group": int(global_group),
                            "grad_norm": float(total_norm),
                            "q_norm": float(q_norm),
                            "k_norm": float(k_norm),
                            "v_norm": float(v_norm),
                            "tp_rank": int(tp_rank),
                        }
                    )

    # Merge across TP ranks (each rank owns a disjoint set of query groups).
    gathered: list[list[dict[str, float | int]]] = [local_events]
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_events)

    merged: dict[tuple[int, int], dict[str, float | int]] = {}
    for rank_events in gathered:
        for event in rank_events:
            key = (int(event["layer"]), int(event["group"]))
            # Same (layer, group) can only come from one owning rank; keep the
            # larger norm defensively in case of pipeline-parallel duplication.
            if key not in merged or float(event["grad_norm"]) > float(merged[key]["grad_norm"]):
                merged[key] = event
    return list(merged.values())
