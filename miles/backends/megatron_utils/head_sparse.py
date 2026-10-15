"""Full-weight attention-head masking for sparse-head fine-tuning."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state

HeadId = tuple[int, str, int]
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_FROZEN_GROUPS_ATTR = "_miles_head_sparse_frozen_groups"
_GROUP_AXIS_ATTR = "_miles_head_sparse_group_axis"
_HEAD_DIM_ATTR = "_miles_head_sparse_head_dim"
_FROZEN_VALUES_ATTR = "_miles_head_sparse_frozen_values"


def _load_manifest(path: str | Path, expected_head_dim: int) -> set[HeadId]:
    manifest_path = Path(path)
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported head mask version in {manifest_path}: {manifest.get('version')!r}")
    if manifest.get("head_dim") != expected_head_dim:
        raise ValueError(
            f"Head mask {manifest_path} was built for head_dim={manifest.get('head_dim')}, "
            f"but the model has head_dim={expected_head_dim}"
        )

    frozen_heads: set[HeadId] = set()
    for item in manifest.get("frozen_heads", []):
        try:
            head = (int(item["layer"]), str(item["projection"]), int(item["head"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid frozen head in {manifest_path}: {item!r}") from exc
        if head[0] < 0 or head[1] not in {"q", "k", "v"} or head[2] < 0:
            raise ValueError(f"Invalid frozen head in {manifest_path}: {item!r}")
        frozen_heads.add(head)
    return frozen_heads


def _global_head_for_local_group(
    local_group: int,
    *,
    q_projection: bool,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    num_query_groups: int,
) -> tuple[str, int]:
    if num_attention_heads % num_query_groups != 0:
        raise ValueError("num_attention_heads must be divisible by num_query_groups")
    if num_attention_heads % tp_size != 0 or num_query_groups % tp_size != 0:
        raise ValueError("Q and KV head counts must be divisible by tensor parallel size")

    q_per_rank = num_attention_heads // tp_size
    kv_per_rank = num_query_groups // tp_size
    if q_projection:
        if local_group >= q_per_rank:
            raise ValueError(f"Local Q head {local_group} exceeds per-rank count {q_per_rank}")
        return "q", tp_rank * q_per_rank + local_group

    q_per_group = num_attention_heads // num_query_groups
    blocks_per_group = q_per_group + 2
    local_query_group, offset = divmod(local_group, blocks_per_group)
    if local_query_group >= kv_per_rank:
        raise ValueError(f"Local fused QKV head {local_group} exceeds the per-rank layout")
    global_query_group = tp_rank * kv_per_rank + local_query_group
    if offset < q_per_group:
        return "q", global_query_group * q_per_group + offset
    if offset == q_per_group:
        return "k", global_query_group
    return "v", global_query_group


def _install_mask(
    param: torch.nn.Parameter,
    frozen_groups: torch.Tensor,
    *,
    group_axis: int,
    head_dim: int,
) -> None:
    frozen_groups = frozen_groups.detach().clone()
    if group_axis == 0:
        grouped = param.data.view(frozen_groups.numel(), head_dim, param.shape[1])
        frozen_values = grouped[frozen_groups].detach().clone()
    else:
        grouped = param.data.view(param.shape[0], frozen_groups.numel(), head_dim)
        frozen_values = grouped[:, frozen_groups, :].detach().clone()
    setattr(param, _FROZEN_GROUPS_ATTR, frozen_groups)
    setattr(param, _GROUP_AXIS_ATTR, group_axis)
    setattr(param, _HEAD_DIM_ATTR, head_dim)
    setattr(param, _FROZEN_VALUES_ATTR, frozen_values)


def freeze_non_attention_parameters(
    model: torch.nn.Module,
    *,
    include_output_projection: bool = True,
    include_mlp: bool = False,
) -> dict[str, float]:
    """Keep only attention projection weights (and optionally MLP weights) trainable."""
    stats = {"trainable_params": 0.0, "frozen_params": 0.0, "trainable_elements": 0.0, "frozen_elements": 0.0}
    for name, param in model.named_parameters():
        trainable = name.endswith("self_attention.linear_qkv.weight") or (
            include_output_projection and name.endswith("self_attention.linear_proj.weight")
        )
        trainable = trainable or (
            include_mlp and (name.endswith("mlp.linear_fc1.weight") or name.endswith("mlp.linear_fc2.weight"))
        )
        param.requires_grad = trainable
        key = "trainable" if trainable else "frozen"
        stats[f"{key}_params"] += 1.0
        stats[f"{key}_elements"] += float(param.numel())
    return stats


def apply_head_mask(
    model: Sequence[torch.nn.Module],
    manifest_path: str | Path,
    head_dim: int,
    num_attention_heads: int,
    num_query_groups: int,
    *,
    include_output_projection: bool = True,
    tp_rank: int | None = None,
    tp_size: int | None = None,
    layer_offset: int = 0,
    num_layers: int | None = None,
) -> dict[str, float] | None:
    """Pin masked Q/K/V rows and matching output-projection columns to initial values."""
    if head_dim <= 0:
        return None
    frozen_heads = _load_manifest(manifest_path, head_dim)
    for layer, projection, head in frozen_heads:
        if num_layers is not None and layer >= num_layers:
            raise ValueError(f"Mask layer {layer} exceeds model layer count {num_layers}")
        max_heads = num_attention_heads if projection == "q" else num_query_groups
        if head >= max_heads:
            raise ValueError(f"Mask {projection.upper()} head {head} exceeds model head count {max_heads}")

    if tp_rank is None or tp_size is None:
        parallel_state = get_parallel_state()
        tp_rank, tp_size = parallel_state.tp.rank, parallel_state.tp.size

    counts = torch.zeros(4, dtype=torch.float64)
    matched: set[HeadId] = set()
    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                is_qkv = name.endswith("self_attention.linear_qkv.weight")
                is_output = include_output_projection and name.endswith("self_attention.linear_proj.weight")
                if param.ndim != 2 or not (is_qkv or is_output):
                    continue
                layer_match = _LAYER_INDEX_RE.search(name)
                if layer_match is None:
                    raise ValueError(f"Cannot extract layer index from attention parameter name: {name}")
                layer = layer_offset + int(layer_match.group(1))
                axis = 0 if is_qkv else 1
                if param.shape[axis] % head_dim != 0:
                    raise ValueError(
                        f"Head axis for {name} is not divisible by head_dim={head_dim}: {tuple(param.shape)}"
                    )

                local_groups = param.shape[axis] // head_dim
                frozen_local = torch.zeros(local_groups, dtype=torch.bool, device=param.device)
                for local_group in range(local_groups):
                    projection, global_head = _global_head_for_local_group(
                        local_group,
                        q_projection=is_output,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        num_attention_heads=num_attention_heads,
                        num_query_groups=num_query_groups,
                    )
                    head_id = (layer, projection, global_head)
                    if head_id in frozen_heads:
                        frozen_local[local_group] = True
                        matched.add(head_id)

                _install_mask(param, frozen_local, group_axis=axis, head_dim=head_dim)
                counts += torch.tensor(
                    [local_groups, int(frozen_local.sum()), int((~frozen_local).sum()), 1], dtype=torch.float64
                )

    reduction_device = next((p.device for chunk in model for p in chunk.parameters()), torch.device("cpu"))
    counts = counts.to(reduction_device)
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        ordered = sorted(frozen_heads)
        matched_flags = torch.tensor([head in matched for head in ordered], dtype=torch.int32, device=reduction_device)
        if matched_flags.numel():
            dist.all_reduce(matched_flags, op=dist.ReduceOp.MAX)
            missing = [head for head, found in zip(ordered, matched_flags.tolist(), strict=True) if not found]
            if missing:
                raise ValueError(f"Head mask contains heads not present in the distributed model: {missing[:10]}")
    elif missing := sorted(frozen_heads - matched):
        raise ValueError(f"Head mask contains heads not present in the model: {missing[:10]}")

    total = float(counts[0].item())
    if total == 0:
        return None
    frozen = float(counts[1].item())
    return {
        "num_groups": total,
        "frozen_groups": frozen,
        "frozen_ratio": frozen / total,
        "train_groups": float(counts[2].item()),
        "train_ratio": float(counts[2].item()) / total,
        "num_tensors": float(counts[3].item()),
        "manifest_groups": float(len(frozen_heads)),
    }


def zero_frozen_gradients(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Exclude masked entries from gradient clipping and optimizer updates."""
    masked_params = masked_elements = 0
    with torch.no_grad():
        for chunk in model:
            for param in chunk.parameters():
                frozen_groups = getattr(param, _FROZEN_GROUPS_ATTR, None)
                if frozen_groups is None:
                    continue
                masked_params += 1
                group_axis = getattr(param, _GROUP_AXIS_ATTR)
                head_dim = getattr(param, _HEAD_DIM_ATTR)
                elements_per_group = head_dim * param.shape[1 - group_axis]
                masked_elements += int(frozen_groups.sum().item()) * elements_per_group
                for grad in (getattr(param, "main_grad", None), param.grad):
                    if grad is not None:
                        frozen = frozen_groups.to(device=grad.device)
                        if group_axis == 0:
                            grad.view(frozen.numel(), head_dim, grad.shape[1])[frozen] = 0
                        else:
                            grad.view(grad.shape[0], frozen.numel(), head_dim)[:, frozen, :] = 0
    return {"masked_params": float(masked_params), "masked_elements": float(masked_elements)}


def restore_frozen_values(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Restore masked entries after an optimizer step."""
    masked_params = frozen_elements = 0
    max_abs_diff = 0.0
    with torch.no_grad():
        for chunk in model:
            for param in chunk.parameters():
                frozen_groups = getattr(param, _FROZEN_GROUPS_ATTR, None)
                values = getattr(param, _FROZEN_VALUES_ATTR, None)
                if frozen_groups is None or values is None:
                    continue
                frozen = frozen_groups.to(device=param.device)
                values = values.to(device=param.device, dtype=param.dtype)
                group_axis = getattr(param, _GROUP_AXIS_ATTR)
                head_dim = getattr(param, _HEAD_DIM_ATTR)
                masked_params += 1
                frozen_elements += values.numel()
                if group_axis == 0:
                    grouped = param.data.view(frozen.numel(), head_dim, param.shape[1])
                    grouped[frozen] = values
                    restored = grouped[frozen]
                else:
                    grouped = param.data.view(param.shape[0], frozen.numel(), head_dim)
                    grouped[:, frozen, :] = values
                    restored = grouped[:, frozen, :]
                if values.numel():
                    max_abs_diff = max(max_abs_diff, float((restored - values).abs().max()))
    return {
        "masked_params": float(masked_params),
        "frozen_elements": float(frozen_elements),
        "max_abs_diff": max_abs_diff,
    }
