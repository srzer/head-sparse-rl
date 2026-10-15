"""Per-neuron (FFN intermediate channel) gradient-importance from the accumulated
gradients of ``linear_fc1`` / ``linear_fc2``.

Neuron-level counterpart of :mod:`miles.utils.head_grad_importance`. The
indivisible unit here is one FFN intermediate channel ``i``, which the dependency
graph binds together as

* ``gate_proj[i, :]`` -- row ``i`` of the first half of the fused ``linear_fc1``,
* ``up_proj[i, :]``   -- row ``i`` of the second half of the fused ``linear_fc1``,
* ``down_proj[:, i]`` -- column ``i`` of ``linear_fc2``.

The score of a neuron is the L2 norm of its accumulated gradient over all three
blocks (i.e. ``sqrt(||gate_i||^2 + ||up_i||^2 + ||down_i||^2)``). Larger norm =
the channel the RL loss most wants to move = more "important". Run it
full-weight over one big global batch (whole dataset) with a tiny lr, exactly
like the head criterion.

Layout notes (Megatron, ``--swiglu``):

* ``linear_fc1`` is a ColumnParallelLinear of output width ``2 * ffn_hidden_size``.
  Each TP rank stores ``[gate_shard ; up_shard]`` contiguously, i.e. local rows
  ``[0, ffn_local)`` are gate and ``[ffn_local, 2 * ffn_local)`` are up
  (``torch.chunk(w, 2, dim=0)`` in ``apply_swiglu_sharded_factory``).
* ``linear_fc2`` is a RowParallelLinear of input width ``ffn_hidden_size``; each
  TP rank owns the matching column shard.
* Rank ``r`` therefore owns global neurons ``[r * ffn_local, (r + 1) * ffn_local)``
  for both tensors.

The gradient is read from ``param.main_grad`` and is only fully reduced when the
data-parallel size is 1 (with the distributed optimizer, DP > 1 reduce-scatters
the grad buffer, so each rank holds correct values for its own shard only). The
caller is expected to enforce DP == 1; see ``assert_single_dp_for_grad_norms``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora_utils import _LAYER_INDEX_RE

_ADAPTER_MARKERS = (".lora_", ".adapter.")
_MOE_MARKERS = (".experts.", "shared_expert")


def assert_single_dp_for_grad_norms(dp_size: int) -> None:
    """Fail loudly when the accumulated gradient is not fully reduced on every rank.

    miles always enables the distributed optimizer, which reduce-scatters the
    gradient buffer across the data-parallel group: with ``dp_size > 1`` each
    rank only holds the reduced gradient for its own shard of the flat buffer
    (an arbitrary slice that does not respect neuron boundaries), so per-neuron
    norms would silently mix reduced and rank-local partial values.
    """
    if dp_size > 1:
        raise ValueError(
            "--mlp-neuron-grad-importance-output requires data-parallel size 1 so that "
            f"param.main_grad is the fully accumulated gradient (got dp_size={dp_size}). "
            "Use tensor/pipeline parallelism to cover the training GPUs, e.g. "
            "--tensor-model-parallel-size <num_gpus>."
        )


def compute_local_mlp_neuron_grad_norms(
    model: Sequence[torch.nn.Module],
    ffn_hidden_size: int,
    *,
    tp_rank: int,
    tp_size: int,
    layer_offset: int = 0,
) -> list[dict[str, object]]:
    """Per-layer, per-neuron gradient-norm components for this rank's TP shard.

    Must be called BEFORE ``optimizer.step`` (which zeroes ``main_grad``). Returns
    one entry per local layer with three CPU float32 tensors of length
    ``ffn_hidden_size // tp_size`` (``gate`` / ``up`` / ``down``), plus the global
    neuron offset of the shard. Keeping only the norms (instead of a full
    gradient snapshot) is what makes this cheap: the fc1/fc2 gradients of a 4B
    model are several GiB.
    """
    if ffn_hidden_size <= 0:
        return []
    if ffn_hidden_size % tp_size != 0:
        raise ValueError(f"ffn_hidden_size ({ffn_hidden_size}) must be divisible by tp_size ({tp_size})")
    ffn_local = ffn_hidden_size // tp_size

    parts: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if any(marker in name for marker in _ADAPTER_MARKERS):
                    continue
                if name.endswith("mlp.linear_fc1.weight"):
                    kind = "fc1"
                elif name.endswith("mlp.linear_fc2.weight"):
                    kind = "fc2"
                else:
                    # Everything else is either unrelated (attention, norms, biases,
                    # TE's fused layer_norm_weight, extra state) or a MoE expert
                    # weight, which this criterion does not model.
                    if any(marker in name for marker in _MOE_MARKERS) and (
                        ".linear_fc1." in name or ".linear_fc2." in name
                    ):
                        raise NotImplementedError(
                            f"MoE MLPs are not supported by the neuron grad-importance criterion: {name}"
                        )
                    continue

                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                if grad is None:
                    continue

                layer_match = _LAYER_INDEX_RE.search(name)
                if layer_match is None:
                    raise ValueError(f"Cannot extract layer index from MLP parameter name: {name}")
                layer = layer_offset + int(layer_match.group(1))

                g = grad.detach().to(torch.float32)
                layer_parts = parts.setdefault(layer, {})
                if kind == "fc1":
                    # [2 * ffn_local, hidden] laid out as [gate_shard ; up_shard].
                    if g.shape[0] != 2 * ffn_local:
                        raise ValueError(
                            f"{name}: fused fc1 grad rows {g.shape[0]} != expected {2 * ffn_local} "
                            f"(ffn_local={ffn_local}); is --swiglu set?"
                        )
                    layer_parts["gate"] = torch.linalg.vector_norm(g[:ffn_local], dim=1).cpu()
                    layer_parts["up"] = torch.linalg.vector_norm(g[ffn_local:], dim=1).cpu()
                else:
                    # [hidden, ffn_local]: one column per neuron.
                    if g.shape[1] != ffn_local:
                        raise ValueError(
                            f"{name}: fc2 grad cols {g.shape[1]} != expected {ffn_local}"
                        )
                    layer_parts["down"] = torch.linalg.vector_norm(g, dim=0).cpu()

    local_parts: list[dict[str, object]] = []
    for layer, layer_parts in sorted(parts.items()):
        missing = {"gate", "up", "down"} - layer_parts.keys()
        if missing:
            raise ValueError(f"layer {layer}: missing MLP gradient blocks {sorted(missing)}")
        local_parts.append(
            {
                "layer": int(layer),
                "neuron_offset": int(tp_rank * ffn_local),
                "gate": layer_parts["gate"],
                "up": layer_parts["up"],
                "down": layer_parts["down"],
            }
        )
    return local_parts


def merge_mlp_neuron_grad_norms(
    local_parts: list[dict[str, object]],
) -> list[dict[str, float | int]]:
    """All-gather the per-shard norms and expand them into one event per neuron.

    Collective: must be called by every rank. The returned list is only meaningful
    on ranks that need it (the writer); it is identical everywhere.
    """
    gathered: list[list[dict[str, object]]] = [local_parts]
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_parts)

    # (layer, neuron_offset) is owned by one TP rank; DP/PP replicas are identical.
    merged: dict[tuple[int, int], dict[str, object]] = {}
    for rank_parts in gathered:
        for part in rank_parts or []:
            key = (int(part["layer"]), int(part["neuron_offset"]))
            merged.setdefault(key, part)

    events: list[dict[str, float | int]] = []
    for (layer, offset), part in sorted(merged.items()):
        gate = part["gate"].to(torch.float32)
        up = part["up"].to(torch.float32)
        down = part["down"].to(torch.float32)
        total = torch.sqrt(gate.square() + up.square() + down.square())
        for i in range(total.numel()):
            events.append(
                {
                    "layer": int(layer),
                    "neuron": int(offset + i),
                    "grad_norm": float(total[i]),
                    "gate_norm": float(gate[i]),
                    "up_norm": float(up[i]),
                    "down_norm": float(down[i]),
                }
            )
    return events
