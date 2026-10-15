"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import hashlib
import json
import logging
import os
import re
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "miles_lora"

# ---------------------------------------------------------------------------
# Unified HF <-> Megatron module name mappings
# ---------------------------------------------------------------------------

# Standard LoRA: merged Q/K/V and merged up/gate
_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_STANDARD_LORA_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate
_CANONICAL_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
}

_CANONICAL_LORA_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Megatron -> HF (inverse mapping, one-to-many)
# Covers both standard LoRA (merged) and CanonicalLoRA (split) module names.
_MEGATRON_TO_HF_MODULES = {
    # Standard LoRA (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # CanonicalLoRA (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
}

_HF_MODULE_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def is_lora_weight_name(name: str) -> bool:
    """Check if a weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


_param_grad_buffer_patched = False


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer to use disable_param_buffers_cpu_backup=True.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  During LoRA training, base weights are
    frozen (requires_grad=False) so DDP only creates buffers for adapter params.

    This patch ensures those buffers are allocated in the "param_buffer" region
    (enable_cpu_backup=False), making them invisible to pause(tag="default") —
    eliminating the need for resume()/pause() around update_weights.

    The patch is idempotent and only takes effect once.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return
    _param_grad_buffer_patched = True

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["disable_param_buffers_cpu_backup"] = True
        kwargs["disable_grad_buffers_cpu_backup"] = True
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (disable cpu backup)")


# ---------------------------------------------------------------------------
# Module name conversion
# ---------------------------------------------------------------------------


def _get_lora_class_name(lora_type: type | object | None) -> str:
    """Resolve LoRA type to its class name string."""
    if lora_type is None:
        return "CanonicalLoRA"
    if isinstance(lora_type, type):
        return lora_type.__name__
    return type(lora_type).__name__


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    lora_type: type | object | None = None,
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.

    HF:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Megatron (LoRA):          linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj,
                              linear_fc1_up, linear_fc1_gate, linear_fc2

    Special values: "all", "all-linear", "all_linear" -> all standard linear modules.
    If input is already in Megatron format, returns as-is.
    """
    class_name = _get_lora_class_name(lora_type)
    is_canonical = class_name == "CanonicalLoRA"

    all_modules = _CANONICAL_LORA_ALL_MODULES if is_canonical else _STANDARD_LORA_ALL_MODULES
    hf_to_megatron = _CANONICAL_LORA_HF_TO_MEGATRON if is_canonical else _STANDARD_LORA_HF_TO_MEGATRON

    # Handle special "all-linear" variants
    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    # Check if already in Megatron format
    if all(m not in _HF_MODULE_NAMES for m in hf_modules if "*" not in m):
        return hf_modules

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)

    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron LoRA target module names to HuggingFace format.

    Supports both standard LoRA and CanonicalLoRA module names.

    Megatron standard:   linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron canonical:  linear_q, linear_k, linear_v, linear_proj,
                         linear_fc1_up, linear_fc1_gate, linear_fc2
    HF:                  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    hf_modules: list[str] = []
    for module in megatron_modules:
        if module in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[module])
        else:
            hf_modules.append(module)
    return hf_modules


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def parse_exclude_modules(args: Namespace, lora_type=None) -> list[str]:
    """Parse and convert exclude_modules argument."""
    exclude_modules: list[str] = []
    raw = getattr(args, "exclude_modules", None)
    if raw:
        if isinstance(raw, str):
            exclude_modules = [m.strip() for m in raw.split(",")]
        else:
            exclude_modules = list(raw)
        exclude_modules = convert_target_modules_to_megatron(exclude_modules, lora_type=lora_type)
    return exclude_modules


def create_lora_instance(args: Namespace):
    """Create a LoRA or CanonicalLoRA instance based on args.

    Returns:
        A LoRA/CanonicalLoRA dataclass instance ready to be applied to a model.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_type_name = getattr(args, "lora_type", "lora").lower()

    if lora_type_name == "canonical_lora":
        lora_cls = CanonicalLoRA
    else:
        lora_cls = LoRA

    target_modules = convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls)
    exclude_modules = parse_exclude_modules(args, lora_type=lora_cls)

    lora = lora_cls(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}, "
        f"exclude_modules={exclude_modules}"
    )
    return lora


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Save LoRA adapter checkpoint to disk.

    Saves in two formats:
    1. **HF PEFT format** (``adapter_model.bin`` + ``adapter_config.json``) for
       external tool compatibility. Uses Megatron-Bridge's ``export_adapter_weights``
       which correctly handles fused QKV / gate-up weight splitting and TP gathering.
    2. **Megatron-native format** (``adapter_megatron_tp{tp}_pp{pp}.pt``) for fast
       checkpoint resume without name/weight conversion. Each TP/PP rank saves its
       own shard with original parameter names.

    When ``optimizer`` is provided, training state (optimizer + LR scheduler) is
    also saved per-rank for checkpoint resume. Base model weights are frozen and
    never change, so they are not saved.

    This function is collective: **all ranks must call it** because the bridge
    export performs TP all-gather internally. Only ``dp_rank == 0`` writes files.
    """
    import json

    from megatron.bridge import AutoBridge

    from miles.utils import megatron_bridge_utils

    save_path = Path(save_dir)
    is_dp_rank_0 = get_parallel_state().intra_dp.rank == 0
    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank

    # Create directory on dp_rank=0, then synchronize
    if is_dp_rank_0:
        save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # ---- Megatron-native format (per TP/PP rank, fast resume) ----
    if is_dp_rank_0:
        adapter_state: dict[str, torch.Tensor] = {}
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if _is_adapter_param_name(name):
                    adapter_state[name] = param.data.cpu()

        native_path = save_path / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")

    # ---- HF PEFT format (uses bridge for correct name/weight conversion) ----
    # Bridge export is collective: all TP ranks participate in the all-gather,
    # so every rank must call export_adapter_weights.
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    lora_state_dict: dict[str, torch.Tensor] = {}
    with megatron_bridge_utils.patch_megatron_model(model):
        for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
            model,
            cpu=True,
            show_progress=False,
        ):
            lora_state_dict[hf_name] = weight

    # Only one rank writes the HF PEFT files (bridge already gathered across TP)
    if is_dp_rank_0 and tp_rank == 0:
        torch.save(lora_state_dict, save_path / "adapter_model.bin")

        target_modules_hf = (
            convert_target_modules_to_hf(list(args.target_modules))
            if args.target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        config = {
            "peft_type": "LORA",
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": target_modules_hf,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        os.sync()
        logger.info(f"Saved HF PEFT adapter to {save_path} with {len(lora_state_dict)} tensors")

    # ---- Training state (optimizer + scheduler) for resume ----
    if optimizer is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(
            {
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            },
            save_path / f"training_state_rank{rank}.pt",
        )
        logger.info(f"Saved optimizer/scheduler state to {save_path}")

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    """Load LoRA adapter weights from a saved checkpoint into the model.

    Attempts to load from Megatron-native format first (per-rank ``.pt`` files),
    which preserves the exact TP/PP sharding and requires no name conversion.
    Falls back to HF PEFT ``adapter_model.bin`` if native files are not found
    (not yet implemented for HF PEFT format).

    When ``optimizer`` is provided, also restores training state (optimizer +
    LR scheduler) from a co-located ``training_state_rank*.pt`` file. Pass
    ``optimizer=None`` to load adapter weights only.

    Args:
        model: List of DDP-wrapped model chunks with LoRA layers already applied.
        adapter_path: Path to the adapter checkpoint directory.
        optimizer: If provided, restore optimizer state for training resume.
            Pass None to skip optimizer/scheduler restore.
        opt_param_scheduler: If provided, restore LR scheduler state.

    Returns:
        ``(loaded, iteration)`` — *loaded* is True if adapter weights were
        successfully loaded; *iteration* is the saved iteration number (or None
        if no training state was found).
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        return False, None

    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank

    # ---- Try Megatron-native format first (fast, no conversion needed) ----
    native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
        loaded = 0
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(device=param.device))
                    loaded += 1
        logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")

        iteration = _load_training_state(adapter_dir, optimizer, opt_param_scheduler)
        return True, iteration

    # ---- HF PEFT format (future work) ----
    hf_path = adapter_dir / "adapter_model.bin"
    if hf_path.exists():
        logger.warning(
            f"Found HF PEFT adapter at {hf_path} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
        )
        return False, None

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None


def _load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    """Restore optimizer/scheduler state saved alongside a LoRA adapter checkpoint."""
    if optimizer is None:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None

    # Optimizer state dicts may contain non-tensor objects (e.g. step counts,
    # param group metadata), so full unpickling is required here.
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(training_state["optimizer"])
    logger.info("Restored optimizer state from LoRA checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from LoRA checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming LoRA training from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# LoRA config dict for weight sync to SGLang
# ---------------------------------------------------------------------------


def _head_prox_A_cols(A: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Compute per-head squared group norms from A columns. A: [rank, n_heads*head_dim]."""
    rank = A.shape[0]
    n_heads = A.shape[1] // head_dim
    return A.view(rank, n_heads, head_dim).pow(2).sum(dim=(0, 2))  # [n_heads]


def _head_prox_B_rows(B: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Compute per-head squared group norms from B rows. B: [n_heads*head_dim, rank]."""
    n_heads = B.shape[0] // head_dim
    return B.view(n_heads, head_dim, B.shape[1]).pow(2).sum(dim=(1, 2))  # [n_heads]


def _head_freeze_mask_A_cols(A: torch.Tensor, head_dim: int, frozen_heads: torch.Tensor) -> torch.Tensor:
    """Build an element mask for A column head blocks. A: [rank, n_heads*head_dim]."""
    rank = A.shape[0]
    n_heads = A.shape[1] // head_dim
    mask = torch.ones((rank, n_heads, head_dim), dtype=A.dtype, device=A.device)
    mask[:, frozen_heads, :] = 0
    return mask.view_as(A)


def _head_freeze_mask_B_rows(B: torch.Tensor, head_dim: int, frozen_heads: torch.Tensor) -> torch.Tensor:
    """Build an element mask for B row head blocks. B: [n_heads*head_dim, rank]."""
    n_heads = B.shape[0] // head_dim
    mask = torch.ones((n_heads, head_dim, B.shape[1]), dtype=B.dtype, device=B.device)
    mask[frozen_heads, :, :] = 0
    return mask.view_as(B)


def _install_lora_freeze_mask(param: torch.nn.Parameter, mask: torch.Tensor) -> None:
    """Persistently mark masked parameter entries as frozen."""
    existing = getattr(param, "_miles_lora_freeze_mask", None)
    if existing is None:
        freeze_mask = mask.detach().clone()
        param._miles_lora_freeze_mask = freeze_mask
    else:
        freeze_mask = existing.to(device=mask.device, dtype=mask.dtype).mul_(mask)
        param._miles_lora_freeze_mask = freeze_mask

    param.data.mul_(param._miles_lora_freeze_mask.to(device=param.data.device, dtype=param.data.dtype))


def _value_frozen_heads_from_param(param: torch.nn.Parameter, head_dim: int, axis: str) -> torch.Tensor | None:
    train_mask = getattr(param, "_miles_lora_value_freeze_train_mask", None)
    if train_mask is None:
        return None
    data = param.data
    if axis == "rows":
        return train_mask.view(-1, head_dim, data.shape[1]).sum(dim=(1, 2)) == 0
    return train_mask.view(data.shape[0], -1, head_dim).sum(dim=(0, 2)) == 0


def collect_lora_freeze_mask_stats(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Measure whether frozen LoRA parameter entries are zero."""
    frozen_params: list[torch.nn.Parameter] = []
    for model_chunk in model:
        for _, param in model_chunk.named_parameters():
            if getattr(param, "_miles_lora_freeze_mask", None) is not None:
                frozen_params.append(param)

    if not frozen_params:
        return {"masked_params": 0.0, "masked_elements": 0.0, "param_max_abs": 0.0}

    masked_elements = 0.0
    param_max_abs = 0.0
    with torch.no_grad():
        for param in frozen_params:
            mask = param._miles_lora_freeze_mask.to(device=param.device)
            frozen = mask == 0
            masked_elements += float(frozen.sum().item())
            if frozen.any():
                param_max_abs = max(param_max_abs, float(param.data[frozen].detach().abs().max().item()))

    return {
        "masked_params": float(len(frozen_params)),
        "masked_elements": masked_elements,
        "param_max_abs": param_max_abs,
    }


def enforce_lora_head_freeze_mask(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Force previously frozen LoRA entries back to zero."""
    with torch.no_grad():
        for model_chunk in model:
            for _, param in model_chunk.named_parameters():
                mask = getattr(param, "_miles_lora_freeze_mask", None)
                if mask is None:
                    continue
                param.data.mul_(mask.to(device=param.data.device, dtype=param.data.dtype))
    return collect_lora_freeze_mask_stats(model)


def _install_lora_value_freeze_mask(param: torch.nn.Parameter, train_mask: torch.Tensor) -> None:
    """Freeze entries where train_mask is 0 while keeping their current values."""
    existing = getattr(param, "_miles_lora_value_freeze_train_mask", None)
    if existing is None:
        merged_train_mask = train_mask.detach().clone()
    else:
        merged_train_mask = existing.to(device=train_mask.device, dtype=train_mask.dtype).mul_(train_mask)

    param._miles_lora_value_freeze_train_mask = merged_train_mask
    param._miles_lora_value_freeze_values = param.data.detach().clone()


def collect_lora_value_freeze_stats(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    frozen_params: list[torch.nn.Parameter] = []
    for model_chunk in model:
        for _, param in model_chunk.named_parameters():
            if getattr(param, "_miles_lora_value_freeze_train_mask", None) is not None:
                frozen_params.append(param)

    if not frozen_params:
        return {"masked_params": 0.0, "frozen_elements": 0.0, "max_abs_diff": 0.0}

    frozen_elements = 0.0
    max_abs_diff = 0.0
    with torch.no_grad():
        for param in frozen_params:
            train_mask = param._miles_lora_value_freeze_train_mask.to(device=param.device)
            frozen = train_mask == 0
            frozen_elements += float(frozen.sum().item())
            if frozen.any():
                values = param._miles_lora_value_freeze_values.to(device=param.device, dtype=param.dtype)
                max_abs_diff = max(max_abs_diff, float((param.data[frozen] - values[frozen]).abs().max().item()))

    return {
        "masked_params": float(len(frozen_params)),
        "frozen_elements": frozen_elements,
        "max_abs_diff": max_abs_diff,
    }


def enforce_lora_value_freeze_mask(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Restore previously frozen LoRA entries to their saved values."""
    with torch.no_grad():
        for model_chunk in model:
            for _, param in model_chunk.named_parameters():
                train_mask = getattr(param, "_miles_lora_value_freeze_train_mask", None)
                values = getattr(param, "_miles_lora_value_freeze_values", None)
                if train_mask is None or values is None:
                    continue
                train_mask = train_mask.to(device=param.data.device, dtype=param.data.dtype)
                values = values.to(device=param.data.device, dtype=param.data.dtype)
                param.data.mul_(train_mask).add_(values * (1.0 - train_mask))
    return collect_lora_value_freeze_stats(model)


def zero_frozen_parameter_grads(model: Sequence[torch.nn.Module]) -> dict[str, float]:
    """Zero gradients for parameter entries covered by a freeze mask.

    Megatron commonly exposes the optimizer gradient as ``main_grad`` while
    ordinary PyTorch optimizers use ``grad``. Mask both when present so frozen
    entries do not contribute to gradient clipping inside ``optimizer.step()``.

    Post-step value restoration remains necessary because weight decay and old
    optimizer state can still move a parameter even when its gradient is zero.
    """
    masked_params = 0
    masked_elements = 0
    with torch.no_grad():
        for model_chunk in model:
            for _, param in model_chunk.named_parameters():
                masks = [
                    getattr(param, "_miles_lora_freeze_mask", None),
                    getattr(param, "_miles_lora_value_freeze_train_mask", None),
                ]
                masks = [mask for mask in masks if mask is not None]
                if not masks:
                    continue

                train_mask = masks[0]
                for mask in masks[1:]:
                    train_mask = train_mask * mask.to(device=train_mask.device, dtype=train_mask.dtype)

                masked_params += 1
                masked_elements += int((train_mask == 0).sum().item())

                main_grad = getattr(param, "main_grad", None)
                grad = param.grad
                if main_grad is not None:
                    main_grad.mul_(train_mask.to(device=main_grad.device, dtype=main_grad.dtype))
                if grad is not None and grad is not main_grad:
                    grad.mul_(train_mask.to(device=grad.device, dtype=grad.dtype))

    return {
        "masked_params": float(masked_params),
        "masked_elements": float(masked_elements),
    }


def apply_lora_head_nonzero_value_freeze(
    model: Sequence[torch.nn.Module],
    head_dim: int,
    threshold: float = 1e-6,
    include_o_proj: bool = False,
    reference_adapter_path: str | None = None,
    reference_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, float] | None:
    """Freeze selected non-zero LoRA head blocks at their current model values.

    When a reference adapter/state is provided, it selects which heads to
    freeze. The values restored by enforcement still come from the current
    loaded model, allowing optimizer/scheduler resume from a later checkpoint
    without freezing heads newly learned after the reference checkpoint.
    """
    if head_dim <= 0:
        return None

    if reference_adapter_path is not None:
        if reference_state is not None:
            raise ValueError("Specify only one of reference_adapter_path and reference_state")
        adapter_dir = Path(reference_adapter_path)
        tp_rank = get_parallel_state().tp.rank
        pp_rank = get_parallel_state().pp.rank
        native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        if not native_path.exists():
            raise FileNotFoundError(f"LoRA freeze reference shard does not exist: {native_path}")
        reference_state = torch.load(native_path, map_location="cpu", weights_only=True)

    stats = {"num_groups": 0.0, "frozen_groups": 0.0, "train_groups": 0.0, "num_tensors": 0.0}
    head_modules = (
        _HEAD_MODULES
        if include_o_proj
        else tuple((marker, param_markers, "rows") for marker, param_markers in _HEAD_QKV_B_MODULES)
    )

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue
                selection_data = data if reference_state is None else reference_state.get(name)
                if selection_data is None:
                    continue
                if selection_data.shape != data.shape:
                    raise ValueError(
                        f"LoRA freeze reference shape mismatch for {name}: "
                        f"reference={tuple(selection_data.shape)}, current={tuple(data.shape)}"
                    )
                selection_data = selection_data.to(device=data.device, dtype=data.dtype)

                for marker, param_markers, axis in head_modules:
                    if marker not in name:
                        continue
                    if not any(param_marker in name for param_marker in param_markers):
                        continue

                    if axis == "rows":
                        if data.shape[0] % head_dim != 0:
                            logger.warning(
                                "%s shape[0]=%d not divisible by head_dim=%d; skipping nonzero freeze",
                                name,
                                data.shape[0],
                                head_dim,
                            )
                            break
                        norms = _head_prox_B_rows(selection_data, head_dim).sqrt()
                        frozen_heads = norms > threshold
                        train_mask = _head_freeze_mask_B_rows(data, head_dim, frozen_heads)
                    else:
                        if data.shape[1] % head_dim != 0:
                            logger.warning(
                                "%s shape[1]=%d not divisible by head_dim=%d; skipping nonzero freeze",
                                name,
                                data.shape[1],
                                head_dim,
                            )
                            break
                        norms = _head_prox_A_cols(selection_data, head_dim).sqrt()
                        frozen_heads = norms > threshold
                        train_mask = _head_freeze_mask_A_cols(data, head_dim, frozen_heads)

                    _install_lora_value_freeze_mask(param, train_mask)
                    stats["num_groups"] += float(frozen_heads.numel())
                    stats["frozen_groups"] += float(frozen_heads.sum().item())
                    stats["train_groups"] += float((~frozen_heads).sum().item())
                    stats["num_tensors"] += 1.0
                    break

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_groups"], stats["frozen_groups"], stats["train_groups"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_groups = float(sums[0].item())
    if num_groups <= 0.0:
        return None

    frozen_groups = float(sums[1].item())
    train_groups = float(sums[2].item())
    return {
        "num_groups": num_groups,
        "frozen_groups": frozen_groups,
        "frozen_ratio": frozen_groups / num_groups,
        "train_groups": train_groups,
        "train_ratio": train_groups / num_groups,
        "num_tensors": float(sums[3].item()),
    }


def apply_lora_mlp_nonzero_value_freeze(
    model: Sequence[torch.nn.Module],
    threshold: float = 1e-6,
    freeze_all: bool = False,
) -> dict[str, float] | None:
    """Freeze MLP LoRA entries at their current values."""
    stats = {"num_elements": 0.0, "frozen_elements": 0.0, "train_elements": 0.0, "num_tensors": 0.0}

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue
                if not any(marker in name for marker in (".linear_fc1.", ".linear_fc2.")):
                    continue
                if not any(marker in name for marker in (".lora_A.", ".lora_B.", ".adapter.linear_in.", ".adapter.linear_out.")):
                    continue

                if freeze_all:
                    frozen_entries = torch.ones_like(data, dtype=torch.bool)
                else:
                    frozen_entries = data.detach().abs() > threshold
                train_mask = torch.ones_like(data)
                train_mask[frozen_entries] = 0
                _install_lora_value_freeze_mask(param, train_mask)

                stats["num_elements"] += float(frozen_entries.numel())
                stats["frozen_elements"] += float(frozen_entries.sum().item())
                stats["train_elements"] += float((~frozen_entries).sum().item())
                stats["num_tensors"] += 1.0

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_elements"], stats["frozen_elements"], stats["train_elements"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_elements = float(sums[0].item())
    if num_elements <= 0.0:
        return None

    frozen_elements = float(sums[1].item())
    train_elements = float(sums[2].item())
    return {
        "num_elements": num_elements,
        "frozen_elements": frozen_elements,
        "frozen_ratio": frozen_elements / num_elements,
        "train_elements": train_elements,
        "train_ratio": train_elements / num_elements,
        "num_tensors": float(sums[3].item()),
    }


def apply_lora_o_proj_value_freeze(model: Sequence[torch.nn.Module]) -> dict[str, float] | None:
    """Freeze all O projection LoRA entries at their current values."""
    stats = {"num_elements": 0.0, "frozen_elements": 0.0, "train_elements": 0.0, "num_tensors": 0.0}

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue
                if ".linear_proj." not in name:
                    continue
                if not any(
                    marker in name
                    for marker in (".lora_A.", ".lora_B.", ".adapter.linear_in.", ".adapter.linear_out.")
                ):
                    continue

                frozen_entries = torch.ones_like(data, dtype=torch.bool)
                train_mask = torch.zeros_like(data)
                _install_lora_value_freeze_mask(param, train_mask)

                stats["num_elements"] += float(frozen_entries.numel())
                stats["frozen_elements"] += float(frozen_entries.sum().item())
                stats["train_elements"] += 0.0
                stats["num_tensors"] += 1.0

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_elements"], stats["frozen_elements"], stats["train_elements"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_elements = float(sums[0].item())
    if num_elements <= 0.0:
        return None

    frozen_elements = float(sums[1].item())
    train_elements = float(sums[2].item())
    return {
        "num_elements": num_elements,
        "frozen_elements": frozen_elements,
        "frozen_ratio": frozen_elements / num_elements,
        "train_elements": train_elements,
        "train_ratio": train_elements / num_elements,
        "num_tensors": float(sums[3].item()),
    }


def apply_lora_qkv_a_value_freeze(model: Sequence[torch.nn.Module]) -> dict[str, float] | None:
    """Freeze all Q/K/V LoRA A entries at their current values."""
    stats = {"num_elements": 0.0, "frozen_elements": 0.0, "train_elements": 0.0, "num_tensors": 0.0}
    qkv_markers = (".linear_q.", ".linear_k.", ".linear_v.", ".linear_qkv.")
    a_markers = (".lora_A.", ".adapter.linear_in.")

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue
                if not any(marker in name for marker in qkv_markers):
                    continue
                if not any(marker in name for marker in a_markers):
                    continue

                frozen_entries = torch.ones_like(data, dtype=torch.bool)
                train_mask = torch.zeros_like(data)
                _install_lora_value_freeze_mask(param, train_mask)

                stats["num_elements"] += float(frozen_entries.numel())
                stats["frozen_elements"] += float(frozen_entries.sum().item())
                stats["train_elements"] += 0.0
                stats["num_tensors"] += 1.0

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_elements"], stats["frozen_elements"], stats["train_elements"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_elements = float(sums[0].item())
    if num_elements <= 0.0:
        return None

    frozen_elements = float(sums[1].item())
    train_elements = float(sums[2].item())
    return {
        "num_elements": num_elements,
        "frozen_elements": frozen_elements,
        "frozen_ratio": frozen_elements / num_elements,
        "train_elements": train_elements,
        "train_ratio": train_elements / num_elements,
        "num_tensors": float(sums[3].item()),
    }


_HEAD_ZERO_EPS = 1e-6


_HEAD_MODULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (".linear_q.", (".lora_B.", ".adapter.linear_out."), "rows"),
    (".linear_k.", (".lora_B.", ".adapter.linear_out."), "rows"),
    (".linear_v.", (".lora_B.", ".adapter.linear_out."), "rows"),
    (".linear_qkv.", (".lora_B.", ".adapter.linear_out."), "rows"),
    (".linear_proj.", (".lora_A.", ".adapter.linear_in."), "cols"),
)


_HEAD_QKV_B_MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (".linear_q.", (".lora_B.", ".adapter.linear_out.")),
    (".linear_k.", (".lora_B.", ".adapter.linear_out.")),
    (".linear_v.", (".lora_B.", ".adapter.linear_out.")),
    (".linear_qkv.", (".lora_B.", ".adapter.linear_out.")),
)

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _load_lora_head_freeze_manifest(
    path: str | Path,
    *,
    expected_head_dim: int | None = None,
) -> set[tuple[int, str, int]]:
    manifest_path = Path(path)
    with manifest_path.open() as f:
        manifest = json.load(f)

    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported LoRA head freeze manifest version in {manifest_path}")
    if expected_head_dim is not None and manifest.get("head_dim") != expected_head_dim:
        raise ValueError(
            f"LoRA head freeze manifest head_dim mismatch: manifest={manifest.get('head_dim')}, "
            f"model={expected_head_dim}"
        )
    entries = manifest.get("frozen_heads")
    if not isinstance(entries, list):
        raise ValueError(f"LoRA head freeze manifest must contain a frozen_heads list: {manifest_path}")

    frozen_heads: set[tuple[int, str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid frozen head entry in {manifest_path}: {entry!r}")
        layer = entry.get("layer")
        projection = entry.get("projection")
        head = entry.get("head")
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"Invalid layer in frozen head entry: {entry!r}")
        if projection not in {"q", "k", "v"}:
            raise ValueError(f"Invalid projection in frozen head entry: {entry!r}")
        if not isinstance(head, int) or head < 0:
            raise ValueError(f"Invalid head index in frozen head entry: {entry!r}")
        key = (layer, projection, head)
        if key in frozen_heads:
            raise ValueError(f"Duplicate frozen head entry in {manifest_path}: {entry!r}")
        frozen_heads.add(key)
    return frozen_heads


def _global_qkv_head_for_local_group(
    local_group: int,
    *,
    projection: str | None,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    num_query_groups: int,
) -> tuple[str, int]:
    if num_attention_heads % tp_size != 0 or num_query_groups % tp_size != 0:
        raise ValueError(
            "Attention heads and query groups must be divisible by tensor parallel size: "
            f"num_attention_heads={num_attention_heads}, num_query_groups={num_query_groups}, tp_size={tp_size}"
        )

    q_per_rank = num_attention_heads // tp_size
    kv_per_rank = num_query_groups // tp_size
    if projection is not None:
        heads_per_rank = q_per_rank if projection == "q" else kv_per_rank
        if local_group >= heads_per_rank:
            raise ValueError(
                f"Local {projection.upper()} head index {local_group} exceeds per-rank count {heads_per_rank}"
            )
        return projection, tp_rank * heads_per_rank + local_group

    q_per_group = num_attention_heads // num_query_groups
    blocks_per_group = q_per_group + 2
    local_query_group, offset_in_group = divmod(local_group, blocks_per_group)
    if local_query_group >= kv_per_rank:
        raise ValueError(
            f"Local fused QKV head index exceeds expected count {kv_per_rank * blocks_per_group}"
        )
    global_query_group = tp_rank * kv_per_rank + local_query_group
    if offset_in_group < q_per_group:
        return "q", global_query_group * q_per_group + offset_in_group
    if offset_in_group == q_per_group:
        return "k", global_query_group
    return "v", global_query_group


def apply_lora_head_value_freeze_from_manifest(
    model: Sequence[torch.nn.Module],
    manifest_path: str | Path,
    head_dim: int,
    num_attention_heads: int,
    num_query_groups: int,
    *,
    tp_rank: int | None = None,
    tp_size: int | None = None,
    layer_offset: int = 0,
    num_layers: int | None = None,
) -> dict[str, float] | None:
    """Freeze semantic Q/K/V LoRA B head blocks listed in a JSON manifest.

    Manifest head IDs are global ``(layer, projection, head)`` tuples. They are
    mapped to the current tensor-parallel shard, including fused QKV B tensors.
    Listed blocks are restored to their values at installation time after every
    optimizer update.
    """
    if head_dim <= 0:
        return None
    frozen_manifest_heads = _load_lora_head_freeze_manifest(manifest_path, expected_head_dim=head_dim)
    for layer, projection, head in frozen_manifest_heads:
        if num_layers is not None and layer >= num_layers:
            raise ValueError(f"Manifest layer {layer} exceeds model layer count {num_layers}")
        max_heads = num_attention_heads if projection == "q" else num_query_groups
        if head >= max_heads:
            raise ValueError(
                f"Manifest {projection.upper()} head {head} exceeds model head count {max_heads}"
            )
    if tp_rank is None or tp_size is None:
        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size

    stats = {"num_groups": 0.0, "frozen_groups": 0.0, "train_groups": 0.0, "num_tensors": 0.0}
    matched_manifest_heads: set[tuple[int, str, int]] = set()

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2 or not any(marker in name for marker in (".lora_B.", ".adapter.linear_out.")):
                    continue

                projection = None
                if ".linear_q." in name:
                    projection = "q"
                elif ".linear_k." in name:
                    projection = "k"
                elif ".linear_v." in name:
                    projection = "v"
                elif ".linear_qkv." not in name:
                    continue

                layer_match = _LAYER_INDEX_RE.search(name)
                if layer_match is None:
                    raise ValueError(f"Cannot extract layer index from Q/K/V LoRA parameter name: {name}")
                layer = layer_offset + int(layer_match.group(1))
                if data.shape[0] % head_dim != 0:
                    raise ValueError(
                        f"LoRA B rows for {name} are not divisible by head_dim={head_dim}: shape={tuple(data.shape)}"
                    )

                num_local_groups = data.shape[0] // head_dim
                frozen_local = torch.zeros(num_local_groups, dtype=torch.bool, device=data.device)
                for local_group in range(num_local_groups):
                    semantic_projection, global_head = _global_qkv_head_for_local_group(
                        local_group,
                        projection=projection,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        num_attention_heads=num_attention_heads,
                        num_query_groups=num_query_groups,
                    )
                    semantic_head = (layer, semantic_projection, global_head)
                    if semantic_head in frozen_manifest_heads:
                        frozen_local[local_group] = True
                        matched_manifest_heads.add(semantic_head)

                _install_lora_value_freeze_mask(param, _head_freeze_mask_B_rows(data, head_dim, frozen_local))
                stats["num_groups"] += float(num_local_groups)
                stats["frozen_groups"] += float(frozen_local.sum().item())
                stats["train_groups"] += float((~frozen_local).sum().item())
                stats["num_tensors"] += 1.0

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_groups"], stats["frozen_groups"], stats["train_groups"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    manifest_head_order = sorted(frozen_manifest_heads)
    matched = torch.tensor(
        [int(head in matched_manifest_heads) for head in manifest_head_order],
        dtype=torch.int32,
        device=device,
    )
    if dist.is_initialized() and matched.numel() > 0:
        dist.all_reduce(matched, op=dist.ReduceOp.MAX)
    if matched.numel() > 0 and not bool(matched.all().item()):
        missing = [head for head, is_matched in zip(manifest_head_order, matched.tolist(), strict=True) if not is_matched]
        raise ValueError(f"Freeze manifest contains heads not present in the distributed model: {missing[:10]}")

    num_groups = float(sums[0].item())
    if num_groups <= 0.0:
        return None

    frozen_groups = float(sums[1].item())
    return {
        "num_groups": num_groups,
        "frozen_groups": frozen_groups,
        "frozen_ratio": frozen_groups / num_groups,
        "train_groups": float(sums[2].item()),
        "train_ratio": float(sums[2].item()) / num_groups,
        "num_tensors": float(sums[3].item()),
        "manifest_groups": float(len(frozen_manifest_heads)),
        "matched_manifest_groups": float(matched.sum().item()),
    }


def _load_neuron_freeze_manifest(
    manifest_path: str | Path,
    *,
    expected_ffn_hidden_size: int,
    expected_num_layers: int | None = None,
) -> dict[int, set[int]]:
    """Read a neuron freeze manifest into ``layer -> frozen neuron indices``."""
    path = Path(manifest_path)
    with path.open() as f:
        manifest = json.load(f)
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported neuron freeze manifest version in {path}: {manifest.get('version')!r}")
    ffn_hidden_size = manifest.get("ffn_hidden_size")
    if ffn_hidden_size != expected_ffn_hidden_size:
        raise ValueError(
            f"Neuron freeze manifest {path} was built for ffn_hidden_size={ffn_hidden_size}, "
            f"but the model has {expected_ffn_hidden_size}"
        )
    frozen_raw = manifest.get("frozen_neurons")
    if not isinstance(frozen_raw, dict):
        raise ValueError(f"Neuron freeze manifest {path} must map layer -> list of neuron indices")

    frozen: dict[int, set[int]] = {}
    for layer_key, neurons in frozen_raw.items():
        layer = int(layer_key)
        if expected_num_layers is not None and not (0 <= layer < expected_num_layers):
            raise ValueError(f"Manifest layer {layer} exceeds model layer count {expected_num_layers}")
        for neuron in neurons:
            if not (0 <= int(neuron) < expected_ffn_hidden_size):
                raise ValueError(f"Manifest neuron {neuron} (layer {layer}) is out of range")
        frozen[layer] = {int(n) for n in neurons}
    return frozen


def apply_full_weight_neuron_value_freeze_from_manifest(
    model: Sequence[torch.nn.Module],
    manifest_path: str | Path,
    ffn_hidden_size: int,
    *,
    tp_rank: int | None = None,
    tp_size: int | None = None,
    layer_offset: int = 0,
    num_layers: int | None = None,
) -> dict[str, float] | None:
    """Freeze MLP neuron groups of the *base* MLP weights (non-LoRA).

    Neuron-level counterpart of the full-weight attention mask implementation.
    A neuron is the group the dependency graph makes indivisible, so all three of
    its blocks are pinned together:

    * ``mlp.linear_fc1.weight`` -- rows ``i`` (gate) and ``ffn_local + i`` (up) of
      the fused SwiGLU shard, which is stored as ``[gate_shard ; up_shard]``;
    * ``mlp.linear_fc2.weight`` -- column ``i`` (down).

    Listed neurons are held at their installation-time values after every
    optimizer update, so only the complementary (trainable) neurons change.
    """
    if ffn_hidden_size <= 0:
        return None
    if tp_rank is None or tp_size is None:
        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size
    if ffn_hidden_size % tp_size != 0:
        raise ValueError(f"ffn_hidden_size ({ffn_hidden_size}) must be divisible by tp_size ({tp_size})")
    ffn_local = ffn_hidden_size // tp_size

    frozen_by_layer = _load_neuron_freeze_manifest(
        manifest_path,
        expected_ffn_hidden_size=ffn_hidden_size,
        expected_num_layers=num_layers,
    )

    stats = {"num_groups": 0.0, "frozen_groups": 0.0, "train_groups": 0.0, "num_tensors": 0.0}
    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue
                is_fc1 = name.endswith("mlp.linear_fc1.weight")
                is_fc2 = name.endswith("mlp.linear_fc2.weight")
                if not (is_fc1 or is_fc2):
                    continue

                layer_match = _LAYER_INDEX_RE.search(name)
                if layer_match is None:
                    raise ValueError(f"Cannot extract layer index from MLP parameter name: {name}")
                layer = layer_offset + int(layer_match.group(1))

                # This rank owns global neurons [tp_rank * ffn_local, +ffn_local).
                frozen_global = frozen_by_layer.get(layer, set())
                frozen_local = torch.zeros(ffn_local, dtype=torch.bool, device=data.device)
                base = tp_rank * ffn_local
                local_idx = [n - base for n in frozen_global if base <= n < base + ffn_local]
                if local_idx:
                    frozen_local[torch.tensor(local_idx, dtype=torch.long, device=data.device)] = True

                train_mask = torch.ones_like(data)
                if is_fc1:
                    if data.shape[0] != 2 * ffn_local:
                        raise ValueError(
                            f"{name}: fused fc1 rows {data.shape[0]} != expected {2 * ffn_local}; is --swiglu set?"
                        )
                    train_mask[:ffn_local][frozen_local] = 0  # gate rows
                    train_mask[ffn_local:][frozen_local] = 0  # up rows
                else:
                    if data.shape[1] != ffn_local:
                        raise ValueError(f"{name}: fc2 cols {data.shape[1]} != expected {ffn_local}")
                    train_mask[:, frozen_local] = 0  # down columns
                _install_lora_value_freeze_mask(param, train_mask)

                # Count neurons once (on fc1); fc2 masks the same neurons' columns.
                if is_fc1:
                    stats["num_groups"] += float(ffn_local)
                    stats["frozen_groups"] += float(frozen_local.sum().item())
                    stats["train_groups"] += float((~frozen_local).sum().item())
                stats["num_tensors"] += 1.0

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_groups"], stats["frozen_groups"], stats["train_groups"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_groups = float(sums[0].item())
    if num_groups <= 0.0:
        return None
    frozen_groups = float(sums[1].item())
    return {
        "num_groups": num_groups,
        "frozen_groups": frozen_groups,
        "frozen_ratio": frozen_groups / num_groups,
        "train_groups": float(sums[2].item()),
        "train_ratio": float(sums[2].item()) / num_groups,
        "num_tensors": float(sums[3].item()),
        "manifest_groups": float(sum(len(v) for v in frozen_by_layer.values())),
    }


def freeze_full_weight_non_mlp_params(model_module: torch.nn.Module) -> dict[str, float]:
    """Freeze (requires_grad=False) every parameter that is not an MLP fc1/fc2 weight.

    Neuron-level counterpart of the full-weight attention parameter filter:
    only ``mlp.linear_fc1.weight`` / ``mlp.linear_fc2.weight`` stay trainable, so
    the MLP-neuron gradient-importance probe pays grad-reduction and optimizer
    state for the FFN only. Like its attention sibling it must be called on the
    raw model *before* gradient-buffer / optimizer construction.
    """
    stats = {
        "trainable_params": 0.0,
        "frozen_params": 0.0,
        "trainable_elements": 0.0,
        "frozen_elements": 0.0,
    }
    for name, param in model_module.named_parameters():
        is_mlp = name.endswith("mlp.linear_fc1.weight") or name.endswith("mlp.linear_fc2.weight")
        param.requires_grad = is_mlp
        key = "trainable" if is_mlp else "frozen"
        stats[f"{key}_params"] += 1.0
        stats[f"{key}_elements"] += float(param.numel())
    return stats


def collect_lora_head_sparsity_stats(
    model: Sequence[torch.nn.Module],
    head_dim: int,
) -> dict[str, float] | None:
    """Collect current head-level LoRA sparsity without modifying parameters."""
    if head_dim <= 0:
        return None

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.zeros(5, dtype=torch.float64, device=device)
    min_norm = torch.tensor(float("inf"), dtype=torch.float64, device=device)
    max_norm = torch.tensor(0.0, dtype=torch.float64, device=device)

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue

                for marker, param_markers, axis in _HEAD_MODULES:
                    if marker not in name:
                        continue
                    if not any(param_marker in name for param_marker in param_markers):
                        continue

                    if axis == "rows":
                        if data.shape[0] % head_dim != 0:
                            logger.warning(
                                "%s shape[0]=%d not divisible by head_dim=%d; skipping sparsity stats",
                                name,
                                data.shape[0],
                                head_dim,
                            )
                            break
                        norms = _head_prox_B_rows(data, head_dim).sqrt()
                    else:
                        if data.shape[1] % head_dim != 0:
                            logger.warning(
                                "%s shape[1]=%d not divisible by head_dim=%d; skipping sparsity stats",
                                name,
                                data.shape[1],
                                head_dim,
                            )
                            break
                        norms = _head_prox_A_cols(data, head_dim).sqrt()

                    norms = norms.detach().to(device=device, dtype=torch.float64)
                    sums += torch.stack(
                        [
                            norms.new_tensor(float(norms.numel())),
                            norms.new_tensor(1.0),
                            (norms > _HEAD_ZERO_EPS).sum(),
                            (norms <= _HEAD_ZERO_EPS).sum(),
                            norms.sum(),
                        ]
                    )
                    min_norm = torch.minimum(min_norm, norms.amin())
                    max_norm = torch.maximum(max_norm, norms.amax())
                    break

    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_norm, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_norm, op=dist.ReduceOp.MAX)

    num_groups = float(sums[0].item())
    if num_groups <= 0.0:
        return None
    active_groups = float(sums[2].item())
    zero_groups = float(sums[3].item())
    return {
        "num_groups": num_groups,
        "num_tensors": float(sums[1].item()),
        "active_groups": active_groups,
        "active_ratio": active_groups / num_groups,
        "zero_groups": zero_groups,
        "zero_ratio": zero_groups / num_groups,
        "norm_mean": float(sums[4].item()) / num_groups,
        "norm_min": float(min_norm.item()),
        "norm_max": float(max_norm.item()),
    }


def collect_lora_qkv_head_activity(
    model: Sequence[torch.nn.Module],
    head_dim: int,
    threshold: float,
    num_attention_heads: int,
    num_query_groups: int,
    *,
    tp_rank: int | None = None,
    tp_size: int | None = None,
    layer_offset: int = 0,
) -> list[dict[str, float | int | str]]:
    """Collect active semantic Q/K/V LoRA B head blocks.

    Returned head IDs use global ``(layer, projection, head)`` coordinates, so
    tensor-parallel shards and fused QKV tensors can be merged deterministically.
    """
    if head_dim <= 0:
        return []
    if tp_rank is None or tp_size is None:
        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size

    local_events: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2 or not any(marker in name for marker in (".lora_B.", ".adapter.linear_out.")):
                    continue

                projection = None
                if ".linear_q." in name:
                    projection = "q"
                elif ".linear_k." in name:
                    projection = "k"
                elif ".linear_v." in name:
                    projection = "v"
                elif ".linear_qkv." not in name:
                    continue

                layer_match = _LAYER_INDEX_RE.search(name)
                if layer_match is None:
                    raise ValueError(f"Cannot extract layer index from Q/K/V LoRA parameter name: {name}")
                layer = layer_offset + int(layer_match.group(1))
                if data.shape[0] % head_dim != 0:
                    raise ValueError(
                        f"LoRA B rows for {name} are not divisible by head_dim={head_dim}: shape={tuple(data.shape)}"
                    )

                norms = _head_prox_B_rows(data, head_dim).sqrt()
                active_local = norms > threshold
                for local_group, is_active in enumerate(active_local.tolist()):
                    if not is_active:
                        continue
                    semantic_projection, global_head = _global_qkv_head_for_local_group(
                        local_group,
                        projection=projection,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        num_attention_heads=num_attention_heads,
                        num_query_groups=num_query_groups,
                    )
                    local_events.append(
                        {
                            "layer": int(layer),
                            "projection": semantic_projection,
                            "head": int(global_head),
                            "norm": float(norms[local_group].item()),
                            "tp_rank": int(tp_rank),
                        }
                    )

    gathered_events: list[list[dict[str, float | int | str]]] = [local_events]
    if dist.is_initialized():
        gathered_events = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered_events, local_events)

    merged: dict[tuple[int, str, int], dict[str, float | int | str]] = {}
    for rank_events in gathered_events:
        for event in rank_events:
            key = (int(event["layer"]), str(event["projection"]), int(event["head"]))
            if key not in merged or float(event["norm"]) > float(merged[key]["norm"]):
                merged[key] = event

    return list(merged.values())


def apply_lora_head_zero_freeze(
    model: Sequence[torch.nn.Module],
    head_dim: int,
    threshold: float = _HEAD_ZERO_EPS,
) -> dict[str, float] | None:
    """Freeze currently-zero head-level LoRA blocks.

    The freeze is persistent for the lifetime of the parameter: frozen entries
    are recorded in a mask. Later prox steps treat frozen head blocks as scale=0,
    and enforce_lora_head_freeze_mask can force the same entries back to zero
    even when no prox step is run. Repeated calls can add newly-zero blocks to
    the frozen set.
    """
    if head_dim <= 0:
        return None

    stats = {
        "num_groups": 0.0,
        "new_frozen_groups": 0.0,
        "total_frozen_groups": 0.0,
        "num_tensors": 0.0,
    }

    with torch.no_grad():
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue

                for marker, param_markers, axis in _HEAD_MODULES:
                    if marker not in name:
                        continue
                    if not any(param_marker in name for param_marker in param_markers):
                        continue

                    if axis == "rows":
                        if data.shape[0] % head_dim != 0:
                            logger.warning(
                                "%s shape[0]=%d not divisible by head_dim=%d; skipping freeze",
                                name,
                                data.shape[0],
                                head_dim,
                            )
                            break
                        norms = _head_prox_B_rows(data, head_dim).sqrt()
                        frozen_heads = norms <= threshold
                        value_frozen_heads = _value_frozen_heads_from_param(param, head_dim, axis)
                        if value_frozen_heads is None:
                            trainable_heads = torch.ones_like(frozen_heads, dtype=torch.bool)
                        else:
                            trainable_heads = ~value_frozen_heads.to(device=frozen_heads.device)
                            frozen_heads = frozen_heads & trainable_heads
                        element_mask = _head_freeze_mask_B_rows(data, head_dim, frozen_heads)
                    else:
                        if data.shape[1] % head_dim != 0:
                            logger.warning(
                                "%s shape[1]=%d not divisible by head_dim=%d; skipping freeze",
                                name,
                                data.shape[1],
                                head_dim,
                            )
                            break
                        norms = _head_prox_A_cols(data, head_dim).sqrt()
                        frozen_heads = norms <= threshold
                        value_frozen_heads = _value_frozen_heads_from_param(param, head_dim, axis)
                        if value_frozen_heads is None:
                            trainable_heads = torch.ones_like(frozen_heads, dtype=torch.bool)
                        else:
                            trainable_heads = ~value_frozen_heads.to(device=frozen_heads.device)
                            frozen_heads = frozen_heads & trainable_heads
                        element_mask = _head_freeze_mask_A_cols(data, head_dim, frozen_heads)

                    if not trainable_heads.any():
                        break

                    existing = getattr(param, "_miles_lora_freeze_mask", None)
                    if existing is None:
                        old_frozen_heads = torch.zeros_like(frozen_heads, dtype=torch.bool)
                    elif axis == "rows":
                        old_frozen_heads = existing.view(-1, head_dim, data.shape[1]).sum(dim=(1, 2)) == 0
                    else:
                        old_frozen_heads = existing.view(data.shape[0], -1, head_dim).sum(dim=(0, 2)) == 0

                    _install_lora_freeze_mask(param, element_mask)
                    current = getattr(param, "_miles_lora_freeze_mask")
                    if axis == "rows":
                        current_frozen_heads = current.view(-1, head_dim, data.shape[1]).sum(dim=(1, 2)) == 0
                    else:
                        current_frozen_heads = current.view(data.shape[0], -1, head_dim).sum(dim=(0, 2)) == 0

                    stats["num_groups"] += float(trainable_heads.sum().item())
                    stats["new_frozen_groups"] += float((frozen_heads & ~old_frozen_heads).sum().item())
                    stats["total_frozen_groups"] += float((current_frozen_heads & trainable_heads).sum().item())
                    stats["num_tensors"] += 1.0
                    break

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [
            stats["num_groups"],
            stats["new_frozen_groups"],
            stats["total_frozen_groups"],
            stats["num_tensors"],
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_groups = float(sums[0].item())
    if num_groups <= 0.0:
        return None

    total_frozen = float(sums[2].item())
    return {
        "num_groups": num_groups,
        "new_frozen_groups": float(sums[1].item()),
        "total_frozen_groups": total_frozen,
        "frozen_ratio": total_frozen / num_groups,
        "num_tensors": float(sums[3].item()),
    }


def apply_lora_head_random_freeze(
    model: Sequence[torch.nn.Module],
    head_dim: int,
    zero_ratio: float,
    seed: int,
    shard_id: str = "",
) -> dict[str, float] | None:
    """Randomly freeze a fixed fraction of head-level LoRA blocks at zero.

    Sampling is stratified per LoRA tensor so every layer/projection retains
    approximately ``1 - zero_ratio`` of its heads. A stable per-tensor seed
    keeps masks reproducible and identical across data-parallel replicas.
    """
    if head_dim <= 0:
        return None
    if not 0.0 <= zero_ratio <= 1.0:
        raise ValueError(f"random head freeze zero_ratio must be in [0, 1], got {zero_ratio}")

    stats = {
        "num_groups": 0.0,
        "frozen_groups": 0.0,
        "train_groups": 0.0,
        "num_tensors": 0.0,
    }

    with torch.no_grad():
        for chunk_index, model_chunk in enumerate(model):
            for name, param in model_chunk.named_parameters():
                data = param.data
                if data.dim() != 2:
                    continue

                for marker, param_markers, axis in _HEAD_MODULES:
                    if marker not in name or not any(param_marker in name for param_marker in param_markers):
                        continue

                    grouped_dim = data.shape[0] if axis == "rows" else data.shape[1]
                    if grouped_dim % head_dim != 0:
                        logger.warning(
                            "%s grouped dimension=%d not divisible by head_dim=%d; skipping random freeze",
                            name,
                            grouped_dim,
                            head_dim,
                        )
                        break

                    num_heads = grouped_dim // head_dim
                    num_frozen = round(num_heads * zero_ratio)
                    stable_key = f"{seed}:{shard_id}:chunk{chunk_index}:{name}".encode()
                    tensor_seed = int.from_bytes(hashlib.sha256(stable_key).digest()[:8], "little")
                    generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
                    frozen_heads = torch.zeros(num_heads, dtype=torch.bool, device=data.device)
                    if num_frozen > 0:
                        frozen_indices = torch.randperm(num_heads, generator=generator)[:num_frozen].to(data.device)
                        frozen_heads[frozen_indices] = True

                    if axis == "rows":
                        element_mask = _head_freeze_mask_B_rows(data, head_dim, frozen_heads)
                    else:
                        element_mask = _head_freeze_mask_A_cols(data, head_dim, frozen_heads)
                    _install_lora_freeze_mask(param, element_mask)

                    stats["num_groups"] += float(num_heads)
                    stats["frozen_groups"] += float(num_frozen)
                    stats["train_groups"] += float(num_heads - num_frozen)
                    stats["num_tensors"] += 1.0
                    break

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sums = torch.tensor(
        [stats["num_groups"], stats["frozen_groups"], stats["train_groups"], stats["num_tensors"]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    num_groups = float(sums[0].item())
    if num_groups <= 0.0:
        return None

    frozen_groups = float(sums[1].item())
    train_groups = float(sums[2].item())
    return {
        "num_groups": num_groups,
        "frozen_groups": frozen_groups,
        "frozen_ratio": frozen_groups / num_groups,
        "train_groups": train_groups,
        "train_ratio": train_groups / num_groups,
        "num_tensors": float(sums[3].item()),
        "seed": float(seed),
    }


def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    target_modules_hf = (
        convert_target_modules_to_hf(list(args.target_modules))
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules_hf,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
