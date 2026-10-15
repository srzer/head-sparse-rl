"""Restore selected attention-head blocks to their base-model weights at export time."""

import json
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open


logger = logging.getLogger(__name__)

_ATTENTION_WEIGHT_RE = re.compile(
    r"(?:^|\.)model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<projection>[qkvo])_proj\.weight$"
)


class TrainedHeadBaseRestorer:
    """Restore the complement of a training freeze mask from a base HF checkpoint.

    A full-weight head-freeze manifest lists the heads that were *frozen* during
    a training stage.  Its complement is therefore the set of heads trained in
    that stage.  This class replaces those trained Q/K/V row blocks, plus the
    matching O-projection column blocks for trained Q heads, with the original
    tensors from ``base_model_path``.
    """

    def __init__(self, base_model_path: str, freeze_manifest_path: str):
        self.base_model_path = Path(base_model_path)
        self.freeze_manifest_path = Path(freeze_manifest_path)
        if not self.base_model_path.is_dir():
            raise ValueError(
                "Restoring trained heads requires --hf-checkpoint to be a local HF model directory; "
                f"got {base_model_path!r}"
            )

        with self.freeze_manifest_path.open() as f:
            manifest = json.load(f)
        with (self.base_model_path / "config.json").open() as f:
            config = json.load(f)

        self.head_dim = int(manifest["head_dim"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_q_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config.get("num_key_value_heads", self.num_q_heads))
        self.frozen_heads: set[tuple[int, str, int]] = set()
        for entry in manifest.get("frozen_heads", []):
            key = (int(entry["layer"]), str(entry["projection"]).lower(), int(entry["head"]))
            layer, projection, head = key
            if key in self.frozen_heads:
                raise ValueError(f"Duplicate head in freeze manifest: {key}")
            if not 0 <= layer < self.num_layers:
                raise ValueError(f"Manifest layer is outside the base model: {key}")
            max_heads = self.num_q_heads if projection == "q" else self.num_kv_heads
            if projection not in {"q", "k", "v"} or not 0 <= head < max_heads:
                raise ValueError(f"Invalid head in freeze manifest: {key}")
            self.frozen_heads.add(key)

        index_path = self.base_model_path / "model.safetensors.index.json"
        single_path = self.base_model_path / "model.safetensors"
        if index_path.is_file():
            with index_path.open() as f:
                self.weight_map = json.load(f)["weight_map"]
        elif single_path.is_file():
            with safe_open(single_path, framework="pt", device="cpu") as f:
                self.weight_map = {name: single_path.name for name in f.keys()}
        else:
            raise ValueError(f"No safetensors weights found under {self.base_model_path}")

        self.restored_tensors = 0
        self.restored_head_blocks = 0

    def restore(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """Return ``weight`` with stage-trained head blocks replaced by base values."""
        match = _ATTENTION_WEIGHT_RE.search(name)
        if match is None:
            return weight

        layer = int(match.group("layer"))
        projection = match.group("projection")
        semantic_projection = "q" if projection == "o" else projection
        num_heads = self.num_q_heads if semantic_projection == "q" else self.num_kv_heads
        axis = 1 if projection == "o" else 0
        expected_axis_size = num_heads * self.head_dim
        if weight.ndim != 2 or weight.shape[axis] != expected_axis_size:
            raise ValueError(
                f"Unexpected shape for {name}: {tuple(weight.shape)}; axis {axis} should have "
                f"{expected_axis_size} entries"
            )

        trained_heads = [
            head for head in range(num_heads) if (layer, semantic_projection, head) not in self.frozen_heads
        ]
        if not trained_heads:
            return weight

        base = self._load_base_tensor(name)
        if base.shape != weight.shape:
            raise ValueError(
                f"Base/exported tensor shape mismatch for {name}: base={tuple(base.shape)}, "
                f"exported={tuple(weight.shape)}"
            )

        indices = torch.cat(
            [torch.arange(head * self.head_dim, (head + 1) * self.head_dim) for head in trained_heads]
        )
        restored = weight.clone()
        device_indices = indices.to(device=weight.device)
        base_blocks = base.index_select(axis, indices).to(device=weight.device, dtype=weight.dtype)
        restored.index_copy_(axis, device_indices, base_blocks)
        self.restored_tensors += 1
        self.restored_head_blocks += len(trained_heads)
        logger.info(
            "Restored %d stage-trained %s head blocks in layer %d from base weights",
            len(trained_heads),
            projection.upper(),
            layer,
        )
        return restored

    def _load_base_tensor(self, name: str) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise ValueError(f"Base HF checkpoint has no tensor named {name}") from exc
        with safe_open(self.base_model_path / shard, framework="pt", device="cpu") as f:
            return f.get_tensor(name)
