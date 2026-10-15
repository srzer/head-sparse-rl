"""CPU tests for full-weight sparse-head masking."""

from __future__ import annotations

import json

import torch

from miles.backends.megatron_utils.head_sparse import (
    apply_head_mask,
    freeze_non_attention_parameters,
    restore_frozen_values,
    zero_frozen_gradients,
)
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-fast")


class TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).view(16, 4))
        self.output = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).view(4, 8))
        self.mlp = torch.nn.Parameter(torch.ones(4, 4))

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        del prefix, recurse, remove_duplicate
        yield "decoder.layers.0.self_attention.linear_qkv.weight", self.qkv
        yield "decoder.layers.0.self_attention.linear_proj.weight", self.output
        yield "decoder.layers.0.mlp.linear_fc1.weight", self.mlp


def test_sparse_head_mask_restores_qkv_and_output_projection(tmp_path):
    model = TinyAttention()
    initial_qkv = model.qkv.detach().clone()
    initial_output = model.output.detach().clone()
    mask_path = tmp_path / "heads.json"
    mask_path.write_text(
        json.dumps(
            {
                "version": 1,
                "head_dim": 2,
                "frozen_heads": [
                    {"layer": 0, "projection": "q", "head": 2},
                    {"layer": 0, "projection": "k", "head": 0},
                    {"layer": 0, "projection": "v", "head": 1},
                ],
            }
        )
    )

    metrics = apply_head_mask(
        [model],
        mask_path,
        head_dim=2,
        num_attention_heads=4,
        num_query_groups=2,
        tp_rank=0,
        tp_size=1,
        num_layers=1,
    )
    assert metrics == {
        "num_groups": 12.0,
        "frozen_groups": 4.0,
        "frozen_ratio": 1 / 3,
        "train_groups": 8.0,
        "train_ratio": 2 / 3,
        "num_tensors": 2.0,
        "manifest_groups": 3.0,
    }

    model.qkv.grad = torch.ones_like(model.qkv)
    model.output.grad = torch.ones_like(model.output)
    zero_frozen_gradients([model])
    assert torch.count_nonzero(model.qkv.grad.view(8, 2, 4)[[2, 4, 7]]) == 0
    assert torch.count_nonzero(model.output.grad.view(4, 4, 2)[:, 2, :]) == 0

    with torch.no_grad():
        model.qkv.add_(1000)
        model.output.add_(1000)
    stats = restore_frozen_values([model])
    assert stats["max_abs_diff"] == 0.0
    assert torch.equal(model.qkv.view(8, 2, 4)[[2, 4, 7]], initial_qkv.view(8, 2, 4)[[2, 4, 7]])
    assert torch.equal(model.output.view(4, 4, 2)[:, 2, :], initial_output.view(4, 4, 2)[:, 2, :])
    assert torch.equal(model.qkv.view(8, 2, 4)[0], initial_qkv.view(8, 2, 4)[0] + 1000)


def test_freeze_non_attention_parameters():
    model = TinyAttention()
    stats = freeze_non_attention_parameters(model)
    assert model.qkv.requires_grad
    assert model.output.requires_grad
    assert not model.mlp.requires_grad
    assert stats["trainable_params"] == 2.0
    assert stats["frozen_params"] == 1.0
