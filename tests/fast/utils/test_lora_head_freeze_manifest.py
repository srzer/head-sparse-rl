"""Tests for semantic LoRA head-freeze manifest preprocessing."""

import torch

from miles.utils.build_lora_head_freeze_manifest import (
    build_manifest,
    extract_qkv_head_activity,
    select_frozen_heads,
)
from tests.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=10, suite="stage-a-fast")


def _adapter_state(active_q_heads: set[int]) -> dict[str, torch.Tensor]:
    q = torch.zeros(8, 2)
    for head in active_q_heads:
        q[head * 2 : (head + 1) * 2] = 1.0
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": q,
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight": torch.ones(4, 2),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.zeros(4, 2),
    }


def test_primary_only_manifest_selection_uses_semantic_head_ids():
    primary_all, primary_active = extract_qkv_head_activity(
        _adapter_state({0, 1, 2}), head_dim=2, threshold=1e-6
    )
    secondary_all, secondary_active = extract_qkv_head_activity(
        _adapter_state({1, 2, 3}), head_dim=2, threshold=1e-6
    )

    frozen = select_frozen_heads(
        primary_all,
        primary_active,
        strategy="primary-only",
        secondary_all=secondary_all,
        secondary_active=secondary_active,
    )

    assert frozen == {(0, "q", 0)}
    manifest = build_manifest(frozen, head_dim=2)
    assert manifest["frozen_heads"] == [{"layer": 0, "projection": "q", "head": 0}]


def test_secondary_inactive_freezes_exact_complement():
    primary_all, primary_active = extract_qkv_head_activity(
        _adapter_state({0, 1, 2}), head_dim=2, threshold=1e-6
    )
    secondary_all, secondary_active = extract_qkv_head_activity(
        _adapter_state({1, 2, 3}), head_dim=2, threshold=1e-6
    )

    frozen = select_frozen_heads(
        primary_all,
        primary_active,
        strategy="secondary-inactive",
        secondary_all=secondary_all,
        secondary_active=secondary_active,
    )

    assert frozen == primary_all - secondary_active
