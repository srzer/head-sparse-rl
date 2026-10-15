"""Unit tests for miles.backends.megatron_utils.lora_utils.

Tests cover module name conversion, LoRA detection helpers, parameter identification,
exclude-module parsing, and LoRA sync config building — all without GPU.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")


import json
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import (
    LORA_ADAPTER_NAME,
    apply_lora_head_nonzero_value_freeze,
    apply_lora_head_random_freeze,
    apply_lora_head_value_freeze_from_manifest,
    apply_lora_mlp_nonzero_value_freeze,
    apply_lora_o_proj_value_freeze,
    apply_lora_qkv_a_value_freeze,
    apply_lora_head_zero_freeze,
    _get_lora_class_name,
    _is_adapter_param_name,
    build_lora_sync_config,
    collect_lora_head_sparsity_stats,
    collect_lora_qkv_head_activity,
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
    enforce_lora_head_freeze_mask,
    enforce_lora_value_freeze_mask,
    is_lora_enabled,
    is_lora_weight_name,
    parse_exclude_modules,
    zero_frozen_parameter_grads,
)


# ---------------------------------------------------------------------------
# _get_lora_class_name
# ---------------------------------------------------------------------------


class TestGetLoraClassName:
    def test_none_returns_canonical(self):
        assert _get_lora_class_name(None) == "CanonicalLoRA"

    def test_type_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA) == "FakeLoRA"

    def test_instance_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA()) == "FakeLoRA"


# ---------------------------------------------------------------------------
# convert_target_modules_to_megatron
# ---------------------------------------------------------------------------


def _make_lora_type(name: str):
    """Helper to create a mock lora_type whose class name matches *name*."""
    mock = MagicMock()
    type(mock).__name__ = name
    return mock


class TestConvertTargetModulesToMegatron:
    # --- "all-linear" variants ------------------------------------------------

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_canonical(self, shorthand):
        result = convert_target_modules_to_megatron(shorthand, lora_type=None)
        assert result == [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_standard(self, shorthand):
        lora_type = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(shorthand, lora_type=lora_type)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_single_element_list(self, shorthand):
        result = convert_target_modules_to_megatron([shorthand], lora_type=None)
        assert len(result) == 7  # CanonicalLoRA has 7 modules

    # --- HF -> Megatron conversion (standard LoRA) ----------------------------

    def test_hf_to_megatron_standard_dedup(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=lora)
        assert result == ["linear_qkv"]

    def test_hf_to_megatron_standard_all_modules(self):
        lora = _make_lora_type("LoRA")
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=lora)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    # --- HF -> Megatron conversion (CanonicalLoRA) ----------------------------

    def test_hf_to_megatron_canonical_split(self):
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=None)
        assert result == ["linear_q", "linear_k", "linear_v"]

    def test_hf_to_megatron_canonical_gate_up(self):
        result = convert_target_modules_to_megatron(["gate_proj", "up_proj"], lora_type=None)
        assert result == ["linear_fc1_gate", "linear_fc1_up"]

    # --- Already in Megatron format -------------------------------------------

    def test_megatron_format_passthrough(self):
        modules = ["linear_qkv", "linear_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    def test_megatron_format_passthrough_canonical(self):
        modules = ["linear_q", "linear_k", "linear_v"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    # --- Single string input --------------------------------------------------

    def test_single_hf_string_input(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron("o_proj", lora_type=lora)
        assert result == ["linear_proj"]


# ---------------------------------------------------------------------------
# convert_target_modules_to_hf
# ---------------------------------------------------------------------------


class TestConvertTargetModulesToHf:
    def test_standard_linear_qkv(self):
        assert convert_target_modules_to_hf(["linear_qkv"]) == ["q_proj", "k_proj", "v_proj"]

    def test_standard_linear_proj(self):
        assert convert_target_modules_to_hf(["linear_proj"]) == ["o_proj"]

    def test_standard_linear_fc1(self):
        assert convert_target_modules_to_hf(["linear_fc1"]) == ["gate_proj", "up_proj"]

    def test_standard_linear_fc2(self):
        assert convert_target_modules_to_hf(["linear_fc2"]) == ["down_proj"]

    def test_canonical_split_modules(self):
        result = convert_target_modules_to_hf(["linear_q", "linear_k", "linear_v"])
        assert result == ["q_proj", "k_proj", "v_proj"]

    def test_canonical_fc1_gate_up(self):
        result = convert_target_modules_to_hf(["linear_fc1_gate", "linear_fc1_up"])
        assert result == ["gate_proj", "up_proj"]

    def test_unknown_module_passthrough(self):
        assert convert_target_modules_to_hf(["some_custom_module"]) == ["some_custom_module"]

    def test_roundtrip_canonical_all_linear(self):
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=None)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def test_roundtrip_standard_all_linear(self):
        lora = _make_lora_type("LoRA")
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=lora)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# is_lora_enabled
# ---------------------------------------------------------------------------


class TestIsLoraEnabled:
    def test_enabled_by_rank(self):
        args = Namespace(lora_rank=32, lora_adapter_path=None)
        assert is_lora_enabled(args) is True

    def test_enabled_by_adapter_path(self):
        args = Namespace(lora_rank=0, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_enabled_by_both(self):
        args = Namespace(lora_rank=16, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_disabled(self):
        args = Namespace(lora_rank=0, lora_adapter_path=None)
        assert is_lora_enabled(args) is False

    def test_disabled_missing_attrs(self):
        args = Namespace()
        assert is_lora_enabled(args) is False


# ---------------------------------------------------------------------------
# is_lora_weight_name / _is_adapter_param_name
# ---------------------------------------------------------------------------


class TestIsLoraWeightName:
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_B.default.weight",
        ],
    )
    def test_positive(self, name):
        assert is_lora_weight_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    def test_negative(self, name):
        assert is_lora_weight_name(name) is False


class TestIsAdapterParamName:
    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.lora_A.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        ],
    )
    def test_positive(self, name):
        assert _is_adapter_param_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.weight",
            "module.decoder.layers.0.mlp.linear_fc1.weight",
            "module.embedding.word_embeddings.weight",
        ],
    )
    def test_negative(self, name):
        assert _is_adapter_param_name(name) is False


# ---------------------------------------------------------------------------
# parse_exclude_modules
# ---------------------------------------------------------------------------


class TestParseExcludeModules:
    def test_none(self):
        args = Namespace(exclude_modules=None)
        assert parse_exclude_modules(args) == []

    def test_single_module_string(self):
        args = Namespace(exclude_modules="o_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert result == ["linear_proj"]

    def test_comma_separated(self):
        args = Namespace(exclude_modules="o_proj, down_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_list_input(self):
        args = Namespace(exclude_modules=["o_proj", "down_proj"])
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_missing_attr(self):
        args = Namespace()
        assert parse_exclude_modules(args) == []


# ---------------------------------------------------------------------------
# build_lora_sync_config
# ---------------------------------------------------------------------------


class TestBuildLoraSyncConfig:
    def test_basic_config(self):
        args = Namespace(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        )
        config = build_lora_sync_config(args)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 32
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.0
        assert config["bias"] == "none"
        assert config["task_type"] == "CAUSAL_LM"
        assert set(config["target_modules"]) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_no_target_modules_uses_default(self):
        args = Namespace(lora_rank=16, lora_alpha=16, lora_dropout=0.0, target_modules=None)
        config = build_lora_sync_config(args)
        assert len(config["target_modules"]) == 7

    def test_canonical_target_modules(self):
        args = Namespace(
            lora_rank=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["linear_q", "linear_k"],
        )
        config = build_lora_sync_config(args)
        assert config["target_modules"] == ["q_proj", "k_proj"]
        assert config["r"] == 8


# ---------------------------------------------------------------------------
# Head-level mask helpers
# ---------------------------------------------------------------------------


class _TinyNamedParamModule:
    def __init__(self, params):
        self.params = {name: torch.nn.Parameter(value) for name, value in params.items()}

    def named_parameters(self):
        yield from self.params.items()


def test_head_sparsity_stats_collects_zero_ratio_without_mutation():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
        }
    )
    param = module.params["layer.self_attention.linear_q.lora_B.weight"]
    before = param.detach().clone()

    metrics = collect_lora_head_sparsity_stats([module], head_dim=2)

    assert metrics is not None
    assert metrics["num_groups"] == 2.0
    assert metrics["active_groups"] == 1.0
    assert metrics["zero_groups"] == 1.0
    assert metrics["zero_ratio"] == 0.5
    assert metrics["norm_mean"] == pytest.approx(2.5)
    assert metrics["norm_min"] == pytest.approx(0.0)
    assert metrics["norm_max"] == pytest.approx(5.0)
    assert torch.allclose(param, before)


def test_qkv_head_activity_uses_global_semantic_head_ids():
    module = _TinyNamedParamModule(
        {
            "decoder.layers.0.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "decoder.layers.0.self_attention.linear_k.lora_B.weight": torch.tensor(
                [[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [12.0, 0.0]]
            ),
            "decoder.layers.0.self_attention.linear_proj.lora_A.weight": torch.ones(2, 4),
        }
    )

    events = collect_lora_qkv_head_activity(
        [module],
        head_dim=2,
        threshold=1e-6,
        num_attention_heads=4,
        num_query_groups=4,
        tp_rank=1,
        tp_size=2,
    )

    assert [
        (event["layer"], event["projection"], event["head"], event["norm"])
        for event in events
    ] == [
        (0, "q", 2, 5.0),
        (0, "k", 3, 13.0),
    ]


def test_head_zero_freeze_marks_blocks_and_enforces_them():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
        }
    )
    param = module.params["layer.self_attention.linear_q.lora_B.weight"]

    metrics = apply_lora_head_zero_freeze([module], head_dim=2, threshold=1e-6)

    assert metrics is not None
    assert metrics["new_frozen_groups"] == 1.0
    assert metrics["total_frozen_groups"] == 1.0

    with torch.no_grad():
        param[2:] = 1.0
    stats = enforce_lora_head_freeze_mask([module])

    assert stats["param_max_abs"] == 0.0
    assert torch.count_nonzero(param[2:]).item() == 0


def test_head_zero_freeze_handles_a_column_blocks():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_proj.adapter.linear_in.weight": torch.tensor(
                [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        }
    )
    param = module.params["layer.self_attention.linear_proj.adapter.linear_in.weight"]

    metrics = apply_lora_head_zero_freeze([module], head_dim=2, threshold=1e-6)

    assert metrics is not None
    assert metrics["new_frozen_groups"] == 1.0
    with torch.no_grad():
        param[:, 2:] = 1.0

    enforce_lora_head_freeze_mask([module])

    assert torch.count_nonzero(param[:, 2:]).item() == 0


def test_head_zero_freeze_enforce_works_without_prox():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
        }
    )
    param = module.params["layer.self_attention.linear_q.lora_B.weight"]

    apply_lora_head_zero_freeze([module], head_dim=2, threshold=1e-6)
    with torch.no_grad():
        param[2:] = 1.0

    stats = enforce_lora_head_freeze_mask([module])

    assert stats["param_max_abs"] == 0.0
    assert torch.count_nonzero(param[2:]).item() == 0


def test_head_random_freeze_is_reproducible_and_persistent():
    def make_module():
        return _TinyNamedParamModule(
            {
                "layer.self_attention.linear_q.lora_B.weight": torch.ones(8, 2),
                "layer.self_attention.linear_k.lora_B.weight": torch.ones(8, 2),
            }
        )

    module_a = make_module()
    module_b = make_module()

    metrics_a = apply_lora_head_random_freeze([module_a], head_dim=2, zero_ratio=0.5, seed=7)
    metrics_b = apply_lora_head_random_freeze([module_b], head_dim=2, zero_ratio=0.5, seed=7)

    assert metrics_a is not None
    assert metrics_a["frozen_groups"] == 4.0
    assert metrics_a["train_groups"] == 4.0
    assert metrics_a["frozen_ratio"] == 0.5
    for name in module_a.params:
        mask_a = module_a.params[name]._miles_lora_freeze_mask
        mask_b = module_b.params[name]._miles_lora_freeze_mask
        assert torch.equal(mask_a, mask_b)
        with torch.no_grad():
            module_a.params[name].fill_(1.0)

    stats = enforce_lora_head_freeze_mask([module_a])

    assert stats["param_max_abs"] == 0.0
    for param in module_a.params.values():
        frozen_heads = param._miles_lora_freeze_mask.view(4, 2, 2).sum(dim=(1, 2)) == 0
        assert frozen_heads.sum().item() == 2
        assert torch.count_nonzero(param.view(4, 2, 2)[frozen_heads]).item() == 0


def test_head_nonzero_value_freeze_restores_b_row_blocks():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
        }
    )
    param = module.params["layer.self_attention.linear_q.lora_B.weight"]

    metrics = apply_lora_head_nonzero_value_freeze([module], head_dim=2, threshold=1e-6)

    assert metrics is not None
    assert metrics["frozen_groups"] == 1.0
    assert metrics["train_groups"] == 1.0

    with torch.no_grad():
        param[:2] = 9.0
        param[2:] = 1.0
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    assert torch.allclose(param[:2], torch.tensor([[3.0, 0.0], [4.0, 0.0]]))
    assert torch.allclose(param[2:], torch.ones_like(param[2:]))


def test_zero_frozen_parameter_grads_masks_main_grad_before_clipping():
    module = _TinyNamedParamModule({"weight": torch.ones(2, 3)})
    param = module.params["weight"]
    param._miles_lora_value_freeze_train_mask = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    param.main_grad = torch.full_like(param, 2.0)

    stats = zero_frozen_parameter_grads([module])

    assert stats == {"masked_params": 1.0, "masked_elements": 3.0}
    assert torch.equal(param.main_grad[0], torch.zeros(3))
    assert torch.equal(param.main_grad[1], torch.full((3,), 2.0))


def test_zero_frozen_parameter_grads_masks_grad_and_combines_freeze_masks():
    module = _TinyNamedParamModule({"weight": torch.ones(2, 2)})
    param = module.params["weight"]
    param._miles_lora_freeze_mask = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    param._miles_lora_value_freeze_train_mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    param.grad = torch.ones_like(param)

    stats = zero_frozen_parameter_grads([module])

    assert stats == {"masked_params": 1.0, "masked_elements": 2.0}
    assert torch.equal(param.grad, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))


def test_zero_frozen_parameter_grads_is_noop_without_masks():
    module = _TinyNamedParamModule({"weight": torch.ones(2, 2)})
    param = module.params["weight"]
    param.grad = torch.ones_like(param)

    stats = zero_frozen_parameter_grads([module])

    assert stats == {"masked_params": 0.0, "masked_elements": 0.0}
    assert torch.equal(param.grad, torch.ones_like(param))


def test_head_manifest_value_freeze_maps_global_heads_to_fused_tp_rows(tmp_path):
    name = "module.decoder.layers.0.self_attention.linear_qkv.lora_B.weight"
    initial = torch.arange(32, dtype=torch.float32).view(16, 2)
    module = _TinyNamedParamModule({name: initial.clone()})
    manifest_path = tmp_path / "freeze_heads.json"
    manifest_path.write_text(
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

    metrics = apply_lora_head_value_freeze_from_manifest(
        [module],
        manifest_path,
        head_dim=2,
        num_attention_heads=8,
        num_query_groups=4,
        tp_rank=0,
        tp_size=2,
    )

    assert metrics is not None
    assert metrics["num_groups"] == 8.0
    assert metrics["frozen_groups"] == 3.0
    param = module.params[name]
    with torch.no_grad():
        param.fill_(99.0)
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    grouped = param.view(8, 2, 2)
    expected = torch.full_like(grouped, 99.0)
    initial_grouped = initial.view(8, 2, 2)
    expected[2] = initial_grouped[2]  # K head 0
    expected[4] = initial_grouped[4]  # Q head 2
    expected[7] = initial_grouped[7]  # V head 1
    assert torch.equal(grouped, expected)


def test_head_nonzero_value_freeze_can_select_heads_from_reference_state():
    name = "layer.self_attention.linear_q.lora_B.weight"
    module = _TinyNamedParamModule(
        {
            name: torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
            )
        }
    )
    param = module.params[name]
    reference_state = {
        name: torch.tensor(
            [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
    }

    metrics = apply_lora_head_nonzero_value_freeze(
        [module],
        head_dim=2,
        threshold=1e-6,
        reference_state=reference_state,
    )

    assert metrics is not None
    assert metrics["frozen_groups"] == 1.0
    assert metrics["train_groups"] == 1.0

    with torch.no_grad():
        param[:2] = 9.0
        param[2:] = 8.0
    enforce_lora_value_freeze_mask([module])

    assert torch.allclose(param[:2], torch.tensor([[3.0, 0.0], [4.0, 0.0]]))
    assert torch.allclose(param[2:], torch.full_like(param[2:], 8.0))


def test_head_zero_freeze_excludes_on_load_value_frozen_blocks():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
        }
    )
    param = module.params["layer.self_attention.linear_q.lora_B.weight"]

    apply_lora_head_nonzero_value_freeze([module], head_dim=2, threshold=1e-6)
    metrics = apply_lora_head_zero_freeze([module], head_dim=2, threshold=1e-6)

    assert metrics is not None
    assert metrics["num_groups"] == 1.0
    assert metrics["new_frozen_groups"] == 1.0
    assert metrics["total_frozen_groups"] == 1.0

    with torch.no_grad():
        param[:2] = 9.0
        param[2:] = 1.0
    enforce_lora_head_freeze_mask([module])
    enforce_lora_value_freeze_mask([module])

    assert torch.allclose(param[:2], torch.tensor([[3.0, 0.0], [4.0, 0.0]]))
    assert torch.count_nonzero(param[2:]).item() == 0


def test_head_nonzero_value_freeze_leaves_o_projection_trainable():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "layer.self_attention.linear_proj.adapter.linear_in.weight": torch.tensor(
                [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
        }
    )
    q_param = module.params["layer.self_attention.linear_q.lora_B.weight"]
    o_param = module.params["layer.self_attention.linear_proj.adapter.linear_in.weight"]

    metrics = apply_lora_head_nonzero_value_freeze([module], head_dim=2, threshold=1e-6)

    assert metrics is not None
    assert metrics["num_tensors"] == 1.0
    assert metrics["frozen_groups"] == 1.0
    assert metrics["train_groups"] == 1.0

    with torch.no_grad():
        q_param[:2] = 9.0
        o_param[:, :2] = 9.0
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    assert torch.allclose(q_param[:2], torch.tensor([[3.0, 0.0], [4.0, 0.0]]))
    assert torch.allclose(o_param[:, :2], torch.full_like(o_param[:, :2], 9.0))


def test_head_nonzero_value_freeze_can_include_o_projection():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "layer.self_attention.linear_proj.adapter.linear_in.weight": torch.tensor(
                [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
        }
    )
    q_param = module.params["layer.self_attention.linear_q.lora_B.weight"]
    o_param = module.params["layer.self_attention.linear_proj.adapter.linear_in.weight"]

    metrics = apply_lora_head_nonzero_value_freeze([module], head_dim=2, threshold=1e-6, include_o_proj=True)

    assert metrics is not None
    assert metrics["num_tensors"] == 2.0
    assert metrics["frozen_groups"] == 2.0
    assert metrics["train_groups"] == 2.0

    with torch.no_grad():
        q_param[:2] = 9.0
        o_param[:, :2] = 9.0
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    assert torch.allclose(q_param[:2], torch.tensor([[3.0, 0.0], [4.0, 0.0]]))
    assert torch.allclose(o_param[:, :2], torch.tensor([[3.0, 4.0], [0.0, 0.0]]))


def test_o_projection_value_freeze_does_not_freeze_qkv():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor(
                [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "layer.self_attention.linear_proj.lora_A.weight": torch.tensor(
                [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 6.0]]
            ),
            "layer.self_attention.linear_proj.lora_B.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
    )
    q_param = module.params["layer.self_attention.linear_q.lora_B.weight"]
    o_a_param = module.params["layer.self_attention.linear_proj.lora_A.weight"]
    o_b_param = module.params["layer.self_attention.linear_proj.lora_B.weight"]

    metrics = apply_lora_o_proj_value_freeze([module])

    assert metrics is not None
    assert metrics["num_tensors"] == 2.0
    assert metrics["frozen_ratio"] == 1.0

    with torch.no_grad():
        q_param.fill_(9.0)
        o_a_param.fill_(9.0)
        o_b_param.fill_(9.0)
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    assert torch.allclose(q_param, torch.full_like(q_param, 9.0))
    assert torch.allclose(
        o_a_param,
        torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 6.0]]),
    )
    assert torch.allclose(o_b_param, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_mlp_value_freeze_can_freeze_all_loaded_entries():
    module = _TinyNamedParamModule(
        {
            "layer.mlp.linear_fc1.lora_A.weight": torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
            "layer.mlp.linear_fc2.adapter.linear_out.weight": torch.tensor([[0.0, 5.0], [6.0, 0.0]]),
        }
    )
    fc1_param = module.params["layer.mlp.linear_fc1.lora_A.weight"]
    fc2_param = module.params["layer.mlp.linear_fc2.adapter.linear_out.weight"]

    metrics = apply_lora_mlp_nonzero_value_freeze([module], threshold=1e-6, freeze_all=True)

    assert metrics is not None
    assert metrics["num_tensors"] == 2.0
    assert metrics["frozen_ratio"] == 1.0

    with torch.no_grad():
        fc1_param.fill_(9.0)
        fc2_param.fill_(9.0)
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    assert torch.allclose(fc1_param, torch.tensor([[3.0, 0.0], [0.0, 4.0]]))
    assert torch.allclose(fc2_param, torch.tensor([[0.0, 5.0], [6.0, 0.0]]))


def test_qkv_a_value_freeze_only_freezes_qkv_a_matrices():
    module = _TinyNamedParamModule(
        {
            "layer.self_attention.linear_q.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer.self_attention.linear_q.lora_B.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "layer.self_attention.linear_proj.lora_A.weight": torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
            "layer.mlp.linear_fc1.lora_A.weight": torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
        }
    )
    q_a_param = module.params["layer.self_attention.linear_q.lora_A.weight"]
    q_b_param = module.params["layer.self_attention.linear_q.lora_B.weight"]
    o_a_param = module.params["layer.self_attention.linear_proj.lora_A.weight"]
    mlp_a_param = module.params["layer.mlp.linear_fc1.lora_A.weight"]

    metrics = apply_lora_qkv_a_value_freeze([module])

    assert metrics is not None
    assert metrics["num_tensors"] == 1.0
    assert metrics["frozen_ratio"] == 1.0

    with torch.no_grad():
        q_a_param.fill_(0.0)
        q_b_param.fill_(0.0)
        o_a_param.fill_(0.0)
        mlp_a_param.fill_(0.0)
    stats = enforce_lora_value_freeze_mask([module])

    assert stats["max_abs_diff"] == 0.0
    assert torch.allclose(q_a_param, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(q_b_param, torch.zeros_like(q_b_param))
    assert torch.allclose(o_a_param, torch.zeros_like(o_a_param))
    assert torch.allclose(mlp_a_param, torch.zeros_like(mlp_a_param))


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"
