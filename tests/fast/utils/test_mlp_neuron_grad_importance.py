import pytest
import torch

from miles.utils.mlp_neuron_grad_importance import (
    assert_single_dp_for_grad_norms,
    compute_local_mlp_neuron_grad_norms,
    merge_mlp_neuron_grad_norms,
)


class _FakeParam(torch.nn.Parameter):
    """Parameter carrying a Megatron-style ``main_grad``."""

    def __new__(cls, data: torch.Tensor, main_grad: torch.Tensor):
        param = super().__new__(cls, data, requires_grad=True)
        param.main_grad = main_grad
        return param


class _FakeChunk:
    def __init__(self, named: dict[str, torch.nn.Parameter]):
        self._named = named

    def named_parameters(self):
        return list(self._named.items())


def _chunk(layers: int, ffn_local: int, hidden: int, *, seed: int = 0, prefix: str = "decoder"):
    generator = torch.Generator().manual_seed(seed)
    named = {}
    for layer in range(layers):
        fc1 = torch.randn(2 * ffn_local, hidden, generator=generator)
        fc2 = torch.randn(hidden, ffn_local, generator=generator)
        named[f"{prefix}.layers.{layer}.mlp.linear_fc1.weight"] = _FakeParam(torch.zeros_like(fc1), fc1)
        named[f"{prefix}.layers.{layer}.mlp.linear_fc2.weight"] = _FakeParam(torch.zeros_like(fc2), fc2)
        # Non-MLP / adapter tensors must be ignored.
        named[f"{prefix}.layers.{layer}.self_attention.linear_qkv.weight"] = _FakeParam(
            torch.zeros(hidden, hidden), torch.ones(hidden, hidden)
        )
        named[f"{prefix}.layers.{layer}.mlp.linear_fc1.lora_b.weight"] = _FakeParam(
            torch.zeros(2 * ffn_local, hidden), torch.ones(2 * ffn_local, hidden)
        )
    return _FakeChunk(named), named


def test_norms_match_manual_gate_up_down_grouping():
    layers, ffn_local, hidden = 2, 4, 3
    chunk, named = _chunk(layers, ffn_local, hidden, seed=1)

    parts = compute_local_mlp_neuron_grad_norms(chunk_list := [chunk], ffn_local, tp_rank=0, tp_size=1)
    events = merge_mlp_neuron_grad_norms(parts)
    assert len(events) == layers * ffn_local
    assert chunk_list  # keep the model alive for the duration of the call

    by_key = {(e["layer"], e["neuron"]): e for e in events}
    for layer in range(layers):
        fc1 = named[f"decoder.layers.{layer}.mlp.linear_fc1.weight"].main_grad
        fc2 = named[f"decoder.layers.{layer}.mlp.linear_fc2.weight"].main_grad
        for neuron in range(ffn_local):
            event = by_key[(layer, neuron)]
            gate = fc1[neuron]
            up = fc1[ffn_local + neuron]
            down = fc2[:, neuron]
            assert event["gate_norm"] == pytest.approx(gate.norm().item(), rel=1e-6)
            assert event["up_norm"] == pytest.approx(up.norm().item(), rel=1e-6)
            assert event["down_norm"] == pytest.approx(down.norm().item(), rel=1e-6)
            expected = torch.cat([gate, up, down]).norm().item()
            assert event["grad_norm"] == pytest.approx(expected, rel=1e-6)


def test_tp_shards_cover_disjoint_global_neurons():
    layers, ffn_local, hidden, tp_size = 2, 4, 3, 2
    ffn_hidden = ffn_local * tp_size
    parts = []
    for tp_rank in range(tp_size):
        chunk, _ = _chunk(layers, ffn_local, hidden, seed=10 + tp_rank)
        parts.extend(
            compute_local_mlp_neuron_grad_norms([chunk], ffn_hidden, tp_rank=tp_rank, tp_size=tp_size)
        )

    events = merge_mlp_neuron_grad_norms(parts)
    assert len(events) == layers * ffn_hidden
    for layer in range(layers):
        neurons = sorted(e["neuron"] for e in events if e["layer"] == layer)
        assert neurons == list(range(ffn_hidden))


def test_missing_fc2_block_raises():
    chunk = _FakeChunk(
        {"decoder.layers.0.mlp.linear_fc1.weight": _FakeParam(torch.zeros(8, 3), torch.ones(8, 3))}
    )
    with pytest.raises(ValueError, match="missing MLP gradient blocks"):
        compute_local_mlp_neuron_grad_norms([chunk], 4, tp_rank=0, tp_size=1)


def test_unexpected_fc1_shape_raises():
    chunk = _FakeChunk(
        {"decoder.layers.0.mlp.linear_fc1.weight": _FakeParam(torch.zeros(5, 3), torch.ones(5, 3))}
    )
    with pytest.raises(ValueError, match="fused fc1 grad rows"):
        compute_local_mlp_neuron_grad_norms([chunk], 4, tp_rank=0, tp_size=1)


def test_moe_mlp_is_rejected():
    chunk = _FakeChunk(
        {
            "decoder.layers.0.mlp.experts.linear_fc1.weight": _FakeParam(
                torch.zeros(8, 3), torch.ones(8, 3)
            )
        }
    )
    with pytest.raises(NotImplementedError, match="MoE"):
        compute_local_mlp_neuron_grad_norms([chunk], 4, tp_rank=0, tp_size=1)


def test_dp_guard():
    assert_single_dp_for_grad_norms(1)
    with pytest.raises(ValueError, match="data-parallel size 1"):
        assert_single_dp_for_grad_norms(2)
