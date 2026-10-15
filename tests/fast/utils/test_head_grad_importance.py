"""Verify head ranking against fully reduced gradients, including DP shards."""

from types import SimpleNamespace
import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

from miles.utils.head_grad_importance import (
    QKVGradientShard,
    collect_qkv_head_grad_norms,
    merge_qkv_head_grad_parts,
    snapshot_qkv_head_gradients,
)


class Chunk(torch.nn.Module):
    def __init__(self, gradients):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros_like(gradient)) for gradient in gradients]
        )
        for parameter, gradient in zip(self.weights, gradients, strict=True):
            parameter.main_grad = gradient

    def named_parameters(self, *args, **kwargs):
        for layer, parameter in enumerate(self.weights):
            yield f"decoder.layers.{layer}.self_attention.linear_qkv.weight", parameter


def test_dp1_matches_full_gradient_norms():
    gradient = torch.arange(48, dtype=torch.float32).view(16, 3)
    model = Chunk([gradient])
    events = collect_qkv_head_grad_norms([model], 2, 4, 2, tp_rank=0, tp_size=1)
    grouped = gradient.view(2, 4, 2, 3)
    for event in events:
        group = grouped[event["group"]]
        assert event["grad_norm"] == pytest.approx(float(group.norm()))
        assert event["q_norm"] == pytest.approx(float(group[:2].norm()))
        assert event["k_norm"] == pytest.approx(float(group[2].norm()))
        assert event["v_norm"] == pytest.approx(float(group[3].norm()))


def test_merge_rejects_missing_or_duplicated_gradient_elements():
    part = dict(layer=0, group=0, tp_rank=0, q_sq=1.0, k_sq=2.0, v_sq=3.0, elements=6, expected_elements=6)
    with pytest.raises(ValueError, match="expected 6"):
        merge_qkv_head_grad_parts([[part], [part]])
    with pytest.raises(ValueError, match="expected 6"):
        merge_qkv_head_grad_parts([[dict(part, elements=3)]])


def test_snapshot_copies_only_owned_range():
    gradient = torch.arange(24, dtype=torch.float32).view(8, 3)
    model = Chunk([gradient])
    parameter = model.weights[0]
    optimizer = SimpleNamespace(
        model_param_gbuf_map={parameter: None},
        _get_model_param_range_map=lambda _: {"param": SimpleNamespace(start=5, end=19)},
    )
    snapshot = snapshot_qkv_head_gradients([model], optimizer)
    shard = next(iter(snapshot.values()))
    assert isinstance(shard, QKVGradientShard)
    assert shard.start == 5
    assert shard.shape == (8, 3)
    assert torch.equal(shard.values, gradient.flatten()[5:19])
    gradient.fill_(100)
    assert not torch.all(shard.values == 100)


def _distributed_worker(rank, world_size, store, tp_size):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world_size)
    tp_rank, dp_rank = rank % tp_size, rank // tp_size
    groups = [dist.new_group(list(range(t, world_size, tp_size))) for t in range(tp_size)]
    dp_group = groups[tp_rank]
    dp_size = world_size // tp_size
    # Two parameters in one bucket, with a cut inside the second parameter's Q block.
    raw = torch.arange(52, dtype=torch.float32) * (dp_rank + 1) + tp_rank
    if dp_rank == 1:
        raw[:24] = -torch.arange(24, dtype=torch.float32) - tp_rank
    reference = raw.clone()
    dist.all_reduce(reference, group=dp_group)
    shard_size = raw.numel() // dp_size
    reduced = torch.empty(shard_size)
    dist.reduce_scatter_tensor(reduced, raw, group=dp_group)
    raw.fill_(float("nan"))  # Unowned buffer contents must never enter ranking.
    begin, end = dp_rank * shard_size, (dp_rank + 1) * shard_size
    raw[begin:end].copy_(reduced)
    model = Chunk([raw[:24].view(8, 3), raw[24:48].view(8, 3)])
    ranges = {}
    for layer, parameter in enumerate(model.weights):
        left, right = max(begin, layer * 24), min(end, (layer + 1) * 24)
        if left < right:
            ranges[parameter] = SimpleNamespace(start=left - layer * 24, end=right - layer * 24)
    optimizer = SimpleNamespace(
        model_param_gbuf_map=ranges,
        _get_model_param_range_map=lambda parameter: {"param": ranges[parameter]},
    )
    snapshot = snapshot_qkv_head_gradients([model], optimizer)
    events = collect_qkv_head_grad_norms(
        [model],
        2,
        tp_size * 2,
        tp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        grad_snapshot=snapshot,
    )
    # Gather an independent all-reduce reference across TP, not the sharded buffers.
    references = [None] * world_size
    dist.all_gather_object(references, (tp_rank, reference.tolist()))
    by_tp = dict(references)
    assert len(events) == 2 * tp_size
    for event in events:
        source = torch.tensor(by_tp[event["tp_rank"]])
        block = source[event["layer"] * 24 : (event["layer"] + 1) * 24].view(4, 2, 3)
        assert event["grad_norm"] == pytest.approx(float(block.norm()), rel=1e-6)
        assert event["q_norm"] == pytest.approx(float(block[:2].norm()), rel=1e-6)
        assert event["k_norm"] == pytest.approx(float(block[2].norm()), rel=1e-6)
        assert event["v_norm"] == pytest.approx(float(block[3].norm()), rel=1e-6)
    assert next(event for event in events if event["layer"] == 0)["grad_norm"] == 0
    dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 2])
def test_dp2_reduce_scatter_matches_all_reduce(tmp_path, tp_size):
    context = mp.get_context("fork")
    processes = [
        context.Process(target=_distributed_worker, args=(rank, 2 * tp_size, str(tmp_path / "store"), tp_size))
        for rank in range(2 * tp_size)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
        assert [p.exitcode for p in processes] == [0] * len(processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
