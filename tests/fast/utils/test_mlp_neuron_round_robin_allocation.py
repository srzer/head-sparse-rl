import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "miles" / "utils" / "build_mlp_neuron_round_robin_allocation_nway.py"

NUM_LAYERS = 2
FFN = 12


def _write_order(path: Path, ranking: list[tuple[int, int]]) -> None:
    with path.open("w") as f:
        for rank, (layer, neuron) in enumerate(ranking):
            f.write(json.dumps({"layer": layer, "neuron": neuron, "importance_rank": rank}) + "\n")


def _all_neurons() -> list[tuple[int, int]]:
    return [(l, n) for l in range(NUM_LAYERS) for n in range(FFN)]


def _run_builder(tmp_path: Path, orders: dict[str, list[tuple[int, int]]], ratio: float = 1 / 3):
    specs = []
    outs = {}
    for name, ranking in orders.items():
        order_path = tmp_path / f"{name}-order.jsonl"
        _write_order(order_path, ranking)
        out = tmp_path / f"{name}-freeze.json"
        outs[name] = out
        specs += ["--task", f"{name}:{order_path}:{out}"]
    subprocess.run(
        [sys.executable, str(BUILDER), *specs, "--active-ratio", str(ratio),
         "--num-layers", str(NUM_LAYERS), "--ffn-hidden-size", str(FFN)],
        check=True, capture_output=True,
    )
    return {name: json.loads(p.read_text()) for name, p in outs.items()}


def _trainable(manifest) -> set[tuple[int, int]]:
    frozen = {(int(l), n) for l, ns in manifest["frozen_neurons"].items() for n in ns}
    return set(_all_neurons()) - frozen


def test_three_way_split_is_disjoint_and_balanced(tmp_path):
    base = _all_neurons()
    manifests = _run_builder(tmp_path, {
        "a": base,
        "b": list(reversed(base)),
        "c": base[FFN:] + base[:FFN],
    })
    allocs = {k: _trainable(m) for k, m in manifests.items()}
    budget = round(NUM_LAYERS * FFN / 3)
    for name, alloc in allocs.items():
        assert len(alloc) == budget, name
    assert allocs["a"] & allocs["b"] == set()
    assert allocs["a"] & allocs["c"] == set()
    assert allocs["b"] & allocs["c"] == set()


def test_each_task_gets_its_own_top_choice(tmp_path):
    base = _all_neurons()
    manifests = _run_builder(tmp_path, {
        "a": base,                       # top choice (0,0)
        "b": list(reversed(base)),       # top choice (1,11)
        "c": base[5:] + base[:5],        # top choice (0,5)
    })
    assert (0, 0) in _trainable(manifests["a"])
    assert (1, FFN - 1) in _trainable(manifests["b"])
    assert (0, 5) in _trainable(manifests["c"])


def test_ratio_above_one_over_n_is_rejected(tmp_path):
    base = _all_neurons()
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_builder(tmp_path, {"a": base, "b": base, "c": base}, ratio=0.5)
    assert b"active_ratio <= 1/num_tasks" in exc.value.stderr


def test_manifest_round_trips_into_a_value_freeze_mask(tmp_path):
    from miles.backends.megatron_utils.lora_utils import (
        apply_full_weight_neuron_value_freeze_from_manifest,
    )

    tp_size, hidden = 2, 3
    ffn_local = FFN // tp_size
    manifest = {
        "version": 1,
        "num_layers": NUM_LAYERS,
        "ffn_hidden_size": FFN,
        # Freeze one neuron owned by rank 0 (2) and one owned by rank 1 (9).
        "frozen_neurons": {"0": [2, 9]},
        "metadata": {},
    }
    path = tmp_path / "neuron-freeze.json"
    path.write_text(json.dumps(manifest))

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Parameter(torch.randn(2 * ffn_local, hidden))
            self.fc2 = torch.nn.Parameter(torch.randn(hidden, ffn_local))

        def named_parameters(self, *a, **k):
            return [("decoder.layers.0.mlp.linear_fc1.weight", self.fc1),
                    ("decoder.layers.0.mlp.linear_fc2.weight", self.fc2)]

    for tp_rank, local in ((0, 2), (1, 9 - ffn_local)):
        chunk = _Chunk()
        stats = apply_full_weight_neuron_value_freeze_from_manifest(
            [chunk], path, FFN, tp_rank=tp_rank, tp_size=tp_size, num_layers=NUM_LAYERS
        )
        assert stats["num_groups"] == ffn_local
        assert stats["frozen_groups"] == 1.0

        fc1_mask = chunk.fc1._miles_lora_value_freeze_train_mask
        fc2_mask = chunk.fc2._miles_lora_value_freeze_train_mask
        # gate row, up row and down column of that neuron are all pinned...
        assert fc1_mask[local].eq(0).all()
        assert fc1_mask[ffn_local + local].eq(0).all()
        assert fc2_mask[:, local].eq(0).all()
        # ...and nothing else is.
        assert fc1_mask.sum().item() == (2 * ffn_local - 2) * hidden
        assert fc2_mask.sum().item() == (ffn_local - 1) * hidden


def test_value_freeze_restores_frozen_neurons_after_a_step(tmp_path):
    from miles.backends.megatron_utils.lora_utils import (
        apply_full_weight_neuron_value_freeze_from_manifest,
        enforce_lora_value_freeze_mask,
    )

    hidden = 3
    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "version": 1, "num_layers": 1, "ffn_hidden_size": 4,
        "frozen_neurons": {"0": [0, 3]}, "metadata": {},
    }))

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Parameter(torch.zeros(8, hidden))
            self.fc2 = torch.nn.Parameter(torch.zeros(hidden, 4))

        def named_parameters(self, *a, **k):
            return [("decoder.layers.0.mlp.linear_fc1.weight", self.fc1),
                    ("decoder.layers.0.mlp.linear_fc2.weight", self.fc2)]

    chunk = _Chunk()
    apply_full_weight_neuron_value_freeze_from_manifest([chunk], path, 4, tp_rank=0, tp_size=1, num_layers=1)
    with torch.no_grad():  # pretend an optimizer step moved everything
        chunk.fc1.add_(1.0)
        chunk.fc2.add_(1.0)
    enforce_lora_value_freeze_mask([chunk])

    for row in (0, 3, 4, 7):  # gate + up rows of neurons 0 and 3
        assert chunk.fc1[row].eq(0).all()
    for row in (1, 2, 5, 6):
        assert chunk.fc1[row].eq(1).all()
    assert chunk.fc2[:, 0].eq(0).all() and chunk.fc2[:, 3].eq(0).all()
    assert chunk.fc2[:, 1].eq(1).all() and chunk.fc2[:, 2].eq(1).all()
