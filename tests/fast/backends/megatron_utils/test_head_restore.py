import json

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.megatron_utils.update_weight.head_restore import TrainedHeadBaseRestorer


def _build_fixture(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 4,
            }
        )
    )
    q_name = "model.layers.0.self_attn.q_proj.weight"
    o_name = "model.layers.0.self_attn.o_proj.weight"
    q_base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    o_base = torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)
    save_file({q_name: q_base, o_name: o_base}, base_dir / "model.safetensors")

    manifest_path = tmp_path / "mask.json"
    manifest_path.write_text(
        json.dumps(
            {
                "head_dim": 2,
                "frozen_heads": [
                    {"layer": 0, "projection": "q", "head": 1},
                    {"layer": 0, "projection": "k", "head": 0},
                    {"layer": 0, "projection": "v", "head": 0},
                ],
            }
        )
    )
    return base_dir, manifest_path, q_name, o_name, q_base, o_base


def test_restores_complement_rows_and_o_columns(tmp_path):
    base_dir, manifest_path, q_name, o_name, q_base, o_base = _build_fixture(tmp_path)
    restorer = TrainedHeadBaseRestorer(str(base_dir), str(manifest_path))

    trained_q = torch.full_like(q_base, 100)
    restored_q = restorer.restore(q_name, trained_q)
    torch.testing.assert_close(restored_q[:2], q_base[:2])
    torch.testing.assert_close(restored_q[2:], trained_q[2:])

    trained_o = torch.full_like(o_base, 200)
    restored_o = restorer.restore(o_name, trained_o)
    torch.testing.assert_close(restored_o[:, :2], o_base[:, :2])
    torch.testing.assert_close(restored_o[:, 2:], trained_o[:, 2:])
    assert restorer.restored_tensors == 2
    assert restorer.restored_head_blocks == 2


def test_rejects_attention_shape_mismatch(tmp_path):
    base_dir, manifest_path, q_name, _, _, _ = _build_fixture(tmp_path)
    restorer = TrainedHeadBaseRestorer(str(base_dir), str(manifest_path))
    with pytest.raises(ValueError, match="Unexpected shape"):
        restorer.restore(q_name, torch.zeros(3, 4))
