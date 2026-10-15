"""Convert a grad-importance JSONL (per GQA group, with per-projection norms)
into a block-level head-order JSONL consumable by
build_head_freeze_manifest.py.

Unlike build_head_order_from_grad_importance.py (which ranks whole groups by the
group's aggregate grad_norm and keeps a group's q/k/v blocks together), this
ranks EACH q/k/v projection block by its OWN gradient norm (q_norm/k_norm/v_norm)
and interleaves them into a single global order sorted largest-norm first. The
4 query heads of a group share that group's q_norm; k/v each use k_norm/v_norm.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grad-importance", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--q-per-group", type=int, default=4)
    a = ap.parse_args()

    groups = [json.loads(l) for l in open(a.grad_importance) if l.strip()]
    blocks = []  # (score, layer, projection, head)
    for g in groups:
        layer, grp = int(g["layer"]), int(g["group"])
        q_norm, k_norm, v_norm = float(g["q_norm"]), float(g["k_norm"]), float(g["v_norm"])
        for i in range(a.q_per_group):
            blocks.append((q_norm, layer, "q", grp * a.q_per_group + i))
        blocks.append((k_norm, layer, "k", grp))
        blocks.append((v_norm, layer, "v", grp))

    # Sort by per-projection grad norm, largest first. Ties keep group's q heads
    # adjacent (stable within equal score by layer/proj/head).
    blocks.sort(key=lambda b: (-b[0], b[1], b[2], b[3]))

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for idx, (score, layer, proj, head) in enumerate(blocks):
            f.write(json.dumps({"layer": layer, "projection": proj, "head": head,
                                "activation_index": idx, "grad_norm": score}) + "\n")
    print(f"wrote {len(blocks)} blocks from {len(groups)} groups -> {out}")


if __name__ == "__main__":
    main()
