"""Convert a grad-importance JSONL (per GQA group) into a block-level head-order
JSONL consumable by build_head_freeze_manifest.py.

Groups are sorted by grad_norm descending (most important first); each group is
expanded into its 6 semantic head blocks (q0..q3, k, v) with a monotonic
activation_index, so the existing manifest builder's rank == grad-importance rank.
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
    groups.sort(key=lambda g: float(g["grad_norm"]), reverse=True)  # most important first
    idx = 0
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for g in groups:
            layer, grp = int(g["layer"]), int(g["group"])
            blocks = [("q", grp * a.q_per_group + i) for i in range(a.q_per_group)]
            blocks += [("k", grp), ("v", grp)]
            for proj, head in blocks:
                f.write(json.dumps({"layer": layer, "projection": proj, "head": head,
                                    "activation_index": idx, "group": grp,
                                    "grad_norm": float(g["grad_norm"])}) + "\n")
                idx += 1
    print(f"wrote {idx} blocks from {len(groups)} groups -> {out}")

if __name__ == "__main__":
    main()
