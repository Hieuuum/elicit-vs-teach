"""Split-half ceilings for the 2026-11-12 carry-pattern set (data/ceiling4x4).

Each model's EAP-IG scores on the 4,096 discovery problems are averaged per
half (2,048 problems, 128 per carry pattern in each half). The ceiling at a
circuit size is the Jaccard of the two halves' top-k edge sets, using the same
ranking as fold_overlap.py (descending signed score, ties by index). Chance is
the expected Jaccard of two random k-edge sets.

    python3 ceiling_report.py --root results/ceiling4x4 \
        base=klword/base elicit_parent=klword/elicit_parent ... \
        --out results/ceiling4x4/ceiling.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from fold_overlap import jaccards_at, rank_of  # noqa: E402

from geode.circuits import edge_tests as et  # noqa: E402

FRACS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def chance_jaccard(k: int, n_edges: int) -> float:
    """Expected Jaccard of two independent uniform k-subsets (intersection k^2/E)."""
    i = k * k / n_edges
    return i / (2 * k - i)


def ceilings(mean_a: torch.Tensor, mean_b: torch.Tensor, fracs=FRACS) -> dict[str, float]:
    if mean_a.numel() != mean_b.numel():
        raise ValueError(f"half sizes differ: {mean_a.numel()} vs {mean_b.numel()}")
    n = mean_a.numel()
    ks = [et.size_to_k(f, n) for f in fracs]
    vals = jaccards_at(rank_of(mean_a.double().numpy()), rank_of(mean_b.double().numpy()), ks)
    return {str(f): v for f, v in zip(fracs, vals)}


def report(root: Path, models: dict[str, str], fracs=FRACS) -> dict:
    out: dict = {"fracs": list(fracs), "models": {}, "chance": {}}
    n_edges = None
    for label, sub in models.items():
        sc = torch.load(root / sub / "scores.pt")
        out["models"][label] = ceilings(sc["mean_a"], sc["mean_b"], fracs)
        san = root / sub / "sanity.json"
        if san.is_file():
            out["models"][label]["exact_match"] = json.loads(san.read_text())["exact_match"]
        n_edges = sc["mean_a"].numel()
    if n_edges is not None:
        out["n_edges"] = n_edges
        out["chance"] = {str(f): chance_jaccard(et.size_to_k(f, n_edges), n_edges) for f in fracs}
    return out


def markdown(rep: dict) -> str:
    fr = rep["fracs"]
    head = "| model | EM | " + " | ".join(f"{100 * f:g}%" for f in fr) + " |"
    lines = [head, "|" + "---|" * (len(fr) + 2)]
    for lab, d in rep["models"].items():
        em = d.get("exact_match")
        lines.append(
            f"| {lab} | {'' if em is None else f'{em:.3f}'} | "
            + " | ".join(f"{d[str(f)]:.3f}" for f in fr)
            + " |"
        )
    if rep["chance"]:
        lines.append(
            "| chance | | " + " | ".join(f"{rep['chance'][str(f)]:.4f}" for f in fr) + " |"
        )
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("models", nargs="+", help="label=subdir/tag (holding scores.pt)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    models = dict(m.split("=", 1) for m in a.models)
    rep = report(a.root, models)
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
    print(markdown(rep))


if __name__ == "__main__":
    main()
