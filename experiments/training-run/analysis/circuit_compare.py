"""Circuit comparison: Jaccard overlap + score rotation between two node maps.

Metric 2 (Prakash-et-al-style circuit overlap) and the shared-node half of
metric 3, computed from two ``circuit_nodes.py`` outputs:

- **Jaccard@k** over top-k |score| node sets, for k in --ks: the fraction of
  circuit membership shared. Elicitation prediction: high (fine-tuning
  reuses the base circuit); teaching: ~chance (no base circuit to reuse).
  Chance level for Jaccard@k over N=528 nodes is ~ k/(2N-k) (reported).
- **Score rotation on shared nodes**: 1 - Spearman correlation of |score|
  over the union of the two top-k sets — "same nodes, changed weighting",
  the node-level shadow of edge rewiring (full edge-EAP is a planned v2).

REFUSES to interpret (prints NOISE verdict) when either sidecar reports a
non-performing regime — a Jaccard against noise is not evidence of anything.
CPU-only.

Usage:
    python3 circuit_compare.py A_stem B_stem [--ks 32 64 128]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(stem: str):
    df = pd.read_parquet(Path(stem).with_suffix(".parquet"))
    meta = json.loads(Path(stem).with_suffix(".json").read_text())
    df["node"] = df["node_type"] + ":" + df["layer"].astype(str) + ":" + df["head"].astype(str)
    return df.set_index("node"), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128])
    args = ap.parse_args()

    da, ma = load(args.a)
    db, mb = load(args.b)
    n_nodes = len(da)
    print(f"[compare] A: {ma['model']} shots={ma['shots']} "
          f"mean_logit_diff={ma['mean_logit_diff']:.3f} "
          f"{'PERFORMING' if ma['performing_regime'] else 'NOT PERFORMING'}")
    print(f"[compare] B: {mb['model']} shots={mb['shots']} "
          f"mean_logit_diff={mb['mean_logit_diff']:.3f} "
          f"{'PERFORMING' if mb['performing_regime'] else 'NOT PERFORMING'}")
    if not (ma["performing_regime"] and mb["performing_regime"]):
        print("[compare] VERDICT GUARD: at least one map is NOISE (non-performing "
              "regime); the numbers below measure noise overlap, not circuit reuse.")

    for k in args.ks:
        top_a = set(da["abs_score"].nlargest(k).index)
        top_b = set(db["abs_score"].nlargest(k).index)
        inter, union = top_a & top_b, top_a | top_b
        jacc = len(inter) / len(union)
        chance = k / (2 * n_nodes - k)
        # Type-matched chance (2026-09-22): an MLP block's score is a dot product
        # over 2048 dims, a head's over 64, so every top-k is MLP-heavy by
        # construction and two unrelated maps share MLPs more often than the
        # uniform null says. Expected Jaccard when each map keeps its own
        # MLP/head counts but the members are drawn at random within type.
        def is_mlp(idx):
            return str(idx).startswith("mlp")
        n_mlp = sum(is_mlp(i) for i in da.index)
        ma_, mb_ = sum(is_mlp(i) for i in top_a), sum(is_mlp(i) for i in top_b)
        e_inter = ma_ * mb_ / max(n_mlp, 1) + (k - ma_) * (k - mb_) / max(n_nodes - n_mlp, 1)
        chance_t = e_inter / (2 * k - e_inter)
        # Spearman by hand (rank -> Pearson): the cluster env has no scipy.
        ra = da.loc[list(union), "abs_score"].rank()
        rb = db.loc[list(union), "abs_score"].rank()
        shared = ra.corr(rb)  # Pearson of ranks == Spearman
        print(f"[compare] k={k:4d}: Jaccard {jacc:.3f} (chance ~{chance:.3f}; "
              f"type-matched ~{chance_t:.3f}, MLPs {ma_}/{mb_} of {k}) "
              f"| shared {len(inter):3d}/{len(union)} "
              f"| union-score Spearman {shared:.3f} (rotation {1 - shared:.3f})")

    # heads-only overlap (2026-09-22): the MLP blocks are 16 of 528 nodes but
    # fill ~half of every top-32 by score size; the attention heads are where
    # two circuits can actually differ. Top-k heads by |score|, chance k/(2H-k).
    heads_a = da[~da.index.astype(str).str.startswith("mlp")]["abs_score"]
    heads_b = db[~db.index.astype(str).str.startswith("mlp")]["abs_score"]
    n_heads = len(heads_a)
    for k in (8, 16, 32):
        if k >= n_heads:  # tiny test models have fewer heads than k
            continue
        ta_, tb_ = set(heads_a.nlargest(k).index), set(heads_b.nlargest(k).index)
        j = len(ta_ & tb_) / len(ta_ | tb_)
        print(f"[compare] heads-only k={k:3d}: Jaccard {j:.3f} (chance ~{k / (2 * n_heads - k):.3f}) "
              f"| shared {len(ta_ & tb_):3d}/{len(ta_ | tb_)}")

    # top-16 nodes side by side, for eyeballing which heads carry the task
    print("[compare] top-16 by |score|:")
    ta = da["abs_score"].nlargest(16).index.tolist()
    tb = db["abs_score"].nlargest(16).index.tolist()
    for x, y in zip(ta, tb):
        both = "  <== shared" if x in tb and y in ta else ""
        print(f"[compare]   A {x:<14} B {y:<14}{both}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
