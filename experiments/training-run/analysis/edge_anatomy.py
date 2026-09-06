"""Anatomy of edge change: is churn REWIRING (stable-node edges) or
RECRUITMENT WIRING (new-node edges)?

Refined metric-3 hypothesis (owner 2026-09-06): ΔS_Edge/ΔS_Node is not a
binary elicit/teach flag but a dial for how much of the ACCESS PATH
pre-exists. Predictions it makes about WHICH edges change:

- Pure elicitation (Llama): new top edges connect STABLE nodes — the same
  machinery, re-routed. "rewired-core" dominates.
- Constructed latency (TS1B-latent): new top edges are written by NEW
  (interface) nodes and read by the layers holding the STABLE engine —
  "recruitment wiring into a stable core" dominates, and kept edges are the
  engine's internal wiring.

Given a node-map pair and an edge-map pair (circuit_nodes / circuit_edges
outputs), classifies every top-K post edge as kept vs new, buckets each by
its writer's node status (stable / recruited / dropped / background), adds
writer-layer histograms for kept vs new edges, and reports score rotation
(Spearman) on the kept edges. CPU, seconds.

Usage:
    python3 edge_anatomy.py --nodes-a PRE --nodes-b POST \
        --edges-a PRE --edges-b POST [--k-nodes 32] [--k-edges 256]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(stem: str):
    df = pd.read_parquet(Path(stem).with_suffix(".parquet"))
    meta = json.loads(Path(stem).with_suffix(".json").read_text())
    return df, meta


def top_nodes(stem: str, k: int) -> set:
    df, meta = load(stem)
    if not meta.get("performing_regime", True):
        print(f"[anatomy] GUARD: {stem} is a NOISE map")
    top = df.sort_values("abs_score", ascending=False).head(k)
    return {(r.node_type, int(r.layer), int(r.head)) for r in top.itertuples()}


def edge_df(stem: str, k: int) -> pd.DataFrame:
    df, meta = load(stem)
    if not meta.get("performing_regime", True):
        print(f"[anatomy] GUARD: {stem} is a NOISE map")
    df = df.sort_values("abs_score", ascending=False).head(k).copy()
    df["key"] = list(zip(df.writer_type, df.writer_layer, df.writer_head,
                         df.reader_type, df.reader_layer))
    return df.set_index("key")


def writer_status(key, stable, recruited, dropped) -> str:
    w = (key[0], int(key[1]), int(key[2]))
    if w in stable:
        return "stable-writer"
    if w in recruited:
        return "recruited-writer"
    if w in dropped:
        return "dropped-writer"
    return "background-writer"


def bucket_counts(keys, stable, recruited, dropped):
    from collections import Counter

    c = Counter(writer_status(k, stable, recruited, dropped) for k in keys)
    n = max(1, len(keys))
    return {b: (c.get(b, 0), c.get(b, 0) / n)
            for b in ("stable-writer", "recruited-writer", "dropped-writer",
                      "background-writer")}


def layer_hist(keys) -> str:
    from collections import Counter

    c = Counter(int(k[1]) for k in keys)
    return " ".join(f"L{layer}:{c[layer]}" for layer in sorted(c))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes-a", required=True, help="PRE node map stem")
    ap.add_argument("--nodes-b", required=True, help="POST node map stem")
    ap.add_argument("--edges-a", required=True, help="PRE edge map stem")
    ap.add_argument("--edges-b", required=True, help="POST edge map stem")
    ap.add_argument("--k-nodes", type=int, default=32)
    ap.add_argument("--k-edges", type=int, default=256)
    args = ap.parse_args()

    na = top_nodes(args.nodes_a, args.k_nodes)
    nb = top_nodes(args.nodes_b, args.k_nodes)
    stable, recruited, dropped = na & nb, nb - na, na - nb
    print(f"[anatomy] nodes @K={args.k_nodes}: stable {len(stable)}  "
          f"recruited {len(recruited)}  dropped {len(dropped)}")

    ea = edge_df(args.edges_a, args.k_edges)
    eb = edge_df(args.edges_b, args.k_edges)
    kept = set(ea.index) & set(eb.index)
    new = set(eb.index) - set(ea.index)
    lost = set(ea.index) - set(eb.index)
    print(f"[anatomy] edges @K={args.k_edges}: kept {len(kept)}  "
          f"new {len(new)}  lost {len(lost)}")

    for label, keys in (("KEPT edges", kept), ("NEW edges", new), ("LOST edges", lost)):
        b = bucket_counts(keys, stable, recruited, dropped)
        line = "  ".join(f"{name} {cnt} ({frac:.0%})" for name, (cnt, frac) in b.items()
                         if cnt or name in ("stable-writer", "recruited-writer"))
        print(f"[anatomy] {label:<10}: {line}")
        print(f"[anatomy]   writer layers: {layer_hist(keys)}")

    # recruitment-wiring focus: where do the NEW writers' new edges READ?
    rec_new = [k for k in new
               if writer_status(k, stable, recruited, dropped) == "recruited-writer"]
    if rec_new:
        readers = [f"{k[3]}:{k[4]}" for k in rec_new]
        from collections import Counter

        top_readers = Counter(readers).most_common(6)
        print("[anatomy] recruited-writer NEW edges read into: "
              + "  ".join(f"{r}({c})" for r, c in top_readers))

    # rotation on kept edges
    if kept:
        ra = ea.loc[list(kept), "abs_score"].rank()
        rb = eb.loc[list(kept), "abs_score"].rank()
        rho = ra.corr(rb)
        print(f"[anatomy] kept-edge score Spearman {rho:.3f} (rotation {1 - rho:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
