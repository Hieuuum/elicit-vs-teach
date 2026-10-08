"""Edge-vs-node change rates between two EAP-IG circuits, with random-set nulls (exploratory, CPU only).

Question (HANDOFF-kl-symbol-nodeedge.md, reading #4): does fine-tuning change edges
more than nodes, beyond what edges outnumbering nodes produces automatically?

For a pair (A, B) and a size k, A and B are the top-k edge sets of the two score
vectors. Change rates (all "fraction of A not in B", so 0 = identical, 1 = disjoint):
  edge rate = |A \\ B| / |A|
  node rate = same on the nodes the edges touch (upstream + receiver node)
  head rate = same on attention-head nodes only
gap = edge rate - node rate.

Nulls (`--n-null` draws each, deterministic in `--seed`, independent of `--workers`):
  N1  independent random k-sets for A and B: the chance level of every rate.
  N2  overlap-matched random pairs: |A'| = |B'| = k, |A' & B'| = the observed
      overlap, edges uniform over the whole graph. The edge rate equals the
      observed one by construction, so N2's node-rate / gap distribution answers
      "do nodes change less than that many random edge changes would imply?".
      Verdict key: one-sided p = P(null gap >= observed gap).
  N3  same, but the pool is the observed union A | B (A', B' re-partition the
      same edges). N2 spreads edges over every layer, which touches more nodes
      than a real, layer-concentrated circuit, so N2 can overstate how many
      nodes random edges touch; N3 keeps the observed node universe and layer
      mix and only re-draws which union edge goes to A, B or both. It is
      cheap, so it is kept.

Rankings: `signed` = top-k by signed score (the circuit definition, as
`topk_edges`); `abs` = top-k by |score| (parent edge signs were at chance).
Pairs: routes, references, plus a per-model split-half (mean_a vs mean_b) as
the ceiling reference, added automatically for every label.

Usage:
  python3 edge_node.py --scores LABEL=DIR ... --pairs A:B ... --out OUTDIR \\
      [--fracs 0.001 ...] [--topn 100 ...] [--n-null 1000] [--seed 0] [--workers 1]
With no `--scores`: the four tags under results_large/ and the four default pairs.
Writes OUTDIR/edge_node.json and OUTDIR/figures/edge_node.png.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TAGS = ("elicit_parent", "elicit_child", "fmt_parent", "teach_child")
DEFAULT_PAIRS = (
    ("elicit_parent", "elicit_child"),
    ("fmt_parent", "teach_child"),
    ("elicit_parent", "fmt_parent"),
    ("elicit_child", "teach_child"),
)
PAIR_KIND = ("route", "route", "reference", "reference")
DEFAULT_FRACS = (0.001, 0.002, 0.005, 0.01)
DEFAULT_TOPN = (100, 500, 1000)
RANKINGS = ("signed", "abs")
NULLS = ("n1", "n2", "n3")
SPLIT_SUFFIX = ":halves"
ALPHA = 0.05


# ------------------------------------------------------------------- inputs


def load_scores(dirs: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    """mean / mean_a / mean_b per label (per_example is memory-mapped, never read)."""
    out = {}
    for label, d in dirs.items():
        try:
            s = torch.load(Path(d) / "scores.pt", map_location="cpu", mmap=True)
        except (RuntimeError, ValueError):  # not a zipfile checkpoint: no mmap
            s = torch.load(Path(d) / "scores.pt", map_location="cpu")
        out[label] = {k: s[k].double().numpy().copy() for k in ("mean", "mean_a", "mean_b")}
        del s
    return out


def graph_index() -> dict:
    """Integer node ids per edge (upstream, receiver) and the head-node mask."""
    from compare import build_graph_ctx
    from geode.circuits import edge_tests as et

    ctx = build_graph_ctx()
    names, inv = np.unique(
        np.asarray(ctx["upstream"] + ctx["receiver"], dtype=object), return_inverse=True
    )
    n = len(ctx["upstream"])
    return {
        "up": inv[:n].astype(np.int64),
        "rc": inv[n:].astype(np.int64),
        "n_nodes": len(names),
        "head": np.isin(names, list(et.heads_only(names.tolist()))),
        "layers": np.asarray(ctx["layers"]),
    }


# --------------------------------------------------------------------- core


def rank_order(scores: np.ndarray, ranking: str) -> np.ndarray:
    """All edges best-first; ties by ascending index (same order as `topk_edges` for `signed`)."""
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}")
    key = np.abs(scores) if ranking == "abs" else scores
    return np.lexsort((np.arange(key.size), -key))


def node_mask(idx: np.ndarray, up: np.ndarray, rc: np.ndarray, n_nodes: int) -> np.ndarray:
    m = np.zeros(n_nodes, dtype=bool)
    m[up[idx]] = True
    m[rc[idx]] = True
    return m


def _frac_gone(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the True entries of mask a that are False in mask b (nan if a is empty)."""
    na = int(a.sum())
    return float((a & ~b).sum() / na) if na else float("nan")


def change_rates(a_idx, b_idx, up, rc, n_nodes, head) -> dict[str, float]:
    """Edge / node / head change rates of edge set A relative to B, plus the gap."""
    a_idx, b_idx = np.asarray(a_idx), np.asarray(b_idx)
    if a_idx.size == 0:
        raise ValueError("A must be nonempty")
    ov = int(np.intersect1d(a_idx, b_idx).size)
    ma, mb = node_mask(a_idx, up, rc, n_nodes), node_mask(b_idx, up, rc, n_nodes)
    edge = 1.0 - ov / a_idx.size
    node = _frac_gone(ma, mb)
    return {
        "overlap": ov,
        "edge_rate": edge,
        "node_rate": node,
        "head_rate": _frac_gone(ma & head, mb & head),
        "gap": edge - node,
        "n_nodes_a": int(ma.sum()),
        "n_nodes_b": int(mb.sum()),
        "n_heads_a": int((ma & head).sum()),
    }


def null_draws(kind, k, overlap, n_edges, up, rc, n_nodes, head, n_draws, seed_seq, pool=None):
    """Arrays (edge, node, head, gap) of length n_draws for null `kind` in N1/N2/N3.

    N2/N3 draw 2k - overlap distinct edges (from the whole graph / from `pool`)
    in random order: the first k are A', the last k are B', so |A' & B'| =
    overlap exactly. N1 draws A' and B' independently.
    """
    rng = np.random.default_rng(seed_seq)
    out = np.empty((4, n_draws))
    for i in range(n_draws):
        if kind == "n1":
            a, b = rng.choice(n_edges, k, replace=False), rng.choice(n_edges, k, replace=False)
        else:
            m = 2 * k - overlap
            s = rng.choice(n_edges, m, replace=False) if kind == "n2" else rng.permutation(pool)
            a, b = s[:k], s[k - overlap :]
        r = change_rates(a, b, up, rc, n_nodes, head)
        out[:, i] = (r["edge_rate"], r["node_rate"], r["head_rate"], r["gap"])
    return out


def _band(x: np.ndarray) -> dict[str, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return {"p5": float(p5), "p50": float(p50), "p95": float(p95)}


def _p_ge(null: np.ndarray, obs: float) -> float:
    """One-sided empirical p = (1 + #{null >= obs}) / (1 + n); nan if obs is nan."""
    null = null[~np.isnan(null)]
    if np.isnan(obs) or null.size == 0:
        return float("nan")
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + null.size))


def summarize_null(draws: np.ndarray, obs: dict) -> dict:
    edge, node, head, gap = draws
    return {
        "edge": _band(edge),
        "node": _band(node),
        "head": _band(head),
        "gap": _band(gap),
        "p_gap": _p_ge(gap, obs["gap"]),
        "p_head_gap": _p_ge(edge - head, obs["edge_rate"] - obs["head_rate"]),
        "n_draws": int(draws.shape[1]),
    }


# --------------------------------------------------------------- the driver

_G: dict = {}


def _task(args):
    kind, k, ov, pool, n_draws, ss = args
    g = _G
    return null_draws(
        kind, k, ov, g["n_edges"], g["up"], g["rc"], g["n_nodes"], g["head"], n_draws, ss, pool
    )


def sizes_of(fracs, topn, n_edges):
    from geode.circuits import edge_tests as et

    out = [(f"frac_{f:g}", et.size_to_k(f, n_edges)) for f in fracs]
    out += [(f"top_{n}", int(n)) for n in topn]
    for _, k in out:
        if not 1 <= k <= n_edges:
            raise ValueError(f"k={k} outside [1, {n_edges}]")
    return out


def pair_vectors(scores, spec):
    """(name, kind, vec_a, vec_b) for 'A:B' or '<label>:halves'."""
    if spec.endswith(SPLIT_SUFFIX):
        lab = spec[: -len(SPLIT_SUFFIX)]
        return spec, "split_half", scores[lab]["mean_a"], scores[lab]["mean_b"]
    a, b = spec.split(":")
    return spec, None, scores[a]["mean"], scores[b]["mean"]


def run(scores, pairs, graph, fracs, topn, n_null, seed, workers):
    """Full analysis -> JSON-able dict. `graph` = dict(up, rc, n_nodes, head)."""
    n_edges = int(graph["up"].size)
    sizes = sizes_of(fracs, topn, n_edges)
    specs = [f"{a}:{b}" for a, b in pairs] + [f"{lab}{SPLIT_SUFFIX}" for lab in scores]
    kind_of = dict(zip(DEFAULT_PAIRS, PAIR_KIND, strict=True))
    kinds = {f"{a}:{b}": kind_of.get((a, b), "pair") for a, b in pairs}
    _G.update(
        n_edges=n_edges,
        up=graph["up"],
        rc=graph["rc"],
        n_nodes=graph["n_nodes"],
        head=graph["head"],
    )
    up, rc, nn, head = graph["up"], graph["rc"], graph["n_nodes"], graph["head"]

    obs, jobs, keys = {}, [], []
    for pi, spec in enumerate(specs):
        _, kind, va, vb = pair_vectors(scores, spec)
        for ri, rk in enumerate(RANKINGS):
            oa, ob = rank_order(va, rk), rank_order(vb, rk)
            for si, (sname, k) in enumerate(sizes):
                a, b = oa[:k], ob[:k]
                r = change_rates(a, b, up, rc, nn, head)
                obs[(spec, rk, sname)] = r
                pool = np.union1d(a, b)
                for ni, kind_n in enumerate(("n2", "n3")):
                    ss = np.random.SeedSequence([seed, pi, ri, si, 2 + ni])
                    jobs.append((kind_n, k, r["overlap"], pool, n_null, ss))
                    keys.append((spec, rk, sname, kind_n))
    for si, (sname, k) in enumerate(sizes):  # N1 depends on k only
        jobs.append(("n1", k, 0, None, n_null, np.random.SeedSequence([seed, 0, 0, si, 1])))
        keys.append((None, None, sname, "n1"))

    if workers > 1:
        with get_context("fork").Pool(workers) as p:
            results = p.map(_task, jobs, chunksize=1)
    else:
        results = [_task(j) for j in jobs]
    drawn = dict(zip(keys, results, strict=True))

    pairs_out = {}
    for spec in specs:
        entry = {
            "kind": kinds.get(spec, "split_half" if spec.endswith(SPLIT_SUFFIX) else "pair"),
            "rankings": {},
        }
        for rk in RANKINGS:
            per_size = {}
            for sname, k in sizes:
                r = dict(obs[(spec, rk, sname)])
                r["k"] = k
                r["null"] = {
                    "n1": summarize_null(drawn[(None, None, sname, "n1")], r),
                    "n2": summarize_null(drawn[(spec, rk, sname, "n2")], r),
                    "n3": summarize_null(drawn[(spec, rk, sname, "n3")], r),
                }
                r["edges_more_than_nodes_n2"] = bool(r["null"]["n2"]["p_gap"] < ALPHA)
                r["edges_more_than_nodes_n3"] = bool(r["null"]["n3"]["p_gap"] < ALPHA)
                per_size[sname] = r
            entry["rankings"][rk] = per_size
        pairs_out[spec] = entry
    return {
        "meta": {
            "n_edges": n_edges,
            "n_nodes": int(nn),
            "n_head_nodes": int(head.sum()),
            "sizes": {s: k for s, k in sizes},
            "n_null": n_null,
            "seed": seed,
            "rankings": list(RANKINGS),
            "alpha": ALPHA,
        },
        "pairs": pairs_out,
    }


# ------------------------------------------------------------------ figure


def plot(result: dict, path: Path, ranking: str = "signed") -> None:
    pairs = result["pairs"]
    names = list(pairs)
    ncol = 4
    nrow = -(-len(names) // ncol)
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), squeeze=False, sharey=True
    )
    for ax in axs.ravel():
        ax.axis("off")
    for ax, name in zip(axs.ravel(), names, strict=False):
        ax.axis("on")
        rows = pairs[name]["rankings"][ranking]
        x = np.arange(len(rows))
        e = [r["edge_rate"] for r in rows.values()]
        n = [r["node_rate"] for r in rows.values()]
        lo = [r["null"]["n2"]["node"]["p5"] for r in rows.values()]
        hi = [r["null"]["n2"]["node"]["p95"] for r in rows.values()]
        ax.fill_between(x, lo, hi, color="0.8", label="N2 node p5-p95")
        ax.plot(x, e, "o-", color="C3", label="edge")
        ax.plot(x, n, "s-", color="C0", label="node")
        ax.set_xticks(
            x, [s.replace("frac_", "").replace("top_", "n") for s in rows], rotation=60, fontsize=7
        )
        ax.set_title(name, fontsize=8)
        ax.set_ylim(0, 1.02)
    axs[0, 0].legend(fontsize=6)
    axs[0, 0].set_ylabel(f"change rate ({ranking})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------- CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores", nargs="*", default=None, metavar="LABEL=DIR")
    p.add_argument("--pairs", nargs="*", default=None, metavar="A:B")
    p.add_argument("--out", type=Path, default=HERE / "results_large" / "edge_node")
    p.add_argument("--fracs", nargs="+", type=float, default=list(DEFAULT_FRACS))
    p.add_argument("--topn", nargs="+", type=int, default=list(DEFAULT_TOPN))
    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    if a.scores is None:
        dirs = {t: HERE / "results_large" / t for t in TAGS}
    else:
        dirs = {}
        for s in a.scores:
            if "=" not in s:
                p.error(f"--scores entry {s!r} is not LABEL=DIR")
            lab, d = s.split("=", 1)
            dirs[lab] = Path(d)
    if a.pairs:
        pairs = []
        for s in a.pairs:
            if s.count(":") != 1:
                p.error(f"--pairs entry {s!r} is not A:B")
            x, y = s.split(":")
            for lab in (x, y):
                if lab not in dirs:
                    p.error(f"pair label {lab!r} has no --scores entry")
            pairs.append((x, y))
    elif set(dirs) == set(TAGS):
        pairs = list(DEFAULT_PAIRS)
    else:
        p.error("--pairs is required unless --scores labels are exactly the four tags")
    a.dirs, a.pair_list = dirs, pairs
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    scores = load_scores(a.dirs)
    result = run(scores, a.pair_list, graph_index(), a.fracs, a.topn, a.n_null, a.seed, a.workers)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "edge_node.json").write_text(json.dumps(result, indent=2, default=float))
    plot(result, a.out / "figures" / "edge_node.png")
    print(f"wrote {a.out / 'edge_node.json'}")


if __name__ == "__main__":
    main()
