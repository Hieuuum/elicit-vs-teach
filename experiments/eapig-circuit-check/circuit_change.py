"""How the 10% EAP-IG circuit changes from parent to child (exploratory, CPU only).

Runbook: `HANDOFF-circuit-change.md`. Reads the four `scores.pt` (fields `mean`,
`mean_a`, `mean_b`; `per_example` is dropped) under `results_large/<tag>/` and
writes `results_large/circuit_change.json` plus PNGs in `results_large/figures/`.
No model is run and no frozen test is touched.

Labels: circuit = top 10% of edges by signed mean score (k = 19,586, same
`topk_edges` call as run.py); kept / added / dropped = in both / only the second
model's / only the first model's; score mass = sum of positive mean scores over
a set; ceiling = Jaccard of the top-k of `mean_a` vs `mean_b` within one model.
Pairs are (a, b) = (parent, child) for the routes and (elicit, other) for the two
references. Score mass is never compared across models in absolute terms (the
parents' m(full) is 0.21 and 0.03), only as shares of each model's own mass.

Usage: `python3 circuit_change.py [--results-dir DIR] [--figures-dir DIR] [--frac F]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from compare import COLOR, INK, MODEL_COLOR, MODEL_LABEL, MODEL_STYLE, build_graph_ctx, dump_json
from geode.circuits import edge_tests as et

HERE = Path(__file__).resolve().parent
TAGS = ("elicit_parent", "elicit_child", "fmt_parent", "teach_child")
PAIRS = {
    "elicit": ("elicit_parent", "elicit_child"),
    "teach": ("fmt_parent", "teach_child"),
    "children": ("elicit_child", "teach_child"),
    "parents": ("elicit_parent", "fmt_parent"),
}
# compare.json key holding each pair's Jaccard (None: not computed there).
COMPARE_KEY = {
    "elicit": ("routes", "elicit"),
    "teach": ("routes", "teach"),
    "children": ("children_reference",),
    "parents": None,
}
SENDER_TYPES = ("embed", "head", "mlp")
RECEIVER_TYPES = ("q", "k", "v", "mlp", "logits")
N_TOP_LIST = 50
N_TOP_SHIFT = 1000
SIGN_TOP_N = (100, 1000)


# ------------------------------------------------------------------- inputs


def load_scores(results_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for tag in TAGS:
        s = torch.load(results_dir / tag / "scores.pt", map_location="cpu")
        out[tag] = {key: s[key].double().numpy() for key in ("mean", "mean_a", "mean_b")}
        del s
    return out


def edge_meta(ctx: dict) -> dict[str, np.ndarray]:
    """Per-edge sender/receiver type and sending/receiving layer.

    Sending layer row 0 = embed, row l+1 = a head or MLP of layer l. Receiving
    layer column l = layer l's q/k/v/MLP input, column 16 = logits.
    """
    graph = ctx["graph"]
    names = graph.edge_names()
    send_type, recv_type, send_layer = [], [], []
    for name in names:
        up, rc = name.split("->")
        if up == "embed":
            send_type.append("embed")
            send_layer.append(0)
        else:
            send_type.append("head" if ".h" in up else "mlp")
            send_layer.append(int(up[1:].split(".")[0]) + 1)
        recv_type.append(
            "logits" if rc == "logits" else "mlp" if rc.endswith(".in") else rc.split(".")[1][0]
        )
    return {
        "names": np.array(names),
        "send_type": np.array(send_type),
        "recv_type": np.array(recv_type),
        "send_layer": np.array(send_layer),
        "recv_layer": ctx["layers"],
    }


def full_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every edge under topk_edges' order (signed desc, index asc)."""
    order = np.lexsort((np.arange(scores.size), -scores))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def pos_mass(scores: np.ndarray, idx: np.ndarray | None = None) -> float:
    v = scores if idx is None else scores[idx]
    return float(np.clip(v, 0, None).sum())


# ---------------------------------------------------------------- analyses


def sign_agreement(a: np.ndarray, b: np.ndarray, n: int) -> float:
    """Pick the top-n edges by |score| in one half, read their sign in the other.

    Selecting in one half and testing in the other keeps the null at 0.5 when
    the signs are noise. Averaged over both directions.
    """
    agree = []
    for x, y in ((a, b), (b, a)):
        idx = np.argsort(-np.abs(x), kind="stable")[:n]
        agree.append(float((np.sign(x[idx]) == np.sign(y[idx])).mean()))
    return float(np.mean(agree))


def reliability(sc: dict[str, np.ndarray], k: int) -> dict:
    """Split-half agreement within one model (halves are 256 problems each)."""
    a, b = sc["mean_a"], sc["mean_b"]
    rho = float(spearmanr(a, b).statistic)
    return {
        "ceiling_jaccard": et.split_half_ceiling(a, b, k),
        "spearman_halves": rho,
        # Spearman-Brown: expected agreement of two full 512-problem score sets.
        "spearman_full_projected": 2 * rho / (1 + rho),
        "spearman_abs_halves": float(spearmanr(np.abs(a), np.abs(b)).statistic),
        "sign_agreement_halves": {str(n): sign_agreement(a, b, n) for n in (*SIGN_TOP_N, k)},
    }


def breakdown(sets: dict[str, np.ndarray], key: np.ndarray, levels, mass: dict) -> dict:
    """Counts and own-mass shares of kept/added/dropped by one categorical key."""
    out = {}
    for lvl in levels:
        row = {}
        for name, idx in sets.items():
            sel = idx[key[idx] == lvl]
            row[f"{name}_n"] = int(sel.size)
            for owner, (scores, total) in mass[name].items():
                row[f"{name}_mass_share_{owner}"] = pos_mass(scores, sel) / total
        row["graph_n"] = int((key == lvl).sum())
        out[lvl] = row
    return out


def layer_grid(idx: np.ndarray, meta: dict, shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=np.int64)
    np.add.at(grid, (meta["send_layer"][idx], meta["recv_layer"][idx]), 1)
    return grid


def edge_rows(
    idx: np.ndarray, own: np.ndarray, own_rank: np.ndarray, other_rank: np.ndarray, meta: dict
) -> list[dict]:
    order = idx[np.lexsort((idx, -own[idx]))][:N_TOP_LIST]
    return [
        {
            "edge": str(meta["names"][e]),
            "score": float(own[e]),
            "rank": int(own_rank[e]),
            "rank_in_other": int(other_rank[e]),
        }
        for e in order
    ]


def compare_pair(
    a: str,
    b: str,
    scores: dict,
    ranks: dict,
    circuits: dict,
    meta: dict,
    k: int,
    grid_shape: tuple[int, int],
) -> dict:
    sa, sb = scores[a]["mean"], scores[b]["mean"]
    ca, cb = circuits[a], circuits[b]
    kept = np.intersect1d(ca, cb)
    added = np.setdiff1d(cb, ca)
    dropped = np.setdiff1d(ca, cb)
    mass_a, mass_b = pos_mass(sa, ca), pos_mass(sb, cb)
    sets = {"kept": kept, "added": added, "dropped": dropped}
    # Each set's mass is read in the model(s) whose circuit contains it.
    mass = {
        "kept": {"a": (sa, mass_a), "b": (sb, mass_b)},
        "added": {"b": (sb, mass_b)},
        "dropped": {"a": (sa, mass_a)},
    }

    top_b = np.lexsort((np.arange(sb.size), -sb))[:N_TOP_SHIFT]
    shift = ranks[a][top_b] - ranks[b][top_b]
    a_rank_of_top_b = ranks[a][top_b]
    abs_top_a = np.argsort(-np.abs(sa), kind="stable")[:k]
    abs_top_b = np.argsort(-np.abs(sb), kind="stable")[:k]
    in_abs_top_a = np.zeros(sa.size, dtype=bool)
    in_abs_top_a[abs_top_a] = True

    g_add, g_drop, g_kept = (layer_grid(x, meta, grid_shape) for x in (added, dropped, kept))
    return {
        "a": a,
        "b": b,
        "how_much": {
            "edge_jaccard": et.jaccard(ca.tolist(), cb.tolist()),
            "chance_jaccard": et.chance_jaccard(k, k, sa.size),
            "spearman_full_vectors": float(spearmanr(sa, sb).statistic),
            # Same as above on |score|: which edges matter, ignoring direction.
            "spearman_abs_vectors": float(spearmanr(np.abs(sa), np.abs(sb)).statistic),
            "abs_topk_jaccard": et.jaccard(abs_top_a.tolist(), abs_top_b.tolist()),
            "n_kept": int(kept.size),
            "n_added": int(added.size),
            "n_dropped": int(dropped.size),
            "b_mass_share_on_kept": pos_mass(sb, kept) / mass_b,
            "a_mass_share_on_kept": pos_mass(sa, kept) / mass_a,
        },
        "where": {
            "by_sender_type": breakdown(sets, meta["send_type"], SENDER_TYPES, mass),
            "by_receiver_type": breakdown(sets, meta["recv_type"], RECEIVER_TYPES, mass),
            "layer_grid_rows": "send layer: 0 = embed, l+1 = layer l head/MLP",
            "layer_grid_cols": "receive layer: l = layer l q/k/v/MLP in, 16 = logits",
            "layer_grid_kept": g_kept.tolist(),
            "layer_grid_added": g_add.tolist(),
            "layer_grid_dropped": g_drop.tolist(),
        },
        "what": {
            "top_added_by_b_score": edge_rows(added, sb, ranks[b], ranks[a], meta),
            "top_dropped_by_a_score": edge_rows(dropped, sa, ranks[a], ranks[b], meta),
            "b_top1000_rank_shift": {
                "definition": "rank in a minus rank in b, for b's top 1000 edges (1-based)",
                "quantiles": {
                    str(q): float(np.quantile(shift, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                },
                "share_in_a_top1000": float((a_rank_of_top_b <= N_TOP_SHIFT).mean()),
                "share_in_a_circuit": float((a_rank_of_top_b <= k).mean()),
                # a's most negative 10%: the edge matters in a but with the other sign.
                "share_in_a_bottom_k": float((a_rank_of_top_b > sa.size - k).mean()),
                "share_in_a_abs_topk": float(in_abs_top_a[top_b].mean()),
                "a_rank_quantiles": {
                    str(q): float(np.quantile(a_rank_of_top_b, q))
                    for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                },
            },
        },
        "_a_rank_of_top_b": a_rank_of_top_b,  # for the plot; stripped before dump
    }


def shape(scores: np.ndarray, k: int) -> dict:
    """Concentration of positive score mass in one model."""
    pos = np.sort(np.clip(scores, 0, None))[::-1]
    total = pos.sum()
    cum = np.cumsum(pos) / total
    n50 = int(np.searchsorted(cum, 0.5) + 1)
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    return {
        "total_pos_mass": float(total),
        "total_neg_mass": float(np.clip(scores, None, 0).sum()),
        "n_positive_edges": int((scores > 0).sum()),
        "n_zero_edges": int((scores == 0).sum()),
        "n_edges_50pct_mass": n50,
        "n_edges_90pct_mass": n90,
        "frac_edges_50pct_mass": n50 / scores.size,
        "frac_edges_90pct_mass": n90 / scores.size,
        "circuit_share_of_pos_mass": float(cum[k - 1]),
    }


# -------------------------------------------------------------------- plots


def _strip(ax) -> None:
    ax.grid(True, color=INK["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_layer_heatmaps(pairs: dict, graph_grid: np.ndarray, path: Path) -> None:
    """Added minus dropped edges per (send layer, receive layer), one panel per route."""
    diffs = {
        n: np.array(pairs[n]["where"]["layer_grid_added"])
        - np.array(pairs[n]["where"]["layer_grid_dropped"])
        for n in ("elicit", "teach")
    }
    vmax = max(np.abs(d).max() for d in diffs.values())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div_red_blue", ["#7a1f1f", COLOR["red"], "#f0efec", COLOR["blue"], "#0d366b"]
    ).with_extremes(bad="#ffffff")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, name in zip(axes, ("elicit", "teach")):
        d = np.ma.masked_where(graph_grid == 0, diffs[name]).astype(float)
        im = ax.imshow(d, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
        a, b = PAIRS[name]
        ax.set_title(f"{name}: {MODEL_LABEL[a]} → {MODEL_LABEL[b]}", fontsize=9)
        ax.set_xlabel("receiving layer (16 = logits)")
        ax.set_xticks(range(0, 17, 2))
        ax.set_yticks(range(0, 17, 2))
        ax.set_yticklabels(["emb"] + [str(i) for i in range(1, 16, 2)])
    axes[0].set_ylabel("sending layer (emb, then layer of head/MLP)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="added − dropped edges (blue = gained)")
    fig.suptitle(
        "Where the top-10% circuit gained and lost edges, parent → child "
        "(blank = no edges in graph; single training seed)",
        fontsize=10,
    )
    fig.savefig(path, dpi=150, facecolor=INK["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_rank_shift(pairs: dict, k: int, path: Path) -> None:
    """Where the second model's top-1000 edges sat in the first model's ranking."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bins = np.logspace(0, np.log10(195_865), 40)
    style = {
        "elicit": (COLOR["blue"], "-"),
        "teach": (COLOR["orange"], "-"),
        "children": (INK["secondary"], "--"),
        "parents": (INK["muted"], ":"),
    }
    for name, pr in pairs.items():
        color, ls = style[name]
        a, b = PAIRS[name]
        ax.hist(
            pr["_a_rank_of_top_b"],
            bins=bins,
            histtype="step",
            color=color,
            linestyle=ls,
            linewidth=2,
            label=f"{name}: {b}'s top 1000, ranked in {a}",
        )
    ax.axvline(k, color=INK["baseline"], linewidth=1)
    ax.text(
        k,
        ax.get_ylim()[1] * 0.95,
        " circuit edge (10%)",
        color=INK["secondary"],
        fontsize=8,
        va="top",
    )
    ax.set_xscale("log")
    ax.set_xlabel("rank in the first model (1 = top)")
    ax.set_ylabel("edges")
    ax.set_title(
        "Rank in the parent of each child's top-1000 edges\n"
        "(references dashed/dotted; single training seed)",
        fontsize=10,
    )
    _strip(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def plot_concentration(scores: dict, k: int, path: Path) -> None:
    """Cumulative share of positive score mass held by the top-n edges, per model."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    n = np.arange(1, 195_866)
    for tag in TAGS:
        pos = np.sort(np.clip(scores[tag]["mean"], 0, None))[::-1]
        ax.plot(
            n,
            np.cumsum(pos) / pos.sum(),
            color=MODEL_COLOR[tag],
            linestyle=MODEL_STYLE[tag],
            linewidth=2,
            label=MODEL_LABEL[tag],
        )
    ax.axvline(k, color=INK["baseline"], linewidth=1)
    for y in (0.5, 0.9):
        ax.axhline(y, color=INK["baseline"], linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("top-n edges by mean score")
    ax.set_ylabel("share of the model's positive score mass")
    ax.set_title(
        "How concentrated each model's positive score mass is\n"
        "(vertical line = 10% circuit; single training seed)",
        fontsize=10,
    )
    _strip(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


# --------------------------------------------------------------------- main


def check_against_compare(pairs: dict, results_dir: Path, frac: float) -> dict:
    """Correctness check: edge Jaccard must reproduce compare.json from the box."""
    cmp = json.loads((results_dir / "compare.json").read_text())
    out = {}
    for name, keypath in COMPARE_KEY.items():
        if keypath is None:
            continue
        node = cmp
        for key in keypath:
            node = node[key]
        ref = node["sizes"][str(frac)]["edge_jaccard"]
        ours = pairs[name]["how_much"]["edge_jaccard"]
        assert abs(ref - ours) < 1e-12, (name, ref, ours)
        out[name] = {"compare_json": ref, "here": ours}
    return out


def run(results_dir: Path, figures_dir: Path, frac: float) -> dict:
    ctx = build_graph_ctx()
    meta = edge_meta(ctx)
    n_edges = ctx["graph"].n_edges
    k = et.size_to_k(frac, n_edges)
    scores = load_scores(results_dir)
    ranks = {t: full_ranks(scores[t]["mean"]) for t in TAGS}
    circuits = {t: np.sort(et.topk_edges(scores[t]["mean"], k)) for t in TAGS}
    for t in TAGS:  # rank order must agree with topk_edges
        assert set(np.flatnonzero(ranks[t] <= k)) == set(circuits[t].tolist()), t
    grid_shape = (ctx["graph"].n_layers + 1, ctx["graph"].n_layers + 1)
    graph_grid = layer_grid(np.arange(n_edges), meta, grid_shape)

    pairs = {
        name: compare_pair(a, b, scores, ranks, circuits, meta, k, grid_shape)
        for name, (a, b) in PAIRS.items()
    }
    result = {
        "exploratory": True,
        "frac": frac,
        "k": k,
        "n_edges": n_edges,
        "reliability": {t: reliability(scores[t], k) for t in TAGS},
        "shape": {t: shape(scores[t]["mean"], k) for t in TAGS},
        "graph_by_sender_type": {s: int((meta["send_type"] == s).sum()) for s in SENDER_TYPES},
        "graph_by_receiver_type": {r: int((meta["recv_type"] == r).sum()) for r in RECEIVER_TYPES},
        "graph_layer_grid": graph_grid.tolist(),
        "compare_json_check": check_against_compare(pairs, results_dir, frac),
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_layer_heatmaps(pairs, graph_grid, figures_dir / "cc_layer_added_minus_dropped.png")
    plot_rank_shift(pairs, k, figures_dir / "cc_rank_in_parent.png")
    plot_concentration(scores, k, figures_dir / "cc_mass_concentration.png")
    for pr in pairs.values():
        pr.pop("_a_rank_of_top_b")
    result["pairs"] = pairs
    dump_json(result, results_dir / "circuit_change.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results_large"))
    ap.add_argument("--figures-dir", default=str(HERE / "results_large" / "figures"))
    ap.add_argument("--frac", type=float, default=0.1)
    a = ap.parse_args()
    run(Path(a.results_dir), Path(a.figures_dir), a.frac)


if __name__ == "__main__":
    main()
