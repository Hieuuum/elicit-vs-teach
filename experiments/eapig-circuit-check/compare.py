"""Parent-vs-child EAP-IG circuit comparison (CPU only, PLAN.md "Parent vs child
comparison" + "Plots"). Reads `run.py`'s per-model outputs under `results/<tag>/`
(`sanity.json`, `scores.pt`, `evaluate.json`) for the four tags `elicit_parent`,
`elicit_child`, `fmt_parent`, `teach_child`, and writes `results/compare.json`,
`results/summary.md`, and five PNG figures. All statistics reuse
`geode.circuits.edge_tests`; nothing here re-derives EAP-IG scores or runs a model.

Routes: elicit = (elicit_parent -> elicit_child), teach = (fmt_parent -> teach_child).
A route whose parent failed `evaluate.json`'s validity gate is labelled
"no parent circuit"; its overlaps are still computed but flagged reference-only.

Usage: `python3 compare.py [--results-dir DIR] [--figures-dir DIR] [--sizes F ...]`.
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

from geode.circuits import edge_tests as et
from geode.circuits.eapig import EdgeGraph

HERE = Path(__file__).resolve().parent
SIZES = (0.001, 0.002, 0.005, 0.01)
TOPK_FIXED = (100, 500, 1000)
TAGS = ("elicit_parent", "elicit_child", "fmt_parent", "teach_child")
ROUTES = {"elicit": ("elicit_parent", "elicit_child"), "teach": ("fmt_parent", "teach_child")}
ACCURACY_GAP_THRESHOLD = 0.15
CEILING_UNRELIABLE = 0.5

# dataviz reference palette (skill `dataviz`, references/palette.md): fixed
# categorical hue order, never cycled or reassigned by rank.
COLOR = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "red": "#e34948",
}
INK = {
    "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "baseline": "#c3c2b7", "surface": "#fcfcfb",
}
MODEL_COLOR = {
    "elicit_parent": COLOR["blue"], "elicit_child": COLOR["blue"],
    "fmt_parent": COLOR["orange"], "teach_child": COLOR["orange"],
}
MODEL_STYLE = {
    "elicit_parent": "-", "elicit_child": "--", "fmt_parent": "-", "teach_child": "--",
}
MODEL_MARKER = {
    "elicit_parent": "o", "elicit_child": "o", "fmt_parent": "s", "teach_child": "s",
}
MODEL_LABEL = {
    "elicit_parent": "elicit parent", "elicit_child": "elicit child (full FT)",
    "fmt_parent": "format-installed parent", "teach_child": "teach child (full FT)",
}
ROUTE_COLOR = {"elicit": COLOR["blue"], "teach": COLOR["orange"]}


# --------------------------------------------------------------------------- io


def load_model(tag: str, results_dir: Path) -> dict | None:
    """Load one tag's `sanity.json`/`scores.pt`/`evaluate.json`; None if any is missing."""
    d = results_dir / tag
    paths = {"sanity": d / "sanity.json", "scores": d / "scores.pt", "evaluate": d / "evaluate.json"}
    if not all(p.exists() for p in paths.values()):
        return None
    scores = torch.load(paths["scores"], map_location="cpu")
    scores.pop("per_example", None)  # not used here; drop early to bound memory
    return {
        "sanity": json.loads(paths["sanity"].read_text()),
        "scores": scores,
        "evaluate": json.loads(paths["evaluate"].read_text()),
    }


def load_all(results_dir: Path) -> dict[str, dict | None]:
    return {tag: load_model(tag, results_dir) for tag in TAGS}


def build_graph_ctx() -> dict:
    """Model-free edge graph + the per-edge metadata every overlap stat needs."""
    graph = EdgeGraph(n_layers=16, n_heads=32, n_kv_heads=8)
    assert graph.n_edges == 195_865, graph.n_edges
    upstream = graph.edge_upstream_node()
    receiver = graph.edge_receiver_node()
    layers = graph.edge_receiving_layer().numpy()
    node_universe_all = set(graph.upstream_names) | set(graph.receiver_node)
    node_universe_heads = et.heads_only(node_universe_all)
    return {
        "graph": graph,
        "upstream": upstream,
        "receiver": receiver,
        "layers": layers,
        "n_all": len(node_universe_all),
        "n_heads": len(node_universe_heads),
    }


def _nearest_size(value: float | None) -> float:
    if value is None:
        return SIZES[-1]
    arr = np.array(SIZES)
    return float(arr[int(np.argmin(np.abs(arr - value)))])


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


# ------------------------------------------------------------------- per-model


def model_summary(entry: dict, ctx: dict) -> dict:
    """Sanity/f-curve/gate/tests/ceiling summary for one model (no overlaps)."""
    graph = ctx["graph"]
    E = graph.n_edges
    scores, ev, sanity = entry["scores"], entry["evaluate"], entry["sanity"]
    ceiling: dict[str, float] = {}
    for frac in SIZES:
        k = et.size_to_k(frac, E)
        ceiling[str(frac)] = et.split_half_ceiling(
            scores["mean_a"].numpy(), scores["mean_b"].numpy(), k
        )
    unreliable = {key: c < CEILING_UNRELIABLE for key, c in ceiling.items()}
    sizes = ev.get("sizes", {})
    return {
        "sanity": {
            "exact_match": sanity.get("exact_match"),
            "n_val": sanity.get("n_val"),
            "m_full": sanity.get("m_full"),
            "m_empty": sanity.get("m_empty"),
            "nonstandard_split_share": sanity.get("nonstandard_split_share"),
        },
        "f_curve": {str(f): sizes.get(str(f), {}).get("f") for f in SIZES},
        "sufficiency_pass": {
            str(f): sizes.get(str(f), {}).get("sufficiency", {}).get("pass") for f in SIZES
        },
        "validity_gate": ev.get("validity_gate", {}).get("pass"),
        "tests": ev.get("tests"),
        "selected_size": ev.get("selected_size"),
        "ceiling": ceiling,
        "ceiling_unreliable": unreliable,
    }


# --------------------------------------------------------------------- overlap


def overlap_at_size(parent_mean, child_mean, frac: float, ctx: dict) -> dict:
    """All parent-vs-child overlap statistics at one circuit size."""
    graph = ctx["graph"]
    E = graph.n_edges
    k = et.size_to_k(frac, E)
    p_edges = et.topk_edges(np.asarray(parent_mean, dtype=np.float64), k)
    c_edges = et.topk_edges(np.asarray(child_mean, dtype=np.float64), k)
    p_set, c_set = set(p_edges.tolist()), set(c_edges.tolist())
    p_nodes = et.nodes_touched(p_set, ctx["upstream"], ctx["receiver"])
    c_nodes = et.nodes_touched(c_set, ctx["upstream"], ctx["receiver"])
    p_heads, c_heads = et.heads_only(p_nodes), et.heads_only(c_nodes)
    added_e, removed_e, rate_e = et.change_rate(p_set, c_set)
    added_n, removed_n, rate_n = et.change_rate(p_nodes, c_nodes)
    added_edges, removed_edges = c_set - p_set, p_set - c_set
    hist_added = et.layer_histogram(added_edges, ctx["layers"], graph.n_layers + 1)
    hist_removed = et.layer_histogram(removed_edges, ctx["layers"], graph.n_layers + 1)
    return {
        "k": k,
        "edge_jaccard": et.jaccard(p_set, c_set),
        "node_jaccard_all": et.jaccard(p_nodes, c_nodes),
        "node_jaccard_heads": et.jaccard(p_heads, c_heads),
        "edge_change": {"added": added_e, "removed": removed_e, "rate": rate_e},
        "node_change": {"added": added_n, "removed": removed_n, "rate": rate_n},
        "layer_hist_added": hist_added.tolist(),
        "layer_hist_removed": hist_removed.tolist(),
        "chance_edge_jaccard": et.chance_jaccard(k, k, E),
        "chance_node_jaccard_all": et.chance_jaccard(len(p_nodes), len(c_nodes), ctx["n_all"]),
        "chance_node_jaccard_heads": et.chance_jaccard(len(p_heads), len(c_heads), ctx["n_heads"]),
    }


def route_summary(
    name: str, parent_tag: str, child_tag: str, models: dict, model_summaries: dict, ctx: dict
) -> dict:
    parent_entry, child_entry = models.get(parent_tag), models.get(child_tag)
    missing = [t for t, e in [(parent_tag, parent_entry), (child_tag, child_entry)] if e is None]
    if missing:
        return {"available": False, "missing": missing, "parent_tag": parent_tag,
                "child_tag": child_tag}
    parent_gate = model_summaries[parent_tag]["validity_gate"]
    label = "ok" if parent_gate else ("no parent circuit" if parent_gate is False else "unknown")
    parent_mean = parent_entry["scores"]["mean"].numpy()
    child_mean = child_entry["scores"]["mean"].numpy()
    sizes_out = {}
    for frac in SIZES:
        ov = overlap_at_size(parent_mean, child_mean, frac, ctx)
        p_ceil = model_summaries[parent_tag]["ceiling"][str(frac)]
        c_ceil = model_summaries[child_tag]["ceiling"][str(frac)]
        ov["ceiling_parent"], ov["ceiling_child"] = p_ceil, c_ceil
        ov["overlap_frac_of_parent_ceiling"] = et.ceiling_fraction(ov["edge_jaccard"], p_ceil)
        ov["overlap_frac_of_child_ceiling"] = et.ceiling_fraction(ov["edge_jaccard"], c_ceil)
        sizes_out[str(frac)] = ov
    topk_overlap = {str(k): et.topk_overlap(parent_mean, child_mean, k) for k in TOPK_FIXED}
    return {
        "available": True,
        "label": label,
        "reference_only": label != "ok",
        "parent_tag": parent_tag,
        "child_tag": child_tag,
        "sizes": sizes_out,
        "topk_overlap": topk_overlap,
        "parent_in_child": child_entry["evaluate"].get("parent_in_child"),
        "hist_size": _nearest_size(model_summaries[child_tag]["selected_size"]),
    }


def children_reference(models: dict, model_summaries: dict, ctx: dict) -> dict:
    """Elicited-child vs taught-child overlap, for reference only (not a route)."""
    a, b = models.get("elicit_child"), models.get("teach_child")
    missing = [t for t, e in [("elicit_child", a), ("teach_child", b)] if e is None]
    if missing:
        return {"available": False, "missing": missing}
    a_mean, b_mean = a["scores"]["mean"].numpy(), b["scores"]["mean"].numpy()
    sizes_out = {}
    for frac in SIZES:
        ov = overlap_at_size(a_mean, b_mean, frac, ctx)
        a_ceil = model_summaries["elicit_child"]["ceiling"][str(frac)]
        b_ceil = model_summaries["teach_child"]["ceiling"][str(frac)]
        ov["ceiling_a"], ov["ceiling_b"] = a_ceil, b_ceil
        ov["overlap_frac_of_a_ceiling"] = et.ceiling_fraction(ov["edge_jaccard"], a_ceil)
        ov["overlap_frac_of_b_ceiling"] = et.ceiling_fraction(ov["edge_jaccard"], b_ceil)
        sizes_out[str(frac)] = ov
    topk_overlap = {str(k): et.topk_overlap(a_mean, b_mean, k) for k in TOPK_FIXED}
    return {
        "available": True, "reference_only": True, "a_tag": "elicit_child", "b_tag": "teach_child",
        "sizes": sizes_out, "topk_overlap": topk_overlap,
    }


def accuracy_gap(models: dict) -> dict:
    ec, tc = models.get("elicit_child"), models.get("teach_child")
    if ec is None or tc is None:
        return {"available": False}
    ec_em, tc_em = ec["sanity"]["exact_match"], tc["sanity"]["exact_match"]
    gap = ec_em - tc_em
    flagged = gap > ACCURACY_GAP_THRESHOLD
    note = None
    if flagged:
        note = (
            "teach_child exact-match is more than 0.15 below elicit_child's; a fair "
            "parent-vs-child f/sufficiency comparison would need f re-evaluated on the "
            "subset of problems both children answer correctly. per_example.npz stores "
            "suff_{frac} arrays (N,3) and 'full' per model, but m_empty is not stored "
            "per-example, so the matched-subset floor can't be reconstructed from saved "
            "artifacts. Flagging the gap only; no corrected numbers are reported here."
        )
    return {
        "available": True, "elicit_child_em": ec_em, "teach_child_em": tc_em,
        "gap": gap, "flag": flagged, "note": note,
    }


# -------------------------------------------------------------------- summary


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return "n/a" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _route_table(route: dict) -> list[str]:
    lines = [
        "| size | edge jacc | node jacc (all) | node jacc (heads) | edge chg rate | "
        "node chg rate | chance (edge) | parent ceil | child ceil | "
        "overlap/parent ceil | overlap/child ceil |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for frac in SIZES:
        ov = route["sizes"][str(frac)]
        lines.append(
            f"| {frac} | {_fmt(ov['edge_jaccard'])} | {_fmt(ov['node_jaccard_all'])} | "
            f"{_fmt(ov['node_jaccard_heads'])} | {_fmt(ov['edge_change']['rate'])} | "
            f"{_fmt(ov['node_change']['rate'])} | {_fmt(ov['chance_edge_jaccard'])} | "
            f"{_fmt(ov['ceiling_parent'])} | {_fmt(ov['ceiling_child'])} | "
            f"{_fmt(ov['overlap_frac_of_parent_ceiling'])} | "
            f"{_fmt(ov['overlap_frac_of_child_ceiling'])} |"
        )
    return lines


def write_summary_md(compare: dict, path: Path) -> None:
    lines = [
        "# EAP-IG parent-vs-child comparison — summary",
        "",
        "Single training seed per child; every elicit-vs-teach difference below is "
        "single-run.",
        "",
    ]
    if compare["missing_models"]:
        lines.append(f"**Missing models (no results dir found):** "
                      f"{', '.join(compare['missing_models'])}")
        lines.append("")
    lines.append("## Per-model")
    for tag in TAGS:
        ms = compare["models"].get(tag)
        lines.append(f"\n### {tag}")
        if ms is None:
            lines.append("not available (missing results).")
            continue
        s = ms["sanity"]
        lines.append(
            f"- sanity: EM {_fmt(s['exact_match'])}, n_val {_fmt(s['n_val'])}, "
            f"m_full {_fmt(s['m_full'])}, m_empty {_fmt(s['m_empty'])}, "
            f"nonstandard split {_fmt(s['nonstandard_split_share'])}"
        )
        lines.append(
            f"- validity gate: {_fmt(ms['validity_gate'])}; "
            f"selected size: {_fmt(ms['selected_size'])}"
        )
        lines.append("")
        lines.append("| size | f | f ceiling | sufficiency | tests passed |")
        lines.append("|---|---|---|---|---|")
        for frac in SIZES:
            key = str(frac)
            ceil, unreliable = ms["ceiling"].get(key), ms["ceiling_unreliable"].get(key)
            ceil_str = _fmt(ceil) + (" (unreliable)" if unreliable else "")
            tests = (ms["tests"] or {}).get(key)
            if tests is None:
                tests_str = "—"
            else:
                passed = sum(1 for v in tests.values() if v is True)
                tests_str = f"{passed}/{len(tests)}"
            lines.append(
                f"| {frac} | {_fmt(ms['f_curve'].get(key))} | {ceil_str} | "
                f"{_fmt(ms['sufficiency_pass'].get(key))} | {tests_str} |"
            )
    lines.append("\n## Per-route overlaps\n")
    for name, route in compare["routes"].items():
        lines.append(f"### {name}")
        if not route.get("available"):
            lines.append(f"not available (missing: {', '.join(route.get('missing', []))}).")
            continue
        flag = " (overlaps for reference only)" if route["reference_only"] else ""
        lines.append(
            f"parent `{route['parent_tag']}` -> child `{route['child_tag']}`; "
            f"label: **{route['label']}**{flag}"
        )
        lines.append("")
        lines.extend(_route_table(route))
        lines.append("")
        lines.append(
            "top-k overlap: " + ", ".join(f"{k}={v}" for k, v in route["topk_overlap"].items())
        )
        lines.append("")
    lines.append("### children vs children (reference)")
    cr = compare["children_reference"]
    if not cr.get("available"):
        lines.append(f"not available (missing: {', '.join(cr.get('missing', []))}).")
    else:
        lines.append("| size | edge jacc | node jacc (all) | node jacc (heads) | chance (edge) |")
        lines.append("|---|---|---|---|---|")
        for frac in SIZES:
            ov = cr["sizes"][str(frac)]
            lines.append(
                f"| {frac} | {_fmt(ov['edge_jaccard'])} | {_fmt(ov['node_jaccard_all'])} | "
                f"{_fmt(ov['node_jaccard_heads'])} | {_fmt(ov['chance_edge_jaccard'])} |"
            )
    lines.append("\n## Accuracy gap (teach vs elicit child)\n")
    gap = compare["accuracy_gap"]
    if not gap.get("available"):
        lines.append("not available (one or both children missing).")
    else:
        flag = " — FLAGGED" if gap["flag"] else ""
        lines.append(
            f"elicit_child EM {_fmt(gap['elicit_child_em'])}, "
            f"teach_child EM {_fmt(gap['teach_child_em'])}, gap {_fmt(gap['gap'])}{flag}"
        )
        if gap["note"]:
            lines.append("")
            lines.append(gap["note"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- plots


def _strip_axes(ax) -> None:
    ax.grid(True, color=INK["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_f_vs_size(models: dict, model_summaries: dict, path: Path) -> None:
    """Claim 1: EAP-IG circuits beat same-size random edge sets at every size."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.array(SIZES)
    any_data = False
    for tag in TAGS:
        ms = model_summaries.get(tag)
        if ms is None:
            continue
        f_vals = [ms["f_curve"].get(str(s)) for s in SIZES]
        if any(v is None for v in f_vals):
            continue
        any_data = True
        color = MODEL_COLOR[tag]
        ax.plot(x, f_vals, color=color, linestyle=MODEL_STYLE[tag], marker=MODEL_MARKER[tag],
                label=MODEL_LABEL[tag], linewidth=2)
        ev = models[tag]["evaluate"]
        lo, hi = [], []
        for s in SIZES:
            rnd = ev["sizes"][str(s)]["f_random"]
            p5, p95 = np.percentile(rnd, [5, 95])
            lo.append(p5)
            hi.append(p95)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("circuit size (fraction of edges)")
    ax.set_ylabel("faithfulness f")
    ax.set_title(
        "Claim: EAP-IG circuits recover more of the model's addition behavior\n"
        "than same-size random edge sets, at every tested size (single training seed)",
        fontsize=10,
    )
    ax.axhline(0, color=INK["baseline"], linewidth=1)
    ax.axhline(1, color=INK["baseline"], linewidth=1, linestyle=":")
    _strip_axes(ax)
    if any_data:
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "no data available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def plot_test_grid(model_summaries: dict, path: Path) -> None:
    """Claim 2: validation tests pass only up to the first fully-passing size."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    n_rows, n_cols = len(TAGS), len(SIZES)
    grid = np.full((n_rows, n_cols), np.nan)
    annot = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for i, tag in enumerate(TAGS):
        ms = model_summaries.get(tag)
        for j, frac in enumerate(SIZES):
            if ms is None:
                continue
            tests = (ms["tests"] or {}).get(str(frac))
            if tests is None:
                annot[i][j] = "—"
                continue
            passed, total = sum(1 for v in tests.values() if v is True), len(tests)
            grid[i, j] = passed / total if total else np.nan
            annot[i][j] = f"{passed}/{total}"
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"]
    ).with_extremes(bad="#f0efec")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            txt = annot[i][j]
            if not txt:
                continue
            val = grid[i, j]
            color = "white" if (not np.isnan(val) and val > 0.6) else INK["primary"]
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{s:g}" for s in SIZES])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([MODEL_LABEL[t] for t in TAGS], fontsize=9)
    ax.set_xlabel("circuit size (fraction of edges)")
    ax.set_title(
        "Claim: validation tests pass only up to the first fully-passing size\n"
        "(cells above the selected size are not evaluated; single training seed)",
        fontsize=10,
    )
    del im
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def plot_overlap_vs_size(routes: dict, path: Path) -> None:
    """Claim 3: elicit keeps more of the parent's edge circuit than teach does."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.array(SIZES)
    any_data, chance_plotted = False, False
    for name, route in routes.items():
        if not route.get("available"):
            continue
        any_data = True
        color = ROUTE_COLOR[name]
        jacc = [route["sizes"][str(s)]["edge_jaccard"] for s in SIZES]
        p_ceil = [route["sizes"][str(s)]["ceiling_parent"] for s in SIZES]
        c_ceil = [route["sizes"][str(s)]["ceiling_child"] for s in SIZES]
        lo, hi = np.minimum(p_ceil, c_ceil), np.maximum(p_ceil, c_ceil)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        label = f"{name} (reference only)" if route["reference_only"] else name
        ax.plot(x, jacc, color=color, marker="o", linewidth=2, label=label)
        if not chance_plotted:
            chance = [route["sizes"][str(s)]["chance_edge_jaccard"] for s in SIZES]
            ax.plot(x, chance, color=INK["muted"], linestyle=":", linewidth=1.5,
                     label="chance (exact)")
            chance_plotted = True
    ax.set_xscale("log")
    ax.set_xlabel("circuit size (fraction of edges)")
    ax.set_ylabel("parent -> child edge Jaccard")
    ax.set_title(
        "Claim: elicit fine-tuning keeps more of the parent's edge circuit\n"
        "than teach fine-tuning does, above chance and the split-half ceiling band\n"
        "(single training seed)",
        fontsize=10,
    )
    _strip_axes(ax)
    if any_data:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def plot_change_and_layers(routes: dict, ctx: dict, path: Path) -> None:
    """Claim 4: elicit and teach redistribute edges across different receiving layers."""
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_rate = fig.add_subplot(gs[0, :])
    x = np.array(SIZES)
    any_data = False
    for name, route in routes.items():
        if not route.get("available"):
            continue
        any_data = True
        color = ROUTE_COLOR[name]
        edge_rate = [route["sizes"][str(s)]["edge_change"]["rate"] for s in SIZES]
        node_rate = [route["sizes"][str(s)]["node_change"]["rate"] for s in SIZES]
        ax_rate.plot(x, edge_rate, color=color, linestyle="-", marker="o", label=f"{name} edges")
        ax_rate.plot(x, node_rate, color=color, linestyle="--", marker="s", label=f"{name} nodes")
    ax_rate.set_xscale("log")
    ax_rate.set_xlabel("circuit size (fraction of edges)")
    ax_rate.set_ylabel("change rate (added+removed)/|parent|")
    _strip_axes(ax_rate)
    if any_data:
        ax_rate.legend(frameon=False, fontsize=8, ncol=2)
    else:
        ax_rate.text(0.5, 0.5, "no data available", ha="center", va="center",
                      transform=ax_rate.transAxes)

    layers = np.arange(ctx["graph"].n_layers + 1)
    for idx, name in enumerate(("elicit", "teach")):
        axh = fig.add_subplot(gs[1, idx])
        route = routes.get(name)
        if not route or not route.get("available"):
            axh.text(0.5, 0.5, "no data", ha="center", va="center", transform=axh.transAxes)
            axh.set_title(name, fontsize=9)
            continue
        size = route["hist_size"]
        ov = route["sizes"][str(size)]
        added, removed = np.array(ov["layer_hist_added"]), np.array(ov["layer_hist_removed"])
        axh.bar(layers, added, color=COLOR["blue"], label="added")
        axh.bar(layers, -removed, color=COLOR["red"], label="removed")
        axh.axhline(0, color=INK["baseline"], linewidth=1)
        axh.set_xlabel("receiving layer (16 = logits)")
        axh.set_title(f"{name} @ size {size:g}", fontsize=9)
        if idx == 0:
            axh.set_ylabel("edge count")
        axh.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Claim: elicit and teach redistribute edges across different receiving "
        "layers\n(single training seed)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def plot_parent_in_child(routes: dict, path: Path) -> None:
    """Claim 5: the parent's circuit stays useful in the elicited child, less so
    in the taught child, relative to each child's own circuit and random band."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    width = 0.35
    x = np.arange(len(SIZES))
    handles_labels = None
    for ax, name in zip(axes, ("elicit", "teach")):
        ax.set_title(name)
        route = routes.get(name)
        pic = route.get("parent_in_child") if route and route.get("available") else None
        if not pic:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        own = [pic[str(s)]["f_own"] for s in SIZES]
        parent_f = [pic[str(s)]["f_parent_circuit"] for s in SIZES]
        bands = [pic[str(s)]["f_random_band"] for s in SIZES]
        ax.bar(x - width / 2, own, width, color=COLOR["blue"], label="child's own circuit")
        ax.bar(x + width / 2, parent_f, width, color=COLOR["orange"],
               label="parent's circuit in child")
        for xi, (p5, _p50, p95) in zip(x, bands):
            ax.fill_betweenx([p5, p95], xi - 0.5, xi + 0.5, color=INK["muted"], alpha=0.15,
                              zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in SIZES])
        ax.axhline(0, color=INK["baseline"], linewidth=1)
        ax.grid(True, axis="y", color=INK["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("faithfulness f")
    if handles_labels is not None:
        fig.legend(*handles_labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Claim: the parent's circuit is more useful inside the elicited child\n"
        "than inside the taught child, relative to each child's own circuit and "
        "random band\n(single training seed)"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=INK["surface"])
    plt.close(fig)


def make_plots(models: dict, model_summaries: dict, routes: dict, ctx: dict,
               figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_f_vs_size(models, model_summaries, figures_dir / "1_f_vs_size.png")
    plot_test_grid(model_summaries, figures_dir / "2_test_grid.png")
    plot_overlap_vs_size(routes, figures_dir / "3_overlap_vs_size.png")
    plot_change_and_layers(routes, ctx, figures_dir / "4_change_and_layers.png")
    plot_parent_in_child(routes, figures_dir / "5_parent_in_child.png")


# ------------------------------------------------------------------------ main


def run_compare(results_dir: Path, figures_dir: Path) -> dict:
    ctx = build_graph_ctx()
    models = load_all(results_dir)
    model_summaries = {t: (model_summary(e, ctx) if e is not None else None)
                        for t, e in models.items()}
    routes = {name: route_summary(name, p, c, models, model_summaries, ctx)
              for name, (p, c) in ROUTES.items()}
    children_ref = children_reference(models, model_summaries, ctx)
    gap = accuracy_gap(models)
    compare = {
        "models": model_summaries,
        "routes": routes,
        "children_reference": children_ref,
        "accuracy_gap": gap,
        "missing_models": [t for t, e in models.items() if e is None],
    }
    dump_json(compare, results_dir / "compare.json")
    write_summary_md(compare, results_dir / "summary.md")
    make_plots(models, model_summaries, routes, ctx, figures_dir)
    return compare


def main() -> None:
    global SIZES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--figures-dir", default=str(HERE / "figures"))
    ap.add_argument("--sizes", type=float, nargs="+", default=list(SIZES),
                    help="must match the sizes run.py evaluated")
    a = ap.parse_args()
    SIZES = tuple(a.sizes)
    run_compare(Path(a.results_dir), Path(a.figures_dir))


if __name__ == "__main__":
    main()
