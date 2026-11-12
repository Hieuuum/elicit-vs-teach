"""Slide figures for the ts1b hard-set accuracy-stop pair (2026-11-12).

Inputs:
  --report   report.json from ts1b_hard_acc_report.py (HF geode-internals:results/ts1b_hard_acc/)
  --runs     experiments/training-run/geode-store/runs (eval_log.jsonl of the four child runs)
  --pg-dir   optional folder of pg/<read>.json files (ts1b_pg.py); the PG figure is skipped if absent
  --out      output folder for PNGs

Old-pair numbers (10-fold means, PLAN.md block 2026-11-04) are constants below: those runs used
other children and another format parent, so they are a comparison, not a recomputation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ELICIT, TEACH, BASE = "#2a78d6", "#eb6834", "#8a8f98"
INK, INK2, GRID = "#1b2430", "#52514e", "#e4e2dc"
BG = "#fbfaf7"
BATCH = 256

# Old pair, 10-fold means (PLAN.md 2026-11-04): elicit child evt-ts1b-elicit-ft-n4000000,
# taught child evt-ts1b-teach-ft-fmt-n4000000 (format parent evt-ts1b-fig2ts-installer).
OLD = {
    "em": {"elicit": 0.960, "teach": 0.111},
    "overlap_1pct": {"elicit": 0.721, "teach": 0.301},
    "functional_1pct": {"elicit": 0.892, "teach": 0.210},
    "cmap_1pct": {"elicit": 0.909, "teach": -1.128},
}
SIZES = ["0.0005", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05"]


def style(ax) -> None:
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=13)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig(w: float, h: float, ncols: int = 1):
    f, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=200)
    f.patch.set_facecolor(BG)
    axes = axes if ncols > 1 else [axes]
    for a in axes:
        style(a)
    return f, axes


def load_eval(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_training(runs: Path, out: Path) -> None:
    """Held-out check accuracy vs examples seen, fresh stage then hard stage, both children."""
    chains = {
        "elicit": ("evt-ts1b-elicit-ft-fresh-b256-c500", "evt-ts1b-elicit-hard-acc", ELICIT),
        "teach": ("evt-ts1b-teach-ft-fresh-b256", "evt-ts1b-teach-hard-acc", TEACH),
    }
    f, (a1, a2) = fig(13, 4.6, 2)
    for name, (fresh, hard, c) in chains.items():
        e1, e2 = load_eval(runs / fresh / "eval_log.jsonl"), load_eval(runs / hard / "eval_log.jsonl")
        x1 = [r["step"] * BATCH / 1e6 for r in e1]
        y1 = [r["format_valid_rate"] for r in e1]
        x2 = [r["step"] * BATCH / 1e6 for r in e2]
        y2 = [r["format_valid_rate"] for r in e2]
        label = "elicit child" if name == "elicit" else "taught child"
        a1.plot(x1, y1, color=c, lw=2, label=label)
        a1.scatter([x1[-1]], [y1[-1]], color=c, s=60, zorder=3, edgecolor=BG, linewidth=2)
        a1.annotate(f"{x1[-1]:.1f}M", (x1[-1], y1[-1]), xytext=(-14, 10), textcoords="offset points",
                    color=INK, fontsize=12)
        a2.plot(x2, y2, color=c, lw=2, marker="o", ms=6, label=label)
        a2.annotate(f"stop: {x2[-1]:.2f}M", (x2[-1], y2[-1]), xytext=(8, -4), textcoords="offset points",
                    color=INK, fontsize=12)
    a1.axhline(0.90, color=INK2, lw=1, ls="--")
    a1.text(0.98, 0.86, "stop bar 0.90 (3 checks)", color=INK2, fontsize=12, ha="right",
            transform=a1.get_yaxis_transform())
    a1.set_xscale("log")
    a1.set_xlabel("training examples seen (millions, log)", color=INK, fontsize=13)
    a1.set_ylabel("held-out accuracy (exact match)", color=INK, fontsize=13)
    a1.set_title("Stage 1: fresh 4M word problems", color=INK, fontsize=15, loc="left")
    a1.set_ylim(0, 1)
    a1.legend(frameon=False, fontsize=13, loc="center left")
    a2.axhline(0.95, color=INK2, lw=1, ls="--")
    a2.text(0.02, 0.952, "stop bar 0.95 (3 checks)", color=INK2, fontsize=12, transform=a2.get_yaxis_transform())
    a2.set_xlabel("examples seen in stage 2 (millions)", color=INK, fontsize=13)
    a2.set_title("Stage 2: continue on hard-weighted problems", color=INK, fontsize=15, loc="left")
    a2.set_ylim(0.93, 0.98)
    a2.set_xlim(0, 0.65)
    f.tight_layout()
    f.savefig(out / "ts1b_hard_acc_training.png", facecolor=BG)
    plt.close(f)


def pct(s: str) -> float:
    return float(s) * 100


def plot_overlap(rep: dict, out: Path) -> None:
    """Edge-overlap Jaccard by circuit size: route vs its blank-base control, one panel per route."""
    pairs = rep["overlap"]["pairs"]
    xs = [pct(s) for s in SIZES]
    f, (a1, a2) = fig(13, 4.6, 2)
    for ax, route, ctrl, c, title, lab in (
        (a1, "ep_sym:ec", "base:ec", ELICIT, "Elicit child vs its parent", "elicit parent ↔ elicit child"),
        (a2, "fp:tc", "base:tc", TEACH, "Taught child vs its parent", "format parent ↔ taught child"),
    ):
        y = [pairs[route][s]["point"] for s in SIZES]
        yb = [pairs[ctrl][s]["point"] for s in SIZES]
        ax.fill_between(xs, yb, y, color=c, alpha=0.12, linewidth=0)
        ax.plot(xs, y, color=c, lw=2.2, marker="o", ms=7, label=lab)
        ax.plot(xs, yb, color=BASE, lw=2, marker="o", ms=6, ls="--", label="blank base ↔ same child")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}%" for x in xs])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_xlabel("circuit size (top % of edges)", color=INK, fontsize=13)
        ax.set_title(title, color=INK, fontsize=15, loc="left")
        ax.legend(frameon=False, fontsize=12, loc="upper right")
    a1.set_ylabel("edge overlap (Jaccard)", color=INK, fontsize=13)
    f.tight_layout()
    f.savefig(out / "ts1b_hard_acc_overlap.png", facecolor=BG)
    plt.close(f)


def bars(ax, groups: list[str], vals: dict[str, list[float]], colors: dict[str, str], width: float = 0.36):
    xs = list(range(len(groups)))
    for k, (name, v) in enumerate(vals.items()):
        pos = [x + (k - 0.5) * (width + 0.04) for x in xs]
        ax.bar(pos, v, width=width, color=colors[name], label=name)
        for p, val in zip(pos, v):
            ax.text(p, val + (0.03 if val >= 0 else -0.03), f"{val:.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", color=INK, fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.axhline(0, color=INK2, lw=1)


def plot_functional(rep: dict, out: Path) -> None:
    fn = rep["functional"]
    sz = ["0.002", "0.01", "0.05"]
    f, (ax,) = fig(8.5, 4.8)
    bars(ax, [f"{pct(s):g}%" for s in sz], {
        "elicit child, parent's circuit": [fn["elicit_child"][s]["share_of_own"]["ep_sym"] for s in sz],
        "taught child, parent's circuit": [fn["teach_child"][s]["share_of_own"]["fp_word"] for s in sz],
    }, {"elicit child, parent's circuit": ELICIT, "taught child, parent's circuit": TEACH})
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("share of the child's own circuit\nperformance kept", color=INK, fontsize=13)
    ax.set_xlabel("circuit size (top % of edges)", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    f.tight_layout()
    f.savefig(out / "ts1b_hard_acc_functional.png", facecolor=BG)
    plt.close(f)


def plot_cmap(rep: dict, out: Path) -> None:
    cm = rep["cmap"]
    sz = ["0.002", "0.01", "0.05"]
    f, (ax,) = fig(8.5, 4.8)
    bars(ax, [f"{pct(s):g}%" for s in sz], {
        "elicit: child edges → parent": [cm["elicit"]["sizes"][s]["child_nologits"]["recovery"] for s in sz],
        "teach: child edges → parent": [cm["teach"]["sizes"][s]["child_nologits"]["recovery"] for s in sz],
    }, {"elicit: child edges → parent": ELICIT, "teach: child edges → parent": TEACH})
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.text(2.62, 1.0, "= child", color=INK2, fontsize=12, va="center", backgroundcolor=BG)
    ax.text(2.62, 0.0, "= parent", color=INK2, fontsize=12, va="center", backgroundcolor=BG)
    ax.set_xlim(-0.6, 3.15)
    ax.set_ylim(-1.6, 1.3)
    ax.set_ylabel("recovery (0 = parent, 1 = child)", color=INK, fontsize=13)
    ax.set_xlabel("circuit size (top % of edges)", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    f.tight_layout()
    f.savefig(out / "ts1b_hard_acc_cmap.png", facecolor=BG)
    plt.close(f)


def plot_old_vs_new(rep: dict, out: Path) -> None:
    """Old pair (10-fold means) vs new matched pair, at 1% for the circuit metrics."""
    new = {
        "em": {"elicit": rep["sets"]["ec"]["exact_match"], "teach": rep["sets"]["tc"]["exact_match"]},
        "overlap_1pct": {"elicit": rep["overlap"]["pairs"]["ep_sym:ec"]["0.01"]["point"],
                         "teach": rep["overlap"]["pairs"]["fp:tc"]["0.01"]["point"]},
        "functional_1pct": {"elicit": rep["functional"]["elicit_child"]["0.01"]["share_of_own"]["ep_sym"],
                            "teach": rep["functional"]["teach_child"]["0.01"]["share_of_own"]["fp_word"]},
        "cmap_1pct": {"elicit": rep["cmap"]["elicit"]["sizes"]["0.01"]["child_nologits"]["recovery"],
                      "teach": rep["cmap"]["teach"]["sizes"]["0.01"]["child_nologits"]["recovery"]},
    }
    titles = {"em": "Child accuracy", "overlap_1pct": "Edge overlap, 1%",
              "functional_1pct": "Functional reuse, 1%", "cmap_1pct": "CMAP recovery, 1%"}
    f, axes = fig(14, 4.4, 4)
    for ax, key in zip(axes, titles):
        for k, (route, c) in enumerate((("elicit", ELICIT), ("teach", TEACH))):
            o, n = OLD[key][route], new[key][route]
            ax.plot([0, 1], [o, n], color=c, lw=2)
            ax.scatter([0], [o], s=70, facecolor=BG, edgecolor=c, linewidth=2, zorder=3)
            ax.scatter([1], [n], s=70, color=c, zorder=3)
            ax.text(-0.08, o, f"{o:.2f}", ha="right", va="center", color=INK, fontsize=12)
            close = abs(new[key]["elicit"] - new[key]["teach"]) < 0.05
            dy = (0.04 if route == "elicit" else -0.04) if close else 0
            ax.text(1.08, n + dy, f"{n:.2f}", ha="left", va="center", color=INK, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["old pair", "new pair"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(titles[key], color=INK, fontsize=14, loc="left")
        lo = -1.5 if key == "cmap_1pct" else 0
        ax.set_ylim(lo, 1.05)
    axes[0].plot([], [], color=ELICIT, lw=2, label="elicit route")
    axes[0].plot([], [], color=TEACH, lw=2, label="teach route")
    axes[0].legend(frameon=False, fontsize=12, loc="center left")
    f.tight_layout()
    f.savefig(out / "ts1b_hard_acc_old_vs_new.png", facecolor=BG)
    plt.close(f)


def plot_pg(pg_dir: Path, out: Path) -> bool:
    reads = [
        ("base_word", "blank base, word", BASE),
        ("fp_word", "format parent, word", TEACH),
        ("ep_word", "elicit parent, word", ELICIT),
        ("base_symbol", "blank base, symbol", BASE),
        ("ep_symbol", "elicit parent, symbol", ELICIT),
    ]
    data = {}
    for key, _, _ in reads:
        p = pg_dir / f"{key}.json"
        if not p.exists():
            return False
        data[key] = json.loads(p.read_text())
    f, (ax,) = fig(10, 4.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    for i, (key, label, c) in enumerate(reads):
        d = data[key]
        lo, hi = d["acc_ci"]
        y = len(reads) - 1 - i
        ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round")
        ax.scatter([d["acc"]], [y], s=90, color=c, zorder=3, edgecolor=BG, linewidth=2)
        ax.text(max(hi, d["acc"]) + 0.012, y, f"{d['acc']:.3f}  ({d['call']})", va="center", color=INK, fontsize=13)
    ax.axvline(0.5, color=INK2, lw=1.2, ls="--")
    ax.text(0.5, len(reads) - 0.45, "chance", ha="center", color=INK2, fontsize=12)
    ax.set_yticks(range(len(reads)))
    ax.set_yticklabels([r[1] for r in reads][::-1], fontsize=13, color=INK)
    lo = min(min(d["acc_ci"][0] for d in data.values()), 0.45)
    hi = max(max(d["acc_ci"][1] for d in data.values()), 0.6)
    ax.set_xlim(lo - 0.02, hi + 0.12)
    ax.set_ylim(-0.6, len(reads) - 0.2)
    ax.set_xlabel("acc*: share of pairs where the true pairing beats the control  (95% CI)", color=INK, fontsize=13)
    f.tight_layout()
    f.savefig(out / "ts1b_pg.png", facecolor=BG)
    plt.close(f)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--pg-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = json.loads(a.report.read_text())
    plot_training(a.runs, a.out)
    plot_overlap(rep, a.out)
    plot_functional(rep, a.out)
    plot_cmap(rep, a.out)
    plot_old_vs_new(rep, a.out)
    if a.pg_dir is not None:
        print("pg figure:", plot_pg(a.pg_dir, a.out))


if __name__ == "__main__":
    main()
