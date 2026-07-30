"""Render the headline runs-5/6 figures (Arm A elicit vs Arm B teach) to PNGs.

The five figures that carry the elicit-vs-teach story, distilled from
pilot_loss_compare.ipynb:

  1. prequential learning curve (epoch 1)   — the paper comparison
  2. EDL accumulation (running excess bits) — where each arm stops paying
  3. held-out val loss                      — the normal comparison
  4. MDL / EDL totals + normalizations      — the headline numbers
  5. G5 zero-shot accuracy + theta_T        — behavioral evidence

Usage: python key_figures.py [--out DIR]
Reads $GEODE_STORE (default: <repo>/geode-store); writes DIR (default:
figures/ next to this file). CPU-only, no cost.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "geode" / "__init__.py").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from geode.edl.metrics import (  # noqa: E402
    edl_nats,
    edl_per_label_token,
    edl_per_param,
    mdl_nats,
    nats_to_bits,
    prefix_edl_curve,
    training_curve,
)

LN2 = float(np.log(2))
STORE = Path(os.environ.get("GEODE_STORE") or REPO_ROOT / "geode-store")
RUNS = {
    "Arm A (elicit) @500K": "evt-run5-armA-target",
    "Arm B (teach) @500K": "evt-run6-armB-target",
}


def load(rid: str) -> dict:
    d = STORE / "runs" / rid
    m = json.loads((d / "manifest.json").read_text())
    tl = d / "eval" / "test_loss.json"
    return {
        "curve": training_curve(rid, store=STORE),
        "val": pd.DataFrame(
            json.loads(ln) for ln in (d / "eval_log.jsonl").read_text().splitlines() if ln.strip()
        ),
        "result": m["experiment"].get("target_result") or {},
        "g5": (m["experiment"].get("gates") or {}).get("G5") or {},
        "theta_T_nats": json.loads(tl.read_text())["loss_per_label_token_nats"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "figures")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    runs = {label: load(rid) for label, rid in RUNS.items()}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1 — prequential learning curve (epoch 1), the paper comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (label, r), color in zip(runs.items(), colors):
        c = r["curve"]
        x = c["example_index"] + 1
        bits = c["loss_per_label_token_nats"] / LN2
        ax.plot(x, bits, color=color, alpha=0.25, lw=0.8)
        ax.plot(x, bits.rolling(9, min_periods=1).mean(), color=color, label=label)
    ax.set(
        xlabel="examples seen (epoch 1)",
        ylabel="prequential loss (bits / label token)",
        xscale="log",
        yscale="log",
        title="Prequential learning curve (epoch 1) — the paper comparison",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(out / "1_prequential_curve.png", dpi=150, bbox_inches="tight")

    # 2 — EDL accumulation: running excess over the converged model's code. The
    # canonical test-floored prefix curve (geode.edl.prefix_edl_curve, V5.73),
    # sampled at the in-loop eval steps rather than every prequential record.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (label, rid), color in zip(RUNS.items(), colors):
        edl = prefix_edl_curve(rid, floor="test", store=STORE)
        ax.plot(
            edl["examples"].to_numpy(),
            (edl["edl_nats"] / LN2).to_numpy(),
            color=color,
            label=label,
        )
    ax.set(
        xlabel="examples seen (epoch 1)",
        ylabel="cumulative excess bits  (≈ EDL(n))",
        xscale="log",
        yscale="log",
        title="EDL accumulation — where each arm stops paying for the task",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(out / "2_edl_accumulation.png", dpi=150, bbox_inches="tight")

    # 3 — held-out val loss (the normal comparison), "x" = last eval
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (label, r), color in zip(runs.items(), colors):
        v = r["val"]
        bits = v["val_loss_nats"] / LN2
        stop = r["result"].get("stop_reason") or "running"
        ax.plot(v["step"], bits, color=color, marker="o", ms=3, label=f"{label} [{stop}]")
        ax.plot(v["step"].iloc[-1], bits.iloc[-1], color=color, marker="x", ms=10, mew=2)
    ax.set(
        xlabel="step",
        ylabel="val loss (bits / label token)",
        xscale="log",
        yscale="log",
        title="Held-out val loss — in-loop evals",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(out / "3_val_loss.png", dpi=150, bbox_inches="tight")

    # 4 — MDL / EDL totals + normalizations (log bars; the point is the gap)
    rows = [
        {
            "run": label,
            "MDL_bits": nats_to_bits(mdl_nats(rid, store=STORE)),
            "EDL_bits": nats_to_bits(edl_nats(rid, store=STORE)),
            "EDL/D_bits": nats_to_bits(edl_per_label_token(rid, store=STORE)),
            "EDL/P_bits": nats_to_bits(edl_per_param(rid, store=STORE)),
        }
        for label, rid in RUNS.items()
    ]
    df = pd.DataFrame(rows)
    ratio = df["EDL_bits"].iloc[1] / df["EDL_bits"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(df))
    for ax, cols, title in [
        (axes[0], ["MDL_bits", "EDL_bits"], "MDL and EDL (bits)"),
        (axes[1], ["EDL/D_bits", "EDL/P_bits"], "EDL per label token / per adapter param"),
    ]:
        for i, col in enumerate(cols):
            bars = ax.bar(x + (i - 0.5) * 0.38, df[col], width=0.36, label=col)
            ax.bar_label(bars, fmt="%.3g", fontsize=8)
        ax.set(title=title, yscale="log", xticks=x, xticklabels=df["run"])
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(f"Information cost of D_target — EDL ratio B/A = {ratio:.1f}x")
    fig.savefig(out / "4_mdl_edl_summary.png", dpi=150, bbox_inches="tight")

    # 5 — G5 behavioral evidence: zero-shot accuracy + theta_T (16-shot omitted:
    # metric invalidated, decisions.md)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(runs)
    acc = [r["g5"].get("zero_shot_accuracy", np.nan) for r in runs.values()]
    theta = [nats_to_bits(r["theta_T_nats"]) for r in runs.values()]
    b0 = axes[0].bar(names, acc, color=colors[: len(runs)])
    axes[0].bar_label(b0, fmt="%.4f")
    axes[0].set(title="G5 zero-shot exact-match accuracy", ylim=(0, 1.05))
    b1 = axes[1].bar(names, theta, color=colors[: len(runs)])
    axes[1].bar_label(b1, fmt="%.4g")
    axes[1].set(title="θ_T — converged test loss (bits / label token)", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.savefig(out / "5_g5_evidence.png", dpi=150, bbox_inches="tight")

    for p in sorted(out.glob("*.png")):
        print(p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
