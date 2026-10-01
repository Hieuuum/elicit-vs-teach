"""Quick-look OCV/EDL preview for the ts1b pf-arm target grid, IN PROGRESS.

Plots whichever ``evt-ts1b-pf-target-n<size>`` runs exist locally under
``--store`` (as few as 1, up to all 5) using the same OCV-floor formula and
plot style as ``edl_converged_val_floor.py`` (EDL(n) = MDL_epoch1(n) -
D(n)*L_val_converged(n), theta_T, converged val floor -- see that script's
module docstring for the full definition). Deliberately NOT wired into that
script's FAMILIES/ARM_MAPS dict: ts1b has no base-arm comparator (owner
declined one), so the family needs a real single-arm/no-comparator design
decision before it belongs in the shared multi-family driver -- see memory
"ts1b OCV/EDL curves" checklist item. This is a standalone throwaway for a
mid-grid look, safe to delete once that design question is resolved and
ts1b is added to the shared script properly.

Usage:
    python3 ocv_ts1b_pf_preview.py [--store <dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geode.edl.metrics import edl_from_totals, epoch1_totals
from geode.zoo import test_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
FIGURES = Path(__file__).resolve().parent / "figures"
LN2 = math.log(2.0)
PATTERN = re.compile(r"^evt-ts1b-pf-target-n(\d+)$")
COLOR = "#ff7f0e"  # matches ts38pf's "preteachfmt" arm color (same project convention)
FULL_GRID = (1000, 4642, 21544, 100000, 316228)


def collect(store: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((store / "runs").iterdir()):
        match = PATTERN.match(run_dir.name)
        if not match or not (run_dir / "logs" / "prequential.jsonl").is_file():
            continue
        n = int(match.group(1))
        mdl, n_label, n_examples = epoch1_totals(run_dir.name, store=store)
        evals = sorted(
            (json.loads(line) for line in (run_dir / "eval_log.jsonl").open() if line.strip()),
            key=lambda r: r["step"],
        )
        l_val_converged = evals[-1]["val_loss_nats"]
        l_val_min = min(r["val_loss_nats"] for r in evals)
        l_test = test_loss(run_dir.name, store=store).loss_per_label_token_nats
        edl = edl_from_totals(mdl, n_label, l_val_converged)
        rows.append(
            {
                "n": n,
                "label_tokens_D": n_label,
                "edl_per_token_bits": edl / n_label / LN2,
                "edl_per_token_bits_test_floor": edl_from_totals(mdl, n_label, l_test)
                / n_label
                / LN2,
                "overshoot_ratio": l_val_converged / l_val_min,
            }
        )
    return pd.DataFrame(rows).sort_values("n")


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.plot(df["n"], df["edl_per_token_bits"], color=COLOR, lw=2.0, alpha=0.9, zorder=2)
    ax.plot(
        df["n"],
        df["edl_per_token_bits"],
        ls="none",
        marker="o",
        ms=7,
        color=COLOR,
        mec="white",
        mew=1.4,
        label="pf-arm (pre-teach-format) — OCV floor",
        zorder=3,
    )
    ax.plot(
        df["n"],
        df["edl_per_token_bits_test_floor"],
        color=COLOR,
        lw=1.4,
        ls="--",
        alpha=0.75,
        marker="s",
        ms=4,
        label="pf-arm — test floor (paper Eq. 3)",
        zorder=2,
    )
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(FULL_GRID[0] * 0.7, FULL_GRID[-1] * 1.4)
    ax.set_xlabel("training examples $n$ (log scale)")
    ax.set_ylabel("EDL/D  (bits per label token)")
    done_n = sorted(df["n"])
    ax.set_title(
        "EDL per label token vs. dataset size — converged-val floor\n"
        f"ts1b pf-arm (TinyStories-1B, no base comparator) — PREVIEW, "
        f"{len(done_n)}/{len(FULL_GRID)} sizes done",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    remaining = [n for n in FULL_GRID if n not in done_n]
    note = ""
    if remaining:
        note = (
            "\nMID-GRID PREVIEW: sizes "
            + ", ".join(f"{n:,}" for n in remaining)
            + " have not finished training yet — this curve will change."
        )
    fig.text(
        0.5,
        0.005,
        r"Solid: OCV floor, EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$ "
        r"(this run's own $\theta_T$ val loss)."
        r"  Dashed: test floor, same $\theta_T$ on the held-out test block (paper Eq. 3)."
        "\nNo base-arm comparator exists at this scale (owner declined it) — read shape only."
        + note,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.12 if note else 0.08, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ocv-ts1b-pf-preview] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()
    df = collect(args.store)
    if df.empty:
        print(f"[ocv-ts1b-pf-preview] no evt-ts1b-pf-target-n* runs found under {args.store / 'runs'}")
        return
    csv_path = Path(__file__).resolve().parent / "edl_converged_val_floor_ts1b_pf_preview.csv"
    df.to_csv(csv_path, index=False)
    plot(df, FIGURES / "edl_converged_val_floor_ts1b_pf_preview.png")
    print(f"[ocv-ts1b-pf-preview] wrote {csv_path}  ({len(df)}/{len(FULL_GRID)} sizes)")
    for _, r in df.iterrows():
        print(
            f"        n={int(r.n):>8d}  EDL/D(OCV)={r.edl_per_token_bits:+.4f} bits  "
            f"overshoot_ratio={r.overshoot_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
