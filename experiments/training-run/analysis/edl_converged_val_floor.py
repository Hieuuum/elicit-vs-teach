"""EDL per label token under the OCV floor (Own-Converged-Validation).

"OCV floor" (named 2026-08-11) is the canonical term for this floor: **O**wn =
that one run's, never shared across dataset sizes; **C**onverged = the last
``eval_log.jsonl`` row (theta_T, the model the stopping rule left), never the
min over the curve; **V**alidation = val loss, never test.

Owner-specified floor, 2026-08-06, and the standing default from here on:

    EDL(n)   = MDL_epoch1(n)  -  D(n) * L_val_converged(n)
    EDL/D(n) = EDL(n) / D(n)

Every term is that ONE run's own. The n=10,000 point subtracts the n=10,000
run's converged validation loss; the n=1,000 point subtracts the n=1,000 run's.
No floor is ever shared between dataset sizes.

``L_val_converged`` is the LAST row of the run's ``eval_log.jsonl`` — the
validation loss of the model the run actually stopped at (theta_T). VERIFIED
2026-08-06 across all 70 runs of both families: the last eval step equals the
manifest's ``experiment.target_result.final_step`` in every case, so the last
eval row is the converged model's own number, not an earlier checkpoint's.

Why this is NOT what ``dataset_size_sweep.py`` plots. That figure uses
``target_result.edl_per_label_token_nats``, whose floor is
``geode.edl.metrics.min_val_nats_from_eval_log`` — the MINIMUM val loss over a
run's eval curve. That minimum is also per-run (the function takes a run_id),
but it is generally reached at some interior step and then left behind: there
is no restore-to-best anywhere in ``geode/edl/loop.py`` or ``train_target.py``,
so the weights that achieved it do not exist at the end of the run. Example,
``evt-llama-fig2nl-noinst-n1000``: val bottoms at 0.400788 (step 16) and the
run stops at 0.471583 (step 35). The min-floored figure subtracts 0.400788; the
converged floor implemented here subtracts 0.471583.

Four distinct floors are in play across this repo. Name the floor whenever a
curve is quoted (decisions.md 2026-07-27):

  moving / per-step      floor recomputed at each step  (prefix_edl_curve only)
  min-over-curve         per run, min of eval_log        (dataset_size_sweep.py)
  CONVERGED val, theta_T per run, last eval_log row      (THIS SCRIPT, default)
  fixed test, theta_T    per run, eval/test_loss.json    (fig2nl_edl_test_floor)

Because the converged floor is >= the min-over-curve floor by construction,
EDL here is <= the min-floored EDL for every run; the gap is exactly
D * (L_val_converged - L_val_min). ``overshoot_ratio`` is carried in the CSV
because overshoot's caveat REVERSES SIGN under this floor rather than
vanishing (decisions.md 2026-08-11): a run whose stopping rule fired on a
HIGH plateau subtracts a larger floor and gets an artificially LOW EDL/D, so
an isolated dip means "run stopped high", not "fast elicitation". Cross-check
``overshoot_ratio`` before quoting any outlier.

Covers both sweep families; writes only ``edl_converged_val_floor*`` names, so
the shipped, immutable §6.10 ``dataset_size_sweep.{parquet,png}`` cannot be
touched (that driver is never invoked from here).

Losses are computed and stored in nats; converted to bits only in the figure
and the ``*_bits`` reporting column. CPU-only, reads the local store, no
network.

Usage:
    python3 edl_converged_val_floor.py [--family {op,nl,both}] [--store <dir>]
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

from geode.edl.metrics import edl_from_totals, edl_nats, epoch1_totals
from geode.zoo import test_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
FIGURES = Path(__file__).resolve().parent / "figures"
LN2 = math.log(2.0)

# --family -> (run-id regex, output stem, figure title tag).
FAMILIES = {
    "op": (
        re.compile(r"^evt-llama-fig2-(noinst|inst)-n(\d+)$"),
        "edl_converged_val_floor_op",
        "operator notation, D_target",
    ),
    "nl": (
        re.compile(r"^evt-llama-fig2nl-(noinst|inst)-n(\d+)$"),
        "edl_converged_val_floor_nl",
        "natural language, D_algo",
    ),
}

# Repo-wide convention, unchanged: base = tab:blue, format-installed = tab:orange.
STYLE = {
    "noinst": ("#1f77b4", "base"),
    "inst": ("#ff7f0e", "format-installed"),
}


def collect(family: str, store: Path) -> pd.DataFrame:
    """One family's per-run table, floored on each run's converged val loss."""
    pattern = FAMILIES[family][0]
    rows = []
    for run_dir in sorted((store / "runs").iterdir()):
        match = pattern.match(run_dir.name)
        if not match or not (run_dir / "logs" / "prequential.jsonl").is_file():
            continue
        condition, n = match.group(1), int(match.group(2))

        mdl, n_label, n_examples = epoch1_totals(run_dir.name, store=store)

        # Masking-parity guard (D-1, V0.5): the library's test-floored EDL must
        # reproduce from our own epoch-1 totals. This validates epoch1_totals and
        # the label-masking path independently of which floor we then apply.
        l_test = test_loss(run_dir.name, store=store).loss_per_label_token_nats
        assert (
            abs(edl_nats(run_dir.name, store=store) - edl_from_totals(mdl, n_label, l_test)) < 1e-6
        )

        evals = sorted(
            (json.loads(line) for line in (run_dir / "eval_log.jsonl").open() if line.strip()),
            key=lambda r: r["step"],
        )
        # theta_T: the model the run actually stopped at. Verified equal to the
        # manifest's final_step for all 70 runs of both families (2026-08-06).
        final_step = evals[-1]["step"]
        l_val_converged = evals[-1]["val_loss_nats"]
        l_val_min = min(r["val_loss_nats"] for r in evals)

        edl = edl_from_totals(mdl, n_label, l_val_converged)
        rows.append(
            {
                "n": n,
                "condition": condition,
                "epoch1_examples": n_examples,
                "label_tokens_D": n_label,
                "mdl_epoch1_nats": mdl,
                "l_val_converged_nats": l_val_converged,
                "edl_nats": edl,
                "edl_per_token_nats": edl / n_label,
                "edl_per_token_bits": edl / n_label / LN2,
                "edl_per_example_nats": edl / n_examples,
                # Reference columns — provenance for the floor, not inputs to it.
                "l_val_min_nats": l_val_min,
                "l_test_nats": l_test,
                "edl_per_token_nats_min_val_floor": edl_from_totals(mdl, n_label, l_val_min)
                / n_label,
                "overshoot_ratio": l_val_converged / l_val_min,
                "final_step": final_step,
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "condition"], ascending=[True, False])


def plot(df: pd.DataFrame, out: Path, family_tag: str) -> None:
    """EDL/D vs. n under the converged-val floor, one curve per condition."""
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for condition, (color, label) in STYLE.items():
        series = df[df["condition"] == condition].sort_values("n")
        if series.empty:
            continue
        ax.plot(series["n"], series["edl_per_token_bits"], color=color, lw=2.0, alpha=0.9, zorder=2)
        ax.plot(
            series["n"],
            series["edl_per_token_bits"],
            ls="none",
            marker="o",
            ms=7,
            color=color,
            mec="white",
            mew=1.4,
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("training examples $n$ (log scale)")
    ax.set_ylabel("EDL/D  (bits per label token)")
    ax.set_title(
        f"EDL per label token vs. dataset size — converged-val floor\nLlama-3.2-1B; {family_tag}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    # A short arm is a stopped sweep, not a diverging curve — say so on the
    # figure, else the gap reads as a result (fig2nl's inst arm ends at n=1e5).
    reach = df.groupby("condition")["n"].max()
    note = ""
    if reach.nunique() > 1:
        short = reach.idxmin()
        note = (
            f"\n{STYLE[short][1]} arm ends at n={int(reach.min()):,} — "
            "sweep stopped there, not a diverging curve."
        )

    fig.text(
        0.5,
        0.005,
        r"Floor = each run's OWN converged ($\theta_T$) validation loss: "
        r"EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$."
        "\nNo floor is shared between dataset sizes." + note,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.10 if note else 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=("op", "nl", "both"), default="both")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    families = ("op", "nl") if args.family == "both" else (args.family,)
    for family in families:
        stem, family_tag = FAMILIES[family][1:]
        df = collect(family, args.store)
        if df.empty:
            print(f"[evt] {family}: no runs found under {args.store / 'runs'} — skipped")
            continue

        csv_path = Path(__file__).resolve().parent / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        plot(df, FIGURES / f"{stem}.png", family_tag)

        negative = df[df["edl_per_token_nats"] < 0]
        print(f"[evt] wrote {csv_path}  ({len(df)} runs)")
        print(
            f"[evt] {family}: {len(df)} runs "
            f"({(df.condition == 'noinst').sum()} noinst / {(df.condition == 'inst').sum()} inst); "
            f"negative EDL/D: {len(negative)}"
        )
        if not negative.empty:
            print(
                "[evt] runs with NEGATIVE EDL/D under this floor (converged val above the epoch-1 mean):"
            )
            for _, r in negative.iterrows():
                print(
                    f"        n={int(r.n):>8d} {r.condition:<7s} EDL/D={r.edl_per_token_nats:+.6f} nats"
                )


if __name__ == "__main__":
    main()
