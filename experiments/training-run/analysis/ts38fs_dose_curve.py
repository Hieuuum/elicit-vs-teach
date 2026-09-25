"""EDL/D vs. target-dataset-size n, one curve per format-install DOSE — the
ts38fs family (EXPERIMENTS §6.20, decisions.md 2026-08-20 "ts38fs
pre-registration").

Same OCV floor as ``edl_converged_val_floor.py`` (owner default 2026-08-06):

    EDL(n) = MDL_epoch1(n) - D(n) * L_val_converged(n)

reusing that module's own floor primitives (``geode.edl.metrics``,
``geode.zoo.test_loss``) rather than recomputing the algebra — see that
file's module docstring for the floor's full derivation and caveats
(overshoot_ratio sign flip, converged vs min-over-curve, etc.).

ts38fs is NOT a 2-arm family (op/nl/ts38/ts38mw/ts38pf/ts38pp all key on one
noinst/inst-style condition), so this does not plug into that module's
FAMILIES/collect()/plot() machinery, which is hardwired to exactly two
conditions. This is a 3-axis grid instead: install size i (the format-install
DOSE) x target size n x seed s. Standalone script, per this launcher's own
closing note ("This family has no edl_converged_val_floor.py FAMILIES entry
yet ... adding one is separate follow-up work").

Grid: i in {1000, 4642, 21544, 100000}, n in {1000, 4642, 21544, 100000,
316228}, s in {316, 1316, 2316}, minus the 5 (i=21544, s=316) cells that
REUSE the ts38pf family's own 5 runs (evt-ts38pf-preteachfmt-n<n>) rather
than retraining under an evt-ts38fs- id — spliced in here by hand, exactly as
launch_ts38fs_family.sh's own final milestone says to.

ts38dense (EXPERIMENTS §6.21, decisions.md 2026-08-21 "ts38dense
pre-registration") extends the i=1000 dose point ONLY, at seed 316 ONLY,
with 5 densified target sizes n in {2154, 10000, 46416, 146780, 215443}
(``evt-ts38fs-i1000-n<n>-s316``, trained from a differently-named overlay,
``ts38fs_dense_i1000_n<n>.yaml``, chosen so the launcher's exactly-55
``ts38fs_i*_n*_s*.yaml`` glob guard stays unaffected — see
``scripts/launch_ts38dense_family.sh``). This is NOT a cartesian extension
against every install or seed: those (i, n, s) combinations were never
trained and would sit "pending" forever if enumerated here. 65 cells total.

This script is SAFE TO RUN WHILE THE SWEEP IS STILL TRAINING: it only reads
whatever cells exist locally and reports the rest as pending, rather than
failing. Re-run any time to refresh the CSV/figure as more cells land.

Usage (CPU, seconds):
    python3 ts38fs_dose_curve.py [--store DIR]
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

INSTALLS = (1000, 4642, 21544, 100000)
SIZES = (1000, 4642, 21544, 100000, 316228)
SEEDS = (316, 1316, 2316)

# ts38dense (EXPERIMENTS §6.21, decisions.md 2026-08-21 "ts38dense
# pre-registration"): 5 densified target sizes at the i=1000 dose point,
# seed 316 ONLY (the i=1000/s=316 row already has a second seed, 1316, at
# the 5 shipped SIZES; the 5 new sizes get seed 316 only — a pre-declared
# asymmetry). Deliberately NOT crossed against every install or seed: those
# (i, n, s) combinations were never run.
DENSE_SIZES = (2154, 10000, 46416, 146780, 215443)
DENSE_INSTALL = 1000
DENSE_SEED = 316
DENSE_CELLS: tuple[tuple[int, int, int], ...] = tuple(
    (DENSE_INSTALL, n, DENSE_SEED) for n in DENSE_SIZES
)

# Total non-reused cells this launcher trains (55) + the 5 reused ts38pf
# rows this script splices in by hand + the 5 ts38dense cells (i=1000,
# s=316 only) = the full 65-cell grid.
TOTAL_CELLS = len(INSTALLS) * len(SIZES) * len(SEEDS) + len(DENSE_CELLS)

TS38FS_RE = re.compile(r"^evt-ts38fs-i(\d+)-n(\d+)-s(\d+)$")
TS38PF_REUSED_RE = re.compile(r"^evt-ts38pf-preteachfmt-n(\d+)$")


def expected_cells() -> list[tuple[int, int, int]]:
    return [(i, n, s) for s in SEEDS for i in INSTALLS for n in SIZES] + list(DENSE_CELLS)


def run_id_for(i: int, n: int, s: int) -> str:
    if i == 21544 and s == 316:
        return f"evt-ts38pf-preteachfmt-n{n}"
    return f"evt-ts38fs-i{i}-n{n}-s{s}"


def collect(store: Path) -> tuple[pd.DataFrame, list[tuple[int, int, int]]]:
    """One row per COMPLETE cell found locally; the rest come back as pending."""
    rows = []
    pending = []
    for i, n, s in expected_cells():
        rid = run_id_for(i, n, s)
        run_dir = store / "runs" / rid
        if not (run_dir / "logs" / "prequential.jsonl").is_file():
            pending.append((i, n, s))
            continue

        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            pending.append((i, n, s))
            continue

        mdl, n_label, n_examples = epoch1_totals(rid, store=store)

        # Same masking-parity guard edl_converged_val_floor.py's collect()
        # applies — validates epoch1_totals/label-masking independently of
        # which floor gets used below.
        l_test = test_loss(rid, store=store).loss_per_label_token_nats
        assert abs(edl_nats(rid, store=store) - edl_from_totals(mdl, n_label, l_test)) < 1e-6

        evals = sorted(
            (json.loads(line) for line in (run_dir / "eval_log.jsonl").open() if line.strip()),
            key=lambda r: r["step"],
        )
        final_step = evals[-1]["step"]
        l_val_converged = evals[-1]["val_loss_nats"]
        l_val_min = min(r["val_loss_nats"] for r in evals)
        stop_reason = manifest["experiment"]["target_result"]["stop_reason"]

        edl = edl_from_totals(mdl, n_label, l_val_converged)
        rows.append(
            {
                "install_i": i,
                "n": n,
                "seed": s,
                "run_id": rid,
                "reused_ts38pf": i == 21544 and s == 316,
                "stop_reason": stop_reason,
                "final_step": final_step,
                "label_tokens_D": n_label,
                "mdl_epoch1_nats": mdl,
                "l_val_converged_nats": l_val_converged,
                "l_val_min_nats": l_val_min,
                "overshoot_ratio": l_val_converged / l_val_min,
                "edl_nats": edl,
                "edl_per_token_nats": edl / n_label,
                "edl_per_token_bits": edl / n_label / LN2,
                "l_test_nats": l_test,
                "edl_per_token_bits_test_floor": edl_from_totals(mdl, n_label, l_test)
                / n_label
                / LN2,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["seed", "install_i", "n"]).reset_index(drop=True)
    return df, pending


# Repo convention (edl_converged_val_floor.py's STYLE dict) reused for i=21544
# so this figure's coloring lines up with ts38pf's own noinst/inst plot;
# the three NEW install sizes get their own distinguishable colors.
DOSE_COLOR = {
    1000: "#2ca02c",
    4642: "#9467bd",
    21544: "#ff7f0e",
    100000: "#d62728",
}


def plot(df: pd.DataFrame, out: Path, n_complete: int) -> None:
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(16, 5.2), sharey=True)
    for ax, s in zip(axes, SEEDS, strict=True):
        sub = df[df["seed"] == s]
        for i in INSTALLS:
            series = sub[sub["install_i"] == i].sort_values("n")
            if series.empty:
                continue
            color = DOSE_COLOR[i]
            ax.plot(
                series["n"], series["edl_per_token_bits"], color=color, lw=1.8, alpha=0.9, zorder=2
            )
            ax.plot(
                series["n"],
                series["edl_per_token_bits"],
                ls="none",
                marker="o",
                ms=6,
                color=color,
                mec="white",
                mew=1.1,
                label=f"install i={i:,}",
                zorder=3,
            )
        non_converged = sub[sub["stop_reason"] != "converged"]
        for _, r in non_converged.iterrows():
            ax.scatter([r["n"]], [r["edl_per_token_bits"]], marker="x", s=90, color="red", zorder=4)
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("target examples $n$ (log scale)")
        ax.set_title(f"seed={s}" + ("  (has reused i=21544 row)" if s == 316 else ""), fontsize=10)
        ax.grid(True, which="both", alpha=0.18, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("EDL/D  (bits per label token, OCV floor)")
    # One shared legend built from whichever axes actually have data — a
    # per-axis legend() call errors/warns on a seed with zero complete cells
    # yet (expected mid-sweep: seed order is 316 -> 1316 -> 2316).
    handles, labels = [], []
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, fontsize=8, frameon=False, loc="upper right", ncol=len(handles))

    fig.suptitle(
        f"ts38fs — EDL/D vs. target size n, by format-install dose i\n"
        f"TinyStories 38.7M, D_algo_bare, r128 LoRA  —  "
        f"PARTIAL: {n_complete}/{TOTAL_CELLS} cells complete (sweep still running)",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        r"EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$, OCV floor "
        "(each run's own converged val loss; no floor shared across n or i). "
        "Red x = stop_reason != converged (bug signal, not a result). "
        "Re-run this script to refresh as more cells land.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ts38fs] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    df, pending = collect(args.store)
    csv_path = Path(__file__).resolve().parent / "edl_converged_val_floor_ts38fs.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ts38fs] wrote {csv_path}  ({len(df)}/{TOTAL_CELLS} cells complete)")

    if pending:
        print(
            f"[ts38fs] {len(pending)} cell(s) still pending (sweep in progress or not yet reached):"
        )
        by_seed: dict[int, list[str]] = {}
        for i, n, s in pending:
            by_seed.setdefault(s, []).append(f"i{i}-n{n}")
        for s in sorted(by_seed):
            print(f"[ts38fs]   seed={s}: {', '.join(by_seed[s])}")

    if df.empty:
        print("[ts38fs] no complete cells yet — nothing to plot")
        return

    non_converged = df[df["stop_reason"] != "converged"]
    if not non_converged.empty:
        print(
            f"[ts38fs] WARNING: {len(non_converged)} cell(s) did NOT converge "
            f"(stop_reason != 'converged') — bug signal, not a result: "
            f"{non_converged['run_id'].tolist()}"
        )
    negative = df[df["edl_per_token_nats"] < 0]
    if not negative.empty:
        print(f"[ts38fs] negative EDL/D on {len(negative)} cell(s): {negative['run_id'].tolist()}")

    plot(df, FIGURES / "edl_converged_val_floor_ts38fs.png", n_complete=len(df))


if __name__ == "__main__":
    raise SystemExit(main())
