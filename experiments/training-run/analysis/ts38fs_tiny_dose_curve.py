"""EDL/D vs. target-dataset-size n, one curve per format-install DOSE i —
the full ts38fs dose axis, extended down to i=2/10/100 by ts38fs-tiny
(EXPERIMENTS §6.20, decisions.md 2026-08-20 "ts38fs sweep killed mid-run,
ts38fs-tiny extends the dose grid down to i=2/10"; i=100 added 2026-08-21,
same entry's follow-on paragraph).

Same OCV floor as ``edl_converged_val_floor.py`` (owner default 2026-08-06)
and ``ts38fs_dose_curve.py`` (this module's non-tiny sibling):

    EDL(n) = MDL_epoch1(n) - D(n) * L_val_converged(n)

Single seed only (316 — ts38fs-tiny's own single-seed design; it is also
the only ts38fs-proper seed with broad size x install coverage, per
decisions.md). i=21544 is spliced in from the ts38pf family's own 5 runs
(evt-ts38pf-preteachfmt-n<n>), exactly as ts38fs proper's own script does.

**Recipe mismatch, i<1000 vs i>=1000 — plotted, not hidden.** i=2/10/100
(ts38fs-tiny) use train-loss full-batch stopping + forced-derangement
labels (cyclic_shift_labels, V5.78); i>=1000 (ts38fs proper) use val-loss
eps/k stopping + random-shuffle labels (permute_labels). The install
RECIPE differs at this boundary even though the target-side OCV-floor EDL
math is identical for every point. This script marks the tiny doses with
hollow markers + dashed connecting lines (proper doses stay solid/filled)
and says so in the caption — deliberately, per the parent memory's
instruction to extend `ts38fs_dose_curve.py` "with a visible
recipe-mismatch note ... never by loosening the regex to swallow both
silently." This is why this is a NEW script and not an edit to that one's
regex.

Also plots the format-acquisition check (theta0 loss_drop_frac vs install
dose i): exact values for i=2/10/100 from
``results/ts38fs_tiny_format_acquisition.json``; i>=1000 shown only as the
0.48-0.59 range decisions.md records for the (killed) proper sweep — no
fabricated per-install points, since that sweep's own per-cell JSON was
never pushed to the relay.

Usage (CPU, seconds; pull run metadata first with
``hf_checkpoint.py pull --run-id <id> --no-weights`` for every id this
script needs):
    python3 ts38fs_tiny_dose_curve.py [--store DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

SEED = 316
DOSES = (2, 10, 100, 1000, 4642, 21544, 100000)
TINY_DOSES = frozenset({2, 10, 100})
SIZES = (1000, 4642, 21544, 100000, 316228)
TOTAL_CELLS = len(DOSES) * len(SIZES)

# decisions.md 2026-08-20 "ts38fs sweep killed mid-run" entry: all 4
# format-acquisition checks at i in {1000,4642,21544,100000} came back
# verdict=LEARNED, loss_drop_frac in this range (no per-install breakdown
# survives — that sweep's own results JSON was never pushed to the relay).
PROPER_LOSS_DROP_RANGE = (0.48, 0.59)


def run_id_for(i: int, n: int) -> str:
    if i == 21544:
        return f"evt-ts38pf-preteachfmt-n{n}"
    if i in TINY_DOSES:
        return f"evt-ts38fs-tiny-i{i}-n{n}-s{SEED}"
    return f"evt-ts38fs-i{i}-n{n}-s{SEED}"


def expected_cells() -> list[tuple[int, int]]:
    return [(i, n) for i in DOSES for n in SIZES]


def collect(store: Path) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """One row per COMPLETE cell found locally; the rest come back as pending."""
    rows = []
    pending = []
    for i, n in expected_cells():
        rid = run_id_for(i, n)
        run_dir = store / "runs" / rid
        if not (run_dir / "logs" / "prequential.jsonl").is_file():
            pending.append((i, n))
            continue

        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            pending.append((i, n))
            continue

        mdl, n_label, n_examples = epoch1_totals(rid, store=store)

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
                "run_id": rid,
                "recipe": "tiny (train-loss stop, cyclic-shift)" if i in TINY_DOSES else "proper (val-loss stop, permute)",
                "reused_ts38pf": i == 21544,
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
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["install_i", "n"]).reset_index(drop=True)
    return df, pending


# Reuses ts38fs_dose_curve.py's own i>=1000 palette so a reader flipping
# between the two figures sees the same color per dose; i=2/10 get new,
# clearly distinguishable colors.
DOSE_COLOR = {
    2: "#17becf",
    10: "#1f77b4",
    100: "#e377c2",
    1000: "#2ca02c",
    4642: "#9467bd",
    21544: "#ff7f0e",
    100000: "#d62728",
}


def plot_edl_curve(df: pd.DataFrame, out: Path, n_complete: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in DOSES:
        series = df[df["install_i"] == i].sort_values("n")
        if series.empty:
            continue
        color = DOSE_COLOR[i]
        tiny = i in TINY_DOSES
        label = f"i={i:,}" + (" (tiny recipe)" if tiny else "") + (" [reused ts38pf]" if i == 21544 else "")
        ax.plot(
            series["n"], series["edl_per_token_bits"],
            color=color, lw=1.8, alpha=0.9, ls="--" if tiny else "-", zorder=2,
        )
        ax.plot(
            series["n"], series["edl_per_token_bits"],
            ls="none", marker="o", ms=7, color=("white" if tiny else color),
            mec=color, mew=1.6, label=label, zorder=3,
        )
    non_converged = df[df["stop_reason"] != "converged"]
    for _, r in non_converged.iterrows():
        ax.scatter([r["n"]], [r["edl_per_token_bits"]], marker="x", s=100, color="red", zorder=4)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("target examples $n$ (log scale)")
    ax.set_ylabel("EDL/D  (bits per label token, OCV floor)")
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="best")

    fig.suptitle(
        f"ts38fs + ts38fs-tiny — EDL/D vs. target size n, by format-install dose i\n"
        f"TinyStories 38.7M, D_algo_bare, r128 LoRA, seed={SEED}  —  "
        f"{n_complete}/{TOTAL_CELLS} cells complete",
        fontsize=12,
    )
    fig.text(
        0.5, 0.005,
        r"EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$, OCV floor "
        "(each run's own converged val loss). Hollow dashed markers = ts38fs-tiny's "
        "different install recipe (train-loss-stop + forced derangement, not val-loss-stop "
        "+ random shuffle) — only the install side differs; target-run EDL math is identical. "
        "Red x = stop_reason != converged (bug signal, not a result).",
        ha="center", va="bottom", fontsize=7.7, color="#555555", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ts38fs-tiny] wrote {out}")


def plot_format_acquisition(out: Path, tiny_json: Path) -> None:
    if not tiny_json.is_file():
        print(f"[ts38fs-tiny] {tiny_json} not found — skipping format-acquisition figure")
        return
    data = json.loads(tiny_json.read_text())
    tiny_points = sorted(
        ((int(i), v["loss_drop_frac"], v["verdict"]) for i, v in data["parents"].items()),
        key=lambda t: t[0],
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo, hi = PROPER_LOSS_DROP_RANGE
    ax.axhspan(lo, hi, xmin=0.0, color="#d62728", alpha=0.12, zorder=1)
    ax.plot(
        [1000, 100000], [(lo + hi) / 2] * 2,
        color="#d62728", lw=0, marker="", zorder=1,
    )
    ax.annotate(
        f"ts38fs proper (i=1,000..100,000): {lo:.2f}-{hi:.2f}\n(range only — per-install JSON never pushed to relay)",
        xy=(3000, hi), xytext=(3000, hi + 0.05), fontsize=8, color="#d62728",
    )
    xs = [p[0] for p in tiny_points]
    ys = [p[1] for p in tiny_points]
    ax.plot(xs, ys, color="#1f77b4", lw=1.8, ls="--", zorder=2)
    for i, y, verdict in tiny_points:
        ax.plot(
            [i], [y], ls="none", marker="o", ms=9, color="white", mec="#1f77b4", mew=1.8, zorder=3
        )
        ax.annotate(f"i={i}: {verdict}\nloss_drop_frac={y:.3f}", xy=(i, y),
                    xytext=(i * 1.5, y + 0.05), ha="left", fontsize=8, zorder=5)
    ax.axhline(0.10, color="#999999", lw=1.0, ls=":", zorder=1)
    ax.annotate("LEARNED threshold (0.10)", xy=(30, 0.10), xytext=(30, 0.12), fontsize=7.5, color="#666666")
    ax.set_xscale("log")
    ax.set_xlabel("format-install size $i$ (log scale)")
    ax.set_ylabel("loss_drop_frac = (base.loss - parent.loss) / base.loss")
    ax.set_ylim(0, max(hi, max(ys)) + 0.15)
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    all_learned = all(verdict == "LEARNED" for _, _, verdict in tiny_points)
    subtitle = (
        "all tested doses clear the LEARNED bar; still no NOT_LEARNED floor"
        if all_learned
        else "NOT_LEARNED floor found — see the point(s) below the threshold line"
    )
    fig.suptitle(
        f"ts38fs-tiny — format-acquisition theta0 check vs. install dose i\n{subtitle}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ts38fs-tiny] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    df, pending = collect(args.store)
    csv_path = Path(__file__).resolve().parent / "edl_converged_val_floor_ts38fs_tiny.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ts38fs-tiny] wrote {csv_path}  ({len(df)}/{TOTAL_CELLS} cells complete)")

    if pending:
        print(f"[ts38fs-tiny] {len(pending)} cell(s) missing locally (not pulled, or not yet trained):")
        for i, n in pending:
            print(f"[ts38fs-tiny]   i={i} n={n}  (run_id={run_id_for(i, n)})")

    if df.empty:
        print("[ts38fs-tiny] no complete cells — nothing to plot")
        return

    non_converged = df[df["stop_reason"] != "converged"]
    if not non_converged.empty:
        print(
            f"[ts38fs-tiny] WARNING: {len(non_converged)} cell(s) did NOT converge "
            f"(stop_reason != 'converged') — bug signal, not a result: "
            f"{non_converged['run_id'].tolist()}"
        )
    negative = df[df["edl_per_token_nats"] < 0]
    if not negative.empty:
        print(f"[ts38fs-tiny] negative EDL/D on {len(negative)} cell(s): {negative['run_id'].tolist()}")

    plot_edl_curve(df, FIGURES / "edl_converged_val_floor_ts38fs_tiny.png", n_complete=len(df))
    plot_format_acquisition(
        FIGURES / "ts38fs_tiny_format_acquisition.png",
        args.store / "results" / "ts38fs_tiny_format_acquisition.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
