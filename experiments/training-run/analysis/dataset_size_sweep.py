"""Figure-2 dataset-size sweep: EDL/D vs. training-set size (Llama pair).

Reproduces the paper's Figure-2 protocol on the Llama target stage: 19
log-spaced prefix-nested dataset sizes (``n = round(10**(3 + i/6))``,
``i = 0..18``, 1,000 .. 1,000,000), each trained fresh to val convergence,
under two conditions:

- ``noinst`` — base ``meta-llama/Llama-3.2-1B``, no format install ("base").
- ``inst``   — base + a format install ("format-installed").

Two sweep FAMILIES share this driver, selected by ``--family`` — the protocol
and every metric are identical, only the target task differs:

- ``op`` (default) — operator-notation add/sub on ``D_target``, run ids
  ``evt-llama-fig2-{noinst,inst}-n<size>``, EXPERIMENTS §6.10. SHIPPED and
  immutable; its ``results/dataset_size_sweep.parquet`` and
  ``figures/dataset_size_sweep.png`` are history.
- ``nl`` — natural-language add/sub on ``D_algo``, run ids
  ``evt-llama-fig2nl-{noinst,inst}-n<size>``, EXPERIMENTS §6.11. Writes to the
  distinct ``_nl`` stem, so an NL run can never overwrite the ``op`` outputs
  (``write_results`` is overwrite-by-name, OQ-6).
- ``nl2`` — same NL target, redesigned installer (NL-format dose at the
  retention-preserving LR, both installer gates enforced), run ids
  ``evt-llama-fig2nl2-{noinst,inst}-n<size>``, EXPERIMENTS §6.12. Distinct
  ``_nl2`` stem for the same overwrite-by-name reason.
- ``nl3`` — scaffold-free NL add/sub on ``D_algo_bare`` (the frozen D_algo
  re-rendered without the Question:/Answer: scaffold), bare dose16 installer,
  run ids ``evt-llama-fig2nl3-{noinst,inst}-n<size>``, EXPERIMENTS §6.13.
  Distinct ``_nl3`` stem.

38 target runs per family (the family's installer run is not a sweep point).

Reads each run's manifest via ``geode.zoo`` — ``experiment.target_result``
(``min_val_nats``, ``stop_reason``, ``edl_epoch1_nats``,
``edl_per_label_token_nats``, ``edl_per_example_nats``, written by
``train_target.py`` at finalize), ``eval/test_loss.json`` (spec 00 §5, the
canonical fixed-floor test loss), and ``experiment.gates.G5`` (zero-shot exact
match, written by ``gates.py g5``) — and writes one long-format parquet,
``results/dataset_size_sweep.parquet``, through the spec 00 §7 / ZOO-6
results-table writer (``layer = -1`` sentinel: every metric here is a
whole-model quantity, not layer-resolved). Losses are stored in **nats**
(field names end ``_nats``); the figure converts to bits only at that
reporting boundary.

``stop_reason != "converged"`` is a loud, per-run WARNING on stdout (a
``max_steps`` stop is a bug signal per the run-until-convergence policy) —
the run is still included in the parquet and the figure, just flagged. A run
whose manifest cannot be loaded yet (not pulled, or genuinely absent) is
skipped with its own warning rather than raising, so the figure is viewable
against a partial store mid-sweep (this script is meant to run against
``hf_checkpoint.py pull --no-weights`` output: metadata only, no model
weights, no GPU, no network of its own).

CPU-only.

Usage:
    python3 dataset_size_sweep.py [--family {op,nl}] [--run-id <rid> ...]
        [--store <dir>] [--fig <path>]
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geode.zoo import load_run, test_loss, write_results
from geode.zoo.manifest import ManifestError

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_LAYER = -1  # sentinel: these metrics are whole-model, not per layer
LN2 = math.log(2.0)

CONDITIONS = {"noinst": "base", "inst": "format-installed"}
SIZES: tuple[int, ...] = tuple(round(10 ** (3 + i / 6)) for i in range(19))

# --family value -> (run-id prefix, results-table + figure stem, title tag).
# The stems MUST stay distinct: write_results is overwrite-by-name (OQ-6), so a
# shared stem would let an NL run silently replace the shipped §6.10 table.
FAMILIES: dict[str, tuple[str, str, str]] = {
    "op": ("evt-llama-fig2", "dataset_size_sweep", "operator notation, D_target"),
    "nl": ("evt-llama-fig2nl", "dataset_size_sweep_nl", "natural language, D_algo"),
    "nl2": (
        "evt-llama-fig2nl2",
        "dataset_size_sweep_nl2",
        "natural language, D_algo; NL-dose installer",
    ),
    "nl3": (
        "evt-llama-fig2nl3",
        "dataset_size_sweep_nl3",
        "scaffold-free NL, D_algo_bare; bare dose16 installer",
    ),
}
DEFAULT_FAMILY = "op"

# All families in one pattern. They cannot cross-match: the "nl"/"nl2"/"nl3"
# infix means an "evt-llama-fig2nl-..." id fails the op reading and vice
# versa, so each id parses unambiguously without the caller declaring its
# family.
RUN_ID_RE = re.compile(r"^evt-llama-fig2(?:nl[23]?)?-(noinst|inst)-n(\d+)$")

# The headline "EDL/D vs dataset size" metric for the figure — D = training
# label tokens in the epoch-1 stream (edl_epoch1_nats / epoch-1 label tokens).
HEADLINE_METRIC = "edl_per_label_token_nats"
# Per-run scalar metrics read straight off experiment.target_result.
TARGET_RESULT_METRICS = (
    "min_val_nats",
    "edl_epoch1_nats",
    "edl_per_label_token_nats",
    "edl_per_example_nats",
)


def default_run_ids(family: str = DEFAULT_FAMILY) -> list[str]:
    """One family's full 38-run sweep: both conditions at every size, ascending n."""
    prefix = FAMILIES[family][0]
    return [f"{prefix}-{cond}-n{n}" for n in SIZES for cond in CONDITIONS]


def _parse_run_id(run_id: str) -> tuple[str, str]:
    """``(condition, curve_label)`` parsed from a fig2 sweep run id.

    Parsed from the run id (not ``manifest.regime``, which is the closed
    elicit/teach/unknown enum and has no base/format-installed distinction) —
    this is the one place that naming convention is assumed.
    """
    m = RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(
            f"{run_id!r} does not match either sweep run_id pattern "
            "'evt-llama-fig2{,nl}-{noinst,inst}-n<size>'"
        )
    condition = m.group(1)
    return condition, CONDITIONS[condition]


def run_rows(run_id: str, store: Path) -> list[dict] | None:
    """One run's long-format rows, or ``None`` if the run must be skipped.

    Skips (with a warning) when the manifest cannot be loaded at all, or
    loads but has no ``experiment.target_result`` (not yet finalized) — both
    mean "nothing to plot for this run yet", not a crash.
    """
    condition, curve_label = _parse_run_id(run_id)
    try:
        manifest = load_run(run_id, store=store)
    except (FileNotFoundError, ManifestError) as exc:
        print(f"[evt] {run_id}: manifest not available ({exc}) — skipped")
        return None

    target_result = manifest.data.get("experiment", {}).get("target_result")
    if target_result is None:
        print(
            f"[evt] WARNING: {run_id}: manifest has no experiment.target_result "
            "(run not finalized yet) — skipped"
        )
        return None

    stop_reason = target_result.get("stop_reason")
    if stop_reason != "converged":
        print(
            f"[evt] WARNING: {run_id}: stop_reason={stop_reason!r} (not 'converged') — "
            "plotted anyway, but max_steps is a stopping-rule bug signal here"
        )

    base = {
        "run_id": run_id,
        "base_model_key": manifest.data["base_model"]["hf_id"],
        "regime": manifest.data["regime"],
        "dataset_size": manifest.data["dataset"]["n_unique_examples"],
        "checkpoint_step": target_result.get("final_step"),
        "layer": GLOBAL_LAYER,
        "condition": condition,
        "curve_label": curve_label,
        "stop_reason": stop_reason,
    }

    rows: list[dict] = []
    for metric_name in TARGET_RESULT_METRICS:
        value = target_result.get(metric_name)
        if value is None:  # not written yet, or genuinely undefined: never fabricate
            continue
        rows.append({**base, "metric_name": metric_name, "metric_value": float(value)})

    try:
        tl = test_loss(run_id, store=store)
        rows.append(
            {
                **base,
                "metric_name": "test_loss_per_label_token_nats",
                "metric_value": tl.loss_per_label_token_nats,
            }
        )
    except FileNotFoundError:
        print(f"[evt] {run_id}: no eval/test_loss.json — test-loss metric skipped")

    g5 = manifest.data.get("experiment", {}).get("gates", {}).get("G5")
    zero_shot = g5.get("zero_shot_accuracy") if g5 else None
    if zero_shot is None:
        print(f"[evt] {run_id}: no G5 zero-shot EM recorded — metric skipped")
    else:
        rows.append({**base, "metric_name": "g5_zero_shot_em", "metric_value": float(zero_shot)})

    print(
        f"[evt] {run_id}: n={base['dataset_size']:,} ({condition}), stop={stop_reason}, "
        f"{len(rows)} metric row(s)"
    )
    return rows


def plot(df: pd.DataFrame, out: Path, family_tag: str = FAMILIES[DEFAULT_FAMILY][2]) -> None:
    """EDL/D (bits/label token) vs. dataset size, log-x, one curve per condition."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"noinst": "tab:blue", "inst": "tab:orange"}

    headline = df[df["metric_name"] == HEADLINE_METRIC]
    for condition, by_cond in headline.groupby("condition", sort=True):
        by_cond = by_cond.sort_values("dataset_size")
        curve_label = by_cond["curve_label"].iloc[0]
        color = colors.get(condition, "gray")
        y = by_cond["metric_value"] / LN2  # nats -> bits at the reporting boundary
        converged = by_cond["stop_reason"] == "converged"
        ax.plot(by_cond["dataset_size"], y, color=color, lw=1.2, alpha=0.6, zorder=1)
        ax.scatter(
            by_cond.loc[converged, "dataset_size"],
            y[converged],
            color=color,
            marker="o",
            s=36,
            label=curve_label,
            zorder=2,
        )
        if (~converged).any():
            ax.scatter(
                by_cond.loc[~converged, "dataset_size"],
                y[~converged],
                facecolors="none",
                edgecolors="red",
                marker="o",
                s=60,
                linewidths=1.5,
                label=f"{curve_label} (NOT converged)",
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("EDL/D (bits per label token; D = training label tokens)")
    ax.set_title(f"Figure 2 sweep: EDL/D vs. dataset size (Llama-3.2-1B; {family_tag})")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--family",
        choices=tuple(FAMILIES),
        default=DEFAULT_FAMILY,
        help="which sweep to read: 'op' = operator notation on D_target "
        "(§6.10, the default and the shipped one), 'nl' = natural language on "
        "D_algo (§6.11). Selects the run-id prefix AND the output stem",
    )
    ap.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="repeatable; default: the chosen --family's full 38-run sweep",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=None,
        help="figure path (default: figures/<stem>.png for the chosen --family)",
    )
    args = ap.parse_args()
    stem, family_tag = FAMILIES[args.family][1:]
    run_ids = args.run_ids or default_run_ids(args.family)
    fig_path = args.fig or Path(__file__).resolve().parent / "figures" / f"{stem}.png"

    rows: list[dict] = []
    n_found = 0
    for rid in run_ids:
        run_result = run_rows(rid, args.store)
        if run_result is None:
            continue
        n_found += 1
        rows.extend(run_result)

    if not rows:
        raise SystemExit(
            "dataset_size_sweep: no run produced any rows — nothing to plot "
            f"(checked {len(run_ids)} run id(s) under {args.store})"
        )

    df = pd.DataFrame(rows)
    path = write_results(df, stem, store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows, {n_found}/{len(run_ids)} runs found)")
    plot(df, fig_path, family_tag)


if __name__ == "__main__":
    main()
