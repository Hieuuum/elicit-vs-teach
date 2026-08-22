"""ts38mt mechanistic-test summary: one row per (arm, n) over the Tier-1/2
grid outputs, plus one row per checkpoint for the Phase-0 parent sweep.

The ten drivers each write one table per run (``handoff-ts38mt-launch.md``
§5b, ``scripts/run_ts38mt_grid_tier{1,2}.sh``); this script folds those
tables into the headline number each driver's own ``print_summary`` already
reports, so the decisions.md Outcome can quote a 3-arm × 10-size table per
test instead of 150 files. Every per-run helper below is a pure function of
one driver's table (unit-tested on synthetic tables); the assembly step only
locates files and tolerates partial grids (missing files ⇒ NaN columns, and
the missing file names are returned and printed — never silently skipped).

Per-run columns (prefix = test):

- ``probe_*`` (test 1, ``resid_probe_<run>.csv``, ``task`` set only): the
  ``resid_probe.layer_summary`` verdict at the FIRST and FINAL snapshot
  (``probe_step_first/final``, ``probe_margin_first/final``,
  ``probe_best_layer_first/final``), ``probe_margin_max`` across snapshots,
  and ``probe_margin_half_step_frac`` — the snapshot-step fraction
  (``step / final step``) at which the margin first reaches half of its
  first→final rise (NaN when the margin does not rise).
- ``gd_*`` (test 8): the six pre-registered ``grad_dynamics`` run-level
  metrics + ``n_steps``, from ``grad_dynamics_all.csv`` (one combined file,
  the grid scripts' layout) or ``grad_dynamics_<run>.csv`` (§5b's per-run
  layout), whichever exists.
- ``wd_*`` (test 9): ``weight_diff.summary_metrics`` — total ``rel_fro``,
  ΔW-mass-weighted ``effective_rank`` and ``overlap_<k>``.
- ``shift_*`` (test 10): the layer of peak ``task`` ``rel_shift``, the task
  and generic shifts there, their ratio (NaN when generic is 0), and the
  task-set direction-consistency metrics at that layer.
- ``jac_*`` (test 7, bridge rows only): ``actual_gain_nats``, the layer with
  the highest ``mean_cos_shift_vs_jac0`` and the cosine + ``pred_gain_ratio``
  there, and the max ``pred_gain_ratio`` over layers.
- ``cp_*`` (test 4): per (direction, scope) the best-recovery layer and
  value (``T_into_0``) or the worst (``0_into_T``, the damage direction),
  and for ``T_into_0`` the earliest layer whose ``recovery_frac`` ≥ 0.5.

Usage:
    python3 ts38mt_mech_summary.py --out ts38mt_mech_summary.csv \\
        --phase0-dir $GEODE_STORE/results/ts38mt_phase0 --phase0-out ts38mt_phase0_summary.csv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from logit_lens import emergence_layer
from mech_lib import write_table
from resid_probe import summary_table as probe_summary_table
from weight_diff import summary_metrics as weight_diff_summary

ARMS: tuple[str, ...] = ("base", "pp", "fmt")
SIZES: tuple[int, ...] = (1000, 2154, 4642, 10000, 21544, 46416, 100000, 146780, 215443, 316228)
PHASE0_MODELS: tuple[str, ...] = ("theta0", "s7773", "s15546", "s23319", "thetaT")
GD_METRICS: tuple[str, ...] = (
    "n_steps",
    "grad_early_mass_frac",
    "grad_peak_ratio",
    "grad_half_step_frac",
    "disp_frac_at_10pct",
    "disp_half_step_frac",
    "loss_half_step_frac",
)
_CP_KEYS: tuple[tuple[str, str, str], ...] = (
    ("T_into_0", "answer", "cp_T0_answer"),
    ("T_into_0", "all", "cp_T0_all"),
    ("0_into_T", "answer", "cp_0T_answer"),
    ("0_into_T", "all", "cp_0T_all"),
)


def run_id(arm: str, n: int) -> str:
    return f"evt-ts38mt-{arm}-n{n}"


def _nan_dict(keys: Sequence[str]) -> dict:
    return {k: math.nan for k in keys}


# ---------------------------------------------------------------- test 1


def probe_run_summary(df: pd.DataFrame) -> dict:
    """First/final/max probe margin over a run's snapshot sweep (``task`` set).
    Raises (via ``resid_probe.layer_summary``) when a curve lacks layer 0."""
    task = df[df["set"] == "task"]
    if task.empty:
        raise ValueError("probe_run_summary: no 'task' rows")
    summ = probe_summary_table(task.to_dict("records")).sort_values("checkpoint_step")
    first, final = summ.iloc[0], summ.iloc[-1]
    m0, m1 = float(first["probe_margin_over_floor"]), float(final["probe_margin_over_floor"])
    half_frac = math.nan
    if m1 > m0:
        target = m0 + 0.5 * (m1 - m0)
        hit = summ[summ["probe_margin_over_floor"] >= target].iloc[0]
        half_frac = float(hit["checkpoint_step"]) / float(final["checkpoint_step"])
    return {
        "probe_step_first": int(first["checkpoint_step"]),
        "probe_margin_first": m0,
        "probe_best_layer_first": int(first["best_layer"]),
        "probe_step_final": int(final["checkpoint_step"]),
        "probe_margin_final": m1,
        "probe_best_layer_final": int(final["best_layer"]),
        "probe_floor": float(final["layer0_floor"]),
        "probe_margin_max": float(summ["probe_margin_over_floor"].max()),
        "probe_margin_half_step_frac": half_frac,
    }


# ---------------------------------------------------------------- test 8


def grad_run_summary(df: pd.DataFrame, rid: str) -> dict:
    """The ``level="run"`` row for ``rid`` from a ``grad_dynamics`` table."""
    row = df[(df["level"] == "run") & (df["run_id"] == rid)]
    if row.empty:
        raise ValueError(f"grad_run_summary: no level='run' row for {rid!r}")
    r = row.iloc[0]
    return {f"gd_{k}": float(r[k]) for k in GD_METRICS}


# ---------------------------------------------------------------- test 9


def weight_run_summary(df: pd.DataFrame) -> dict:
    return {f"wd_{k}": v for k, v in weight_diff_summary(df).items()}


# ---------------------------------------------------------------- test 10

SHIFT_KEYS = (
    "shift_peak_layer",
    "shift_task_at_peak",
    "shift_generic_at_peak",
    "shift_ratio_at_peak",
    "shift_task_cos_at_peak",
    "shift_task_evr_at_peak",
)


def shift_run_summary(df: pd.DataFrame) -> dict:
    task = df[df["set"] == "task"].set_index("layer").sort_index()
    generic = df[df["set"] == "generic"].set_index("layer")
    if task.empty:
        raise ValueError("shift_run_summary: no 'task' rows")
    peak = int(task["rel_shift"].idxmax())
    t = float(task.loc[peak, "rel_shift"])
    g = float(generic.loc[peak, "rel_shift"]) if peak in generic.index else math.nan
    return {
        "shift_peak_layer": peak,
        "shift_task_at_peak": t,
        "shift_generic_at_peak": g,
        "shift_ratio_at_peak": t / g if g > 0 else math.nan,
        "shift_task_cos_at_peak": float(task.loc[peak, "mean_cos_to_mean"]),
        "shift_task_evr_at_peak": float(task.loc[peak, "top_pc_evr"]),
    }


# ---------------------------------------------------------------- test 7

JAC_KEYS = (
    "jac_actual_gain_nats",
    "jac_best_cos_layer",
    "jac_best_cos",
    "jac_pred_gain_ratio_at_best_cos",
    "jac_pred_gain_ratio_max",
)


def jacobian_run_summary(df: pd.DataFrame) -> dict:
    """Bridge-row readout; all-NaN (not an error) when the table has no
    ``--model-b`` bridge rows."""
    if "mean_cos_shift_vs_jac0" not in df:
        return _nan_dict(JAC_KEYS)
    a = df.dropna(subset=["mean_cos_shift_vs_jac0"]).sort_values("layer")
    if a.empty:
        return _nan_dict(JAC_KEYS)
    best = a.loc[a["mean_cos_shift_vs_jac0"].idxmax()]
    return {
        "jac_actual_gain_nats": float(a["actual_gain_nats"].iloc[0]),
        "jac_best_cos_layer": int(best["layer"]),
        "jac_best_cos": float(best["mean_cos_shift_vs_jac0"]),
        "jac_pred_gain_ratio_at_best_cos": float(best["pred_gain_ratio"]),
        "jac_pred_gain_ratio_max": float(a["pred_gain_ratio"].max()),
    }


# ---------------------------------------------------------------- test 4


def _cp_keys() -> list[str]:
    keys = []
    for direction, _scope, key in _CP_KEYS:
        if direction == "T_into_0":
            keys += [f"{key}_best_layer", f"{key}_best_rec", f"{key}_first_layer_ge_half"]
        else:
            keys += [f"{key}_min_layer", f"{key}_min_rec"]
    return keys


CP_KEYS = tuple(_cp_keys())


def cross_patch_run_summary(df: pd.DataFrame) -> dict:
    out = _nan_dict(CP_KEYS)
    for direction, scope, key in _CP_KEYS:
        sub = df[(df["direction"] == direction) & (df["scope"] == scope)].dropna(
            subset=["recovery_frac"]
        )
        if sub.empty:
            continue
        if direction == "T_into_0":
            best = sub.loc[sub["recovery_frac"].idxmax()]
            out[f"{key}_best_layer"] = int(best["layer"])
            out[f"{key}_best_rec"] = float(best["recovery_frac"])
            ge_half = sub[sub["recovery_frac"] >= 0.5].sort_values("layer")
            if not ge_half.empty:
                out[f"{key}_first_layer_ge_half"] = int(ge_half["layer"].iloc[0])
        else:
            worst = sub.loc[sub["recovery_frac"].idxmin()]
            out[f"{key}_min_layer"] = int(worst["layer"])
            out[f"{key}_min_rec"] = float(worst["recovery_frac"])
    return out


# ---------------------------------------------------------------- assembly


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _grad_table(results_dir: Path, rid: str) -> Path | None:
    combined = results_dir / "grad_dynamics_all.csv"
    per_run = results_dir / f"grad_dynamics_{rid}.csv"
    if per_run.is_file():
        return per_run
    if combined.is_file():
        return combined
    return None


def summarize_grid(
    results_dir: Path,
    arms: Sequence[str] = ARMS,
    sizes: Sequence[int] = SIZES,
) -> tuple[pd.DataFrame, list[str]]:
    """One row per (n, arm) in that order; ``missing`` lists every expected
    table that was not found (its columns are NaN in the row)."""
    rows: list[dict] = []
    missing: list[str] = []
    for n in sizes:
        for arm in arms:
            rid = run_id(arm, n)
            row: dict = {"arm": arm, "n": n, "run_id": rid}
            specs = [
                (results_dir / f"resid_probe_{rid}.csv", probe_run_summary),
                (results_dir / f"weight_diff_{rid}.parquet", weight_run_summary),
                (results_dir / f"resid_shift_{rid}.csv", shift_run_summary),
                (results_dir / f"jacobian_lens_{rid}.csv", jacobian_run_summary),
                (results_dir / f"cross_patch_{rid}.csv", cross_patch_run_summary),
            ]
            for path, fn in specs:
                if path.is_file():
                    row.update(fn(_read(path)))
                else:
                    missing.append(path.name)
            gd = _grad_table(results_dir, rid)
            if gd is None:
                missing.append(f"grad_dynamics_{rid}.csv")
            else:
                try:
                    row.update(grad_run_summary(_read(gd), rid))
                except ValueError:
                    missing.append(f"grad_dynamics_{rid}.csv")
            rows.append(row)
    return pd.DataFrame(rows), missing


def summarize_phase0(
    phase0_dir: Path, models: Sequence[str] = PHASE0_MODELS
) -> tuple[pd.DataFrame, list[str]]:
    """One row per Phase-0 checkpoint (``run_ts38mt_phase0.sh`` layout):
    probe margin on ``task`` and ``op``, logit-lens emergence layer on both,
    and the weight/shift/Jacobian readouts (absent for ``theta0`` by design,
    reported as missing only for the others)."""
    rows: list[dict] = []
    missing: list[str] = []
    for m in models:
        row: dict = {"model": m}
        p = phase0_dir / f"resid_probe_{m}.csv"
        if p.is_file():
            summ = probe_summary_table(_read(p).to_dict("records"))
            for s, sub in summ.groupby("set"):
                r = sub.iloc[0]
                row[f"probe_{s}_floor"] = float(r["layer0_floor"])
                row[f"probe_{s}_best_layer"] = int(r["best_layer"])
                row[f"probe_{s}_best_acc"] = float(r["best_acc"])
                row[f"probe_{s}_margin"] = float(r["probe_margin_over_floor"])
        else:
            missing.append(p.name)
        for s in ("task", "op"):
            p = phase0_dir / f"logit_lens_{m}_{s}.csv"
            if p.is_file():
                d = _read(p)
                row[f"lens_{s}_emergence_layer"] = emergence_layer(d.to_dict("records"))
                fa = d[d["position_kind"] == "first_answer"].sort_values("layer")
                row[f"lens_{s}_final_top1"] = float(fa["top1_acc"].iloc[-1])
            else:
                missing.append(p.name)
        if m != "theta0":
            specs = [
                (phase0_dir / f"weight_diff_{m}.parquet", weight_run_summary),
                (phase0_dir / f"resid_shift_{m}.csv", shift_run_summary),
                (phase0_dir / f"jacobian_lens_{m}.csv", jacobian_run_summary),
            ]
            for path, fn in specs:
                if path.is_file():
                    row.update(fn(_read(path)))
                else:
                    missing.append(path.name)
        rows.append(row)
    return pd.DataFrame(rows), missing


HEADLINE_COLS: tuple[str, ...] = (
    "probe_margin_first",
    "probe_margin_final",
    "probe_best_layer_final",
    "gd_grad_early_mass_frac",
    "gd_grad_half_step_frac",
    "gd_disp_frac_at_10pct",
    "gd_disp_half_step_frac",
    "wd_rel_fro",
    "wd_effective_rank",
    "wd_overlap_32",
    "shift_peak_layer",
    "shift_ratio_at_peak",
    "shift_task_cos_at_peak",
    "jac_best_cos",
    "jac_pred_gain_ratio_max",
    "cp_T0_answer_best_layer",
    "cp_T0_answer_best_rec",
    "cp_T0_all_first_layer_ge_half",
)


def print_grid(df: pd.DataFrame, cols: Sequence[str] = HEADLINE_COLS) -> None:
    """One arm × n pivot per headline column, ``[evt]``-prefixed."""
    for col in cols:
        if col not in df or df[col].isna().all():
            continue
        pivot = df.pivot(index="arm", columns="n", values=col).reindex(
            [a for a in ARMS if a in set(df["arm"])]
        )
        print(f"[evt] {col}")
        print(
            "[evt] " + pivot.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n[evt] ")
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="grid tables (default $GEODE_STORE/results/ts38mt_mech)",
    )
    ap.add_argument("--sizes", default=None, help="comma-separated subset of the grid")
    ap.add_argument("--arms", default=None, help="comma-separated subset of base,pp,fmt")
    ap.add_argument("--out", type=Path, default=None, help="grid summary table (csv/parquet)")
    ap.add_argument("--phase0-dir", type=Path, default=None, help="Phase-0 tables (optional)")
    ap.add_argument("--phase0-out", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        store = os.environ.get("GEODE_STORE")
        if store is None:
            raise SystemExit("--results-dir or $GEODE_STORE required")
        results_dir = Path(store) / "results" / "ts38mt_mech"
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(SIZES)
    arms = args.arms.split(",") if args.arms else list(ARMS)

    grid, missing = summarize_grid(results_dir, arms, sizes)
    print(f"[evt] grid: {len(grid)} cells, {len(missing)} missing tables")
    for name in missing:
        print(f"[evt]   missing: {name}")
    print_grid(grid)
    if args.out is not None:
        write_table(grid, args.out)
        print(f"[evt] wrote {args.out} ({len(grid)} rows)")

    if args.phase0_dir is not None:
        p0, p0_missing = summarize_phase0(args.phase0_dir)
        for name in p0_missing:
            print(f"[evt]   phase0 missing: {name}")
        print("[evt] phase0\n[evt] " + p0.to_string(index=False).replace("\n", "\n[evt] "))
        if args.phase0_out is not None:
            write_table(p0, args.phase0_out)
            print(f"[evt] wrote {args.phase0_out} ({len(p0)} rows)")


if __name__ == "__main__":
    main()
