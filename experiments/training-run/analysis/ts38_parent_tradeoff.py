"""ts38 pre-taught parent: does run-until-convergence cost G8, or is the
G1/G8 tradeoff intrinsic to the arithmetic install itself?

Every ts38 parent trained so far (full-FT ladder, decisions.md 2026-08-15
"ts38 ladder CLOSED"; LoRA sweep + install, decisions.md 2026-08-15 "ts38
LoRA-installed parent") passes G1 (exact-match >= 0.95 on D_algo val) only
after G8 (TinyStories retention, mean next-token loss on run 1's frozen val
stream <= base 1.0718 + 0.10 = 1.1718 nats) has already failed. The eps/k
convergence rule that stops every run only ever fires AFTER the arithmetic
val curve has bottomed out and re-risen a little, so it is plausible that an
earlier, non-converged checkpoint would have cleared G1 while G8 was still
under the bar. This script replays the recorded logs (no GPU, no new
checkpoints) to quantify that window, or show there isn't one.

Two things are ESTIMATED, not measured, and are labeled as such everywhere
they appear: G1 is only ever measured at each run's converged final
checkpoint, so "the step G1 first clears 0.95" is read off the ARITHMETIC
val curve (eval_log.jsonl, evaluated every 1000 steps during training) via a
G1-vs-val map fit across all 11 measured (val, G1) points, not from a direct
G1(step) curve (no intermediate checkpoints were saved, decisions.md
2026-08-15: "train_sft.py saves the final checkpoint only, spec 02 SS6").
G8 is measured at even fewer points per run (1 or 2), so G8(step) is a
2-point extrapolation (linear or power-law in step), never a measured curve.

Ground-truth (step, G1, G8) table: hardcoded below from decisions.md
2026-08-15 entries "ts38 parent ladder HALT at 1e-5", "ts38 ladder CLOSED",
and "ts38 LoRA-installed parent" (the sweep table + the two full-run rows),
cross-checked against the pulled ladder/sweep JSONs and run manifests -- see
`cross_check()`. No disagreements were found between the hardcoded table and
the store's JSONs/manifests (2026-08-15 pull); rounding-only differences are
noted inline.

Item-4 note: the task brief's example list of "single-point" G8-vs-step runs
names four (full-FT 3e-4/1e-4/3e-5, LoRA 1e-3). A fifth run in the pulled set
also has exactly one G8 measurement and is not part of any 2-point family:
the LoRA 3e-5 sweep rung (G8=1.0756 @8000). It is modeled under the identical
single-point rule (anchored at (0, 1.0718)) for consistency; flagged here
since the brief did not name it explicitly.

CPU-only, reads the local store, no network, no torch.

Usage:
    python3 analysis/ts38_parent_tradeoff.py [--store DIR] [--out-dir DIR] [--fig PATH]
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
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/training-run/analysis"

G8_BASE = 1.0718  # evt-run1-base-v3-ext TinyStories converged val loss (decisions.md :866)
G8_BAR = 1.1718  # G8_BASE + ratified delta 0.10 (decisions.md 2026-08-14 "G8 delta ratified")
G1_BAR = 0.95

# --- Ground-truth (step, G1, G8) table --------------------------------------
# Hardcoded from the task brief / decisions.md 2026-08-15 entries "ts38 parent
# ladder HALT at 1e-5", "ts38 ladder CLOSED: 1e-5 converged", and "ts38
# LoRA-installed parent". `final_step` is the run's OWN stopping step (theta_T);
# g1/g8 were scored on that checkpoint, never on the min-val checkpoint.
# `family` groups runs that share a G8-vs-step model: a `pair` has 2 measured
# G8 points at different steps of nominally the same (method, lr) install; a
# `single` has exactly 1 and is modeled through the anchor (0, G8_BASE).
RUN_SPECS = [
    dict(
        run="full_ft_lr3.0e-4",
        dir="runs-failed/evt-ts38-pretaught-parent-lr3.0e-4",
        method="full_ft",
        lr=3.0e-4,
        final_step=21000,
        g1=0.9883,
        g8=9.9579,
        family="single",
    ),
    dict(
        run="full_ft_lr1.0e-4",
        dir="runs-failed/evt-ts38-pretaught-parent-lr1.0e-4",
        method="full_ft",
        lr=1.0e-4,
        final_step=25000,
        g1=0.9863,
        g8=3.5983,
        family="single",
    ),
    dict(
        run="full_ft_lr3.0e-5",
        dir="runs-failed/evt-ts38-pretaught-parent-lr3.0e-5",
        method="full_ft",
        lr=3.0e-5,
        final_step=40000,
        g1=0.9785,
        g8=1.2074,
        family="single",
    ),
    dict(
        run="full_ft_lr1.0e-5_ceil40k",
        dir="runs-failed/evt-ts38-pretaught-parent-lr1.0e-5-ceil40k",
        method="full_ft",
        lr=1.0e-5,
        final_step=40000,
        g1=0.8809,
        g8=1.1431,
        family="pair:full_ft_1e-5",
    ),
    dict(
        run="full_ft_lr1.0e-5_g1fail",
        dir="runs-failed/evt-ts38-pretaught-parent-lr1.0e-5-g1fail",
        method="full_ft",
        lr=1.0e-5,
        final_step=68000,
        g1=0.9404,
        g8=1.1904,
        family="pair:full_ft_1e-5",
    ),
    dict(
        run="lora_sweep_lr1e-3",
        dir="runs/evt-ts38-parent-lorasweep-lr1e-3",
        method="lora",
        lr=1.0e-3,
        final_step=8000,
        g1=0.9473,
        g8=1.2549,
        family="single",
    ),
    dict(
        run="lora_sweep_lr3e-4",
        dir="runs/evt-ts38-parent-lorasweep-lr3e-4",
        method="lora",
        lr=3.0e-4,
        final_step=8000,
        g1=0.8672,
        g8=1.1379,
        family="pair:lora_3e-4",
    ),
    dict(
        run="lora_sweep_lr1e-4",
        dir="runs/evt-ts38-parent-lorasweep-lr1e-4",
        method="lora",
        lr=1.0e-4,
        final_step=8000,
        g1=0.3828,
        g8=1.0842,
        family="pair:lora_1e-4",
    ),
    dict(
        run="lora_sweep_lr3e-5",
        dir="runs/evt-ts38-parent-lorasweep-lr3e-5",
        method="lora",
        lr=3.0e-5,
        final_step=8000,
        g1=0.0605,
        g8=1.0756,
        family="single",
    ),
    dict(
        run="lora_parent_lr3e-4",
        dir="runs-failed/evt-ts38-pretaught-parent-lora-lr3e-4",
        method="lora",
        lr=3.0e-4,
        final_step=24000,
        g1=0.9775,
        g8=1.1855,
        family="pair:lora_3e-4",
    ),
    dict(
        run="lora_parent_lr1e-4",
        dir="runs-failed/evt-ts38-pretaught-parent-lora-lr1e-4",
        method="lora",
        lr=1.0e-4,
        final_step=55000,
        g1=0.9658,
        g8=1.1994,
        family="pair:lora_1e-4",
    ),
]

# Anchor point, not a run: the untrained base's premise (~0 EM) and its own
# TinyStories retention floor. Used only to anchor single-point G8 models at
# step 0; excluded from the G1-vs-val map (no comparable arithmetic val
# number was pulled for it -- ts38_step0_baseline.json is scored on a
# different eval, arith_bare_addsub, not the op-notation arith_op_addsub val
# split the ladder/sweep curves and G1 gate use).
BASE_ROW = dict(
    run="base_evt-run1-base-v3-ext", method="-", lr=None, final_step=0, g1=0.0, g8=G8_BASE
)


def load_eval_log(store: Path, rel_dir: str) -> list[tuple[int, float]]:
    path = store / rel_dir / "eval_log.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    rows.sort(key=lambda r: r["step"])
    return [(r["step"], r["val_loss_nats"]) for r in rows]


def load_stop_reason(store: Path, rel_dir: str) -> str:
    return json.loads((store / rel_dir / "training_meta.json").read_text())["stop_reason"]


def cross_check(store: Path) -> list[str]:
    """Compare RUN_SPECS' method/lr/final_step/g1/g8 against manifest.json /
    training_meta.json where present. Returns a list of disagreement strings
    (empty if none)."""
    problems = []
    for spec in RUN_SPECS:
        d = store / spec["dir"]
        manifest = json.loads((d / "manifest.json").read_text())
        meta = json.loads((d / "training_meta.json").read_text())
        t = manifest["training"]
        if t["method"] != spec["method"]:
            problems.append(
                f"{spec['run']}: manifest method={t['method']!r} != table {spec['method']!r}"
            )
        manifest_lr = t["optimizer"]["lr"]
        if abs(manifest_lr - spec["lr"]) > 1e-12:
            problems.append(f"{spec['run']}: manifest lr={manifest_lr!r} != table {spec['lr']!r}")
        if meta["final_step"] != spec["final_step"]:
            problems.append(
                f"{spec['run']}: training_meta final_step={meta['final_step']} != table {spec['final_step']}"
            )
        gates = manifest.get("experiment", {}).get("gates", {})
        if "G1" in gates:
            manifest_g1 = round(gates["G1"]["accuracy"], 4)
            if abs(manifest_g1 - spec["g1"]) > 5e-5:
                problems.append(f"{spec['run']}: manifest G1={manifest_g1} != table {spec['g1']}")
    # ladder.json / lora_sweep.json cross-check
    ladder = json.loads((store / "results/ts38_parent_ladder.json").read_text())
    ladder_by_lr = {row["lr"]: row for row in ladder}
    sweep = json.loads((store / "results/ts38_parent_lora_sweep.json").read_text())
    sweep_by_lr = {row["lr"]: row for row in sweep}
    checks = [
        ("full_ft_lr3.0e-4", ladder_by_lr.get("3.0e-4")),
        ("full_ft_lr1.0e-4", ladder_by_lr.get("1.0e-4")),
        ("full_ft_lr3.0e-5", ladder_by_lr.get("3.0e-5")),
        ("full_ft_lr1.0e-5_g1fail", ladder_by_lr.get("1.0e-5")),
        ("lora_sweep_lr1e-3", sweep_by_lr.get("1e-3")),
        ("lora_sweep_lr3e-4", sweep_by_lr.get("3e-4")),
        ("lora_sweep_lr1e-4", sweep_by_lr.get("1e-4")),
        ("lora_sweep_lr3e-5", sweep_by_lr.get("3e-5")),
    ]
    spec_by_run = {s["run"]: s for s in RUN_SPECS}
    for run, row in checks:
        if row is None:
            problems.append(f"{run}: not found in ladder/sweep JSON")
            continue
        spec = spec_by_run[run]
        if abs(row["g1_accuracy"] - spec["g1"]) > 5e-5:
            problems.append(f"{run}: json g1={row['g1_accuracy']} != table {spec['g1']}")
        if abs(row["g8_val_loss_nats"] - spec["g8"]) > 5e-5:
            problems.append(f"{run}: json g8={row['g8_val_loss_nats']} != table {spec['g8']}")
    return problems


# --- Isotonic regression (pool-adjacent-violators, implemented in numpy) ----


def pav_increasing(y: np.ndarray) -> np.ndarray:
    """Isotonic (non-decreasing) fit to y, equal unit weights. Classic PAVA
    via a stack of pooled blocks (value, weight, count)."""
    blocks: list[list[float]] = []  # [mean, weight, count]
    for yi in y:
        blocks.append([float(yi), 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            w = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / w, w, c1 + c2])
    out = []
    for v, _w, c in blocks:
        out.extend([v] * c)
    return np.array(out)


def g1_vs_val_map(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """points: list of (val_at_final_step, g1), any order. Returns
    (val_sorted_ascending, isotonic_g1_fit, val_star) where val_star is the
    val at which the isotonic map crosses G1_BAR (linear interpolation
    between the two fitted points bracketing it)."""
    pts = sorted(points, key=lambda p: p[0])
    vals = np.array([p[0] for p in pts])
    g1s = np.array([p[1] for p in pts])
    # G1 should be non-increasing in val -> fit isotonic non-decreasing to -g1.
    fit = -pav_increasing(-g1s)
    val_star = interp_crossing(vals, fit, G1_BAR)
    return vals, fit, val_star


def interp_crossing(xs: np.ndarray, ys: np.ndarray, target: float) -> float:
    """xs ascending, ys monotonic non-increasing in xs. Linear-interpolate the
    x at which ys crosses `target`. Raises if target is out of range."""
    if target > ys[0] or target < ys[-1]:
        raise ValueError(f"target {target} outside fitted range [{ys[-1]}, {ys[0]}]")
    for i in range(len(xs) - 1):
        if ys[i] >= target >= ys[i + 1]:
            if ys[i] == ys[i + 1]:
                return float(xs[i])
            frac = (target - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + frac * (xs[i + 1] - xs[i]))
    return float(xs[-1])


def nearest_bracket(points: list[tuple[float, float]], target: float) -> tuple[float, float]:
    """Plain linear-interpolation bracket: the two measured points whose G1
    values most tightly bracket `target` (one just below, one just above),
    interpolated in (val, G1) space. Returns val* from this bracket alone."""
    below = [(v, g) for v, g in points if g <= target]
    above = [(v, g) for v, g in points if g >= target]
    lo = max(below, key=lambda p: p[1])  # largest g1 <= target
    hi = min(above, key=lambda p: p[1])  # smallest g1 >= target
    if lo[1] == hi[1]:
        return lo[0]
    frac = (target - lo[1]) / (hi[1] - lo[1])
    return lo[0] + frac * (hi[0] - lo[0])


# --- S_G1: first / persistent crossing of val* -------------------------------


def s_g1(curve: list[tuple[int, float]], val_star: float) -> tuple[int | None, int | None]:
    """curve: [(step, val), ...] ascending by step. Returns (S_G1_first,
    S_G1_persist): first step with val <= val_star, and first step where val
    <= val_star at that step AND at the next recorded eval (2 consecutive)."""
    first = None
    persist = None
    for i, (step, val) in enumerate(curve):
        if val <= val_star:
            if first is None:
                first = step
            if persist is None and i + 1 < len(curve) and curve[i + 1][1] <= val_star:
                persist = step
    return first, persist


# --- G8-vs-step models --------------------------------------------------------


class G8Model:
    """excess(step) = c * step**p, G8(step) = G8_BASE + excess(step)."""

    def __init__(self, kind: str, c: float, p: float):
        self.kind = kind  # "linear-2pt" | "power" | "linear-anchor" | "power-p0.5"
        self.c = c
        self.p = p

    def g8_at(self, step: float) -> float:
        return G8_BASE + self.c * step**self.p

    def step_at_g8(self, g8_target: float) -> float:
        excess = g8_target - G8_BASE
        return (excess / self.c) ** (1.0 / self.p)


class G8LinearTwoPoint:
    """Full 2-point line through (step1, g8_1), (step2, g8_2) -- NOT forced
    through the (0, G8_BASE) anchor. Used for `pair` families only."""

    kind = "linear-2pt"

    def __init__(self, step1: float, g8_1: float, step2: float, g8_2: float):
        self.slope = (g8_2 - g8_1) / (step2 - step1)
        self.intercept = g8_1 - self.slope * step1

    def g8_at(self, step: float) -> float:
        return self.slope * step + self.intercept

    def step_at_g8(self, g8_target: float) -> float:
        return (g8_target - self.intercept) / self.slope


def fit_pair_models(step1: float, g8_1: float, step2: float, g8_2: float) -> dict[str, object]:
    excess1, excess2 = g8_1 - G8_BASE, g8_2 - G8_BASE
    p = math.log(excess2 / excess1) / math.log(step2 / step1)
    c = excess1 / step1**p
    return {
        "linear": G8LinearTwoPoint(step1, g8_1, step2, g8_2),
        "power": G8Model("power", c, p),
    }


def fit_single_models(step_i: float, g8_i: float) -> dict[str, object]:
    excess_i = g8_i - G8_BASE
    c_linear = excess_i / step_i  # p = 1
    c_power = excess_i / step_i**0.5  # p = 0.5
    return {
        "linear": G8Model("linear-anchor(p=1)", c_linear, 1.0),
        "power": G8Model("power-anchor(p=0.5)", c_power, 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir
    fig_path = args.fig or (out_dir / "figures" / "ts38_parent_tradeoff.png")

    problems = cross_check(args.store)
    if problems:
        print("[evt] GROUND-TRUTH vs STORE disagreements (not auto-fixed):")
        for p in problems:
            print(f"        {p}")
    else:
        print("[evt] ground-truth table matches store JSONs/manifests exactly (no disagreements)")

    curves = {spec["run"]: load_eval_log(args.store, spec["dir"]) for spec in RUN_SPECS}
    for spec in RUN_SPECS:
        spec["stop_reason"] = load_stop_reason(args.store, spec["dir"])

    # --- G1-vs-val map ---
    map_points = []
    for spec in RUN_SPECS:
        val_at_final = dict(curves[spec["run"]])[spec["final_step"]]
        spec["val_at_final_step"] = val_at_final
        spec["min_val"] = min(v for _s, v in curves[spec["run"]])
        map_points.append((val_at_final, spec["g1"]))

    vals_sorted, g1_fit, val_star_map = g1_vs_val_map(map_points)
    val_star_bracket = nearest_bracket(map_points, G1_BAR)
    print(f"\n[evt] val* (isotonic G1-vs-val map, G1={G1_BAR}): {val_star_map:.6f} nats")
    print(
        f"[evt] val* (plain linear-interp bracket of nearest measured points): {val_star_bracket:.6f} nats"
    )
    if not np.isclose(val_star_map, val_star_bracket, atol=1e-9):
        print(
            "        NOTE: map and bracket differ -- isotonic regression pooled a "
            "violator (G1 was NOT already monotone-decreasing in val)."
        )
    else:
        print(
            "        (map == bracket: the 11 measured points were already monotone in val, no PAV pooling)"
        )
    val_star = val_star_map

    # Sensitivity: the pooled bracket above is set by two LoRA points (the
    # nearest measured pair straddling G1=0.95 happens to both be LoRA). A
    # full-FT-only bracket is a useful cross-check since it's an independent
    # subset of the same measured points.
    full_ft_points = [
        (spec["val_at_final_step"], spec["g1"]) for spec in RUN_SPECS if spec["method"] == "full_ft"
    ]
    val_star_full_ft_only = nearest_bracket(full_ft_points, G1_BAR)
    print(f"[evt] val* (full-FT-only bracket, cross-check): {val_star_full_ft_only:.6f} nats")

    # --- per-run S_G1 ---
    for spec in RUN_SPECS:
        first, persist = s_g1(curves[spec["run"]], val_star)
        spec["s_g1_first"] = first
        spec["s_g1_persist"] = persist

    # --- G8 models per family ---
    pair_keys = sorted({s["family"] for s in RUN_SPECS if s["family"].startswith("pair:")})
    family_models: dict[str, dict[str, object]] = {}
    for key in pair_keys:
        members = sorted(
            (s for s in RUN_SPECS if s["family"] == key), key=lambda s: s["final_step"]
        )
        m1, m2 = members
        family_models[key] = fit_pair_models(m1["final_step"], m1["g8"], m2["final_step"], m2["g8"])
    for spec in RUN_SPECS:
        if spec["family"] == "single":
            family_models[f"single:{spec['run']}"] = fit_single_models(
                spec["final_step"], spec["g8"]
            )
            spec["model_key"] = f"single:{spec['run']}"
        else:
            spec["model_key"] = spec["family"]

    # --- per-run G8 estimates ---
    estimate_rows = []
    for spec in RUN_SPECS:
        models = family_models[spec["model_key"]]
        row = dict(
            run=spec["run"],
            method=spec["method"],
            lr=spec["lr"],
            final_step=spec["final_step"],
            val_at_final_step_nats=spec["val_at_final_step"],
            min_val_nats=spec["min_val"],
            g1_measured=spec["g1"],
            g8_measured=spec["g8"],
            g8_model_family=spec["model_key"],
            val_star_nats=val_star,
            s_g1_first=spec["s_g1_first"],
            s_g1_persist=spec["s_g1_persist"],
        )
        # s_g1_status distinguishes WHY val never reached val* within a run's
        # recorded curve -- these are different findings, not the same "None":
        #   reached               -- val* was crossed, S_G1 is a real number.
        #   not_reached_max_steps -- the run was capped (LoRA sweep rungs,
        #     8000-step 1-epoch cut); continuing might still cross val*, and
        #     for the 3e-4/1e-4 sweep rungs the paired full run shows it does.
        #   not_reached_converged -- the eps/k rule fired (plateau) BEFORE val
        #     ever reached val*: this run's own stopping rule says it is
        #     done, and it is still above the G1 threshold. That is evidence
        #     the G1/G8 tradeoff is intrinsic for this arm, not an artifact
        #     of running past an earlier pass point (full-FT 1e-5: min_val
        #     equals the final val, i.e. it was still strictly descending at
        #     convergence yet still never dipped below val*).
        if spec["s_g1_first"] is not None:
            s_g1_status = "reached"
        elif spec["stop_reason"] == "converged":
            s_g1_status = "not_reached_converged"
        else:
            s_g1_status = "not_reached_max_steps"
        row["s_g1_status"] = s_g1_status
        row["stop_reason"] = spec["stop_reason"]
        for label, model in models.items():
            s_g8 = model.step_at_g8(G8_BAR)
            row[f"s_g8_{label}"] = s_g8
            row[f"g8_model_kind_{label}"] = model.kind
            row[f"power_p_{label}"] = getattr(model, "p", None)
            for tag in ("first", "persist"):
                s_g1_val = spec[f"s_g1_{tag}"]
                row[f"g8_at_s_g1_{tag}_{label}"] = (
                    model.g8_at(s_g1_val) if s_g1_val is not None else None
                )
            s1 = spec["s_g1_persist"]
            if s1 is not None and s1 < s_g8:
                row[f"window_{label}"] = f"[{s1}, {s_g8:.0f}]"
                row[f"window_width_{label}"] = s_g8 - s1
                row[f"g8_headroom_at_s_g1_persist_{label}"] = G8_BAR - model.g8_at(s1)
            else:
                row[f"window_{label}"] = "none"
                row[f"window_width_{label}"] = None
                row[f"g8_headroom_at_s_g1_persist_{label}"] = None
        # Catastrophic single-point full-FT runs (G8 > 1.4, the same cutoff
        # panel (c) uses): forgetting this severe is not a linear or sqrt
        # process from (0, G8_BASE) -- these S_G8 / G8_at_S_G1 numbers are
        # NOT meaningful estimates of anything, just what a 2-point line
        # through an implausible functional form says. Flag, don't quote.
        row["not_interpretable"] = spec["family"] == "single" and spec["g8"] > 1.4
        estimate_rows.append(row)

    estimates_df = pd.DataFrame(estimate_rows)

    # --- print report table ---
    print("\n[evt] per-run estimates (S_G8 / G8_at_S_G1 are ESTIMATES from 2-point step models):")
    print(
        "[evt] NOTE on power-law p: the only run where p is FIT TO DATA from that same "
        "method is full_ft_lr1.0e-5 (p=0.96, near-linear, from its own ceil40k/g1fail pair). "
        "The p=0.5 variant reported for full-FT single-point runs is IMPORTED from the LoRA "
        "3e-4 fit (p=0.49), not fit to any full-FT data -- weight the linear model more for "
        "full-FT single-point runs; the two models are not equally well-supported there."
    )
    for row in estimate_rows:
        lr_s = f"{row['lr']:.1e}" if row["lr"] else "-"
        print(
            f"\n  {row['run']:<28s} method={row['method']:<7s} lr={lr_s} final_step={row['final_step']}"
        )
        print(
            f"    val@final={row['val_at_final_step_nats']:.5f} min_val={row['min_val_nats']:.5f} "
            f"g1_measured={row['g1_measured']:.4f} g8_measured={row['g8_measured']:.4f}"
        )
        print(
            f"    S_G1_first={row['s_g1_first']} S_G1_persist={row['s_g1_persist']} "
            f"status={row['s_g1_status']} (stop_reason={row['stop_reason']})"
        )
        if row["not_interpretable"]:
            print(
                "    NOT INTERPRETABLE: G8 > 1.4 (catastrophic), single-point step-model is "
                "not a real estimate -- forgetting this severe is not linear/sqrt from origin."
            )
        for label in ("linear", "power"):
            p_val = row.get(f"power_p_{label}")
            p_s = f" p={p_val:.4f}" if p_val is not None else ""
            print(
                f"    [{label}{p_s} | {row[f'g8_model_kind_{label}']}] "
                f"S_G8={row[f's_g8_{label}']:.0f}  "
                f"G8_at_S_G1_first={row[f'g8_at_s_g1_first_{label}']} "
                f"G8_at_S_G1_persist={row[f'g8_at_s_g1_persist_{label}']}  "
                f"window={row[f'window_{label}']}  "
                f"headroom@persist={row[f'g8_headroom_at_s_g1_persist_{label}']}"
            )

    # --- val* sensitivity: LoRA 3e-4 parent's S_G1_persist=15000 sits within
    # 1.2e-5 nats of a threshold crossing (step 16000's val, 0.0342798, is
    # just under val_star=0.0342916). Show how much S_G1/window/headroom move
    # if val* were picked slightly differently, instead of quoting 15000 as
    # if it had no error bar.
    print(
        "\n[evt] val* SENSITIVITY (lora_parent_lr3e-4, the run whose S_G1_persist=15000 is "
        "closest to a val* threshold flip -- step 16000's val is only 1.2e-5 nats under val*):"
    )
    lora34_models = family_models["pair:lora_3e-4"]
    s_g8_lin_lora34 = lora34_models["linear"].step_at_g8(G8_BAR)
    for vs, tag in (
        (0.0300, "tighter"),
        (val_star_full_ft_only, "full-FT-only bracket"),
        (val_star, "primary (pooled map)"),
        (0.0380, "looser"),
    ):
        f, p = s_g1(curves["lora_parent_lr3e-4"], vs)
        if p is not None:
            g8_at_p = lora34_models["linear"].g8_at(p)
            headroom = G8_BAR - g8_at_p
            window = (
                f"[{p}, {s_g8_lin_lora34:.0f}] width={s_g8_lin_lora34 - p:.0f}"
                if p < s_g8_lin_lora34
                else "none"
            )
        else:
            g8_at_p = headroom = window = None
        print(
            f"    val*={vs:.4f} ({tag}): S_G1_persist={p}  G8_at_S_G1(linear)={g8_at_p}  "
            f"headroom={headroom}  window={window}"
        )

    # --- CSV 1: measured points (11 runs + base anchor) ---
    measured_rows = []
    for spec in RUN_SPECS:
        measured_rows.append(
            dict(
                run=spec["run"],
                method=spec["method"],
                lr=spec["lr"],
                final_step=spec["final_step"],
                val_at_final_step_nats=spec["val_at_final_step"],
                min_val_nats=spec["min_val"],
                g1_accuracy=spec["g1"],
                g8_val_loss_nats=spec["g8"],
                g8_pass=spec["g8"] <= G8_BAR,
                g1_pass=spec["g1"] >= G1_BAR,
            )
        )
    measured_rows.append(
        dict(
            run=BASE_ROW["run"],
            method=BASE_ROW["method"],
            lr=None,
            final_step=0,
            val_at_final_step_nats=None,
            min_val_nats=None,
            g1_accuracy=BASE_ROW["g1"],
            g8_val_loss_nats=BASE_ROW["g8"],
            g8_pass=True,
            g1_pass=False,
        )
    )
    measured_df = pd.DataFrame(measured_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ts38_parent_tradeoff.csv"
    measured_df.to_csv(csv_path, index=False)
    est_csv_path = out_dir / "ts38_parent_tradeoff_estimates.csv"
    estimates_df.to_csv(est_csv_path, index=False)
    print(f"\n[evt] wrote {csv_path} ({len(measured_df)} rows)")
    print(f"[evt] wrote {est_csv_path} ({len(estimates_df)} rows)")

    make_figure(args.store, curves, val_star, family_models, estimate_rows, fig_path)
    print(f"[evt] wrote {fig_path}")


# --- Figure -------------------------------------------------------------------

METHOD_COLOR = {
    "full_ft": "#1f77b4",
    "lora": "#d62728",
}  # tab:blue / brick red -- fixed, never cycled
LR_STYLE = {
    3.0e-4: "-",
    1.0e-4: "--",
    3.0e-5: "-.",
    1.0e-5: ":",
    1.0e-3: (0, (1, 1)),
}


def run_label(spec: dict) -> str:
    lr_s = f"{spec['lr']:.0e}" if spec["lr"] else "-"
    return f"{spec['method']} {lr_s}"


def make_figure(
    store: Path,
    curves: dict[str, list[tuple[int, float]]],
    val_star: float,
    family_models: dict[str, dict[str, object]],
    estimate_rows: list[dict],
    fig_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    ax_val, ax_g8, ax_scatter = axes

    # Panel (a): val vs step, log-y, one line per run, val* line, G8 markers.
    # The full-FT 1e-5 pair (ceil40k, g1fail) is a verified bit-exact replay
    # prefix (decisions.md 2026-08-15): draw only the longer curve per
    # (method, lr) so the identical shared prefix isn't plotted twice.
    longest_curve_len: dict[tuple[str, float | None], int] = {}
    for spec in RUN_SPECS:
        key = (spec["method"], spec["lr"])
        longest_curve_len[key] = max(longest_curve_len.get(key, 0), len(curves[spec["run"]]))
    plotted_curve_keys: set[tuple[str, float | None]] = set()
    for spec in RUN_SPECS:
        key = (spec["method"], spec["lr"])
        curve = curves[spec["run"]]
        steps = [s for s, _ in curve]
        vv = [v for _, v in curve]
        color = METHOD_COLOR[spec["method"]]
        ls = LR_STYLE.get(spec["lr"], "-")
        if key not in plotted_curve_keys and len(steps) >= longest_curve_len[key]:
            plotted_curve_keys.add(key)
            ax_val.plot(
                steps,
                vv,
                color=color,
                linestyle=ls,
                linewidth=1.6,
                label=run_label(spec),
                alpha=0.9,
            )
        # G8 measurement marker + annotation at THIS run's own final_step,
        # drawn for every run regardless of curve dedup above (each of the
        # 11 runs is its own distinct G8 measurement, even the two that
        # share a replayed curve prefix).
        val_at_final = dict(curve)[spec["final_step"]]
        ax_val.scatter([spec["final_step"]], [val_at_final], color=color, s=28, zorder=5)
        ax_val.annotate(
            f"G8={spec['g8']:.3f}",
            (spec["final_step"], val_at_final),
            fontsize=6.5,
            color=color,
            xytext=(3, 4),
            textcoords="offset points",
        )
    ax_val.axhline(
        val_star, color="black", linewidth=1, linestyle=":", label=f"val* = {val_star:.4f}"
    )
    ax_val.set_yscale("log")
    ax_val.set_xlabel("step")
    ax_val.set_ylabel("D_algo val loss (nats/label-token, log scale)")
    ax_val.set_title("(a) arithmetic val curves vs. step")
    ax_val.legend(fontsize=6, ncol=2, loc="upper right")

    # Panel (b): G8 vs step, measured points + fitted curves for the pair
    # families (LoRA 3e-4 emphasized, LoRA 1e-4 and full-FT 1e-5 lighter),
    # bar/base lines, shaded S_G1 windows.
    pair_specs = {
        "pair:lora_3e-4": ("LoRA 3e-4", 1.0),
        "pair:lora_1e-4": ("LoRA 1e-4", 0.35),
        "pair:full_ft_1e-5": ("full-FT 1e-5", 0.35),
    }
    step_grid_by_family = {}
    for fam, (label, alpha) in pair_specs.items():
        members = [s for s in RUN_SPECS if s["family"] == fam]
        steps_m = [m["final_step"] for m in members]
        g8_m = [m["g8"] for m in members]
        ax_g8.scatter(steps_m, g8_m, color="black", s=24, zorder=5, alpha=alpha)
        lo, hi = 0, max(steps_m) * 1.4
        grid = np.linspace(lo, hi, 200)
        step_grid_by_family[fam] = grid
        models = family_models[fam]
        ax_g8.plot(
            grid,
            [models["linear"].g8_at(s) for s in grid],
            color="#1f77b4",
            alpha=alpha,
            linewidth=1.4,
            linestyle="-",
            label=f"{label} linear" if alpha == 1.0 else None,
        )
        ax_g8.plot(
            grid,
            [models["power"].g8_at(max(s, 1.0)) for s in grid],
            color="#d62728",
            alpha=alpha,
            linewidth=1.4,
            linestyle="--",
            label=f"{label} power (p={models['power'].p:.2f})" if alpha == 1.0 else None,
        )
    for spec in RUN_SPECS:
        if spec["family"] == "single":
            ax_g8.scatter(
                [spec["final_step"]], [spec["g8"]], color="gray", s=14, alpha=0.6, marker="x"
            )
    ax_g8.axhline(G8_BAR, color="black", linewidth=1, linestyle=":", label=f"G8 bar = {G8_BAR}")
    ax_g8.axhline(G8_BASE, color="gray", linewidth=1, linestyle="--", label=f"base = {G8_BASE}")
    # Shaded S_G1 windows (persist) for the 3 pair families, linear model.
    for fam, (label, alpha) in pair_specs.items():
        members = [s for s in RUN_SPECS if s["family"] == fam]
        s1_vals = [
            r["s_g1_persist"] for r in estimate_rows if r["run"] in {m["run"] for m in members}
        ]
        s1_vals = [v for v in s1_vals if v is not None]
        if not s1_vals:
            continue
        s1 = min(s1_vals)
        s_g8 = family_models[fam]["linear"].step_at_g8(G8_BAR)
        if s1 < s_g8:
            ax_g8.axvspan(s1, s_g8, color="#1f77b4", alpha=0.06 * (1.5 if alpha == 1.0 else 1.0))
    ax_g8.set_xlabel("step")
    ax_g8.set_ylabel("G8: TinyStories val loss (nats)")
    ax_g8.set_ylim(1.0, 1.35)
    ax_g8.set_title("(b) G8 vs. step: measured points + 2-point fits (ESTIMATES)")
    ax_g8.legend(fontsize=6.5, loc="upper left")

    # Panel (c): G8 vs G1 scatter, bars drawn as lines, catastrophic full-FT
    # points annotated with an arrow so the <=1.3 region stays readable.
    for spec in RUN_SPECS:
        color = METHOD_COLOR[spec["method"]]
        g8 = spec["g8"]
        g1 = spec["g1"]
        label_txt = f"{spec['method']} {spec['lr']:.0e}@{spec['final_step']}"
        if g8 > 1.4:
            # Catastrophic point off the readable range: draw an arrow from
            # near the bar toward it, annotate the true value, don't distort
            # the axis for everyone else.
            ax_scatter.annotate(
                f"{label_txt}\nG8={g8:.2f}",
                xy=(g1, 1.30),
                xytext=(g1, 1.24 + 0.02 * (list(RUN_SPECS).index(spec) % 3)),
                fontsize=6,
                color=color,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )
        else:
            ax_scatter.scatter([g1], [g8], color=color, s=30, zorder=5)
            ax_scatter.annotate(
                label_txt, (g1, g8), fontsize=6, xytext=(3, 3), textcoords="offset points"
            )
    ax_scatter.axhline(G8_BAR, color="black", linewidth=1, linestyle=":")
    ax_scatter.axvline(G1_BAR, color="black", linewidth=1, linestyle=":")
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(1.0, 1.32)
    ax_scatter.set_xlabel("G1: exact-match accuracy")
    ax_scatter.set_ylabel("G8: TinyStories val loss (nats), clipped at 1.32")
    ax_scatter.set_title("(c) G8 vs. G1, both-pass quadrant = lower-right of the dotted lines")
    ax_scatter.text(
        G1_BAR + 0.01,
        1.01,
        "both pass",
        fontsize=7,
        color="green",
        ha="left",
        va="bottom",
    )

    fig.suptitle("ts38 pre-taught parent: G1 (arithmetic) vs. G8 (TinyStories retention) tradeoff")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
