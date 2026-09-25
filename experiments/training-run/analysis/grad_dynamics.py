"""Gradient-norm decay + weight displacement from snapshots (Phase 0 test 8,
EXPERIMENTS.md §6.22): does training move fast then stop, or keep working?

Owner's pre-registered signature (EXPERIMENTS.md §6.22 Tier-1 table): elicit
= "one big early gradient step, then collapse"; teach = "gradual, sustained
descent". This driver turns that qualitative table entry into numbers, per
run, from three artifacts already on disk (no new instrumentation, per the
§6.22 build notes):

- ``logs/gradstats.jsonl`` (spec 00 §4) — LoRA target runs only
  (``geode.edl.loop``/``train_prequential``): one ``global_grad_norm`` per
  logged step (stride-configurable; step 0 is always present).
  ``geode.train.sft.train_sft`` (the full-FT trainer, used for the ts38pp
  and ts38mt-fmt PARENTS) never writes this file — it logs ``grad_norm``
  inline in ``train_log.jsonl`` instead (``geode/train/sft.py:315-323``).
  So ``read_grad_norms`` tries ``gradstats.jsonl`` first and, only when that
  file is entirely absent, falls back to ``train_log.jsonl``'s own
  ``grad_norm`` field — this fallback is NOT in the original test-8 sketch,
  it is required for the driver to say anything at all about a full-FT run;
  when neither source has a value for a step, ``grad_norm`` is NaN, never
  fabricated.
- ``train_log.jsonl`` / ``eval_log.jsonl`` at the run root (spec 02 §6
  extension) — ``train_loss_nats`` every step, ``val_loss_nats`` at eval
  steps only (NaN elsewhere).
- The run's own weight snapshots — displacement ``‖θ_k − θ_ref‖``, dispatched
  on ``manifest.training.method``:

  - ``lora``: ``snapshots/step_<k>/adapter.safetensors``. The trajectory
    point is the merged update ``ΔW = (α/2r)·B@A`` (``adapters.lora_delta_w``
    / ``trajectory._lora_state`` + ``_lora_point``, imported as siblings —
    private-named but this repo's own convention, see ``trajectory.py``'s
    own docstring). ``B`` is zero-initialised, so ``ΔW ≡ 0`` at true init
    exactly: ``net_displacement(k) = ‖ΔW_k‖`` needs no reference subtraction,
    and the reference "step" for interpolation purposes is 0 (true init).
  - ``full_ft``: ``sft_snapshots/step_<k:07d>/model.safetensors``
    (``save_pretrained`` dirs). ``θ_ref`` is EITHER the model directory
    passed as ``--ref-dir`` (a plain ``model.safetensors`` dir, e.g. the
    base model's checkpoint — the correct reference for a full-FT parent
    whose true init IS the base model) OR, when ``--ref-dir`` is omitted,
    the earliest discovered snapshot as a stand-in (``trajectory.py``'s own
    full-FT convention: step 1 is already one optimizer update in, so it is
    the closest available substitute for init, not init itself — NOT a
    faithful ``θ_ref``). With ``--ref-dir`` every discovered step gets a row
    and the interpolation reference step is 0; with the fallback, the
    earliest snapshot itself gets no row (nothing to compare it to) and the
    interpolation reference step is that snapshot's own step number.

Streaming: at most one reference state + one snapshot being processed are
ever held in memory (never all snapshots at once), matching
``trajectory.py``/``adapters.py``.

Rows, two ``level`` values in one table (``mech_lib.write_table``, following
``weight_diff.py``'s precedent for a driver whose row shapes differ by
level — NOT ``geode.zoo.write_results``'s single-shape long format):

- ``level="step"``: ``run_id, regime, dataset_size, step, grad_norm,
  train_loss_nats, val_loss_nats, net_displacement, displacement_frac``.
  ``displacement_frac = net_displacement / final_displacement`` (NaN unless
  both are defined; the final snapshot's own displacement_frac is exactly
  1.0 by construction).
- ``level="run"``: ``run_id, regime, dataset_size, n_steps`` plus the six
  pre-registered summary metrics below. Every summary metric is a PURE
  function of a ``{step: value}`` curve (or, for the displacement metrics,
  the already-computed ``displacement_frac`` curve) plus ``n_steps`` — none
  of them touch a file, so they are unit-tested directly on synthetic
  curves. All six treat "fraction of training" as ``step / n_steps``, where
  ``n_steps`` is the run's own final training step (``max`` of
  ``train_log.jsonl``'s step column) — one consistent denominator across
  every metric, so they are comparable to each other within a run. Every
  windowed metric uses ``max(1, ceil(q * n_steps))`` as its step threshold,
  so a run shorter than ``1/q`` steps still gets a well-defined (>= 1 step)
  window instead of an empty one.

  * ``grad_early_mass_frac(q=0.1)`` — Σ grad_norm over the first ``q``
    fraction of steps / Σ over all steps. Elicit ("one big early step") ⇒
    ≫ q; teach ("gradual, sustained descent") ⇒ ≈ q (mass spread evenly).
  * ``grad_peak_ratio`` — max grad_norm in the first ``q`` fraction / median
    grad_norm in the last 50%. Elicit ⇒ ≫ 1 (a spike against a flat tail);
    teach ⇒ ≈ 1 (comparable early/late magnitude).
  * ``grad_half_step_frac`` — step fraction at which cumulative grad-norm
    MASS (running sum, not running min) first reaches 50% of its total.
    Elicit ⇒ ≈ 0; teach ⇒ ≈ 0.5 (mass accrues at a roughly constant rate).
  * ``disp_frac_at_10pct`` — ``displacement_frac`` at the LATEST snapshot at
    or before the first ``q`` fraction of training. Elicit ⇒ already large
    (near 1) this early; teach ⇒ still small (near 0).
  * ``disp_half_step_frac`` — step fraction at which ``displacement_frac``
    first reaches >= 0.5, linearly interpolated between the two bracketing
    points (the reference state — step = the run's interpolation reference
    step above, displacement_frac = 0.0 by construction — is always the
    first point, so this is well-defined even when the very first snapshot
    is already past 0.5). Elicit ⇒ ≈ 0; teach ⇒ ≈ 0.5.
  * ``loss_half_step_frac`` — step fraction at which the RUNNING-MINIMUM
    (min-so-far) train loss first recovers >= 50% of its own total drop
    ``L0 − L_final``, where ``L0``/``L_final`` are the running-minimum's own
    first/last values. The min-so-far smoothing exists because a raw
    per-batch loss is noisy enough that a naive first-crossing check would
    fire on a downward blip rather than the sustained drop; using the
    running minimum makes the crossing monotone-safe. NaN when the run's
    running minimum never drops (``L0 == L_final``) — a genuinely undefined
    "half of a zero drop", never fabricated as 0.0.

  All six are NaN, not a crash or a fabricated 0.0/1.0, whenever their input
  curve has no usable (non-NaN) points at all — e.g. gradstats entirely
  absent and no ``train_log.jsonl`` fallback either.

Usage:
    python3 grad_dynamics.py --run-id evt-ts38mt-pp-n21544 \\
        --run-id evt-ts38mt-fmt-n21544 --out grad_dynamics_n21544.csv
    python3 grad_dynamics.py --run-id evt-ts38mt-fmt-parent \\
        --ref-dir $GEODE_STORE/runs/evt-run1-base-v3-ext/model \\
        --out grad_dynamics_fmt_parent.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from safetensors.torch import load_file

from adapters import _discover_steps
from mech_lib import write_table
from trajectory import _lora_point, _lora_state

from geode.probe import load_probe_dumps
from geode.zoo import load_run
from geode.zoo.store import gradstats_log_path, run_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- log/snapshot reading ---------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Every JSON record in ``path``, or ``[]`` if the file does not exist."""
    if not path.is_file():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_grad_norms(run_id: str, store: Path) -> dict[int, float]:
    """``{step: global_grad_norm}``, preferring ``gradstats.jsonl`` (spec 00
    §4) and falling back to ``train_log.jsonl``'s inline ``grad_norm`` field
    (the full-FT trainer's only grad-norm source — module docstring). ``{}``
    when neither source exists; callers read missing steps as NaN."""
    rows = _read_jsonl(gradstats_log_path(run_id, store=store))
    if rows:
        return {int(r["step"]): float(r["global_grad_norm"]) for r in rows}
    train_rows = _read_jsonl(run_dir(run_id, store=store) / "train_log.jsonl")
    return {int(r["step"]): float(r["grad_norm"]) for r in train_rows if "grad_norm" in r}


def read_train_log(run_id: str, store: Path) -> dict[int, float]:
    """``{step: train_loss_nats}`` from the run-root ``train_log.jsonl``."""
    rows = _read_jsonl(run_dir(run_id, store=store) / "train_log.jsonl")
    if not rows:
        raise FileNotFoundError(
            f"{run_id}: no train_log.jsonl (or it is empty) under {run_dir(run_id, store=store)}"
        )
    return {int(r["step"]): float(r["train_loss_nats"]) for r in rows}


def read_val_log(run_id: str, store: Path) -> dict[int, float]:
    """``{step: val_loss_nats}`` from ``eval_log.jsonl``; ``{}`` if absent."""
    rows = _read_jsonl(run_dir(run_id, store=store) / "eval_log.jsonl")
    return {int(r["step"]): float(r["val_loss_nats"]) for r in rows}


def _lora_displacements(run_id: str, store: Path, manifest) -> dict[int, float]:
    """``{step: ‖ΔW_k‖}`` for a LoRA run — no reference subtraction (ΔW ≡ 0 at
    init exactly, ``trajectory.py``'s own justification)."""
    lora_cfg = manifest.data["training"]["lora"]
    alpha, rank = float(lora_cfg["alpha"]), int(lora_cfg["rank"])
    snap_root = run_dir(run_id, store=store) / "snapshots"
    steps = _discover_steps(snap_root)
    if not steps:
        raise FileNotFoundError(f"{run_id}: no snapshots under {snap_root}")
    out: dict[int, float] = {}
    for k in steps:
        point = _lora_point(_lora_state(run_id, k, store), alpha, rank)
        sq = sum(float(dw.double().pow(2).sum().item()) for dw in point.values())
        out[k] = sq**0.5
    return out


def _full_ft_displacements(
    run_id: str, store: Path, ref_dir: Path | None
) -> tuple[dict[int, float], int | None]:
    """``({step: ‖θ_k − θ_ref‖}, ref_step)`` for a full-FT run.

    ``ref_step`` is ``None`` when ``--ref-dir`` supplies an external
    reference (conceptually "step 0", true init) and an int (the earliest
    snapshot's own step) when that snapshot is used as the stand-in ref —
    see the module docstring's full-FT paragraph. Only one of θ_ref and the
    current snapshot is ever resident at a time.
    """
    snap_root = run_dir(run_id, store=store) / "sft_snapshots"
    steps = load_probe_dumps(snap_root, marker="model.safetensors")
    if not steps:
        raise FileNotFoundError(f"{run_id}: no sft_snapshots under {snap_root}")

    def load_step(k: int) -> dict[str, torch.Tensor]:
        return load_file(str(snap_root / f"step_{k:07d}" / "model.safetensors"))

    if ref_dir is not None:
        ref_file = Path(ref_dir) / "model.safetensors"
        if not ref_file.is_file():
            raise FileNotFoundError(f"--ref-dir {ref_dir}: no model.safetensors there")
        ref_state = load_file(str(ref_file))
        walk, ref_step = steps, None
    else:
        ref_state = load_step(steps[0])
        walk, ref_step = steps[1:], steps[0]
    if not walk:
        raise FileNotFoundError(
            f"{run_id}: only {len(steps)} sft snapshot(s) and no usable reference step — "
            "full-FT displacement needs at least one step away from θ_ref (pass --ref-dir, "
            "or provide a second snapshot for the earliest-snapshot fallback)"
        )

    keys = sorted(key for key, t in ref_state.items() if t.is_floating_point())
    out: dict[int, float] = {}
    for k in walk:
        state = load_step(k)
        missing = [key for key in keys if key not in state]
        if missing:
            raise ValueError(
                f"{run_id}@step_{k}: snapshot is missing {len(missing)} tensor(s) present in "
                f"θ_ref (first: {missing[0]!r})"
            )
        sq = sum(
            float(
                (state[key].to(torch.float32).double() - ref_state[key].to(torch.float32).double())
                .pow(2)
                .sum()
                .item()
            )
            for key in keys
        )
        out[k] = sq**0.5
    return out, ref_step


@dataclass
class RunCurves:
    """One run's raw curves, everything the per-step rows and the six
    pure summary metrics need. ``ref_step`` is the x-axis step at which
    displacement is 0 by construction (module docstring)."""

    run_id: str
    method: str
    regime: str
    dataset_size: int
    n_steps: int
    ref_step: int
    train: dict[int, float]
    val: dict[int, float]
    grad: dict[int, float]
    disp: dict[int, float]

    @property
    def final_disp(self) -> float:
        return self.disp[max(self.disp)] if self.disp else math.nan

    @property
    def disp_frac(self) -> dict[int, float]:
        fd = self.final_disp
        if not self.disp or math.isnan(fd) or fd <= 0.0:
            return {}
        return {s: v / fd for s, v in self.disp.items()}


def load_run_curves(run_id: str, store: Path, ref_dir: Path | None = None) -> RunCurves:
    """Dispatch on ``manifest.training.method``, read every artifact for one run."""
    manifest = load_run(run_id, store=store)
    method = manifest.data["training"]["method"]
    train = read_train_log(run_id, store)
    val = read_val_log(run_id, store)
    grad = read_grad_norms(run_id, store)
    if method == "lora":
        disp, ref_step = _lora_displacements(run_id, store, manifest), 0
    elif method == "full_ft":
        disp, fallback_ref_step = _full_ft_displacements(run_id, store, ref_dir)
        ref_step = 0 if fallback_ref_step is None else fallback_ref_step
    else:
        raise ValueError(f"{run_id}: unsupported training.method {method!r} (lora or full_ft)")
    return RunCurves(
        run_id=run_id,
        method=method,
        regime=manifest.data["regime"],
        dataset_size=manifest.data["dataset"]["n_unique_examples"],
        n_steps=max(train),
        ref_step=ref_step,
        train=train,
        val=val,
        grad=grad,
        disp=disp,
    )


# --- pure metric functions (unit-tested directly on synthetic curves) ------


def _window_threshold(n_steps: int, q: float) -> int:
    """Step at which the first ``q`` fraction of training ends; >= 1 always."""
    return max(1, math.ceil(q * n_steps))


def grad_early_mass_frac(grad: dict[int, float], n_steps: int, q: float = 0.1) -> float:
    """Σ grad_norm over the first ``q`` fraction of steps / Σ over all steps.

    See module docstring for the pre-registered elicit/teach reading.
    """
    pts = {s: v for s, v in grad.items() if not math.isnan(v)}
    if not pts:
        return math.nan
    total = sum(pts.values())
    if total <= 0.0:
        return math.nan
    threshold = _window_threshold(n_steps, q)
    early = sum(v for s, v in pts.items() if s <= threshold)
    return early / total


def grad_peak_ratio(grad: dict[int, float], n_steps: int, q: float = 0.1) -> float:
    """max grad_norm in the first ``q`` fraction / median grad_norm in the
    last 50% of steps. NaN if either window has no data or the late median
    is exactly 0 (undefined ratio, never fabricated as inf)."""
    pts = {s: v for s, v in grad.items() if not math.isnan(v)}
    if not pts:
        return math.nan
    threshold = _window_threshold(n_steps, q)
    half = n_steps / 2.0
    early = [v for s, v in pts.items() if s <= threshold]
    late = [v for s, v in pts.items() if s > half]
    if not early or not late:
        return math.nan
    median_late = statistics.median(late)
    if median_late <= 0.0:
        return math.nan
    return max(early) / median_late


def grad_half_step_frac(grad: dict[int, float], n_steps: int) -> float:
    """Step fraction at which cumulative grad-norm MASS (running sum) first
    reaches 50% of its total. NaN with no data or zero total mass."""
    pts = sorted((s, v) for s, v in grad.items() if not math.isnan(v))
    if not pts:
        return math.nan
    total = sum(v for _, v in pts)
    if total <= 0.0:
        return math.nan
    half = 0.5 * total
    cum = 0.0
    for s, v in pts:
        cum += v
        if cum >= half:
            return s / n_steps
    return pts[-1][0] / n_steps  # float-rounding guard; the loop above always fires by here


def disp_frac_at_10pct(disp_frac_by_step: dict[int, float], n_steps: int, q: float = 0.1) -> float:
    """``displacement_frac`` at the latest snapshot at or before the first
    ``q`` fraction of training. NaN if no snapshot falls in that window."""
    threshold = _window_threshold(n_steps, q)
    candidates = {
        s: f for s, f in disp_frac_by_step.items() if s <= threshold and not math.isnan(f)
    }
    if not candidates:
        return math.nan
    return candidates[max(candidates)]


def disp_half_step_frac(disp_frac_by_step: dict[int, float], ref_step: int, n_steps: int) -> float:
    """Step fraction at which ``displacement_frac`` first reaches >= 0.5,
    linearly interpolated between bracketing points. The reference state
    (step=``ref_step``, displacement_frac=0.0 by construction) is always the
    first point, so this is defined even when the earliest real snapshot is
    already past 0.5. NaN if there is no displacement data at all."""
    pts = sorted((s, f) for s, f in disp_frac_by_step.items() if not math.isnan(f))
    if not pts:
        return math.nan
    series = [(ref_step, 0.0), *pts]
    for (s0, f0), (s1, f1) in zip(series, series[1:]):
        if f1 >= 0.5:
            frac_step = s1 if f1 == f0 else s0 + (0.5 - f0) / (f1 - f0) * (s1 - s0)
            return frac_step / n_steps
    return math.nan  # final displacement_frac (== 1.0 by construction) never reached 0.5


def loss_half_step_frac(train: dict[int, float], n_steps: int) -> float:
    """Step fraction at which the running-minimum train loss first recovers
    >= 50% of its own total drop. NaN with no data or zero drop."""
    pts = sorted((s, v) for s, v in train.items() if not math.isnan(v))
    if not pts:
        return math.nan
    running_min = math.inf
    smoothed = []
    for s, v in pts:
        running_min = min(running_min, v)
        smoothed.append((s, running_min))
    l0, l_final = smoothed[0][1], smoothed[-1][1]
    drop = l0 - l_final
    if drop <= 0.0:
        return math.nan
    target = l0 - 0.5 * drop
    for s, v in smoothed:
        if v <= target:
            return s / n_steps
    return math.nan  # unreachable: smoothed[-1][1] == l_final <= target by construction


# --- row builders ------------------------------------------------------------


def run_step_rows(curves: RunCurves) -> list[dict]:
    """One ``level="step"`` row per ``train_log.jsonl`` step (module docstring)."""
    disp_frac = curves.disp_frac
    common = {"run_id": curves.run_id, "regime": curves.regime, "dataset_size": curves.dataset_size}
    rows = []
    for step in sorted(curves.train):
        rows.append(
            {
                "level": "step",
                **common,
                "step": step,
                "grad_norm": curves.grad.get(step, math.nan),
                "train_loss_nats": curves.train[step],
                "val_loss_nats": curves.val.get(step, math.nan),
                "net_displacement": curves.disp.get(step, math.nan),
                "displacement_frac": disp_frac.get(step, math.nan),
            }
        )
    return rows


def run_summary_row(curves: RunCurves) -> dict:
    """The one ``level="run"`` row: the six pre-registered summary metrics."""
    disp_frac = curves.disp_frac
    return {
        "level": "run",
        "run_id": curves.run_id,
        "regime": curves.regime,
        "dataset_size": curves.dataset_size,
        "n_steps": curves.n_steps,
        "grad_early_mass_frac": grad_early_mass_frac(curves.grad, curves.n_steps),
        "grad_peak_ratio": grad_peak_ratio(curves.grad, curves.n_steps),
        "grad_half_step_frac": grad_half_step_frac(curves.grad, curves.n_steps),
        "disp_frac_at_10pct": disp_frac_at_10pct(disp_frac, curves.n_steps),
        "disp_half_step_frac": disp_half_step_frac(disp_frac, curves.ref_step, curves.n_steps),
        "loss_half_step_frac": loss_half_step_frac(curves.train, curves.n_steps),
    }


def plot(step_rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(step_rows)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (rid, by_run) in enumerate(df.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        by_run = by_run.sort_values("step")
        label = f"{rid} ({by_run['regime'].iloc[0]})"
        axes[0].plot(by_run["step"], by_run["grad_norm"], color=color, lw=1.2, label=label)
        axes[1].plot(
            by_run["step"],
            by_run["displacement_frac"],
            color=color,
            lw=1.6,
            marker=".",
            label=label,
        )
        axes[2].plot(by_run["step"], by_run["train_loss_nats"], color=color, lw=1.2, label=label)
    for ax, ylabel in zip(axes, ("grad_norm", "displacement_frac", "train_loss_nats")):
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7)
    axes[2].set_xlabel("step")
    axes[0].set_title("grad-norm decay + weight displacement")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def print_summary(summary_rows: list[dict]) -> None:
    df = pd.DataFrame(summary_rows)
    cols = [
        "run_id",
        "regime",
        "n_steps",
        "grad_early_mass_frac",
        "grad_peak_ratio",
        "grad_half_step_frac",
        "disp_frac_at_10pct",
        "disp_half_step_frac",
        "loss_half_step_frac",
    ]
    print("[evt] " + df[cols].to_string(index=False).replace("\n", "\n[evt] "))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", action="append", dest="run_ids", required=True, help="repeatable")
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="full_ft only: a plain model.safetensors dir used as θ_ref (e.g. the base model's "
        "checkpoint); default falls back to the earliest sft_snapshots step, a stand-in, not "
        "true init (module docstring)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "grad_dynamics.csv"
    )
    ap.add_argument(
        "--fig", type=Path, default=None, help="optional PNG; nothing written if omitted"
    )
    args = ap.parse_args()

    step_rows: list[dict] = []
    summary_rows: list[dict] = []
    for rid in args.run_ids:
        curves = load_run_curves(rid, args.store, args.ref_dir)
        step_rows.extend(run_step_rows(curves))
        summary_rows.append(run_summary_row(curves))
        n_grad = sum(1 for v in curves.grad.values() if not math.isnan(v))
        note = "" if n_grad else " (no grad-norm data found in gradstats.jsonl or train_log.jsonl)"
        print(
            f"[evt] {rid}: {curves.method}, {curves.n_steps} steps, {len(curves.disp)} "
            f"displacement snapshots, {n_grad} grad-norm points{note}"
        )

    df = pd.concat([pd.DataFrame(step_rows), pd.DataFrame(summary_rows)], ignore_index=True)
    write_table(df, args.out)
    print(f"[evt] wrote {args.out} ({len(df)} rows)")
    print_summary(summary_rows)
    if args.fig is not None:
        plot(step_rows, args.fig)


if __name__ == "__main__":
    main()
