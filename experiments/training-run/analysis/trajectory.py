"""Weight-space trajectory geometry driver (spec 02 §7, runs 7/8).

Asks one question of the snapshot weights: **does elicitation walk a straight,
self-consistent path through weight space while teaching wanders?** For each
run it replays the optimizer's path θ_ref → θ_1 → … → θ_final and measures its
geometry — how far each step moves, how far the run has actually got from
where it started, how much of the walking was wasted, and whether consecutive
steps agree on a direction.

Per snapshot k (with d_k = θ_k − θ_{k−1} the step-delta and Δ_final =
θ_final − θ_ref the run's overall displacement) the driver emits:

- ``delta_step_norm``   ‖d_k‖
- ``path_length_cum``   Σ_{j≤k} ‖d_j‖ — arc length walked so far
- ``net_displacement``  ‖θ_k − θ_ref‖ — straight-line distance from the start
- ``path_efficiency``   net_displacement / path_length_cum ∈ (0, 1]; 1.0 means
  a perfectly straight walk, → 0 means the run is milling around
- ``step_cosine``       cos(d_k, d_{k−1}) — local self-consistency (no row for
  the first delta, which has no predecessor)
- ``cos_to_final_dir``  cos(d_k, Δ_final) — is this step buying displacement in
  the direction the run ultimately ends up going?

Results land as one long-format parquet through the ZOO-4 writer
(``results/trajectory_geometry.parquet``, join key run_id/checkpoint_step);
the overlay figure lands in ``analysis/figures/`` (gitignored).

Design notes:

- **Full-FT only.** ``manifest.training.method`` must be ``"full_ft"`` (runs
  7/8). LoRA runs are refused: their snapshots store the A/B factors, so a
  weight-space step-delta is ``ΔW_k − ΔW_{k−1}`` of the *merged* update
  (``adapters.lora_delta_w``), a different and much more expensive
  materialisation, and the factorisation's rotational gauge freedom makes raw
  factor differences meaningless. Out of scope here.
- **Reference state**: the earliest discovered snapshot, exactly as
  ``adapters.py`` picks it (this driver reuses its ``_discover_steps`` /
  ``_load_state``). Step 1 is already one optimizer update in, so it is the
  closest available stand-in for init, not init itself. The reference step
  gets no rows — every metric is defined relative to it.
- ``layer = -1`` on every row: the mandated ``layer`` column is a **global
  sentinel** here. Trajectory geometry is a whole-parameter-vector quantity;
  the per-layer allocation of update energy already lives in the
  ``adapter_diffs`` parquet (``delta_w_layer_alloc``), so it is not repeated.
- **Streaming, RAM-bounded**: a single pass in step order holding at most
  θ_ref, Δ_final, θ_prev and d_{k−1} (plus the snapshot being read) — never
  all snapshots, and never one giant flattened parameter vector. Norms and dot
  products are accumulated in python floats, upcasting each tensor pair to
  fp64; held states/deltas stay fp32 (as in ``adapters.py``).
- Cosine rows are emitted only when both operands have nonzero norm — an
  undefined cosine is dropped rather than fabricated as 0.0.

CPU-only; runs wherever the snapshots are (the box).

Usage:
    python3 trajectory.py [--run-id <rid> ...] [--store <dir>]
        [--fig figures/trajectory_geometry.png]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

from adapters import _discover_steps, _load_state

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
GLOBAL_LAYER = -1  # sentinel: these metrics are whole-parameter-vector, not per layer
LATE_STEP = 1000  # summary print averages step_cosine over steps beyond this


def _float_keys(state: dict[str, torch.Tensor]) -> list[str]:
    """Sorted floating-point tensor keys — the parameters the trajectory lives in."""
    return sorted(key for key, tensor in state.items() if tensor.is_floating_point())


def _require_keys(run_id: str, step: int, state: dict[str, torch.Tensor], keys: list[str]) -> None:
    missing = [key for key in keys if key not in state]
    if missing:
        raise ValueError(
            f"{run_id}@step_{step}: snapshot is missing {len(missing)} tensor(s) present in the "
            f"reference snapshot (first: {missing[0]!r}) — the parameter vector must be identical "
            "across snapshots for trajectory geometry to mean anything"
        )


def _difference(
    state: dict[str, torch.Tensor], other: dict[str, torch.Tensor], keys: list[str]
) -> dict[str, torch.Tensor]:
    """``state − other`` per key, in fp32 (upcast from the stored dtype)."""
    return {key: state[key].to(torch.float32) - other[key].to(torch.float32) for key in keys}


def _norm(delta: dict[str, torch.Tensor], keys: list[str]) -> float:
    """‖delta‖, accumulated per tensor in fp64."""
    return sum(float(delta[key].double().pow(2).sum().item()) for key in keys) ** 0.5


def _cosine(dot: float, norm_a: float, norm_b: float) -> float | None:
    """cos(a, b) from a precomputed dot product; None when either operand is degenerate."""
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return dot / (norm_a * norm_b)


def _step_geometry(
    state: dict[str, torch.Tensor],
    prev: dict[str, torch.Tensor],
    ref: dict[str, torch.Tensor],
    delta_final: dict[str, torch.Tensor],
    prev_delta: dict[str, torch.Tensor] | None,
    keys: list[str],
) -> tuple[dict[str, torch.Tensor], float, float, float, float]:
    """One streaming step: build d_k = θ_k − θ_prev and its four fp64 accumulators.

    Returns ``(d_k, ‖d_k‖, ‖θ_k − θ_ref‖, d_k·d_{k−1}, d_k·Δ_final)``. Every
    quantity is accumulated tensor by tensor so that no flattened copy of the
    parameter vector is ever materialised; ``d_k`` is kept in fp32 because the
    caller carries it into the next step as ``prev_delta``.
    """
    delta: dict[str, torch.Tensor] = {}
    d_sq = net_sq = dot_prev = dot_final = 0.0
    for key in keys:
        cur = state[key].double()
        d = cur - prev[key].double()
        d_sq += float(d.pow(2).sum().item())
        net_sq += float((cur - ref[key].double()).pow(2).sum().item())
        dot_final += float((d * delta_final[key].double()).sum().item())
        if prev_delta is not None:
            dot_prev += float((d * prev_delta[key].double()).sum().item())
        delta[key] = d.to(torch.float32)
    return delta, d_sq**0.5, net_sq**0.5, dot_prev, dot_final


def run_rows(run_id: str, store: Path) -> list[dict]:
    manifest = load_run(run_id, store=store)
    method = manifest.data["training"]["method"]
    if method != "full_ft":
        raise ValueError(
            f"{run_id}: training.method is {method!r}, but trajectory.py handles 'full_ft' only. "
            "LoRA trajectories need the merged ΔW per step (adapters.lora_delta_w) and the "
            "factorisation's gauge freedom makes raw A/B differences meaningless — out of scope."
        )
    common = {
        "run_id": run_id,
        "base_model_key": manifest.data["base_model"]["hf_id"],
        "regime": manifest.data["regime"],
        "dataset_size": manifest.data["dataset"]["n_unique_examples"],
    }
    snap_root = run_dir(run_id, store=store) / "snapshots"
    steps = _discover_steps(snap_root)
    if len(steps) < 2:
        raise FileNotFoundError(
            f"{run_id}: found {len(steps)} snapshot(s) under {snap_root} — trajectory geometry "
            "needs at least a reference snapshot and one step after it"
        )

    ref_step = steps[0]
    ref_state = _load_state(run_id, ref_step, store)
    keys = _float_keys(ref_state)
    final_state = _load_state(run_id, steps[-1], store)
    _require_keys(run_id, steps[-1], final_state, keys)
    delta_final = _difference(final_state, ref_state, keys)
    del final_state  # the last snapshot is re-read in step order; keep only Δ_final
    final_norm = _norm(delta_final, keys)

    prev_state = ref_state
    prev_delta: dict[str, torch.Tensor] | None = None
    prev_norm = 0.0
    path_cum = 0.0
    rows: list[dict] = []
    for k in steps[1:]:
        state = _load_state(run_id, k, store)
        _require_keys(run_id, k, state, keys)
        delta, d_norm, net, dot_prev, dot_final = _step_geometry(
            state, prev_state, ref_state, delta_final, prev_delta, keys
        )
        path_cum += d_norm
        base = {**common, "checkpoint_step": k, "layer": GLOBAL_LAYER}
        metrics: dict[str, float | None] = {
            "delta_step_norm": d_norm,
            "path_length_cum": path_cum,
            "net_displacement": net,
            "path_efficiency": (net / path_cum) if path_cum > 0.0 else 0.0,
            "step_cosine": _cosine(dot_prev, d_norm, prev_norm),
            "cos_to_final_dir": _cosine(dot_final, d_norm, final_norm),
        }
        for name, value in metrics.items():
            if value is None:  # undefined cosine (zero-length step): drop, never fabricate
                continue
            rows.append({**base, "metric_name": name, "metric_value": value})
        prev_state, prev_delta, prev_norm = state, delta, d_norm
    print(
        f"[evt] {run_id}: {len(steps)} snapshots ({method}), ref step {ref_step}, "
        f"‖Δ_final‖={final_norm:.4f}, {len(rows)} rows"
    )
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    eff = df[df["metric_name"] == "path_efficiency"]
    for i, (rid, by_run) in enumerate(eff.groupby("run_id", sort=True)):
        by_run = by_run.sort_values("checkpoint_step")
        regime = by_run["regime"].iloc[0]
        axes[0].plot(
            by_run["checkpoint_step"],
            by_run["metric_value"],
            color=colors[i % len(colors)],
            lw=2.0,
            label=f"{rid} ({regime})",
        )

    cos = df[df["metric_name"] == "step_cosine"]
    for i, (rid, by_run) in enumerate(cos.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        by_run = by_run.sort_values("checkpoint_step")
        regime = by_run["regime"].iloc[0]
        axes[1].plot(
            by_run["checkpoint_step"], by_run["metric_value"], color=color, alpha=0.2, lw=0.8
        )
        smooth = by_run["metric_value"].rolling(9, min_periods=3).median()
        axes[1].plot(
            by_run["checkpoint_step"], smooth, color=color, lw=2.0, label=f"{rid} ({regime})"
        )

    for ax, ylabel in zip(axes, ("path efficiency (net / arc length)", "cos(d_k, d_{k-1})")):
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("checkpoint step")
    axes[0].set_title("weight-space trajectory geometry (bottom: raw faint, rolling median bold)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help=f"repeatable; default: {', '.join(DEFAULT_RUNS)}",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "trajectory_geometry.png",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store))
    df = pd.DataFrame(rows)
    path = write_results(df, "trajectory_geometry", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    for rid, by_run in df.groupby("run_id", sort=True):
        n_snapshots = by_run["checkpoint_step"].nunique() + 1  # + the reference snapshot
        eff = by_run[by_run["metric_name"] == "path_efficiency"].sort_values("checkpoint_step")
        final_eff = float(eff["metric_value"].iloc[-1])
        late = by_run[
            (by_run["metric_name"] == "step_cosine") & (by_run["checkpoint_step"] > LATE_STEP)
        ]["metric_value"]
        late_txt = f"{late.mean():.4f} (n={len(late)})" if len(late) else "n/a (no such steps)"
        print(
            f"[evt] {rid}: {n_snapshots} snapshots processed, final path_efficiency "
            f"{final_eff:.4f}, mean step_cosine over steps>{LATE_STEP} {late_txt}"
        )


if __name__ == "__main__":
    main()
