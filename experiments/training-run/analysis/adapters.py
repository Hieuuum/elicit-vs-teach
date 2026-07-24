"""Adapter-diff analysis driver (spec 02 §7 "Adapter diffs", §9 deliverable).

Reads each run's snapshot *weights* straight off disk (no model
instantiation, no probe dumps) and, per (snapshot, module), summarises the
cumulative weight update ΔW: its Frobenius norm ‖ΔW‖_F, its spectral-entropy
effective rank (``geode.probe.effective_rank``), and the per-layer allocation
of update energy. Results land as one long-format parquet through the ZOO-4
writer (``results/adapter_diffs.parquet``, join key run_id/checkpoint_step/
layer, ``regime`` = elicit/teach); the overlay figure lands in
``analysis/figures/`` (gitignored).

Two ΔW paths, decided per run from ``manifest.training.method``:

- ``full_ft`` (runs 7/8): ΔW(module, k) = W_k − W_ref, taken against a
  *reference* snapshot — the earliest discovered step by default (``--ref-step``
  overrides). NOTE: step 1 is already one optimizer update in, so it is only
  the closest available stand-in for init, not init itself. 2-D weight tensors
  get ‖ΔW‖_F + effective rank; 1-D tensors (norms, biases) feed the Frobenius
  totals but get no rank.
- ``lora`` (runs 5/6): ΔW(module, k) = ``(α / 2r) · (B_k @ A_k)`` — this repo's
  pinned V5.47 scaling, not PEFT's α/r. Because B is zero-initialised, this is
  already the diff from run init, so there is NO reference subtraction on the
  LoRA path (α, r come from the manifest's lora block).

All diffs and norms are computed in fp32 (upcast from the stored dtype);
``effective_rank`` re-upcasts to fp64 internally. CPU-only, runs wherever the
snapshots are.

Usage:
    python3 adapters.py [--run-id <rid> ...] [--store <dir>] [--ref-step <k>]
        [--fig figures/adapter_diffs.png]
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from safetensors.torch import load_file

from geode.probe import effective_rank
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
# Row metric_name values: delta_w_fro, delta_w_effective_rank (per 2-D module),
# delta_w_layer_alloc (per layer), delta_w_fro_total (per step).
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\b")


class _Delta(NamedTuple):
    """One module's weight update at a step: ``dw`` is None for 1-D tensors."""

    module: str
    layer: int
    dw: torch.Tensor | None  # 2-D ΔW (rank-bearing); None for norms/biases
    fro_sq: float  # ‖ΔW‖²_F, always counted toward the totals


def lora_delta_w(b: torch.Tensor, a: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
    """Effective weight update of one LoRA module: ``(alpha / (2*rank)) * (b @ a)``.

    ``b`` is ``[d_out, r]`` and ``a`` is ``[r, d_in]`` (the ``B``/``A`` factors as
    stored), so the product is the ``[d_out, d_in]`` update the adapter adds to
    the frozen base weight. The ``alpha / (2*rank)`` scaling is this repo's
    pinned V5.47 convention (``geode.train.lora.LoRALinear.scaling`` /
    ``merge_lora``), deliberately NOT PEFT's ``alpha / rank``. Because ``B`` is
    zero-initialised, this is already ΔW from run init — no reference is
    subtracted.
    """
    return (alpha / (2 * rank)) * (b @ a)


def _module_name(tensor_key: str) -> str:
    """Drop a trailing ``.weight``/``.bias`` to name the owning module."""
    for suffix in (".weight", ".bias"):
        if tensor_key.endswith(suffix):
            return tensor_key[: -len(suffix)]
    return tensor_key


def _module_layer(module: str) -> int:
    """Map a module name to its transformer layer (embeddings/head/final norm -> -1)."""
    m = _LAYER_RE.search(module)
    return int(m.group(1)) if m else -1


def _load_state(run_id: str, step: int, store: Path) -> dict[str, torch.Tensor]:
    """Raw tensors of θ_step: base ∪ adapter (adapter wins), or a legacy full save.

    Mirrors ``geode.edl.loop.load_snapshot``'s file layout but needs no model:
    weight diffing is pure tensor math. New format merges the once-per-run
    ``snapshots/base/model.safetensors`` (absent for a pure legacy run) with the
    step's ``adapter.safetensors``; legacy snapshots are a single
    ``model.safetensors``.
    """
    snap_dir = run_dir(run_id, store=store) / "snapshots" / f"step_{step}"
    legacy = snap_dir / "model.safetensors"
    if legacy.is_file():
        return load_file(str(legacy))
    base_file = run_dir(run_id, store=store) / "snapshots" / "base" / "model.safetensors"
    base = load_file(str(base_file)) if base_file.is_file() else {}
    return base | load_file(str(snap_dir / "adapter.safetensors"))


def _discover_steps(snap_root: Path) -> list[int]:
    """Sorted step indices under ``snap_root`` carrying an adapter or full save."""
    return sorted(
        int(p.name.removeprefix("step_"))
        for p in snap_root.glob("step_*")
        if (p / "adapter.safetensors").is_file() or (p / "model.safetensors").is_file()
    )


def _full_ft_deltas(
    state: dict[str, torch.Tensor], ref_state: dict[str, torch.Tensor]
) -> list[_Delta]:
    """ΔW = W_k − W_ref for every floating-point tensor (2-D rank-bearing, 1-D not)."""
    deltas: list[_Delta] = []
    for key, tensor in state.items():
        if not tensor.is_floating_point() or key not in ref_state:
            continue
        dw = tensor.to(torch.float32) - ref_state[key].to(torch.float32)
        fro_sq = float(dw.pow(2).sum().item())
        module, layer = _module_name(key), _module_layer(key)
        deltas.append(_Delta(module, layer, dw if dw.ndim == 2 else None, fro_sq))
    return deltas


def _lora_deltas(state: dict[str, torch.Tensor], alpha: float, rank: int) -> list[_Delta]:
    """ΔW = (α/2r)·(B@A) per module carrying both ``.A.weight`` and ``.B.weight``."""
    factors: dict[str, dict[str, str]] = defaultdict(dict)
    for key in state:
        for role, suffix in (("A", ".A.weight"), ("B", ".B.weight")):
            if key.endswith(suffix):
                factors[key[: -len(suffix)]][role] = key
    deltas: list[_Delta] = []
    for module, pair in sorted(factors.items()):
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"lora module {module!r}: found only {sorted(pair)} (need A and B)")
        a = state[pair["A"]].to(torch.float32)
        b = state[pair["B"]].to(torch.float32)
        dw = lora_delta_w(b, a, alpha, rank)
        deltas.append(_Delta(module, _module_layer(module), dw, float(dw.pow(2).sum().item())))
    return deltas


def _step_rows(step: int, deltas: list[_Delta], common: dict) -> list[dict]:
    """The four metrics for one snapshot: per-module fro/rank, per-layer alloc, total."""
    rows: list[dict] = []
    total_sq = sum(d.fro_sq for d in deltas)
    layer_sq: dict[int, float] = defaultdict(float)
    for d in deltas:
        layer_sq[d.layer] += d.fro_sq
        if d.dw is None:  # 1-D: feeds the totals only, no rank
            continue
        rows.append(
            {
                **common,
                "checkpoint_step": step,
                "layer": d.layer,
                "module": d.module,
                "metric_name": "delta_w_fro",
                "metric_value": d.fro_sq**0.5,
            }
        )
        if d.fro_sq > 0.0:  # all-zero ΔW (e.g. the ref step) has no spectrum
            rows.append(
                {
                    **common,
                    "checkpoint_step": step,
                    "layer": d.layer,
                    "module": d.module,
                    "metric_name": "delta_w_effective_rank",
                    "metric_value": effective_rank(d.dw),
                }
            )
    for layer, sq in layer_sq.items():
        rows.append(
            {
                **common,
                "checkpoint_step": step,
                "layer": layer,
                "module": "all",
                "metric_name": "delta_w_layer_alloc",
                "metric_value": (sq / total_sq) if total_sq > 0.0 else 0.0,
            }
        )
    rows.append(
        {
            **common,
            "checkpoint_step": step,
            "layer": -1,
            "module": "all",
            "metric_name": "delta_w_fro_total",
            "metric_value": total_sq**0.5,
        }
    )
    return rows


def run_rows(run_id: str, store: Path, ref_step: int | None) -> list[dict]:
    manifest = load_run(run_id, store=store)
    method = manifest.data["training"]["method"]
    common = {
        "run_id": run_id,
        "base_model_key": manifest.data["base_model"]["hf_id"],
        "regime": manifest.data["regime"],
        "dataset_size": manifest.data["dataset"]["n_unique_examples"],
    }
    snap_root = run_dir(run_id, store=store) / "snapshots"
    steps = _discover_steps(snap_root)
    if not steps:
        raise FileNotFoundError(f"{run_id}: no snapshots under {snap_root}")

    ref_state: dict[str, torch.Tensor] = {}
    if method == "full_ft":
        ref = ref_step if ref_step is not None else steps[0]
        if ref not in steps:
            raise ValueError(f"{run_id}: --ref-step {ref} not among snapshots {steps}")
        ref_state = _load_state(run_id, ref, store)
    elif method == "lora":
        lora_cfg = manifest.data["training"]["lora"]
        alpha, rank = float(lora_cfg["alpha"]), int(lora_cfg["rank"])
    else:
        raise ValueError(f"{run_id}: unsupported training.method {method!r} (lora or full_ft)")

    rows: list[dict] = []
    for k in steps:
        state = _load_state(run_id, k, store)
        if method == "full_ft":
            deltas = _full_ft_deltas(state, ref_state)
        else:
            deltas = _lora_deltas(state, alpha, rank)
        rows.extend(_step_rows(k, deltas, common))
    print(f"[evt] {run_id}: {len(steps)} snapshots ({method}), {len(rows)} rows")
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    total = df[df["metric_name"] == "delta_w_fro_total"]
    per_mod = df[df["metric_name"] == "delta_w_fro"]
    for i, (rid, by_run) in enumerate(total.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        for _, by_mod in per_mod[per_mod["run_id"] == rid].groupby("module"):
            by_mod = by_mod.sort_values("checkpoint_step")
            axes[0].plot(
                by_mod["checkpoint_step"], by_mod["metric_value"], color=color, alpha=0.15, lw=0.8
            )
        by_run = by_run.sort_values("checkpoint_step")
        regime = by_run["regime"].iloc[0]
        axes[0].plot(
            by_run["checkpoint_step"],
            by_run["metric_value"],
            color=color,
            lw=2.0,
            label=f"{rid} ({regime})",
        )

    erank = df[df["metric_name"] == "delta_w_effective_rank"]
    for i, (rid, by_run) in enumerate(erank.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        for _, by_mod in by_run.groupby("module"):
            by_mod = by_mod.sort_values("checkpoint_step")
            axes[1].plot(
                by_mod["checkpoint_step"], by_mod["metric_value"], color=color, alpha=0.15, lw=0.8
            )
        mean = by_run.groupby("checkpoint_step")["metric_value"].mean()
        regime = by_run["regime"].iloc[0]
        axes[1].plot(mean.index, mean.values, color=color, lw=2.0, label=f"{rid} ({regime})")

    for ax, ylabel in zip(axes, ("‖ΔW‖_F", "ΔW effective rank")):
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("checkpoint step")
    axes[0].set_title("adapter ΔW (bold: total / module-mean, faint: per-module)")
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
        "--ref-step",
        type=int,
        default=None,
        help="full_ft reference snapshot (default: earliest discovered step)",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "adapter_diffs.png",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store, args.ref_step))
    df = pd.DataFrame(rows)
    path = write_results(df, "adapter_diffs", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    for rid, by_run in df.groupby("run_id", sort=True):
        last = int(by_run["checkpoint_step"].max())
        at_last = by_run[by_run["checkpoint_step"] == last]
        total = float(
            at_last.loc[at_last["metric_name"] == "delta_w_fro_total", "metric_value"].iloc[0]
        )
        erank = at_last.loc[
            at_last["metric_name"] == "delta_w_effective_rank", "metric_value"
        ].mean()
        alloc = at_last[at_last["metric_name"] == "delta_w_layer_alloc"].nlargest(3, "metric_value")
        top3 = ", ".join(f"L{int(r.layer)}={r.metric_value:.3f}" for r in alloc.itertuples())
        print(
            f"[evt] {rid} @ step {last}: ‖ΔW‖={total:.4f}, mean erank={erank:.2f}, "
            f"top-3 layers by alloc [{top3}]"
        )


if __name__ == "__main__":
    main()
