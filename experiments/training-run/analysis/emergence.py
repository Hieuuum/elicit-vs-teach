"""Steering-direction emergence driver (spec 02 §7, runs 7/8).

``steering.py`` takes ONE direction per hook — the mean pooled-activation shift
between the earliest and the final probe dump — and shows causally that arm A's
version of it recovers ~14% of the target task when injected into arm A's
parent. This driver asks the timing question that experiment leaves open:
**when during training does that direction come into being?**

For every dumped snapshot k it forms the partial direction

    direction_k = mean_i [ pooled_i(dump k) − pooled_i(dump ref) ]

with exactly the pooling ``steering.py`` uses (fp64 mask-mean over label
positions, via the shared ``pooled_label_mean``), and compares it to the run's
final direction ``direction_last`` — the vector the steering square actually
injected. Per snapshot and hook point:

- ``dir_cos_to_final``    cos(direction_k, direction_last) — is the run already
  pointing where it ends up, regardless of how far it has gone?
- ``dir_norm_frac``       ‖direction_k‖ / ‖direction_last‖ — how far it has gone.
- ``dir_progress``        ⟨direction_k, direction_last⟩ / ‖direction_last‖² —
  the signed fraction of the final displacement already achieved *along* it
  (= cos × norm_frac). 0 at the reference, exactly 1 at the last snapshot.
- ``dir_cos_split_half``  cos(direction_k over examples [0, n/2),
  direction_k over [n/2, n)) — the measurement's own reproducibility.

Because ``direction_k`` is a mean over the SAME frozen probe examples pooled
through the SAME mask at every snapshot, it collapses to a difference of global
means, μ_k − μ_ref; the input_ids and probe-hash guards below are what make
that identity hold, so they are not optional.

Design notes:

- **Split-half is the noise floor that matters.** A cosine only means something
  above the level at which the measurement reproduces itself. Comparing
  ``dir_cos_to_final`` against a random vector's 1/√d is the weaker test:
  ``dir_cos_split_half`` says what half the probe set thinks the other half's
  direction is at that same step, so wherever the two curves meet, the
  cos-to-final reading is noise and the direction is not yet resolved. Both
  halves reuse the reference's own halves, so the split is a partition of the
  measurement, not of the model.
- **Reference snapshot**: the earliest dump on disk (``--ref-step`` overrides),
  which is step 1 — one optimizer update in, the closest available stand-in for
  init, not init itself. Its rows exist and are identically 0 (norm_frac,
  progress); the undefined ratios (both cosines, direction is the zero vector)
  emit no row rather than a NaN.
- **The last snapshot is a self-check, asserted not just emitted**: cos and
  progress must both be 1.0 to fp tolerance there by construction, so a
  ref/last bookkeeping slip fails loudly instead of shifting the whole curve.
- **Never-moving hooks emit no cosine.** Under LoRA the embedding table is
  frozen, so ``hook_embed``'s activations are bit-identical at every snapshot
  and its direction is exactly the zero vector — every ratio against it is
  0/0. Those hooks emit the two magnitude rows (truthfully 0.0) and no
  cosines, which also keeps them out of the hook-mean curves rather than
  dragging them toward 0. A second assertion checks such a hook never moved
  *mid-run* either, so a zero endpoint that is really a cancellation cannot
  masquerade as a frozen parameter.
- **Cross-arm timing is NOT read off ``train_frac``.** Runs 7 and 8 have
  different dump schedules (stride ~9 vs ~11) and different lengths (5991 vs
  10969), so step fraction conflates "earlier" with "shorter". Compare arms at
  matched capability via ``results/performance_matching.parquet`` (V5.16) at
  report time; this driver emits ``train_frac`` for within-run reading only.
- **The two arms' curves do not carry the same claim.** Arm A's final direction
  is causally potent (steering square: 0.1406 EM vs random 0.0273); arm B's is
  behaviourally inert on the same target (0.0352 ≈ random). So arm A's curve
  times the emergence of a direction that *does something*, while arm B's times
  the stabilisation of its net displacement — a geometry fact, not an
  elicitation one. Do not report them symmetrically.
- Directions live in each run's own parent's residual basis, and the two
  parents are separately fine-tuned models; cross-run cosines between arms are
  therefore not interpretable here (the steering square measured them at
  −0.05..+0.21) and this driver does not compute any.

CPU-only, dumps are memory-mapped and only ``acts`` is touched (the equally
large ``grads`` file is never paged in).

Usage:
    python3 emergence.py [--run-id <rid> ...] [--store <dir>] [--ref-step <k>]
        [--probe-local <path>] [--fig figures/direction_emergence.png]
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geode.arith import order_hash
from geode.probe import load_probe_dump, residual_hook_names
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

# The direction's pooling semantics must be the ones steering.py injected, not
# a re-derivation: if that changes, this curve has to change with it. Sibling
# import, as trajectory.py takes its deltas from adapters.py.
from steering import _guard_probe_hash, _load_probe, pooled_label_mean  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
RESULTS_NAME = "direction_emergence"
METRIC_COS = "dir_cos_to_final"
METRIC_NORM = "dir_norm_frac"
METRIC_PROGRESS = "dir_progress"
METRIC_SPLIT = "dir_cos_split_half"
SELF_CHECK_TOL = 1e-9


def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    """Cosine between two fp64 vectors; 0.0 if either is degenerate."""
    nu, nv = float(u.norm().item()), float(v.norm().item())
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(torch.dot(u, v).item()) / (nu * nv)


def _snapshot_means(
    acts: dict[str, torch.Tensor], mask: torch.Tensor, names: list[str], half: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per hook: (mean over all probe examples, mean over half A, mean over half B).

    One pooling pass per hook; the halves are means of the same ``[n, d]``
    pooled matrix, so the split costs nothing extra.
    """
    out: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name in names:
        pooled = pooled_label_mean(acts[name], mask)
        out[name] = (pooled.mean(dim=0), pooled[:half].mean(dim=0), pooled[half:].mean(dim=0))
    return out


def run_rows(run_id: str, store: Path, probe_hash: str, ref_step: int | None) -> list[dict]:
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = sorted(
        int(p.name.removeprefix("step_"))
        for p in probe_root.glob("step_*")
        if (p / "meta.json").is_file()
    )
    if len(steps) < 2:
        raise FileNotFoundError(
            f"{run_id}: need >= 2 probe dumps under {probe_root} to take a direction, "
            f"found {len(steps)} — run scripts/extract.py first"
        )
    ref = steps[0] if ref_step is None else ref_step
    if ref not in steps:
        raise ValueError(f"{run_id}: --ref-step {ref} not among dumped steps {steps}")
    if ref == steps[-1]:
        raise ValueError(f"{run_id}: --ref-step {ref} is the last dump — no direction to take")
    manifest = load_run(run_id, store=store)
    regime = manifest.data["regime"]
    dataset_size = manifest.data["dataset"]["n_unique_examples"]

    ref_capture, ref_meta = load_probe_dump(run_id, ref, store=store)
    _guard_probe_hash(run_id, ref, ref_meta.probe_set_hash, probe_hash)
    names = residual_hook_names(len(ref_capture.acts) - 1)
    mask = ref_capture.label_mask
    n_examples = int(mask.shape[0])
    half = n_examples // 2
    ref_means = _snapshot_means(ref_capture.acts, mask, names, half)

    # Pass 1: the pooled means per snapshot. Each dump is ~0.25 GiB of acts and
    # is dropped as soon as it is reduced to 9 x 3 d-vectors.
    means: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for k in steps:
        if k == ref:
            capture, meta = ref_capture, ref_meta
        else:
            capture, meta = load_probe_dump(run_id, k, store=store)
        _guard_probe_hash(run_id, k, meta.probe_set_hash, probe_hash)
        if not torch.equal(capture.input_ids, ref_capture.input_ids):
            raise ValueError(
                f"{run_id}: step {k} input_ids differ from ref step {ref} — the probe "
                "tokenization must be identical across snapshots to pool one mask"
            )
        means[k] = _snapshot_means(capture.acts, mask, names, half)
        if k != ref:
            del capture

    # Pass 2: directions relative to ref, compared against the final direction.
    last = steps[-1]
    final_dirs = {name: means[last][name][0] - ref_means[name][0] for name in names}
    final_norms = {name: float(final_dirs[name].norm().item()) for name in names}
    # A hook whose final direction is EXACTLY zero never moved at all: under
    # LoRA the embedding table is frozen, so identical probe input_ids give
    # bit-identical hook_embed activations at every snapshot (the steering
    # square saw the same thing — direction_norm 0 there, and alpha scaled it
    # to nothing). Every ratio against it is 0/0, so those hooks emit the two
    # magnitude rows (truthfully 0.0: no movement) and no cosine rows.
    degenerate = {name for name in names if final_norms[name] == 0.0}
    rows: list[dict] = []
    for k in steps:
        for layer, name in enumerate(names):
            base = {
                "run_id": run_id,
                "base_model_key": ref_meta.base_model_key,
                "regime": regime,
                "dataset_size": dataset_size,
                "checkpoint_step": k,
                "train_frac": k / last,
                "layer": layer,
                "hook_name": name,
                "arm": ref_meta.arm,
                "ref_step": ref,
                "final_step": last,
                "n_examples": n_examples,
            }
            final = final_dirs[name]
            final_norm = final_norms[name]
            direction = means[k][name][0] - ref_means[name][0]
            rows.append(
                {
                    **base,
                    "metric_name": METRIC_NORM,
                    "metric_value": (
                        float(direction.norm().item()) / final_norm if final_norm > 0 else 0.0
                    ),
                }
            )
            rows.append(
                {
                    **base,
                    "metric_name": METRIC_PROGRESS,
                    "metric_value": (
                        float(torch.dot(direction, final).item()) / (final_norm**2)
                        if final_norm > 0
                        else 0.0
                    ),
                }
            )
            if k == ref or name in degenerate:
                continue  # zero direction: both cosines are 0/0, emit nothing
            rows.append({**base, "metric_name": METRIC_COS, "metric_value": _cos(direction, final)})
            dir_a = means[k][name][1] - ref_means[name][1]
            dir_b = means[k][name][2] - ref_means[name][2]
            rows.append({**base, "metric_name": METRIC_SPLIT, "metric_value": _cos(dir_a, dir_b)})

    # Self-check 1: at the last snapshot the direction IS the final direction,
    # so all three live metrics are 1.0 there by construction.
    final_rows = [
        r for r in rows if r["checkpoint_step"] == last and r["hook_name"] not in degenerate
    ]
    for metric in (METRIC_COS, METRIC_PROGRESS, METRIC_NORM):
        vals = [r["metric_value"] for r in final_rows if r["metric_name"] == metric]
        worst = max(abs(v - 1.0) for v in vals)
        if worst > SELF_CHECK_TOL:
            raise AssertionError(
                f"{run_id}: {metric} at the final snapshot (step {last}) is not 1.0 "
                f"(max deviation {worst:.3e}) — the reference/final bookkeeping is wrong"
            )
    # Self-check 2: a hook that ends where it started must have been there the
    # whole time. A frozen table cannot move and return; a nonzero intermediate
    # would mean the zero endpoint is a cancellation, not a frozen parameter.
    for name in sorted(degenerate):
        moved = max(float((means[k][name][0] - ref_means[name][0]).norm().item()) for k in steps)
        if moved > 0.0:
            raise AssertionError(
                f"{run_id}: {name} has a zero FINAL direction but moves to {moved:.3e} "
                "mid-run — that endpoint is a cancellation, not a frozen parameter"
            )
    d_model = int(final_dirs[names[0]].shape[0])
    frozen = ", ".join(sorted(degenerate)) if degenerate else "none"
    print(
        f"[evt] {run_id} ({regime}): {len(steps)} dumps, ref step {ref} → final {last}, "
        f"{len(names)} hooks, {n_examples} probe examples (split {half}/{n_examples - half}), "
        f"d_model {d_model}, random-vector cosine scale 1/√d = {1 / math.sqrt(d_model):.4f}; "
        f"never-moving hooks (no cosine rows): {frozen}"
    )
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 12))
    runs = sorted(df["run_id"].unique())
    colors = {rid: c for rid, c in zip(runs, ("tab:blue", "tab:orange"))}

    def hook_mean(rid: str, metric: str) -> pd.DataFrame:
        sub = df[(df["run_id"] == rid) & (df["metric_name"] == metric)]
        return sub.groupby("checkpoint_step")["metric_value"].mean().reset_index()

    for rid in runs:
        regime = df[df["run_id"] == rid]["regime"].iloc[0]
        cos, split = hook_mean(rid, METRIC_COS), hook_mean(rid, METRIC_SPLIT)
        axes[0].plot(
            cos["checkpoint_step"], cos["metric_value"], color=colors[rid], label=f"{regime} {rid}"
        )
        axes[0].plot(
            split["checkpoint_step"],
            split["metric_value"],
            color=colors[rid],
            ls=":",
            alpha=0.7,
            label=f"{regime} split-half (noise ceiling)",
        )
        prog = hook_mean(rid, METRIC_PROGRESS)
        axes[1].plot(prog["checkpoint_step"], prog["metric_value"], color=colors[rid], label=regime)
        norm = hook_mean(rid, METRIC_NORM)
        axes[1].plot(
            norm["checkpoint_step"],
            norm["metric_value"],
            color=colors[rid],
            ls="--",
            alpha=0.6,
            label=f"{regime} ‖·‖ fraction",
        )
    axes[0].set_ylabel("cos(direction_k, final direction)")
    axes[0].set_title(
        "Direction emergence: when does the steered direction come into being?\n"
        "(solid: cos to final / dotted: split-half reliability of the same measurement)"
    )
    axes[1].set_ylabel("fraction of the final direction")
    axes[1].set_title("progress along the final direction (solid) and its norm fraction (dashed)")
    for ax in axes[:2]:
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)

    cmap = plt.get_cmap("viridis")
    layers = sorted(int(x) for x in df["layer"].unique())
    denom = max(layers) or 1
    for rid, style in zip(runs, ("-", "--")):
        sub = df[(df["run_id"] == rid) & (df["metric_name"] == METRIC_COS)]
        for layer, g in sub.groupby("layer"):
            g = g.sort_values("checkpoint_step")
            axes[2].plot(
                g["train_frac"],
                g["metric_value"],
                color=cmap(int(layer) / denom),
                ls=style,
                lw=1.2,
                alpha=0.85,
                label=f"L{int(layer)}" if style == "-" else None,
            )
    axes[2].set_xlabel("fraction of the run's own training (within-run reading only)")
    axes[2].set_ylabel("cos to final direction")
    axes[2].set_title(f"per hook point — solid: {runs[0]}, dashed: {runs[-1]}")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=6, ncol=3)
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
        help="reference snapshot step (default: earliest dumped step)",
    )
    ap.add_argument(
        "--probe-local", type=Path, default=None, help="skip the HF download of probe.parquet"
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "direction_emergence.png",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    df_probe = _load_probe(args.probe_local)
    probe_hash = order_hash(df_probe.to_dict("records"))
    print(f"[evt] probe set: {len(df_probe)} examples, probe_set_hash={probe_hash[:12]}…")

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store, probe_hash, args.ref_step))
    df = pd.DataFrame(rows)
    path = write_results(df, RESULTS_NAME, store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    for rid, by_run in df.groupby("run_id", sort=True):
        regime = by_run["regime"].iloc[0]
        cos = (
            by_run[by_run["metric_name"] == METRIC_COS]
            .groupby("checkpoint_step")["metric_value"]
            .mean()
        )
        split = (
            by_run[by_run["metric_name"] == METRIC_SPLIT]
            .groupby("checkpoint_step")["metric_value"]
            .mean()
        )
        prog = (
            by_run[by_run["metric_name"] == METRIC_PROGRESS]
            .groupby("checkpoint_step")["metric_value"]
            .mean()
        )
        last = int(by_run["checkpoint_step"].max())
        for label, series, thresh in (
            ("cos>=0.5", cos, 0.5),
            ("cos>=0.9", cos, 0.9),
            ("progress>=0.5", prog, 0.5),
            ("progress>=0.9", prog, 0.9),
        ):
            hit = series[series >= thresh]
            when = (
                f"step {int(hit.index[0])} ({hit.index[0] / last:.1%} of the run)"
                if len(hit)
                else "never"
            )
            print(f"[evt] {rid} ({regime}): first {label} at {when}")
        # Where the direction first clears its own measurement noise, and stays clear.
        common = cos.index.intersection(split.index)
        clear = (cos[common] > split[common]).sort_index()
        streak = clear[::-1].cummin()[::-1]  # True only if True from here to the end
        resolved = streak[streak]
        first = (
            f"step {int(resolved.index[0])} ({resolved.index[0] / last:.1%})"
            if len(resolved)
            else "never"
        )
        print(
            f"[evt] {rid} ({regime}): cos exceeds split-half reliability from {first} onward; "
            f"split-half at the earliest step {float(split.iloc[0]):.4f}, at the end "
            f"{float(split.iloc[-1]):.4f}"
        )


if __name__ == "__main__":
    main()
