"""Linear-decodability analysis driver (spec 02 §7, elicit-vs-teach §9).

For each run's probe dumps (written by ``scripts/extract.py``), fits a linear
readout of the **first answer token** from the residual stream at the position
that generates it, at every snapshot and every residual hook point, and reports
held-out accuracy. Results land as one long-format parquet through the ZOO-4
writer (``results/linear_probes.parquet``, join key run_id/checkpoint_step
/layer, ``regime`` = elicit/teach); the overlay figure lands in
``analysis/figures/`` (gitignored).

Scientific question: elicitation predicts the answer is linearly decodable
EARLY (the capability is latent in the parent and training only surfaces it);
teaching predicts decodability arrives only as training builds it. The curve
to read is ``probe_test_acc`` vs step against the majority-class baseline.

Design notes:

- Feature/target. For probe example i, ``p_i`` = index of the first True in
  ``label_mask[i]`` (the first answer-label token); the target is
  ``y_i = input_ids[i, p_i]`` and the feature is the residual state at
  ``p_i - 1`` — the position whose next-token prediction *is* that answer
  token (causal LM; the same off-by-one the extraction guard enforces).
  Features are cast fp32; storage is bf16.
- Class space is fit ONCE, from the earliest dump of the first run, and reused
  everywhere — accuracies are only comparable across arms and steps if they
  are accuracies over the same label alphabet on the same examples. The frozen
  probe set is identical across arms by construction, so every dump of every
  run must match that reference's ``input_ids`` AND ``label_mask`` exactly;
  a mismatch refuses loudly rather than silently probing a different question.
- Row alignment (spec 00 §6): the parquet ``order_hash`` must equal the
  ``probe_set_hash`` of every dump used, same guard as ``drift.py``. The
  parquet is not otherwise read here (the labels come from the dumps).
- Split: one seeded permutation of n, first half train / second half test,
  computed once and reused for every (run, step, hook) — so every accuracy in
  the table is measured on the same held-out examples.
- Classifier: multinomial logistic regression on train-standardised features,
  zero-initialised (the objective is convex ⇒ no init-seed sensitivity),
  full-batch L-BFGS with strong-Wolfe line search, L2 on the weights only.
  fp32, CPU, torch-only (scikit-learn is not a repo dependency).
- Efficiency: reads ``acts.safetensors`` + ``probe_data.safetensors`` directly
  rather than ``load_probe_dump``, which would also pull ``grads`` — about
  half the dump bytes, unused here (same narrow-read reason as
  ``matching.py``). Features for all hooks are extracted first, then the acts
  are dropped, so the fits run with only ``[n, d]`` matrices resident.

Usage:
    python3 probes.py [--run-id <rid> ...] [--store <dir>] [--probe-local <path>]
        [--seed 0] [--l2 1e-3] [--every 1] [--fig figures/linear_probes.png]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from geode.arith import order_hash
from geode.probe import assert_probe_alignment, load_probe_dumps, residual_hook_names
from geode.probe.extract import ACTS_FILE, META_FILE, PROBE_DATA_FILE, ProbeMeta
from geode.zoo import load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")
METRIC_TEST = "probe_test_acc"
METRIC_TRAIN = "probe_train_acc"
PROBE_REPO = "mhieuuu/elicit-vs-teach-arith"
PROBE_FILE = "probe.parquet"
MAX_ITER = 150  # L-BFGS iterations per fit; one fit measured well under 2 s
GOOD_ACC = 0.9  # "decodable" threshold used only by the [evt] summary


def _load_probe(probe_local: Path | None) -> pd.DataFrame:
    """The frozen probe parquet — a local path, or downloaded from HF (spec §7)."""
    if probe_local is not None:
        path = probe_local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(PROBE_REPO, PROBE_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def _dumped_steps(run_id: str, store: Path) -> list[int]:
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = load_probe_dumps(probe_root, marker=META_FILE)
    if not steps:
        raise FileNotFoundError(
            f"{run_id}: no probe dumps under {probe_root} — run scripts/extract.py first"
        )
    return steps


def _load_dump(
    run_id: str, step: int, store: Path
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], ProbeMeta]:
    """One dump's acts + probe data + meta — never the grads (unused here)."""
    step_dir = run_dir(run_id, store=store) / "probe" / f"step_{step}"
    meta = ProbeMeta(**json.loads((step_dir / META_FILE).read_text()))
    return load_file(step_dir / ACTS_FILE), load_file(step_dir / PROBE_DATA_FILE), meta


def label_positions(label_mask: torch.Tensor) -> torch.Tensor:
    """First-True index per row of ``label_mask``; refuses empty or position-0 spans.

    A causal LM predicts token p from position p-1, so a label span starting at
    0 has no generating position (the same guard ``extract_probe`` applies at
    capture time).
    """
    n, seq = label_mask.shape
    idx = torch.arange(seq).expand(n, seq)
    pos = torch.where(label_mask.bool(), idx, torch.full_like(idx, seq)).min(dim=1).values
    empty = (pos == seq).nonzero().flatten().tolist()
    if empty:
        raise ValueError(
            f"label_mask rows {empty[:5]} have no label token — every probe example "
            "must have an answer span to decode"
        )
    at_zero = (pos == 0).nonzero().flatten().tolist()
    if at_zero:
        raise ValueError(
            f"label_mask rows {at_zero[:5]} start their label span at position 0, which "
            "no position generates (causal LM predicts token p from p-1)"
        )
    return pos


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    l2: float,
) -> tuple[float, float]:
    """Multinomial logistic regression; returns (train_acc, test_acc).

    Features are standardised with TRAIN mean/std (std clamped at 1e-6 so a
    constant feature cannot blow up); weights and bias start at exactly zero
    and the objective (cross-entropy + ``l2 * ||W||²``) is convex, so the fit
    is deterministic — no init seed enters.
    """
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    xt = (x_train - mean) / std
    xs = (x_test - mean) / std

    w = torch.zeros(xt.shape[1], n_classes, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=MAX_ITER, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(xt @ w + b, y_train) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        train_acc = ((xt @ w + b).argmax(dim=1) == y_train).double().mean().item()
        test_acc = ((xs @ w + b).argmax(dim=1) == y_test).double().mean().item()
    return train_acc, test_acc


class _Reference:
    """The globally shared probe question: positions, targets, class space, split.

    Built once from the earliest dump of the first run. Every other dump is
    asserted identical on ``input_ids`` and ``label_mask`` — the frozen probe
    set is the same for both arms and every snapshot, and accuracies are only
    comparable if the decoded question and the held-out examples are the same.
    """

    def __init__(self, data: dict[str, torch.Tensor], seed: int, label: str) -> None:
        self.label = label
        self.input_ids = data["input_ids"]
        self.label_mask = data["label_mask"]
        self.pos = label_positions(self.label_mask)
        targets = self.input_ids[torch.arange(self.pos.shape[0]), self.pos]
        self.class_ids = torch.unique(targets).sort().values
        lookup = {int(t): i for i, t in enumerate(self.class_ids.tolist())}
        self.y = torch.tensor([lookup[int(t)] for t in targets.tolist()], dtype=torch.long)
        self.n_classes = int(self.class_ids.shape[0])

        n = int(self.y.shape[0])
        gen = torch.Generator()
        gen.manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
        self.train_idx = perm[: n // 2]
        self.test_idx = perm[n // 2 :]
        counts = torch.bincount(self.y[self.test_idx], minlength=self.n_classes)
        self.majority_test_acc = float(counts.max().item()) / float(self.test_idx.shape[0])

    def assert_matches(self, run_id: str, step: int, data: dict[str, torch.Tensor]) -> None:
        for field in ("input_ids", "label_mask"):
            if not torch.equal(data[field], getattr(self, field)):
                raise ValueError(
                    f"{run_id}@step_{step}: {field} differ from the reference dump "
                    f"({self.label}) — the frozen probe set must be identical across "
                    "arms and snapshots for decodability to be comparable"
                )


def run_rows(
    run_id: str,
    store: Path,
    probe_hash: str,
    ref: _Reference,
    every: int,
    l2: float,
) -> list[dict]:
    """One run's rows: a probe fit per (dumped step, hook point)."""
    steps = _dumped_steps(run_id, store)[::every]
    manifest = load_run(run_id, store=store)
    regime = manifest.data["regime"]
    dataset_size = manifest.data["dataset"]["n_unique_examples"]

    rows: list[dict] = []
    total = len(steps)
    t0 = time.time()
    n_fits = 0
    for i, k in enumerate(steps, start=1):
        acts, data, meta = _load_dump(run_id, k, store)
        assert_probe_alignment(meta.probe_set_hash, probe_hash, run_id=run_id, step=k)
        ref.assert_matches(run_id, k, data)
        names = residual_hook_names(len(acts) - 1)
        # Extract the [n, d] feature matrices first, then drop the dump: the
        # fits below need only these, and the acts are ~0.5 GiB.
        rowsel = torch.arange(ref.pos.shape[0])
        feats = {name: acts[name][rowsel, ref.pos - 1, :].to(torch.float32) for name in names}
        del acts, data
        if i % 10 == 0 or i == 1 or i == total:
            print(f"[evt] {run_id}: step {k} ({i}/{total})")
        for layer, name in enumerate(names):
            train_acc, test_acc = fit_linear_probe(
                feats[name][ref.train_idx],
                ref.y[ref.train_idx],
                feats[name][ref.test_idx],
                ref.y[ref.test_idx],
                ref.n_classes,
                l2,
            )
            n_fits += 1
            base = {
                "run_id": run_id,
                "base_model_key": meta.base_model_key,
                "regime": regime,
                "dataset_size": dataset_size,
                "checkpoint_step": k,
                "layer": layer,
                "hook_name": name,
                "arm": meta.arm,
                "n_classes": ref.n_classes,
                "majority_test_acc": ref.majority_test_acc,
            }
            rows.append(
                {
                    **base,
                    "metric_name": METRIC_TEST,
                    "metric_value": test_acc,
                    "n_examples": int(ref.test_idx.shape[0]),
                }
            )
            rows.append(
                {
                    **base,
                    "metric_name": METRIC_TRAIN,
                    "metric_value": train_acc,
                    "n_examples": int(ref.train_idx.shape[0]),
                }
            )
        del feats
    elapsed = time.time() - t0
    print(
        f"[evt] {run_id}: {total} dumps, {n_fits} probe fits in {elapsed:.1f}s "
        f"({elapsed / max(n_fits, 1):.2f}s/fit)"
    )
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sub = df[df["metric_name"] == METRIC_TEST]
    for i, (rid, by_run) in enumerate(sub.groupby("run_id", sort=True)):
        color = colors[i % len(colors)]
        for _, by_hook in by_run.groupby("hook_name"):
            by_hook = by_hook.sort_values("checkpoint_step")
            ax.plot(
                by_hook["checkpoint_step"], by_hook["metric_value"], color=color, alpha=0.15, lw=0.8
            )
        mean = by_run.groupby("checkpoint_step")["metric_value"].mean()
        regime = by_run["regime"].iloc[0]
        ax.plot(mean.index, mean.values, color=color, lw=2.0, label=f"{rid} ({regime})")
    baseline = float(sub["majority_test_acc"].iloc[0])
    ax.axhline(baseline, ls="--", color="gray", lw=1.0, label=f"majority baseline ({baseline:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("linear probe test accuracy (first answer token)")
    ax.set_title("linear decodability of the answer (faint: hooks, bold: hook mean)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def print_summary(df: pd.DataFrame) -> None:
    sub = df[df["metric_name"] == METRIC_TEST]
    baseline = float(sub["majority_test_acc"].iloc[0])
    print(f"[evt] majority baseline (test split): {baseline:.4f}")
    for rid, by_run in sub.groupby("run_id", sort=True):
        regime = by_run["regime"].iloc[0]
        mean = by_run.groupby("checkpoint_step")["metric_value"].mean().sort_index()
        first, last = int(mean.index[0]), int(mean.index[-1])
        good = mean[mean > GOOD_ACC]
        when = f"step {int(good.index[0])}" if len(good) else "never"
        print(
            f"[evt] {rid} ({regime}): hook-mean test acc {mean.iloc[0]:.4f} @ step {first} -> "
            f"{mean.iloc[-1]:.4f} @ step {last}; first > {GOOD_ACC}: {when}"
        )


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
        "--probe-local", type=Path, default=None, help="skip the HF download of probe.parquet"
    )
    ap.add_argument("--seed", type=int, default=0, help="train/test split permutation seed")
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on the readout weights")
    ap.add_argument(
        "--every", type=int, default=1, help="process every k-th dumped step (cheap reruns)"
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "linear_probes.png",
    )
    args = ap.parse_args()
    if args.every < 1:
        ap.error("--every must be >= 1")
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    df_probe = _load_probe(args.probe_local)
    probe_hash = order_hash(df_probe.to_dict("records"))
    print(f"[evt] probe set: {len(df_probe)} examples, probe_set_hash={probe_hash[:12]}…")

    ref_run = run_ids[0]
    ref_step = _dumped_steps(ref_run, args.store)[0]
    _, ref_data, ref_meta = _load_dump(ref_run, ref_step, args.store)
    assert_probe_alignment(ref_meta.probe_set_hash, probe_hash, run_id=ref_run, step=ref_step)
    ref = _Reference(ref_data, args.seed, f"{ref_run}@step_{ref_step}")
    del ref_data
    print(
        f"[evt] reference {ref.label}: n={int(ref.y.shape[0])}, {ref.n_classes} target classes, "
        f"split {int(ref.train_idx.shape[0])} train / {int(ref.test_idx.shape[0])} test, "
        f"majority test acc {ref.majority_test_acc:.4f}"
    )

    rows: list[dict] = []
    for rid in run_ids:
        rows.extend(run_rows(rid, args.store, probe_hash, ref, args.every, args.l2))
    df = pd.DataFrame(rows)
    path = write_results(df, "linear_probes", store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)
    print_summary(df)


if __name__ == "__main__":
    main()
