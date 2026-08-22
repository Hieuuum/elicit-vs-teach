"""Residual-stream linear probe for the sum (Phase 0 / Tier 1 test 1): is the
answer linearly decodable from the residual stream at theta0, even where the
model's own zero-shot generation is near-zero (decisions.md 2026-08-16
evening ts38pp OUTCOME)?

For one model (or every snapshot of one run) and one or more prompt sets,
fits a multinomial logistic-regression probe of the FIRST answer token from
the residual stream at the position that generates it (``label_span[0] - 1``
— same off-by-one as ``logit_lens.position_target_pairs``'s ``first_answer``
target), at every residual-hook point (``mech_lib``'s layer convention: 0 =
``hook_embed``, i+1 = ``blocks.{i}.hook_resid_post``). ``fit_linear_probe``
below is copied verbatim from ``probes.py`` (multinomial LBFGS logistic
regression on train-standardised features) rather than imported: importing
``probes.py`` pulls in ``matplotlib`` at module load, which is a ``dev``-only
optional dependency (``pyproject.toml``), not a runtime one — the other three
Phase-0 drivers (``logit_lens.py``/``weight_diff.py``/``resid_shift.py``,
meant to run "as soon as any box exists") have no such dependency, and this
one should not be the exception. ``probes.py``'s ``label_positions`` is not
reused either: it decodes a position from a ``label_mask`` tensor written by
the probe-dump extractor, and every example here already carries its
position directly as ``SftExample.label_span``.

**Test 1 discriminator (pre-registered, not optional — decisions.md
2026-08-21 night "ts38mt pre-registration").** The sum is a deterministic
function of the operand tokens, so a linear probe with enough capacity can in
principle *compute* it rather than *read* a computed representation. The
layer-0 (post-embedding, pre-compute) accuracy is reported as an explicit
FLOOR: a probe that already scores high at layer 0 is reading raw digit
tokens, not a computed representation, and that reading undercuts a "latent
capability" claim regardless of how high later-layer accuracy climbs.
Task-probe accuracy at theta0 must be judged against that floor and against
the op-notation positive control's ceiling (meaningful only on the elicit/pp
parent — the teach/fmt and base parents were never taught the op mapping
either), never in absolute terms.

Owner's Tier-1 signature table (EXPERIMENTS.md §6.22, decisions.md same
entry): elicit = probe accuracy high at theta0, well above the layer-0
floor; teach = low / near-chance at theta0.

Class space and the seeded half/half train-test split are fixed ONCE per
prompt set (``build_reference``, same discipline as ``probes.py``'s
``_Reference``) and reused across every layer and every checkpoint, so
accuracies are comparable.

Two ways to pick the model(s):

- ``--model run:<run_id>`` or ``--model dir:<path>``: one theta, one probe
  pass. ``checkpoint_step`` is recorded as ``-1`` (there is no snapshot step
  for a plain theta load — picked over NaN so the column stays integer).
- ``--run-id <id> [--snapshot-steps 1,2,4,...]``: every (or the selected)
  snapshot of a LoRA target run (``runs/<id>/snapshots/step_*/adapter
  .safetensors`` — the LoRA-wrapped model tree is built once via
  ``geode.zoo.load_model`` and mutated in place per step by
  ``geode.edl.loop.load_snapshot``) or a full-FT parent
  (``runs/<id>/sft_snapshots/step_*/`` — plain ``save_pretrained`` dirs,
  loaded fresh per step via ``dir:``). Default steps: every ``step_*`` dir
  carrying its kind's marker file, under whichever of those two directories
  has any (``discover_snapshot_dir``/``snapshot_steps_in_dir``); an explicit
  ``--snapshot-steps`` is checked against that same usable set up front.

Usage:
    python3 resid_probe.py --model run:evt-ts38pp-parent \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --prompt-parquet ../data/full/probe.parquet --set-name op \\
        --out resid_probe_ts38pp_theta0.csv

    python3 resid_probe.py --run-id evt-ts38mt-pp-n21544 \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --out resid_probe_ts38mt_pp_n21544.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from mech_lib import (
    capture_residuals,
    load_any_model,
    load_task_examples,
    pad_examples,
    write_table,
)

from geode.arith.spans import SftExample
from geode.edl.loop import load_snapshot
from geode.zoo import load_model
from geode.zoo.store import run_dir

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"
NO_CHECKPOINT_STEP = -1  # a plain theta load has no snapshot step (documented above)
MAX_ITER = 150  # L-BFGS iterations per fit — probes.py's own measured budget
SNAPSHOT_MARKER = {"lora": "adapter.safetensors", "full_ft": "model.safetensors"}


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    l2: float,
) -> tuple[float, float]:
    """Multinomial logistic regression; returns (train_acc, test_acc).

    Copied verbatim from ``probes.py`` (module docstring explains why, not
    imported). Features are standardised with TRAIN mean/std (std clamped at
    1e-6 so a constant feature cannot blow up); weights and bias start at
    exactly zero and the objective (cross-entropy + ``l2 * ||W||^2``) is
    convex, so the fit is deterministic — no init seed enters.
    """
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    xt = (x_train - mean) / std
    xs = (x_test - mean) / std
    y_train = y_train.to(xt.device)
    y_test = y_test.to(xt.device)

    w = torch.zeros(
        xt.shape[1], n_classes, dtype=torch.float32, device=xt.device, requires_grad=True
    )
    b = torch.zeros(n_classes, dtype=torch.float32, device=xt.device, requires_grad=True)
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


@dataclass
class ProbeReference:
    """The frozen probe question for one prompt set: examples, target class
    ids, and a seeded half/half train-test split — built ONCE and reused for
    every model, checkpoint, and layer probed on this set."""

    set_name: str
    examples: list[SftExample]
    y: torch.Tensor
    class_ids: torch.Tensor
    n_classes: int
    train_idx: torch.Tensor
    test_idx: torch.Tensor
    majority_test_acc: float


def feature_positions(examples: Sequence[SftExample]) -> list[int]:
    """The residual position each example's probe feature is read from:
    ``label_span[0] - 1`` — a causal LM predicts the token at ``p`` from
    position ``p - 1``, the same off-by-one ``logit_lens.position_target_pairs``
    uses for its ``first_answer`` positions."""
    return [ex.label_span[0] - 1 for ex in examples]


def build_reference(examples: Sequence[SftExample], set_name: str, seed: int) -> ProbeReference:
    """Target classes (the FIRST answer token id, ``input_ids[label_span[0]]``)
    plus a seeded half/half train-test permutation split for one prompt set —
    ``probes.py``'s ``_Reference`` discipline, without the ``label_mask``
    decode step (every example here already carries ``label_span`` directly).
    """
    if not examples:
        raise ValueError(f"build_reference: empty example list for set {set_name!r}")
    targets = torch.tensor([ex.input_ids[ex.label_span[0]] for ex in examples], dtype=torch.long)
    class_ids = torch.unique(targets).sort().values
    lookup = {int(t): i for i, t in enumerate(class_ids.tolist())}
    y = torch.tensor([lookup[int(t)] for t in targets.tolist()], dtype=torch.long)
    n_classes = int(class_ids.shape[0])

    n = len(examples)
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    train_idx = perm[: n // 2]
    test_idx = perm[n // 2 :]
    counts = torch.bincount(y[test_idx], minlength=n_classes)
    majority_test_acc = float(counts.max().item()) / float(test_idx.shape[0])

    return ProbeReference(
        set_name=set_name,
        examples=list(examples),
        y=y,
        class_ids=class_ids,
        n_classes=n_classes,
        train_idx=train_idx,
        test_idx=test_idx,
        majority_test_acc=majority_test_acc,
    )


def extract_features(model: nn.Module, examples: Sequence[SftExample]) -> dict[str, torch.Tensor]:
    """Residual-stream feature per layer at ``feature_positions(examples)``,
    one row per example, in ``mech_lib.capture_residuals``'s layer order."""
    if not examples:
        raise ValueError("extract_features: empty example list")
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_examples(examples, device=str(device))
    acts = capture_residuals(model, input_ids, attention_mask)
    ex_idx = torch.arange(len(examples), device=device)
    pos = torch.tensor(feature_positions(examples), device=device)
    return {name: acts[name][ex_idx, pos, :].to(torch.float32) for name in acts}


def probe_rows_for_model(
    model: nn.Module,
    refs: dict[str, ProbeReference],
    model_label: str,
    checkpoint_step: int,
    l2: float = 1e-3,
) -> list[dict]:
    """One row per (set, layer) for this model/checkpoint: refit the probe at
    every layer on each set's FIXED reference split (only the features come
    from this model's own forward pass; class space and split are frozen)."""
    rows: list[dict] = []
    for set_name, ref in refs.items():
        feats = extract_features(model, ref.examples)
        for layer, name in enumerate(feats):
            train_acc, test_acc = fit_linear_probe(
                feats[name][ref.train_idx],
                ref.y[ref.train_idx],
                feats[name][ref.test_idx],
                ref.y[ref.test_idx],
                ref.n_classes,
                l2,
            )
            rows.append(
                {
                    "model": model_label,
                    "checkpoint_step": checkpoint_step,
                    "set": set_name,
                    "layer": layer,
                    "hook_name": name,
                    "probe_train_acc": train_acc,
                    "probe_test_acc": test_acc,
                    "majority_test_acc": ref.majority_test_acc,
                    "n_classes": ref.n_classes,
                    "n_train": int(ref.train_idx.shape[0]),
                    "n_test": int(ref.test_idx.shape[0]),
                }
            )
    return rows


def snapshot_steps_in_dir(snap_dir: Path, marker: str) -> list[int]:
    """Sorted step ints from ``step_<k>`` subdirectories of ``snap_dir`` that
    carry ``marker`` (``adapter.safetensors`` for a LoRA run,
    ``model.safetensors`` for a full-FT parent — ``SNAPSHOT_MARKER``). A
    ``step_*`` dir without the marker is a killed-mid-write snapshot, not a
    usable checkpoint, and is skipped rather than raising mid-sweep — the
    same discipline ``train_target.py``'s own ``snapshots_taken`` glob uses
    (``if (p / "adapter.safetensors").is_file()``). A non-numeric sibling —
    e.g. the LoRA schedule's ``base/`` sidecar — is skipped too."""
    steps: list[int] = []
    for p in snap_dir.glob("step_*"):
        if not p.is_dir():
            continue
        suffix = p.name.removeprefix("step_")
        if suffix.isdigit() and (p / marker).is_file():
            steps.append(int(suffix))
    if not steps:
        raise FileNotFoundError(f"{snap_dir}: no usable step_<k>/{marker} snapshots found")
    return sorted(steps)


def discover_snapshot_dir(run_id: str, store: Path | None) -> tuple[Path, str]:
    """Which snapshot directory this run has USABLE snapshots in: LoRA
    target runs write ``runs/<id>/snapshots/step_*/adapter.safetensors``;
    full-FT parents write ``runs/<id>/sft_snapshots/step_*/model
    .safetensors`` (plain ``save_pretrained`` dirs). Returns ``(dir, kind)``
    with ``kind`` in ``{"lora", "full_ft"}``. Dispatches on which directory
    actually has a usable ``step_*``, not on bare directory existence — an
    empty or killed-mid-write ``snapshots/`` must fall through to
    ``sft_snapshots/`` rather than winning by existing first."""
    base = run_dir(run_id, store=store)
    candidates = (("lora", base / "snapshots"), ("full_ft", base / "sft_snapshots"))
    for kind, snap_dir in candidates:
        if not snap_dir.is_dir():
            continue
        try:
            snapshot_steps_in_dir(snap_dir, SNAPSHOT_MARKER[kind])
        except FileNotFoundError:
            continue
        return snap_dir, kind
    named = " nor ".join(str(d) for _, d in candidates)
    raise FileNotFoundError(f"{run_id}: no usable snapshots under {named} — nothing to sweep")


def probe_margin_over_floor(best_acc: float, layer0_acc: float) -> float:
    """How far the best-layer probe accuracy climbs above the layer-0 floor —
    the pre-registered Test-1 verdict helper (decisions.md 2026-08-21 night):
    a large positive margin is consistent with a representation that is
    computed with depth rather than read off raw tokens at layer 0."""
    return best_acc - layer0_acc


def layer_summary(sub: pd.DataFrame) -> dict:
    """Layer-0 floor, best layer + accuracy, final layer + accuracy, and the
    margin-over-floor verdict for one (model, checkpoint, set) probe-accuracy
    curve across layers. ``sub`` must have one row per layer with ``layer``
    and ``probe_test_acc`` columns, including layer 0 (the floor is
    mandatory — there is no verdict without it)."""
    if sub.empty:
        raise ValueError("layer_summary: empty layer curve")
    sub = sub.sort_values("layer")
    layer0 = sub[sub["layer"] == 0]
    if layer0.empty:
        raise ValueError("layer_summary: no layer-0 row — the floor is mandatory")
    layer0_acc = float(layer0["probe_test_acc"].iloc[0])
    best = sub.loc[sub["probe_test_acc"].idxmax()]
    final = sub.iloc[-1]
    best_acc = float(best["probe_test_acc"])
    return {
        "layer0_floor": layer0_acc,
        "best_layer": int(best["layer"]),
        "best_acc": best_acc,
        "final_layer": int(final["layer"]),
        "final_acc": float(final["probe_test_acc"]),
        "probe_margin_over_floor": probe_margin_over_floor(best_acc, layer0_acc),
    }


def summary_table(rows: list[dict]) -> pd.DataFrame:
    """One row per (model, checkpoint_step, set): ``layer_summary`` over that
    group's per-layer curve."""
    df = pd.DataFrame(rows)
    out = []
    for (model, step, set_name), sub in df.groupby(["model", "checkpoint_step", "set"], sort=False):
        out.append({"model": model, "checkpoint_step": step, "set": set_name, **layer_summary(sub)})
    return pd.DataFrame(out)


def print_summary(rows: list[dict]) -> None:
    summary = summary_table(rows)
    for (model, set_name), sub in summary.groupby(["model", "set"], sort=False):
        print(f"[evt] {model} / {set_name}:")
        for r in sub.sort_values("checkpoint_step").itertuples():
            print(
                f"[evt]   step={r.checkpoint_step:>8}  layer0_floor={r.layer0_floor:.4f}  "
                f"best=layer{r.best_layer}:{r.best_acc:.4f}  "
                f"final=layer{r.final_layer}:{r.final_acc:.4f}  "
                f"margin_over_floor={r.probe_margin_over_floor:+.4f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="run:<run_id> or dir:<path> — one theta")
    ap.add_argument(
        "--model-name", default=None, help="'model' column label (default: --model/--run-id)"
    )
    ap.add_argument("--run-id", default=None, help="sweep this run's snapshots instead of --model")
    ap.add_argument(
        "--snapshot-steps",
        default=None,
        help="comma-separated steps to probe (default: every step_* dir found)",
    )
    ap.add_argument(
        "--prompt-parquet", action="append", dest="prompt_parquets", type=Path, required=True
    )
    ap.add_argument("--set-name", action="append", dest="set_names", required=True)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="train/test split permutation seed")
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on the readout weights")
    ap.add_argument(
        "--limit", type=int, default=None, help="use only the first N prompt rows per set"
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "resid_probe.csv")
    args = ap.parse_args()

    if bool(args.model) == bool(args.run_id):
        ap.error("exactly one of --model or --run-id is required")
    if args.snapshot_steps and not args.run_id:
        ap.error("--snapshot-steps requires --run-id")
    if len(args.prompt_parquets) != len(args.set_names):
        ap.error("--prompt-parquet and --set-name must be given the same number of times")

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    refs: dict[str, ProbeReference] = {}
    for path, set_name in zip(args.prompt_parquets, args.set_names):
        df = pd.read_parquet(path)
        examples = load_task_examples(df, tokenizer, limit=args.limit)
        refs[set_name] = build_reference(examples, set_name, args.seed)
        ref = refs[set_name]
        print(
            f"[evt] set {set_name!r}: n={len(examples)}, {ref.n_classes} classes, "
            f"majority test acc {ref.majority_test_acc:.4f}"
        )

    rows: list[dict] = []
    if args.model:
        model = load_any_model(args.model, device=args.device, store=args.store)
        model_label = args.model_name or args.model
        rows = probe_rows_for_model(model, refs, model_label, NO_CHECKPOINT_STEP, l2=args.l2)
    else:
        snap_dir, kind = discover_snapshot_dir(args.run_id, args.store)
        available = snapshot_steps_in_dir(snap_dir, SNAPSHOT_MARKER[kind])
        if args.snapshot_steps:
            steps = sorted(int(s) for s in args.snapshot_steps.split(","))
            missing = sorted(set(steps) - set(available))
            if missing:
                ap.error(
                    f"--snapshot-steps {missing} not found (or missing their "
                    f"{SNAPSHOT_MARKER[kind]!r} marker) under {snap_dir}; usable steps: {available}"
                )
        else:
            steps = available
        model_label = args.model_name or args.run_id
        print(f"[evt] {args.run_id}: {kind} snapshots, {len(steps)} steps: {steps}")
        lora_model = (
            load_model(args.run_id, store=args.store, device=args.device)
            if kind == "lora"
            else None
        )
        for step in steps:
            if kind == "lora":
                model = load_snapshot(lora_model, args.run_id, step, store=args.store)
            else:
                model = load_any_model(
                    f"dir:{snap_dir / f'step_{step}'}", device=args.device, store=args.store
                )
            rows.extend(probe_rows_for_model(model, refs, model_label, step, l2=args.l2))

    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)


if __name__ == "__main__":
    main()
