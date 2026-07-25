"""Steering-vector transfer driver (Wang et al. 2025 template, spec 02 §7).

Wang et al. 2025 found that an OOCR-style capability shift is approximately a
**constant residual-stream direction**: adding one vector reproduces much of
what training produced. This driver runs that test causally on the
elicit-vs-teach pair — the only analysis here that touches a model rather
than only dumps:

1. **Extract** (dumps only, no model): for each residual hook point L, the
   mean activation shift training produced,
   ``direction_L = mean_i [ pooled_i(final dump) − pooled_i(earliest dump) ]``
   where ``pooled_i`` is the fp64 mask-mean over example i's label positions
   (identical pooling semantics to ``geode.probe.representation_drift`` —
   padding activations are NOT zero, so pooling must go through the mask).
   The earliest dump (step 1) is one optimizer update in: the closest
   available stand-in for the pre-training-run state, not init itself.
2. **Inject** into the PARENT model — the run's ``base_model.hf_id``
   (``zoo-run/<parent_run_id>``), loaded through ``geode.zoo.load_model``
   (the manifest is the authority on how a checkpoint loads, V0.9; a raw
   ``from_pretrained`` on a wrapped checkpoint silently random-initialises
   projections). No training, no weight edit: one forward hook adds
   ``alpha * direction_L`` to hook L's residual output.
   ``--target-parent`` overrides the injection target with any other run in
   the store — the **cross-arm** cell (arm A's direction into arm B's parent
   and vice versa), which asks whether a direction is a transferable
   capability or a key that only opens a lock the target already has. A cross
   cell must be written under its own ``--results-name`` (guarded below), and
   its null is only interpretable next to the per-hook cosine between the two
   arms' directions: a direction is defined in the residual basis it was
   measured in, and two separately fine-tuned parents need not share one.
3. **Evaluate** target-task exact match + mean loss (nats) on the frozen
   ``D_target_eval`` reporting block — the same eval path gates.py g5 uses.

Causal question: does the target capability appear in the parent from the
direction alone? A large EM jump at some (hook, alpha) says training's effect
is largely a constant shift (elicitation-flavoured); no jump says it is not.

Design decisions:

- **All positions.** The shift is added at every sequence position (including
  padding and, under KV-cache generation, each newly generated position),
  not only label positions: at eval time the answer tokens do not exist yet,
  so a label-position-only injection could not steer the completion at all.
  Pad positions are attention-masked, so adding there is inert.
- **alpha = 0 is the no-injection baseline and runs through the hook.** It is
  a row like any other, computed with the hook installed, so the baseline
  shares every code path with the steered configs (a hook that corrupted the
  forward would move the baseline too, not hide behind it).
- **Random control.** ``direction_kind="random"`` is a seeded Gaussian
  rescaled to the learned ‖direction_L‖ per hook, so a "steering works"
  claim has to beat an equally large arbitrary push. Its seed is
  ``seed + layer``, so a ``--hooks`` subset draws exactly the vectors the
  full run would.
- **Leak check.** After every config the hook is removed and one cached
  example's logits must match the un-hooked baseline bit-exactly; a leaked
  hook would silently contaminate every later config.
- **checkpoint_step** in the results table is the direction's *source* step
  (the final dumped snapshot) — the rows measure a direction taken there,
  applied to a model that never took that step.
- Eval questions are the same fixed slice G5 scores (reporting block, after
  the 16 exemplar rows), so the alpha=0 row is directly comparable to the
  parent's recorded G5 zero-shot accuracy. Prompts are token-level prefixes
  of the training tokenization and the loss tokenization carries the EOS
  inside the label span (2026-07-21 incident; V5.43) — both via
  ``geode.arith.tokenize_with_spans``, never re-tokenized char slices.

Usage:
    python3 steering.py [--run-id evt-run7-armA-target-1m] [--store <dir>]
        [--alphas 0,0.5,1.0,2.0] [--hooks hook_embed,blocks.7.hook_resid_post]
        [--n-eval 256] [--seed 0] [--device cuda] [--probe-local <path>]
        [--eval-config <yaml>] [--eval-local <parquet>]
        [--target-parent <run_id>] [--results-name steering_transfer]
        [--fig figures/steering_transfer.png]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geode.arith import exact_match, greedy_completions, order_hash, tokenize_with_spans
from geode.edl.masking import TaskFormat
from geode.probe import load_probe_dump, residual_hook_names

# The single hook-name -> module map (embedding + decoder blocks) that
# extraction hooks with; injection must hook exactly the same points, so it
# reuses the helper rather than re-deriving the module tree.
from geode.probe.extract import _residual_modules
from geode.train import evaluate_sft_nll_nats
from geode.zoo import load_model, load_run, write_results
from geode.zoo.store import run_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "experiments" / "training-run" / "scripts"
CONFIGS_DIR = REPO_ROOT / "experiments" / "training-run" / "configs"
sys.path.insert(0, str(SCRIPTS_DIR))

# The eval-slice protocol constants and the frozen-file loader live in
# scripts/, next to the gates that define them — imported, never re-derived.
from gates import G5_N_SHOTS  # noqa: E402
from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402
from train_target import EVAL_STOP_ROWS  # noqa: E402

DEFAULT_RUN = "evt-run7-armA-target-1m"
DEFAULT_EVAL_CONFIG = CONFIGS_DIR / "eval_target_data.yaml"
METRIC_EM = "steer_em"
METRIC_LOSS = "steer_loss_nats"
DEFAULT_RESULTS_NAME = "steering_transfer"
DIRECTION_KINDS = ("learned", "random")
PROBE_REPO = "mhieuuu/elicit-vs-teach-arith"
PROBE_FILE = "probe.parquet"
ZOO_PREFIX = "zoo-run/"


def _load_probe(probe_local: Path | None) -> pd.DataFrame:
    """The frozen probe parquet — a local path, or downloaded from HF (spec §7)."""
    if probe_local is not None:
        path = probe_local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(PROBE_REPO, PROBE_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def _guard_probe_hash(run_id: str, step: int, meta_hash: str, probe_hash: str) -> None:
    """Row-alignment guard (spec 00 §6): parquet order == every dump's probe set."""
    if meta_hash != probe_hash:
        raise ValueError(
            f"{run_id}@step_{step}: dump probe_set_hash {meta_hash} != probe parquet "
            f"order_hash {probe_hash} — parquet row i and dump row i must be the same "
            "frozen probe example (spec 00 §6); refusing to align mismatched sets"
        )


def pooled_label_mean(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-example mean activation over ``mask``-True positions: ``[n, seq, d] -> [n, d]``.

    fp64 regardless of the stored dtype (dumps are bf16); padding activations
    are not zero, so pooling goes through the mask, never a flatten.
    """
    if acts.ndim != 3:
        raise ValueError(f"pooled_label_mean: expected [n, seq, d] acts, got {tuple(acts.shape)}")
    if tuple(mask.shape) != tuple(acts.shape[:2]):
        raise ValueError(
            f"pooled_label_mean: mask shape {tuple(mask.shape)} != acts [n, seq] "
            f"{tuple(acts.shape[:2])}"
        )
    m = mask.to(torch.bool)
    counts = m.sum(dim=1)
    if bool((counts == 0).any()):
        zero_rows = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"pooled_label_mean: examples {zero_rows} have zero mask-True positions")
    weighted = (acts.to(torch.float64) * m.to(torch.float64)[:, :, None]).sum(dim=1)
    return weighted / counts.to(torch.float64)[:, None]


def steering_directions(
    run_id: str, store: Path, probe_hash: str
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    """Mean activation shift per hook between the earliest and final dump.

    Returns ``(directions, first_step, last_step, n_examples)``; each
    direction is an fp64 ``[d_model]`` vector.
    """
    probe_root = run_dir(run_id, store=store) / "probe"
    steps = sorted(
        int(p.name.removeprefix("step_"))
        for p in probe_root.glob("step_*")
        if (p / "meta.json").is_file()
    )
    if len(steps) < 2:
        raise FileNotFoundError(
            f"{run_id}: need >= 2 probe dumps under {probe_root} to take a shift, "
            f"found {len(steps)} — run scripts/extract.py first"
        )
    first, last = steps[0], steps[-1]
    cap_first, meta_first = load_probe_dump(run_id, first, store=store)
    cap_last, meta_last = load_probe_dump(run_id, last, store=store)
    _guard_probe_hash(run_id, first, meta_first.probe_set_hash, probe_hash)
    _guard_probe_hash(run_id, last, meta_last.probe_set_hash, probe_hash)
    if not torch.equal(cap_first.input_ids, cap_last.input_ids):
        raise ValueError(
            f"{run_id}: step {last} input_ids differ from step {first} — the probe "
            "tokenization must be identical across snapshots to pool one mask"
        )
    names = residual_hook_names(len(cap_first.acts) - 1)
    mask = cap_first.label_mask
    directions = {
        name: (
            pooled_label_mean(cap_last.acts[name], mask)
            - pooled_label_mean(cap_first.acts[name], mask)
        ).mean(dim=0)
        for name in names
    }
    n_examples = int(mask.shape[0])
    print(
        f"[evt] {run_id}: direction = dump step {last} − step {first}, "
        f"{n_examples} probe examples, {len(names)} hook points"
    )
    return directions, first, last, n_examples


def add_injection_hook(
    model: torch.nn.Module, hook_name: str, direction: torch.Tensor, alpha: float
):
    """Register a forward hook adding ``alpha * direction`` at hook ``hook_name``.

    The shift lands at **all** sequence positions of that residual point (see
    the module docstring), cast to the residual's dtype/device at call time so
    the driver stays device-agnostic. Returns the handle; the caller removes it.
    """
    embed, blocks = _residual_modules(model)
    names = residual_hook_names(len(blocks))
    if hook_name not in names:
        raise ValueError(f"unknown hook {hook_name!r}: model exposes {names}")
    index = names.index(hook_name)
    module = embed if index == 0 else blocks[index - 1]
    shift = direction.detach() * float(alpha)

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        shifted = tensor + shift.to(device=tensor.device, dtype=tensor.dtype)
        if isinstance(output, tuple):
            return (shifted, *output[1:])
        return shifted

    return module.register_forward_hook(hook)


def random_directions(
    directions: dict[str, torch.Tensor], names: list[str], seed: int
) -> dict[str, torch.Tensor]:
    """Norm-matched Gaussian controls: seed ``seed + layer``, fp64, CPU."""
    controls: dict[str, torch.Tensor] = {}
    for layer, name in enumerate(names):
        learned = directions[name]
        generator = torch.Generator().manual_seed(seed + layer)
        vec = torch.randn(learned.shape, generator=generator, dtype=torch.float64)
        norm = float(vec.norm().item())
        controls[name] = vec * (float(learned.norm().item()) / norm) if norm > 0 else vec
    return controls


def load_eval_examples(
    cfg: dict, tokenizer, n_eval: int, eval_local: Path | None
) -> tuple[list[list[int]], list[int], list]:
    """The G5 question slice: token-prefix prompts, answers, loss examples.

    Same rows G5 scores (reporting block after the 16 exemplars — fixed
    slices, no sampling) and the same two tokenizations: prompts are token
    prefixes of the training tokenization; the loss examples carry the EOS
    inside the label span (V5.43).
    """
    if eval_local is not None:
        # Offline/smoke path: the pinned order_hash cannot describe a local
        # stand-in file, so the frozen-file verification is skipped with it.
        print(f"[evt] eval data: local {eval_local} (pinned order_hash NOT verified)")
        df = pd.read_parquet(eval_local)
    else:
        df = load_frozen_parquet(cfg)
    q_start = EVAL_STOP_ROWS + G5_N_SHOTS
    if len(df) < q_start + n_eval:
        raise ValueError(
            f"--n-eval {n_eval}: eval file has {len(df)} rows, needs {q_start + n_eval}"
        )
    rows = df.iloc[q_start : q_start + n_eval]
    texts = rows["full_text"].tolist()
    char_spans = list(
        zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int))
    )
    answers = rows["true_answer"].astype(int).tolist()
    prompt_examples = tokenize_with_spans(texts, char_spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in prompt_examples]
    loss_examples = tokenize_with_spans(texts, char_spans, tokenizer, append_eos=True)
    return prompt_ids, answers, loss_examples


def evaluate(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: list[list[int]],
    answers: list[int],
    loss_examples: list,
    task_format: TaskFormat,
    *,
    device: str,
    batch_size: int,
) -> tuple[float, float]:
    """Target-task exact match + mean label loss (nats) of ``model`` as it stands."""
    completions = greedy_completions(
        model, tokenizer, prompt_ids, device=device, batch_size=batch_size
    )
    # The completion starts inside the answer slot, so "Answer:" + completion
    # re-enters the tested parser with the marker it expects (gates.py).
    hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
    loss = evaluate_sft_nll_nats(
        model, loss_examples, task_format, batch_size=batch_size, device=device
    )
    return sum(hits) / len(hits), loss


@torch.no_grad()
def _logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids).logits


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    cmap = plt.get_cmap("viridis")
    layers = sorted(int(x) for x in df["layer"].unique())
    denom = max(layers) if max(layers) > 0 else 1
    linestyles = {"learned": "-", "random": ":"}
    for ax, metric, ylabel in (
        (axes[0], METRIC_EM, "target-task exact match"),
        (axes[1], METRIC_LOSS, "target-task loss (nats)"),
    ):
        sub = df[df["metric_name"] == metric]
        baseline = sub.loc[sub["alpha"] == 0.0, "metric_value"]
        if not baseline.empty:
            ax.axhline(
                float(baseline.mean()),
                color="black",
                lw=0.9,
                ls="--",
                alpha=0.6,
                label="alpha=0 baseline (no injection)",
            )
        for (layer, kind), group in sub.groupby(["layer", "direction_kind"]):
            group = group.sort_values("alpha")
            ax.plot(
                group["alpha"],
                group["metric_value"],
                color=cmap(int(layer) / denom),
                ls=linestyles[str(kind)],
                marker="o",
                ms=3,
                lw=1.4,
                label=f"L{int(layer)} {group['hook_name'].iloc[0]}" if kind == "learned" else None,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2, title="solid: learned / dotted: norm-matched random")
    axes[1].set_xlabel("alpha (steering-vector scale)")
    axes[0].set_title("steering-vector transfer into the parent model (no training)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=DEFAULT_RUN, help="direction source run (its probe dumps)")
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--probe-local", type=Path, default=None, help="skip the HF download of probe.parquet"
    )
    ap.add_argument("--alphas", default="0,0.5,1.0,2.0", help="comma-separated steering scales")
    ap.add_argument("--hooks", default=None, help="comma-separated subset (default: all hooks)")
    ap.add_argument("--n-eval", type=int, default=256, help="eval questions (G5 reporting slice)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random-direction control")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    ap.add_argument(
        "--eval-local", type=Path, default=None, help="local eval parquet (skips the HF download)"
    )
    ap.add_argument(
        "--target-parent",
        default=None,
        help="inject into this run's checkpoint instead of --run-id's own parent (cross-arm cell)",
    )
    ap.add_argument(
        "--results-name",
        default=DEFAULT_RESULTS_NAME,
        help=f"results table name (default {DEFAULT_RESULTS_NAME}; a cross cell needs its own)",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "steering_transfer.png",
    )
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        raise SystemExit("[evt] --alphas is empty")

    df_probe = _load_probe(args.probe_local)
    probe_hash = order_hash(df_probe.to_dict("records"))
    print(f"[evt] probe set: {len(df_probe)} examples, probe_set_hash={probe_hash[:12]}…")

    directions, first_step, last_step, n_probe = steering_directions(
        args.run_id, args.store, probe_hash
    )
    all_names = list(directions)
    names = [h.strip() for h in args.hooks.split(",")] if args.hooks else all_names
    unknown = [n for n in names if n not in directions]
    if unknown:
        raise SystemExit(f"[evt] unknown hook(s) {unknown}; model exposes {all_names}")
    controls = random_directions(directions, all_names, args.seed)

    manifest = load_run(args.run_id, store=args.store)
    base_model_key = manifest.data["base_model"]["hf_id"]
    if not base_model_key.startswith(ZOO_PREFIX):
        raise SystemExit(
            f"[evt] {args.run_id}: base_model.hf_id {base_model_key!r} is not a "
            f"'{ZOO_PREFIX}<run_id>' parent — steering transfers into a parent run "
            "checkpoint held in this store, not an external hub model"
        )
    source_parent_id = base_model_key[len(ZOO_PREFIX) :]
    parent_id = args.target_parent or source_parent_id
    # A cross cell writes a different question's answer; sharing the default
    # table name would silently overwrite the own-parent diagonals
    # (write_results is overwrite-by-name, OQ-6).
    if parent_id != source_parent_id and args.results_name == DEFAULT_RESULTS_NAME:
        raise SystemExit(
            f"[evt] --target-parent {parent_id} != {args.run_id}'s own parent "
            f"{source_parent_id}: a cross-arm cell must be written under its own "
            f"--results-name, not the default {DEFAULT_RESULTS_NAME!r} which holds "
            "the own-parent diagonals"
        )

    cfg = load_config(args.eval_config, None)
    from transformers import AutoTokenizer

    local = (args.eval_config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    task_format = TaskFormat(name=cfg["task"]["name"], format_version=cfg["task"]["format_version"])
    prompt_ids, answers, loss_examples = load_eval_examples(
        cfg, tokenizer, args.n_eval, args.eval_local
    )

    # V0.9: a run checkpoint loads only the way its manifest says it was
    # trained — never a bare from_pretrained.
    print(f"[evt] loading parent {parent_id} (steering target, untrained on D_target) ...")
    model = load_model(parent_id, store=args.store, device=args.device)
    # A direction is only meaningful in the residual basis it was measured in;
    # a shape mismatch means the two models are not even the same architecture.
    d_model = int(model.config.hidden_size)
    d_direction = int(next(iter(directions.values())).shape[0])
    if d_direction != d_model:
        raise SystemExit(
            f"[evt] {args.run_id} direction is {d_direction}-dim but target {parent_id} "
            f"has hidden size {d_model} — a direction cannot be injected into a "
            "different architecture"
        )

    n_configs = len(names) * len(alphas) * len(DIRECTION_KINDS)
    print(
        f"[evt] {n_configs} configs x {args.n_eval} examples "
        f"({len(names)} hooks x {len(alphas)} alphas x {len(DIRECTION_KINDS)} direction kinds)"
    )

    # Cached leak-check example: un-hooked logits of the first eval prompt.
    leak_ids = torch.tensor([prompt_ids[0]], dtype=torch.long, device=args.device)
    leak_baseline = _logits(model, leak_ids)

    common = {
        "run_id": args.run_id,
        # The model actually injected, not the direction source's own parent —
        # otherwise a groupby on base_model_key silently mixes cross cells in
        # with the diagonals.
        "base_model_key": ZOO_PREFIX + parent_id,
        "source_parent_id": source_parent_id,
        "regime": manifest.data["regime"],
        "dataset_size": manifest.data["dataset"]["n_unique_examples"],
        "checkpoint_step": last_step,  # the direction's source snapshot
        "parent_id": parent_id,
        "n_eval": args.n_eval,
        "n_probe_examples": n_probe,
        "ref_step": first_step,
    }
    rows: list[dict] = []
    for name in names:
        layer = all_names.index(name)
        for kind in DIRECTION_KINDS:
            direction = directions[name] if kind == "learned" else controls[name]
            norm = float(direction.norm().item())
            for alpha in alphas:
                handle = add_injection_hook(model, name, direction, alpha)
                try:
                    em, loss = evaluate(
                        model,
                        tokenizer,
                        prompt_ids,
                        answers,
                        loss_examples,
                        task_format,
                        device=args.device,
                        batch_size=args.batch_size,
                    )
                finally:
                    handle.remove()
                if not torch.equal(_logits(model, leak_ids), leak_baseline):
                    raise RuntimeError(
                        f"{name} alpha={alpha} ({kind}): removing the injection hook did "
                        "not restore the baseline logits — a leaked hook would "
                        "contaminate every later config"
                    )
                print(f"[evt] {name} alpha={alpha:g} {kind}: EM {em:.4f}, loss {loss:.4f} nats")
                for metric_name, value in ((METRIC_EM, em), (METRIC_LOSS, loss)):
                    rows.append(
                        {
                            **common,
                            "layer": layer,
                            "hook_name": name,
                            "metric_name": metric_name,
                            "metric_value": value,
                            "alpha": alpha,
                            "direction_kind": kind,
                            "direction_norm": norm,
                        }
                    )

    df = pd.DataFrame(rows)
    path = write_results(df, args.results_name, store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows)")
    plot(df, args.fig)

    em_rows = df[df["metric_name"] == METRIC_EM]
    baseline = em_rows[em_rows["alpha"] == 0.0]["metric_value"]
    if not baseline.empty:
        print(f"[evt] baseline (alpha=0, no injection): EM {float(baseline.mean()):.4f}")
    for kind in DIRECTION_KINDS:
        steered = em_rows[(em_rows["direction_kind"] == kind) & (em_rows["alpha"] != 0.0)]
        if steered.empty:
            continue
        best = steered.loc[steered["metric_value"].idxmax()]
        print(
            f"[evt] best {kind}: {best['hook_name']} alpha={best['alpha']:g} -> "
            f"EM {best['metric_value']:.4f}"
        )


if __name__ == "__main__":
    main()
