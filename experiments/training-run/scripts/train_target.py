"""Launch script for the LoRA target runs 5-6 (spec 02 §1 rows 5-6, §6 "Runs 5-6").

Protocol-exempt glue, mirroring ``train_sft.py``: parses run YAML (+ pilot
overlay), downloads the frozen D_target parquet AND the frozen shared eval
file (``data.eval_file``, owner 2026-07-22) and refuses on ``order_hash``
mismatch for either (V5.40), takes the ``data.n_examples`` PREFIX of the
frozen order (deterministic, comparable across OPEN(2) pilot arms; the full
1M is trainable — the eval questions live in their own disjoint file, not a
reserved tail), converts char to token spans with a label-covered EOS
(V5.38/V5.43), enforces the parent-gate DAG rule and the G7 cross-run
data-order match, computes the snapshot schedule
(``geode.probe.snapshot_steps``) and writes it into the manifest BEFORE
training, prints a cost estimate, and refuses to train without --confirm-cost
(CLAUDE.md budget rule). Training runs through the prequential EDL harness
(``geode.edl.train_prequential`` — the measurement itself), which wraps the
parent checkpoint with the pinned LoRA adapter (``geode.train.apply_lora``,
spec 02 §6) and writes prequential/gradstats/snapshots/test-loss per spec 00.
The spec 02 §6 logging extension (per-step LR + train-accuracy) and the loss
ε/k stopping rule ride on the harness's ``step_callback``; ``max_steps`` is a
pure cost ceiling. This file must stay thin.

Usage (run 5):
    python3 train_target.py --config configs/run5_target.yaml \
        [--override configs/pilot/target_pilot_100k_armA_lr1e-3.yaml] \
        --init-from $GEODE_STORE/runs/evt-run3-armA-inst/model \
        [--device cuda] [--store <dir>] [--confirm-cost]
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from geode.arith import load_frozen_parquet as _load_frozen_parquet, tokenize_with_spans
from geode.edl import EVAL_STOP_ROWS
from geode.edl.loop import PrequentialStepInfo, train_prequential
from geode.edl.masking import TaskFormat, masking_config_hash
from geode.probe import snapshot_steps
from geode.train import ConvergenceTracker, StoppingRule, evaluate_sft_nll_nats
from geode.zoo import checkpoint_dir, load_run, register_run, require_parent_ready, tokenizer_hash
from train import REPO_ROOT, git_commit, load_config, phase

# Regimes attach to the target runs. Run 10 (arm "llama") records "unknown"
# (spec 00: regime is design-known at creation; whether real Llama holds the
# capability latent is exactly what that run measures — run-9 SFT precedent).
# Explicit entry, not .get(default): a typo'd arm must still fail loudly.
ARM_REGIME = {"A": "elicit", "B": "teach", "llama": "unknown", "warmstart": "unknown"}

# EVAL_STOP_ROWS (spec 02 §5) is imported from geode.edl (V5.74) and re-exported
# here so `from train_target import EVAL_STOP_ROWS` still resolves
# (dump_g5_predictions.py, unlock_embedding.py). Rows [0, EVAL_STOP_ROWS) of the
# frozen eval file are the in-loop ε/k stopping block; everything after is the
# reporting block (θ_T test loss + G5), so no reported number ever touches the
# rows that drove the stop decision.

# Curve evals (owner 2026-07-22): extra in-loop evals at a dense-then-log-
# spaced set of steps (same tested scheduler as snapshots) so the val learning
# curve has resolution on a log step axis. Logged-only — rows carry
# stopping_eval: false and are NEVER fed to the ε/k tracker; the ratified
# stopping rule consumes exactly the eval_every cadence (+ the ceiling eval).
# Density 64→256 / dense 16→30 (owner 2026-07-24, "record val a lot for
# plotting"): ~161 curve evals land <= step 2000 over the 46,878 ceiling —
# the expected early-stop region. Worst case at 1.24B: one eval ≈ 16 fp32
# forward batches ≈ 5 update steps of wall-clock, so even an early-stopping
# run 10 pays tens of minutes, dollars-cents at box rates. Train loss needs
# no knob: train_log.jsonl + logs/prequential.jsonl record EVERY step.
EVAL_CURVE_N = 256
EVAL_CURVE_DENSE_UNTIL = 30


def load_frozen_parquet(d: dict):
    """Data-block adapter over ``geode.arith.load_frozen_parquet`` (V5.70).

    ``d`` needs ``hf_id``/``file``/``order_hash`` (+ optional ``local_path``);
    the hash is recomputed over the FULL file, so the recorded hash always names
    the frozen artifact, not the ``data.n_examples`` prefix the caller slices
    afterwards. A relative ``local_path`` resolves against ``REPO_ROOT``
    (2026-07-27 reconciliation with train_sft.py), an absolute one as-is.
    """
    return _load_frozen_parquet(d, root=REPO_ROOT)


def lora_trainable_param_count(model: torch.nn.Module, lora_cfg: dict) -> int:
    """Trainable adapter parameters the harness will create: Σ r·(d_in+d_out)
    over the target linears (the V5.49 formula), computed before the wrap so
    the manifest can be registered ahead of training."""
    targets = set(lora_cfg["target_modules"])
    rank = int(lora_cfg["r"])
    return sum(
        rank * (m.in_features + m.out_features)
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets
    )


def manifest_fields(
    cfg: dict,
    *,
    n_train: int,
    n_trainable: int,
    epochs_total: int,
    schedule: list[int],
    est_usd: float,
    init_from: Path,
    mask_hash: str,
    precision: str,
) -> dict[str, Any]:
    t = cfg["train"]
    lora = cfg["lora"]
    exp = cfg["experiment"]
    # A null parent_run_id + external_base names a non-zoo base (e.g. a
    # checkpoint carrying a recorded failing gate that require_parent_ready
    # refuses as a zoo parent); it is recorded verbatim. Mirrors train_sft.py.
    base_hf_id = (
        exp["external_base"]
        if not exp.get("parent_run_id") and exp.get("external_base")
        else f"zoo-run/{exp['parent_run_id']}"
    )
    return {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": git_commit(),
        "regime": ARM_REGIME[cfg["experiment"]["arm"]],
        "base_model": {
            "hf_id": base_hf_id,
            "revision": "none",
        },
        "task": {"name": cfg["task"]["name"], "format_version": cfg["task"]["format_version"]},
        "dataset": {
            "name": f"{cfg['data']['hf_id']}:{cfg['data']['file']}",
            # OQ-4: the example_ids universe is the TRAIN split (0..n_train-1).
            "n_unique_examples": n_train,
            "seed": cfg["data"]["seed"],
        },
        "training": {
            "method": "lora",
            "lora": {
                "rank": lora["r"],
                "alpha": lora["alpha"],
                "target_modules": list(lora["target_modules"]),
                "dropout": lora["dropout"],
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": cfg["optimizer"]["name"],
                "lr": t["lr"],
                "batch_size": t["batch_size"],
                "micro_batch_size": t["batch_size"],  # prequential loop: no accumulation
                "betas": list(cfg["optimizer"]["betas"]),
                "weight_decay": cfg["optimizer"]["weight_decay"],
                "grad_clip": cfg["training"]["grad_clip"],
            },
            "lr_schedule": "constant",  # structural: the prequential loop has no scheduler
            "min_lr": None,
            # Resolved value after any CPU fallback (spec 00 §2). The harness
            # reads THIS field: bf16 autocasts only its update forward — every
            # recorded loss stays fp32 (V5.62; Llama chain, owner 2026-07-24).
            # Small-model runs ship no train.precision and stay fp32.
            "precision": precision,
            "eval_every": t["eval_every"],
            "max_steps": t["max_steps"],
            "stopping": {  # loss branch of the V0.7 union (no `metric` key)
                "eps_nats": t["stopping"]["eps_nats"],
                "k": t["stopping"]["k"],
                "min_steps": t["stopping"].get("min_steps", 0),
            },
            # Resolved epoch budget covering the max_steps ceiling; the
            # callback stops at max_steps exactly, so later epochs are
            # reachable-but-unused headroom, never a planned duration.
            "epochs_total": epochs_total,
            "seed": t["seed"],
        },
        "trainable_param_count": n_trainable,
        "snapshot_steps": schedule,
        "cost": {"gpu_type": cfg["gpu"]["type"], "est_usd": est_usd, "actual_usd": None},
        "status": "running",
        # Extras ride as preserved unknowns (spec 00 V0.2).
        "experiment": cfg["experiment"]
        | {
            "gates": {},
            "init_from": str(init_from),
            "masking_config_hash": mask_hash,
            "data_order_hash": cfg["data"]["order_hash"],  # FULL frozen file
            "n_examples": cfg["data"].get("n_examples"),  # prefix taken
            # Shared fixed eval file (spec 02 §5): stopping block first,
            # reporting block (θ_T test loss + G5) after.
            "eval_data": {
                "file": cfg["data"]["eval_file"],
                "order_hash": cfg["data"]["eval_order_hash"],
                "stop_rows": EVAL_STOP_ROWS,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, default=None, help="e.g. a pilot overlay")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--init-from",
        type=Path,
        required=True,
        help="save_pretrained checkpoint dir of the parent installer run "
        "(targets never start from random)",
    )
    parser.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE")
    parser.add_argument(
        "--parent-run-id",
        default=None,
        help="override experiment.parent_run_id for a pre-selected warm-start candidate",
    )
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    phase(1, "config + tokenizer")
    cfg = load_config(args.config, args.override)
    if args.parent_run_id is not None:
        if cfg["experiment"]["arm"] != "warmstart":
            raise ValueError("--parent-run-id is only valid for arm: warmstart")
        cfg["experiment"]["parent_run_id"] = args.parent_run_id
    if cfg["train"].get("lr") is None:
        print(
            "[evt] train.lr is null — the LoRA target LR is unpinned until the pilot "
            "mini-sweep pins it (run-2/run-3 precedent); a placeholder-lr launch is a "
            "silent redo (2026-07-21 incident class). Exiting."
        )
        return 1
    if cfg["train"].get("max_steps") is None:
        print("[evt] train.max_steps is null — the cost ceiling must be pinned. Exiting.")
        return 1

    from transformers import AutoTokenizer, LlamaForCausalLM

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    phase(2, "dataset — frozen train + eval parquets, order_hash verified, prefix, tokenize")
    d = cfg["data"]
    df = load_frozen_parquet(d)
    n_examples = d.get("n_examples")
    if n_examples is None:
        raise ValueError(
            "data.n_examples is null — pin the prefix explicitly (OPEN(2)); "
            f"the full frozen file is n_examples: {len(df)}"
        )
    if not (0 < n_examples <= len(df)):
        raise ValueError(f"data.n_examples={n_examples} outside 1..{len(df)}")
    df = df.head(n_examples)  # PREFIX of the frozen order (OPEN(2) comparability)
    print(
        f"[evt] {d['file']}: order_hash verified on full file; using "
        f"{len(df)} rows (prefix {n_examples})",
        flush=True,
    )

    def tokenize_df(frame):
        texts = frame["full_text"].tolist()
        spans = list(
            zip(frame["answer_char_start"].astype(int), frame["answer_char_end"].astype(int))
        )
        return tokenize_with_spans(texts, spans, tokenizer, append_eos=True)

    train_examples = tokenize_df(df)  # the FULL prefix trains — no val carve
    max_len = max(len(ex.input_ids) for ex in train_examples)
    print(f"[evt] tokenized: max {max_len} tokens/example (expected ≈34 incl. EOS)", flush=True)

    # Shared fixed eval file (owner 2026-07-22): disjoint from all training
    # data by construction, identical for every run — stopping block first,
    # reporting block (final θ_T test loss, G5) after.
    eval_df = load_frozen_parquet(
        {
            "hf_id": d["hf_id"],
            "file": d["eval_file"],
            "order_hash": d["eval_order_hash"],
            "local_path": d.get("eval_local_path"),
        }
    )
    eval_examples = tokenize_df(eval_df)
    stop_examples = eval_examples[:EVAL_STOP_ROWS]
    test_examples = eval_examples[EVAL_STOP_ROWS:]
    print(
        f"[evt] {d['eval_file']}: order_hash verified; stopping block "
        f"{len(stop_examples)}, reporting block {len(test_examples)}",
        flush=True,
    )

    phase(3, "model — parent checkpoint (the harness applies the pinned LoRA)")
    print(f"[evt] loading init checkpoint {args.init_from} ...", flush=True)
    model = LlamaForCausalLM.from_pretrained(args.init_from)
    m = cfg["model"]
    got = (model.config.vocab_size, model.config.hidden_size, model.config.num_hidden_layers)
    want = (len(tokenizer), m["hidden_size"], m["num_hidden_layers"])
    if got != want:
        raise ValueError(
            f"--init-from arch mismatch: checkpoint has (vocab, hidden, layers)={got}, "
            f"config + tokenizer want {want}"
        )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = lora_trainable_param_count(model, cfg["lora"])

    phase(4, "parent gates + G7 data-order match + cost estimate + confirm gate")
    if args.store is not None:
        os.environ["GEODE_STORE"] = str(args.store)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    parent = cfg["experiment"]["parent_run_id"]
    external_base = cfg["experiment"].get("external_base")
    if not parent:
        if not external_base:
            print("[evt] experiment.parent_run_id is null — pin it to the installer run. Exiting.")
            return 1
        # Deliberate bypass (mirrors train_sft.py): the true base carries a
        # recorded failing gate that require_parent_ready correctly refuses, so
        # the config declares no zoo parent and names the base via external_base.
        # --init-from still carries the real weights; experiment.init_from
        # records the exact path. Owner-directed only ("ignore the gate").
        print(
            f"[evt] external_base '{external_base}' — no zoo parent, skipping parent-gate check",
            flush=True,
        )
    else:
        require_parent_ready(
            parent,
            required_gates=tuple(cfg["experiment"].get("parent_required_gates", ())),
            store=store,
        )
        if cfg["experiment"]["arm"] == "warmstart":
            expected_parent_checkpoint = checkpoint_dir(parent, store=store)
            if args.init_from.resolve() != expected_parent_checkpoint.resolve():
                raise ValueError(
                    f"--init-from {args.init_from} is not declared warm-start parent "
                    f"'{parent}' checkpoint {expected_parent_checkpoint}"
                )
            print(f"[evt] parent '{parent}' complete, gates pass, checkpoint matches", flush=True)
        else:
            print(f"[evt] parent '{parent}' complete, gates pass", flush=True)

    match_with = cfg["experiment"].get("match_data_order_with")
    if match_with:
        # G7 (spec 02 §8): both target runs must consume the identical frozen
        # data (same file order, same prefix length) — enforced at launch.
        other = load_run(match_with, store=store).data.get("experiment", {})
        ours = (cfg["data"]["order_hash"], n_examples)
        theirs = (other.get("data_order_hash"), other.get("n_examples"))
        if ours != theirs:
            raise ValueError(
                f"G7 refusal: (data_order_hash, n_examples) mismatch vs '{match_with}': "
                f"this run {ours}, matched run {theirs} — arms must train on identical "
                "data in identical order (spec 02 §8 G7)"
            )
        print(f"[evt] G7: data order matches '{match_with}'", flush=True)

    t = cfg["train"]
    steps_per_epoch = math.ceil(len(train_examples) / t["batch_size"])  # no drop-last
    fixed_prefix_one_pass = bool(cfg["experiment"].get("fixed_prefix_one_pass", False))
    if fixed_prefix_one_pass:
        if cfg["experiment"]["arm"] != "warmstart":
            raise ValueError("fixed_prefix_one_pass is only valid for arm: warmstart")
        if t["max_steps"] != steps_per_epoch:
            raise ValueError(
                "fixed-prefix warm-start target must set max_steps to exactly one "
                f"no-drop-last pass ({steps_per_epoch}), got {t['max_steps']}"
            )
        if t["stopping"].get("min_steps", 0) != steps_per_epoch:
            raise ValueError(
                "fixed-prefix warm-start target must prevent convergence before its "
                f"one-pass ceiling at step {steps_per_epoch}"
            )
    gpu = cfg["gpu"]
    # Same CPU fallback as train_sft.py; spec 00 §2 records the resolved value.
    precision = t.get("precision", "fp32") if args.device != "cpu" else "fp32"
    # The ceiling is what the budget must cover: max_steps batches of
    # right-padded rows (set-max padding, geode.edl.loop convention).
    flops = 6.0 * n_params * (t["max_steps"] * t["batch_size"] * max_len)
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    est_usd = hours * gpu["usd_per_hour"]
    epochs_total = math.ceil(t["max_steps"] / steps_per_epoch)
    print(
        f"[evt] run_id={cfg['run_id']} train={len(train_examples)} "
        f"stop-eval={len(stop_examples)} test={len(test_examples)} "
        f"steps/epoch={steps_per_epoch} max_steps={t['max_steps']} (ceiling) "
        f"trainable={n_trainable / 1e6:.1f}M of {n_params / 1e6:.1f}M"
    )
    print(f"[evt] estimated cost: ${est_usd:,.2f}  ({hours:.2f} GPU-h @ ${gpu['usd_per_hour']}/h)")
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    phase(5, "register (snapshot schedule in manifest BEFORE training) + train")
    snaps = t["snapshots"]
    # n: 0 = save no snapshots at all (LR-sweep pilots, owner 2026-07-24);
    # the scheduler keeps its n >= 1 contract, so bypass it here.
    schedule = (
        []
        if snaps["n"] == 0
        else snapshot_steps(
            total_steps=t["max_steps"], n=snaps["n"], dense_until=snaps.get("dense_until", 30)
        )
    )
    task_format = TaskFormat(name=cfg["task"]["name"], format_version=cfg["task"]["format_version"])
    tok_hash = tokenizer_hash(tokenizer)
    mask_hash = masking_config_hash(task_format, tok_hash)
    manifest = register_run(
        manifest_fields(
            cfg,
            n_train=len(train_examples),
            n_trainable=n_trainable,
            epochs_total=epochs_total,
            schedule=schedule,
            est_usd=est_usd,
            init_from=args.init_from,
            mask_hash=mask_hash,
            precision=precision,
        ),
        store=store,
    )
    out_dir = store / "runs" / cfg["run_id"]
    print(f"[evt] store={store}  snapshots scheduled: {len(schedule)}", flush=True)

    # Stopping (spec 02 §6 / OPEN(3) inheritance): loss-convergence ε/k on the
    # fixed stopping block (first EVAL_STOP_ROWS rows of the frozen eval
    # file — identical data for every run), evaluated in-loop every
    # eval_every steps. Extra curve evals (EVAL_CURVE_*) run at log-spaced
    # steps for learning-curve resolution but never feed the tracker;
    # max_steps is a pure cost ceiling. The spec-02 §6
    # logging extension (per-step LR + train-accuracy) lands in
    # train_log.jsonl at the run root; stopping evals in eval_log.jsonl
    # (field name val_loss_nats kept for continuity). Both ride the
    # harness's step_callback so the EDL loop itself stays untouched.
    s = t["stopping"]
    tracker = ConvergenceTracker(
        StoppingRule(eps_nats=s["eps_nats"], k=s["k"], min_steps=s.get("min_steps", 0))
    )
    state: dict[str, Any] = {"final_step": 0, "stop_reason": None}
    eval_every, max_steps, batch_size = t["eval_every"], t["max_steps"], t["batch_size"]
    curve_steps = set(snapshot_steps(max_steps, n=EVAL_CURVE_N, dense_until=EVAL_CURVE_DENSE_UNTIL))
    out_dir.mkdir(parents=True, exist_ok=True)

    with (
        (out_dir / "train_log.jsonl").open("w") as train_f,
        (out_dir / "eval_log.jsonl").open("w") as eval_f,
    ):

        def step_callback(info: PrequentialStepInfo) -> bool:
            train_f.write(
                json.dumps(
                    {
                        "step": info.step,
                        "train_loss_nats": info.train_loss_nats,
                        "train_accuracy": info.train_accuracy,
                        "lr": info.lr,
                        "time_unix": time.time(),
                    }
                )
                + "\n"
            )
            train_f.flush()
            state["final_step"] = info.step
            at_ceiling = info.step >= max_steps
            stopping_eval = info.step % eval_every == 0 or at_ceiling
            if stopping_eval or info.step in curve_steps:
                val_loss_nats = evaluate_sft_nll_nats(
                    model, stop_examples, task_format, batch_size=batch_size, device=args.device
                )
                eval_f.write(
                    json.dumps(
                        {
                            "step": info.step,
                            "val_loss_nats": val_loss_nats,
                            "stopping_eval": stopping_eval,
                            "time_unix": time.time(),
                        }
                    )
                    + "\n"
                )
                eval_f.flush()
                if stopping_eval:
                    if tracker.update(val_loss_nats, step=info.step):
                        state["stop_reason"] = "converged"  # wins the ceiling tie-break
                    elif at_ceiling:
                        state["stop_reason"] = "max_steps"
            return state["stop_reason"] is not None

        train_started = time.time()
        train_prequential(
            model,
            {"train": train_examples, "test": test_examples, "tokenizer_hash": tok_hash},
            task_format,
            manifest,
            device=args.device,
            seed=t["seed"],
            # Default 1 = every step, the behaviour of every run before
            # 2026-07-27. _gradstat calls .item() once per trainable tensor
            # (112 of them at LoRA r=128 over 8 layers), and each is a
            # device->host sync inside the inner loop, so on a model this small
            # it is a real share of step time. Runs that do not read
            # gradstats.jsonl can raise the stride; step 0 is always logged
            # (0 % N == 0), so the spec 00 §4 artifact stays non-empty.
            gradstats_stride=t.get("gradstats_stride", 1),
            store=store,
            step_callback=step_callback,
        )
        train_wall_s = time.time() - train_started

    phase(6, "finalize — checkpoint + manifest (emergent snapshot truncation)")
    # Final checkpoint at runs/<id>/model (flat layout, spec 00 §1): the
    # complete state_dict, base + adapter tensors — same save path as
    # geode.train (save_pretrained); reload via geode.train.reapply_lora.
    model.save_pretrained(str(out_dir / "model"))
    taken = sorted(
        int(p.name.removeprefix("step_"))
        for p in (out_dir / "snapshots").glob("step_*")
        if (p / "adapter.safetensors").is_file()
    )
    stop_reason = state["stop_reason"] or "max_steps"
    if fixed_prefix_one_pass and (
        stop_reason != "max_steps" or state["final_step"] != steps_per_epoch
    ):
        raise RuntimeError(
            "fixed-prefix warm-start target did not complete its exact one-pass budget: "
            f"stop_reason={stop_reason!r}, final_step={state['final_step']}, "
            f"expected_step={steps_per_epoch}"
        )
    manifest.data["status"] = "complete"
    # Scheduled steps past the stopping point never materialize; the emergent
    # truncation is recorded here (declared schedule stays in snapshot_steps).
    manifest.data["snapshots_taken"] = taken
    manifest.data["experiment"]["target_result"] = {
        "final_step": state["final_step"],
        "stop_reason": stop_reason,
        # null if the run stopped before its first in-loop eval (JSON has no inf)
        "best_val_nats": None if math.isinf(tracker.best_nats) else tracker.best_nats,
        "min_val_nats": None if math.isinf(tracker.min_nats) else tracker.min_nats,
    }
    manifest.save(out_dir / "manifest.json")
    print(
        f"[evt] {cfg['run_id']} done: {stop_reason} at step {state['final_step']}, "
        f"min val {tracker.min_nats:.4f} nats (eps-gated best {tracker.best_nats:.4f}); "
        f"snapshots taken {len(taken)}/{len(schedule)}. Checkpoint: {out_dir / 'model'}"
    )
    # Printed, not stored: no run in this project has ever recorded wall clock,
    # so "would a faster GPU help?" has never been answerable. steps/s against
    # the FLOP estimate above is what says whether the run is compute-bound or
    # overhead-bound on a model this small.
    steps_per_s = state["final_step"] / train_wall_s if train_wall_s > 0 else float("nan")
    print(
        f"[evt] wall clock: {train_wall_s / 60:.1f} min of training "
        f"({steps_per_s:.2f} steps/s over {state['final_step']} steps, "
        f"batch {batch_size}, eval_every {eval_every})"
    )
    print(json.dumps(manifest.data["experiment"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
