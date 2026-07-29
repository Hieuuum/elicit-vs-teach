"""Launch script for the SFT runs 2-4 (spec 02 §6, §6.2).

Protocol-exempt glue, mirroring ``train.py`` (pretrain): parses run YAML,
loads a frozen arithmetic dataset from HF, converts answer char spans to
token spans, enforces the parent-gate DAG rule (EXPERIMENTS.md §3.1),
registers the run in geode.zoo, prints a cost estimate, and refuses to train
without --confirm-cost (CLAUDE.md budget rule). All learning logic lives in
``geode.train.sft``; this file must stay thin.

Usage (run 9):
    python3 train_sft.py --config configs/archive/runs/run9_llama1b_inst.yaml \
        [--override configs/sweeps/llama9/llama9_smoke.yaml] \
        --init-from meta-llama/Llama-3.2-1B \
        [--device cuda] [--confirm-cost]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from geode.arith import (
    format_valid,
    greedy_completions,
    load_frozen_parquet as _load_frozen_parquet,
    tokenize_with_spans,
)
from geode.edl.masking import TaskFormat, masking_config_hash
from geode.train import (
    BehavioralStoppingRule,
    StoppingRule,
    apply_lora,
    evaluate_sft_nll_nats,
    split_indices,
    train_sft,
)
from geode.zoo import register_run, require_parent_ready, tokenizer_hash
from train import REPO_ROOT, git_commit, load_config, phase


def load_frozen_parquet(cfg: dict):
    """Full-cfg adapter over ``geode.arith.load_frozen_parquet`` (V5.70).

    Binds the repo root so every existing importer keeps the
    ``load_frozen_parquet(cfg)`` contract (gates.py, dump_g5_predictions.py,
    train_embedding_warmstart.py). ``data.local_path`` (repo-root-relative or
    absolute) skips the download but never the hash check.
    """
    return _load_frozen_parquet(cfg["data"], root=REPO_ROOT)


def own_lora_block(cfg: dict, config_path: Path, override_path: Path | None) -> dict | None:
    """The run's LoRA config, iff the run YAML itself declares ``lora:``.

    ``common.yaml`` carries a shared ``lora:`` block for the runs-5/6 LoRA
    target family (its own comment: "kept here so both arm configs inherit
    identical values"); ``load_config``'s deep_merge means every install-stage
    config (run2/3/4, full FT) inherits it too, even though the run's own
    YAML never mentions LoRA. Gating on the *merged* ``cfg["lora"]`` would
    silently switch every existing full-FT launch to LoRA. Only a run (or its
    pilot override) that explicitly declares its own ``lora:`` block — run9
    style — opts in; the returned dict is still the fully merged
    ``cfg["lora"]``, so an override like run9's r/alpha still layers on
    common.yaml's target_modules/dropout defaults.
    """
    own = yaml.safe_load(config_path.read_text()) or {}
    if "lora" in own:
        return cfg["lora"]
    if override_path is not None:
        override = yaml.safe_load(override_path.read_text()) or {}
        if "lora" in override:
            return cfg["lora"]
    return None


def manifest_fields(
    cfg: dict,
    n_params: int,
    n_rows: int,
    est_usd: float,
    init_from: Path,
    mask_hash: str,
    *,
    precision: str,
    lora_cfg: dict | None,
    step0: dict[str, float],
    device: str,
) -> dict[str, Any]:
    t = cfg["train"]
    parent = cfg["experiment"]["parent_run_id"]
    external_base = cfg["experiment"].get("external_base")
    base_hf_id = external_base if not parent and external_base else f"zoo-run/{parent}"
    return {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": git_commit(),
        "regime": "unknown",  # regimes attach to target runs (5-6)
        "base_model": {
            "hf_id": base_hf_id,
            "revision": "none",
        },
        "task": {"name": cfg["task"]["name"], "format_version": cfg["task"]["format_version"]},
        "dataset": {
            "name": f"{cfg['data']['hf_id']}:{cfg['data']['file']}",
            "n_unique_examples": n_rows,
            "seed": cfg["data"]["seed"],
        },
        "training": {
            "method": "lora" if lora_cfg else "full_ft",
            "lora": (
                {
                    "rank": lora_cfg["r"],
                    "alpha": lora_cfg["alpha"],
                    "target_modules": list(lora_cfg["target_modules"]),
                    "dropout": lora_cfg["dropout"],
                    "sparse_param_count": None,
                }
                if lora_cfg
                else {
                    "rank": None,
                    "alpha": None,
                    "target_modules": [],
                    "dropout": None,
                    "sparse_param_count": None,
                }
            ),
            "optimizer": {
                "name": cfg["optimizer"]["name"],
                "lr": t["lr"],
                "batch_size": t["batch_size"],
                "micro_batch_size": t["batch_size"],  # geode.train.sft: no grad accumulation
                "betas": list(cfg["optimizer"]["betas"]),
                "weight_decay": cfg["optimizer"]["weight_decay"],
                "grad_clip": cfg["training"]["grad_clip"],
            },
            "lr_schedule": "constant",  # structural: geode.train.sft has no scheduler
            "min_lr": None,
            "precision": precision,
            "eval_every": t["eval_every"],
            "max_steps": t["max_steps"],
            "stopping": (
                {
                    "metric": "format_validity",
                    "threshold": t["stopping"]["threshold"],
                    "k": t["stopping"]["k"],
                    "n_prompts": t["stopping"]["n_prompts"],
                    "prompt_seed": t["stopping"]["prompt_seed"],
                }
                if t["stopping"].get("metric") == "format_validity"
                else {
                    # "train_loss" (dose runs, 2026-07-26) records its metric;
                    # the legacy val-loss rule stays keyless for byte-stable
                    # re-registration of existing configs.
                    **(
                        {"metric": "train_loss"}
                        if t["stopping"].get("metric") == "train_loss"
                        else {}
                    ),
                    "eps_nats": t["stopping"]["eps_nats"],
                    "k": t["stopping"]["k"],
                    "min_steps": t["stopping"].get("min_steps", 0),
                }
            ),
            "epochs_total": t["epochs_total_planned"],
            "seed": t["seed"],
        },
        "trainable_param_count": n_params,
        "snapshot_steps": [],  # runs 2-4: final checkpoint only (spec 02 §6)
        "cost": {"gpu_type": cfg["gpu"]["type"], "est_usd": est_usd, "actual_usd": None},
        "status": "running",
        # Extras ride as preserved unknowns (spec 00 V0.2).
        "experiment": cfg["experiment"]
        | {
            "gates": {},
            "init_from": str(init_from),
            "masking_config_hash": mask_hash,
            "data_order_hash": cfg["data"]["order_hash"],
            # What the stopping rule read BEFORE the first update (phase-0
            # lesson, 2026-07-26): runs 3/4 stopped at the k×eval_every floor
            # with G4 already 1.0000 on the Arm A parent, and no eval log
            # started before step 250, so "the rule fired" and "the rule
            # measured nothing" were indistinguishable for weeks.
            "step0": step0,
            # The dose rule's eps/k is calibrated from a recorded loss
            # trajectory, and a trajectory is only reproducible on the device
            # that produced it. Recording the device makes a mixed-device dose
            # curve visible in the manifests instead of silent (2026-07-26).
            "device": device,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, default=None, help="e.g. a sweep overlay")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--init-from",
        type=Path,
        required=True,
        help="save_pretrained checkpoint dir of the parent run (SFT never starts from random)",
    )
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    phase(1, "config + tokenizer")
    cfg = load_config(args.config, args.override)
    if cfg["train"].get("lr") is None:
        print(
            "[evt] train.lr is null — pin it from the installer LR sweep (or pass a sweep "
            "--override) before a canonical launch; a placeholder-lr run is a redo "
            "(run-2 incident, 2026-07-21). Exiting."
        )
        return 1
    if cfg["train"].get("max_steps") is None:
        print("[evt] train.max_steps is null — the cost ceiling must be pinned. Exiting.")
        return 1
    if cfg["train"]["stopping"].get("metric") == "train_loss" and (
        cfg["train"]["stopping"].get("eps_nats") is None
    ):
        print(
            "[evt] train.stopping.eps_nats is null — the dose rule's eps/k is pinned at "
            "config time from the calibration pilots (decisions.md 2026-07-26), never "
            "inherited from the target-stage rule by default. Exiting."
        )
        return 1
    lora_cfg = own_lora_block(cfg, args.config, args.override)

    from transformers import AutoTokenizer, LlamaForCausalLM

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    phase(2, "dataset — download frozen parquet, verify order_hash, tokenize spans")
    df = load_frozen_parquet(cfg)
    if cfg["data"].get("max_rows"):  # pilot/smoke overlays; hash checked on the full file first
        df = df.head(cfg["data"]["max_rows"])
    n_rows = len(df)
    texts = df["full_text"].tolist()
    char_spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    print(f"[evt] {cfg['data']['file']}: {n_rows} rows, order_hash verified", flush=True)
    examples = tokenize_with_spans(texts, char_spans, tokenizer, append_eos=True)
    max_len = max(len(ex.input_ids) for ex in examples)
    print(f"[evt] tokenized: max {max_len} tokens/example (expected ≈34 incl. EOS)", flush=True)
    d = cfg["data"]
    if cfg["train"]["stopping"].get("metric") == "train_loss":
        # Dose runs (spec 02 §6, 2026-07-26): the whole tiny set trains; the
        # ε/k rule reads the full-dose training loss, so no val carve exists
        # (split_indices would refuse n=1 anyway). Gates score externally.
        train_examples, val_examples = list(examples), []
    else:
        train_idx, val_idx = split_indices(n_rows, d["val_fraction"], d["seed"])
        train_examples = [examples[i] for i in train_idx]
        val_examples = [examples[i] for i in val_idx]

    phase(3, "model — warm start from parent checkpoint")
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

    if lora_cfg:
        apply_lora(model, rank=lora_cfg["r"], alpha=lora_cfg["alpha"], seed=cfg["train"]["seed"])
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[evt] LoRA applied: r={lora_cfg['r']} alpha={lora_cfg['alpha']} "
            f"trainable={n_trainable / 1e6:.1f}M of {n_params / 1e6:.1f}M",
            flush=True,
        )
    else:
        n_trainable = n_params

    phase(4, "parent gates + cost estimate + confirm gate")
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    parent = cfg["experiment"]["parent_run_id"]
    external_base = cfg["experiment"].get("external_base")
    if not parent:
        if not external_base:
            print(
                "[evt] experiment.parent_run_id is null — pin it to the floor-1 run id "
                "(see configs/archive/runs/run2_algo.yaml). Exiting."
            )
            return 1
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
        print(f"[evt] parent '{parent}' complete, gates pass", flush=True)

    t = cfg["train"]
    gpu = cfg["gpu"]
    epochs = cfg.get("cost", {}).get("assumed_epochs_for_estimate", 1)
    # Set-max right padding (geode.train.sft) means every row costs max_len.
    # LoRA doesn't change the forward/backward FLOP order here (still a full
    # forward + backward through every layer), so the estimate always uses
    # full-model n_params regardless of training.method.
    flops = 6.0 * n_params * (len(train_examples) * max_len) * epochs
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    est_usd = hours * gpu["usd_per_hour"]
    steps_per_epoch = len(train_examples) // t["batch_size"]
    print(
        f"[evt] run_id={cfg['run_id']} train={len(train_examples)} val={len(val_examples)} "
        f"steps/epoch={steps_per_epoch} max_steps={t['max_steps']}"
    )
    print(f"[evt] estimated cost: ${est_usd:,.2f}  ({hours:.2f} GPU-h @ ${gpu['usd_per_hour']}/h)")
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    phase(5, "stopping rule + step-0 baseline (what the rule reads before any update)")
    task_format = TaskFormat(name=cfg["task"]["name"], format_version=cfg["task"]["format_version"])
    mask_hash = masking_config_hash(task_format, tokenizer_hash(tokenizer))
    precision = t.get("precision", "bf16") if args.device != "cpu" else "fp32"

    s = t["stopping"]
    if s.get("metric") == "format_validity":
        # Behavioral stop (spec 02 §6, runs 3-4): greedy decode on held-out
        # token-prefix prompts, stop at the k-th consecutive eval >= threshold.
        picks = random.Random(s["prompt_seed"]).sample(range(len(val_examples)), s["n_prompts"])
        prompt_ids = [val_examples[i].input_ids[: val_examples[i].label_span[0]] for i in picks]

        def behavioral_eval() -> float:
            completions = greedy_completions(
                model, tokenizer, prompt_ids, device=args.device, batch_size=t["batch_size"]
            )
            return sum(format_valid("Answer:" + c) for c in completions) / len(completions)

        stopping = BehavioralStoppingRule(threshold=s["threshold"], k=s["k"])
        print(
            f"[evt] behavioral stop: format_validity >= {s['threshold']} on "
            f"{s['n_prompts']} held-out prompts, k={s['k']}, every {t['eval_every']} steps",
            flush=True,
        )
    else:
        behavioral_eval = None
        stopping = StoppingRule(
            eps_nats=s["eps_nats"],
            k=s["k"],
            min_steps=s.get("min_steps", 0),
        )
        if s.get("metric") == "train_loss":
            print(
                f"[evt] dose stop: eps/k plateau on the full-dose training loss "
                f"(eps {s['eps_nats']} nats, k={s['k']}, eval every {t['eval_every']} steps)",
                flush=True,
            )

    # Step 0, always recorded (2026-07-26): the metric the stopping rule is
    # about to read, measured before the first update, so "the rule fired"
    # can be told apart from "the rule was already satisfied" (phase 0).
    model.to(args.device)
    step0: dict[str, float] = {}
    if s.get("metric") == "train_loss":
        step0["dose_loss_nats"] = evaluate_sft_nll_nats(
            model, train_examples, task_format, batch_size=t["batch_size"], device=args.device
        )
    else:
        step0["val_loss_nats"] = evaluate_sft_nll_nats(
            model, val_examples, task_format, batch_size=t["batch_size"], device=args.device
        )
    if behavioral_eval is not None:
        step0["format_validity"] = float(behavioral_eval())
    print(f"[evt] step 0 (before any update): {step0}", flush=True)

    manifest = register_run(
        manifest_fields(
            cfg,
            n_trainable,
            n_rows,
            est_usd,
            args.init_from,
            mask_hash,
            precision=precision,
            lora_cfg=lora_cfg,
            step0=step0,
            device=args.device,
        )
    )
    # Flat run layout (spec 00 §1, 2026-07-21): logs + training_meta.json at
    # the run root, checkpoint at runs/<id>/model. Pre-migration runs keep an
    # sft/ phase dir; readers accept both, migrate_store_layout.py converts
    # explicitly.
    out_dir = store / "runs" / cfg["run_id"]
    print(f"[evt] store={store}", flush=True)

    phase(6, "train — progress lands in eval_log.jsonl; stopping is automatic")
    train_started = time.time()
    result = train_sft(
        model,
        train_examples,
        val_examples,
        task_format,
        lr=t["lr"],
        batch_size=t["batch_size"],
        stopping=stopping,
        behavioral_eval=behavioral_eval,
        eval_every=t["eval_every"],
        max_steps=t["max_steps"],
        grad_clip=cfg["training"]["grad_clip"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
        device=args.device,
        seed=t["seed"],
        out_dir=out_dir,
        precision=precision,
        stopping_metric=("train_loss" if s.get("metric") == "train_loss" else "val_loss"),
    )
    train_wall_s = time.time() - train_started

    phase(7, "finalize — manifest + checkpoint")
    manifest.data["status"] = "complete"
    manifest.data["experiment"]["sft_result"] = {
        "final_step": result.final_step,
        "best_val_nats": result.best_val_nats,
        "min_val_nats": result.min_val_nats,
        "stop_reason": result.stop_reason,
    }
    manifest.save(store / "runs" / cfg["run_id"] / "manifest.json")
    print(
        f"[evt] {cfg['run_id']} done: {result.stop_reason} at step {result.final_step}, "
        f"min val {result.min_val_nats:.4f} nats "
        f"(eps-gated best {result.best_val_nats:.4f}). Checkpoint: {result.checkpoint_dir}"
    )
    # See train_target.py: no run in this project has recorded wall clock, so
    # whether these runs are compute-bound has never been checkable. Compare
    # steps/s against the FLOP-derived estimate printed in phase 4.
    steps_per_s = result.final_step / train_wall_s if train_wall_s > 0 else float("nan")
    print(
        f"[evt] wall clock: {train_wall_s / 60:.1f} min of training "
        f"({steps_per_s:.2f} steps/s over {result.final_step} steps, "
        f"batch {t['batch_size']}, eval_every {t['eval_every']})"
    )
    print(json.dumps(manifest.data["experiment"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
