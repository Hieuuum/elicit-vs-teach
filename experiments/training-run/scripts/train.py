"""Launch script for the training-run experiment (spec 02 §6.2).

Protocol-exempt glue: parses run YAML, packs data, registers the run in
geode.zoo, prints a cost estimate, and refuses to train without
--confirm-cost (CLAUDE.md budget rule). All learning logic lives in
geode.train; this file must stay thin.

Usage:
    python train.py --config configs/run1_pretrain.yaml [--override configs/pilot/run1_pretrain.yaml]
        [--device cuda] [--confirm-cost]
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from geode.train import StoppingRule, pack_corpus, train_full, train_val_split
from geode.zoo import register_run


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Path, override_path: Path | None) -> dict:
    here = config_path.parent
    common = (
        yaml.safe_load((here / "common.yaml").read_text())
        if (here / "common.yaml").exists()
        else {}
    )
    cfg = deep_merge(common, yaml.safe_load(config_path.read_text()))
    if override_path is not None:
        cfg = deep_merge(cfg, yaml.safe_load(override_path.read_text()))
    return cfg


def build_model(cfg: dict, vocab_size: int) -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    m = cfg["model"]
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        num_key_value_heads=m["num_key_value_heads"],
        rope_theta=m.get("rope_theta", 500000.0),
        tie_word_embeddings=m.get("tie_word_embeddings", True),
        max_position_embeddings=cfg["data"]["seq_len"],
    )
    return LlamaForCausalLM(config)


def load_texts(cfg: dict) -> list[str]:
    from datasets import load_dataset

    d = cfg["data"]
    ds = load_dataset(d["hf_id"], split=d["train_split"])
    texts = ds[d["text_field"]]
    if d.get("max_documents"):
        texts = texts[: d["max_documents"]]
    return list(texts)


def estimate_cost_usd(cfg: dict, n_params: int, n_train_tokens: int) -> tuple[float, str]:
    gpu = cfg["gpu"]
    epochs = cfg.get("cost", {}).get("assumed_epochs_for_estimate", 2)
    flops = 6.0 * n_params * n_train_tokens * epochs
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    usd = hours * gpu["usd_per_hour"]
    detail = (
        f"params={n_params / 1e6:.1f}M tokens/epoch={n_train_tokens / 1e9:.3f}B "
        f"assumed_epochs={epochs} -> {hours:.1f} GPU-h @ ${gpu['usd_per_hour']}/h "
        f"(util={gpu['utilization']}, {gpu['tflops_bf16']} TFLOPs bf16)"
    )
    return usd, detail


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def manifest_fields(cfg: dict, n_params: int, n_docs: int, est_usd: float) -> dict[str, Any]:
    t = cfg["train"]
    return {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": git_commit(),
        "regime": "unknown",  # run 1 is shared; regimes attach to target runs
        "base_model": {"hf_id": "random-init/llama-3.2-1b-arch", "revision": "none"},
        "task": {"name": "tinystories_pretrain", "format_version": "v1"},
        "dataset": {
            "name": cfg["data"]["hf_id"],
            "n_unique_examples": n_docs,
            "seed": cfg["data"]["seed"],
        },
        "training": {
            "method": "full_ft",
            "lora": {
                "rank": None,
                "alpha": None,
                "target_modules": [],
                "dropout": None,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": cfg["optimizer"]["name"],
                "lr": t["lr"],
                "batch_size": t["batch_size"],
                "weight_decay": cfg["optimizer"]["weight_decay"],
            },
            "epochs_total": t["epochs_total_planned"],
            "seed": t["seed"],
        },
        "trainable_param_count": n_params,
        "snapshot_steps": [],  # runs 1-4: final checkpoint only (spec 02 §6)
        "cost": {"gpu_type": cfg["gpu"]["type"], "est_usd": est_usd, "actual_usd": None},
        "status": "running",
        # Extra fields below ride as preserved unknowns (spec 00 V0.2) until
        # the experiment-block validation task lands (spec 02 §4).
        "experiment": cfg["experiment"] | {"gates": {}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, default=None, help="e.g. a pilot overlay")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["hf_id"])
    print(f"[evt] loading + packing {cfg['data']['hf_id']} ...", flush=True)
    texts = load_texts(cfg)
    seqs = pack_corpus(texts, tokenizer, cfg["data"]["seq_len"])
    train_seqs, val_seqs = train_val_split(seqs, cfg["data"]["val_fraction"], cfg["data"]["seed"])

    model = build_model(cfg, vocab_size=len(tokenizer))
    n_params = sum(p.numel() for p in model.parameters())
    est_usd, detail = estimate_cost_usd(cfg, n_params, train_seqs.numel())

    print(
        f"[evt] run_id={cfg['run_id']} docs={len(texts)} train_seqs={len(train_seqs)} "
        f"val_seqs={len(val_seqs)} seq_len={cfg['data']['seq_len']}"
    )
    print(f"[evt] estimated cost: ${est_usd:,.2f}  ({detail})")
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    store = Path(os.environ["GEODE_STORE"])
    manifest = register_run(manifest_fields(cfg, n_params, len(texts), est_usd))
    out_dir = store / "runs" / cfg["run_id"] / "pretrain"

    t = cfg["train"]
    result = train_full(
        model.to(args.device),
        train_seqs,
        val_seqs,
        lr=t["lr"],
        batch_size=t["batch_size"],
        stopping=StoppingRule(eps_nats=t["stopping"]["eps_nats"], k=t["stopping"]["k"]),
        eval_every=t["eval_every"],
        max_steps=t["max_steps"],
        grad_clip=cfg["training"]["grad_clip"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
        device=args.device,
        seed=t["seed"],
        out_dir=out_dir,
        precision=t.get("precision", "bf16") if args.device != "cpu" else "fp32",
    )

    manifest.data["status"] = "complete"
    manifest.data["experiment"]["pretrain_result"] = {
        "final_step": result.final_step,
        "best_val_nats": result.best_val_nats,
        "stop_reason": result.stop_reason,
    }
    manifest.save(store / "runs" / cfg["run_id"] / "manifest.json")
    print(
        f"[evt] done: {result.stop_reason} at step {result.final_step}, "
        f"best val {result.best_val_nats:.4f} nats. Checkpoint: {result.checkpoint_dir}"
    )
    print(json.dumps(manifest.data["experiment"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
