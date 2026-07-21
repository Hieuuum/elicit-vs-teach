"""Launch script for the training-run experiment (spec 02 §6.2).

Protocol-exempt glue: parses run YAML, packs data, registers the run in
geode.zoo, prints a cost estimate, and refuses to train without
--confirm-cost (CLAUDE.md budget rule). All learning logic lives in
geode.train; this file must stay thin.

Usage:
    python train.py --config configs/run1_pretrain.yaml [--override configs/pilot/run1_pretrain.yaml]
        [--device cuda] [--init-from <checkpoint_dir>] [--confirm-cost]
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

from geode.train import StoppingRule, pack_corpus, split_documents, train_full, train_val_split
from geode.zoo import register_run

REPO_ROOT = Path(__file__).resolve().parents[3]


def phase(n: int, msg: str) -> None:
    """Six-phase banner: every long silent stretch (packing, training)
    must be attributable to a phase on screen."""
    print(f"[evt] ===== phase {n}/6: {msg} =====", flush=True)


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
        rope_theta=m.get("rope_theta", 10000.0),
        tie_word_embeddings=m.get("tie_word_embeddings", True),
        max_position_embeddings=m.get("max_position_embeddings", cfg["data"]["seq_len"]),
    )
    return LlamaForCausalLM(config)


def load_texts(cfg: dict) -> list[str]:
    # TinyStories-v2 exists only as a delimiter-separated txt file inside the
    # v1 repo (verified 2026-07-18); the repo's parquet config is v1 data.
    from huggingface_hub import hf_hub_download

    d = cfg["data"]
    path = hf_hub_download(d["hf_id"], d["file"], repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        if d.get("max_documents"):
            texts = []
            for doc in split_documents(f):
                texts.append(doc)
                if len(texts) == d["max_documents"]:
                    break
        else:
            texts = list(split_documents(f))
    return texts


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


def manifest_fields(
    cfg: dict,
    n_params: int,
    n_docs: int,
    est_usd: float,
    init_from: Path | None,
    *,
    tokenizer: Any,
    precision: str,
    n_train_seqs: int,
    n_val_seqs: int,
) -> dict[str, Any]:
    # The manifest alone must answer "how exactly was this trained" —
    # spec 00 §2 (2026-07-20): record RESOLVED values, not config defaults
    # (the v2 cosine-vs-constant question required digging through
    # training_meta.json).
    from geode.zoo import tokenizer_hash

    t = cfg["train"]
    d = cfg["data"]
    return {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": git_commit(),
        "regime": "unknown",  # run 1 is shared; regimes attach to target runs
        "base_model": {"hf_id": "random-init/llama-arch-d512-L8", "revision": "none"},
        "task": {"name": "tinystories_pretrain", "format_version": "v1"},
        "dataset": {
            "name": f"{d['hf_id']}:{d['file']}",
            "n_unique_examples": n_docs,
            "seed": d["seed"],
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
                "micro_batch_size": t.get("micro_batch_size") or t["batch_size"],
                "betas": list(cfg["optimizer"]["betas"]),
                "weight_decay": cfg["optimizer"]["weight_decay"],
                "grad_clip": cfg["training"]["grad_clip"],
            },
            "lr_schedule": t.get("lr_schedule", "constant"),
            "min_lr": t.get("min_lr"),
            "precision": precision,
            "eval_every": t["eval_every"],
            "max_steps": t["max_steps"],
            "stopping": {"eps_nats": t["stopping"]["eps_nats"], "k": t["stopping"]["k"]},
            "epochs_total": t["epochs_total_planned"],
            "seed": t["seed"],
        },
        "trainable_param_count": n_params,
        "snapshot_steps": [],  # runs 1-4: final checkpoint only (spec 02 §6)
        "cost": {"gpu_type": cfg["gpu"]["type"], "est_usd": est_usd, "actual_usd": None},
        "status": "running",
        # Extra fields below ride as preserved unknowns (spec 00 V0.2) until
        # the experiment-block validation task lands (spec 02 §4).
        "experiment": cfg["experiment"]
        | {
            "gates": {},
            "model_config": dict(cfg["model"]) | {"vocab_size": len(tokenizer)},
            "tokenizer": {
                "path": str(cfg["tokenizer"]["path"]),
                "sha256": tokenizer_hash(tokenizer),
            },
            "data_config": {
                "file": d["file"],
                "seq_len": d["seq_len"],
                "val_fraction": d["val_fraction"],
                "max_documents": d.get("max_documents"),
                "n_train_seqs": n_train_seqs,
                "n_val_seqs": n_val_seqs,
            },
        }
        | ({"init_from": str(init_from)} if init_from is not None else {}),
    }


def main() -> int:
    # Wall-clock from script start: covers packing + train + save, i.e.
    # what the box actually bills, not just the GPU phase.
    run_started = datetime.datetime.now(datetime.UTC)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, default=None, help="e.g. a pilot overlay")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--packed-cache",
        type=Path,
        default=None,
        help="pack once, reuse: .pt of packed rows; loaded if present, else written after packing",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="save_pretrained checkpoint dir to load instead of random init "
        "(extension runs: pair with a constant-LR override and a NEW run_id)",
    )
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()
    if args.packed_cache is not None:
        # torch.save does not create parent dirs; discovering that only at
        # the cache write threw away a finished 40-min pack (2026-07-20).
        args.packed_cache.parent.mkdir(parents=True, exist_ok=True)

    phase(1, "config + tokenizer")
    cfg = load_config(args.config, args.override)

    from transformers import AutoTokenizer

    tok_path = cfg["tokenizer"]["path"]
    if not tok_path:
        print(
            "[evt] tokenizer.path is null — train the custom tokenizer first "
            "(scripts/make_tokenizer.py; see configs/run1_pretrain.yaml). Exiting."
        )
        return 1
    # Relative paths resolve against the config dir; anything that doesn't
    # resolve to a local directory passes through as an HF id.
    local = (args.config.parent / tok_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else tok_path)
    d = cfg["data"]
    # Packing 2.7M documents takes tens of CPU-minutes; the cache lets the LR
    # sweep + production launches reuse one packing pass. The key guards
    # against silently training on rows packed under a different config.
    cache_key = {
        "data_file": d["file"],
        "seq_len": d["seq_len"],
        "max_documents": d.get("max_documents"),
        "tokenizer_path": str(cfg["tokenizer"]["path"]),
    }
    phase(2, "corpus — load packed cache, or download + pack (~30-60 min CPU on a miss)")
    if args.packed_cache is not None and args.packed_cache.exists():
        # First output of the run: the multi-GB torch.load below is a silent
        # ~90 s window that already caused a double launch (2026-07-19).
        print(f"[evt] pid={os.getpid()} loading packed cache {args.packed_cache} ...", flush=True)
        blob = torch.load(args.packed_cache)
        if blob["key"] != cache_key:
            raise ValueError(
                f"--packed-cache mismatch: cache built for {blob['key']}, config wants {cache_key}"
            )
        # .long() restores int64 for the model/CE; no-op on pre-uint16 caches.
        seqs, n_docs = blob["seqs"].long(), blob["n_docs"]
        print(f"[evt] loaded {len(seqs)} packed rows from {args.packed_cache}", flush=True)
    else:
        print(f"[evt] loading + packing {d['hf_id']} ...", flush=True)
        texts = load_texts(cfg)
        seqs = pack_corpus(texts, tokenizer, d["seq_len"])
        n_docs = len(texts)
        if args.packed_cache is not None:
            # uint16 storage (vocab 10K << 65536): 4x smaller blob, 4x faster
            # load — the int64 cache was a 4.3 GB / ~90 s startup stall.
            if len(tokenizer) - 1 > torch.iinfo(torch.uint16).max:
                raise ValueError(f"--packed-cache: vocab {len(tokenizer)} overflows uint16 storage")
            torch.save(
                {"key": cache_key, "seqs": seqs.to(torch.uint16), "n_docs": n_docs},
                args.packed_cache,
            )
            print(f"[evt] wrote packed cache {args.packed_cache}", flush=True)
    train_seqs, val_seqs = train_val_split(seqs, d["val_fraction"], d["seed"])

    phase(3, "model — warm start from checkpoint" if args.init_from else "model — random init")
    if args.init_from is not None:
        from transformers import LlamaForCausalLM

        print(f"[evt] loading init checkpoint {args.init_from} ...", flush=True)
        model = LlamaForCausalLM.from_pretrained(args.init_from)
        # A wrong checkpoint path would silently train the wrong model on
        # real GPU budget; check the loaded arch against config + tokenizer.
        m = cfg["model"]
        got = (model.config.vocab_size, model.config.hidden_size, model.config.num_hidden_layers)
        want = (len(tokenizer), m["hidden_size"], m["num_hidden_layers"])
        if got != want:
            raise ValueError(
                f"--init-from arch mismatch: checkpoint has (vocab, hidden, layers)={got}, "
                f"config + tokenizer want {want}"
            )
    else:
        model = build_model(cfg, vocab_size=len(tokenizer))
    n_params = sum(p.numel() for p in model.parameters())
    est_usd, detail = estimate_cost_usd(cfg, n_params, train_seqs.numel())

    phase(4, "cost estimate + confirm gate")
    print(
        f"[evt] run_id={cfg['run_id']} docs={n_docs} train_seqs={len(train_seqs)} "
        f"val_seqs={len(val_seqs)} seq_len={cfg['data']['seq_len']}"
    )
    print(f"[evt] estimated cost: ${est_usd:,.2f}  ({detail})")
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    phase(5, "train — progress lands in eval_log.jsonl; stopping is automatic")
    # Store default (spec 00 §1, 2026-07-20): repo-root/geode-store when the
    # env var is unset. setdefault, because register_run resolves the env
    # var internally and both must agree on one root.
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    t = cfg["train"]
    precision = t.get("precision", "bf16") if args.device != "cpu" else "fp32"
    manifest = register_run(
        manifest_fields(
            cfg,
            n_params,
            n_docs,
            est_usd,
            args.init_from,
            tokenizer=tokenizer,
            precision=precision,
            n_train_seqs=len(train_seqs),
            n_val_seqs=len(val_seqs),
        )
    )
    out_dir = store / "runs" / cfg["run_id"] / "pretrain"
    print(f"[evt] store={store}", flush=True)

    result = train_full(
        model.to(args.device),
        train_seqs,
        val_seqs,
        lr=t["lr"],
        batch_size=t["batch_size"],
        micro_batch_size=t.get("micro_batch_size"),
        stopping=StoppingRule(eps_nats=t["stopping"]["eps_nats"], k=t["stopping"]["k"]),
        eval_every=t["eval_every"],
        max_steps=t["max_steps"],
        lr_schedule=t.get("lr_schedule", "constant"),
        min_lr=t.get("min_lr"),
        grad_clip=cfg["training"]["grad_clip"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
        device=args.device,
        seed=t["seed"],
        out_dir=out_dir,
        precision=precision,
    )

    phase(6, "finalize — manifest + checkpoint")
    run_ended = datetime.datetime.now(datetime.UTC)
    duration_s = (run_ended - run_started).total_seconds()
    manifest.data["status"] = "complete"
    manifest.data["experiment"]["pretrain_result"] = {
        "final_step": result.final_step,
        "best_val_nats": result.best_val_nats,
        "min_val_nats": result.min_val_nats,
        "stop_reason": result.stop_reason,
        "run_started_utc": run_started.isoformat(),
        "run_ended_utc": run_ended.isoformat(),
        "run_duration_s": round(duration_s, 1),
    }
    manifest.save(store / "runs" / cfg["run_id"] / "manifest.json")
    print(
        f"[evt] done: {result.stop_reason} at step {result.final_step}, "
        f"min val {result.min_val_nats:.4f} nats "
        f"(eps-gated best {result.best_val_nats:.4f}) "
        f"in {datetime.timedelta(seconds=round(duration_s))}. "
        f"Checkpoint: {result.checkpoint_dir}"
    )
    print(json.dumps(manifest.data["experiment"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
