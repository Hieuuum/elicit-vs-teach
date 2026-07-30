"""Persist a fixed-dose Phase-3 input-embedding warm-start.

One invocation trains and persists one LR candidate from the identical operator-
addition parent. It moves only the configured input rows on 512 correct NL-add
examples, then records a disjoint held-out NL score plus operator retention. The
target stream and D_p3_nl_eval are never read by this script; selection happens
after all four durable candidates exist.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch

from geode.arith import exact_match_accuracy, tokenize_with_spans
from geode.edl.masking import TaskFormat, masking_config_hash
from geode.train import (
    assert_untie_is_noop,
    evaluate_sft_nll_nats,
    resolve_warmstart_rows,
    split_indices,
    train_embedding_rows,
)
from geode.zoo import checkpoint_dir, load_model, register_run, require_parent_ready, tokenizer_hash
from train import REPO_ROOT, git_commit, load_config, phase
from train_sft import load_frozen_parquet


def _tokenize_rows(df, tokenizer, *, append_eos: bool):
    spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    return tokenize_with_spans(df["full_text"].tolist(), spans, tokenizer, append_eos=append_eos)


def _accuracy(
    model: torch.nn.Module,
    tokenizer,
    examples,
    answers: list[int],
    *,
    device: str,
    batch_size: int,
) -> float:
    prompts = [example.input_ids[: example.label_span[0]] for example in examples]
    accuracy, _ = exact_match_accuracy(
        model, tokenizer, prompts, answers, device=device, batch_size=batch_size
    )
    return accuracy


def _operator_eval(cfg: dict, tokenizer) -> tuple[list, list[int]]:
    df = load_frozen_parquet(cfg)
    _, val_indices = split_indices(len(df), cfg["data"]["val_fraction"], cfg["data"]["seed"])
    picked = random.Random(316).sample(val_indices, 1024)
    rows = df.iloc[picked]
    return _tokenize_rows(rows, tokenizer, append_eos=False), rows["true_answer"].astype(
        int
    ).tolist()


def candidate_run_id(base_run_id: str, lr: float) -> str:
    """Return the stable run id for one member of the frozen LR grid."""
    tags = {0.001: "1e-3", 0.01: "1e-2", 0.1: "1e-1", 1.0: "1e0"}
    for value, tag in tags.items():
        if abs(lr - value) <= 1e-12:
            return f"{base_run_id}-lr{tag}"
    raise ValueError(f"embedding warm-start: lr={lr} is outside the pinned grid")


def manifest_fields(
    cfg: dict,
    *,
    candidate: dict[str, Any],
    n_trainable: int,
    est_usd: float,
    init_from: Path,
    mask_hash: str,
) -> dict[str, Any]:
    warm = cfg["warmstart"]
    parent = cfg["experiment"]["parent_run_id"]
    return {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": git_commit(),
        "regime": "unknown",
        "base_model": {"hf_id": f"zoo-run/{parent}", "revision": "none"},
        "task": {"name": cfg["task"]["name"], "format_version": cfg["task"]["format_version"]},
        "dataset": {
            "name": f"{cfg['data']['hf_id']}:{cfg['data']['file']}",
            "n_unique_examples": warm["train_rows"],
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
                "name": "adamw",
                "lr": candidate["lr"],
                "batch_size": warm["train_rows"],
                "micro_batch_size": warm["micro_batch_size"],
                "betas": list(cfg["optimizer"]["betas"]),
                "weight_decay": 0.0,
                "grad_clip": cfg["training"]["grad_clip"],
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": None,
            "max_steps": warm["steps"],
            "stopping": {"eps_nats": None, "k": None, "min_steps": None},
            "epochs_total": warm["steps"],
            "seed": warm["seed"],
        },
        "trainable_param_count": n_trainable,
        "snapshot_steps": [],
        "cost": {"gpu_type": cfg["gpu"]["type"], "est_usd": est_usd, "actual_usd": None},
        "status": "running",
        "experiment": cfg["experiment"]
        | {
            "gates": {},
            "init_from": str(init_from),
            "masking_config_hash": mask_hash,
            "data_order_hash": cfg["data"]["order_hash"],
            "embedding_warmstart": {
                "tokens": list(warm["tokens"]),
                "rows": list(warm["rows"]),
                "lr_grid": list(warm["lr_grid"]),
                "candidate_lr": candidate["lr"],
                "train_rows": warm["train_rows"],
                "selection_rows": warm["selection_rows"],
                "steps": warm["steps"],
                "micro_batch_size": warm["micro_batch_size"],
                "example_presentations": warm["train_rows"] * warm["steps"],
                "selection_rule": (
                    "maximize held-out NL exact match; minimize held-out NL NLL; "
                    "then smaller LR; operator retention report-only"
                ),
                "final": candidate,
            },
        },
    }


def _verify_saved_checkpoint(
    parent_checkpoint: Path, output_checkpoint: Path, rows: set[int]
) -> None:
    """Require the persisted checkpoint to differ only in declared input rows."""
    from safetensors import safe_open
    from transformers import AutoConfig

    embedding_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"
    parent_path = parent_checkpoint / "model.safetensors"
    output_path = output_checkpoint / "model.safetensors"
    with (
        safe_open(parent_path, framework="pt") as parent,
        safe_open(output_path, framework="pt") as output,
    ):
        parent_keys = set(parent.keys())
        output_keys = set(output.keys())
        if output_keys != parent_keys | {head_key}:
            raise RuntimeError(
                "embedding warm-start: persisted state keys differ beyond the untied lm_head"
            )
        parent_embedding = parent.get_tensor(embedding_key)
        output_embedding = output.get_tensor(embedding_key)
        keep = [row for row in range(parent_embedding.shape[0]) if row not in rows]
        if not torch.equal(parent_embedding[keep], output_embedding[keep]):
            raise RuntimeError("embedding warm-start: an undeclared persisted row changed")
        if any(torch.equal(parent_embedding[row], output_embedding[row]) for row in rows):
            raise RuntimeError("embedding warm-start: a declared persisted row did not move")
        if not torch.equal(output.get_tensor(head_key), parent_embedding):
            raise RuntimeError("embedding warm-start: frozen lm_head differs from the parent")
        for key in parent_keys - {embedding_key}:
            if not torch.equal(parent.get_tensor(key), output.get_tensor(key)):
                raise RuntimeError(f"embedding warm-start: persisted parameter {key} changed")
    if AutoConfig.from_pretrained(output_checkpoint).tie_word_embeddings:
        raise RuntimeError("embedding warm-start: saved checkpoint re-tied the embeddings")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    phase(1, "config + tokenizer + frozen datasets")
    cfg = load_config(args.config, args.override)
    warm = cfg["warmstart"]
    lr = float(args.lr)
    if not any(abs(lr - float(value)) <= 1e-12 for value in warm["lr_grid"]):
        raise ValueError(f"warm-start --lr {lr} is outside pinned grid {warm['lr_grid']}")
    cfg["run_id"] = candidate_run_id(cfg["run_id"], lr)
    if warm["train_rows"] + warm["selection_rows"] != cfg["data"]["n_examples"]:
        raise ValueError("warm-start train_rows + selection_rows must equal data.n_examples")
    if warm["steps"] != 200 or warm["train_rows"] != 512:
        raise ValueError("warm-start dose drift: expected 512 rows x 200 steps")
    if [float(lr) for lr in warm["lr_grid"]] != [0.001, 0.01, 0.1, 1.0]:
        raise ValueError("warm-start LR-grid drift")

    from transformers import AutoTokenizer

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = resolve_warmstart_rows(tokenizer, warm["tokens"])
    if tuple(warm["rows"]) != rows:
        raise ValueError(f"warm-start config rows {warm['rows']} != resolved token rows {rows}")

    df = load_frozen_parquet(cfg)
    if len(df) != cfg["data"]["n_examples"]:
        raise ValueError(
            f"warm-start data has {len(df)} rows, expected {cfg['data']['n_examples']}"
        )
    train_df = df.iloc[: warm["train_rows"]]
    selection_df = df.iloc[warm["train_rows"] :]
    train_examples = _tokenize_rows(train_df, tokenizer, append_eos=True)
    selection_train_examples = _tokenize_rows(selection_df, tokenizer, append_eos=True)
    selection_prompt_examples = _tokenize_rows(selection_df, tokenizer, append_eos=False)
    selection_answers = selection_df["true_answer"].astype(int).tolist()
    operator_config = args.config.parent / warm["operator_eval_config"]
    operator_cfg = load_config(operator_config, None)
    operator_examples, operator_answers = _operator_eval(operator_cfg, tokenizer)
    task_format = TaskFormat(name=cfg["task"]["name"], format_version=cfg["task"]["format_version"])

    phase(2, "parent + architecture + cost gate")
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    parent = cfg["experiment"]["parent_run_id"]
    require_parent_ready(
        parent,
        required_gates=tuple(cfg["experiment"]["parent_required_gates"]),
        store=store,
    )
    resolved_parent_checkpoint = checkpoint_dir(parent, store=store)
    if args.init_from.resolve() != resolved_parent_checkpoint.resolve():
        raise ValueError(
            f"warm-start --init-from {args.init_from} is not parent checkpoint "
            f"{resolved_parent_checkpoint}"
        )
    model = load_model(parent, store=store, device="cpu", checkpoint=args.init_from)
    got = (model.config.vocab_size, model.config.hidden_size, model.config.num_hidden_layers)
    want = (len(tokenizer), cfg["model"]["hidden_size"], cfg["model"]["num_hidden_layers"])
    if got != want:
        raise ValueError(f"warm-start architecture {got} != config {want}")
    n_params = sum(parameter.numel() for parameter in model.parameters())
    max_len = max(len(example.input_ids) for example in train_examples)
    flops = 6.0 * n_params * len(train_examples) * max_len * warm["steps"]
    gpu = cfg["gpu"]
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    est_usd = hours * gpu["usd_per_hour"]
    print(
        f"[evt] run_id={cfg['run_id']} rows={rows} dose={len(train_examples)} x "
        f"{warm['steps']} steps, candidate LR={lr}"
    )
    print(f"[evt] estimated training-only cost: ${est_usd:,.2f} ({hours:.3f} GPU-h)")
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    phase(3, "untie + fixed-protocol baselines")
    probe_len = min(len(example.input_ids) for example in operator_examples[:8])
    probe = torch.tensor([example.input_ids[:probe_len] for example in operator_examples[:8]])
    assert_untie_is_noop(model, probe, device=args.device)
    embedding = model.get_input_embeddings().weight
    original_embedding = embedding.detach().clone()
    baseline = {
        "lr": 0.0,
        "operator_accuracy": _accuracy(
            model,
            tokenizer,
            operator_examples,
            operator_answers,
            device=args.device,
            batch_size=warm["eval_batch_size"],
        ),
        "nl_accuracy": _accuracy(
            model,
            tokenizer,
            selection_prompt_examples,
            selection_answers,
            device=args.device,
            batch_size=warm["eval_batch_size"],
        ),
        "nl_loss_nats": evaluate_sft_nll_nats(
            model,
            selection_train_examples,
            task_format,
            batch_size=warm["eval_batch_size"],
            device=args.device,
        ),
    }
    print(f"[evt] baseline: {baseline}", flush=True)

    phase(4, "fixed-dose candidate + persisted checkpoint")
    embedding.data.copy_(original_embedding)
    result = train_embedding_rows(
        model,
        train_examples,
        task_format,
        rows=rows,
        lr=lr,
        steps=warm["steps"],
        micro_batch_size=warm["micro_batch_size"],
        device=args.device,
        betas=tuple(cfg["optimizer"]["betas"]),
        grad_clip=cfg["training"]["grad_clip"],
        seed=warm["seed"],
    )
    if set(result.moved_rows) != set(rows):
        raise RuntimeError(f"warm-start declared rows {rows} did not all move: {result.moved_rows}")
    candidate = {
        "lr": lr,
        "train_loss_nats": result.final_loss_nats,
        "row_delta_l2": result.row_delta_l2,
        "operator_accuracy": _accuracy(
            model,
            tokenizer,
            operator_examples,
            operator_answers,
            device=args.device,
            batch_size=warm["eval_batch_size"],
        ),
        "nl_accuracy": _accuracy(
            model,
            tokenizer,
            selection_prompt_examples,
            selection_answers,
            device=args.device,
            batch_size=warm["eval_batch_size"],
        ),
        "nl_loss_nats": evaluate_sft_nll_nats(
            model,
            selection_train_examples,
            task_format,
            batch_size=warm["eval_batch_size"],
            device=args.device,
        ),
    }
    mask_hash = masking_config_hash(task_format, tokenizer_hash(tokenizer))
    manifest = register_run(
        manifest_fields(
            cfg,
            candidate=candidate,
            n_trainable=len(rows) * model.config.hidden_size,
            est_usd=est_usd,
            init_from=args.init_from,
            mask_hash=mask_hash,
        ),
        store=store,
    )
    run_dir = store / "runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "warmstart_metrics.json").write_text(
        json.dumps({"baseline": baseline, "candidate": candidate}, indent=2) + "\n"
    )
    checkpoint = run_dir / "model"
    model.save_pretrained(checkpoint)
    _verify_saved_checkpoint(args.init_from, checkpoint, set(rows))

    phase(5, "reload verification + complete manifest")
    reloaded = load_model(cfg["run_id"], store=store, device="cpu", checkpoint=checkpoint)
    reloaded_embedding = reloaded.get_input_embeddings().weight.detach()
    if not torch.equal(reloaded_embedding, model.get_input_embeddings().weight.detach().cpu()):
        raise RuntimeError("embedding warm-start: reloaded checkpoint embedding differs")
    manifest.data["status"] = "complete"
    manifest.data["experiment"]["embedding_warmstart"]["baseline"] = baseline
    manifest.data["masking_config_hash"] = mask_hash
    manifest.save(run_dir / "manifest.json")
    print(f"[evt] {cfg['run_id']} complete: {candidate}; checkpoint {checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
