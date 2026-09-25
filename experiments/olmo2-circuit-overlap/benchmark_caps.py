"""Repeat saved native prompts at shorter output caps; no probes or diagnostics."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from geode.circuits.artifacts import read_jsonl, sha256_file, write_json
from geode.circuits.checkpoints import select_checkpoints
from geode.circuits.data import grade
from geode.circuits.execution import RunBudget, generate_answers
from geode.circuits.plan import example_from_dict
from geode.zoo.activations import tokenizer_hash

TASKS = ("cruxeval_input", "cruxeval_output", "gsm_symbolic")
IDENTITY = ("id", "task", "group", "prompt", "answer", "metadata", "options", "label")


def compare_rows(old: list[dict], new: list[dict]) -> dict:
    """Match identity exactly; a lost answer must be balanced by the accuracy delta."""
    old_by_id = {r["id"]: r for r in old}
    new_by_id = {r["id"]: r for r in new}
    if (
        len(old_by_id) != len(old)
        or len(new_by_id) != len(new)
        or old_by_id.keys() != new_by_id.keys()
    ):
        raise ValueError("Duplicate or mismatched saved example IDs")
    records, lost, gained = [], [], []
    for row in new:
        before = old_by_id[row["id"]]
        for key in IDENTITY:
            a, b = before[key], row[key]
            if key == "options":
                a, b = tuple(a), tuple(b)
            if a != b:
                raise ValueError(f"Saved input identity changed: {row['id']}/{key}")
        if before["correct"] and not row["correct"]:
            lost.append(row["id"])
        if row["correct"] and not before["correct"]:
            gained.append(row["id"])
        keys = (
            "output_tokens",
            "truncated",
            "correct",
            "prediction",
            "generation",
            "context_overflow",
            "parse_failure",
            "timed_out",
            "execution_failure",
        )
        records.append(
            {
                **{key: row[key] for key in IDENTITY},
                "old": {key: before.get(key) for key in keys},
                "new": {key: row.get(key) for key in keys},
            }
        )
    return {
        "n_rows": len(new),
        "old_correct": sum(r["correct"] for r in old),
        "new_correct": sum(r["correct"] for r in new),
        "lost_ids": lost,
        "gained_ids": gained,
        "old_output_tokens": sum(r["output_tokens"] for r in old),
        "new_output_tokens": sum(r["output_tokens"] for r in new),
        "old_truncated": sum(r["truncated"] for r in old),
        "new_truncated": sum(r["truncated"] for r in new),
        "examples": records,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=["init", "stage2", "rlvr2"])
    p.add_argument("--coding-max-new-tokens", type=int, default=192)
    p.add_argument("--math-max-new-tokens", type=int, default=384)
    p.add_argument("--hourly-rate", type=float, required=True)
    p.add_argument("--confirm-cost", action="store_true")
    return p


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parser().parse_args()
    budget = RunBudget(900, args.hourly_rate, args.confirm_cost)
    if min(args.coding_max_new_tokens, args.math_max_new_tokens) < 1:
        raise ValueError("Positive output caps required")
    checkpoints = select_checkpoints(args.stages)
    examples = [
        example_from_dict(r)
        for r in read_jsonl(args.pilot / "selected_examples.jsonl")
        if r["task"] in TASKS
    ]
    by_task = {task: [ex for ex in examples if ex.task == task] for task in TASKS}
    if any(not rows for rows in by_task.values()):
        raise ValueError("All three saved generative task pools are required")
    # Validate all inputs and the previous 2048/batch8 protocol before loading a model.
    prior, checkpoint_records = {}, {}
    for cp in checkpoints:
        checkpoint_records[cp.stage] = json.loads(
            (args.pilot / cp.stage / "checkpoint.json").read_text()
        )
        if any(
            checkpoint_records[cp.stage][key] != getattr(cp, key)
            for key in ("repo", "revision", "stage")
        ):
            raise ValueError("Saved checkpoint identity differs from pinned checkpoint")
        prior[cp.stage] = [
            r for r in read_jsonl(args.pilot / cp.stage / "behavior.jsonl") if r["task"] in TASKS
        ]
        if {r["id"] for r in prior[cp.stage]} != {ex.id for ex in examples}:
            raise ValueError("Saved behavior does not exactly cover selected examples")
        selected = {ex.id: asdict(ex) for ex in examples}
        compare_rows(prior[cp.stage], [{**r, **selected[r["id"]]} for r in prior[cp.stage]])
        timings = json.loads((args.pilot / cp.stage / "timing.json").read_text())
        for task in TASKS:
            row = next(r for r in timings if r["component"] == "behavior" and r["task"] == task)
            if (row["batch_size"], row["max_context"], row["max_new_tokens"]) != (8, 4096, 2048):
                raise ValueError("Baseline must use batch8/context4096/output2048")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "maximum_compute_usd": budget.estimated_usd,
        "protocol": "same native prompts, greedy BF16 SDPA batch8 context4096; only output cap changes",
        "selected_examples_sha256": sha256_file(args.pilot / "selected_examples.jsonl"),
        "script_sha256": sha256_file(Path(__file__)),
        "startup": [],
        "tasks": [],
    }
    path = args.output / "cap_benchmark.json"
    write_json(path, result)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 GPU required")
        for cp in checkpoints:
            budget.check()
            started = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(
                cp.repo, revision=cp.revision, local_files_only=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer_hash(tokenizer) != checkpoint_records[cp.stage]["tokenizer_hash"]:
                raise ValueError("Saved checkpoint tokenizer hash mismatch")
            torch.manual_seed(0)
            model = (
                AutoModelForCausalLM.from_pretrained(
                    cp.repo,
                    revision=cp.revision,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    local_files_only=True,
                )
                .to("cuda")
                .eval()
            )
            result["startup"].append(
                {
                    **cp.to_dict(),
                    "seconds": time.monotonic() - started,
                    "tokenizer_hash": tokenizer_hash(tokenizer),
                    "baseline_behavior_sha256": sha256_file(
                        args.pilot / cp.stage / "behavior.jsonl"
                    ),
                }
            )
            for task, pool in by_task.items():
                budget.check()
                cap = (
                    args.math_max_new_tokens
                    if task == "gsm_symbolic"
                    else args.coding_max_new_tokens
                )
                stops = pool[0].metadata.get("stop_strings")
                if any(ex.metadata.get("stop_strings") != stops for ex in pool):
                    raise ValueError("Task stop strings differ across saved examples")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.monotonic()
                generated = generate_answers(
                    model,
                    tokenizer,
                    [ex.prompt for ex in pool],
                    device="cuda",
                    batch_size=8,
                    max_context=4096,
                    max_new_tokens=cap,
                    stop_strings=stops,
                    budget=budget,
                )
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                rows = [
                    {**asdict(ex), **gen, **grade(ex, gen["generation"])}
                    for ex, gen in zip(pool, generated, strict=True)
                ]
                comparison = compare_rows([r for r in prior[cp.stage] if r["task"] == task], rows)
                comparison.update(
                    stage=cp.stage,
                    task=task,
                    max_new_tokens=cap,
                    batch_size=8,
                    max_context=4096,
                    stop_strings=stops,
                    generation_seconds=elapsed,
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
                result["tasks"].append(comparison)
                write_json(path, result)
                print(
                    json.dumps(
                        {key: value for key, value in comparison.items() if key != "examples"}
                    ),
                    flush=True,
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        budget.check()
        result.update(status="complete", elapsed_seconds=time.monotonic() - budget.started)
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(path, result)


if __name__ == "__main__":
    main()
