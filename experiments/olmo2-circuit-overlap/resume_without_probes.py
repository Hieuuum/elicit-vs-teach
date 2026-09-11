"""Resume saved behavior and circuits under an explicit no-further-probes amendment.

The original runner and frozen sampling plan remain unchanged. Completed stages
are retained; complete behavioral tasks are verified and reused. Circuit maps
are computed together so the original random-control RNG sequence is preserved.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from geode.circuits import runner
from geode.circuits.artifacts import (
    environment_provenance,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
    validate_matched_rows,
)
from geode.circuits.checkpoints import select_checkpoints
from geode.circuits.data import ethics_controls, native_aggregate, verify_data
from geode.circuits.execution import RunBudget
from geode.circuits.plan import example_from_dict, load_plan, validate_plan
from geode.circuits.sanity import validate_interventions


def reusable_tasks(rows, examples, tokenizer, args):
    """Reuse only whole tasks with identical prompts, references and generation caps."""
    expected, saved = defaultdict(list), defaultdict(list)
    for ex in examples:
        expected[ex.task].append(ex)
        if ex.task.startswith("ethics_"):
            expected[ex.task].extend(ethics_controls(ex, tokenizer))
    for row in rows:
        saved[row["task"]].append(row)
    if set(saved) - set(expected):
        raise ValueError("Saved behavior contains an unexpected task")
    for task, actual in saved.items():
        target = expected[task]
        if len(actual) != len(target):
            raise ValueError(f"Partial saved behavioral task: {task}")
        for row, ex in zip(actual, target):
            if example_from_dict({key: row[key] for key in asdict(ex)}) != ex:
                raise ValueError(f"Saved behavioral input changed: {ex.id}")
            if not task.startswith("ethics_") and row.get(
                "max_new_tokens"
            ) != runner.generation_cap(args, task):
                raise ValueError(f"Saved generation cap changed: {task}")
    return set(saved)


def saved_circuit_times(stage, directory, tasks):
    """Reuse a complete circuit phase only after checking its saved arrays and controls."""
    events = {}
    with (directory.parent / "full.log").open() as stream:
        for line in stream:
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") == stage.name and row.get("event") == "circuits_done":
                events[row["task"]] = {k: v for k, v in row.items() if k != "event"}
    if set(events) != set(tasks):
        return None
    for task in tasks:
        base = stage / "circuits" / task
        meta = json.loads(base.with_suffix(".json").read_text())
        reference = json.loads((directory / "init" / "circuits" / f"{task}.json").read_text())
        validate_matched_rows(reference, meta)
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
            scores = data["scores"]
            if (
                scores.shape != (len(meta["item_ids"]), len(meta["node_names"]))
                or not np.isfinite(scores).all()
            ):
                raise ValueError(f"Invalid saved circuit array: {task}")
        validate_interventions(
            json.loads(base.with_name(task + "_interventions.json").read_text()),
            meta,
            require_typed=True,
        )
    return [events[task] for task in sorted(tasks)]


def resume(directory: Path, max_wall_seconds: float, confirm_cost: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from geode.zoo.activations import tokenizer_hash

    meta_path = directory / "run_metadata.json"
    metadata = json.loads(meta_path.read_text())
    args = argparse.Namespace(**metadata["config"])
    args.max_wall_seconds = max_wall_seconds
    args.confirm_cost = confirm_cost
    runner.validate_config(args)
    budget = RunBudget(max_wall_seconds, args.hourly_rate, confirm_cost)
    source = verify_data(args.data)
    plan = load_plan(args.plan, args, source)
    # The original runner persists its metadata fingerprint only on exit; a
    # stopped worker can still have the initial metadata plus its frozen plan.
    original_plan = load_plan(directory / "frozen_plan.json", args, source)
    if plan["fingerprint"] != original_plan["fingerprint"] or metadata.get(
        "plan_fingerprint", original_plan["fingerprint"]
    ) != plan["fingerprint"]:
        raise ValueError("Resume plan differs from original run")
    metadata["plan_fingerprint"] = plan["fingerprint"]
    del original_plan
    examples = [example_from_dict(row) for row in plan["payload"]["behavior"]]
    pipeline_path = directory.parent / "pipeline_status.json"
    pipeline = json.loads(pipeline_path.read_text())
    stages = select_checkpoints(args.stages)
    policy = {
        cp.stage: "complete" if (directory / cp.stage / "hardware.json").exists() else "skipped"
        for cp in stages
    }
    amendment = {
        "reason": "User requested continuing accuracy, answer scores, top nodes and interventions while stopping further linear probes.",
        "started_utc": utc_now(),
        "probe_policy_by_stage": policy,
        "activation_extraction": "No new probe activations; completed artifacts retained.",
        "provenance": environment_provenance(ROOT),
        "reused_behavior_sha256": {},
    }
    metadata.setdefault("amendments", []).append(amendment)
    metadata["probe_policy_by_stage"] = policy
    metadata["status"] = "running"
    pipeline.update(
        status="running", phase="full_evaluation_without_probes", probe_policy_by_stage=policy
    )
    write_json(meta_path, metadata)
    write_json(pipeline_path, pipeline)
    torch.manual_seed(args.seed)
    all_timings = []
    try:
        for cp in stages:
            stage = directory / cp.stage
            if policy[cp.stage] == "complete":
                all_timings.extend(json.loads((stage / "timing.json").read_text()))
                continue
            budget.check()
            started = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(cp.repo, revision=cp.revision)
            tokenizer.pad_token = tokenizer.eos_token
            token_hash = tokenizer_hash(tokenizer)
            validate_plan(plan, args, source, token_hash)
            rows = (
                read_jsonl(stage / "behavior.jsonl") if (stage / "behavior.jsonl").exists() else []
            )
            done = reusable_tasks(rows, examples, tokenizer, args)
            if rows:
                amendment["reused_behavior_sha256"][cp.stage] = sha256_file(
                    stage / "behavior.jsonl"
                )
            times = (
                json.loads((stage / "timing.json").read_text())
                if (stage / "timing.json").exists()
                else []
            )
            times = [t for t in times if t["component"] == "behavior" and t["task"] in done]
            if {t["task"] for t in times} != done:
                raise ValueError("Saved behavior lacks matching timing records")
            circuit_times = saved_circuit_times(stage, directory, args.tasks)
            if done == set(args.tasks) and circuit_times is not None:
                checkpoint = json.loads((stage / "checkpoint.json").read_text())
                for key, expected in {**cp.to_dict(), "tokenizer_hash": token_hash}.items():
                    if checkpoint[key] != expected:
                        raise ValueError(f"Reused checkpoint mismatch: {key}")
                times.extend(circuit_times)
                for row in times:
                    row["stage"] = cp.stage
                write_json(stage / "timing.json", times)
                write_json(
                    stage / "hardware.json",
                    {
                        "device": args.device,
                        "gpu": torch.cuda.get_device_name(),
                        "peak_allocated_gb": None,
                        "stage_seconds": sum(t["seconds"] for t in times),
                        "timing_scope": "Saved behavior and circuit phases; interrupted probe work excluded. Peak memory was not recorded before interruption.",
                        "probes": "remaining work skipped by user request",
                    },
                )
                all_timings.extend(times)
                write_json(directory / "timing.json", all_timings)
                write_json(meta_path, metadata)
                print(
                    json.dumps(
                        {"event": "checkpoint_reused", "stage": cp.stage, "probes": "skipped"}
                    ),
                    flush=True,
                )
                del rows
                continue
            model = (
                AutoModelForCausalLM.from_pretrained(
                    cp.repo,
                    revision=cp.revision,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                .to(args.device)
                .eval()
            )
            torch.cuda.reset_peak_memory_stats()
            stage.mkdir(exist_ok=True)
            checkpoint = {
                **cp.to_dict(),
                "config": model.config.to_dict(),
                "tokenizer_hash": token_hash,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
            if (stage / "checkpoint.json").exists():
                old = json.loads((stage / "checkpoint.json").read_text())
                for key in ("stage", "repo", "revision", "tokenizer_hash", "parameter_count"):
                    if old[key] != checkpoint[key]:
                        raise ValueError(f"Resume checkpoint mismatch: {key}")
            else:
                write_json(stage / "checkpoint.json", checkpoint)
            print(
                json.dumps(
                    {"event": "checkpoint_resumed", "stage": cp.stage, "reused_tasks": sorted(done)}
                ),
                flush=True,
            )
            startup = time.monotonic() - started
            for task in sorted(set(args.tasks) - done):
                with tempfile.TemporaryDirectory(prefix=".resume-", dir=stage) as temporary:
                    new_rows, new_times = runner.behavior(
                        model,
                        tokenizer,
                        [ex for ex in examples if ex.task == task],
                        args=args,
                        budget=budget,
                        output=Path(temporary),
                    )
                rows.extend(new_rows)
                rows.sort(key=lambda row: row["task"])
                times.extend(new_times)
                write_jsonl(stage / "behavior.jsonl", rows)
                write_json(stage / "behavior_summary.json", native_aggregate(rows))
                write_json(stage / "timing.json", times)
                print(
                    json.dumps({"event": "behavior_saved", "stage": cp.stage, "task": task}),
                    flush=True,
                )
            del rows
            circuit_times, _ = runner.circuit_maps(
                model,
                tokenizer,
                [],
                args=args,
                budget=budget,
                output=stage,
                tokenizer_hash=token_hash,
                resolved_plan=plan["payload"]["circuits"],
            )
            times.extend(circuit_times + [{"component": "startup", "seconds": startup}])
            for row in times:
                row["stage"] = cp.stage
            write_json(stage / "timing.json", times)
            write_json(
                stage / "hardware.json",
                {
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name(),
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "stage_seconds": time.monotonic() - started,
                    "timing_scope": "Resumed work only; reused behavior timing retained separately.",
                    "probes": "skipped by user request",
                },
            )
            all_timings.extend(times)
            write_json(directory / "timing.json", all_timings)
            write_json(meta_path, metadata)
            print(
                json.dumps({"event": "checkpoint_done", "stage": cp.stage, "probes": "skipped"}),
                flush=True,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        metadata.update(
            status="complete",
            finished_utc=utc_now(),
            elapsed_seconds=(
                datetime.now(timezone.utc) - datetime.fromisoformat(metadata["started_utc"])
            ).total_seconds(),
        )
        amendment["elapsed_seconds"] = time.monotonic() - budget.started
        pipeline.update(
            status="complete", phase="awaiting_local_backup_audit_report", finished_utc=utc_now()
        )
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        pipeline.update(status="failed", error=metadata["error"])
        raise
    finally:
        write_json(meta_path, metadata)
        write_json(pipeline_path, pipeline)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--max-wall-seconds", type=float, required=True)
    p.add_argument("--confirm-cost", action="store_true")
    cli = p.parse_args()
    resume(cli.directory.resolve(), cli.max_wall_seconds, cli.confirm_cost)
