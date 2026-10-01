"""Measure full-sized RLVR2 probes and small frozen-plan smoke tests before a full run.

The CLI supervisor enforces a hard two-hour maximum, including a stuck BLAS fit.
No scientific-accuracy threshold is used: this is a technical execution gate.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from geode.circuits import runner
from geode.circuits.artifacts import (
    environment_provenance,
    sha256_file,
    utc_now,
    write_json,
)
from geode.circuits.checkpoints import CHECKPOINTS
from geode.circuits.data import verify_data
from geode.circuits.execution import RunBudget
from geode.circuits.plan import example_from_dict, validate_plan
from geode.zoo.activations import tokenizer_hash


def full_args(plan: dict, cli: argparse.Namespace) -> SimpleNamespace:
    """Keep the immutable full-plan binding and production full-run guard intact."""
    args = runner.parser().parse_args(["run"])
    vars(args).update(plan["binding"]["settings"])
    vars(args).update(
        stages=[c.stage for c in CHECKPOINTS],
        device="cuda",
        batch_size=8,
        max_new_tokens=cli.max_new_tokens,
        coding_max_new_tokens=getattr(cli, "coding_max_new_tokens", None),
        math_max_new_tokens=getattr(cli, "math_max_new_tokens", None),
        data=cli.data,
        output=cli.output,
        max_wall_seconds=cli.max_wall_seconds,
        hourly_rate=cli.hourly_rate,
        confirm_cost=cli.confirm_cost,
    )
    runner.validate_config(args)
    if args.mode != "full" or args.probe_groups != 128 or args.max_context != 4096:
        raise ValueError("Pilot requires frozen full mode, 128 target probe groups, context 4096")
    if set(args.tasks) != set(runner.TASKS):
        raise ValueError("Pilot requires all eight tasks")
    if set(plan["payload"]["probes"]) != set(args.tasks):
        raise ValueError("Frozen probe task coverage differs from full settings")
    return args


def smoke_plan(plan: dict) -> tuple[list, dict]:
    """Take one fixed source group per task; preserve every ETHICS control row.

    Math behavior uses one native instance rather than all 50 template variants.
    Circuit groups keep every repeated row of the selected pair. The full plan
    itself is never mutated; attribution/intervention separation is checked over
    the entire source plan before selecting the smoke subset.
    """
    payload = plan["payload"]
    behavioral = []
    for task in sorted({row["task"] for row in payload["behavior"]}):
        pool = sorted((r for r in payload["behavior"] if r["task"] == task), key=lambda r: r["id"])
        chosen = [r for r in pool if r["group"] == pool[0]["group"]]
        behavioral.extend(
            example_from_dict(r) for r in (chosen if task.startswith("ethics_") else chosen[:1])
        )
    circuits = {"tasks": {}, "coverage": {"scope": "one deterministic source-pair group per task"}}
    for task, records in sorted(payload["circuits"]["tasks"].items()):
        source_sets = {
            key: {row[side]["group"] for row in records[key] for side in ("clean", "corrupt")}
            for key in ("pairs", "interventions")
        }
        if source_sets["pairs"] & source_sets["interventions"]:
            raise ValueError(f"Attribution/intervention source leakage: {task}")
        selected = {}
        for key in ("pairs", "interventions"):
            pool = records[key]
            if not pool:
                raise ValueError(f"Missing full-plan smoke coverage: {task}/{key}")
            first = min(pool, key=lambda r: (r["clean"]["id"], r["corrupt"]["id"]))
            pair = (first["clean"]["group"], first["corrupt"]["group"])
            selected[key] = [
                r for r in pool if (r["clean"]["group"], r["corrupt"]["group"]) == pair
            ]
        circuits["tasks"][task] = selected
    return behavioral, circuits


def assert_finite(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite(child)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError("Nonfinite artifact value")


def audit_probe(path: Path, rows: list, *, n_layers: int, hidden_size: int) -> dict:
    """Check saved extraction identity and independently reconstruct split/scaler invariants."""
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        raise ValueError(f"Full probe did not complete: {path.stem}: {report.get('reason')}")
    assert_finite(report)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        features, labels, groups = saved["features"], saved["labels"], saved["groups"]
    if features.shape != (len(rows), n_layers, hidden_size) or not np.isfinite(features).all():
        raise ValueError("Extraction shape or finite-value mismatch")
    if report["item_ids"] != [r[0] for r in rows] or not np.array_equal(
        groups, [r[1] for r in rows]
    ):
        raise ValueError("Frozen probe item/group identity changed")
    if not np.array_equal(labels, [r[2] for r in rows]):
        raise ValueError("Frozen probe labels changed")
    expected_answer = any(row[4] is not None for row in rows)
    if expected_answer != ("answer_only" in report):
        raise ValueError("Candidate answer-only control missing or unexpected")
    canonical_split = None
    for kind, section in [("main", report)] + (
        [("answer_only", report["answer_only"])] if expected_answer else []
    ):
        if len(section["layers"]) != n_layers:
            raise ValueError("Missing probe layers")
        for index, layer in enumerate(section["layers"]):
            split = layer["split_indices"]
            if set(split) != {"train", "validation", "test"}:
                raise ValueError("Missing probe partition")
            flat = [i for part in split.values() for i in part]
            if sorted(flat) != list(range(len(rows))):
                raise ValueError("Probe split rows overlap or omit examples")
            sets = [set(groups[split[key]]) for key in ("train", "validation", "test")]
            if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("Probe source-group split leakage")
            if canonical_split is None:
                canonical_split = split
            if split != canonical_split:
                raise ValueError("Probe layers/control use different heldout splits")
            if len(layer["shuffled_test_accuracies"]) != 5:
                raise ValueError("Full pilot must perform five shuffled fits")
            for key in ("test_accuracy", "validation_accuracy", "majority_test_accuracy"):
                if not 0 <= layer[key] <= 1:
                    raise ValueError("Invalid probe accuracy")
            if kind == "main":
                train = np.asarray(features[split["train"], index, :], dtype=np.float64)
                mean, scale = train.mean(axis=0), train.std(axis=0)
                scale[scale <= np.finfo(np.float64).eps] = 1.0
                if not np.allclose(layer["training_mean"], mean, rtol=1e-6, atol=1e-7):
                    raise ValueError("Probe scaler mean is not train-only")
                if not np.allclose(layer["training_scale"], scale, rtol=1e-6, atol=1e-7):
                    raise ValueError("Probe scaler scale is not train-only")
    return {
        "status": "passed",
        "n_rows": len(rows),
        "n_groups": len(set(groups)),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_shuffles": 5,
        "answer_only": expected_answer,
        "split_n_groups": report["layers"][0]["split_n_groups"],
        "split_n_examples": report["layers"][0]["split_n_examples"],
        "feature_artifact_sha256": sha256_file(path.with_suffix(".npz")),
        "report_sha256": sha256_file(path),
    }


def memory(device: str) -> dict:
    result = {
        "process_lifetime_peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024
        / 1e9
    }
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        result.update(
            gpu_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            gpu_peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        )
    return result


@contextmanager
def instrument_probes(metrics: dict, device: str):
    """Time the actual production extraction and layerwise fitting calls unchanged."""
    originals = {
        name: getattr(runner, name)
        for name in ("extract_residual_features", "evaluate_layerwise_probes")
    }

    def wrapped(name: str):
        def call(*args, **kwargs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.monotonic()
            result = originals[name](*args, **kwargs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            key = "extraction" if name == "extract_residual_features" else "fitting"
            metrics[key + "_seconds"] += time.monotonic() - started
            metrics[key + "_calls"] += 1
            if key == "extraction" and not torch.isfinite(result).all():
                raise ValueError("Nonfinite extracted activation")
            return result

        return call

    try:
        for name in originals:
            setattr(runner, name, wrapped(name))
        yield
    finally:
        for name, original in originals.items():
            setattr(runner, name, original)


def load_rlvr2() -> tuple:
    """Pinned cached model only; pilot cannot silently download new assets."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    checkpoint = next(c for c in CHECKPOINTS if c.stage == "rlvr2")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repo, revision=checkpoint.revision, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint.repo,
            revision=checkpoint.revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    if any(p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("Loaded model is not entirely BF16 CUDA")
    return model, tokenizer


def execute(plan: dict, args: SimpleNamespace, output: Path, *, loader=load_rlvr2) -> dict:
    """Execute and audit; loader injection is solely for tiny offline tests."""
    budget = RunBudget(args.max_wall_seconds, args.hourly_rate, args.confirm_cost)
    stage = output / "rlvr2"
    stage.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running",
        "started_utc": utc_now(),
        "full_plan_fingerprint": plan["fingerprint"],
        "plan_binding": plan["binding"],
        "checkpoint": next(c.to_dict() for c in CHECKPOINTS if c.stage == "rlvr2"),
        "gate": "technical completion, finite values, exact plan extraction and disjoint probe splits; no accuracy threshold",
        "probe_audits": {},
        "phase_timings": [],
    }
    write_json(output / "pilot_status.json", status)
    timings = []
    try:
        examples, smoke = smoke_plan(plan)
        write_json(
            output / "smoke_plan.json",
            {"behavior_ids": [e.id for e in examples], "circuits": smoke},
        )
        started = time.monotonic()
        torch.manual_seed(args.seed)
        model, tokenizer = loader()
        if tokenizer_hash(tokenizer) != plan["binding"]["tokenizer_hash"]:
            raise ValueError("Loaded tokenizer differs from frozen plan")
        status["phase_timings"].append(
            {"component": "startup", "seconds": time.monotonic() - started, **memory(args.device)}
        )
        for task, task_plan in sorted(plan["payload"]["probes"].items()):
            budget.check()
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            metrics = {
                "component": "probes",
                "task": task,
                "extraction_seconds": 0.0,
                "fitting_seconds": 0.0,
                "extraction_calls": 0,
                "fitting_calls": 0,
            }
            started = time.monotonic()
            with instrument_probes(metrics, args.device):
                measured = runner.probe_features(
                    model,
                    tokenizer,
                    [],
                    args=args,
                    budget=budget,
                    output=stage,
                    resolved_plan={task: task_plan},
                )
            expected_extractions = sum(1 + (r[4] is not None) for r in task_plan["rows"])
            if metrics["extraction_calls"] != expected_extractions:
                raise ValueError("Not every frozen full probe row was extracted")
            metrics.update(seconds=time.monotonic() - started, **memory(args.device))
            status["phase_timings"].append(metrics)
            audit_start = time.monotonic()
            status["probe_audits"][task] = audit_probe(
                stage / "probes" / f"{task}.json",
                task_plan["rows"],
                n_layers=model.config.num_hidden_layers + 1,
                hidden_size=model.config.hidden_size,
            )
            status["probe_audits"][task]["audit_seconds"] = time.monotonic() - audit_start
            timings.extend(measured)
            write_json(output / "pilot_status.json", status)
            write_json(stage / "timing.json", timings)
            print(json.dumps({"event": "full_probe_audited", **metrics}), flush=True)
        for component in ("behavior", "circuits"):
            budget.check()
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            if component == "behavior":
                rows, measured = runner.behavior(
                    model, tokenizer, examples, args=args, budget=budget, output=stage
                )
                assert_finite(rows)
                if any(r["context_overflow"] or r["answer_log_prob_nats"] is None for r in rows):
                    raise ValueError("Behavior smoke context/scoring failure")
                status["behavior_smoke"] = {
                    "n_rows": len(rows),
                    "n_truncated": sum(r["truncated"] for r in rows),
                    "n_correct": sum(r["correct"] for r in rows),
                }
            else:
                measured, _ = runner.circuit_maps(
                    model,
                    tokenizer,
                    [],
                    args=args,
                    budget=budget,
                    output=stage,
                    tokenizer_hash=plan["binding"]["tokenizer_hash"],
                    resolved_plan=smoke,
                )
                for task in smoke["tasks"]:
                    with np.load(stage / "circuits" / f"{task}.npz", allow_pickle=False) as saved:
                        if not np.isfinite(saved["scores"]).all():
                            raise ValueError("Nonfinite circuit scores")
                    effects = json.loads(
                        (stage / "circuits" / f"{task}_interventions.json").read_text()
                    )
                    assert_finite(effects)
                    if not effects or any(
                        not {"top", "random", "random_type_matched"}.issubset(row)
                        for row in effects
                    ):
                        raise ValueError("Missing intervention controls")
            timings.extend(measured)
            status["phase_timings"].append(
                {
                    "component": component,
                    "seconds": time.monotonic() - started,
                    **memory(args.device),
                }
            )
            write_json(stage / "timing.json", timings)
            write_json(output / "pilot_status.json", status)
        budget.check()
        status.update(
            status="passed",
            finished_utc=utc_now(),
            elapsed_seconds=time.monotonic() - budget.started,
        )
    except BaseException as exc:
        status.update(status="failed", finished_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "pilot_status.json", status)
    return status


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hourly-rate", type=float, required=True)
    p.add_argument("--max-wall-seconds", type=float, default=7200)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Behavior smoke generation cap; must match the approved full-run protocol",
    )
    p.add_argument("--confirm-cost", action="store_true")
    p.add_argument("--coding-max-new-tokens", type=int)
    p.add_argument("--math-max-new-tokens", type=int)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> None:
    cli = parser().parse_args()
    RunBudget(cli.max_wall_seconds, cli.hourly_rate, cli.confirm_cost)
    if cli.max_wall_seconds > 7200:
        raise ValueError("Full-plan pilot maximum is two hours")
    output = Path(cli.output)
    if cli.worker:
        plan = json.loads(Path(cli.plan).read_text())
        args = full_args(plan, cli)
        source = verify_data(cli.data)
        validate_plan(plan, args, source)
        write_json(
            output / "provenance.json",
            {
                "plan_sha256": sha256_file(Path(cli.plan)),
                "dataset_manifest": source,
                "environment": environment_provenance(ROOT),
                "configuration": vars(args),
                "maximum_compute_usd": cli.max_wall_seconds * cli.hourly_rate / 3600,
            },
        )
        execute(plan, args, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "pilot_status.json", {"status": "running", "started_utc": utc_now()})
    worker = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        start_new_session=False,
    )
    try:
        code = worker.wait(timeout=cli.max_wall_seconds)
        if code:
            raise RuntimeError(f"Pilot worker exited {code}")
        if json.loads((output / "pilot_status.json").read_text()).get("status") != "passed":
            raise RuntimeError("Worker exited without a passing technical audit")
    except BaseException as exc:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        status = json.loads((output / "pilot_status.json").read_text())
        status.update(
            status="failed", finished_utc=utc_now(), supervisor_error=f"{type(exc).__name__}: {exc}"
        )
        write_json(output / "pilot_status.json", status)
        raise


if __name__ == "__main__":
    main()
