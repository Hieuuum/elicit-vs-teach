"""Run a full-plan pilot, then the full experiment only if its gate passes.

The wall-clock budget covers both subprocesses, including CPU fitting. It ends
the evaluation processes; instance billing/retention is managed separately.
No cloud-account credentials are needed on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from geode.circuits.artifacts import sha256_file, utc_now, write_json
from geode.circuits.checkpoints import CHECKPOINTS


def budget_seconds(max_compute_usd: float, hourly_rate: float) -> float:
    if not all(math.isfinite(v) and v > 0 for v in (max_compute_usd, hourly_rate)):
        raise ValueError("Budget and hourly rate must be finite and positive")
    return max_compute_usd / hourly_rate * 3600


def validate_pilot_gate(path: Path, full_plan_fingerprint: str) -> dict:
    gate = json.loads(path.read_text())
    if gate.get("status") != "passed":
        raise ValueError("Full-plan pilot did not pass; full run will not start")
    if gate.get("full_plan_fingerprint") != full_plan_fingerprint:
        raise ValueError("Pilot gate belongs to a different full plan")
    return gate


def run_bounded(command: list[str], log: Path, timeout_seconds: float) -> None:
    """Terminate the entire evaluation process group if its deadline expires."""
    if timeout_seconds <= 0:
        raise TimeoutError("Evaluation compute budget exhausted")
    with log.open("x") as output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout_seconds)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Evaluation subprocess exited with code {code}; see {log.name}")


def full_command(args: argparse.Namespace, remaining_seconds: float) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "geode.circuits.runner",
        "run",
        "--mode",
        "full",
        "--stages",
        *[cp.stage for cp in CHECKPOINTS],
        "--data",
        str(args.data),
        "--plan",
        str(args.plan),
        "--output",
        str(args.output / "results"),
        "--pairs",
        "512",
        "--probe-groups",
        "128",
        "--interventions",
        "128",
        "--batch-size",
        "8",
        "--max-context",
        "4096",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        "0",
        "--cpu-threads",
        "2",
        "--max-wall-seconds",
        str(max(1.0, remaining_seconds - 5)),
        "--hourly-rate",
        str(args.hourly_rate),
        "--skip-report",
        "--confirm-cost",
    ] + cap_arguments(args)


def cap_arguments(args: argparse.Namespace) -> list[str]:
    result = []
    for field in ("coding_max_new_tokens", "math_max_new_tokens"):
        value = getattr(args, field, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"{field} must be positive")
            result.extend(["--" + field.replace("_", "-"), str(value)])
    return result


def execute(args: argparse.Namespace) -> dict:
    if not args.confirm_cost:
        raise ValueError("GPU execution requires --confirm-cost")
    allowed = budget_seconds(args.max_compute_usd, args.hourly_rate)
    if not math.isfinite(args.pilot_max_seconds) or args.pilot_max_seconds <= 0:
        raise ValueError("Pilot time limit must be finite and positive")
    if args.max_new_tokens < 1:
        raise ValueError("Generation token limit must be positive")
    task_caps = cap_arguments(args)
    plan = json.loads(args.plan.read_text())
    # Both children perform full source/data/tokenizer binding validation.
    # This controller also prevents accidentally passing a pilot-mode plan.
    if plan["binding"]["settings"]["mode"] != "full":
        raise ValueError("Next-stage pipeline requires a full-mode frozen plan")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {
        "status": "running",
        "phase": "full_plan_pilot",
        "started_utc": utc_now(),
        "full_plan_fingerprint": plan["fingerprint"],
        "full_plan_sha256": sha256_file(args.plan),
        "controller_sha256": sha256_file(Path(__file__)),
        "max_compute_usd": args.max_compute_usd,
        "hourly_rate": args.hourly_rate,
        "max_new_tokens": args.max_new_tokens,
        "coding_max_new_tokens": getattr(args, "coding_max_new_tokens", None),
        "math_max_new_tokens": getattr(args, "math_max_new_tokens", None),
        "max_evaluation_seconds": allowed,
        "budget_scope": "Pilot plus full evaluation wall time; instance billing is separate.",
        "instance_action": "No instance deletion or stop is performed by this script.",
    }
    path = args.output / "pipeline_status.json"
    write_json(path, status)
    print(json.dumps(status), flush=True)
    try:
        pilot = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("pilot_full_plan.py")),
            "--plan",
            str(args.plan),
            "--data",
            str(args.data),
            "--output",
            str(args.output / "pilot"),
            "--hourly-rate",
            str(args.hourly_rate),
            "--max-wall-seconds",
            str(min(args.pilot_max_seconds, allowed)),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--confirm-cost",
        ]
        pilot.extend(task_caps)
        run_bounded(pilot, args.output / "pilot.log", min(args.pilot_max_seconds, allowed))
        gate = validate_pilot_gate(args.output / "pilot" / "pilot_status.json", plan["fingerprint"])
        status.update(
            phase="full_evaluation", pilot_status=gate["status"], pilot_finished_utc=utc_now()
        )
        write_json(path, status)
        remaining = allowed - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("No evaluation budget remains after the pilot")
        run_bounded(full_command(args, remaining), args.output / "full.log", remaining)
        run = json.loads((args.output / "results" / "run_metadata.json").read_text())
        if run.get("status") != "complete" or run.get("plan_fingerprint") != plan["fingerprint"]:
            raise ValueError("Full evaluation did not save a complete matching run")
        status.update(status="complete", phase="awaiting_local_backup_audit_report")
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status.update(
            finished_utc=utc_now(),
            elapsed_seconds=time.monotonic() - started,
            estimated_compute_usd=(time.monotonic() - started) / 3600 * args.hourly_rate,
        )
        write_json(path, status)
        print(json.dumps(status), flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--max-compute-usd", type=float, default=25)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--coding-max-new-tokens", type=int)
    parser.add_argument("--math-max-new-tokens", type=int)
    parser.add_argument("--pilot-max-seconds", type=float, default=7200)
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()
    args.plan, args.data, args.output = (p.resolve() for p in (args.plan, args.data, args.output))
    execute(args)
