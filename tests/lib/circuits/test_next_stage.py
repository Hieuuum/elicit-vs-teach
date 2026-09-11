"""The unattended controller never advances through an invalid pilot gate."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.fixture
def controller():
    path = Path(__file__).resolve().parents[3] / "experiments/olmo2-circuit-overlap/next_stage.py"
    spec = importlib.util.spec_from_file_location("next_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "state,fingerprint", [("failed", "expected"), ("running", "expected"), ("passed", "other")]
)
def test_failed_or_unmatched_pilot_blocks_full_run(controller, tmp_path, state, fingerprint):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"status": state, "full_plan_fingerprint": fingerprint}))
    with pytest.raises(ValueError):
        controller.validate_pilot_gate(gate, "expected")


def test_passed_matching_pilot_and_explicit_generation_budget(controller, tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"status": "passed", "full_plan_fingerprint": "expected"}))
    assert controller.validate_pilot_gate(gate, "expected")["status"] == "passed"
    args = argparse.Namespace(
        data=tmp_path, plan=gate, output=tmp_path, hourly_rate=0.6, max_new_tokens=256
    )
    command = controller.full_command(args, 300)
    assert command[command.index("--max-new-tokens") + 1] == "256"
    assert command[command.index("--max-wall-seconds") + 1] == "295"
    stages = command[command.index("--stages") + 1 : command.index("--data")]
    assert stages == ["init", "stage1", "stage2", "sft", "dpo", "rlvr1", "rlvr2"]
    assert "--confirm-cost" in command and "--skip-report" in command
    assert controller.budget_seconds(25, 0.5) == 180000


@pytest.mark.parametrize(
    "budget,rate", [(0, 0.6), (25, 0), (float("inf"), 0.6), (25, float("nan"))]
)
def test_invalid_budget_rejected(controller, budget, rate):
    with pytest.raises(ValueError):
        controller.budget_seconds(budget, rate)


def test_task_caps_propagate_to_full_command_and_reject_invalid(controller, tmp_path):
    args = argparse.Namespace(
        data=tmp_path,
        plan=tmp_path / "plan",
        output=tmp_path,
        hourly_rate=0.6,
        max_new_tokens=384,
        coding_max_new_tokens=192,
        math_max_new_tokens=384,
    )
    command = controller.full_command(args, 300)
    assert command[command.index("--coding-max-new-tokens") + 1] == "192"
    assert command[command.index("--math-max-new-tokens") + 1] == "384"
    args.coding_max_new_tokens = 0
    with pytest.raises(ValueError):
        controller.full_command(args, 300)


def test_deadline_ends_child_process(controller, tmp_path):
    marker = tmp_path / "should_not_exist"
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys,time;time.sleep(.5);pathlib.Path(sys.argv[1]).touch()",
        str(marker),
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        controller.run_bounded(command, tmp_path / "timeout.log", 0.05)
    time.sleep(0.6)
    assert not marker.exists()


def test_subprocess_failure_is_loud_and_logs_are_never_overwritten(controller, tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="code 3"):
        controller.run_bounded([sys.executable, "-c", "raise SystemExit(3)"], log, 5)
    with pytest.raises(FileExistsError):
        controller.run_bounded([sys.executable, "-c", "pass"], log, 5)
