"""Offline execution-gate tests: identity, source leakage and genuine tiny fitting."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geode.circuits import plan, runner
from geode.circuits.artifacts import fingerprint, write_json
from geode.circuits.data import parse_ethics
from geode.circuits.probes import evaluate_layerwise_probes
from geode.zoo.activations import tokenizer_hash

from .test_runner_integration import args as args, model as model, tokenizer as tokenizer

SCRIPT = (
    Path(__file__).resolve().parents[3] / "experiments/olmo2-circuit-overlap/pilot_full_plan.py"
)
spec = importlib.util.spec_from_file_location("full_plan_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


@pytest.fixture
def probe_artifact(tmp_path):
    rng = np.random.default_rng(6)
    features = rng.normal(size=(24, 2, 4)).astype(np.float32)
    features[:, :, 3] = 1  # Train-only constant feature has unit scale.
    labels = np.tile([0, 1], 12)
    groups = np.repeat([f"g{i}" for i in range(12)], 2)
    rows = [
        [f"row{i}", str(g), int(y), [1, 2], None, None]
        for i, (g, y) in enumerate(zip(groups, labels))
    ]
    report = evaluate_layerwise_probes(features, labels, groups, seed=0, n_shuffles=5)
    report.update(status="complete", item_ids=[r[0] for r in rows])
    path = tmp_path / "probe.json"
    write_json(path, report)
    np.savez_compressed(path.with_suffix(".npz"), features=features, labels=labels, groups=groups)
    return path, rows, report


def test_audit_checks_actual_train_only_moments_and_full_shuffle_count(probe_artifact):
    path, rows, _ = probe_artifact
    result = pilot.audit_probe(path, rows, n_layers=2, hidden_size=4)
    assert result["n_rows"] == 24
    assert result["n_groups"] == 12
    assert result["n_shuffles"] == 5


@pytest.mark.parametrize(
    "defect", ["scaler", "split", "shuffles", "identity", "candidate", "finite"]
)
def test_gate_rejects_silent_probe_corruption(probe_artifact, defect):
    path, rows, report = probe_artifact
    if defect == "scaler":
        report["layers"][0]["training_mean"][0] += 1
    elif defect == "split":
        report["layers"][0]["split_indices"]["test"][0] = report["layers"][0]["split_indices"][
            "train"
        ][0]
    elif defect == "shuffles":
        report["layers"][0]["shuffled_test_accuracies"].pop()
    elif defect == "identity":
        rows[0][0] = "different"
    elif defect == "candidate":
        rows[0][4] = [2]
    else:
        report["layers"][0]["test_accuracy"] = float("nan")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        pilot.audit_probe(path, rows, n_layers=2, hidden_size=4)


@pytest.fixture
def tiny_plan(args, tokenizer):
    examples = parse_ethics(
        "commonsense", [{"label": str(i % 2), "input": f"w{i % 5} w0 w1"} for i in range(24)]
    )
    train = parse_ethics(
        "commonsense",
        [{"label": str(i % 2), "input": f"w{i % 5} w0 w1"} for i in range(24)],
        split="train",
    )
    args.probe_groups = 12
    frozen = plan.build_plan(
        args,
        tokenizer,
        {"fixture": True},
        tokenizer_hash(tokenizer),
        pools=(examples, examples, train),
    )
    args.mode = "full"  # Production probe_features must execute all five controls.
    return frozen


def test_smoke_selection_preserves_factorial_and_rejects_hidden_source_leakage(tiny_plan):
    before = fingerprint(tiny_plan)
    _, smoke = pilot.smoke_plan(tiny_plan)
    task = smoke["tasks"]["ethics_commonsense"]
    assert len(task["pairs"]) == len(task["interventions"]) == 8
    assert fingerprint(tiny_plan) == before
    leaked = deepcopy(tiny_plan)
    records = leaked["payload"]["circuits"]["tasks"]["ethics_commonsense"]
    # Leak outside the selected first pair must also fail.
    records["interventions"][-1]["clean"]["group"] = records["pairs"][-1]["clean"]["group"]
    with pytest.raises(ValueError, match="source leakage"):
        pilot.smoke_plan(leaked)


def test_full_settings_keep_all_seven_stage_guard(args):
    args.mode = "full"
    args.tasks = list(runner.TASKS)
    args.pairs, args.interventions, args.probe_groups, args.max_context = 512, 128, 128, 4096
    frozen = {
        "binding": {"settings": {key: getattr(args, key) for key in plan.PLAN_SETTINGS}},
        "payload": {"probes": dict.fromkeys(runner.TASKS)},
    }
    cli = SimpleNamespace(
        data="unused",
        output="unused",
        max_wall_seconds=7200,
        max_new_tokens=2048,
        coding_max_new_tokens=192,
        math_max_new_tokens=384,
        hourly_rate=0.61,
        confirm_cost=True,
    )
    resolved = pilot.full_args(frozen, cli)
    assert len(resolved.stages) == 7
    assert resolved.mode == "full" and resolved.max_new_tokens == 2048
    assert resolved.coding_max_new_tokens == 192 and resolved.math_max_new_tokens == 384
    frozen["binding"]["settings"]["mode"] = "pilot"
    with pytest.raises(ValueError, match="frozen full"):
        pilot.full_args(frozen, cli)


def test_tiny_model_executes_real_full_fits_and_both_smoke_paths(
    tiny_plan, args, tokenizer, model, tmp_path
):
    original = runner.extract_residual_features
    output = tmp_path / "full-size-pilot-fixture"
    result = pilot.execute(tiny_plan, args, output, loader=lambda: (model, tokenizer))
    assert result["status"] == "passed"
    assert result["full_plan_fingerprint"] == tiny_plan["fingerprint"]
    audit = result["probe_audits"]["ethics_commonsense"]
    assert audit["n_rows"] == len(tiny_plan["payload"]["probes"]["ethics_commonsense"]["rows"])
    assert audit["n_shuffles"] == 5
    phases = result["phase_timings"]
    probe = next(row for row in phases if row["component"] == "probes")
    assert probe["extraction_calls"] == audit["n_rows"]
    assert probe["fitting_calls"] == 1
    assert probe["fitting_seconds"] > 0 and probe["extraction_seconds"] > 0
    assert {row["component"] for row in phases} == {"startup", "probes", "behavior", "circuits"}
    assert runner.extract_residual_features is original
    assert json.loads((output / "pilot_status.json").read_text())["status"] == "passed"


def test_loader_failure_persists_failed_gate(tiny_plan, args, tmp_path):
    def failed_loader():
        raise RuntimeError("fixture OOM")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture OOM"):
        pilot.execute(tiny_plan, args, output, loader=failed_loader)
    assert json.loads((output / "pilot_status.json").read_text())["status"] == "failed"


def test_controller_selected_generation_cap_reaches_behavior_without_changing_plan(args):
    args.mode = "full"
    args.tasks = list(runner.TASKS)
    args.pairs, args.interventions, args.probe_groups, args.max_context = 512, 128, 128, 4096
    frozen = {
        "binding": {"settings": {key: getattr(args, key) for key in plan.PLAN_SETTINGS}},
        "payload": {"probes": dict.fromkeys(runner.TASKS)},
    }
    before = fingerprint(frozen)
    cli = SimpleNamespace(
        data="unused",
        output="unused",
        max_wall_seconds=7200,
        max_new_tokens=128,
        hourly_rate=0.61,
        confirm_cost=True,
    )
    assert pilot.full_args(frozen, cli).max_new_tokens == 128
    assert fingerprint(frozen) == before


def test_supervisor_timeout_fails_closed_and_keeps_controller_process_group(monkeypatch, tmp_path):
    output = tmp_path / "supervised"
    cli = SimpleNamespace(
        plan="unused",
        data="unused",
        output=str(output),
        hourly_rate=0.61,
        max_wall_seconds=1,
        max_new_tokens=128,
        confirm_cost=True,
        worker=False,
    )
    monkeypatch.setattr(pilot, "parser", lambda: SimpleNamespace(parse_args=lambda: cli))
    calls = []

    class Worker:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                raise pilot.subprocess.TimeoutExpired("fixture", 1)
            return -15

        def poll(self):
            return None

        def terminate(self):
            calls.append(("terminate",))

    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is False
        assert command[-1] == "--worker"
        return Worker()

    monkeypatch.setattr(pilot.subprocess, "Popen", launch)
    with pytest.raises(pilot.subprocess.TimeoutExpired):
        pilot.main()
    result = json.loads((output / "pilot_status.json").read_text())
    assert result["status"] == "failed"
    assert "TimeoutExpired" in result["supervisor_error"]
    assert ("terminate",) in calls
