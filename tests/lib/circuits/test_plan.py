"""Frozen plans must preserve sample semantics and never repeat CPU matching on GPU runs."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from geode.circuits import plan, runner
from geode.circuits.data import parse_ethics, parse_gsm
from geode.zoo.activations import tokenizer_hash

from .test_runner_integration import args as args
from .test_runner_integration import model as model
from .test_runner_integration import tokenizer as tokenizer


@pytest.fixture
def pools():
    ethics = parse_ethics(
        "commonsense", [{"label": str(i % 2), "input": f"w{i % 5} w0 w1"} for i in range(24)]
    )
    train = parse_ethics(
        "commonsense",
        [{"label": str(i % 2), "input": f"w{i % 5} w0 w1"} for i in range(24)],
        split="train",
    )
    math = parse_gsm(
        [
            {"id": i, "instance": 0, "question": "How many?", "answer": f"reason #### {i + 10}"}
            for i in range(24)
        ]
    )
    math = [replace(example, prompt="w0 w1 Answer:\n") for example in math]
    return ethics[:2] + math[:2], ethics + math, train + math


def make_plan(args, tokenizer, pools):
    return plan.build_plan(
        args, tokenizer, {"sources": "fixture"}, tokenizer_hash(tokenizer), pools=pools
    )


def test_plan_roundtrip_preserves_pairs_encodings_and_source_disjoint_interventions(
    args, tokenizer, pools, tmp_path
):
    frozen = make_plan(args, tokenizer, pools)
    path = tmp_path / "plan.json"
    plan.save_plan(path, frozen)
    restored = plan.load_plan(path, args, {"sources": "fixture"}, tokenizer_hash(tokenizer))
    assert restored["fingerprint"] == frozen["fingerprint"]
    for task in restored["payload"]["circuits"]["tasks"].values():
        sources = set()
        for record in task["pairs"]:
            a, b, ids = plan.decode_pair(record)
            assert json.loads(json.dumps(ids)) == json.loads(
                json.dumps(runner._pair_inputs(a, b, tokenizer))
            )
            sources.update((a.group, b.group))
        for record in task["interventions"]:
            a, b, _ = plan.decode_pair(record)
            assert a.group not in sources and b.group not in sources
    assert [plan.example_from_dict(e) for e in restored["payload"]["behavior"]] == pools[0]


def test_plan_keeps_complete_ethics_factorials_and_reciprocal_frequency_balance(
    args, tokenizer, pools
):
    frozen = make_plan(args, tokenizer, pools)
    pairs = frozen["payload"]["circuits"]["tasks"]["ethics_commonsense"]["pairs"]
    clustered = defaultdict(list)
    for row in pairs:
        a, b, _ = plan.decode_pair(row)
        clustered[(a.group, b.group)].append((a, b))
    assert all(len(rows) == 8 for rows in clustered.values())
    probe_rows = frozen["payload"]["probes"]["gsm_symbolic"]["rows"]
    clusters = defaultdict(list)
    for row in probe_rows:
        clusters[row[1]].append(row)
    assert clusters
    for rows in clusters.values():
        assert len(rows) == 4
        assert Counter(row[5] for row in rows if row[2] == 1) == Counter(
            row[5] for row in rows if row[2] == 0
        )


def test_rebuilding_same_plan_is_deterministic_and_checkpoint_independent(args, tokenizer, pools):
    first = make_plan(args, tokenizer, pools)
    args.stages = ["init", "stage1", "stage2", "sft", "dpo", "rlvr1", "rlvr2"]
    args.device = "cuda"
    args.hourly_rate = 99.0
    args.output = "/tmp/a-different-output"
    second = make_plan(args, tokenizer, pools)
    assert first["fingerprint"] == second["fingerprint"]
    plan.validate_plan(first, args, {"sources": "fixture"}, tokenizer_hash(tokenizer))


@pytest.mark.parametrize(
    "field", ["seed", "pairs", "probe_groups", "max_context", "groups", "interventions"]
)
def test_changed_sampling_settings_cannot_silently_reuse_plan(args, tokenizer, pools, field):
    frozen = make_plan(args, tokenizer, pools)
    setattr(args, field, getattr(args, field) + 1)
    with pytest.raises(ValueError, match="settings"):
        plan.validate_plan(frozen, args, {"sources": "fixture"}, tokenizer_hash(tokenizer))


def test_changed_dataset_tokenizer_and_planning_source_are_rejected(
    args, tokenizer, pools, monkeypatch
):
    frozen = make_plan(args, tokenizer, pools)
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        plan.validate_plan(frozen, args, {"sources": "changed"}, tokenizer_hash(tokenizer))
    with pytest.raises(ValueError, match="tokenizer_hash"):
        plan.validate_plan(frozen, args, {"sources": "fixture"}, "different tokenizer")
    monkeypatch.setattr(plan, "planning_source_fingerprint", lambda: "changed source")
    with pytest.raises(ValueError, match="source_fingerprint"):
        plan.validate_plan(frozen, args, {"sources": "fixture"}, tokenizer_hash(tokenizer))


def test_tampered_token_ids_and_group_labels_fail_integrity_check(args, tokenizer, pools):
    frozen = make_plan(args, tokenizer, pools)
    corrupt = deepcopy(frozen)
    corrupt["payload"]["circuits"]["tasks"]["ethics_commonsense"]["pairs"][0]["inputs"][0][0] += 1
    with pytest.raises(ValueError, match="content fingerprint"):
        plan.validate_plan(corrupt, args, {"sources": "fixture"})
    corrupt = deepcopy(frozen)
    corrupt["payload"]["behavior"][0]["group"] = "changed group"
    with pytest.raises(ValueError, match="content fingerprint"):
        plan.validate_plan(corrupt, args, {"sources": "fixture"})


def test_plan_path_cannot_be_overwritten_by_a_different_sample(args, tokenizer, pools, tmp_path):
    first = make_plan(args, tokenizer, pools)
    path = tmp_path / "frozen.json"
    plan.save_plan(path, first)
    plan.save_plan(path, first)
    args.seed += 1
    with pytest.raises(FileExistsError, match="overwrite"):
        plan.save_plan(path, make_plan(args, tokenizer, pools))
    assert json.loads(path.read_text())["fingerprint"] == first["fingerprint"]


def test_two_checkpoint_run_consumes_cached_pairs_and_probes_without_matching_or_parsing(
    args,
    model,
    tokenizer,
    monkeypatch,
    tmp_path,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ethics = parse_ethics(
        "commonsense", [{"label": str(i % 2), "input": "w0 w1"} for i in range(12)]
    )
    train = parse_ethics(
        "commonsense", [{"label": str(i % 2), "input": "w0 w1"} for i in range(12)], split="train"
    )
    args.tasks = ["ethics_commonsense"]
    args.stages = ["stage2", "rlvr2"]
    frozen = make_plan(args, tokenizer, (ethics[:2], ethics, train))
    args.plan = str(tmp_path / "frozen.json")
    plan.save_plan(args.plan, frozen)

    def no_matching(*_args, **_kwargs):
        pytest.fail("cached execution must never recompute matching or diagnostic token IDs")

    monkeypatch.setattr(plan, "resolve_pools", no_matching)
    monkeypatch.setattr(plan, "circuit_plan", no_matching)
    monkeypatch.setattr(plan, "probe_plan", no_matching)
    monkeypatch.setattr(runner, "controlled_pairs", no_matching)
    monkeypatch.setattr(runner, "_pair_inputs", no_matching)
    monkeypatch.setattr(runner, "verify_data", lambda _: {"sources": "fixture"})
    monkeypatch.setattr(runner, "environment_provenance", lambda _: {"fixture": True})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_k: deepcopy(tokenizer))
    loaded = []

    def load_model(*_args, **_kwargs):
        loaded.append(True)
        return deepcopy(model)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    from geode.circuits import report

    args.skip_report = True
    monkeypatch.setattr(
        report,
        "generate_report",
        lambda *_a, **_k: pytest.fail("skip-report must defer CPU reporting"),
    )
    result = runner.run(args)
    assert len(loaded) == 2
    assert result["status"] == "complete"
    assert result["plan_fingerprint"] == frozen["fingerprint"]


def test_wrong_plan_tokenizer_is_rejected_before_model_weights_load(
    args, tokenizer, pools, monkeypatch, tmp_path
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frozen = make_plan(args, tokenizer, pools)
    args.plan = str(tmp_path / "frozen.json")
    plan.save_plan(args.plan, frozen)
    changed = deepcopy(tokenizer)
    changed.add_tokens(["new vocabulary item"])
    monkeypatch.setattr(runner, "verify_data", lambda _: {"sources": "fixture"})
    monkeypatch.setattr(runner, "environment_provenance", lambda _: {"fixture": True})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_k: changed)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *_a, **_k: pytest.fail("weights must not load"),
    )
    with pytest.raises(ValueError, match="tokenizer_hash"):
        runner.run(args)


def test_prepare_plan_cli_downloads_only_tokenizer_and_does_not_require_cost_confirmation(
    args, tokenizer, pools, monkeypatch, tmp_path
):
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = tmp_path / "prepared.json"
    monkeypatch.setattr(sys, "argv", ["geode", "prepare-plan", "--plan", str(path)])
    monkeypatch.setattr(plan, "resolve_pools", lambda _: pools)
    monkeypatch.setattr(runner, "verify_data", lambda _: {"sources": "fixture"})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_k: deepcopy(tokenizer))
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *_a, **_k: pytest.fail("CPU preparation must never load weights"),
    )
    runner.main()
    saved = json.loads(path.read_text())
    assert saved["fingerprint"]
    monkeypatch.setattr(
        plan, "build_plan", lambda *_a, **_k: pytest.fail("valid plan must be reused")
    )
    runner.main()
