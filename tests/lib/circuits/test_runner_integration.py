"""Integration properties for the pilot runner; no model or dataset downloads."""

from dataclasses import asdict
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

from geode.circuits import plan, runner
from geode.circuits.artifacts import read_jsonl, validate_matched_rows
from geode.circuits.data import Example, grade, parse_ethics
from geode.circuits.execution import RunBudget


@pytest.fixture
def tokenizer():
    tokens = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "0",
        "1",
        "A",
        "B",
        "C",
        "D",
        ":",
        "2",
        "3",
        "4",
        "5",
        "6",
        "Answer",
        "w0",
        "w1",
        "w2",
        "w3",
        "w4",
    ]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )


@pytest.fixture
def model(tokenizer):
    with torch.random.fork_rng():
        torch.manual_seed(14)
        config = Olmo2Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            pad_token_id=0,
            eos_token_id=1,
        )
        config._attn_implementation = "eager"
        return Olmo2ForCausalLM(config).eval()


@pytest.fixture
def args(tmp_path):
    args = runner.parser().parse_args(["run", "--device", "cpu", "--confirm-cost"])
    args.output = str(tmp_path / "run")
    args.data = str(tmp_path / "unused-data")
    args.max_context = 512
    args.max_new_tokens = 2
    args.batch_size = 4
    args.pairs = 2
    args.interventions = 1
    args.probe_groups = 8
    return args


def commonsense(n=8):
    return parse_ethics(
        "commonsense", [{"label": str(i % 2), "input": f"w{i % 5} w0 w1"} for i in range(n)]
    )


def math_examples(n=8):
    return [
        Example(
            id=f"math:{i}",
            task="gsm_symbolic",
            prompt=f"w{i % 5} w1 Answer:",
            answer=f"gold reasoning #### {i % 2 + 2}",
            group=f"math:{i}",
            metadata={"split": "test", "variant": "original", "group_size": 1},
        )
        for i in range(n)
    ]


def test_ethics_behavior_retains_original_and_all_controls_with_semantic_grades(
    model, tokenizer, args, tmp_path
):
    base = commonsense(2)
    budget = RunBudget(120, 0, True)
    rows, timing = runner.behavior(
        model, tokenizer, base, args=args, budget=budget, output=tmp_path
    )
    assert len(rows) == 18
    assert len(timing) == 1
    assert read_jsonl(tmp_path / "behavior.jsonl") == json.loads(json.dumps(rows))
    originals = [row for row in rows if row["metadata"]["variant"] == "original"]
    assert [row["prompt"] for row in originals] == [ex.prompt for ex in base]
    assert [row["answer"] for row in originals] == [ex.answer for ex in base]
    for ex in base:
        variants = [row for row in rows if row["group"] == ex.group]
        assert len(variants) == 9
        assert len({row["metadata"]["variant"] for row in variants}) == 9
        assert {row["label"] for row in variants} == {ex.label}
        for row in variants:
            rendered = Example(**{key: row[key] for key in asdict(ex)})
            assert row["correct"] == grade(rendered, row["generation"])["correct"]
            assert np.isfinite(row["logit_margin"])
            assert row["output_tokens"] == 1


def test_behavior_produces_all_eight_tasks_and_preserves_native_group_sizes(
    model, tokenizer, args, tmp_path, monkeypatch
):
    args.coding_max_new_tokens = 3
    args.math_max_new_tokens = 4
    observed_caps = []
    generate = runner.generate_answers

    def capture_generation(*positional, **kwargs):
        observed_caps.append(kwargs["max_new_tokens"])
        return generate(*positional, **kwargs)

    monkeypatch.setattr(runner, "generate_answers", capture_generation)
    examples = commonsense(2)
    examples += parse_ethics(
        "justice", [{"label": str(i % 2), "scenario": "w0 w1"} for i in range(4)]
    )
    examples += parse_ethics(
        "deontology", [{"label": str(i % 2), "scenario": "w0", "excuse": "w1"} for i in range(4)]
    )
    examples += parse_ethics(
        "virtue", [{"label": str(i % 2), "scenario": "w0 [SEP] w1"} for i in range(5)]
    )
    examples += parse_ethics("utilitarianism", [{"better": "w0", "worse": "w1"}])
    examples += math_examples(1)
    for direction in ("input", "output"):
        examples.append(
            Example(
                id=f"crux:{direction}:0",
                task=f"cruxeval_{direction}",
                prompt="w0 w1 Answer:",
                answer="assert f(2) == 3\n[/ANSWER]",
                group="function:0",
                metadata={
                    "split": "test",
                    "variant": "original",
                    "group_size": 1,
                    "direction": direction,
                    "source": {"code": "def f(x):\n    return x + 1", "input": "2", "output": "3"},
                },
            )
        )

    def local_grade(ex, generated):
        if ex.task.startswith("cruxeval_"):
            return {"correct": generated == ex.answer, "parse_failure": False}
        return grade(ex, generated)

    monkeypatch.setattr(runner, "grade", local_grade)
    rows, timings = runner.behavior(
        model, tokenizer, examples, args=args, budget=RunBudget(120, 0, True), output=tmp_path
    )
    assert {row["task"] for row in rows} == set(runner.TASKS)
    assert len(timings) == 8
    assert sorted(observed_caps) == [3, 3, 4]
    for timing in timings:
        expected = (
            1
            if timing["task"].startswith("ethics_")
            else (4 if timing["task"] == "gsm_symbolic" else 3)
        )
        assert timing["max_new_tokens"] == expected
    for row in rows:
        if not row["task"].startswith("ethics_"):
            assert row["max_new_tokens"] == (4 if row["task"] == "gsm_symbolic" else 3)
    assert len(rows) == 9 * (2 + 4 + 4 + 5 + 1) + 3
    for task, size in [("ethics_justice", 4), ("ethics_deontology", 4), ("ethics_virtue", 5)]:
        originals = [
            r for r in rows if r["task"] == task and r["metadata"]["variant"] == "original"
        ]
        assert len(originals) == size
        assert len({r["group"] for r in originals}) == 1
    math = next(row for row in rows if row["task"] == "gsm_symbolic")
    assert math["prompt"] == math_examples(1)[0].prompt
    assert math["diagnostic_answer"] == "2"
    assert "gold reasoning" not in math["diagnostic_prompt"]
    assert np.isfinite(math["answer_log_prob_nats"])


def test_circuit_pair_metadata_matches_across_changed_model_weights(
    model, tokenizer, args, tmp_path
):
    pool = commonsense(8) + math_examples(8)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    runner.circuit_maps(
        model,
        tokenizer,
        pool,
        args=args,
        budget=RunBudget(120, 0, True),
        output=a_dir,
        tokenizer_hash="same-tokenizer",
    )
    with torch.no_grad():
        model.lm_head.weight.mul_(1.1)
    runner.circuit_maps(
        model,
        tokenizer,
        pool,
        args=args,
        budget=RunBudget(120, 0, True),
        output=b_dir,
        tokenizer_hash="same-tokenizer",
    )
    for task in ("ethics_commonsense", "gsm_symbolic"):
        a = json.loads((a_dir / "circuits" / f"{task}.json").read_text())
        b = json.loads((b_dir / "circuits" / f"{task}.json").read_text())
        validate_matched_rows(a, b)
        assert len(a["node_names"]) == 10
        x = np.load(a_dir / "circuits" / f"{task}.npz")["scores"]
        y = np.load(b_dir / "circuits" / f"{task}.npz")["scores"]
        assert x.shape == y.shape == (len(a["item_ids"]), 10)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert not np.allclose(x, y, atol=1e-8)
        interventions = json.loads((a_dir / "circuits" / f"{task}_interventions.json").read_text())
        from geode.circuits.sanity import validate_interventions

        validate_interventions(interventions, a, require_typed=True)
        scored_ids = {identifier for pair in a["item_ids"] for identifier in pair.split("|")}
        for patch in interventions:
            assert patch["clean_id"] not in scored_ids
            assert patch["corrupt_id"] not in scored_ids


def test_pair_inputs_teacher_force_same_complete_clean_answer(tokenizer):
    a, b = math_examples(2)
    a, b = runner.diagnostic_example(a), runner.diagnostic_example(b)
    clean, corrupt, mask, negative = runner._pair_inputs(a, b, tokenizer)
    assert len(clean) == len(corrupt) == len(mask)
    assert [token for token, keep in zip(clean, mask) if keep] == [
        token for token, keep in zip(corrupt, mask) if keep
    ]
    assert negative is None
    assert not any("gold reasoning" in ex.prompt for ex in (a, b))


def test_cost_timeout_after_run_start_records_failure_before_any_download(args, monkeypatch):
    monkeypatch.setattr(runner, "verify_data", lambda _path: {"synthetic": True})
    monkeypatch.setattr(plan, "load_examples", lambda *_a, **_k: commonsense(2))
    monkeypatch.setattr(runner, "environment_provenance", lambda _root: {"test": True})

    def timed_out(_self):
        raise TimeoutError("synthetic exhausted pilot budget")

    monkeypatch.setattr(RunBudget, "check", timed_out)
    with pytest.raises(TimeoutError, match="exhausted"):
        runner.run(args)
    from pathlib import Path

    output = Path(args.output)
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "TimeoutError" in metadata["error"]
    assert (output / "selected_examples.jsonl").exists()
    assert not (output / "stage2").exists()


def test_unconfirmed_run_rejected_before_accessing_data(args, monkeypatch):
    args.confirm_cost = False

    def fail(_path):
        raise AssertionError("Data must not be accessed before compute confirmation")

    monkeypatch.setattr(runner, "verify_data", fail)
    with pytest.raises(ValueError, match="confirm-cost"):
        runner.run(args)


def test_controlled_pairs_freeze_sources_then_keep_eight_clustered_variants(tokenizer):
    pairs, coverage = runner.controlled_pairs(
        commonsense(8), tokenizer, count=2, seed=7, max_context=512
    )
    assert len(pairs) == 16
    grouped = {}
    for a, b in pairs:
        grouped.setdefault((a.group, b.group), []).append((a, b))
        assert a.metadata["variant"] == b.metadata["variant"]
        assert a.label != b.label
        assert a.group != b.group
    assert len(grouped) == 2
    assert len({g for pair in grouped for g in pair}) == 4
    assert all(len({a.metadata["variant"] for a, _ in rows}) == 8 for rows in grouped.values())
    assert coverage["rejected_unaligned_control_groups"] == 0


def test_utilitarian_corruption_swaps_scenarios_within_source_and_keeps_mapping(tokenizer):
    pool = parse_ethics(
        "utilitarianism",
        [{"better": "w0 w1", "worse": "w2 w3"}, {"better": "w1 w2", "worse": "w3 w4"}],
    )
    pairs, coverage = runner.controlled_pairs(pool, tokenizer, count=1, seed=0, max_context=512)
    assert len(pairs) == 8
    assert len({a.group for a, _ in pairs}) == 1
    for a, b in pairs:
        assert a.group == b.group
        assert a.prompt != b.prompt
        assert a.options == b.options
        assert a.answer != b.answer
        clean, corrupt, mask, negative = runner._pair_inputs(a, b, tokenizer)
        assert len(clean) == len(corrupt) == len(negative)
        assert [c for c, selected in zip(clean, mask) if selected] == [
            c for c, selected in zip(corrupt, mask) if selected
        ]
    assert coverage["utilitarian_source_groups"] == 1


@pytest.mark.parametrize(
    "field,value", [("pairs", 0), ("max_context", -1), ("batch_size", 0), ("probe_groups", 0)]
)
def test_invalid_workload_config_rejected(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        runner.validate_config(args)


def test_full_mode_rejects_pilot_stages_and_pair_counts(args):
    args.mode = "full"
    with pytest.raises(ValueError, match="all seven stages"):
        runner.validate_config(args)
    args.stages = [checkpoint.stage for checkpoint in runner.CHECKPOINTS]
    with pytest.raises(ValueError, match="probe-groups"):
        runner.validate_config(args)
    args.probe_groups = 40
    args.interventions = 128
    with pytest.raises(ValueError, match="pairs"):
        runner.validate_config(args)


def test_run_with_actual_tiny_models_writes_complete_reviewable_artifacts(
    model, tokenizer, args, monkeypatch
):
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pool = commonsense(8)
    train = parse_ethics(
        "commonsense",
        [{"label": str(i % 2), "input": f"w{i % 5} w0 w1"} for i in range(8)],
        split="train",
    )
    args.tasks = ["ethics_commonsense"]
    args.groups = 2
    args.stages = ["stage2"]
    monkeypatch.setattr(runner, "verify_data", lambda _path: {"synthetic": True})
    monkeypatch.setattr(
        plan,
        "load_examples",
        lambda *_a, **kwargs: pool + train if kwargs.get("include_train") else pool,
    )
    monkeypatch.setattr(runner, "environment_provenance", lambda _root: {"test": True})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_kw: deepcopy(tokenizer))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *_a, **_kw: deepcopy(model))
    metadata = runner.run(args)
    assert metadata["status"] == "complete"
    output = Path(args.output)
    stage = output / "stage2"
    assert len(read_jsonl(stage / "behavior.jsonl")) == 18
    assert (stage / "circuits" / "ethics_commonsense.npz").exists()
    assert (stage / "hardware.json").exists()
    probe = json.loads((stage / "probes" / "ethics_commonsense.json").read_text())
    assert probe["n_groups"] == 8
    assert all(":train:" in group for group in probe["group_ids"])
    assert (output / "report.md").exists()
