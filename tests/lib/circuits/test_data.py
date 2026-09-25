"""Offline properties guarding label semantics, grouping and native scoring."""

from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import hashlib

import pytest

from geode.circuits import data

from geode.circuits.data import (
    _sample_groups,
    candidate_negative,
    ethics_controls,
    grade,
    gsm_prompt,
    native_aggregate,
    numeric_answer,
    parse_crux,
    parse_ethics,
    parse_gsm,
    single_token_labels,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def ethics_rows(task, size):
    return [
        {
            "label": str(i % 2),
            "input": f"action {i}",
            "scenario": f"scenario {i}",
            "excuse": f"excuse {i}",
        }
        for i in range(size)
    ]


@pytest.mark.parametrize("task,size", [("justice", 4), ("deontology", 4), ("virtue", 5)])
def test_native_ethics_metric_requires_every_group_member_correct(task, size):
    examples = parse_ethics(task, ethics_rows(task, 2 * size))
    records = [{**asdict(e), "correct": i != 0} for i, e in enumerate(examples)]
    score = native_aggregate(records)[f"ethics_{task}/test/original"]
    assert score["accuracy"] == 0.5
    assert score["row_accuracy"] == (2 * size - 1) / (2 * size)
    assert score["n_groups"] == 2


@pytest.mark.parametrize("task,size", [("justice", 4), ("deontology", 4), ("virtue", 5)])
def test_partial_ethics_groups_are_never_silently_scored(task, size):
    with pytest.raises(ValueError, match="incomplete"):
        parse_ethics(task, ethics_rows(task, size - 1))
    examples = parse_ethics(task, ethics_rows(task, size))
    with pytest.raises(ValueError, match="incomplete"):
        native_aggregate([{**asdict(e), "correct": True} for e in examples[:-1]])


def test_ethics_macro_weights_tasks_equally_and_keeps_splits_separate():
    records = []
    for task, size in [("commonsense", 10), ("justice", 4), ("deontology", 4), ("virtue", 5)]:
        for e in parse_ethics(task, ethics_rows(task, size)):
            records.append({**asdict(e), "correct": task != "commonsense"})
    for e in parse_ethics("utilitarianism", [("happy", "sad")]):
        records.append({**asdict(e), "correct": True})
    scores = native_aggregate(records)
    assert scores["ethics_macro/test/original"]["accuracy"] == 0.8
    assert "ethics_macro/test_hard/original" not in scores


def test_native_aggregation_rejects_duplicate_rows():
    e = parse_ethics("commonsense", ethics_rows("commonsense", 1))[0]
    row = {**asdict(e), "correct": True}
    with pytest.raises(ValueError, match="duplicate"):
        native_aggregate([row, row])


@pytest.mark.parametrize("label", [0, 1])
def test_ethics_controls_are_full_factorial_and_preserve_semantic_judgment(label):
    e = parse_ethics("commonsense", [{"label": str(label), "input": "I helped a person."}])[0]
    controls = ethics_controls(e, CharacterTokenizer())
    assert len(controls) == len({c.id for c in controls}) == 8
    assert {c.group for c in controls} == {e.group}
    assert {c.metadata["semantic_label"] for c in controls} == {label}
    assert Counter(c.answer for c in controls) == {"A": 2, "B": 2, "C": 2, "D": 2}
    positions = Counter(c.metadata["option_positions"].index(label) for c in controls)
    assert positions == {0: 4, 1: 4}
    for c in controls:
        assert c.answer == c.options[label]
        assert grade(c, c.answer)["correct"]
        assert not grade(c, candidate_negative(c, []))["correct"]
        assert c.metadata["source"] == e.metadata["source"]


def test_utilitarian_controls_remove_always_first_scenario_shortcut():
    e = parse_ethics("utilitarianism", [("won a prize", "lost a friend")])[0]
    controls = ethics_controls(e, CharacterTokenizer())
    assert Counter(c.label for c in controls) == {0: 4, 1: 4}
    assert Counter(c.answer for c in controls) == {"A": 2, "B": 2, "C": 2, "D": 2}
    for c in controls:
        preferred = "Scenario 2: won a prize" if c.label else "Scenario 1: won a prize"
        assert preferred in c.prompt
        assert c.answer == c.options[c.label]
        assert c.metadata["semantic_label"] == 0


def test_single_token_check_detects_boundary_merging_not_only_label_length():
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith(":A"):
                return super().encode(text[:-2]) + [999]
            return super().encode(text)

    with pytest.raises(ValueError, match="stable token"):
        single_token_labels("Answer:", ["A", "B"], MergingTokenizer())
    with pytest.raises(ValueError, match="stable token"):
        single_token_labels("Answer:", ["AA", "BB"], CharacterTokenizer())
    e = parse_ethics("commonsense", ethics_rows("commonsense", 1))[0]
    assert e.prompt.endswith("Answer:\n")
    assert len(ethics_controls(e, MergingTokenizer())) == 8


def test_single_token_check_rejects_collapsed_unknown_labels():
    class UnknownTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [0] * len(text)

    with pytest.raises(ValueError, match="same tokenizer token"):
        single_token_labels("Answer:", ["A", "B"], UnknownTokenizer())


def test_group_sampling_is_complete_deterministic_and_seed_sensitive():
    examples = parse_ethics("justice", ethics_rows("justice", 80))
    selected = _sample_groups(examples, 4, 13)
    assert selected == _sample_groups(examples, 4, 13)
    assert selected != _sample_groups(examples, 4, 14)
    assert len(selected) == 16
    assert set(Counter(e.group for e in selected).values()) == {4}
    assert _sample_groups(examples, 100, 0) == examples


def test_train_virtue_and_deontology_group_by_scenario_not_shuffled_row_blocks():
    for task in ("virtue", "deontology"):
        rows = ethics_rows(task, 6)
        rows[0]["scenario"] = "shared [SEP] generous"
        rows[5]["scenario"] = "shared [SEP] selfish"
        examples = parse_ethics(task, rows, "train")
        assert examples[0].group == examples[5].group
        assert examples[0].group != examples[1].group


@pytest.mark.parametrize(
    "text,value",
    [
        ("The final answer is 1,200.", Decimal(1200)),
        ("3.00", Decimal(3)),
        ("first 100; final -2.50", Decimal("-2.5")),
        ("#### 0", Decimal(0)),
        ("No numeric answer", None),
        ("3/4", Decimal(4)),
        ("1e3", Decimal(3)),
    ],
)
def test_gsm_numeric_parser_matches_published_last_number_heuristic(text, value):
    assert numeric_answer(text) == value


def test_gsm_prompt_has_fixed_eight_demos_and_target_without_gold_reasoning():
    prompt = gsm_prompt("How much is 111 plus 222?")
    assert prompt.count("Q:") == 9
    assert prompt.count("The final answer is") == 8
    assert prompt.endswith("Q: How much is 111 plus 222?\nA: Let's think step by step.")
    assert "333" not in prompt


def test_gsm_templates_are_shared_groups_across_instances_and_errors_count():
    rows = [
        {"id": 7, "instance": i, "question": "How many?", "answer": "reason 10 #### 4"}
        for i in range(2)
    ]
    examples = parse_gsm(rows)
    assert examples[0].group == examples[1].group
    assert examples[0].id != examples[1].id
    assert grade(examples[0], "final 4.0 Q: How many 88?")["correct"]
    assert grade(examples[0], "I cannot solve this")["parse_failure"]
    assert not grade(examples[0], candidate_negative(examples[0], []))["correct"]


def test_crux_tasks_share_function_groups_and_prompts_do_not_reveal_unknown():
    row = {
        "id": "sample_7",
        "code": "def f(x):\n    return x + 1",
        "input": "4321",
        "output": "4322",
    }
    input_e, output_e = parse_crux([row])
    assert input_e.group == output_e.group
    assert input_e.id != output_e.id
    assert "4321" not in input_e.prompt
    assert "4322" not in output_e.prompt
    assert "assert f(??) == 4322" in input_e.prompt
    assert "assert f(4321) == ??" in output_e.prompt
    assert grade(input_e, input_e.answer)["correct"]
    assert grade(output_e, output_e.answer)["correct"]


def test_ethics_original_preserves_source_strings_and_strict_label_parsing():
    text = "I helped.\nThen I went home."
    e = parse_ethics("commonsense", [{"label": "0", "input": text}])[0]
    assert e.prompt.startswith(text)
    assert grade(e, " 0 \n")["correct"]
    assert grade(e, "0 or 1")["parse_failure"]
    assert not grade(e, "1")["correct"]


def test_control_group_scoring_keeps_each_variant_separate():
    examples = parse_ethics("justice", ethics_rows("justice", 4))
    controls = [c for e in examples for c in ethics_controls(e, CharacterTokenizer())]
    scores = native_aggregate([{**asdict(c), "correct": True} for c in controls])
    assert len(scores) == 8
    assert all(s["n_examples"] == 4 and s["accuracy"] == 1 for s in scores.values())


def test_cached_data_are_offline_and_tampering_is_detected(tmp_path, monkeypatch):
    payload = b"public fixture bytes"
    monkeypatch.setattr(
        data,
        "SOURCES",
        {
            "fixture.jsonl": {
                "url": "https://example.invalid/no-network",
                "revision": "fixed-revision",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    )
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    (tmp_path / "fixture.jsonl").write_bytes(payload)
    manifest = data.prepare_data(tmp_path)
    assert data.verify_data(tmp_path) == manifest
    (tmp_path / "fixture.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance mismatch"):
        data.verify_data(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        data.prepare_data(tmp_path)
