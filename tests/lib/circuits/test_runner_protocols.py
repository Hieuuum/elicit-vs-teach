"""Pure runner protocol tests; no pretrained models, network, GPU or artifact writes."""

from collections import Counter, defaultdict

import pytest

from geode.circuits.checkpoints import CHECKPOINTS
from geode.circuits.data import Example, parse_crux, parse_ethics, parse_gsm
from geode.circuits.runner import (
    cap_instances,
    controlled_pairs,
    diagnostic_example,
    parser,
    validate_config,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_math_diagnostic_never_conditions_on_gold_rationale_and_keeps_source_group():
    original = parse_gsm(
        [
            {
                "id": 12,
                "instance": 3,
                "question": "How many blue objects?",
                "answer": "SECRET_RATIONALE 25 divided by 5 is 5. #### 5",
            }
        ]
    )[0]
    diagnostic = diagnostic_example(original)
    assert "SECRET_RATIONALE" not in diagnostic.prompt
    assert diagnostic.answer == "5"
    assert diagnostic.prompt.startswith(original.prompt)
    assert diagnostic.id == original.id and diagnostic.group == original.group
    assert original.answer.startswith("SECRET_RATIONALE")
    assert not original.prompt.endswith("The final answer is ")


def test_ethics_diagnostic_keeps_original_prompt_answer_and_semantics():
    original = parse_ethics("commonsense", [{"label": "1", "input": "I stole money."}])[0]
    assert diagnostic_example(original) == original


def test_crux_diagnostic_scores_only_unknown_answer_component():
    originals = parse_crux(
        [
            {
                "id": "sample",
                "code": "def f(x):\n    return x+1",
                "input": "123",
                "output": "124",
            }
        ]
    )
    input_example, output_example = map(diagnostic_example, originals)
    assert input_example.answer == "f(123)"
    assert output_example.answer == "124"
    assert "124" not in input_example.answer
    assert "123" not in output_example.answer
    for original, diagnostic in zip(originals, (input_example, output_example)):
        assert diagnostic.prompt.startswith(original.prompt)
        assert diagnostic.group == original.group
        assert diagnostic.metadata == original.metadata


def test_pilot_instance_cap_changes_only_math_and_preserves_ethics_groups():
    ethics = parse_ethics(
        "justice", [{"label": str(i % 2), "scenario": f"scenario {i}"} for i in range(8)]
    )
    maths = parse_gsm(
        [
            {"id": template, "instance": i, "question": "How many?", "answer": "#### 3"}
            for template in range(3)
            for i in range(5)
        ]
    )
    crux = Example("crux", "cruxeval_input", "prompt", "answer", "function0")
    examples = [*ethics, *maths, crux]
    selected = cap_instances(examples, 2)
    assert [e for e in selected if e.task == "ethics_justice"] == ethics
    assert crux in selected
    counts = Counter(e.group for e in selected if e.task == "gsm_symbolic")
    assert len(counts) == 3 and set(counts.values()) == {2}
    assert cap_instances(examples, None) == examples
    assert cap_instances(list(reversed(examples)), 2) == selected


def _commonsense_pool():
    return parse_ethics(
        "commonsense", [{"label": str(i % 2), "input": f"action {i:02d}"} for i in range(12)]
    )


def test_control_pairs_freeze_sources_then_expand_complete_factorials():
    tokenizer = CharacterTokenizer()
    pool = _commonsense_pool()
    pairs, coverage = controlled_pairs(pool, tokenizer, count=3, seed=17, max_context=4096)
    assert len(pairs) == 3 * 8
    clusters = defaultdict(list)
    for a, b in pairs:
        clusters[(a.group, b.group)].append((a, b))
        assert a.task == b.task
        assert a.group != b.group
        assert a.answer != b.answer
        assert a.metadata["variant"] == b.metadata["variant"]
        assert a.options == b.options
        assert len(tokenizer.encode(a.prompt)) == len(tokenizer.encode(b.prompt))
    assert len(clusters) == 3
    assert all(len(v) == 8 for v in clusters.values())
    all_sources = [group for cluster in clusters for group in cluster]
    assert len(all_sources) == len(set(all_sources))
    for cluster in clusters.values():
        assert len({a.metadata["variant"] for a, _ in cluster}) == 8
    assert controlled_pairs(pool, tokenizer, count=3, seed=17, max_context=4096)[0] == pairs
    assert (
        controlled_pairs(list(reversed(pool)), tokenizer, count=3, seed=17, max_context=4096)[0]
        == pairs
    )
    assert coverage["control_renderings"] == len(pairs)


def test_one_unaligned_rendering_rejects_whole_control_cluster():
    class VariantLengthTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text)
            if "alpha" in text and "A: not morally wrong" in text:
                return [999] + ids
            return ids

    pool = parse_ethics(
        "commonsense",
        [
            {"label": "0", "input": "alpha"},
            {"label": "1", "input": "bravo"},
        ],
    )
    pairs, coverage = controlled_pairs(
        pool, VariantLengthTokenizer(), count=1, seed=0, max_context=4096
    )
    assert pairs == []
    assert coverage["rejected_unaligned_control_groups"] == 1


def test_utilitarian_corruption_swaps_actual_scenarios_with_same_labels_both_directions():
    tokenizer = CharacterTokenizer()
    pool = parse_ethics("utilitarianism", [("won a prize", "lost a friend"), ("healthy", "ill")])
    pairs, coverage = controlled_pairs(pool, tokenizer, count=1, seed=0, max_context=4096)
    assert len(pairs) == 8
    assert len({a.group for a, _ in pairs}) == 1
    for a, b in pairs:
        assert a.group == b.group
        assert a.options == b.options
        assert a.answer != b.answer
        assert a.label == 1 - b.label
        assert a.metadata["scenario_order_swapped"] != b.metadata["scenario_order_swapped"]
        assert len(tokenizer.encode(a.prompt)) == len(tokenizer.encode(b.prompt))
        assert (b, a) in pairs
    assert Counter(a.label for a, _ in pairs) == {0: 4, 1: 4}
    assert coverage["utilitarian_source_groups"] == 1


def test_context_limit_removes_entire_control_pair_without_partial_labels():
    pairs, coverage = controlled_pairs(
        _commonsense_pool(), CharacterTokenizer(), count=2, seed=0, max_context=10
    )
    assert pairs == []
    assert coverage["excluded_context_overflow"] == 12


@pytest.mark.parametrize(
    "field",
    [
        "groups",
        "instances_per_group",
        "pairs",
        "pair_pool_groups",
        "probe_groups",
        "interventions",
        "batch_size",
        "max_context",
        "max_new_tokens",
        "cpu_threads",
    ],
)
def test_config_rejects_nonpositive_sampling_and_execution_counts(field):
    args = parser().parse_args(["run"])
    setattr(args, field, 0)
    with pytest.raises(ValueError, match=field):
        validate_config(args)


def test_full_config_cannot_silently_run_subset_of_checkpoint_lineage():
    args = parser().parse_args(["run", "--mode", "full"])
    with pytest.raises(ValueError, match="all seven stages"):
        validate_config(args)
    args.stages = [checkpoint.stage for checkpoint in CHECKPOINTS]
    with pytest.raises(ValueError, match="probe-groups.*interventions"):
        validate_config(args)
    args.probe_groups = 40
    args.interventions = 128
    with pytest.raises(ValueError, match="pairs 512"):
        validate_config(args)
    args.pairs = 512
    validate_config(args)


@pytest.mark.parametrize("tasks", [[], ["gsm_symbolic", "gsm_symbolic"]])
def test_config_rejects_empty_or_repeated_task_domains(tasks):
    args = parser().parse_args(["run"])
    args.tasks = tasks
    with pytest.raises(ValueError, match="unique list"):
        validate_config(args)
