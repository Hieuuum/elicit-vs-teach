"""Offline batching/teacher-forcing properties, with actual tiny OLMo 2."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

from geode.circuits.execution import (
    RunBudget,
    complete_answer_log_probs,
    encode_answer,
    generate_answers,
    make_pairs,
    pad_encoded,
    score_answers,
    score_choices,
    select_groups,
)


@pytest.fixture
def tokenizer():
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}
    vocab.update({f"w{i}": i + 3 for i in range(28)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(3)
        config = Olmo2Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            eos_token_id=1,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        return Olmo2ForCausalLM(config).eval()


def test_explicit_token_boundary_protocol_and_empty_rejection(tokenizer):
    # WordLevel merges "w1w2" into an UNK token if concatenated as raw text.
    ids, mask = encode_answer(tokenizer, "w1", "w2")
    assert ids == tokenizer.encode("w1") + tokenizer.encode("w2")
    assert ids != tokenizer.encode("w1w2")
    assert mask == [False, True]
    for prompt, answer in [("", "w1"), ("w1", "")]:
        with pytest.raises(ValueError, match="each contain"):
            encode_answer(tokenizer, prompt, answer)


def test_causal_shift_full_answer_mean_and_no_gradient_on_unselected_logits():
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 9, requires_grad=True)
    ids = torch.tensor([[2, 3, 4, 5, 0], [2, 3, 4, 5, 6]])
    mask = torch.tensor([[False, False, True, True, False], [False, True, True, True, True]])
    result = complete_answer_log_probs(logits, ids, mask)
    expected = torch.stack(
        [
            torch.stack(
                [logits[b, t - 1].log_softmax(-1)[ids[b, t]] for t in range(1, 5) if mask[b, t]]
            ).mean()
            for b in range(2)
        ]
    )
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 3:].abs().sum() == 0
    assert logits.grad[0, 1:3].abs().sum() > 0


def test_unselected_nan_logits_do_not_contaminate_answer_metric():
    logits = torch.zeros(1, 4, 6)
    logits[0, 0] = float("nan")
    logits[0, 3] = float("nan")
    mask = torch.tensor([[False, False, True, True]])
    result = complete_answer_log_probs(logits, torch.tensor([[1, 2, 3, 4]]), mask)
    assert result.item() == pytest.approx(-torch.log(torch.tensor(6.0)).item())


def test_scoring_actual_model_batch_padding_order_and_singletons(model, tokenizer):
    prompts = ["w1 w2 w3 w4", "w3", "w1 w6"]
    answers = ["w7 w8", "w9", "w2 w3 w4"]
    values = score_answers(model, tokenizer, prompts, answers, device="cpu", batch_size=3)
    individual = score_answers(model, tokenizer, prompts, answers, device="cpu", batch_size=1)
    assert values == pytest.approx(individual, abs=5e-7)
    for p, a, actual in zip(prompts, answers, values, strict=True):
        ids, mask = encode_answer(tokenizer, p, a)
        with torch.no_grad():
            logits = model(torch.tensor([ids]), use_cache=False).logits[0]
        expected = (
            torch.stack([logits[t - 1].log_softmax(-1)[ids[t]] for t in range(len(ids)) if mask[t]])
            .mean()
            .item()
        )
        assert actual == pytest.approx(expected, abs=5e-7)


def test_padding_masks_exclude_only_padding_and_preserve_targets():
    ids, attention, targets = pad_encoded(
        [([3, 4, 5], [False, True, True]), ([6, 7], [False, True])], 0, "cpu"
    )
    assert ids.tolist() == [[3, 4, 5], [6, 7, 0]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert targets.tolist() == [[False, True, True], [False, True, False]]


class EchoGenerator(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        if self.fail:
            raise RuntimeError("deliberate generation failure")
        assert (attention_mask[:, -1] == 1).all()
        assert kwargs["do_sample"] is False
        self.calls.append((input_ids.clone(), kwargs))
        suffix = input_ids[:, -1:].repeat(1, kwargs["max_new_tokens"])
        return torch.cat([input_ids, suffix], dim=1)


def test_generation_restores_order_padding_and_flags_every_overflow(tokenizer):
    prompts = ["w1 w2 w3", "w4", " ".join(["w1"] * 8), "", "w5 w6"]
    rows = generate_answers(
        EchoGenerator(),
        tokenizer,
        prompts,
        device="cpu",
        batch_size=3,
        max_new_tokens=2,
        max_context=8,
    )
    assert tokenizer.padding_side == "right"
    assert len(rows) == len(prompts)
    assert [r["generation"] for r in rows] == ["w3 w3", "w4 w4", "", "", "w6 w6"]
    assert [r["context_overflow"] for r in rows] == [False, False, True, True, False]
    assert [r["truncated"] for r in rows] == [True, True, False, False, True]


def test_generation_context_cap_is_independent_of_batchmates(tokenizer):
    prompts = ["w1", "w2 w3 w4 w5"]
    one = generate_answers(
        EchoGenerator(),
        tokenizer,
        prompts,
        device="cpu",
        batch_size=1,
        max_new_tokens=3,
        max_context=6,
    )
    together = generate_answers(
        EchoGenerator(),
        tokenizer,
        prompts,
        device="cpu",
        batch_size=2,
        max_new_tokens=3,
        max_context=6,
    )
    assert one == together
    assert [row["output_tokens"] for row in together] == [3, 2]


def test_generation_restores_padding_after_error(tokenizer):
    with pytest.raises(RuntimeError, match="deliberate"):
        generate_answers(EchoGenerator(fail=True), tokenizer, ["w1"], device="cpu")
    assert tokenizer.padding_side == "right"


def test_eos_removes_batch_padding_and_is_not_marked_truncated(tokenizer):
    class EosGenerator(EchoGenerator):
        def generate(self, input_ids, attention_mask, **kwargs):
            suffix = torch.tensor([[6, 1, 0], [7, 8, 1]])
            return torch.cat([input_ids, suffix], dim=1)

    rows = generate_answers(EosGenerator(), tokenizer, ["w1", "w2"], device="cpu", max_new_tokens=3)
    assert [r["generation"] for r in rows] == ["w3", "w4 w5"]
    assert [r["output_tokens"] for r in rows] == [2, 3]
    assert not any(r["truncated"] for r in rows)


def test_real_model_greedy_batching_matches_singletons(model, tokenizer):
    prompts = ["w1 w2 w3", "w4", "w5 w6"]
    single = generate_answers(
        model, tokenizer, prompts, device="cpu", batch_size=1, max_new_tokens=3, max_context=32
    )
    batched = generate_answers(
        model, tokenizer, prompts, device="cpu", batch_size=3, max_new_tokens=3, max_context=32
    )
    assert single == batched


def example(
    i, task="math", group=None, prompt="w1 w2", answer=None, variant="original", split="test"
):
    return SimpleNamespace(
        id=str(i),
        task=task,
        group=group or str(i),
        prompt=prompt,
        answer=answer or f"w{int(i) % 2 + 3}",
        metadata={"variant": variant, "split": split},
    )


def test_group_selection_retains_complete_variants_and_stratifies_task_split():
    rows = [
        example(i, task=task, group=f"{split}:{i}", variant=variant, split=split)
        for task in ["math", "code"]
        for split in ["test", "train"]
        for i in range(8)
        for variant in ["original", "control"]
    ]
    selected = select_groups(rows, 2, seed=8)
    assert len(selected) == 2 * 2 * 2 * 2
    assert [(e.task, e.group, e.metadata) for e in selected] == [
        (e.task, e.group, e.metadata) for e in select_groups(rows, 2, seed=8)
    ]
    for task in ["math", "code"]:
        for split in ["test", "train"]:
            subset = [e for e in selected if e.task == task and e.metadata["split"] == split]
            assert len({e.group for e in subset}) == 2
            for group in {e.group for e in subset}:
                assert {e.metadata["variant"] for e in subset if e.group == group} == {
                    "original",
                    "control",
                }


def test_pairing_disjoint_within_task_but_crux_directions_can_share_functions(tokenizer):
    rows = [
        example(i, task=task, group=f"function:{i}")
        for task in ["crux_i", "crux_o"]
        for i in range(8)
    ]
    pairs, report = make_pairs(rows, tokenizer, max_pairs_per_task=4, seed=9)
    assert len(pairs) == 8
    assert report["pairs_per_task"] == {"crux_i": 4, "crux_o": 4}
    for task in ["crux_i", "crux_o"]:
        used = [e.group for pair in pairs for e in pair if e.task == task]
        assert len(used) == len(set(used)) == 8
    assert all(a.task == b.task and a.answer != b.answer for a, b in pairs)


def test_pairs_obey_format_length_context_and_seed_reproducibility(tokenizer):
    rows = [
        example(i, prompt="w1 " * (2 + i // 4), variant="a" if i < 8 else "b") for i in range(16)
    ]
    pairs, report = make_pairs(rows, tokenizer, max_pairs_per_task=8, seed=6, max_context=5)
    assert report["excluded_context_overflow"] == 4
    assert report["eligible_examples"] == 12
    assert pairs == make_pairs(rows, tokenizer, max_pairs_per_task=8, seed=6, max_context=5)[0]
    assert pairs
    for a, b in pairs:
        assert a.metadata == b.metadata
        assert len(tokenizer.encode(a.prompt)) == len(tokenizer.encode(b.prompt))
        assert len(tokenizer.encode(a.prompt)) + len(tokenizer.encode(a.answer)) <= 5


def test_pair_capping_does_not_always_prefer_shortest_bucket(tokenizer):
    rows = [example(i, prompt="w1 " * (1 if i < 4 else 5)) for i in range(8)]
    selected_lengths = {
        len(
            tokenizer.encode(
                make_pairs(rows, tokenizer, max_pairs_per_task=1, seed=seed)[0][0][0].prompt
            )
        )
        for seed in range(20)
    }
    assert selected_lengths == {1, 5}


def test_scoring_context_overflow_fails_before_model_forward(tokenizer):
    with pytest.raises(ValueError, match="context overflow"):
        score_answers(EchoGenerator(), tokenizer, ["w1 w2"], ["w3 w4"], device="cpu", max_context=3)


def test_budget_requires_confirmation_and_stops_at_exact_deadline(monkeypatch):
    with pytest.raises(ValueError, match="confirm-cost"):
        RunBudget(30, 0.6, False)
    monkeypatch.setattr("geode.circuits.execution.time.monotonic", lambda: 100.0)
    budget = RunBudget(30, 0.6, True)
    assert budget.estimated_usd == pytest.approx(0.005)
    budget.check()
    monkeypatch.setattr("geode.circuits.execution.time.monotonic", lambda: 130.0)
    with pytest.raises(TimeoutError):
        budget.check()


def grouped_repeat_rows(n_groups=4, rows_per_group=6, task="math"):
    return [
        example(
            group * rows_per_group + row,
            task=task,
            group=f"template:{group}",
            prompt="w1 " * (2 + row % 2),
            answer="w3" if group % 2 == 0 else "w4",
        )
        for group in range(n_groups)
        for row in range(rows_per_group)
    ]


def test_group_pair_repeats_preserve_locked_partners_orientation_and_row_disjointness(tokenizer):
    rows = grouped_repeat_rows()
    base, _ = make_pairs(rows, tokenizer, max_pairs_per_task=20, seed=3)
    repeated, report = make_pairs(
        rows, tokenizer, max_pairs_per_task=20, seed=3, repeats_per_group_pair=3
    )
    assert repeated[: len(base)] == base
    assert len(base) == 2
    assert len(repeated) == 6
    assert {(a.group, b.group) for a, b in repeated} == {(a.group, b.group) for a, b in base}
    ids = [ex.id for pair in repeated for ex in pair]
    assert len(ids) == len(set(ids))
    partners = {}
    for a, b in repeated:
        partners.setdefault(a.group, set()).add(b.group)
        partners.setdefault(b.group, set()).add(a.group)
        assert a.metadata == b.metadata
        assert len(tokenizer.encode(a.prompt)) == len(tokenizer.encode(b.prompt))
        assert a.answer != b.answer
    assert all(len(others) == 1 for others in partners.values())
    assert report["independent_group_pairs"] == 2
    assert report["repeated_pairs"] == 4
    assert report["group_pairs_per_task"] == {"math": 2}
    assert {row["n_pairs"] for row in report["group_pair_counts"]} == {3}


def test_repeat_extension_respects_total_task_cap_and_balances_partnerships(tokenizer):
    rows = grouped_repeat_rows(task="math") + grouped_repeat_rows(task="code")
    pairs, report = make_pairs(
        rows, tokenizer, max_pairs_per_task=5, seed=9, repeats_per_group_pair=50
    )
    assert len(pairs) == 10
    assert report["pairs_per_task"] == {"code": 5, "math": 5}
    for task in ("code", "math"):
        counts = sorted(
            row["n_pairs"] for row in report["group_pair_counts"] if row["task"] == task
        )
        assert counts == [2, 3]
        ids = [ex.id for pair in pairs for ex in pair if ex.task == task]
        assert len(ids) == len(set(ids))


def test_repeat_extension_exhausts_rows_without_reusing_them(tokenizer):
    rows = grouped_repeat_rows(n_groups=2, rows_per_group=4)
    pairs, report = make_pairs(
        rows, tokenizer, max_pairs_per_task=50, seed=0, repeats_per_group_pair=50
    )
    assert len(pairs) == 4
    assert len({ex.id for pair in pairs for ex in pair}) == 8
    assert report["group_pair_counts"][0]["n_pairs"] == 4


def test_default_repetition_protocol_remains_one_pair_per_independent_partnership(tokenizer):
    rows = grouped_repeat_rows()
    default, _ = make_pairs(rows, tokenizer, max_pairs_per_task=20, seed=9)
    explicit, report = make_pairs(
        rows, tokenizer, max_pairs_per_task=20, seed=9, repeats_per_group_pair=1
    )
    assert default == explicit
    assert report["repeated_pairs"] == 0
    assert report["matched_pairs"] == report["independent_group_pairs"] == 2


def test_gsm_scale_repeats_yield_512_rows_in_50_independent_clusters(tokenizer):
    rows = grouped_repeat_rows(n_groups=100, rows_per_group=50)
    pairs, report = make_pairs(
        rows, tokenizer, max_pairs_per_task=512, seed=7, repeats_per_group_pair=50
    )
    assert len(pairs) == 512
    assert report["independent_group_pairs"] == 50
    assert len({ex.id for pair in pairs for ex in pair}) == 1024
    assert max(row["n_pairs"] for row in report["group_pair_counts"]) <= 11
    assert {row["n_pairs"] for row in report["group_pair_counts"]} == {10, 11}


def test_duplicate_row_ids_cannot_enter_pair_pool(tokenizer):
    rows = grouped_repeat_rows()
    with pytest.raises(ValueError, match="IDs must be unique"):
        make_pairs(
            rows + [rows[0]], tokenizer, max_pairs_per_task=20, seed=0, repeats_per_group_pair=2
        )


def test_choice_scoring_matches_full_teacher_forcing_and_preserves_orders(model, tokenizer):
    prompts = ["w1 w2 w3 ", "w4 ", "w1 w6 "]
    labels = [["w7", "w8"], ["w9", "w10", "w11"], ["w2", "w3"]]
    actual = score_choices(model, tokenizer, prompts, labels, device="cpu", batch_size=3)
    individual = score_choices(model, tokenizer, prompts, labels, device="cpu", batch_size=1)
    expected = [
        score_answers(
            model, tokenizer, [prompt] * len(options), options, device="cpu", batch_size=1
        )
        for prompt, options in zip(prompts, labels, strict=True)
    ]
    for a, b, c in zip(actual, individual, expected, strict=True):
        assert a == pytest.approx(b, abs=5e-7)
        assert a == pytest.approx(c, abs=5e-7)
    reversed_options = score_choices(
        model,
        tokenizer,
        prompts,
        [list(reversed(row)) for row in labels],
        device="cpu",
        batch_size=3,
    )
    for a, b in zip(actual, reversed_options, strict=True):
        assert a == list(reversed(b))


def test_choice_scoring_projects_only_one_position_per_unique_prompt(model, tokenizer):
    projections = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda _module, inputs: projections.append(inputs[0].shape[:2])
    )
    try:
        score_choices(
            model, tokenizer, ["w1 w2 w3 ", "w4 "], [["w5", "w6"]] * 2, device="cpu", batch_size=2
        )
    finally:
        handle.remove()
    assert projections == [(2, 1)]


def test_choice_scoring_restores_model_state_on_success_and_failure(model, tokenizer):
    model.train()
    model.model.layers[0].eval()
    states = [module.training for module in model.modules()]
    score_choices(model, tokenizer, ["w1 "], [["w2", "w3"]], device="cpu")
    assert [module.training for module in model.modules()] == states

    def fail(*_args):
        raise RuntimeError("injected model failure")

    handle = model.lm_head.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            score_choices(model, tokenizer, ["w1 "], [["w2", "w3"]], device="cpu")
        assert [module.training for module in model.modules()] == states
        assert tokenizer.padding_side == "right"
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "prompts,labels,error",
    [
        (["w1"], [["w2", "w3"]], "stable token"),
        (["w1 "], [["w2 w3", "w4"]], "stable token"),
        (["w1 "], [["w2", "w2"]], "collapse"),
        ([""], [["w2", "w3"]], "nonempty"),
    ],
)
def test_choice_scoring_rejects_unstable_multitoken_or_degenerate_labels(
    model, tokenizer, prompts, labels, error
):
    with pytest.raises(ValueError, match=error):
        score_choices(model, tokenizer, prompts, labels, device="cpu")


def test_choice_context_guard_accounts_for_teacher_forced_label(model, tokenizer):
    with pytest.raises(ValueError, match="context overflow"):
        score_choices(model, tokenizer, ["w1 w2 "], [["w3", "w4"]], device="cpu", max_context=2)
    assert (
        len(
            score_choices(
                model, tokenizer, ["w1 w2 "], [["w3", "w4"]], device="cpu", max_context=3
            )[0]
        )
        == 2
    )


def test_rejected_partner_does_not_consume_group_or_pair_quota(tokenizer):
    rows = grouped_repeat_rows(n_groups=4, rows_per_group=1)
    original, _ = make_pairs(rows, tokenizer, max_pairs_per_task=2, seed=4)
    a, b = original[0]
    c, d = original[1]
    other = c if c.answer != a.answer else d
    remaining = d if other is c else c
    allowed = {frozenset((a.id, other.id)), frozenset((b.id, remaining.id))}
    checked = []

    def verifier(left, right):
        checked.append((left.id, right.id))
        return frozenset((left.id, right.id)) in allowed

    pairs, report = make_pairs(rows, tokenizer, max_pairs_per_task=2, seed=4, pair_filter=verifier)
    assert checked[0] == (a.id, b.id)
    assert {frozenset((left.id, right.id)) for left, right in pairs} == allowed
    assert report["matched_pairs"] == 2
    assert report["used_groups"] == 4
    assert report["filter_candidates_rejected"] >= 1


def test_repeated_pairs_apply_directional_filter_and_cache_candidate_checks(tokenizer):
    rows = grouped_repeat_rows(n_groups=2, rows_per_group=6)
    checks = []

    def verifier(a, b):
        checks.append((a.id, b.id))
        # Require the same within-template row index, in one direction only.
        return a.group == "template:0" and int(a.id) % 6 == int(b.id) % 6

    pairs, report = make_pairs(
        rows,
        tokenizer,
        max_pairs_per_task=20,
        seed=2,
        repeats_per_group_pair=6,
        pair_filter=verifier,
    )
    assert len(pairs) == 6
    assert all(a.group == "template:0" and int(a.id) % 6 == int(b.id) % 6 for a, b in pairs)
    assert len(checks) == len(set(checks))
    assert report["filter_candidates_tested"] == len(checks)
    assert report["independent_group_pairs"] == 1
    assert report["repeated_pairs"] == 5


def test_filter_rejecting_every_pair_leaves_all_groups_available(tokenizer):
    pairs, report = make_pairs(
        grouped_repeat_rows(),
        tokenizer,
        max_pairs_per_task=4,
        seed=0,
        pair_filter=lambda _a, _b: False,
    )
    assert pairs == []
    assert report["used_groups"] == 0
    assert report["filter_candidates_tested"] == report["filter_candidates_rejected"] > 0
