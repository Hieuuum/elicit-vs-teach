"""Embedding warm-start properties: untie no-op and exact row isolation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from geode.arith import SftExample
from geode.edl.masking import TaskFormat
from geode.train import assert_untie_is_noop, train_embedding_rows


def _model() -> LlamaForCausalLM:
    torch.manual_seed(7)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )


def _examples() -> list[SftExample]:
    return [
        SftExample(input_ids=[1, 5, 6, 7, 2], label_span=(3, 5)),
        SftExample(input_ids=[1, 8, 6, 9, 2], label_span=(3, 5)),
    ]


def test_v5_68_embedding_untie_is_logit_noop() -> None:
    model = _model()
    ids = torch.tensor([[1, 5, 6, 7]])
    assert_untie_is_noop(model, ids, device="cpu")
    assert not model.config.tie_word_embeddings
    assert model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()


def test_v5_68_embedding_only_declared_rows_move() -> None:
    model = _model()
    assert_untie_is_noop(model, torch.tensor([[1, 5, 6, 7]]), device="cpu")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = train_embedding_rows(
        model,
        _examples(),
        TaskFormat(name="arith_nl_add", format_version="v1"),
        rows=(5, 6),
        lr=0.1,
        steps=2,
        micro_batch_size=2,
        device="cpu",
        seed=11,
    )

    assert set(result.moved_rows) <= {5, 6}
    assert result.row_delta_l2[5] > 0
    assert result.row_delta_l2[6] > 0
    after = model.state_dict()
    for name, value in before.items():
        if name == "model.embed_tokens.weight":
            unchanged = [row for row in range(value.shape[0]) if row not in {5, 6}]
            assert torch.equal(value[unchanged], after[name][unchanged])
        else:
            assert torch.equal(value, after[name]), name


def test_v5_68_absent_embedding_row_leaves_prompt_logits_unchanged() -> None:
    model = _model()
    prompt = torch.tensor([[1, 5, 6, 7]])
    assert_untie_is_noop(model, prompt, device="cpu")
    model.eval()
    with torch.no_grad():
        before = model(prompt).logits.clone()
    result = train_embedding_rows(
        model,
        _examples(),
        TaskFormat(name="arith_nl_add", format_version="v1"),
        rows=(8,),
        lr=0.1,
        steps=2,
        micro_batch_size=2,
        device="cpu",
        seed=11,
    )
    model.eval()
    with torch.no_grad():
        after = model(prompt).logits

    assert result.row_delta_l2[8] > 0
    assert torch.equal(before, after)


def test_v5_68_embedding_rows_require_untied_head() -> None:
    model = _model()
    try:
        train_embedding_rows(
            model,
            _examples(),
            TaskFormat(name="arith_nl_add", format_version="v1"),
            rows=(5,),
            lr=0.1,
            steps=1,
            micro_batch_size=2,
            device="cpu",
            seed=11,
        )
    except ValueError as exc:
        assert "untie lm_head" in str(exc)
    else:
        raise AssertionError("tied head was accepted")


def test_v5_69_warmstart_candidate_selection_is_deterministic() -> None:
    from geode.train import select_warmstart_candidate

    candidates = [
        {"lr": 0.001, "operator_accuracy": 0.96, "nl_accuracy": 0.2, "nl_loss_nats": 1.0},
        {"lr": 0.1, "operator_accuracy": 0.94, "nl_accuracy": 0.9, "nl_loss_nats": 0.1},
        {"lr": 0.01, "operator_accuracy": 0.96, "nl_accuracy": 0.2, "nl_loss_nats": 0.8},
    ]
    # Operator retention is reported but does not filter: the NL-best row wins.
    assert select_warmstart_candidate(candidates)["lr"] == 0.1


def test_warmstart_token_ids_are_pinned() -> None:
    from geode.train import resolve_warmstart_rows

    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: {"Ġs": 261, "um": 492}[token])
    assert resolve_warmstart_rows(tokenizer, ["Ġs", "um"]) == (261, 492)
