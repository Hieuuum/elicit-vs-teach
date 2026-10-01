"""Offline fixtures for pre-rental provenance, tokenizer and context guards."""

import copy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from geode.circuits.data import Example
from geode.circuits.preflight import (
    EXPECTED_OLMO2_SHAPE,
    _sample_source_groups,
    architecture_signature,
    audit_contexts,
    metadata_file_names,
    token_length_summary,
    tokenizer_signature,
    validate_checkpoint_compatibility,
)


def tiny_tokenizer(extra=False):
    vocab = {"<eos>": 0, "<unk>": 1, "A": 2, "B": 3}
    if extra:
        vocab["C"] = 4
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>")


def fixture_record(stage="init"):
    return {
        "stage": stage,
        "revision": "a" * 40,
        "resolved_revision": "a" * 40,
        "architecture": architecture_signature(
            {**EXPECTED_OLMO2_SHAPE, "max_position_embeddings": 4096}
        ),
        "tokenizer": tokenizer_signature(tiny_tokenizer()),
        "weight_files": [{"filename": "model.safetensors", "advertised_sha256": "b" * 64}],
        "stored_config_dtype": "float32",
    }


def test_metadata_allowlist_never_downloads_weight_files():
    files = [
        SimpleNamespace(rfilename=name, size=128)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model-00001-of-00002.safetensors",
            "optimizer.pt",
            "model.safetensors.index.json",
            "chat_template.jinja",
        )
    ]
    names = metadata_file_names(files)
    assert "model.safetensors.index.json" in names
    assert "chat_template.jinja" in names
    assert not any(name.endswith((".safetensors", ".bin", ".pt")) for name in names)


def test_metadata_asset_size_guard_and_required_tokenizer():
    with pytest.raises(ValueError, match="required"):
        metadata_file_names([SimpleNamespace(rfilename="config.json", size=128)])
    with pytest.raises(ValueError, match="large"):
        metadata_file_names([SimpleNamespace(rfilename="tokenizer.json", size=70 * 1024**2)])


def test_pad_eos_policy_is_part_of_tokenizer_signature():
    tokenizer = tiny_tokenizer()
    assert tokenizer.pad_token_id is None
    signature = tokenizer_signature(tokenizer)
    assert signature["pad_token_id"] == signature["eos_token_id"] == 0
    assert signature == tokenizer_signature(tiny_tokenizer())


def test_tokenizer_vocabulary_and_wrapper_drift_are_detected():
    original = fixture_record()
    other = fixture_record("stage1")
    other["tokenizer"] = tokenizer_signature(tiny_tokenizer(extra=True))
    with pytest.raises(ValueError, match="Tokenizer"):
        validate_checkpoint_compatibility([original, other])
    other = fixture_record("stage1")
    other["tokenizer"]["padding_side"] = "left"
    with pytest.raises(ValueError, match="Tokenizer"):
        validate_checkpoint_compatibility([original, other])


@pytest.mark.parametrize(
    "field,value",
    [("num_hidden_layers", 32), ("max_position_embeddings", 8192), ("hidden_size", 1024)],
)
def test_architecture_changes_are_rejected(field, value):
    original, other = fixture_record(), fixture_record("stage1")
    other["architecture"][field] = value
    with pytest.raises(ValueError, match="Architecture"):
        validate_checkpoint_compatibility([original, other])


def test_stored_dtype_differences_do_not_falsely_imply_architecture_drift():
    original, other = fixture_record(), fixture_record("stage1")
    other["stored_config_dtype"] = "bfloat16"
    assert validate_checkpoint_compatibility([original, other])["status"] == "passed"


def test_unpinned_or_missing_weight_sha_provenance_is_rejected():
    item = fixture_record()
    item["resolved_revision"] = "c" * 40
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_checkpoint_compatibility([item])
    item = fixture_record()
    item["weight_files"][0]["advertised_sha256"] = None
    with pytest.raises((ValueError, TypeError)):
        validate_checkpoint_compatibility([item])


def test_context_counting_distinguishes_input_overflow_from_generation_reserve():
    result = token_length_summary([3, 8, 10, 11], max_context=10, generation_reserve=3)
    assert result["n_prompt_overflow"] == 1
    assert result["n_no_generation_slot"] == 2
    assert result["n_generation_budget_exceeds_context"] == 3
    assert "Not observable" in result["actual_generation_truncation"]


def examples(n=40):
    return [
        Example(
            id=f"item-{group}-{i}",
            task="gsm_symbolic",
            group=f"g-{group:02d}",
            prompt="x" * (group + 1),
            answer="0",
            metadata={"split": "test"},
        )
        for group in range(n)
        for i in range(2)
    ]


def test_context_sample_preserves_groups_long_texts_and_order_invariance():
    rows = examples()
    a = _sample_source_groups(rows, count=4, seed=7)
    b = _sample_source_groups(rows[::-1], count=4, seed=7)
    assert [ex.id for ex in a] == [ex.id for ex in b]
    assert "g-39" in {ex.group for ex in a}
    assert all(sum(ex.group == group for ex in a) == 2 for group in {ex.group for ex in a})


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}


def test_full_native_audit_never_substitutes_sample_maximum_for_global_bound():
    rows = examples()
    report = audit_contexts(
        rows,
        CharacterTokenizer(),
        max_context=25,
        max_new_tokens=5,
        sample_groups=2,
        diagnostic_builder=lambda ex: ex,
    )
    native = report["native"]["gsm_symbolic/test"]
    assert native["n_rows"] == 80 and native["max_observed_tokens"] == 40
    assert native["n_prompt_overflow"] == 30
    sampled = report["diagnostics"]["gsm_symbolic/test"]
    assert sampled["n_rows"] < 80
    assert "NOT a global bound" in sampled["scope"]


def test_ethics_controls_are_checked_at_real_label_boundaries_without_mutation():
    row = Example(
        id="ethics-0",
        task="ethics_commonsense",
        group="source-0",
        prompt="Judge: ",
        answer="0",
        label=0,
        options=("0", "1"),
        metadata={"split": "test", "source_text": "I returned the lost book."},
    )
    before = copy.deepcopy(row)
    report = audit_contexts(
        [row],
        CharacterTokenizer(),
        max_context=4096,
        sample_groups=1,
        diagnostic_builder=lambda ex: ex,
    )
    assert report["single_token_label_checks"] == 18
    assert report["ethics_controls"]["ethics_commonsense/test"]["n_rows"] == 8
    assert row == before
