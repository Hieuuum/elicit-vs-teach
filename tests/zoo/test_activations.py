"""ZOO-3 tests: activation store with matched-input enforcement.

Derived from specs/00-interfaces.md — §1 "Directory layout", §6 "Activation
storage" — validation property V0.4, and resolved decisions (specs/00 §9) OQ-5
(position policy is exactly the enum ``{all, answer_only, last}``) and OQ-7
(``tokenizer_hash`` = sha256 hex digest of the canonical tokenizer JSON;
deterministic across instances, sensitive to vocab changes).

Written test-first: the imports below target the public API fixed in the retired build plan (git history)
"### ZOO-3" (module ``geode.zoo.activations``) and are expected to fail until
ZOO-3 is implemented.

Contract note: spec V0.4 demands "a clear error" but names no exception class.
These tests pin ``ValueError`` as the refusal contract for matched-input
mismatches; the implementer must conform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from geode.zoo.activations import (
    ActivationMeta,
    load_activations,
    load_matched_pair,
    save_activations,
    tokenizer_hash,
)

# ---------------------------------------------------------------------------
# Shared constants and helpers (spec 00 §1 layout, §6 storage schema).
# ---------------------------------------------------------------------------

MODEL_A = "model_a"
MODEL_B = "model_b"
DATASET = "ds_x"
# §6: tensor name == hook name (TransformerLens convention). Contains dots.
HOOK = "blocks.1.hook_resid_post"

# OQ-7: tokenizer_hash is a sha256 hex digest — 64 hex chars. Sentinels here.
TOK_HASH_A = "aa" * 32
TOK_HASH_B = "bb" * 32

# §6: shape [n_samples, n_positions_kept, d_model].
N_SAMPLES = 4
N_POS = 3
D_MODEL = 8

# Exact sidecar schema: one JSON key per ActivationMeta field.
META_KEYS = {
    "model_key",
    "hf_id",
    "revision",
    "dataset_key",
    "position_policy",
    "n_samples",
    "tokenizer_hash",
}


def _meta(
    *,
    model_key: str = MODEL_A,
    hf_id: str = "org/tiny-model",
    revision: str = "main",
    dataset_key: str = DATASET,
    position_policy: str = "all",
    tok_hash: str = TOK_HASH_A,
) -> ActivationMeta:
    """A fully populated ActivationMeta; override only the fields under test."""
    return ActivationMeta(
        model_key=model_key,
        hf_id=hf_id,
        revision=revision,
        dataset_key=dataset_key,
        position_policy=position_policy,
        n_samples=N_SAMPLES,
        tokenizer_hash=tok_hash,
    )


def _acts(seed: int) -> torch.Tensor:
    """A small seeded float32 activation tensor of shape [N_SAMPLES, N_POS, D_MODEL]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(N_SAMPLES, N_POS, D_MODEL, generator=gen)


def _save(meta: ActivationMeta, store: Path, *, seed: int = 0, hook: str = HOOK) -> Path:
    """Save a seeded tensor for ``meta`` under ``store``; return the written path."""
    return save_activations(_acts(seed), meta, hook, store=store)


def _sidecar(store: Path, model_key: str, dataset_key: str, hook: str = HOOK) -> Path:
    """The §1 sidecar path: activations/{model}/{dataset}/{hook}.meta.json."""
    return store / "activations" / model_key / dataset_key / f"{hook}.meta.json"


def _tamper_sidecar(
    store: Path, model_key: str, path_dataset_key: str, **overrides: object
) -> None:
    """Rewrite a stored sidecar so its recorded provenance disagrees with the path."""
    path = _sidecar(store, model_key, path_dataset_key)
    data = json.loads(path.read_text())
    data.update(overrides)
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# §1 + §6 — save/load roundtrip: layout, tensor name, shape, dtype, values.
# ---------------------------------------------------------------------------


def test_roundtrip_shape_dtype_name(geode_store: Path) -> None:
    """save then load preserves shape/dtype; the tensor is stored under the hook
    name, at the spec §1 path, downcast to float16 by default, and the values
    round-trip exactly (via the float16 cast)."""
    meta = _meta()
    acts = _acts(seed=0)  # float32 input
    path = save_activations(acts, meta, HOOK)  # store resolved from $GEODE_STORE

    # §1: activations/{model_key}/{dataset_key}/{hook_name}.safetensors, sidecar beside it.
    expected = geode_store / "activations" / MODEL_A / DATASET / f"{HOOK}.safetensors"
    assert path == expected
    assert expected.exists()
    assert _sidecar(geode_store, MODEL_A, DATASET).exists()

    # §6: tensor name inside the file == hook name; dtype float16 by default.
    on_disk = load_file(path)
    assert set(on_disk) == {HOOK}
    stored = on_disk[HOOK]
    assert stored.shape == (N_SAMPLES, N_POS, D_MODEL)
    assert stored.dtype == torch.float16

    # load_activations returns (tensor, meta): shape/dtype/values/meta round-trip.
    loaded, loaded_meta = load_activations(MODEL_A, DATASET, HOOK)
    assert loaded.shape == (N_SAMPLES, N_POS, D_MODEL)
    assert loaded.dtype == torch.float16
    assert torch.equal(loaded, acts.to(torch.float16))
    assert loaded_meta == meta


def test_sidecar_meta_roundtrip(geode_store: Path) -> None:
    """Every ActivationMeta field survives save/load exactly, on disk and via the loader."""
    meta = ActivationMeta(
        model_key="mdl-key",
        hf_id="org/some-model",
        revision="abc123",
        dataset_key="dataset-key",
        position_policy="answer_only",  # OQ-5: a non-default enum member
        n_samples=N_SAMPLES,
        tokenizer_hash=TOK_HASH_A,
    )
    path = _save(meta, geode_store)

    # On disk: exactly the ActivationMeta field set, with values preserved.
    on_disk = json.loads((path.parent / f"{HOOK}.meta.json").read_text())
    assert set(on_disk) == META_KEYS
    assert on_disk == {
        "model_key": "mdl-key",
        "hf_id": "org/some-model",
        "revision": "abc123",
        "dataset_key": "dataset-key",
        "position_policy": "answer_only",
        "n_samples": N_SAMPLES,
        "tokenizer_hash": TOK_HASH_A,
    }

    _, loaded_meta = load_activations("mdl-key", "dataset-key", HOOK)
    assert loaded_meta == meta
    assert type(loaded_meta.n_samples) is int
    assert loaded_meta.position_policy == "answer_only"


# ---------------------------------------------------------------------------
# V0.4 — matched-input loader refuses comparisons across unmatched provenance.
# ---------------------------------------------------------------------------


def test_matched_pair_dataset_key_mismatch_raises(geode_store: Path) -> None:
    """V0.4: when a model's recorded dataset_key disagrees with the storage
    path (inconsistent provenance), the loader must refuse rather than trust
    the path — comparisons across models require identical dataset_key."""
    _save(_meta(model_key=MODEL_A), geode_store, seed=1)
    _save(_meta(model_key=MODEL_B), geode_store, seed=2)
    # Model A's sidecar now claims a different dataset than its storage path.
    _tamper_sidecar(geode_store, MODEL_A, DATASET, dataset_key="ds_other")

    with pytest.raises(ValueError):
        load_matched_pair(MODEL_A, MODEL_B, DATASET, HOOK, store=geode_store)


def test_matched_pair_position_policy_mismatch_raises(geode_store: Path) -> None:
    """V0.4: differing position policy (only) between the two models is refused."""
    _save(_meta(model_key=MODEL_A, position_policy="all"), geode_store, seed=1)
    _save(_meta(model_key=MODEL_B, position_policy="last"), geode_store, seed=2)

    with pytest.raises(ValueError):
        load_matched_pair(MODEL_A, MODEL_B, DATASET, HOOK, store=geode_store)


def test_matched_pair_tokenizer_hash_mismatch_raises(geode_store: Path) -> None:
    """V0.4: differing tokenizer hash (only) between the two models is refused."""
    _save(_meta(model_key=MODEL_A, tok_hash=TOK_HASH_A), geode_store, seed=1)
    _save(_meta(model_key=MODEL_B, tok_hash=TOK_HASH_B), geode_store, seed=2)

    with pytest.raises(ValueError):
        load_matched_pair(MODEL_A, MODEL_B, DATASET, HOOK, store=geode_store)


def test_matched_pair_error_message_is_clear(geode_store: Path) -> None:
    """V0.4: spec §6 demands "a clear error". The message must name the
    mismatched field and both differing values so it is debuggable at
    GPU-run time (contract pinned by the tests; the implementer must conform)."""
    _save(_meta(model_key=MODEL_A, tok_hash=TOK_HASH_A), geode_store, seed=1)
    _save(_meta(model_key=MODEL_B, tok_hash=TOK_HASH_B), geode_store, seed=2)

    with pytest.raises(ValueError) as excinfo:
        load_matched_pair(MODEL_A, MODEL_B, DATASET, HOOK, store=geode_store)
    msg = str(excinfo.value)
    assert "tokenizer_hash" in msg
    assert TOK_HASH_A in msg
    assert TOK_HASH_B in msg


def test_matched_pair_row_count_mismatch_raises(geode_store: Path) -> None:
    """Identical dataset_key/position_policy/tokenizer_hash but a differing
    row count between the two stored tensors is refused — an undetected
    count mismatch would silently misalign every downstream matched-pair
    comparison. The message names both model keys and both counts so the
    mismatch is debuggable at GPU-run time."""
    _save(_meta(model_key=MODEL_A), geode_store, seed=1)  # N_SAMPLES rows
    short = torch.randn(
        N_SAMPLES - 1, N_POS, D_MODEL, generator=torch.Generator().manual_seed(2)
    )
    save_activations(short, _meta(model_key=MODEL_B), HOOK, store=geode_store)

    with pytest.raises(ValueError) as excinfo:
        load_matched_pair(MODEL_A, MODEL_B, DATASET, HOOK, store=geode_store)
    msg = str(excinfo.value)
    assert MODEL_A in msg
    assert MODEL_B in msg
    assert str(N_SAMPLES) in msg
    assert str(N_SAMPLES - 1) in msg


def test_matched_pair_matched_returns_both(geode_store: Path) -> None:
    """Positive V0.4 contract: with identical dataset_key/position_policy/
    tokenizer_hash, the loader returns both tensors — in (a, b) order — and
    the shared provenance. hf_id/revision differ deliberately: real matched
    pairs are different models, and §6 gates only on the three matched-input
    fields, so a gate demanding whole-meta equality must fail here."""
    _save(
        _meta(model_key=MODEL_A, hf_id="org/model-a", revision="rev-a"),
        geode_store,
        seed=1,
    )
    _save(
        _meta(model_key=MODEL_B, hf_id="org/model-b", revision="rev-b"),
        geode_store,
        seed=2,
    )

    acts_a, acts_b, meta = load_matched_pair(MODEL_A, MODEL_B, DATASET, HOOK, store=geode_store)

    assert acts_a.dtype == torch.float16
    assert acts_b.dtype == torch.float16
    # Exact values pin the (a, b) return order: slot a is model A's saved
    # tensor (seed 1), slot b is model B's (seed 2), both float16 on disk —
    # a swapped (acts_b, acts_a) return must fail.
    assert torch.equal(acts_a, _acts(1).to(torch.float16))
    assert torch.equal(acts_b, _acts(2).to(torch.float16))
    # The returned meta carries the shared matched-input provenance (§6).
    assert meta.dataset_key == DATASET
    assert meta.position_policy == "all"
    assert meta.tokenizer_hash == TOK_HASH_A


# ---------------------------------------------------------------------------
# OQ-7 — tokenizer_hash is deterministic across instances and vocab-sensitive.
# ---------------------------------------------------------------------------


def _wordlevel_tokenizer(words: list[str]) -> PreTrainedTokenizerFast:
    """An in-process word-level tokenizer over ``words`` (conftest pattern)."""
    specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    vocab = {tok: i for i, tok in enumerate([*specials, *words])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


def test_tokenizer_hash_deterministic_and_sensitive(tiny_tokenizer) -> None:
    """Same construction ⇒ identical hash across two separately-built
    instances; a different vocab ⇒ a different hash (OQ-7)."""
    h1 = tokenizer_hash(tiny_tokenizer())
    h2 = tokenizer_hash(tiny_tokenizer())
    assert h1 == h2

    # sha256 hex digest (OQ-7).
    assert isinstance(h1, str)
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)

    # Sensitivity: a smaller vocabulary changes the canonical tokenizer JSON.
    h_small = tokenizer_hash(tiny_tokenizer(vocab_size=64))
    assert h_small != h1

    # Content sensitivity at equal vocab size (OQ-7: the hash covers the
    # canonical tokenizer JSON, not just the vocab size): renaming a single
    # word token must change the hash even though both vocabs have the same
    # size — a hash of vocab_size alone must fail here.
    words = [f"t{i}" for i in range(8)]
    renamed = ["z0", *words[1:]]
    h_words = tokenizer_hash(_wordlevel_tokenizer(words))
    h_renamed = tokenizer_hash(_wordlevel_tokenizer(renamed))
    assert h_words != h_renamed
