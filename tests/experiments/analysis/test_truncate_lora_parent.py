"""``truncate_lora_parent.py`` — layer-truncated-adapter parent builder for
the ts38tr positive-control family.

Silent failure modes guarded: a wrong block index in ``zero_lora_blocks``
(off-by-one on ``i >= keep_blocks``) would silently move the truncation
boundary by one layer, corrupting the entire premise of the positive
control (the whole point is that block K-1 IS the last one carrying the
trained adapter). A wrong ``merge_lora`` interaction with a zeroed ``B``
would silently perturb the "untrained" blocks instead of reproducing the
base function exactly. A manifest that fails to validate, or validates with
the wrong ``training.method``, would only be discovered deep inside
``register_run`` on a real launch (the same failure class
``test_config_completeness.py``'s own module docstring calls out for
``manifest_fields``). Every property below is checked against a value known
by construction (hand-computed weights, a deep-copied reference model, or a
plain ``merge_lora``/base-model comparison), not just "did not crash".

Loaded via ``tests._scriptloader.load`` (supports the ``experiments/
training-run/scripts`` dir, where this script lives, in addition to
``analysis/``). CPU-only, tiny random-init fixtures (conftest's
``tiny_llama``), no network, no real checkpoints.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from geode.train.lora import LoRALinear, apply_lora, merge_lora
from geode.zoo import RunManifest, load_model, load_run, register_run
from tests._scriptloader import load
from tests.lib.zoo.test_manifest import make_manifest

tlp = load("truncate_lora_parent")
mech_lib = load("mech_lib")

RANK = 2
ALPHA = 4.0
TARGETS = ("q_proj", "v_proj")


def _perturb_adapters(model: torch.nn.Module, seed: int) -> None:
    """Give every LoRA A/B nonzero values, as training would (V0.9 precedent,
    test_model_io.py's own ``perturb_adapters``): a zero B leaves every
    property below trivially true regardless of whether the code is correct."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.A.weight.copy_(torch.randn(module.A.weight.shape, generator=g))
                module.B.weight.copy_(torch.randn(module.B.weight.shape, generator=g))


def _lora_linears(block: torch.nn.Module) -> dict[str, LoRALinear]:
    return {name: m for name, m in block.named_modules() if isinstance(m, LoRALinear)}


# --------------------------------------------------------------------------
# Property 1: per-block weight identity after zero + merge
# --------------------------------------------------------------------------


def test_property1_blocks_below_k_get_delta_blocks_at_or_above_k_are_base(tiny_llama):
    keep_blocks = 2
    model = tiny_llama(seed=0, n_layers=4, d_model=32, vocab_size=64)
    apply_lora(model, rank=RANK, alpha=ALPHA, seed=1, target_modules=TARGETS)
    _perturb_adapters(model, seed=2)

    # Capture ground truth BEFORE any mutation (zero_lora_blocks/merge_lora
    # both mutate in place) — the reference values a correct implementation
    # must reproduce, not a copy of whatever the code under test produced.
    expected: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for i, block in enumerate(model.model.layers):
        for name, module in _lora_linears(block).items():
            expected[(i, name)] = {
                "base": module.base.weight.detach().clone(),
                "delta": (
                    module.scaling * (module.B.weight.detach() @ module.A.weight.detach())
                ).clone(),
            }
    assert len(expected) == 4 * len(TARGETS)  # sanity: 4 layers x 2 targets

    counts = tlp.zero_lora_blocks(model, keep_blocks=keep_blocks)
    assert counts == {"zeroed": 2 * len(TARGETS), "kept": 2 * len(TARGETS)}
    merge_lora(model)

    for (i, name), vals in expected.items():
        merged = model.model.layers[i].get_submodule(name)
        assert isinstance(merged, torch.nn.Linear) and not isinstance(merged, LoRALinear)
        if i < keep_blocks:
            torch.testing.assert_close(merged.weight.detach(), vals["base"] + vals["delta"])
        else:
            assert torch.equal(merged.weight.detach(), vals["base"]), (i, name)


# --------------------------------------------------------------------------
# Property 2: residual-stream prefix identical, final logits diverge
# --------------------------------------------------------------------------


def test_property2_residual_prefix_matches_final_logits_diverge(tiny_llama):
    keep_blocks = 2
    n_layers = 4
    model_full = tiny_llama(
        seed=10, n_layers=n_layers, d_model=32, vocab_size=64, tie_word_embeddings=False
    )
    apply_lora(model_full, rank=RANK, alpha=ALPHA, seed=11, target_modules=TARGETS)
    _perturb_adapters(model_full, seed=12)
    model_full.eval()

    model_trunc = copy.deepcopy(model_full)  # independent tensors: mutation-safe
    tlp.zero_lora_blocks(model_trunc, keep_blocks=keep_blocks)
    merge_lora(model_trunc)
    model_trunc.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(4, 64, (2, 6))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    acts_full = mech_lib.capture_residuals(model_full, input_ids, attention_mask)
    acts_trunc = mech_lib.capture_residuals(model_trunc, input_ids, attention_mask)

    # mech_lib's own layer convention: hook_embed = layer 0,
    # blocks.{i}.hook_resid_post = layer i+1 — so "residual after block K-1"
    # is blocks.{K-1}.hook_resid_post, i.e. layer K.
    layer_name = f"blocks.{keep_blocks - 1}.hook_resid_post"
    torch.testing.assert_close(acts_trunc[layer_name], acts_full[layer_name])

    with torch.no_grad():
        logits_full = model_full(input_ids=input_ids, attention_mask=attention_mask.long()).logits
        logits_trunc = model_trunc(input_ids=input_ids, attention_mask=attention_mask.long()).logits
    # Same tolerance as the "identical" check above, so a passing "differ"
    # assertion can't be an artifact of a looser bound.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(logits_trunc, logits_full)


# --------------------------------------------------------------------------
# Property 3: the two boundary cases
# --------------------------------------------------------------------------


def test_property3a_keep_blocks_equals_n_layers_is_a_plain_full_merge(tiny_llama):
    n_layers = 3
    model_a = tiny_llama(seed=20, n_layers=n_layers, d_model=32, vocab_size=64)
    apply_lora(model_a, rank=RANK, alpha=ALPHA, seed=21, target_modules=TARGETS)
    _perturb_adapters(model_a, seed=22)
    model_b = copy.deepcopy(model_a)

    counts = tlp.zero_lora_blocks(model_a, keep_blocks=n_layers)
    assert counts == {"zeroed": 0, "kept": n_layers * len(TARGETS)}
    merge_lora(model_a)  # truncated path, but nothing was zeroed
    merge_lora(model_b)  # plain full merge, no truncation step at all

    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), name_a


def test_property3b_keep_blocks_zero_reproduces_the_base_model(tiny_llama):
    n_layers = 3
    base = tiny_llama(seed=30, n_layers=n_layers, d_model=32, vocab_size=64)
    trunc = copy.deepcopy(base)
    apply_lora(trunc, rank=RANK, alpha=ALPHA, seed=31, target_modules=TARGETS)
    _perturb_adapters(trunc, seed=32)

    counts = tlp.zero_lora_blocks(trunc, keep_blocks=0)
    assert counts == {"zeroed": n_layers * len(TARGETS), "kept": 0}
    merge_lora(trunc)

    torch.manual_seed(0)
    ids = torch.randint(4, 64, (2, 5))
    with torch.no_grad():
        logits_base = base(input_ids=ids).logits
        logits_trunc = trunc(input_ids=ids).logits
    assert torch.equal(logits_base, logits_trunc)


# --------------------------------------------------------------------------
# Property 4: out-of-range keep_blocks refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keep_blocks", [-1, 4])
def test_property4_keep_blocks_out_of_range_refuses(tiny_llama, keep_blocks):
    model = tiny_llama(
        seed=40, n_layers=3, d_model=32, vocab_size=64
    )  # 3 layers: 4 is out of range
    apply_lora(model, rank=RANK, alpha=ALPHA, seed=41, target_modules=TARGETS)
    with pytest.raises(ValueError, match="keep_blocks"):
        tlp.zero_lora_blocks(model, keep_blocks=keep_blocks)


def test_zero_lora_blocks_on_an_unwrapped_model_is_a_silent_zero_count(tiny_llama):
    """``zero_lora_blocks`` does not itself enforce its "must be apply_lora-
    wrapped" precondition -- an unwrapped model has no LoRALinear anywhere,
    so it returns {"zeroed": 0, "kept": 0} rather than raising. That's fine
    because ``main()`` refuses upstream on training.method != "lora" before
    this ever runs, and the immediately following ``merge_lora`` call DOES
    raise on an unwrapped model ("no LoRALinear modules to merge") -- so a
    direct caller still gets a loud failure, just one function later. Pinned
    here so this division of labor is a documented choice, not an accident."""
    model = tiny_llama(seed=45, n_layers=3, d_model=32, vocab_size=64)  # never apply_lora'd
    assert tlp.zero_lora_blocks(model, keep_blocks=1) == {"zeroed": 0, "kept": 0}
    with pytest.raises(ValueError, match="no LoRALinear"):
        merge_lora(model)


# --------------------------------------------------------------------------
# Property 5: the manifest itself
# --------------------------------------------------------------------------


def test_property5_manifest_validates_full_ft_no_gates_has_truncation():
    source = {
        "run_id": "evt-ts38mt-base-n316228",
        "task": {"name": "arith_bare_addsub", "format_version": "v1"},
        "dataset": {
            "name": "mhieuuu/elicit-vs-teach-arith:D_algo_bare.parquet",
            "n_unique_examples": 316228,
            "seed": 316,
        },
    }
    counts = {"zeroed": 6, "kept": 10}

    fields = tlp.truncated_parent_manifest(
        source, "evt-ts38tr-k7-parent", 7, 8, counts, n_params=38_000_000
    )

    RunManifest(data=fields).validate()  # must not raise
    assert fields["training"]["method"] == "full_ft"
    assert fields["status"] == "complete"
    assert fields["experiment"]["gates"] == {}
    assert fields["base_model"] == {"hf_id": "zoo-run/evt-ts38mt-base-n316228", "revision": "none"}
    assert fields["trainable_param_count"] == 38_000_000
    assert fields["task"] == source["task"]
    assert fields["dataset"] == source["dataset"]
    assert fields["truncation"] == {
        "source_run_id": "evt-ts38mt-base-n316228",
        "keep_blocks": 7,
        "n_blocks": 8,
        "zeroed_modules": 6,
        "kept_modules": 10,
        "method": "zero_B_then_merge",
    }


# --------------------------------------------------------------------------
# Property 6: end-to-end main() over a fake zoo LoRA source run
# --------------------------------------------------------------------------


def _write_source_run(store: Path, run_id: str, model: torch.nn.Module) -> None:
    d = store / "runs" / run_id
    d.mkdir(parents=True)
    model.save_pretrained(str(d / "model"))
    fields = make_manifest(run_id)
    fields["training"]["method"] = "lora"
    fields["training"]["lora"].update(
        rank=RANK, alpha=ALPHA, target_modules=list(TARGETS), dropout=0.0
    )
    register_run(fields, store=store)


def test_property6_end_to_end_main_over_fake_lora_run(tiny_llama, tmp_path, monkeypatch):
    import sys

    store = tmp_path / "store"
    source_id, out_id = "evt-test-src", "evt-test-truncated-parent"
    keep_blocks = 2

    source_model = tiny_llama(
        seed=50, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=False
    )
    apply_lora(source_model, rank=RANK, alpha=ALPHA, seed=51, target_modules=TARGETS)
    _perturb_adapters(source_model, seed=52)
    _write_source_run(store, source_id, source_model)

    # Independent "expected" computation over a fresh reload of the SAME
    # saved weights — not the in-memory source_model object, so this also
    # exercises the reapply_lora round trip main() itself goes through.
    expected = load_model(source_id, store=store, device="cpu")
    tlp.zero_lora_blocks(expected, keep_blocks=keep_blocks)
    merge_lora(expected)

    argv = [
        "truncate_lora_parent.py",
        "--source-run-id",
        source_id,
        "--keep-blocks",
        str(keep_blocks),
        "--out-run-id",
        out_id,
        "--store",
        str(store),
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert tlp.main() == 0

    manifest = load_run(out_id, store=store)
    assert manifest.data["training"]["method"] == "full_ft"
    assert manifest.data["status"] == "complete"

    loaded = load_model(out_id, store=store, device="cpu")  # exercises the full_ft load path
    torch.manual_seed(1)
    ids = torch.randint(4, 64, (2, 5))
    with torch.no_grad():
        logits_expected = expected(input_ids=ids).logits
        logits_loaded = loaded(input_ids=ids).logits
    torch.testing.assert_close(logits_loaded, logits_expected)

    # The contract also names mech_lib's `dir:` loader and train_target.py's
    # `--init-from` (both go through plain from_pretrained on this same
    # directory) — one extra load covers both call sites.
    model_dir = store / "runs" / out_id / "model"
    via_dir = mech_lib.load_any_model(f"dir:{model_dir}", device="cpu")
    with torch.no_grad():
        logits_via_dir = via_dir(input_ids=ids).logits
    torch.testing.assert_close(logits_via_dir, logits_expected)

    # Re-running refuses: the output run_dir already exists.
    assert tlp.main() == 1
