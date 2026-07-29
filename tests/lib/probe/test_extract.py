"""Property tests for the offline probe extraction (specs/02 §7, V5.9-V5.12).

Derived from the spec's extraction-pass contract:

- V5.9  extraction captures n_layers+1 hook points with correct shapes and
  stored mask on a tiny fixture model.
- V5.10 per-example gradients match an explicit one-example-at-a-time backward
  reference (sum reduction ⇒ equality), and are nonzero iff the example's loss
  is nonzero.
- V5.11 dump ↔ load roundtrip preserves values (bf16), names, metadata.
- V5.12 matched-load guard raises on probe_set_hash / tokenizer_hash /
  template mismatch with a clear error.

Fixture models are the shared ``tiny_llama`` (random-init, in-process, CPU);
all references (embedding outputs, logits reconstruction, single-example
backward) are computed through torch/transformers directly, never through the
module under test's own storage path.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from geode.edl.masking import TaskFormat, label_mask
from geode.probe import (
    ProbeMeta,
    extract_probe,
    load_matched_probe_pair,
    load_probe_dump,
    save_probe_dump,
)
from tests.conftest import BOS_ID, TaskExample

MODEL_SEED = 21


def _task_format() -> TaskFormat:
    return TaskFormat(name="copy_token", format_version="1")


def _varied_examples() -> list[TaskExample]:
    """Lengths 4, 3, 5, 3 with varied span starts — pad structure is exercised."""
    return [
        TaskExample(input_ids=[BOS_ID, 5, 6, 7], label_span=(2, 4)),
        TaskExample(input_ids=[BOS_ID, 8, 9], label_span=(2, 3)),
        TaskExample(input_ids=[BOS_ID, 10, 11, 12, 13], label_span=(2, 5)),
        TaskExample(input_ids=[BOS_ID, 14, 15], label_span=(1, 3)),
    ]


def _meta(**overrides) -> ProbeMeta:
    fields = dict(
        run_id="run-a",
        arm="armA_elicit",
        step=17,
        probe_set_hash="ab" * 32,
        tokenizer_hash="cd" * 32,
        base_model_key="zoo-run/evt-run3-armA-inst",
        task_name="copy_token",
        format_version="1",
    )
    fields.update(overrides)
    return ProbeMeta(**fields)


# ---------------------------------------------------------------------------
# V5.9 — n_layers+1 hook points, shapes, dtype, stored masks
# ---------------------------------------------------------------------------


def test_v5_9_hook_points_shapes_dtype_and_mask(tiny_llama):
    """n_layers+1 residual points with correct shapes/dtype, plus the padded
    input_ids, attention mask, and shared M1 label mask stored alongside.

    Independent anchors: ``hook_embed`` must equal the embedding module's own
    output, and final-norm + lm_head applied to the LAST captured residual
    must reproduce the model's logits — so the capture really is the residual
    stream, not some other tensor of the right shape.
    """
    model = tiny_llama(seed=MODEL_SEED, n_layers=2, d_model=64)
    examples = _varied_examples()
    cap = extract_probe(model, examples, _task_format(), device="cpu")

    names = ["hook_embed", "blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    assert list(cap.acts) == names  # n_layers + 1 = 3, embedding output first
    assert list(cap.grads) == names
    for name in names:
        assert cap.acts[name].shape == (4, 5, 64)  # [n, max_len, d_model]
        assert cap.grads[name].shape == (4, 5, 64)
        assert cap.acts[name].dtype == torch.float32  # model compute dtype
        assert cap.grads[name].dtype == torch.float32

    assert cap.input_ids.shape == (4, 5) and cap.input_ids.dtype == torch.long
    assert cap.attention_mask.shape == (4, 5) and cap.attention_mask.dtype == torch.bool
    lengths = [len(ex.input_ids) for ex in examples]
    for i, n_real in enumerate(lengths):
        assert cap.attention_mask[i, :n_real].all() and not cap.attention_mask[i, n_real:].any()
        assert cap.input_ids[i, :n_real].tolist() == examples[i].input_ids
        assert (cap.input_ids[i, n_real:] == 0).all()  # pad id 0, the edl convention
    assert torch.equal(cap.label_mask, label_mask(examples, _task_format()))  # M1 shared path
    assert cap.probe_loss_nats.shape == (4,)
    assert cap.probe_loss_nats.dtype == torch.float32
    assert (cap.probe_loss_nats > 0).all()

    # Anchor 1: the first hook point IS the embedding output.
    with torch.no_grad():
        emb = model.get_input_embeddings()(cap.input_ids)
    assert torch.equal(cap.acts["hook_embed"], emb)

    # Anchor 2: final norm + lm_head over the last residual reproduces logits.
    with torch.no_grad():
        logits_ref = model(input_ids=cap.input_ids).logits
        logits_from_resid = model.lm_head(model.model.norm(cap.acts[names[-1]]))
    assert torch.allclose(logits_from_resid, logits_ref, rtol=1e-5, atol=1e-6)


def test_v5_9_hook_count_tracks_n_layers(tiny_llama):
    """The hook count is n_layers+1, never a hardcoded 9 (spec §7)."""
    model = tiny_llama(seed=MODEL_SEED + 1, n_layers=3, d_model=64)
    cap = extract_probe(model, _varied_examples(), _task_format(), device="cpu")
    assert list(cap.acts) == [
        "hook_embed",
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
    ]
    assert list(cap.grads) == list(cap.acts)


def test_v5_9_label_span_at_position_zero_refused(tiny_llama):
    """The loop's p=0 guard holds at this surface too: a label at position 0
    has no predecessor under the causal shift."""
    model = tiny_llama(seed=MODEL_SEED)
    bad = [TaskExample(input_ids=[5, 6, 7], label_span=(0, 2))]
    with pytest.raises(ValueError, match="position 0"):
        extract_probe(model, bad, _task_format(), device="cpu")


# ---------------------------------------------------------------------------
# V5.10 — batched per-example gradients == one-example-at-a-time reference
# ---------------------------------------------------------------------------


def test_v5_10_batched_grads_match_single_example_reference(tiny_llama):
    """Each example's gradient rows from the batched extraction equal the
    extraction of that example ALONE (sum reduction ⇒ no cross-example terms;
    causal attention + right-padding ⇒ no cross-example ops at all).

    Tolerance calibration (scratchpad, 2026-07-22, this exact fixture): the
    batched-vs-single discrepancy is pure kernel shape-dependence, max |Δ| ≤
    9e-7 across every (example, hook); per-example gradients have max |g| ≥
    1.3 at every hook, and examples' gradients differ from each other at the
    ~3.0 scale — so the 1e-6/1e-5 tolerance is >10^6 below any actual
    gradient-mixing signal. Padding rows must be EXACTLY zero (no loss term
    depends on them), which also fails any implementation that leaks other
    examples' gradients into the pad region.
    """
    model = tiny_llama(seed=MODEL_SEED)
    examples = _varied_examples()
    batched = extract_probe(model, examples, _task_format(), device="cpu")

    for i, ex in enumerate(examples):
        single = extract_probe(model, [ex], _task_format(), device="cpu")
        n_real = len(ex.input_ids)
        assert batched.probe_loss_nats[i].item() == pytest.approx(
            single.probe_loss_nats[0].item(), rel=1e-5, abs=1e-6
        )
        for name in batched.grads:
            ref = single.grads[name][0]
            got = batched.grads[name][i, :n_real]
            assert ref.abs().max() > 1e-3, "reference gradient too small to discriminate"
            assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6), (
                f"example {i}, {name}: batched gradient != single-example reference "
                "(cross-example mixing or padding leakage)"
            )
            pad = batched.grads[name][i, n_real:]
            assert pad.numel() == 0 or pad.abs().max().item() == 0.0, (
                f"example {i}, {name}: nonzero gradient at padding positions"
            )
            # Activations agree too (same forward function batched vs alone).
            assert torch.allclose(
                batched.acts[name][i, :n_real], single.acts[name][0], rtol=1e-5, atol=1e-6
            )


def test_v5_10_grads_nonzero_iff_loss_nonzero(tiny_llama):
    """Per-example gradients are nonzero iff the example's loss is nonzero.

    An empty label span (start == end) gives an exactly-zero loss — its
    gradient rows must be EXACTLY zero at every hook point (sum reduction: no
    loss term touches that example). Every nonzero-loss example must have a
    nonzero gradient at every hook point.
    """
    model = tiny_llama(seed=MODEL_SEED)
    examples = [
        TaskExample(input_ids=[BOS_ID, 5, 6, 7], label_span=(2, 4)),
        TaskExample(input_ids=[BOS_ID, 8, 9], label_span=(2, 2)),  # empty span
        TaskExample(input_ids=[BOS_ID, 10, 11], label_span=(1, 3)),
    ]
    cap = extract_probe(model, examples, _task_format(), device="cpu")

    assert cap.probe_loss_nats[1].item() == 0.0
    assert cap.probe_loss_nats[0].item() > 0 and cap.probe_loss_nats[2].item() > 0
    for name in cap.grads:
        assert cap.grads[name][1].abs().max().item() == 0.0, (
            f"{name}: zero-loss example has nonzero gradient"
        )
        for i in (0, 2):
            assert cap.grads[name][i].abs().sum().item() > 0, (
                f"{name}: nonzero-loss example {i} has all-zero gradient"
            )


# ---------------------------------------------------------------------------
# V5.11 — dump ↔ load roundtrip preserves values (bf16), names, metadata
# ---------------------------------------------------------------------------


def test_v5_11_dump_load_roundtrip(geode_store, tiny_llama):
    """Save writes the spec-§7 layout (acts/grads safetensors + probe data +
    sidecar meta) under ``runs/{run_id}/probe/step_{k}``; load returns exactly
    the stored values: acts/grads in bf16 (bit-equal to the fp32 capture cast
    to bf16), input_ids/masks exact, per-example loss fp32-exact, metadata
    field-for-field identical."""
    model = tiny_llama(seed=MODEL_SEED)
    cap = extract_probe(model, _varied_examples(), _task_format(), device="cpu")
    meta = _meta()

    out = save_probe_dump(cap, meta)
    assert out == geode_store / "runs" / "run-a" / "probe" / "step_17"
    for fname in ("acts.safetensors", "grads.safetensors", "probe_data.safetensors", "meta.json"):
        assert (out / fname).is_file(), f"missing {fname}"

    loaded, loaded_meta = load_probe_dump("run-a", 17)
    assert loaded_meta == meta
    assert loaded_meta.dtype == "bfloat16"  # the spec-§7 storage pin

    assert set(loaded.acts) == set(cap.acts)  # tensor names preserved
    assert set(loaded.grads) == set(cap.grads)
    for name in cap.acts:
        assert loaded.acts[name].dtype == torch.bfloat16
        assert loaded.grads[name].dtype == torch.bfloat16
        assert torch.equal(loaded.acts[name], cap.acts[name].to(torch.bfloat16))
        assert torch.equal(loaded.grads[name], cap.grads[name].to(torch.bfloat16))

    assert torch.equal(loaded.input_ids, cap.input_ids)
    assert loaded.input_ids.dtype == torch.long
    assert torch.equal(loaded.attention_mask, cap.attention_mask)
    assert loaded.attention_mask.dtype == torch.bool
    assert torch.equal(loaded.label_mask, cap.label_mask)
    assert loaded.label_mask.dtype == torch.bool
    assert torch.equal(loaded.probe_loss_nats, cap.probe_loss_nats)  # fp32, never down-cast
    assert loaded.probe_loss_nats.dtype == torch.float32


# ---------------------------------------------------------------------------
# V5.12 — matched-load guard for cross-arm comparison
# ---------------------------------------------------------------------------


def test_v5_12_matched_load_guard(geode_store, tiny_llama):
    """The pairwise loader refuses on probe_set_hash / tokenizer_hash /
    template (task_name, format_version) mismatch, naming the field — and does
    NOT refuse on the fields that legitimately differ across arms and
    snapshots (run_id, arm, step, base_model_key)."""
    model = tiny_llama(seed=MODEL_SEED)
    cap = extract_probe(model, _varied_examples(), _task_format(), device="cpu")

    meta_a = _meta()
    meta_b = _meta(
        run_id="run-b",
        arm="armB_teach",
        step=99,
        base_model_key="zoo-run/evt-run4-armB-inst",
    )
    save_probe_dump(cap, meta_a)
    save_probe_dump(cap, meta_b)

    # Matched identity, differing run/arm/step/base_model_key: must load.
    cap_a, cap_b, got_a, got_b = load_matched_probe_pair("run-a", 17, "run-b", 99)
    assert got_a == meta_a and got_b == meta_b
    assert torch.equal(cap_a.input_ids, cap_b.input_ids)

    mismatches = {
        "probe_set_hash": "ff" * 32,
        "tokenizer_hash": "ee" * 32,
        "task_name": "arith_op_addsub",
        "format_version": "2",
    }
    for field, bad_value in mismatches.items():
        save_probe_dump(cap, dataclasses.replace(meta_b, **{field: bad_value}))
        with pytest.raises(ValueError, match=field):
            load_matched_probe_pair("run-a", 17, "run-b", 99)
