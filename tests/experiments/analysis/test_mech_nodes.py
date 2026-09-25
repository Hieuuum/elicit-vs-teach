"""``mech_nodes.py``: the shared node/edge mechanistic-interpretability
library the ts38mt Tier-2/3 drivers (cross_patch.py / test 4, and the
gated-on-Tier-1 tests 2/3/5) build on.

Silent failure modes guarded (CLAUDE.md's promotion rule — this module is
imported by three other drivers, so its correctness bar is the highest in
the ts38mt tree):

- A wrong head slice or a wrong LoRA/full-FT weight dispatch in the
  residual-write reconstruction would silently corrupt every attribution
  and edge score downstream with no crash — checked against the model's
  own hooked forward output exactly (plain AND LoRA-wrapped trees).
- ``patch_nodes``/``patch_residual`` hooks left installed after an
  exception would corrupt every subsequent call on the same model — tested
  by raising inside the ``with`` block and checking the model is back to
  its clean forward pass.
- A total-gradient (instead of direct-path) bug in the edge-attribution
  gradient hook would make ``Σ_v score(u->v)`` overcount ``node score(u)``
  by roughly the number of downstream consumers — checked as an exact
  first-order identity, not a loose sanity bound (module docstring's
  edge-attribution section explains why the clone-before-retain_grad makes
  this exact rather than approximate).
- A backward-graph bug (e.g. accidentally requiring param grads, or
  cross-example leakage from a batched reduction) would silently give
  wrong or batch-order-dependent per-example gradients — checked against
  finite differences and against re-running a single example alone.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``), no
network, no real checkpoints.
"""

from __future__ import annotations

import copy
import math
import random

import pytest
import torch

from geode.arith.spans import SftExample
from geode.train.lora import apply_lora

from tests._scriptloader import load

mn = load("mech_nodes")
mech = load("mech_lib")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


def _lora_wrap_with_signal(base_model, *, rank=4, alpha=8.0, seed=0):
    """A LoRA-wrapped copy of ``base_model`` with nonzero ``B`` (a fresh
    ``apply_lora`` leaves ``B`` at zero — no adapter signal to distinguish
    the LoRA weight-reconstruction path from the full-FT one)."""
    wrapped = copy.deepcopy(base_model)
    apply_lora(wrapped, rank=rank, alpha=alpha, seed=seed)
    for module in wrapped.modules():
        if hasattr(module, "B") and hasattr(module, "A"):
            with torch.no_grad():
                module.B.weight.uniform_(-0.2, 0.2)
    wrapped.eval()
    return wrapped


def _assert_residual_decomposes(model, ids, mask):
    """resid_post(block i) == resid_pre(block i) + Σ_heads writes + mlp_out,
    checked against the model's own hooked forward output — the shared
    helper the plain-tree and LoRA-tree tests both call."""
    resid = mech.capture_residuals(model, ids, mask)
    names = list(resid.keys())
    acts = mn.capture_nodes(model, ids, mask)
    n_heads, d_head = mn._num_heads_and_dhead(model)
    _, blocks = mn.residual_modules(model)
    for i, block in enumerate(blocks):
        resid_pre = resid[names[i]]
        w, bias = mn._o_proj_weight_bias(mn._o_proj_module(block))
        head_sum = torch.zeros_like(resid_pre)
        for h in range(n_heads):
            a = acts[mn.NodeId(i, "head", h)]
            w_slice = w[:, h * d_head : (h + 1) * d_head]
            head_sum = head_sum + torch.einsum("od,bsd->bso", w_slice, a)
        if bias is not None:
            head_sum = head_sum + bias
        mlp_out = acts[mn.NodeId(i, "mlp", 0)]
        recon = resid_pre + head_sum + mlp_out
        actual = resid[names[i + 1]]
        assert torch.allclose(recon, actual, atol=1e-4), (
            f"layer {i} residual decomposition mismatch"
        )


class TestNodeId:
    def test_str_round_trip_head(self):
        nid = mn.NodeId(3, "head", 2)
        assert str(nid) == "a3.h2"
        assert mn.NodeId.parse("a3.h2") == nid

    def test_str_round_trip_mlp(self):
        nid = mn.NodeId(5, "mlp", 0)
        assert str(nid) == "m5"
        assert mn.NodeId.parse("m5") == nid

    def test_str_round_trip_attn_sink(self):
        nid = mn.NodeId(2, "attn", 0)
        assert str(nid) == "a2"
        assert mn.NodeId.parse("a2") == nid

    def test_parse_rejects_garbage(self):
        for bad in ("", "x1", "a1.h", "m", "a1.h2.x", "h1"):
            with pytest.raises(ValueError):
                mn.NodeId.parse(bad)

    def test_rejects_bad_kind(self):
        with pytest.raises(ValueError):
            mn.NodeId(0, "bogus", 0)

    def test_rejects_negative_layer_or_index(self):
        with pytest.raises(ValueError):
            mn.NodeId(-1, "head", 0)
        with pytest.raises(ValueError):
            mn.NodeId(0, "head", -1)

    def test_rejects_nonzero_index_on_mlp_and_attn(self):
        with pytest.raises(ValueError):
            mn.NodeId(0, "mlp", 1)
        with pytest.raises(ValueError):
            mn.NodeId(0, "attn", 1)

    def test_ordering_heads_before_mlp_same_layer_then_layer_major(self):
        nodes = [
            mn.NodeId(1, "mlp", 0),
            mn.NodeId(0, "head", 3),
            mn.NodeId(0, "mlp", 0),
            mn.NodeId(0, "head", 0),
            mn.NodeId(1, "head", 0),
        ]
        ordered = sorted(nodes)
        assert [str(n) for n in ordered] == ["a0.h0", "a0.h3", "m0", "a1.h0", "m1"]


class TestCaptureNodes:
    def test_shapes_and_count(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=3, d_model=32, vocab_size=64)
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        acts = mn.capture_nodes(model, ids, mask)
        n_heads, d_head = mn._num_heads_and_dhead(model)
        assert len(acts) == 3 * (n_heads + 1)
        for h in range(n_heads):
            assert acts[mn.NodeId(0, "head", h)].shape == (2, 6, d_head)
        assert acts[mn.NodeId(0, "mlp", 0)].shape == (2, 6, 32)
        # keys sorted: layer-major, heads before mlp
        keys = list(acts.keys())
        assert keys == sorted(keys)

    def test_refuses_non_llama_tree(self):
        class NotLlama(torch.nn.Module):
            pass

        with pytest.raises(ValueError, match="Llama"):
            mn.capture_nodes(
                NotLlama(), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool)
            )

    def test_head_dhead_gqa_width_assertion_passes_on_fixture(self, tiny_llama):
        # tiny_llama's num_key_value_heads == num_attention_heads (no GQA),
        # but the assertion must still hold: o_proj.in_features == H*d_head.
        model = tiny_llama(seed=1, n_layers=1, d_model=32, vocab_size=64)
        n_heads, d_head = mn._num_heads_and_dhead(model)
        o_proj = mn._o_proj_module(model.model.layers[0])
        assert o_proj.in_features == n_heads * d_head

    def test_residual_decomposition_plain_model(self, tiny_llama):
        model = tiny_llama(seed=2, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        ids = torch.randint(4, 64, (3, 7))
        mask = torch.ones(3, 7, dtype=torch.bool)
        _assert_residual_decomposes(model, ids, mask)

    def test_residual_decomposition_lora_wrapped_model(self, tiny_llama):
        base = tiny_llama(seed=3, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        wrapped = _lora_wrap_with_signal(base, rank=4, alpha=8.0, seed=7)
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        _assert_residual_decomposes(wrapped, ids, mask)


class TestCaptureResidualInputs:
    def test_pre_block_0_is_embed_output(self, tiny_llama):
        model = tiny_llama(seed=4, n_layers=2, d_model=32, vocab_size=64)
        ids = torch.randint(4, 64, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        pre_block, _pre_mlp = mn.capture_residual_inputs(model, ids, mask)
        resid = mech.capture_residuals(model, ids, mask)
        assert torch.allclose(pre_block[0], resid["hook_embed"], atol=1e-6)

    def test_pre_mlp_equals_pre_block_plus_attn_write(self, tiny_llama):
        model = tiny_llama(seed=5, n_layers=2, d_model=32, vocab_size=64)
        ids = torch.randint(4, 64, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        pre_block, pre_mlp = mn.capture_residual_inputs(model, ids, mask)
        acts = mn.capture_nodes(model, ids, mask)
        n_heads, d_head = mn._num_heads_and_dhead(model)
        _, blocks = mn.residual_modules(model)
        for i, block in enumerate(blocks):
            w, bias = mn._o_proj_weight_bias(mn._o_proj_module(block))
            head_sum = torch.zeros_like(pre_block[i])
            for h in range(n_heads):
                a = acts[mn.NodeId(i, "head", h)]
                w_slice = w[:, h * d_head : (h + 1) * d_head]
                head_sum = head_sum + torch.einsum("od,bsd->bso", w_slice, a)
            if bias is not None:
                head_sum = head_sum + bias
            assert torch.allclose(pre_block[i] + head_sum, pre_mlp[i], atol=1e-4)

    def test_shapes(self, tiny_llama):
        model = tiny_llama(seed=6, n_layers=3, d_model=16, vocab_size=32)
        ids = torch.randint(4, 32, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)
        pre_block, pre_mlp = mn.capture_residual_inputs(model, ids, mask)
        assert set(pre_block) == {0, 1, 2}
        assert set(pre_mlp) == {0, 1, 2}
        for i in range(3):
            assert pre_block[i].shape == (2, 4, 16)
            assert pre_mlp[i].shape == (2, 4, 16)


class TestPatchNodes:
    def test_noop_with_own_clean_activations(self, tiny_llama):
        model = tiny_llama(seed=7, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            base_logits = model(input_ids=ids, attention_mask=mask.long()).logits
        clean = mn.capture_nodes(model, ids, mask)
        with mn.patch_nodes(model, clean), torch.no_grad():
            logits2 = model(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(base_logits, logits2, atol=1e-5)

    def test_hooks_removed_after_exception(self, tiny_llama):
        model = tiny_llama(seed=8, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            base_logits = model(input_ids=ids, attention_mask=mask.long()).logits
        with pytest.raises(RuntimeError, match="boom"):
            with mn.patch_nodes(model, {mn.NodeId(0, "mlp", 0): torch.zeros(2, 6, 32)}):
                raise RuntimeError("boom")
        with torch.no_grad():
            logits_after = model(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(base_logits, logits_after, atol=1e-6)
        # also no lingering hooks on the specific patched module
        o_proj = mn._o_proj_module(model.model.layers[0])
        mlp = mn._mlp_module(model.model.layers[0])
        assert len(o_proj._forward_pre_hooks) == 0
        assert len(mlp._forward_hooks) == 0

    def test_callable_replacement_matches_manual_zero_ablation(self, tiny_llama):
        model = tiny_llama(seed=9, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        clean = mn.capture_nodes(model, ids, mask)
        nid = mn.NodeId(1, "head", 1)
        with mn.patch_nodes(model, {nid: lambda x: torch.zeros_like(x)}), torch.no_grad():
            logits_callable = model(input_ids=ids, attention_mask=mask.long()).logits
        zero_tensor = torch.zeros_like(clean[nid])
        with mn.patch_nodes(model, {nid: zero_tensor}), torch.no_grad():
            logits_tensor = model(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(logits_callable, logits_tensor, atol=1e-6)

    def test_position_restricted_patch_changes_only_that_position(self, tiny_llama):
        # Checked at the PATCHED node/layer itself, not a downstream layer:
        # causal attention means a later block's later positions legitimately
        # change too (they attend back to the patched position) — the
        # invariant _masked_merge actually guarantees is local to the node
        # being patched.
        model = tiny_llama(seed=10, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        ids = torch.randint(4, 64, (3, 6))
        mask = torch.ones(3, 6, dtype=torch.bool)
        clean = mn.capture_nodes(model, ids, mask)
        nid = mn.NodeId(0, "mlp", 0)
        replacement = torch.randn_like(clean[nid])
        pos = torch.tensor([1, 3, 4])
        with mn.patch_nodes(model, {nid: replacement}, positions=pos):
            patched = mn.capture_nodes(model, ids, mask)
        for b in range(3):
            for s in range(6):
                if s == int(pos[b]):
                    assert torch.allclose(patched[nid][b, s], replacement[b, s], atol=1e-6)
                else:
                    assert torch.allclose(patched[nid][b, s], clean[nid][b, s], atol=1e-6), (
                        f"position {s} of example {b} changed despite not being patched"
                    )

    def test_value_dependent_callable_backprops_to_mask_param(self, tiny_llama):
        """Regression (2026-08-21, found by the DCM driver): the head path
        used to write the replacement in-place into a clone AFTER a
        value-dependent callable had read the same slice, so ``backward()``
        raised "modified by an inplace operation". A relaxed-mask mix
        ``m·a_T + (1−m)·a`` through BOTH a head and the mlp of one block must
        backprop to ``m`` with a nonzero gradient and no error."""
        model = tiny_llama(seed=21, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        for p in model.parameters():
            p.requires_grad_(False)
        rng = random.Random(5)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=7, span=(4, 6))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)
        clean = mn.capture_nodes(model, ids, mask)
        head, mlp = mn.NodeId(1, "head", 2), mn.NodeId(1, "mlp", 0)
        a_t = {nid: clean[nid] + torch.randn_like(clean[nid]) for nid in (head, mlp)}
        m = torch.full((2,), 0.3, requires_grad=True)

        def mix(i, nid):
            return lambda a: m[i] * a_t[nid] + (1 - m[i]) * a

        with mn.patch_nodes(model, {head: mix(0, head), mlp: mix(1, mlp)}), torch.enable_grad():
            loss = -mn.answer_logprob(model, ids, mask, pos, targets).mean()
            loss.backward()
        assert m.grad is not None and torch.all(m.grad != 0)
        assert all(p.grad is None for p in model.parameters())

    def test_rejects_attn_kind_node(self, tiny_llama):
        model = tiny_llama(seed=11, n_layers=1, d_model=16, vocab_size=32)
        with pytest.raises(ValueError, match="head.*mlp"):
            with mn.patch_nodes(model, {mn.NodeId(0, "attn", 0): torch.zeros(1, 1, 16)}):
                pass

    def test_rejects_out_of_range_layer(self, tiny_llama):
        model = tiny_llama(seed=12, n_layers=2, d_model=16, vocab_size=32)
        with pytest.raises(ValueError, match="layer"):
            with mn.patch_nodes(model, {mn.NodeId(5, "mlp", 0): torch.zeros(1, 1, 16)}):
                pass


class TestPatchResidual:
    def test_final_layer_reproduces_model_b_logits_exactly(self, tiny_llama):
        model_a = tiny_llama(
            seed=13, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.o_proj.weight.data += 0.01
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            logits_b = model_b(input_ids=ids, attention_mask=mask.long()).logits
        resid_b = mech.capture_residuals(model_b, ids, mask)
        final_layer = 2  # n_layers
        with (
            mn.patch_residual(model_a, final_layer, resid_b["blocks.1.hook_resid_post"]),
            torch.no_grad(),
        ):
            logits_patched = model_a(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(logits_patched, logits_b, atol=1e-5)

    def test_early_layer_reproduces_b_when_only_earlier_block_differs(self, tiny_llama):
        model_a = tiny_llama(
            seed=14, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.o_proj.weight.data += 0.02
        ids = torch.randint(4, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            logits_b = model_b(input_ids=ids, attention_mask=mask.long()).logits
        resid_b = mech.capture_residuals(model_b, ids, mask)
        # layer 1 = block 0's output; blocks 1,2 and norm/head are shared -> patching here reproduces B fully
        with mn.patch_residual(model_a, 1, resid_b["blocks.0.hook_resid_post"]), torch.no_grad():
            logits_patched = model_a(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(logits_patched, logits_b, atol=1e-5)

    def test_layer_0_patches_embed_output(self, tiny_llama):
        # tie_word_embeddings=False here: perturbing the embedding table on a
        # TIED model would also perturb model_b's lm_head (same tensor), and
        # patch_residual only ever touches the residual stream, not weights
        # — with tying, the two models would then no longer "share
        # everything after this point" (the docstring's precondition for
        # exact reproduction), for a reason unrelated to patch_residual
        # itself.
        model_a = tiny_llama(
            seed=15, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=False
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.get_input_embeddings().weight.data += 0.05
        ids = torch.randint(4, 32, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            logits_b = model_b(input_ids=ids, attention_mask=mask.long()).logits
        resid_b = mech.capture_residuals(model_b, ids, mask)
        with mn.patch_residual(model_a, 0, resid_b["hook_embed"]), torch.no_grad():
            logits_patched = model_a(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(logits_patched, logits_b, atol=1e-5)

    def test_hooks_removed_after_exception(self, tiny_llama):
        model = tiny_llama(seed=16, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        ids = torch.randint(4, 32, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            base_logits = model(input_ids=ids, attention_mask=mask.long()).logits
        with pytest.raises(RuntimeError):
            with mn.patch_residual(model, 1, torch.zeros(2, 5, 16)):
                raise RuntimeError("boom")
        with torch.no_grad():
            logits_after = model(input_ids=ids, attention_mask=mask.long()).logits
        assert torch.allclose(base_logits, logits_after, atol=1e-6)

    def test_rejects_out_of_range_layer(self, tiny_llama):
        model = tiny_llama(seed=17, n_layers=2, d_model=16, vocab_size=32)
        with pytest.raises(ValueError):
            with mn.patch_residual(model, 99, torch.zeros(1, 1, 16)):
                pass

    def test_position_restricted_leaves_other_positions_unchanged(self, tiny_llama):
        model_a = tiny_llama(
            seed=18, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.o_proj.weight.data += 0.03
        ids = torch.randint(4, 64, (3, 6))
        mask = torch.ones(3, 6, dtype=torch.bool)
        resid_b = mech.capture_residuals(model_b, ids, mask)
        resid_a_clean = mech.capture_residuals(model_a, ids, mask)
        pos = torch.tensor([2, 3, 4])
        with mn.patch_residual(model_a, 1, resid_b["blocks.0.hook_resid_post"], positions=pos):
            patched = mech.capture_residuals(model_a, ids, mask)
        patched_layer1 = patched["blocks.0.hook_resid_post"]
        clean_layer1 = resid_a_clean["blocks.0.hook_resid_post"]
        b_layer1 = resid_b["blocks.0.hook_resid_post"]
        for b in range(3):
            for s in range(6):
                if s == int(pos[b]):
                    assert torch.allclose(patched_layer1[b, s], b_layer1[b, s], atol=1e-5)
                else:
                    assert torch.allclose(patched_layer1[b, s], clean_layer1[b, s], atol=1e-5)


class TestAnswerLogprob:
    def test_hand_computed(self):
        class _FakeOutput:
            def __init__(self, logits):
                self.logits = logits

        class _FakeModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self._logits = logits
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, attention_mask):
                return _FakeOutput(self._logits)

        # batch of 1, seq len 2, vocab 3; want logprob at position 0 of target 1
        logits = torch.tensor([[[1.0, 5.0, 2.0], [0.0, 0.0, 0.0]]])
        model = _FakeModel(logits)
        positions = torch.tensor([0])
        targets = torch.tensor([1])
        with torch.no_grad():
            out = mn.answer_logprob(
                model,
                torch.zeros(1, 2, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.bool),
                positions,
                targets,
            )
        expected = torch.log_softmax(logits[0, 0].double(), dim=-1)[1]
        assert math.isclose(float(out[0]), float(expected), rel_tol=1e-9)

    def test_matches_logit_lens_first_answer_convention(self, tiny_llama, tiny_tokenizer):
        model = tiny_llama(seed=19, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(1)
        exs = _examples(rng, n=3, vocab_size=32, seq_len=7, span=(4, 6))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)
        with torch.no_grad():
            out = mn.answer_logprob(model, ids, mask, pos, targets)
            logits = model(input_ids=ids, attention_mask=mask.long()).logits
        for i, ex in enumerate(exs):
            p = ex.label_span[0] - 1
            t = ex.input_ids[ex.label_span[0]]
            assert pos[i].item() == p
            assert targets[i].item() == t
            expected = torch.log_softmax(logits[i, p].double(), dim=-1)[t]
            assert math.isclose(float(out[i]), float(expected), rel_tol=1e-6)


class TestAnswerLogprobAndNodeGrads:
    def test_finite_difference_matches_analytic_grad(self, tiny_llama):
        model = tiny_llama(
            seed=20, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        ).double()
        rng = random.Random(2)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=7, span=(4, 6))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)

        metric, grads = mn.answer_logprob_and_node_grads(model, ids, mask, pos, targets)
        assert metric.dtype == torch.float64
        assert metric.shape == (2,)

        # raw fp64 (no fp32 downcast) clean capture, for a clean FD baseline
        raw: dict[tuple[str, int], torch.Tensor] = {}
        handles = []
        _, blocks = mn.residual_modules(model)

        def make_o(i):
            def hook(_m, args):
                raw[("head", i)] = args[0].detach().clone()

            return hook

        for i, block in enumerate(blocks):
            handles.append(mn._o_proj_module(block).register_forward_pre_hook(make_o(i)))
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask.long())
        for h in handles:
            h.remove()

        n_heads, d_head = mn._num_heads_and_dhead(model)

        def metric_fn():
            with torch.no_grad():
                return mn.answer_logprob(model, ids, mask, pos, targets)

        eps = 1e-2
        for nid, head_idx in [(mn.NodeId(0, "head", 1), 1), (mn.NodeId(1, "head", 2), 2)]:
            layer = nid.layer
            a = raw[("head", layer)][..., head_idx * d_head : (head_idx + 1) * d_head]
            b_idx, s_idx, f_idx = 0, int(pos[0].item()), 0
            direction = torch.zeros_like(a)
            direction[b_idx, s_idx, f_idx] = eps
            with mn.patch_nodes(model, {nid: a + direction}):
                m_plus = metric_fn()
            with mn.patch_nodes(model, {nid: a - direction}):
                m_minus = metric_fn()
            fd = float((m_plus[b_idx] - m_minus[b_idx]) / (2 * eps))
            analytic = float(grads[nid][b_idx, s_idx, f_idx])
            assert math.isclose(fd, analytic, rel_tol=5e-2, abs_tol=1e-4), (
                f"{nid}: fd={fd}, analytic={analytic}"
            )

    def test_per_example_independence(self, tiny_llama):
        model = tiny_llama(seed=21, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(3)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=7, span=(4, 6))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)
        _metric_batch, grads_batch = mn.answer_logprob_and_node_grads(
            model, ids, mask, pos, targets
        )

        # example 0 alone
        ids_solo = ids[0:1]
        mask_solo = mask[0:1]
        pos_solo = pos[0:1]
        targets_solo = targets[0:1]
        _metric_solo, grads_solo = mn.answer_logprob_and_node_grads(
            model, ids_solo, mask_solo, pos_solo, targets_solo
        )
        for nid in grads_batch:
            assert torch.allclose(grads_batch[nid][0], grads_solo[nid][0], atol=1e-4), (
                f"{nid}: example 0's grad changed when other batch examples were present"
            )

    def test_params_requires_grad_restored(self, tiny_llama):
        model = tiny_llama(seed=22, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        # mix requires_grad flags to make sure restoration is per-param, not blanket
        params = list(model.parameters())
        for i, p in enumerate(params):
            p.requires_grad_(i % 2 == 0)
        before = [p.requires_grad for p in params]
        rng = random.Random(4)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)
        mn.answer_logprob_and_node_grads(model, ids, mask, pos, targets)
        after = [p.requires_grad for p in params]
        assert before == after

    def test_no_grad_leaks_onto_model_parameters(self, tiny_llama):
        model = tiny_llama(seed=23, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        for p in model.parameters():
            p.grad = None
        rng = random.Random(5)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)
        mn.answer_logprob_and_node_grads(model, ids, mask, pos, targets)
        assert all(p.grad is None for p in model.parameters())


class TestAttributionScores:
    def test_zero_ablation_equals_negative_activation_dot_grad(self, tiny_llama):
        model = tiny_llama(seed=24, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(6)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=7, span=(4, 6))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos, targets = mn.answer_targets(exs)
        clean = mn.capture_nodes(model, ids, mask)
        _metric, grads = mn.answer_logprob_and_node_grads(model, ids, mask, pos, targets)
        scores = mn.attribution_scores(model, exs, ablation="zero", positions="answer")
        for nid in clean:
            expected = -(clean[nid].to(torch.float64) * grads[nid].to(torch.float64)).sum(dim=-1)
            expected = expected.gather(1, pos[:, None]).squeeze(1)
            assert torch.allclose(scores[nid], expected, atol=1e-8), f"{nid} mismatch"

    def test_mean_ablation_of_batch_constant_node_is_zero(self):
        # Pure-function property of _ablated_acts: if a node's clean
        # activation is identical across every example in the batch, its
        # per-position mean over the batch equals that same value, so
        # mean-ablation gives delta=0 (hence score=0) regardless of the
        # gradient — the real property the "constant plant" idea is after,
        # isolated from any live-model/hook mechanics.
        const_vec = torch.randn(1, 5, 8)  # [1, s, d]
        nid = mn.NodeId(0, "mlp", 0)
        clean_acts = {nid: const_vec.expand(4, 5, 8).clone()}  # same value for all 4 examples
        ablated = mn._ablated_acts(clean_acts, "mean", None)
        assert torch.allclose(ablated[nid], clean_acts[nid], atol=1e-12)
        grad = torch.randn(4, 5, 8, dtype=torch.float64)
        delta = (ablated[nid] - clean_acts[nid]).to(torch.float64)
        score = (delta * grad).sum(dim=-1)
        assert torch.allclose(score, torch.zeros_like(score))

    def test_planted_causal_head_dominates_by_far(self, tiny_llama):
        model = tiny_llama(seed=26, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(8)
        exs = _examples(rng, n=5, vocab_size=64, seq_len=8, span=(5, 7))
        _pos, targets = mn.answer_targets(exs)
        embed = model.get_input_embeddings().weight.detach()
        n_heads, d_head = mn._num_heads_and_dhead(model)
        plant_layer, plant_head = 1, 2
        plant_node = mn.NodeId(plant_layer, "head", plant_head)
        block = model.model.layers[plant_layer]
        w, _bias = mn._o_proj_weight_bias(mn._o_proj_module(block))
        w_slice = w[:, plant_head * d_head : (plant_head + 1) * d_head]
        w_pinv = torch.linalg.pinv(w_slice)  # [d_head, d_model]
        huge = 3.0

        def plant_hook(x):
            target_dirs = embed[targets]  # [b, d_model]
            delta_head = torch.einsum("hd,bd->bh", w_pinv, huge * target_dirs)
            return x + delta_head[:, None, :]

        with mn.patch_nodes(model, {plant_node: plant_hook}):
            scores = mn.attribution_scores(model, exs, ablation="zero", positions="answer")
        max_abs = {nid: v.abs().max().item() for nid, v in scores.items()}
        ranked = sorted(max_abs.items(), key=lambda kv: -kv[1])
        assert ranked[0][0] == plant_node
        assert ranked[0][1] > 4.0 * ranked[1][1]

    def test_invalid_ablation_raises(self, tiny_llama):
        model = tiny_llama(seed=27, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(9)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        with pytest.raises(ValueError, match="ablation"):
            mn.attribution_scores(model, exs, ablation="bogus")

    def test_invalid_positions_raises(self, tiny_llama):
        model = tiny_llama(seed=28, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(10)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        with pytest.raises(ValueError, match="positions"):
            mn.attribution_scores(model, exs, positions="bogus")

    def test_empty_batch_raises(self, tiny_llama):
        model = tiny_llama(seed=29, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            mn.attribution_scores(model, [])

    def test_restrict_positions_all_vs_answer_hand_computed(self):
        per_position = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        answer_pos = torch.tensor([1, 2])
        answer_only = mn._restrict_positions(per_position, "answer", answer_pos)
        all_positions = mn._restrict_positions(per_position, "all", answer_pos)
        assert torch.equal(answer_only, torch.tensor([2.0, 6.0]))
        assert torch.equal(all_positions, torch.tensor([6.0, 15.0]))


class TestEdgeAttributionScores:
    def test_upstream_only_no_edge_into_non_downstream_v(self, tiny_llama):
        model = tiny_llama(seed=30, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(11)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=8, span=(5, 7))
        edges = mn.edge_attribution_scores(model, exs, ablation="mean", positions="answer")
        n_layers = 3
        for u, v in edges:
            assert mn._node_pos(u, n_layers) < mn._node_pos(v, n_layers), (
                f"{u} -> {v} is not upstream"
            )

    def test_edge_count_matches_combinatorics(self, tiny_llama):
        model = tiny_llama(seed=31, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(12)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=8, span=(5, 7))
        edges = mn.edge_attribution_scores(model, exs, ablation="mean", positions="answer")
        n_heads, _d_head = mn._num_heads_and_dhead(model)
        n_layers = 3
        expected = 0
        for layer in range(n_layers):
            for kind in ("head", "mlp"):
                u_pos = 2 * layer if kind == "head" else 2 * layer + 1
                n_downstream = sum(1 for p in range(2 * n_layers + 1) if p > u_pos)
                count = n_heads if kind == "head" else 1
                expected += count * n_downstream
        assert len(edges) == expected

    def test_sum_over_v_equals_node_score_first_order_identity(self, tiny_llama):
        model = tiny_llama(seed=32, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(13)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=9, span=(5, 8))
        node_scores = mn.attribution_scores(model, exs, ablation="mean", positions="answer")
        edge_scores = mn.edge_attribution_scores(model, exs, ablation="mean", positions="answer")
        sums: dict = {}
        for (u, _v), s in edge_scores.items():
            sums[u] = sums.get(u, torch.zeros_like(s)) + s
        for u, node_score in node_scores.items():
            edge_sum = sums[u]
            assert torch.allclose(edge_sum, node_score, rtol=1e-3, atol=1e-6), (
                f"{u}: edge sum {edge_sum} != node score {node_score}"
            )

    def test_sum_over_v_identity_holds_for_positions_all_too(self, tiny_llama):
        model = tiny_llama(seed=33, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(14)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=8, span=(4, 7))
        node_scores = mn.attribution_scores(model, exs, ablation="zero", positions="all")
        edge_scores = mn.edge_attribution_scores(model, exs, ablation="zero", positions="all")
        sums: dict = {}
        for (u, _v), s in edge_scores.items():
            sums[u] = sums.get(u, torch.zeros_like(s)) + s
        for u, node_score in node_scores.items():
            assert torch.allclose(sums[u], node_score, rtol=1e-3, atol=1e-6)

    def test_empty_batch_raises(self, tiny_llama):
        model = tiny_llama(seed=34, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            mn.edge_attribution_scores(model, [])


class TestNodePos:
    def test_attention_upstream_of_own_block_mlp(self):
        u = mn.NodeId(2, "attn", 0)
        v = mn.NodeId(2, "mlp", 0)
        assert mn._node_pos(u, 5) < mn._node_pos(v, 5)

    def test_mlp_not_upstream_of_own_block_attention(self):
        u = mn.NodeId(2, "mlp", 0)
        v = mn.NodeId(2, "attn", 0)
        assert mn._node_pos(u, 5) >= mn._node_pos(v, 5)

    def test_out_is_downstream_of_everything(self):
        n_layers = 4
        for layer in range(n_layers):
            for kind in ("attn", "mlp"):
                assert mn._node_pos(mn.NodeId(layer, kind, 0), n_layers) < mn._node_pos(
                    "out", n_layers
                )

    def test_rejects_unknown_sink_string(self):
        with pytest.raises(ValueError):
            mn._node_pos("bogus", 3)
