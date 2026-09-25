"""``dcm.py`` (Phase-0/Tier-3 test 5, gated on Tier 1): Desiderata-based
Component Masking, adapted cross-model -- learn a sparse mask over
(head, mlp) nodes such that patching ONLY the masked nodes' activations
from θ_T into θ0 recovers θ_T's answer.

Silent failure modes guarded:

- The relaxed mask's gradient flows through ``mech_nodes.patch_nodes``'s
  functional per-head rebuild (its original in-place write broke autograd
  for value-dependent callables — fixed 2026-08-21, pinned in
  ``test_mech_nodes.py``). A regression there would either crash every fit
  with an autograd in-place error or, worse, silently give WRONG gradients --
  checked against finite differences on a live model, and against an exact
  known-answer planted scenario.
- A sign/anchor bug in ``recovery_frac`` (mirrors ``cross_patch.py``'s own
  guard) would silently flip "recovered" and "damaged" readings -- checked
  against hand-computed values and the exact 0.0/1.0 bounds a genuinely
  isolated single-node perturbation must produce.
- A gradient leak onto model parameters (instead of only the mask logits)
  would silently corrupt every subsequent forward pass on a shared model
  object -- checked by asserting every parameter's ``.grad`` stays
  ``None`` after a fit.
- Hooks left installed after an exception in ``fit_mask``/``evaluate_mask``
  would corrupt every later call on the same model -- checked by raising
  mid-fit and confirming the model's plain forward pass is unaffected and
  no hooks remain on the patched modules.
- A binarisation off-by-one (``>`` vs ``>=`` at the threshold, or reading
  the wrong sign of the mask value) would silently mis-select nodes with
  no crash -- checked with a hand-picked boundary value.
- A row-level bookkeeping bug (the "fit" row's ``n_selected`` disagreeing
  with a hand-count over the "node" rows for the same λ) would silently
  desync the two output row levels -- checked by cross-checking them.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``), no
network, no real checkpoints. Kept fast: the heaviest fit-loop tests use a
2-3 layer, d_model=32 model with <= 80 steps.
"""

from __future__ import annotations

import copy
import math
import random
import sys

import pandas as pd
import pytest
import torch

from geode.arith.spans import SftExample

from tests._scriptloader import load

# Load order matters: dcm.py does ``from mech_nodes import NodeId`` at
# import time. If ``dcm`` were loaded FIRST, that bare import would run
# BEFORE this file's own ``load("mech_nodes")`` registers the
# scriptloader's module object in ``sys.modules``, so Python's normal
# import machinery would create a SEPARATE ``mech_nodes`` module (and a
# SEPARATE ``NodeId`` class) transitively -- every ``mn.NodeId(...)``
# built in this file would then silently fail ``==``/``in`` checks against
# the ``NodeId`` instances ``dcm``'s own functions return, despite
# identical field values and identical ``str()`` (this was hit and fixed
# while writing these tests: dict/``in`` lookups raised ``KeyError`` on a
# node whose string form was plainly present). Loading ``mech_nodes``
# (and, for the same reason, ``mech_lib``) before ``dcm`` makes ``dcm``'s
# own bare imports resolve to these exact module objects instead.
mn = load("mech_nodes")
mech = load("mech_lib")
dcm = load("dcm")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


def _plant_last_mlp_shift(tiny_llama, *, seed, n_layers, d_model, vocab_size, huge=10.0):
    """``(model_a, model_b, plant_nid, examples)``: ``model_b`` is a deep
    copy of ``model_a`` with its LAST block's mlp forward permanently
    monkey-patched to add ``huge * embed[target]`` to its output at every
    position. Because it is the LAST block, the mlp's write is the entire
    remaining residual delta (no later block to propagate through, and
    nothing else about model_b differs), so patching model_a's own last-mlp
    node with model_b's captured activation there reproduces model_b's
    logits EXACTLY -- the "genuinely isolated single node" construction
    used by every exact-bound test below (mirrors ``cross_patch.py``'s
    ``_perturbed_pair``, but at node granularity via a functional forward
    override rather than a weight delta, so ONLY that one node's captured
    activation differs -- a real weight perturbation on an earlier op
    would also move sibling nodes, which a weight-only construction cannot
    avoid, verified by hand before writing this fixture)."""
    model_a = tiny_llama(
        seed=seed,
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )
    for p in model_a.parameters():
        p.requires_grad_(False)
    model_b = copy.deepcopy(model_a)

    rng = random.Random(seed + 1000)
    exs = _examples(rng, n=4, vocab_size=vocab_size, seq_len=9, span=(5, 8))
    _pos, targets = mn.answer_targets(exs)
    embed = model_a.get_input_embeddings().weight.detach()
    shift = huge * embed[targets]  # [b, d_model]

    last = n_layers - 1
    orig_forward = model_b.model.layers[last].mlp.forward

    def shifted_forward(hidden_states):
        return orig_forward(hidden_states) + shift[:, None, :]

    model_b.model.layers[last].mlp.forward = shifted_forward
    plant_nid = mn.NodeId(last, "mlp", 0)
    return model_a, model_b, plant_nid, exs


class TestRecoveryFrac:
    def test_hand_computed(self):
        assert math.isclose(dcm.recovery_frac(5.0, 2.0, 8.0), 0.5, rel_tol=1e-12)
        assert math.isclose(dcm.recovery_frac(2.0, 2.0, 8.0), 0.0, rel_tol=1e-12)
        assert math.isclose(dcm.recovery_frac(8.0, 2.0, 8.0), 1.0, rel_tol=1e-12)

    def test_denominator_guard_returns_nan(self):
        assert math.isnan(dcm.recovery_frac(1.0, 1.0, 1.0))
        assert math.isnan(dcm.recovery_frac(3.0, 1.0, 1.0 + dcm._DENOM_EPS / 10))

    def test_denominator_above_threshold_is_not_nan(self):
        val = dcm.recovery_frac(1.0, 0.0, 1.0 + dcm._DENOM_EPS * 10)
        assert not math.isnan(val)


class TestBinarizeMask:
    def test_hand_computed_boundary(self):
        nid0 = mn.NodeId(0, "mlp", 0)
        nid1 = mn.NodeId(0, "head", 0)
        nid2 = mn.NodeId(0, "head", 1)
        mask_values = {nid0: 0.5, nid1: 0.49999, nid2: 0.50001}
        selected = dcm.binarize_mask(mask_values, threshold=0.5)
        assert selected[nid0] is True  # exactly at threshold -> selected (>=)
        assert selected[nid1] is False
        assert selected[nid2] is True

    def test_all_below_threshold_gives_empty(self):
        nids = [mn.NodeId(0, "head", h) for h in range(4)]
        mask_values = {n: 0.1 for n in nids}
        selected = dcm.binarize_mask(mask_values, threshold=0.5)
        assert not any(selected.values())

    def test_all_above_threshold_gives_full(self):
        nids = [mn.NodeId(0, "head", h) for h in range(4)]
        mask_values = {n: 0.9 for n in nids}
        selected = dcm.binarize_mask(mask_values, threshold=0.5)
        assert all(selected.values())


class TestSelectedNodeListAndCounts:
    def test_selected_node_list_str_sorted_and_filters(self):
        nids = {
            mn.NodeId(1, "mlp", 0): True,
            mn.NodeId(0, "head", 3): True,
            mn.NodeId(0, "head", 0): False,
            mn.NodeId(0, "mlp", 0): True,
        }
        s = dcm.selected_node_list_str(nids)
        assert s == "a0.h3,m0,m1"

    def test_selected_node_list_str_empty(self):
        assert dcm.selected_node_list_str({mn.NodeId(0, "head", 0): False}) == ""

    def test_per_layer_selected_counts_hand_computed(self):
        selected = {
            mn.NodeId(0, "head", 0): True,
            mn.NodeId(0, "head", 1): True,
            mn.NodeId(0, "mlp", 0): False,
            mn.NodeId(1, "head", 0): False,
            mn.NodeId(1, "mlp", 0): True,
        }
        counts = dcm.per_layer_selected_counts(selected)
        assert counts == {0: 2, 1: 1}

    def test_per_layer_counts_str_includes_zero_layers(self):
        counts_str = dcm.per_layer_counts_str({0: 2, 1: 0, 2: 1})
        assert counts_str == "0:2,1:0,2:1"


class TestPrepareBatch:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            dcm.prepare_batch(model, model, [])

    def test_mismatched_head_geometry_raises(self, tiny_llama):
        model_a = tiny_llama(
            seed=1, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = tiny_llama(
            seed=2, n_layers=2, d_model=16, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(1)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=7, span=(4, 6))
        with pytest.raises(ValueError, match="head geometry"):
            dcm.prepare_batch(model_a, model_b, exs)

    def test_mismatched_layer_count_raises(self, tiny_llama):
        model_a = tiny_llama(
            seed=3, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = tiny_llama(
            seed=4, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(2)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=7, span=(4, 6))
        with pytest.raises(ValueError, match="blocks"):
            dcm.prepare_batch(model_a, model_b, exs)

    def test_identical_models_clean_metrics_equal(self, tiny_llama):
        model = tiny_llama(seed=5, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(3)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=8, span=(5, 7))
        batch = dcm.prepare_batch(model, model, exs)
        assert math.isclose(batch.clean_logprob_a, batch.clean_logprob_b, abs_tol=1e-9)
        assert batch.n == 3
        n_heads, _d = mn._num_heads_and_dhead(model)
        assert len(batch.nodes) == 2 * (n_heads + 1)


class TestEvaluateMaskExactBounds:
    """The genuinely-isolated-single-node construction (``_plant_last_mlp_shift``):
    patching EXACTLY that node must recover model_b's own clean metric
    EXACTLY (recovery_frac == 1.0), and patching NOTHING must reproduce
    model_a's own clean metric EXACTLY (recovery_frac == 0.0)."""

    def test_single_correct_node_gives_exact_recovery_one(self, tiny_llama):
        model_a, model_b, plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=10, n_layers=3, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        selected = {nid: (nid == plant_nid) for nid in batch.nodes}
        ev = dcm.evaluate_mask(model_a, batch, selected, positions="all")
        assert math.isclose(ev["masked_logprob"], batch.clean_logprob_b, abs_tol=1e-5)
        assert math.isclose(ev["recovery_frac"], 1.0, abs_tol=1e-5)
        assert ev["n_selected"] == 1

    def test_empty_mask_gives_exact_recovery_zero(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=11, n_layers=3, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        selected = {nid: False for nid in batch.nodes}
        ev = dcm.evaluate_mask(model_a, batch, selected, positions="all")
        assert math.isclose(ev["masked_logprob"], batch.clean_logprob_a, abs_tol=1e-9)
        assert math.isclose(ev["recovery_frac"], 0.0, abs_tol=1e-9)
        assert ev["n_selected"] == 0

    def test_wrong_node_alone_does_not_reach_full_recovery(self, tiny_llama):
        model_a, model_b, plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=12, n_layers=3, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        wrong_nid = next(n for n in batch.nodes if n != plant_nid and n.kind == "head")
        selected = {nid: (nid == wrong_nid) for nid in batch.nodes}
        ev = dcm.evaluate_mask(model_a, batch, selected, positions="all")
        assert not math.isclose(ev["recovery_frac"], 1.0, abs_tol=1e-3)

    def test_gain_is_large_and_well_separated(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=13, n_layers=3, d_model=32, vocab_size=64, huge=10.0
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        assert batch.clean_logprob_b - batch.clean_logprob_a > 0.3


class TestEvaluateMaskPositions:
    def test_invalid_positions_raises(self, tiny_llama):
        model = tiny_llama(seed=14, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(4)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        batch = dcm.prepare_batch(model, model, exs)
        with pytest.raises(ValueError, match="positions"):
            dcm.evaluate_mask(model, batch, {n: False for n in batch.nodes}, positions="bogus")

    def test_answer_vs_all_differ_for_early_layer_perturbation(self, tiny_llama):
        # Plant at an EARLY (non-final) layer so causal attention propagates
        # differently depending on whether every position or only the
        # answer position carries the patched value (mirrors
        # cross_patch.py's own scope-dependence test).
        n_layers, d_model, vocab_size = 3, 32, 64
        model_a = tiny_llama(
            seed=15,
            n_layers=n_layers,
            d_model=d_model,
            vocab_size=vocab_size,
            tie_word_embeddings=True,
        )
        model_b = copy.deepcopy(model_a)
        rng = random.Random(5)
        exs = _examples(rng, n=4, vocab_size=vocab_size, seq_len=9, span=(5, 8))
        _pos, targets = mn.answer_targets(exs)
        embed = model_a.get_input_embeddings().weight.detach()
        shift = 10.0 * embed[targets]
        early = 0
        orig_forward = model_b.model.layers[early].mlp.forward

        def shifted_forward(hidden_states):
            return orig_forward(hidden_states) + shift[:, None, :]

        model_b.model.layers[early].mlp.forward = shifted_forward
        plant_nid = mn.NodeId(early, "mlp", 0)

        batch = dcm.prepare_batch(model_a, model_b, exs)
        selected = {nid: (nid == plant_nid) for nid in batch.nodes}
        ev_answer = dcm.evaluate_mask(model_a, batch, selected, positions="answer")
        ev_all = dcm.evaluate_mask(model_a, batch, selected, positions="all")
        assert not math.isclose(ev_answer["masked_logprob"], ev_all["masked_logprob"], abs_tol=1e-4)


class TestFitMaskGradientIsolation:
    def test_grad_flows_only_to_mask_logits(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=20, n_layers=2, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        # _plant_last_mlp_shift freezes model_a's params for its OWN reasons
        # (needed elsewhere); re-enable requires_grad on BOTH models here so
        # this test actually exercises fit_mask's freeze/restore instead of
        # trivially passing because nothing could ever require grad in the
        # first place -- this is the real production state
        # (mech_lib.load_any_model's checkpoints have requires_grad=True).
        for p in model_a.parameters():
            p.requires_grad_(True)
            p.grad = None
        for p in model_b.parameters():
            p.requires_grad_(True)
            p.grad = None
        dcm.fit_mask(
            model_a, batch, lam=0.01, steps=3, lr=0.1, seed=0, init_logit=-3.0, positions="all"
        )
        assert all(p.grad is None for p in model_a.parameters())
        assert all(p.grad is None for p in model_b.parameters())

    def test_model_a_requires_grad_restored(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=21, n_layers=2, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        params = list(model_a.parameters())
        for i, p in enumerate(params):
            p.requires_grad_(i % 2 == 0)
        before = [p.requires_grad for p in params]
        dcm.fit_mask(
            model_a, batch, lam=0.01, steps=3, lr=0.1, seed=0, init_logit=-3.0, positions="all"
        )
        after = [p.requires_grad for p in params]
        assert before == after


class TestFitMaskInvalidArgs:
    def test_invalid_positions_raises(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=22, n_layers=1, d_model=16, vocab_size=32
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        with pytest.raises(ValueError, match="positions"):
            dcm.fit_mask(
                model_a,
                batch,
                lam=0.01,
                steps=2,
                lr=0.1,
                seed=0,
                init_logit=-3.0,
                positions="bogus",
            )

    def test_zero_steps_raises(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=23, n_layers=1, d_model=16, vocab_size=32
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        with pytest.raises(ValueError, match="steps"):
            dcm.fit_mask(
                model_a, batch, lam=0.01, steps=0, lr=0.1, seed=0, init_logit=-3.0, positions="all"
            )


class TestFitMaskConvergenceProperties:
    def test_loss_decreases_over_steps(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=30, n_layers=3, d_model=32, vocab_size=64, huge=10.0
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        fit = dcm.fit_mask(
            model_a, batch, lam=0.01, steps=40, lr=0.3, seed=0, init_logit=-3.0, positions="all"
        )
        assert fit.loss_final < fit.loss_initial

    def test_optimizer_recovers_single_planted_node(self, tiny_llama):
        model_a, model_b, plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=31, n_layers=3, d_model=32, vocab_size=64, huge=10.0
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        fit = dcm.fit_mask(
            model_a, batch, lam=0.01, steps=60, lr=0.3, seed=0, init_logit=-3.0, positions="all"
        )
        selected = dcm.binarize_mask(fit.mask_values, threshold=0.5)
        assert selected[plant_nid] is True
        assert sum(selected.values()) == 1

    def test_determinism_under_seed(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=32, n_layers=2, d_model=32, vocab_size=64, huge=10.0
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        fit1 = dcm.fit_mask(
            model_a, batch, lam=0.01, steps=10, lr=0.2, seed=7, init_logit=-3.0, positions="all"
        )
        fit2 = dcm.fit_mask(
            model_a, batch, lam=0.01, steps=10, lr=0.2, seed=7, init_logit=-3.0, positions="all"
        )
        for nid in batch.nodes:
            assert math.isclose(fit1.mask_values[nid], fit2.mask_values[nid], abs_tol=1e-9)
        assert math.isclose(fit1.loss_initial, fit2.loss_initial, abs_tol=1e-9)
        assert math.isclose(fit1.loss_final, fit2.loss_final, abs_tol=1e-9)

    def test_large_lambda_gives_empty_mask(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=33, n_layers=2, d_model=32, vocab_size=64, huge=10.0
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        fit = dcm.fit_mask(
            model_a, batch, lam=1000.0, steps=30, lr=0.5, seed=0, init_logit=-3.0, positions="all"
        )
        selected = dcm.binarize_mask(fit.mask_values, threshold=0.5)
        assert not any(selected.values())

    def test_zero_lambda_high_init_logit_keeps_all_selected(self, tiny_llama):
        # model_b == model_a (identical): a_T[nid] == a_clean[nid] pointwise
        # for every node, so the interpolated activation m*a_T+(1-m)*a_clean
        # equals a_clean REGARDLESS of m -- the metric term contributes
        # EXACTLY zero gradient to every mask logit. With lambda=0 the whole
        # objective is gradient-free, so a high init_logit stays high.
        model = tiny_llama(seed=34, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        for p in model.parameters():
            p.requires_grad_(False)
        rng = random.Random(6)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=8, span=(5, 7))
        batch = dcm.prepare_batch(model, model, exs)
        fit = dcm.fit_mask(
            model, batch, lam=0.0, steps=20, lr=0.5, seed=0, init_logit=5.0, positions="all"
        )
        selected = dcm.binarize_mask(fit.mask_values, threshold=0.5)
        assert all(selected.values())
        for v in fit.mask_values.values():
            assert v > 0.99


class TestFitMaskAnalyticGradientMatchesFiniteDifference:
    def test_single_node_grad_matches_finite_difference(self, tiny_llama):
        """Isolates ONE node's mask logit (only that node is in the
        ``patch_nodes`` replacements dict, so no other node's interpolation
        contributes to the metric at all) and checks the analytic gradient
        through ``patch_nodes``'s functional (``torch.cat``) head rebuild
        against a symmetric finite difference."""
        model_a, model_b, plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=40, n_layers=2, d_model=32, vocab_size=64, huge=10.0
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        a_t = batch.a_T[plant_nid]

        def metric_mean(logit_value: float) -> tuple[torch.Tensor, torch.Tensor]:
            logit = torch.tensor(logit_value, requires_grad=True)

            def repl(a_clean: torch.Tensor) -> torch.Tensor:
                m = torch.sigmoid(logit)
                return m * a_t + (1 - m) * a_clean

            with mn.patch_nodes(model_a, {plant_nid: repl}, None), torch.enable_grad():
                metric = mn.answer_logprob(
                    model_a, batch.input_ids, batch.attention_mask, batch.answer_pos, batch.targets
                )
            return metric.mean(), logit

        m0, logit0 = metric_mean(-1.0)
        m0.backward()
        analytic = float(logit0.grad)

        eps = 1e-2
        m_plus, _ = metric_mean(-1.0 + eps)
        m_minus, _ = metric_mean(-1.0 - eps)
        fd = float((m_plus.detach() - m_minus.detach()) / (2 * eps))
        assert math.isclose(analytic, fd, rel_tol=0.05, abs_tol=1e-3)


class TestHooksCleanup:
    def test_fit_mask_hooks_removed_after_exception(self, tiny_llama, monkeypatch):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=50, n_layers=2, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)

        real_answer_logprob = dcm.answer_logprob
        # _plant_last_mlp_shift already froze model_a's params (needed for
        # its own gradient-isolation use elsewhere); record that as the
        # PRE-CALL state so this test checks restoration to whatever it
        # was, not an assumed True.
        before_requires_grad = [p.requires_grad for p in model_a.parameters()]

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(dcm, "answer_logprob", boom)
        with pytest.raises(RuntimeError, match="boom"):
            dcm.fit_mask(
                model_a, batch, lam=0.01, steps=3, lr=0.1, seed=0, init_logit=-3.0, positions="all"
            )
        monkeypatch.setattr(dcm, "answer_logprob", real_answer_logprob)

        with torch.no_grad():
            base_logits = model_a(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask.long()
            ).logits
        for block in model_a.model.layers:
            assert len(mn._o_proj_module(block)._forward_pre_hooks) == 0
            assert len(mn._mlp_module(block)._forward_hooks) == 0
        with torch.no_grad():
            logits_after = model_a(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask.long()
            ).logits
        assert torch.allclose(base_logits, logits_after, atol=1e-6)
        after_requires_grad = [p.requires_grad for p in model_a.parameters()]
        assert after_requires_grad == before_requires_grad

    def test_evaluate_mask_hooks_removed_after_exception(self, tiny_llama, monkeypatch):
        model_a, model_b, plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=51, n_layers=2, d_model=32, vocab_size=64
        )
        batch = dcm.prepare_batch(model_a, model_b, exs)
        # Select the planted mlp node AND a head node so BOTH of
        # patch_nodes's hook installation paths (o_proj forward-PRE-hook
        # for heads, mlp forward-hook for mlp) actually fire before the
        # exception -- selecting only the mlp node would leave the head
        # path's cleanup entirely unchecked.
        head_nid = next(n for n in batch.nodes if n.kind == "head")
        selected = {nid: (nid in (plant_nid, head_nid)) for nid in batch.nodes}

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(dcm, "_forward_metrics", boom)
        with pytest.raises(RuntimeError, match="boom"):
            dcm.evaluate_mask(model_a, batch, selected, positions="all")

        for block in model_a.model.layers:
            assert len(mn._o_proj_module(block)._forward_pre_hooks) == 0
            assert len(mn._mlp_module(block)._forward_hooks) == 0
        with torch.no_grad():
            model_a(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask.long()
            )  # no raise


class TestDcmRows:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=60, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            dcm.dcm_rows(
                model,
                model,
                [],
                [0.01],
                steps=2,
                lr=0.1,
                seed=0,
                init_logit=-3.0,
                threshold=0.5,
                positions="all",
            )

    def test_empty_lambdas_raises(self, tiny_llama):
        model = tiny_llama(seed=61, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(7)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        with pytest.raises(ValueError):
            dcm.dcm_rows(
                model,
                model,
                exs,
                [],
                steps=2,
                lr=0.1,
                seed=0,
                init_logit=-3.0,
                threshold=0.5,
                positions="all",
            )

    def test_row_levels_and_columns(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=62, n_layers=2, d_model=32, vocab_size=64
        )
        rows = dcm.dcm_rows(
            model_a,
            model_b,
            exs,
            [0.01, 0.1],
            steps=3,
            lr=0.1,
            seed=0,
            init_logit=-3.0,
            threshold=0.5,
            positions="all",
        )
        df = pd.DataFrame(rows)
        assert set(df["level"]) == {"fit", "node"}
        fit_df = df[df["level"] == "fit"]
        assert len(fit_df) == 2  # one per lambda
        n_heads, _d = mn._num_heads_and_dhead(model_a)
        n_nodes = 2 * (n_heads + 1)
        node_df = df[df["level"] == "node"]
        assert len(node_df) == 2 * n_nodes  # n_nodes per lambda
        for col in (
            "lambda",
            "n_selected",
            "frac_selected",
            "recovery_frac",
            "masked_top1_acc",
            "clean_logprob_a",
            "clean_logprob_b",
            "selected_nodes",
            "per_layer_counts",
        ):
            assert col in fit_df.columns

    def test_fit_row_n_selected_matches_node_row_count(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=63, n_layers=2, d_model=32, vocab_size=64, huge=10.0
        )
        rows = dcm.dcm_rows(
            model_a,
            model_b,
            exs,
            [0.01],
            steps=20,
            lr=0.3,
            seed=0,
            init_logit=-3.0,
            threshold=0.5,
            positions="all",
        )
        fit_row = next(r for r in rows if r["level"] == "fit")
        node_rows = [r for r in rows if r["level"] == "node"]
        hand_count = sum(1 for r in node_rows if r["selected"])
        assert fit_row["n_selected"] == hand_count

    def test_reused_batch_gives_identical_clean_metrics_across_lambdas(self, tiny_llama):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=64, n_layers=2, d_model=32, vocab_size=64
        )
        rows = dcm.dcm_rows(
            model_a,
            model_b,
            exs,
            [0.001, 0.01, 0.1],
            steps=2,
            lr=0.1,
            seed=0,
            init_logit=-3.0,
            threshold=0.5,
            positions="all",
        )
        fit_rows = [r for r in rows if r["level"] == "fit"]
        clean_a_vals = {r["clean_logprob_a"] for r in fit_rows}
        clean_b_vals = {r["clean_logprob_b"] for r in fit_rows}
        assert len(clean_a_vals) == 1
        assert len(clean_b_vals) == 1

    def test_print_summary_does_not_crash(self, tiny_llama, capsys):
        model_a, model_b, _plant_nid, exs = _plant_last_mlp_shift(
            tiny_llama, seed=65, n_layers=1, d_model=16, vocab_size=32
        )
        rows = dcm.dcm_rows(
            model_a,
            model_b,
            exs,
            [0.01],
            steps=2,
            lr=0.1,
            seed=0,
            init_logit=-3.0,
            threshold=0.5,
            positions="all",
        )
        dcm.print_summary(rows)
        out = capsys.readouterr().out
        assert "lambda" in out


def _save_tiny_model(tiny_llama, tmp_path, seed, name, n_layers=2, d_model=16, vocab_size=32):
    model = tiny_llama(
        seed=seed,
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )
    out = tmp_path / name
    model.save_pretrained(out)
    return out


def _save_tokenizer(tiny_tokenizer, tmp_path):
    tok = tiny_tokenizer(vocab_size=32)
    out = tmp_path / "tokenizer"
    tok.save_pretrained(out)
    return out


def _task_parquet(tmp_path):
    texts = [f"t1 t2 t{i}" for i in range(3, 9)]
    starts = [len(t) - len(t.split()[-1]) for t in texts]
    ends = [len(t) for t in texts]
    df = pd.DataFrame({"full_text": texts, "answer_char_start": starts, "answer_char_end": ends})
    p = tmp_path / "task.parquet"
    df.to_parquet(p)
    return p


class TestMainSmoke:
    def test_main_single_lambda_smoke(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        model_a_dir = _save_tiny_model(tiny_llama, tmp_path, 70, "model_a")
        model_b_dir = _save_tiny_model(tiny_llama, tmp_path, 71, "model_b")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "dcm.py",
            "--model-a",
            f"dir:{model_a_dir}",
            "--model-b",
            f"dir:{model_b_dir}",
            "--prompt-parquet",
            str(parquet),
            "--tokenizer",
            str(tok_dir),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--steps",
            "6",
            "--lr",
            "0.2",
            "--lambda",
            "0.01",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        dcm.main()
        assert out.is_file()
        df = pd.read_csv(out)
        assert set(df["level"]) == {"fit", "node"}

    def test_main_lambda_sweep_smoke(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        model_a_dir = _save_tiny_model(tiny_llama, tmp_path, 72, "model_a")
        model_b_dir = _save_tiny_model(tiny_llama, tmp_path, 73, "model_b")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "dcm.py",
            "--model-a",
            f"dir:{model_a_dir}",
            "--model-b",
            f"dir:{model_b_dir}",
            "--prompt-parquet",
            str(parquet),
            "--tokenizer",
            str(tok_dir),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--steps",
            "5",
            "--lr",
            "0.2",
            "--lambdas",
            "0.001,0.1",
            "--positions",
            "answer",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        dcm.main()
        assert out.is_file()
        df = pd.read_csv(out)
        fit_df = df[df["level"] == "fit"]
        assert set(fit_df["lambda"]) == {0.001, 0.1}
        assert set(fit_df["positions"]) == {"answer"}
