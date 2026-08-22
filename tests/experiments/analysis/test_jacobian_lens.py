"""``jacobian_lens.py`` (Phase 0 test 7, Tier 2, "J-lens" read as a Jacobian
lens — module docstring states the naming assumption).

Silent failure modes guarded:

- A wrong hook wiring (missing ``retain_grad()``, or differentiating
  through a DETACHED copy instead of the real forward graph) would produce
  a plausible-looking but numerically wrong Jacobian with no crash —
  checked against a fully independent autograd path at the final layer
  (test a) and against a hand-planted first-order perturbation (test c).
- Padding interacting with gradients (a batched example's Jacobian being
  polluted by another example in the same batch, or by its own trailing
  pad tokens) would silently corrupt every real multi-example run — checked
  by comparing a mixed-length batch against each example run alone (test
  b), which is also the empirical justification for summing before ONE
  ``.backward()`` (see ``jacobian_and_logprob``'s docstring).
- ``label_span[0] == 0`` silently wrapping ``pos = -1`` to the LAST
  sequence position (Python negative indexing) instead of raising.
- The identical-models edge case reading NaN instead of the documented 0.0
  convention on three of the four bridge columns, and reading a wrong non-
  NaN value on the fourth (``pred_gain_ratio``, a genuine 0/0 there).
- Hooks not removed after a mid-forward error, leaving a corrupted model
  for any caller that reuses it.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``), no
network, no real checkpoints.
"""

from __future__ import annotations

import copy
import math
import random

import pandas as pd
import torch

from geode.arith.spans import SftExample

from tests._scriptloader import load

# Load order matters: jacobian_lens.py does `from logit_lens import ...` and
# `from resid_shift import ...` as bare sibling imports, resolved through
# sys.modules. Loading those two (and mech_lib, which both of them import)
# BEFORE jacobian_lens means jacobian_lens's imports reuse the SAME module
# objects rather than re-executing fresh copies under the same names -- the
# `is` identity check below (test_position_reuses_logit_lens_position_target_pairs)
# only holds with this order.
ll = load("logit_lens")
mech = load("mech_lib")
rs = load("resid_shift")
jl = load("jacobian_lens")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


class TestCaptureResidualsGrad:
    def test_names_shapes_and_logits_match_plain_forward(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        ids = torch.randint(4, 64, (3, 6))
        mask = torch.ones(3, 6, dtype=torch.bool)
        with torch.no_grad():
            plain_logits = model(input_ids=ids, attention_mask=mask.long()).logits

        captured, logits = jl.capture_residuals_grad(model, ids, mask)
        assert list(captured.keys()) == mech.residual_hook_names(3)
        for t in captured.values():
            assert t.shape == (3, 6, 32)
        assert torch.allclose(logits, plain_logits, atol=1e-5)
        assert logits.requires_grad
        assert captured["hook_embed"].requires_grad
        assert captured["hook_embed"].is_leaf

    def test_f_hooks_removed_even_on_mid_forward_error(self, tiny_llama):
        model = tiny_llama(seed=10, n_layers=3, d_model=16, vocab_size=32, tie_word_embeddings=True)
        ids = torch.randint(4, 32, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)

        def boom(*_a, **_k):
            raise RuntimeError("boom")

        orig_forward = model.model.layers[1].forward
        model.model.layers[1].forward = boom
        try:
            try:
                jl.capture_residuals_grad(model, ids, mask)
            except RuntimeError as e:
                assert "boom" in str(e)
            else:
                raise AssertionError("expected RuntimeError")
        finally:
            model.model.layers[1].forward = orig_forward

        assert model.get_input_embeddings()._forward_hooks == {}
        for block in model.model.layers:
            assert block._forward_hooks == {}


class TestFreezeParams:
    def test_sets_requires_grad_false(self, tiny_llama):
        model = tiny_llama(seed=1, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        assert any(p.requires_grad for p in model.parameters())
        jl.freeze_params(model)
        assert all(not p.requires_grad for p in model.parameters())

    def test_does_not_change_the_jacobian(self, tiny_llama):
        model_frozen = tiny_llama(
            seed=2, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        model_unfrozen = copy.deepcopy(model_frozen)
        jl.freeze_params(model_frozen)

        rng = random.Random(0)
        exs = _examples(rng, n=4, vocab_size=32, seq_len=6, span=(3, 5))
        ex_idx_l, pos_l, target_l = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        ids, mask = mech.pad_examples(exs, device="cpu")
        ex_idx = torch.tensor(ex_idx_l)
        pos = torch.tensor(pos_l)
        target = torch.tensor(target_l)

        jac_f, _, logp_f = jl.jacobian_and_logprob(model_frozen, ids, mask, ex_idx, pos, target)
        jac_u, _, logp_u = jl.jacobian_and_logprob(model_unfrozen, ids, mask, ex_idx, pos, target)

        for name in jac_f:
            assert torch.allclose(jac_f[name], jac_u[name], atol=1e-6)
        assert torch.allclose(logp_f, logp_u, atol=1e-6)


class TestJacobianAndLogprob:
    def test_a_final_layer_matches_independent_autograd(self, tiny_llama):
        model = tiny_llama(
            seed=1, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        ).double()
        jl.freeze_params(model)
        rng = random.Random(0)
        exs = _examples(rng, n=5, vocab_size=64, seq_len=9, span=(5, 8))
        ids, mask = mech.pad_examples(exs, device="cpu")
        ex_idx_l, pos_l, target_l = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        ex_idx = torch.tensor(ex_idx_l)
        pos = torch.tensor(pos_l)
        target = torch.tensor(target_l)

        jac, _resid, logp = jl.jacobian_and_logprob(model, ids, mask, ex_idx, pos, target)
        final_name = list(jac.keys())[-1]

        # Fully independent path: mech_lib's OWN (no-grad) residual capture,
        # then a hand-built leaf + norm/head/log-softmax + its own backward.
        with torch.no_grad():
            acts = mech.capture_residuals(model, ids, mask)
        h_final = acts[final_name][ex_idx, pos, :].detach().clone().requires_grad_(True)
        norm, head = mech.final_norm_and_head(model)
        logits_manual = head(norm(h_final))
        logp_manual = torch.log_softmax(logits_manual, dim=-1)
        target_logp_manual = logp_manual.gather(1, target[:, None]).squeeze(1)
        target_logp_manual.sum().backward()

        assert torch.allclose(jac[final_name], h_final.grad, atol=1e-10, rtol=1e-8)
        assert torch.allclose(logp, target_logp_manual.detach(), atol=1e-10)

    def test_b_batching_invariance_matches_single_example_backward(self, tiny_llama):
        model = tiny_llama(seed=3, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        jl.freeze_params(model)
        rng = random.Random(2)
        exs = [
            SftExample(input_ids=[rng.randrange(4, 64) for _ in range(6)], label_span=(4, 6)),
            SftExample(input_ids=[rng.randrange(4, 64) for _ in range(9)], label_span=(7, 9)),
            SftExample(input_ids=[rng.randrange(4, 64) for _ in range(5)], label_span=(3, 5)),
            SftExample(input_ids=[rng.randrange(4, 64) for _ in range(8)], label_span=(5, 8)),
        ]
        ex_idx_l, pos_l, target_l = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        ids, mask = mech.pad_examples(exs, device="cpu")
        ex_idx = torch.tensor(ex_idx_l)
        pos = torch.tensor(pos_l)
        target = torch.tensor(target_l)
        jac_batch, resid_batch, logp_batch = jl.jacobian_and_logprob(
            model, ids, mask, ex_idx, pos, target
        )

        for i, ex in enumerate(exs):
            ids1, mask1 = mech.pad_examples([ex], device="cpu")
            ei, pi, ti = jl.position_target_pairs([ex], jl.FIRST_ANSWER)
            jac1, resid1, logp1 = jl.jacobian_and_logprob(
                model, ids1, mask1, torch.tensor(ei), torch.tensor(pi), torch.tensor(ti)
            )
            for name in jac1:
                assert torch.allclose(jac_batch[name][i], jac1[name][0], atol=1e-4, rtol=1e-3), (
                    f"{name} at example {i}: batched Jacobian diverges from solo"
                )
                assert torch.allclose(resid_batch[name][i], resid1[name][0], atol=1e-5)
            assert math.isclose(float(logp_batch[i]), float(logp1[0]), rel_tol=1e-3, abs_tol=1e-5)

    def test_c_first_order_prediction_error_shrinks_and_is_small(self, tiny_llama):
        model = tiny_llama(
            seed=4, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=True
        ).double()
        jl.freeze_params(model)
        rng = random.Random(3)
        exs = _examples(rng, n=8, vocab_size=64, seq_len=10, span=(6, 9))
        ex_idx_l, pos_l, target_l = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        ids, mask = mech.pad_examples(exs, device="cpu")
        ex_idx = torch.tensor(ex_idx_l)
        pos = torch.tensor(pos_l)
        target = torch.tensor(target_l)

        jac, _resid, logp0 = jl.jacobian_and_logprob(model, ids, mask, ex_idx, pos, target)
        target_layer_idx = 2  # blocks.{target_layer_idx - 1}.hook_resid_post
        layer_name = list(jac.keys())[target_layer_idx]
        v = torch.randn(model.config.hidden_size, dtype=torch.float64)

        # All examples share one span, so one shared position p. The planted
        # perturbation must land ONLY at that position -- the Jacobian is
        # d(logp_i)/d(h_L[i, p, :]), a single position's gradient. Perturbing
        # every position (a naive `output + eps*v` broadcast) would ALSO
        # perturb positions < p, which causally reach position p through
        # later attention layers, adding a real first-order term the
        # single-position Jacobian never claims to capture -- that mismatch
        # doesn't shrink with epsilon, so this guards a genuine "perturbed
        # the wrong thing" bug, not just numerical noise.
        p = pos_l[0]
        assert all(pp == p for pp in pos_l)

        def make_hook(eps: float):
            def hook(_module, _inputs, output):
                t = output[0] if isinstance(output, tuple) else output
                t = t.clone()
                t[:, p, :] = t[:, p, :] + eps * v
                return (t,) + output[1:] if isinstance(output, tuple) else t

            return hook

        errors = []
        for eps in (1e-2, 1e-5):
            handle = model.model.layers[target_layer_idx - 1].register_forward_hook(make_hook(eps))
            try:
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=mask.long()).logits
            finally:
                handle.remove()
            sel = logits[ex_idx, pos, :]
            logp_pert = torch.log_softmax(sel, dim=-1).gather(1, target[:, None]).squeeze(1)
            actual_delta = float((logp_pert - logp0).mean())
            pred_delta = float((eps * (jac[layer_name] @ v)).mean())
            errors.append(abs(actual_delta - pred_delta) / (abs(actual_delta) + 1e-12))

        assert errors[1] < errors[0] * 0.3, errors  # relative error shrinks with epsilon
        assert errors[1] < 0.05, errors  # AND is small outright at the tiny epsilon

    def test_negative_position_raises_instead_of_wrapping(self, tiny_llama):
        model = tiny_llama(seed=11, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        exs = [SftExample(input_ids=[10, 11], label_span=(0, 1))]
        ex_idx_l, pos_l, target_l = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        assert pos_l == [-1]  # confirms the trap this guard exists to catch
        ids, mask = mech.pad_examples(exs, device="cpu")
        try:
            jl.jacobian_and_logprob(
                model,
                ids,
                mask,
                torch.tensor(ex_idx_l),
                torch.tensor(pos_l),
                torch.tensor(target_l),
            )
        except ValueError as e:
            assert "position" in str(e).lower() or "label_span" in str(e).lower()
        else:
            raise AssertionError("expected ValueError")


class TestPureMetrics:
    def test_jacobian_norm_stats_hand_computed(self):
        jac = torch.tensor([[3.0, 4.0], [0.0, 5.0]])  # norms 5, 5
        h = torch.tensor([[1.0, 0.0], [0.0, 2.0]])  # norms 1, 2
        mean_jn, jn_rel = jl.jacobian_norm_stats(jac, h)
        assert math.isclose(mean_jn, 5.0, rel_tol=1e-9)
        assert math.isclose(jn_rel, (5.0 * 1.0 + 5.0 * 2.0) / 2, rel_tol=1e-9)

    def test_bridge_layer_stats_hand_computed(self):
        d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        jac0 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        logp0 = torch.tensor([-2.0, -3.0])
        logp_t = torch.tensor([-1.0, -2.5])
        out = jl.bridge_layer_stats(d, jac0, logp0, logp_t)
        assert math.isclose(out["mean_cos_shift_vs_jac0"], 0.5, rel_tol=1e-9)  # (1.0 + 0.0) / 2
        assert math.isclose(out["mean_pred_gain_nats"], 0.5, rel_tol=1e-9)  # (1.0 + 0.0) / 2
        assert math.isclose(out["actual_gain_nats"], 0.75, rel_tol=1e-9)  # (1.0 + 0.5) / 2
        assert math.isclose(out["pred_gain_ratio"], 0.5 / 0.75, rel_tol=1e-9)

    def test_bridge_layer_stats_zero_shift_zero_not_nan_except_ratio(self):
        d = torch.zeros(5, 8)
        jac0 = torch.randn(5, 8)
        logp0 = torch.full((5,), -1.5)
        logp_t = logp0.clone()
        out = jl.bridge_layer_stats(d, jac0, logp0, logp_t)
        assert out["mean_cos_shift_vs_jac0"] == 0.0
        assert out["mean_pred_gain_nats"] == 0.0
        assert out["actual_gain_nats"] == 0.0
        assert math.isnan(out["pred_gain_ratio"])  # genuine 0/0, not the norm-clamp convention


class TestJacobianLensRows:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=9, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        try:
            jl.jacobian_lens_rows(model, [], "m", "s")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_position_reuses_logit_lens_position_target_pairs(self):
        assert jl.position_target_pairs is ll.position_target_pairs
        assert jl.FIRST_ANSWER == ll.FIRST_ANSWER
        exs = [
            SftExample(input_ids=[10, 11, 12, 13, 14], label_span=(2, 4)),
            SftExample(input_ids=[20, 21, 22, 23, 24], label_span=(3, 5)),
        ]
        ex_idx, pos, target = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        assert pos == [1, 2]  # label_span[0] - 1, matching logit_lens's convention exactly
        assert target == [12, 23]

    def test_determinism_same_seed_twice_identical_rows(self, tiny_llama):
        model = tiny_llama(seed=5, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(4)
        exs = _examples(rng, n=5, vocab_size=32, seq_len=6, span=(3, 5))
        rows1 = jl.jacobian_lens_rows(model, exs, "m", "s")
        rows2 = jl.jacobian_lens_rows(model, exs, "m", "s")
        pd.testing.assert_frame_equal(
            pd.DataFrame(rows1), pd.DataFrame(rows2), check_exact=False, atol=1e-8, rtol=1e-6
        )

    def test_identical_models_bridge_columns_zero_not_nan_except_ratio(self, tiny_llama):
        model = tiny_llama(seed=6, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        model_b = copy.deepcopy(model)
        rng = random.Random(5)
        exs = _examples(rng, n=4, vocab_size=32, seq_len=6, span=(3, 5))
        rows = jl.jacobian_lens_rows(model, exs, "a", "s", model_b=model_b, model_b_label="b")
        df = pd.DataFrame(rows)
        a_rows = df[df["model"] == "a"]
        b_rows = df[df["model"] == "b"]

        assert (a_rows["mean_cos_shift_vs_jac0"] == 0.0).all()
        assert (a_rows["mean_pred_gain_nats"] == 0.0).all()
        assert (a_rows["actual_gain_nats"] == 0.0).all()
        assert a_rows["pred_gain_ratio"].isna().all()
        # bridge columns are defined relative to theta0's Jacobian only:
        # absent on theta_T's own rows, read back as NaN once concatenated.
        assert b_rows["mean_cos_shift_vs_jac0"].isna().all()

        # And model_b gets its OWN standard columns (its own Jacobian, not zero).
        assert len(b_rows) == len(a_rows)
        assert b_rows["mean_jac_norm"].notna().all()

    def test_consistency_columns_hand_checked_constant_vs_varied_examples(self, tiny_llama):
        model = tiny_llama(seed=8, n_layers=2, d_model=24, vocab_size=32, tie_word_embeddings=True)
        jl.freeze_params(model)

        same_ids = [10, 11, 12, 13, 14]
        same_examples = [SftExample(input_ids=list(same_ids), label_span=(3, 5)) for _ in range(5)]
        rows_const = jl.jacobian_lens_rows(model, same_examples, "m", "s")
        for r in rows_const:
            assert math.isclose(r["mean_cos_to_mean"], 1.0, abs_tol=1e-5)
            assert math.isclose(r["top_pc_evr"], 1.0, abs_tol=1e-5)

        rng = random.Random(9)
        varied_examples = _examples(rng, n=20, vocab_size=32, seq_len=6, span=(3, 5))
        rows_varied = jl.jacobian_lens_rows(model, varied_examples, "m", "s")
        for r in rows_varied:
            assert r["mean_cos_to_mean"] < 0.95

    def test_consistency_columns_wired_to_shift_consistency_directly(self, tiny_llama):
        model = tiny_llama(seed=7, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        jl.freeze_params(model)
        rng = random.Random(6)
        exs = _examples(rng, n=6, vocab_size=32, seq_len=6, span=(3, 5))
        rows = jl.jacobian_lens_rows(model, exs, "m", "s")

        ex_idx_l, pos_l, target_l = jl.position_target_pairs(exs, jl.FIRST_ANSWER)
        ids, mask = mech.pad_examples(exs, device="cpu")
        jac, _resid, _logp = jl.jacobian_and_logprob(
            model, ids, mask, torch.tensor(ex_idx_l), torch.tensor(pos_l), torch.tensor(target_l)
        )
        names = list(jac.keys())
        for layer, name in enumerate(names):
            expected_cos, expected_evr = rs.shift_consistency(jac[name])
            row = next(r for r in rows if r["layer"] == layer)
            assert math.isclose(row["mean_cos_to_mean"], expected_cos, rel_tol=1e-9)
            assert math.isclose(row["top_pc_evr"], expected_evr, rel_tol=1e-9)

    def test_print_summary_smoke(self, tiny_llama):
        model = tiny_llama(seed=12, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        model_b = copy.deepcopy(model)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.q_proj.weight.data += 0.05
        rng = random.Random(11)
        exs = _examples(rng, n=4, vocab_size=32, seq_len=6, span=(3, 5))
        rows = jl.jacobian_lens_rows(model, exs, "a", "s", model_b=model_b, model_b_label="b")
        jl.print_summary(rows)  # smoke: must not raise
