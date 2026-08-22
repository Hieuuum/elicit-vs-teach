"""``cross_patch.py`` (Phase-0/Tier-2 test 4): cross-model residual patching
θ_T -> θ0.

Silent failure modes guarded:

- A sign/anchor bug in ``recovery_frac`` (e.g. swapping which clean metric
  is the anchor) would silently flip "recovered" and "damaged" readings —
  checked against a hand-computed value and against the exact 0.0/1.0
  bounds a planted, block-local perturbation must produce.
- Reusing ``patch_residual``'s ``positions=None`` for scope="all" is only
  correct because padding never leaks into a real position's computation
  through the model's attention mask — a broken assumption there would
  silently give the WRONG "all" numbers with no crash. Checked indirectly
  via the exact planted-perturbation bounds, which "all" must hit exactly
  at every layer >= k+1 (not just the final one, unlike scope="answer").
- A denominator-guard bug (dividing by ~0 instead of returning NaN) would
  produce a huge, meaningless finite ratio for the identical-models case
  instead of the documented NaN.
- Any position/target off-by-one relative to ``logit_lens.position_target_
  pairs`` would silently score the wrong token as "the answer" — checked
  by direct comparison, not just internal self-consistency.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``), no
network, no real checkpoints.
"""

from __future__ import annotations

import copy
import math
import random

import pandas as pd
import pytest
import torch

from geode.arith.spans import SftExample

from tests._scriptloader import load

cp = load("cross_patch")
mn = load("mech_nodes")
ll = load("logit_lens")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


def _perturbed_pair(tiny_llama, *, seed, n_layers, d_model, vocab_size, perturb_layer, delta=0.05):
    """``(model_a, model_b, perturb_layer)``: ``model_b`` is ``model_a`` with
    ``perturb_layer``'s ``o_proj`` weight shifted — the "only block k
    differs" scenario the exact recovery-bound tests need."""
    model_a = tiny_llama(
        seed=seed,
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )
    model_b = copy.deepcopy(model_a)
    with torch.no_grad():
        model_b.model.layers[perturb_layer].self_attn.o_proj.weight.data += delta
    return model_a, model_b


class TestLogprobAndTop1:
    def test_hand_computed(self):
        logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 0.5]])
        targets = torch.tensor([1, 0])
        logp, top1 = cp._logprob_and_top1(logits, targets)
        assert torch.equal(top1, torch.tensor([True, True]))
        expected = torch.log_softmax(logits.double(), dim=-1).gather(1, targets[:, None]).squeeze(1)
        assert torch.allclose(logp, expected)

    def test_wrong_prediction_top1_false(self):
        logits = torch.tensor([[5.0, 1.0, 2.0]])
        targets = torch.tensor([1])
        _logp, top1 = cp._logprob_and_top1(logits, targets)
        assert top1.item() is False


class TestRecoveryFrac:
    def test_hand_computed(self):
        assert math.isclose(cp.recovery_frac(5.0, 2.0, 8.0), 0.5, rel_tol=1e-12)
        assert math.isclose(cp.recovery_frac(2.0, 2.0, 8.0), 0.0, rel_tol=1e-12)
        assert math.isclose(cp.recovery_frac(8.0, 2.0, 8.0), 1.0, rel_tol=1e-12)

    def test_denominator_guard_returns_nan(self):
        assert math.isnan(cp.recovery_frac(1.0, 1.0, 1.0))
        assert math.isnan(cp.recovery_frac(3.0, 1.0, 1.0 + cp._DENOM_EPS / 10))

    def test_denominator_above_threshold_is_not_nan(self):
        val = cp.recovery_frac(1.0, 0.0, 1.0 + cp._DENOM_EPS * 10)
        assert not math.isnan(val)


class TestPositionAlignment:
    def test_matches_logit_lens_first_answer(self, tiny_tokenizer):
        tok = tiny_tokenizer(vocab_size=64)
        texts = ["t1 t2 t3 t4 t5", "t6 t7 t8 t9 t10 t11"]
        answers = ["t5", "t11"]  # answer must be the last token (append_eos requires span-at-end)
        starts = [len(t) - len(a) for t, a in zip(texts, answers)]
        ends = [len(t) for t in texts]
        df = pd.DataFrame(
            {"full_text": texts, "answer_char_start": starts, "answer_char_end": ends}
        )
        from mech_lib import load_task_examples

        examples = load_task_examples(df, tok)
        ex_idx, ll_pos, ll_target = ll.position_target_pairs(examples, ll.FIRST_ANSWER)
        mn_pos, mn_target = mn.answer_targets(examples)
        assert ex_idx == list(range(len(examples)))
        assert mn_pos.tolist() == ll_pos
        assert mn_target.tolist() == ll_target


class TestCrossPatchRows:
    def test_identical_models_recovery_nan_and_metrics_equal(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(1)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=8, span=(5, 7))
        rows = cp.cross_patch_rows(model, model, exs)
        df = pd.DataFrame(rows)
        assert df["recovery_frac"].isna().all()
        assert (df["metric_clean_a"] == df["metric_clean_b"]).all()
        assert (df["metric_patched"] - df["metric_clean_a"]).abs().max() < 1e-6

    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=1, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            cp.cross_patch_rows(model, model, [])

    def test_row_count_and_columns(self, tiny_llama):
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=2, n_layers=3, d_model=32, vocab_size=64, perturb_layer=1
        )
        rng = random.Random(2)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=8, span=(5, 7))
        rows = cp.cross_patch_rows(model_a, model_b, exs)
        # (n_layers+1) layers * 2 scopes * 2 directions
        assert len(rows) == 4 * 2 * 2
        expected_cols = {
            "layer",
            "scope",
            "direction",
            "metric_clean_a",
            "metric_clean_b",
            "metric_patched",
            "recovery_frac",
            "top1_clean_a",
            "top1_clean_b",
            "top1_patched",
            "n",
        }
        assert expected_cols <= set(rows[0].keys())
        assert all(r["n"] == 2 for r in rows)

    def test_planted_perturbation_all_scope_exact_bounds_both_directions(self, tiny_llama):
        k = 1
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=3, n_layers=4, d_model=32, vocab_size=64, perturb_layer=k
        )
        rng = random.Random(3)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=9, span=(5, 8))
        rows = cp.cross_patch_rows(model_a, model_b, exs)
        df = pd.DataFrame(rows)
        all_scope = df[df["scope"] == "all"]

        t_into_0 = all_scope[all_scope["direction"] == "T_into_0"].set_index("layer")
        for layer in range(0, k + 1):
            assert math.isclose(t_into_0.loc[layer, "recovery_frac"], 0.0, abs_tol=1e-9)
        for layer in range(k + 1, 5):
            assert math.isclose(t_into_0.loc[layer, "recovery_frac"], 1.0, abs_tol=1e-6)

        zero_into_t = all_scope[all_scope["direction"] == "0_into_T"].set_index("layer")
        for layer in range(0, k + 1):
            assert math.isclose(zero_into_t.loc[layer, "recovery_frac"], 1.0, abs_tol=1e-6)
        for layer in range(k + 1, 5):
            assert math.isclose(zero_into_t.loc[layer, "recovery_frac"], 0.0, abs_tol=1e-9)

    def test_planted_perturbation_answer_scope_zero_bound_also_exact(self, tiny_llama):
        # The layer<=k / recovery=0.0 bound is scope-independent: A and B's
        # residuals are bit-identical up to layer k regardless of which
        # positions get patched, since block k hasn't run yet at that point.
        k = 1
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=4, n_layers=4, d_model=32, vocab_size=64, perturb_layer=k
        )
        rng = random.Random(4)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=9, span=(5, 8))
        rows = cp.cross_patch_rows(model_a, model_b, exs)
        df = pd.DataFrame(rows)
        answer_scope = df[(df["scope"] == "answer") & (df["direction"] == "T_into_0")].set_index(
            "layer"
        )
        for layer in range(0, k + 1):
            assert math.isclose(answer_scope.loc[layer, "recovery_frac"], 0.0, abs_tol=1e-9)

    def test_scope_answer_vs_all_differ_at_intermediate_layer(self, tiny_llama):
        # At an intermediate layer (strictly between the perturbation and
        # the final residual), "answer" patches only one position while
        # attention still mixes in the OTHER positions' un-patched values —
        # "all" replaces every position, giving a different (exact) result.
        # At the FINAL layer both scopes coincide (final norm+lm_head are
        # position-wise), so this checks a layer before the last one.
        k = 0
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=5, n_layers=3, d_model=32, vocab_size=64, perturb_layer=k
        )
        rng = random.Random(5)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=9, span=(5, 8))
        rows = cp.cross_patch_rows(model_a, model_b, exs)
        df = pd.DataFrame(rows)
        intermediate_layer = k + 1  # < n_layers (3), so not the final layer
        assert intermediate_layer < 3
        sub = df[(df["layer"] == intermediate_layer) & (df["direction"] == "T_into_0")].set_index(
            "scope"
        )
        assert not math.isclose(
            sub.loc["answer", "recovery_frac"], sub.loc["all", "recovery_frac"], abs_tol=1e-4
        )
        # "all" hits the exact bound at this layer; "answer" does not (yet).
        assert math.isclose(sub.loc["all", "recovery_frac"], 1.0, abs_tol=1e-6)
        assert not math.isclose(sub.loc["answer", "recovery_frac"], 1.0, abs_tol=1e-3)

    def test_final_layer_scope_independent(self, tiny_llama):
        k = 0
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=6, n_layers=3, d_model=32, vocab_size=64, perturb_layer=k
        )
        rng = random.Random(6)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=9, span=(5, 8))
        rows = cp.cross_patch_rows(model_a, model_b, exs)
        df = pd.DataFrame(rows)
        final_layer = 3  # n_layers
        sub = df[(df["layer"] == final_layer) & (df["direction"] == "T_into_0")].set_index("scope")
        assert math.isclose(
            sub.loc["answer", "recovery_frac"], sub.loc["all", "recovery_frac"], abs_tol=1e-9
        )

    def test_hooks_removed_after_full_run(self, tiny_llama):
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=7, n_layers=2, d_model=16, vocab_size=32, perturb_layer=0
        )
        rng = random.Random(7)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        cp.cross_patch_rows(model_a, model_b, exs)
        embed_a = model_a.get_input_embeddings()
        embed_b = model_b.get_input_embeddings()
        assert len(embed_a._forward_hooks) == 0
        assert len(embed_b._forward_hooks) == 0
        for block in list(model_a.model.layers) + list(model_b.model.layers):
            assert len(block._forward_hooks) == 0

    def test_print_summary_does_not_crash(self, tiny_llama):
        model_a, model_b = _perturbed_pair(
            tiny_llama, seed=8, n_layers=2, d_model=16, vocab_size=32, perturb_layer=0
        )
        rng = random.Random(8)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        rows = cp.cross_patch_rows(model_a, model_b, exs)
        cp.print_summary(rows)  # smoke: must not raise
