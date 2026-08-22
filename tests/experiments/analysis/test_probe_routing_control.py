"""``probe_routing_control.py`` — the routing-vs-computing control for
``resid_probe.py``'s residual-stream probe (Phase 0 / Tier 1 test 1).

Silent failure modes guarded:

- ``top_position_prediction``/``top_position_determined`` must match the
  pre-registered hand-worked examples exactly (module docstring's own sanity
  list) — a sign error in the ``op == '-'`` branches, or the ``ta == tb``
  edge case, would silently mislabel which test examples are "affected"
  without crashing.
- the KEY property this whole script exists to provide: a probe that only
  recovers the naive top-position PREDICTION (not the true class) must score
  well on "determined" test examples (prediction == truth there, by
  construction) but stay near chance on "affected" ones (prediction != truth
  there, decorrelated from truth); a probe that recovers the TRUE class must
  score well on both. If this didn't hold, ``acc_affected`` would not be a
  valid routing-vs-computed discriminator.
- ``split_mask`` must line up 1:1 with whatever ``--limit`` truncation built
  the examples (example i <-> row i) — a silent misalignment would score the
  wrong example's determined/affected status with no crash.
- refusals: a prompt parquet missing ``a``/``b``/``op``, a set with zero
  AFFECTED test examples (no routing-vs-computed signal at all), a mismatched
  example/row count, and a malformed ``--model`` spec must all raise loudly.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``/
``tiny_tokenizer``), no network, no real checkpoints.
"""

from __future__ import annotations

import math
import random
import sys

import pandas as pd
import pytest
import torch

from geode.arith.spans import SftExample

from tests._scriptloader import load

rp = load("resid_probe")
prc = load("probe_routing_control")


# ---------------------------------------------------------------------------
# Sanity list (module docstring / task contract) — one row per hand-worked
# example: (a, b, op, expected_prediction, expected_determined).
# ---------------------------------------------------------------------------
SANITY_EXAMPLES = [
    (943, 5881, "+", "5", False),  # carry: pred '5', truth '6'
    (7465, 8497, "+", "1", True),  # pred '1', truth '1'
    (5898, 3, "-", "5", True),  # pred '5', truth '5'
    (560, 1188, "-", "-", True),  # pred '-', truth '-'
    (26, 279, "-", "-", True),  # pred '-', truth '-'
    (500, 480, "-", "1", False),  # pred '1', truth '2'
    (523, 480, "-", "1", False),  # pred '1', truth '4'
    (45, 47, "+", "8", False),  # pred '8', truth '9'
    (55, 55, "-", None, False),  # ta == tb -> None; truth '0'
]


class TestTopPositionRule:
    @pytest.mark.parametrize("a,b,op,expected_pred,expected_determined", SANITY_EXAMPLES)
    def test_sanity_examples_exact(self, a, b, op, expected_pred, expected_determined):
        assert prc.top_position_prediction(a, b, op) == expected_pred
        assert prc.top_position_determined(a, b, op) == expected_determined

    def test_rejects_negative_operands(self):
        with pytest.raises(ValueError):
            prc.top_position_prediction(-1, 2, "+")
        with pytest.raises(ValueError):
            prc.top_position_determined(1, -2, "-")

    def test_rejects_bad_op(self):
        with pytest.raises(ValueError):
            prc.top_position_prediction(1, 2, "*")
        with pytest.raises(ValueError):
            prc.top_position_determined(1, 2, "*")


class TestSplitMask:
    def test_matches_top_position_determined_elementwise(self):
        rows = [(a, b, op) for a, b, op, _, _ in SANITY_EXAMPLES]
        df = pd.DataFrame(
            {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
        )
        mask = prc.split_mask(df)
        expected = torch.tensor(
            [not prc.top_position_determined(*r) for r in rows], dtype=torch.bool
        )
        assert torch.equal(mask, expected)

    def test_missing_op_column_raises(self):
        with pytest.raises(ValueError):
            prc.split_mask(pd.DataFrame({"a": [1], "b": [2]}))

    def test_missing_a_and_b_columns_raises(self):
        with pytest.raises(ValueError):
            prc.split_mask(pd.DataFrame({"op": ["+"]}))

    def test_limit_truncation_commutes_with_masking(self):
        """``split_mask(df.iloc[:limit])`` must equal the first ``limit``
        entries of ``split_mask(df)`` — the alignment ``build_routing_
        reference`` (and ``main``) relies on between example i and row i
        after ``mech_lib.load_task_examples``'s own ``--limit`` truncation.
        """
        rng = random.Random(11)
        rows = [
            (rng.randrange(0, 100000), rng.randrange(0, 100000), rng.choice(["+", "-"]))
            for _ in range(20)
        ]
        df = pd.DataFrame(
            {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
        )
        full_mask = prc.split_mask(df)
        for limit in (1, 5, 12, 20):
            truncated_mask = prc.split_mask(df.iloc[:limit])
            assert truncated_mask.shape[0] == limit
            assert torch.equal(truncated_mask, full_mask[:limit])


class TestTokenFeatures:
    def test_shape(self):
        assert prc.token_features(943, 5881, "+").shape == (98,)

    def test_op_onehot_values(self):
        plus = prc.token_features(1, 2, "+")
        minus = prc.token_features(1, 2, "-")
        assert plus[0:2].tolist() == [1.0, 0.0]
        assert minus[0:2].tolist() == [0.0, 1.0]

    def test_length_onehots_sum_to_one(self):
        feats = prc.token_features(943, 47, "+")
        assert feats[2:6].sum().item() == 1.0  # len(a)=3 -> bucket index 2
        assert feats[2:6].tolist() == [0.0, 0.0, 1.0, 0.0]
        assert feats[6:10].sum().item() == 1.0  # len(b)=2 -> bucket index 1
        assert feats[6:10].tolist() == [0.0, 1.0, 0.0, 0.0]

    def test_length_onehot_clamps_beyond_max_digit_positions(self):
        feats = prc.token_features(12345, 1, "+")  # a has 5 digits, clamped to bucket 4
        assert feats[2:6].tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_digit_onehots_each_sum_to_one(self):
        feats = prc.token_features(943, 5881, "+")
        digits = feats[10:]
        assert digits.shape == (88,)
        for k in range(8):
            block = digits[k * 11 : (k + 1) * 11]
            assert block.sum().item() == 1.0

    def test_digit_values_and_absent_positions(self):
        # LEFT-aligned: position 0 = top (most significant) digit.
        # a=943 (3 digits): top=9, next=4, next=3, position3=absent.
        # b=3 (1 digit): top=3, positions 1-3 (past its length)=absent.
        feats = prc.token_features(943, 3, "-")
        a_digits = feats[10:54]
        b_digits = feats[54:98]
        assert a_digits[0 * 11 : 1 * 11].tolist().index(1.0) == 9
        assert a_digits[1 * 11 : 2 * 11].tolist().index(1.0) == 4
        assert a_digits[2 * 11 : 3 * 11].tolist().index(1.0) == 3
        assert a_digits[3 * 11 : 4 * 11].tolist().index(1.0) == prc.ABSENT_DIGIT
        assert b_digits[0 * 11 : 1 * 11].tolist().index(1.0) == 3
        for pos in range(1, 4):
            block = b_digits[pos * 11 : pos * 11 + 11]
            assert block.tolist().index(1.0) == prc.ABSENT_DIGIT

    def test_top_digit_always_at_position_zero_regardless_of_length(self):
        """The whole point of left-alignment: position 0's one-hot block is
        the operand's TOP digit no matter how many digits it has -- unlike
        right-alignment, where the top digit's feature index shifts with
        length and a linear probe can't select it without cross-referencing
        the length one-hots."""
        for x, top_digit in [(7, 7), (94, 9), (943, 9), (5881, 5), (12345, 1)]:
            feats = prc.token_features(x, 1, "+")
            a_pos0 = feats[10:21]
            assert a_pos0.tolist().index(1.0) == top_digit

    def test_rejects_negative_operands_and_bad_op(self):
        with pytest.raises(ValueError):
            prc.token_features(-1, 2, "+")
        with pytest.raises(ValueError):
            prc.token_features(1, 2, "*")


class TestMajorityAccuracy:
    def test_hand_computed(self):
        y = torch.tensor([0, 0, 0, 1, 1, 2])
        assert math.isclose(prc.majority_accuracy(y, n_classes=3), 3.0 / 6.0, rel_tol=1e-12)

    def test_empty_is_nan(self):
        assert math.isnan(prc.majority_accuracy(torch.tensor([], dtype=torch.long), n_classes=3))


class TestFitLinearProbeEquivalence:
    def test_predictions_and_accuracies_agree(self):
        torch.manual_seed(0)
        n_classes = 3
        x_train = torch.randn(30, 5)
        y_train = torch.randint(0, n_classes, (30,))
        x_test = torch.randn(12, 5)
        y_test = torch.randint(0, n_classes, (12,))
        train_acc, test_acc = rp.fit_linear_probe(x_train, y_train, x_test, y_test, n_classes, 1e-3)
        train_pred, test_pred = rp.fit_linear_probe_predictions(
            x_train, y_train, x_test, y_test, n_classes, 1e-3
        )
        assert math.isclose(
            train_acc, (train_pred == y_train).double().mean().item(), rel_tol=1e-12
        )
        assert math.isclose(test_acc, (test_pred == y_test).double().mean().item(), rel_tol=1e-12)


class TestDeterminedFrac:
    def test_matches_mean_of_top_position_determined_on_test_set(self):
        rng = random.Random(42)
        n = 100
        triples = [
            (rng.randrange(0, 100000), rng.randrange(0, 100000), rng.choice(["+", "-"]))
            for _ in range(n)
        ]
        exs = [SftExample(input_ids=[4 + i, 5 + i, 6 + i], label_span=(2, 3)) for i in range(n)]
        df = pd.DataFrame(
            {
                "a": [t[0] for t in triples],
                "b": [t[1] for t in triples],
                "op": [t[2] for t in triples],
            }
        )
        ref = rp.build_reference(exs, "task", seed=3)
        rr = prc.build_routing_reference(df, ref, l2=1e-3)

        test_determined_flags = [
            prc.top_position_determined(*triples[i]) for i in ref.test_idx.tolist()
        ]
        expected = sum(test_determined_flags) / len(test_determined_flags)
        assert math.isclose(rr.determined_frac, expected, rel_tol=1e-12)


class TestTokenBaselineTracksRoutingNotCarries:
    """``token_baseline_acc``/``token_baseline_acc_affected`` are the
    script's model-FREE null. Every ``TestTokenFeatures`` check above pins
    shape/one-hot structure but never actually fits a probe on
    ``token_features`` against a REAL first-answer-character target -- a bug
    in the digit encoding (wrong absent sentinel, wrong alignment direction,
    shifted concat order, ...) would leave every one of those tests green
    while silently breaking the one thing this baseline exists to measure.
    This builds examples whose target token genuinely IS the true first
    answer character of real ``(a, b, op)`` triples (operand range matched
    to the real ``D_algo_eval_bare.parquet``: ``a, b`` in ``[0, 9999]``, so
    every digit lands inside ``MAX_DIGIT_POSITIONS`` with no truncation) and
    checks the two-sided property the module docstring claims: with LEFT-
    aligned digit features, position 0 IS the top digit at every operand
    length, so the baseline recovers the top-position-ROUTABLE part of the
    signal (well above majority overall, already with a few thousand test
    examples -- re-measured empirically against the real parquet, see the
    module docstring) but not the carry/borrow-computed part (affected-
    subset accuracy stays close to that subset's own chance level, clearly
    below the baseline's own overall accuracy, even though it can drift
    slightly above chance itself -- a real, reproducible weak correlation
    between top digits and the true leading digit even under a carry,
    verified across 20 seeds during calibration, never exceeding roughly
    +0.11 over ``majority_affected_acc``).
    """

    CHARS = "-0123456789"  # every possible first answer character, incl. the
    # rare a==b subtraction case (answer '0') -- 9/100000 rows in the real
    # parquet, but common enough in a random draw over a 10000-value range
    # (birthday-paradox collisions) that the mapping must cover it.

    def _token_id(self, char: str) -> int:
        return 100 + self.CHARS.index(char)

    def _build(self, seed: int, n: int):
        rng = random.Random(seed)
        exs, triples = [], []
        for _ in range(n):
            a = rng.randrange(0, 10000)
            b = rng.randrange(0, 10000)
            op = rng.choice(["+", "-"])
            truth = str(a + b)[0] if op == "+" else str(a - b)[0]
            exs.append(SftExample(input_ids=[4, 5, self._token_id(truth)], label_span=(2, 3)))
            triples.append((a, b, op))
        return exs, triples

    def test_baseline_tracks_routing_not_carries(self):
        exs, triples = self._build(seed=99, n=3000)
        df = pd.DataFrame(
            {
                "a": [t[0] for t in triples],
                "b": [t[1] for t in triples],
                "op": [t[2] for t in triples],
            }
        )
        ref = rp.build_reference(exs, "task", seed=0)
        rr = prc.build_routing_reference(df, ref, l2=1e-3)

        # Overall: real, learnable signal above the majority-class floor --
        # a linear readout over LEFT-aligned routed digits recovers the
        # top-position-determined majority of examples directly.
        assert rr.token_baseline_acc > ref.majority_test_acc + 0.15
        # Affected subset: carry/borrow/cancellation/sign-tie logic is not a
        # linear function of one-hot digit positions, so the SAME baseline
        # stays close to that subset's own chance level -- well below its
        # own overall accuracy, and not dramatically above the affected
        # floor (calibration across 20 seeds never exceeded +0.11).
        assert rr.token_baseline_acc_affected < rr.token_baseline_acc - 0.2
        assert rr.token_baseline_acc_affected <= rr.majority_affected_acc + 0.15


class TestRefusals:
    def test_zero_affected_test_examples_raises(self):
        triples = [
            (1, 2, "+"),
            (3, 1, "+"),
            (9, 0, "-"),
            (500, 100, "-"),
            (26, 279, "-"),
            (5898, 3, "-"),
        ]
        assert all(prc.top_position_determined(*t) for t in triples)  # sanity: all determined
        exs = [SftExample(input_ids=[4 + i, 5 + i, 6 + i], label_span=(2, 3)) for i in range(6)]
        df = pd.DataFrame(
            {
                "a": [t[0] for t in triples],
                "b": [t[1] for t in triples],
                "op": [t[2] for t in triples],
            }
        )
        ref = rp.build_reference(exs, "task", seed=0)
        with pytest.raises(ValueError):
            prc.build_routing_reference(df, ref, l2=1e-3)

    def test_row_count_mismatch_raises(self):
        triples = [(943, 5881, "+"), (500, 480, "-")]
        exs = [SftExample(input_ids=[4, 5, 6], label_span=(2, 3))]  # 1 example, 2 df rows
        df = pd.DataFrame(
            {
                "a": [t[0] for t in triples],
                "b": [t[1] for t in triples],
                "op": [t[2] for t in triples],
            }
        )
        ref = rp.build_reference(exs, "task", seed=0)
        with pytest.raises(ValueError):
            prc.build_routing_reference(df, ref, l2=1e-3)

    def test_malformed_model_spec_raises(self):
        with pytest.raises(ValueError):
            prc._parse_model_arg("dir:/x/y")  # missing 'name='


class TestPlantedSignalSeparatesRoutingFromComputing:
    """The central discriminator this script exists to provide, checked
    causally against a real (tiny) forward pass rather than trusted by
    inspection: a probe that only recovers the naive top-position
    PREDICTION must score well on determined test examples but stay near
    chance on affected ones; a probe that recovers the TRUE class must
    score well on both.
    """

    N_CLASSES = 4
    ANSWER_TOKENS = [10, 20, 30, 40]
    SEQ_LEN = 8
    SPAN = (5, 7)
    N_EXAMPLES = 120
    TARGET_LAYER_IDX = 2
    SCALE = 60.0

    def _build_examples_and_affected_mask(self, seed):
        rng = random.Random(seed)
        exs = []
        determined = []
        for i in range(self.N_EXAMPLES):
            ids = [rng.randrange(4, 64) for _ in range(self.SEQ_LEN)]
            truth_idx = i % self.N_CLASSES
            ids[self.SPAN[0]] = self.ANSWER_TOKENS[truth_idx]
            exs.append(SftExample(input_ids=ids, label_span=self.SPAN))
            determined.append(i % 2 == 0)  # arbitrary split, unrelated to real arithmetic
        return exs, ~torch.tensor(determined, dtype=torch.bool)

    def _routing_reference(self, ref, affected_mask):
        y_test = ref.y[ref.test_idx]
        test_affected = affected_mask[ref.test_idx]
        return prc.RoutingReference(
            base=ref,
            affected_mask=affected_mask,
            determined_frac=float((~test_affected).double().mean().item()),
            majority_affected_acc=prc.majority_accuracy(y_test[test_affected], ref.n_classes),
            n_test_determined=int((~test_affected).sum().item()),
            n_test_affected=int(test_affected.sum().item()),
            token_baseline_acc=float("nan"),
            token_baseline_acc_affected=float("nan"),
        )

    def test_a_prediction_only_signal_high_on_determined_chance_on_affected(self, tiny_llama):
        model = tiny_llama(seed=90, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=True)
        exs, affected_mask = self._build_examples_and_affected_mask(seed=1)
        ref = rp.build_reference(exs, "task", seed=0)
        rr = self._routing_reference(ref, affected_mask)
        assert rr.n_test_affected > 0 and rr.n_test_determined > 0

        # pred_y: equals the TRUE class on determined examples (the naive
        # rule is right there, by construction); an INDEPENDENT random draw
        # -- decorrelated from truth -- on affected ones (the naive rule is
        # wrong there and carries no information about which class is true).
        gen_pred = torch.Generator().manual_seed(7)
        indep_draw = torch.randint(0, self.N_CLASSES, (self.N_EXAMPLES,), generator=gen_pred)
        pred_y = torch.where(affected_mask, indep_draw, ref.y)

        gen_dirs = torch.Generator().manual_seed(123)
        class_dirs = torch.randn(self.N_CLASSES, 32, generator=gen_dirs)

        def perturb_hook(_module, _inputs, output):
            add = (self.SCALE * class_dirs[pred_y]).unsqueeze(1)
            if isinstance(output, tuple):
                return (output[0] + add,) + output[1:]
            return output + add

        handle = model.model.layers[self.TARGET_LAYER_IDX - 1].register_forward_hook(perturb_hook)
        try:
            rows = prc.routing_probe_rows_for_model(model, {"task": rr}, "m", checkpoint_step=0)
        finally:
            handle.remove()

        df = pd.DataFrame(rows).set_index("layer").sort_index()
        for layer in range(self.TARGET_LAYER_IDX, len(df)):
            assert df.loc[layer, "acc_determined"] >= 0.9, (
                f"layer {layer}: determined not recovered"
            )
            assert df.loc[layer, "acc_affected"] <= df.loc[layer, "majority_affected_acc"] + 0.35, (
                f"layer {layer}: prediction-only signal leaked TRUE-class info on affected examples"
            )

    def test_b_true_class_signal_high_on_both_subsets(self, tiny_llama):
        model = tiny_llama(seed=91, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=True)
        exs, affected_mask = self._build_examples_and_affected_mask(seed=2)
        ref = rp.build_reference(exs, "task", seed=0)
        rr = self._routing_reference(ref, affected_mask)
        assert rr.n_test_affected > 0 and rr.n_test_determined > 0

        gen_dirs = torch.Generator().manual_seed(456)
        class_dirs = torch.randn(self.N_CLASSES, 32, generator=gen_dirs)

        def perturb_hook(_module, _inputs, output):
            add = (self.SCALE * class_dirs[ref.y]).unsqueeze(1)
            if isinstance(output, tuple):
                return (output[0] + add,) + output[1:]
            return output + add

        handle = model.model.layers[self.TARGET_LAYER_IDX - 1].register_forward_hook(perturb_hook)
        try:
            rows = prc.routing_probe_rows_for_model(model, {"task": rr}, "m", checkpoint_step=0)
        finally:
            handle.remove()

        df = pd.DataFrame(rows).set_index("layer").sort_index()
        for layer in range(self.TARGET_LAYER_IDX, len(df)):
            assert df.loc[layer, "acc_determined"] >= 0.9
            assert df.loc[layer, "acc_affected"] >= 0.9, (
                f"layer {layer}: a representation encoding the TRUE class must score high on "
                "affected examples too -- this is what distinguishes it from routing"
            )


def _save_tiny_model(tiny_llama, tmp_path, seed, name):
    model = tiny_llama(seed=seed, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
    out = tmp_path / name
    model.save_pretrained(out)
    return out


def _save_tokenizer(tiny_tokenizer, tmp_path):
    tok = tiny_tokenizer(vocab_size=32)
    out = tmp_path / "tokenizer"
    tok.save_pretrained(out)
    return out


def _routing_parquet(tmp_path):
    texts = [f"t1 t2 t{i}" for i in range(3, 9)]  # 6 rows
    starts = [len(t) - len(t.split()[-1]) for t in texts]
    ends = [len(t) for t in texts]
    triples = [
        (943, 5881, "+"),
        (500, 480, "-"),
        (523, 480, "-"),
        (45, 47, "+"),
        (55, 55, "-"),
        (91, 19, "-"),
    ]  # all AFFECTED (per TestTopPositionRule / hand-checked) -- guarantees
    # build_routing_reference won't refuse for lacking affected test examples.
    assert all(not prc.top_position_determined(*t) for t in triples)
    df = pd.DataFrame(
        {
            "full_text": texts,
            "answer_char_start": starts,
            "answer_char_end": ends,
            "a": [t[0] for t in triples],
            "b": [t[1] for t in triples],
            "op": [t[2] for t in triples],
        }
    )
    p = tmp_path / "task.parquet"
    df.to_parquet(p)
    return p


class TestMainSmoke:
    def test_main_smoke(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        model_dir = _save_tiny_model(tiny_llama, tmp_path, 70, "model")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _routing_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "probe_routing_control.py",
            "--model",
            f"m1=dir:{model_dir}",
            "--prompt-parquet",
            str(parquet),
            "--set-name",
            "task",
            "--tokenizer",
            str(tok_dir),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        prc.main()
        assert out.is_file()
        df = pd.read_csv(out)
        expected_cols = {
            "model",
            "checkpoint_step",
            "set",
            "layer",
            "hook_name",
            "probe_test_acc",
            "acc_determined",
            "acc_affected",
            "n_test_determined",
            "n_test_affected",
            "majority_test_acc",
            "majority_affected_acc",
            "determined_frac",
            "token_baseline_acc",
            "token_baseline_acc_affected",
            "n_classes",
            "n_train",
            "n_test",
        }
        assert expected_cols.issubset(df.columns)
        assert (df["model"] == "m1").all()
        assert (df["checkpoint_step"] == prc.NO_CHECKPOINT_STEP).all()
        assert (df["n_test_determined"] == 0).all()  # every row is AFFECTED by construction

    def test_malformed_model_spec_argv_propagates(self, tiny_tokenizer, tmp_path, monkeypatch):
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _routing_parquet(tmp_path)
        argv = [
            "probe_routing_control.py",
            "--model",
            "dir:/nonexistent",  # malformed: no 'name='
            "--prompt-parquet",
            str(parquet),
            "--set-name",
            "task",
            "--tokenizer",
            str(tok_dir),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(ValueError):
            prc.main()
