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
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Cross-format transfer probe (decisions.md 2026-08-22 "probe trajectory").
# ---------------------------------------------------------------------------


class TestOpTwinFrame:
    def test_exact_strings_known_rows(self):
        df = pd.DataFrame({"a": [189, 560], "b": [937, 1188], "op": ["+", "-"]})
        twin = prc.op_twin_frame(df)
        row0, row1 = twin.iloc[0], twin.iloc[1]
        assert row0["full_text"] == "Question: 189 + 937\nAnswer: 1126"
        assert (row0["answer_char_start"], row0["answer_char_end"]) == (28, 32)
        assert row0["answer_text"] == "1126"
        assert row1["full_text"] == "Question: 560 - 1188\nAnswer: -628"
        assert row1["answer_text"] == "-628"

    def test_row_order_preserved(self):
        rows = [(1, 2, "+"), (500, 480, "-"), (943, 5881, "+")]
        df = pd.DataFrame(
            {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
        )
        twin = prc.op_twin_frame(df)
        assert twin["a"].tolist() == [r[0] for r in rows]
        assert twin["b"].tolist() == [r[1] for r in rows]
        assert twin["op"].tolist() == [r[2] for r in rows]

    def test_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "op": ["+"]})
        twin = prc.op_twin_frame(df)
        assert set(twin.columns) == {
            "a",
            "b",
            "op",
            "full_text",
            "answer_char_start",
            "answer_char_end",
            "answer_text",
        }

    def test_span_slices_to_answer_text_every_row(self):
        rng = random.Random(17)
        rows = [
            (rng.randrange(0, 100000), rng.randrange(0, 100000), rng.choice(["+", "-"]))
            for _ in range(25)
        ]
        df = pd.DataFrame(
            {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
        )
        twin = prc.op_twin_frame(df)
        for row in twin.itertuples(index=False):
            assert row.full_text[row.answer_char_start : row.answer_char_end] == row.answer_text

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            prc.op_twin_frame(pd.DataFrame({"a": [1], "op": ["+"]}))
        with pytest.raises(ValueError):
            prc.op_twin_frame(pd.DataFrame({"a": [1], "b": [2]}))


class TestPositiveFirstDigitClasses:
    def test_hand_computed(self):
        rows = [
            (560, 1188, "-"),  # -628 -> -1
            (55, 55, "-"),  # a==b, answer 0 -> -1
            (189, 937, "+"),  # 1126 -> class 0
            (0, 9, "+"),  # 9 -> class 8
            (5898, 3, "-"),  # 5895 -> class 4
        ]
        df = pd.DataFrame(
            {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
        )
        y = prc.positive_first_digit_classes(df)
        assert y.tolist() == [-1, -1, 0, 8, 4]
        assert y.dtype == torch.long

    def test_length_matches_df(self):
        rng = random.Random(9)
        rows = [
            (rng.randrange(0, 100000), rng.randrange(0, 100000), rng.choice(["+", "-"]))
            for _ in range(13)
        ]
        df = pd.DataFrame(
            {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
        )
        y = prc.positive_first_digit_classes(df)
        assert y.shape[0] == len(df) == 13

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            prc.positive_first_digit_classes(pd.DataFrame({"a": [1], "op": ["+"]}))


def _synthetic_examples(n: int, offset: int = 0) -> list[SftExample]:
    """``n`` distinct-token SftExamples -- content is irrelevant to the
    transfer reference / feats-based tests below, only the count matters."""
    return [
        SftExample(
            input_ids=[4 + offset + (i % 10), 5 + offset + (i % 10), 6 + offset + (i % 10)],
            label_span=(2, 3),
        )
        for i in range(n)
    ]


def _routing_reference_for_transfer(rows, exs_seed, split_seed):
    """Builds ``(df, rr)`` via the real ``build_routing_reference`` path, the
    way ``build_transfer_reference``'s docstring/callers do."""
    df = pd.DataFrame(
        {"a": [r[0] for r in rows], "b": [r[1] for r in rows], "op": [r[2] for r in rows]}
    )
    exs = _synthetic_examples(len(rows), offset=exs_seed)
    ref = rp.build_reference(exs, "task", seed=split_seed)
    rr = prc.build_routing_reference(df, ref, l2=1e-3)
    return df, rr


class TestBuildTransferReference:
    def test_refuses_twin_length_mismatch(self):
        rows = [(a, b, "-") for a, b in [(500, 480), (523, 480), (55, 55), (91, 19)]]
        df, rr = _routing_reference_for_transfer(rows, exs_seed=0, split_seed=0)
        short_twin = _synthetic_examples(len(rows) - 1, offset=100)
        with pytest.raises(ValueError):
            prc.build_transfer_reference(df, rr, short_twin)

    def test_refuses_zero_affected_in_positive_test_subset(self):
        # index0: determined, positive (train). index1: determined, positive
        # (test). index2: affected, negative (train filler). index3:
        # affected, negative (test) -- the only affected row in the test
        # half is NOT positive, so the positive test subset has zero
        # affected examples.
        rows = [
            (7465, 8497, "+"),  # determined, +15962
            (5898, 3, "-"),  # determined, +5895
            (120, 150, "-"),  # affected (ta==tb), -30
            (311, 350, "-"),  # affected (ta==tb), -39
        ]
        assert [prc.top_position_determined(*r) for r in rows] == [True, True, False, False]
        df, rr = _routing_reference_for_transfer(rows, exs_seed=0, split_seed=33)
        assert rr.base.train_idx.tolist() == [0, 2] and rr.base.test_idx.tolist() == [1, 3]
        assert rr.n_test_affected == 1  # build_routing_reference itself is happy
        twin = _synthetic_examples(len(rows), offset=200)
        with pytest.raises(ValueError):
            prc.build_transfer_reference(df, rr, twin)

    def test_refuses_no_positive_examples_at_all(self):
        rows = [
            (120, 150, "-"),  # affected, -30
            (26, 279, "-"),  # determined, -253
            (200, 300, "-"),  # determined, -100
            (150, 180, "-"),  # affected, -30
        ]
        df, rr = _routing_reference_for_transfer(rows, exs_seed=0, split_seed=0)
        assert rr.n_test_affected > 0  # build_routing_reference itself is happy
        twin = _synthetic_examples(len(rows), offset=200)
        with pytest.raises(ValueError):
            prc.build_transfer_reference(df, rr, twin)

    def test_happy_path_hand_computable(self):
        rng = random.Random(42)
        n = 30
        rows = [
            (rng.randrange(1000, 5000), rng.randrange(1000, 5000), rng.choice(["+", "-"]))
            for _ in range(n)
        ]
        df, rr = _routing_reference_for_transfer(rows, exs_seed=0, split_seed=3)
        twin = _synthetic_examples(n, offset=100)
        tr = prc.build_transfer_reference(df, rr, twin)

        y_digit = prc.positive_first_digit_classes(df)
        expected_train = sorted(i for i in rr.base.train_idx.tolist() if y_digit[i] >= 0)
        expected_test = sorted(i for i in rr.base.test_idx.tolist() if y_digit[i] >= 0)
        assert sorted(tr.train_idx.tolist()) == expected_train
        assert sorted(tr.test_idx.tolist()) == expected_test
        assert set(tr.train_idx.tolist()) <= set(rr.base.train_idx.tolist())
        assert set(tr.test_idx.tolist()) <= set(rr.base.test_idx.tolist())

        y_test = y_digit[tr.test_idx]
        expected_majority = prc.majority_accuracy(y_test, len(prc.TRANSFER_CLASSES))
        assert math.isclose(tr.majority_acc, expected_majority, rel_tol=1e-12)

        test_affected = rr.affected_mask[tr.test_idx]
        expected_majority_affected = prc.majority_accuracy(
            y_test[test_affected], len(prc.TRANSFER_CLASSES)
        )
        assert math.isclose(tr.majority_affected_acc, expected_majority_affected, rel_tol=1e-12)
        assert tr.n_test_affected == int(test_affected.sum().item())


class TestTransferRowsFromFeats:
    """Synthetic-feature key-property test (no model, no forward pass): a
    representation that is the SAME in both formats must transfer; one that
    is format-specific must not.
    """

    N = 400
    D = 32
    LAYERS = ("l0", "l1")

    def _tr(self):
        rng = random.Random(21)
        rows = [(rng.randrange(1000, 5000), rng.randrange(1000, 5000), "+") for _ in range(self.N)]
        df, rr = _routing_reference_for_transfer(rows, exs_seed=0, split_seed=7)
        twin = _synthetic_examples(self.N, offset=100)
        return prc.build_transfer_reference(df, rr, twin)

    def _feats_from_dirs(
        self, dirs: torch.Tensor, y: torch.Tensor, gen: torch.Generator
    ) -> torch.Tensor:
        return dirs[y] + 0.5 * torch.randn(y.shape[0], dirs.shape[1], generator=gen)

    def test_shared_representation_transfers(self):
        tr = self._tr()
        n_classes = len(prc.TRANSFER_CLASSES)
        class_dirs = (
            torch.randn(n_classes, self.D, generator=torch.Generator().manual_seed(123)) * 5.0
        )
        feats_task = {
            layer: self._feats_from_dirs(
                class_dirs, tr.y_digit, torch.Generator().manual_seed(10 + i)
            )
            for i, layer in enumerate(self.LAYERS)
        }
        feats_op = {
            layer: self._feats_from_dirs(
                class_dirs, tr.y_digit, torch.Generator().manual_seed(20 + i)
            )
            for i, layer in enumerate(self.LAYERS)
        }
        rows = prc.transfer_rows_from_feats(tr, feats_task, feats_op, "m", 0, l2=1e-3)
        df_rows = pd.DataFrame(rows)
        assert len(df_rows) == len(prc.TRANSFER_DIRECTIONS) * len(self.LAYERS)
        assert set(df_rows.columns) == {
            "model",
            "checkpoint_step",
            "set",
            "direction",
            "layer",
            "hook_name",
            "acc",
            "acc_affected",
            "majority_acc",
            "majority_affected_acc",
            "n_train",
            "n_test",
            "n_test_affected",
            "n_classes",
        }
        for direction in ("op_to_task", "task_to_op"):
            sub = df_rows[df_rows["direction"] == direction]
            assert (sub["acc"] >= 0.9).all()
            assert (sub["acc_affected"] >= 0.9).all()

    def test_format_specific_representation_does_not_transfer(self):
        tr = self._tr()
        n_classes = len(prc.TRANSFER_CLASSES)
        dirs_task = (
            torch.randn(n_classes, self.D, generator=torch.Generator().manual_seed(200)) * 5.0
        )
        dirs_op = torch.randn(n_classes, self.D, generator=torch.Generator().manual_seed(300)) * 5.0
        feats_task = {
            layer: self._feats_from_dirs(
                dirs_task, tr.y_digit, torch.Generator().manual_seed(21 + i)
            )
            for i, layer in enumerate(self.LAYERS)
        }
        feats_op = {
            layer: self._feats_from_dirs(dirs_op, tr.y_digit, torch.Generator().manual_seed(23 + i))
            for i, layer in enumerate(self.LAYERS)
        }
        rows = prc.transfer_rows_from_feats(tr, feats_task, feats_op, "m", 0, l2=1e-3)
        df_rows = pd.DataFrame(rows)
        for direction in ("op_to_op", "task_to_task"):
            sub = df_rows[df_rows["direction"] == direction]
            assert (sub["acc"] >= 0.9).all()
        for direction in ("op_to_task", "task_to_op"):
            sub = df_rows[df_rows["direction"] == direction]
            majority_ceiling = sub["majority_affected_acc"].iloc[0] + 0.25
            assert (sub["acc"] <= majority_ceiling).all()
            assert (sub["acc_affected"] <= majority_ceiling).all()

    def test_raises_on_layer_mismatch(self):
        tr = self._tr()
        feats_task = {"l0": torch.randn(self.N, self.D)}
        feats_op = {"l1": torch.randn(self.N, self.D)}
        with pytest.raises(ValueError):
            prc.transfer_rows_from_feats(tr, feats_task, feats_op, "m", 0)


class TestProbeModel:
    def _rr(self):
        rng = random.Random(50)
        rows = [(rng.randrange(1000, 5000), rng.randrange(1000, 5000), "+") for _ in range(20)]
        return _routing_reference_for_transfer(rows, exs_seed=0, split_seed=1)

    def test_empty_transfer_matches_routing_only(self, tiny_llama):
        model = tiny_llama(seed=40, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        _df, rr = self._rr()
        torch.manual_seed(1)
        routing_rows, transfer_rows = prc.probe_model(model, {"task": rr}, {}, "m", 0, l2=1e-3)
        assert transfer_rows == []
        torch.manual_seed(999)  # fit is deterministic -- RNG state must not matter
        expected = prc.routing_probe_rows_for_model(model, {"task": rr}, "m", 0, l2=1e-3)
        assert routing_rows == expected
        assert len(routing_rows) > 0

    def test_transfer_set_produces_rows(self, tiny_llama):
        model = tiny_llama(seed=41, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        df, rr = self._rr()
        twin = _synthetic_examples(len(df), offset=20)
        tr = prc.build_transfer_reference(df, rr, twin)
        routing_rows, transfer_rows = prc.probe_model(
            model, {"task": rr}, {"task": tr}, "m", 0, l2=1e-3
        )
        assert routing_rows
        assert transfer_rows
        assert {r["direction"] for r in transfer_rows} == set(prc.TRANSFER_DIRECTIONS)
        assert all(r["model"] == "m" and r["checkpoint_step"] == 0 for r in transfer_rows)


def _fullft_snapshot_run(tiny_llama, store: Path, run_id: str, steps: tuple[int, ...]) -> None:
    snap_dir = store / "runs" / run_id / "sft_snapshots"
    for step in steps:
        _save_tiny_model(tiny_llama, snap_dir, seed=100 + step, name=f"step_{step}")


class TestMainRunIdSweep:
    def test_full_ft_snapshot_sweep(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        store = tmp_path / "store"
        _fullft_snapshot_run(tiny_llama, store, "evt-fake-fullft", (1, 3))
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _routing_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "probe_routing_control.py",
            "--run-id",
            "evt-fake-fullft",
            "--model-name",
            "m",
            "--prompt-parquet",
            str(parquet),
            "--set-name",
            "task",
            "--tokenizer",
            str(tok_dir),
            "--store",
            str(store),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        prc.main()
        df = pd.read_csv(out)
        assert set(df["checkpoint_step"]) == {1, 3}
        assert (df["model"] == "m").all()

    def test_snapshot_steps_selects_subset(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        store = tmp_path / "store"
        _fullft_snapshot_run(tiny_llama, store, "evt-fake-fullft2", (1, 3))
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _routing_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "probe_routing_control.py",
            "--run-id",
            "evt-fake-fullft2",
            "--snapshot-steps",
            "3",
            "--prompt-parquet",
            str(parquet),
            "--set-name",
            "task",
            "--tokenizer",
            str(tok_dir),
            "--store",
            str(store),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        prc.main()
        df = pd.read_csv(out)
        assert set(df["checkpoint_step"]) == {3}

    def test_missing_snapshot_step_raises_system_exit(
        self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch
    ):
        store = tmp_path / "store"
        _fullft_snapshot_run(tiny_llama, store, "evt-fake-fullft3", (1, 3))
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _routing_parquet(tmp_path)
        argv = [
            "probe_routing_control.py",
            "--run-id",
            "evt-fake-fullft3",
            "--snapshot-steps",
            "5",
            "--prompt-parquet",
            str(parquet),
            "--set-name",
            "task",
            "--tokenizer",
            str(tok_dir),
            "--store",
            str(store),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            prc.main()


class TestMainArgparseRefusals:
    def _base_argv(self, extra: list[str]) -> list[str]:
        return [
            "probe_routing_control.py",
            *extra,
            "--prompt-parquet",
            "nonexistent.parquet",
            "--set-name",
            "task",
        ]

    def test_both_model_and_run_id_raises(self, monkeypatch):
        argv = self._base_argv(["--model", "m=dir:/x", "--run-id", "rid"])
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            prc.main()

    def test_neither_model_nor_run_id_raises(self, monkeypatch):
        argv = self._base_argv([])
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            prc.main()

    def test_snapshot_steps_without_run_id_raises(self, monkeypatch):
        argv = self._base_argv(["--model", "m=dir:/x", "--snapshot-steps", "1"])
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            prc.main()

    def test_transfer_set_not_in_set_names_raises(self, monkeypatch):
        argv = self._base_argv(["--model", "m=dir:/x", "--transfer-set", "other"])
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            prc.main()


def _transfer_smoke_parquet(tmp_path: Path) -> Path:
    """A bigger (20-row) routing parquet for the ``--transfer-set`` CLI
    smoke: mostly positive additions (some with carries), a handful of
    subtractions, real ``a``/``b``/``op`` so the operator-notation twin and
    the routing split both carry genuine signal. ``full_text`` uses
    tiny-vocab words (the tokenizer's [UNK] path aligns spans fine -- see
    module docstring) matched with dummy answer_char span"""
    rng = random.Random(2)
    n = 20
    rows = [
        (rng.randrange(100, 9999), rng.randrange(100, 9999), rng.choice(["+", "+", "+", "-"]))
        for _ in range(n)
    ]
    texts = [f"t1 t2 t{3 + (i % 25)}" for i in range(n)]
    starts = [len(t) - len(t.split()[-1]) for t in texts]
    ends = [len(t) for t in texts]
    df = pd.DataFrame(
        {
            "full_text": texts,
            "answer_char_start": starts,
            "answer_char_end": ends,
            "a": [r[0] for r in rows],
            "b": [r[1] for r in rows],
            "op": [r[2] for r in rows],
        }
    )
    p = tmp_path / "task_transfer.parquet"
    df.to_parquet(p)
    return p


class TestMainTransferSetSmoke:
    def test_transfer_set_writes_transfer_csv(
        self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch
    ):
        model_dir = _save_tiny_model(tiny_llama, tmp_path, 70, "model")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _transfer_smoke_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "probe_routing_control.py",
            "--model",
            f"m1=dir:{model_dir}",
            "--prompt-parquet",
            str(parquet),
            "--set-name",
            "task",
            "--transfer-set",
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
        transfer_out = out.with_name(f"{out.stem}_transfer.csv")
        assert transfer_out.is_file()
        df = pd.read_csv(transfer_out)
        expected_cols = {
            "model",
            "checkpoint_step",
            "set",
            "direction",
            "layer",
            "hook_name",
            "acc",
            "acc_affected",
            "majority_acc",
            "majority_affected_acc",
            "n_train",
            "n_test",
            "n_test_affected",
            "n_classes",
        }
        assert expected_cols.issubset(df.columns)
        assert set(df["direction"]) == set(prc.TRANSFER_DIRECTIONS)
