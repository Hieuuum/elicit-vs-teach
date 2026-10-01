"""``circuit_jaccard.py`` (ts38mt Tier-3 test 2, gated on Tier 1): do two
models' top-k node/edge circuits overlap?

Silent failure modes guarded:

- A wrong tie-break in ``top_k`` (hash/insertion order instead of
  ``(-|score|, str(key))``) would make circuits non-reproducible across
  runs/machines without ever raising -- checked by hand-computed exact-set
  equality on a deliberately tied input.
- ``jaccard``'s both-empty case silently returning 0.0 instead of NaN would
  make a genuinely-undefined comparison look like "zero overlap" (a real,
  meaningful, DIFFERENT claim) -- checked against both branches.
- ``spearman``'s tie handling and constant-input guard (torch-only,
  hand-rolled rank averaging, no scipy) -- checked against exact hand
  computations, a perfect anti-correlation (-1), and the NaN branch.
- ``mass_overlap``'s asymmetry (swapping which side's mass is the
  denominator changes the number) -- checked against an exact hand
  computation showing both directions differ on the same intersection.
- ``circuit_scores``'s batching must not change the reported mean(|score|)
  under ``ablation="zero"`` -- checked batch_size=2 vs batch_size=len(examples)
  to 1e-6. Conversely, under the DEFAULT ``ablation="mean"`` batching MUST
  change it (batch-local ablation reference, documented in the script's own
  docstring) -- pinned as a checked fact, not left as an untested caveat, so
  a future change that silently made "mean" ablation batch-invariant would
  be caught.
- A model whose forward pass now routes the answer through one artificially
  dominant head (the same "hook plants a huge answer direction on one
  head's o_proj input" construction ``test_mech_nodes.py`` uses for
  ``attribution_scores``, here installed as a PERMANENT hook so it survives
  every internal forward call ``circuit_scores`` makes) must make that node
  the model's own top-1 circuit member, drop the pair jaccard against an
  unplanted twin, and raise concentration -- checked end to end through
  ``circuit_jaccard_rows``, not just the underlying attribution math (that
  math is already covered in ``test_mech_nodes.py``; this file only owns
  the aggregation/circuit-comparison layer built on top of it).
- ``top_k``'s budget-clipping (k > available count) must not crash, and
  must degrade to "the whole set" rather than silently truncating to
  nothing or raising -- checked at both the pure-function and the
  ``circuit_jaccard_rows`` level.
- Edge-key string serialisation must round-trip exactly (a lossy
  serialisation would silently corrupt every edge circuit's printed/CSV
  form without any crash) -- checked in both directions plus garbage
  rejection.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``/
``tiny_tokenizer``), no network, no real checkpoints.
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


# Load order matters: circuit_jaccard.py does `from mech_nodes import NodeId`
# at its own module-exec time. `load()` always execs a FRESH module object
# (unlike a normal `import`, which reuses `sys.modules`), so if "mech_nodes"
# loads AFTER circuit_jaccard.py, `cj`'s bound `NodeId` and `mn.NodeId` end
# up as two structurally-identical but DIFFERENT classes -- the dataclass
# `__eq__` then fails on same-field instances (checked via `.__class__ is`),
# silently breaking every `mn.NodeId(...) == cj...NodeId(...)` comparison
# below. Load `mech_nodes` first so circuit_jaccard's own import reuses it.
mn = load("mech_nodes")
cj = load("circuit_jaccard")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


def _fixed_target_examples(
    rng: random.Random,
    n: int,
    vocab_size: int,
    seq_len: int,
    span: tuple[int, int],
    target_token: int,
):
    """Like ``_examples`` but every example's answer token is forced to
    ``target_token`` -- lets the planted-head test use one fixed plant
    direction instead of a per-example one."""
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        ids[span[0]] = target_token
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


def _install_causal_head_plant(model, layer: int, head: int, target_token: int, huge: float = 3.0):
    """Permanently hooks head ``head`` of block ``layer``'s o_proj input to
    add ``huge * embed[target_token]`` in the direction recoverable through
    that head's o_proj weight slice -- the exact construction
    ``test_mech_nodes.py``'s ``test_planted_causal_head_dominates_by_far``
    uses, just installed as a raw persistent hook (not
    ``mn.patch_nodes``'s context manager) since ``circuit_scores`` calls
    ``attribution_scores``/``edge_attribution_scores`` directly with no
    patching context of its own. The plant direction is FIXED (not
    per-example), so every example scored against this model must share
    ``target_token`` as its own answer (``_fixed_target_examples``).
    """
    with torch.no_grad():
        embed = model.get_input_embeddings().weight.detach()
        n_heads, d_head = mn._num_heads_and_dhead(model)
        _, blocks = mn.residual_modules(model)
        o_proj = mn._o_proj_module(blocks[layer])
        w, _bias = mn._o_proj_weight_bias(o_proj)
        w_slice = w[:, head * d_head : (head + 1) * d_head]
        w_pinv = torch.linalg.pinv(w_slice)  # [d_head, d_model]
        delta_head = w_pinv @ (huge * embed[target_token])  # [d_head]

    def hook(_module, args):
        x = args[0].clone()
        sl = slice(head * d_head, (head + 1) * d_head)
        x[..., sl] = x[..., sl] + delta_head
        return (x,) + args[1:]

    o_proj.register_forward_pre_hook(hook)


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------


class TestTopK:
    def test_hand_computed(self):
        scores = {"a": 5.0, "b": -3.0, "c": 1.0, "d": -4.5}
        assert cj.top_k(scores, 2) == {"a", "d"}

    def test_tie_break_deterministic(self):
        # a, b, c tied at |score|=2.0 -> alphabetical: a, b, c; d is last.
        scores = {"b": 2.0, "a": 2.0, "c": -2.0, "d": 1.0}
        assert cj.top_k(scores, 2) == {"a", "b"}

    def test_k_larger_than_dict_clips_without_crash(self):
        scores = {"a": 1.0, "b": 2.0}
        assert cj.top_k(scores, 100) == {"a", "b"}

    def test_k_zero_gives_empty_set(self):
        assert cj.top_k({"a": 1.0}, 0) == set()

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            cj.top_k({"a": 1.0}, -1)


class TestJaccard:
    def test_hand_computed_partial_overlap(self):
        assert math.isclose(cj.jaccard({1, 2, 3}, {2, 3, 4}), 0.5)

    def test_both_empty_is_nan(self):
        assert math.isnan(cj.jaccard(set(), set()))

    def test_one_empty_is_zero_not_nan(self):
        assert cj.jaccard(set(), {1, 2}) == 0.0

    def test_identical_sets_is_one(self):
        assert cj.jaccard({1, 2}, {1, 2}) == 1.0

    def test_disjoint_sets_is_zero(self):
        assert cj.jaccard({1}, {2}) == 0.0


class TestSpearman:
    def test_hand_computed_no_ties(self):
        # x ranks: 10->1, 15->2, 20->3; y ranks match exactly -> rho = 1.
        assert math.isclose(cj.spearman([10.0, 20.0, 15.0], [1.0, 3.0, 2.0]), 1.0, abs_tol=1e-9)

    def test_perfect_anticorrelation_is_negative_one(self):
        assert math.isclose(
            cj.spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]), -1.0, abs_tol=1e-9
        )

    def test_ties_use_average_rank_and_still_correlate_perfectly(self):
        x = [1.0, 2.0, 2.0, 3.0]
        y = [1.0, 2.0, 2.0, 3.0]
        assert math.isclose(cj.spearman(x, y), 1.0, abs_tol=1e-9)

    def test_ties_hand_computed_partial_correlation(self):
        # x ranks (avg-tie): [1, 2.5, 2.5, 4]; y ranks: [1, 2, 3, 4] (no ties).
        # Pearson-on-ranks by hand: rx=[1,2.5,2.5,4]-2.5=[-1.5,0,0,1.5],
        # ry=[1,2,3,4]-2.5=[-1.5,-0.5,0.5,1.5]. cov=(-1.5*-1.5)+(0)+(0)+(1.5*1.5)=4.5
        # var_x=1.5^2*2=4.5, var_y=1.5^2+0.5^2+0.5^2+1.5^2=5.0
        x = [1.0, 2.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0]
        expected = 4.5 / math.sqrt(4.5 * 5.0)
        assert math.isclose(cj.spearman(x, y), expected, abs_tol=1e-9)

    def test_constant_input_is_nan(self):
        assert math.isnan(cj.spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]))

    def test_constant_both_is_nan(self):
        assert math.isnan(cj.spearman([5.0, 5.0], [7.0, 7.0]))

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            cj.spearman([1.0, 2.0], [1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cj.spearman([], [])


class TestMassOverlap:
    def test_hand_computed(self):
        scores_a = {"x": 10.0, "y": 5.0, "z": 1.0}
        top_a = {"x", "y"}
        top_b = {"y", "z"}
        assert math.isclose(cj.mass_overlap(scores_a, top_a, top_b), 5.0 / 15.0)

    def test_full_overlap_is_one(self):
        scores_a = {"x": 3.0, "y": 2.0}
        assert cj.mass_overlap(scores_a, {"x", "y"}, {"x", "y", "z"}) == 1.0

    def test_no_overlap_is_zero(self):
        scores_a = {"x": 3.0, "y": 2.0}
        assert cj.mass_overlap(scores_a, {"x", "y"}, {"z"}) == 0.0

    def test_empty_top_a_is_nan(self):
        assert math.isnan(cj.mass_overlap({"x": 1.0}, set(), {"x"}))

    def test_asymmetric_direction_matters(self):
        # Same intersection ({"y"}), but each side's OWN top-set mass
        # differs, so the two directions must give different numbers.
        scores_a = {"x": 9.0, "y": 1.0}
        assert math.isclose(cj.mass_overlap(scores_a, {"x", "y"}, {"y"}), 0.1)


class TestConcentration:
    def test_hand_computed(self):
        scores = {"a": 8.0, "b": 1.0, "c": 1.0}
        assert math.isclose(cj.concentration(scores, {"a"}), 0.8)

    def test_full_set_is_one(self):
        scores = {"a": 3.0, "b": -2.0}
        assert cj.concentration(scores, {"a", "b"}) == 1.0

    def test_empty_top_is_zero(self):
        assert cj.concentration({"a": 3.0}, set()) == 0.0

    def test_all_zero_scores_is_nan(self):
        assert math.isnan(cj.concentration({"a": 0.0, "b": 0.0}, {"a"}))


class TestEdgeKeySerialization:
    def test_to_str_head_to_mlp(self):
        assert cj.edge_key_to_str(mn.NodeId(0, "head", 1), mn.NodeId(2, "mlp", 0)) == "a0.h1->m2"

    def test_to_str_mlp_to_out(self):
        assert cj.edge_key_to_str(mn.NodeId(3, "mlp", 0), "out") == "m3->out"

    def test_to_str_head_to_attn_sink(self):
        assert cj.edge_key_to_str(mn.NodeId(0, "head", 2), mn.NodeId(1, "attn", 0)) == "a0.h2->a1"

    def test_round_trip(self):
        pairs = [
            (mn.NodeId(0, "head", 1), mn.NodeId(2, "mlp", 0)),
            (mn.NodeId(3, "mlp", 0), "out"),
            (mn.NodeId(1, "head", 0), mn.NodeId(4, "attn", 0)),
        ]
        for u, v in pairs:
            assert cj.edge_key_from_str(cj.edge_key_to_str(u, v)) == (u, v)

    def test_from_str_rejects_garbage(self):
        for bad in ("", "nosep", "a0.h1", "bogus->m2", "a0.h1->bogus"):
            with pytest.raises(ValueError):
                cj.edge_key_from_str(bad)


class TestPairRowsHelper:
    def test_raises_on_no_common_keys(self):
        with pytest.raises(ValueError):
            cj._pair_rows("a", "b", {"x": 1.0}, {"y": 2.0}, (1,), "node", 5)


class TestArgParsing:
    def test_parse_model_arg(self):
        assert cj._parse_model_arg("pp0=dir:/x/y") == ("pp0", "dir:/x/y")

    def test_parse_model_arg_rejects_missing_equals(self):
        with pytest.raises(ValueError):
            cj._parse_model_arg("dir:/x/y")

    def test_parse_budgets(self):
        assert cj._parse_budgets("5,10,20") == (5, 10, 20)


# --------------------------------------------------------------------------
# Model plumbing / end-to-end
# --------------------------------------------------------------------------


class TestCircuitScores:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(
            seed=200, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        with pytest.raises(ValueError):
            cj.circuit_scores(model, [])

    def test_invalid_batch_size_raises(self, tiny_llama):
        model = tiny_llama(
            seed=201, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        exs = [SftExample(input_ids=[4, 5, 6, 7], label_span=(2, 4))]
        with pytest.raises(ValueError):
            cj.circuit_scores(model, exs, batch_size=0)

    def test_batching_invariance_with_zero_ablation(self, tiny_llama):
        model = tiny_llama(
            seed=202, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(60)
        exs = _examples(rng, n=6, vocab_size=64, seq_len=8, span=(5, 7))
        node_whole, edge_whole = cj.circuit_scores(model, exs, ablation="zero", batch_size=len(exs))
        node_chunked, edge_chunked = cj.circuit_scores(model, exs, ablation="zero", batch_size=2)
        assert set(node_whole) == set(node_chunked)
        assert set(edge_whole) == set(edge_chunked)
        for nid in node_whole:
            assert math.isclose(node_whole[nid], node_chunked[nid], abs_tol=1e-6), nid
        for key in edge_whole:
            assert math.isclose(edge_whole[key], edge_chunked[key], abs_tol=1e-6), key

    def test_batching_NON_invariance_with_mean_ablation(self, tiny_llama):
        """Pins the caveat documented in ``circuit_scores``'s own docstring:
        unlike ``ablation="zero"``, the default ``ablation="mean"`` uses a
        batch-LOCAL reference (``attribution_scores``'s own per-call "mean"
        semantics -- no ``reference_acts`` passed here), so chunking the
        SAME example set differently must give a genuinely DIFFERENT
        mean(|score|), not just float noise. This converts the module
        docstring's caveat into a checked, named fact: if a future change
        silently made "mean" ablation batch-invariant (e.g. an accidental
        shared/cached reference), this test would need updating -- that's
        the point, not a bug in the test.
        """
        model = tiny_llama(
            seed=208, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(64)
        exs = _examples(rng, n=6, vocab_size=64, seq_len=8, span=(5, 7))
        node_whole, _edge_whole = cj.circuit_scores(
            model, exs, ablation="mean", batch_size=len(exs)
        )
        node_chunked, _edge_chunked = cj.circuit_scores(model, exs, ablation="mean", batch_size=2)
        assert set(node_whole) == set(node_chunked)
        max_diff = max(abs(node_whole[nid] - node_chunked[nid]) for nid in node_whole)
        assert max_diff > 1e-4, (
            "expected ablation='mean' to be batch-size DEPENDENT (batch-local "
            "reference), but chunking left every node score unchanged"
        )


class TestCircuitJaccardRows:
    def test_empty_models_dict_raises(self):
        exs = [SftExample(input_ids=[4, 5, 6, 7], label_span=(2, 4))]
        with pytest.raises(ValueError):
            cj.circuit_jaccard_rows({}, exs)

    def test_one_model_raises(self, tiny_llama):
        model = tiny_llama(
            seed=203, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        exs = [SftExample(input_ids=[4, 5, 6, 7], label_span=(2, 4))]
        with pytest.raises(ValueError):
            cj.circuit_jaccard_rows({"a": model}, exs)

    def test_empty_examples_raises(self, tiny_llama):
        model_a = tiny_llama(
            seed=204, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with pytest.raises(ValueError):
            cj.circuit_jaccard_rows({"a": model_a, "b": model_b}, [])

    def test_same_model_twice_gives_jaccard_and_spearman_of_one(self, tiny_llama):
        # Default ablation="mean": both "models" are the SAME weights fed
        # the SAME single batch (n=4 < the default batch_size=32), so the
        # mean-ablation reference is identical for both sides and the
        # default dispatch path gets a real numeric assertion here, not
        # just the ablation="zero" path other tests exercise.
        model_a = tiny_llama(
            seed=205, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        rng = random.Random(61)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=7, span=(4, 6))
        rows = cj.circuit_jaccard_rows(
            {"a": model_a, "b": model_b},
            exs,
            node_budgets=(2, 5),
            edge_budgets=(5, 10),
        )
        df = pd.DataFrame(rows)
        pair = df[df["level"] == "pair"]
        assert (pair["jaccard"] == 1.0).all()
        assert (pair["mass_overlap_ab"] == 1.0).all()
        assert (pair["mass_overlap_ba"] == 1.0).all()
        assert pair["spearman"].apply(lambda v: math.isclose(v, 1.0, abs_tol=1e-9)).all()
        assert (pair["n"] == len(exs)).all()

    def test_planted_causal_head_enters_topk_drops_jaccard_raises_concentration(self, tiny_llama):
        model_a = tiny_llama(
            seed=206, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        target_token = 10
        rng = random.Random(62)
        exs = _fixed_target_examples(
            rng, n=5, vocab_size=64, seq_len=8, span=(5, 7), target_token=target_token
        )
        plant_layer, plant_head = 1, 2
        _install_causal_head_plant(model_b, plant_layer, plant_head, target_token, huge=3.0)
        plant_node_str = str(mn.NodeId(plant_layer, "head", plant_head))

        rows = cj.circuit_jaccard_rows(
            {"a": model_a, "b": model_b},
            exs,
            node_budgets=(1, 3),
            edge_budgets=(5,),
            ablation="zero",
        )
        df = pd.DataFrame(rows)
        model_rows = df[df["level"] == "model"]

        def model_row(name, kind, k):
            sub = model_rows[
                (model_rows["model"] == name)
                & (model_rows["kind"] == kind)
                & (model_rows["k"] == k)
            ]
            return sub.iloc[0]

        b_k1 = model_row("b", "node", 1)
        assert b_k1["circuit"] == plant_node_str

        a_k1 = model_row("a", "node", 1)
        assert a_k1["circuit"] != plant_node_str  # a was never planted

        pair_rows = df[(df["level"] == "pair") & (df["kind"] == "node")]
        jac_k1 = pair_rows[pair_rows["k"] == 1].iloc[0]["jaccard"]
        assert jac_k1 == 0.0  # b's sole top-1 (the plant) differs from a's

        conc_a = model_row("a", "node", 3)["concentration"]
        conc_b = model_row("b", "node", 3)["concentration"]
        assert conc_b > conc_a
        assert conc_b > 0.5

    def test_node_budget_larger_than_node_count_clips_without_crash(self, tiny_llama):
        model_a = tiny_llama(
            seed=207, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        rng = random.Random(63)
        exs = _examples(rng, n=3, vocab_size=32, seq_len=6, span=(3, 5))
        rows = cj.circuit_jaccard_rows(
            {"a": model_a, "b": model_b},
            exs,
            node_budgets=(1000,),
            edge_budgets=(1000,),
            ablation="zero",
        )
        df = pd.DataFrame(rows)
        n_heads, _d_head = mn._num_heads_and_dhead(model_a)
        expected_node_count = 1 * (n_heads + 1)  # n_layers=1
        model_node_rows = df[(df["level"] == "model") & (df["kind"] == "node")]
        assert (model_node_rows["k_effective"] == expected_node_count).all()
        pair_node_rows = df[(df["level"] == "pair") & (df["kind"] == "node")]
        assert (
            pair_node_rows["jaccard"] == 1.0
        ).all()  # identical weights, full circuit both sides
        pair_edge_rows = df[(df["level"] == "pair") & (df["kind"] == "edge")]
        assert (pair_edge_rows["jaccard"] == 1.0).all()


# --------------------------------------------------------------------------
# print_summary
# --------------------------------------------------------------------------


class TestPrintSummary:
    def test_runs_without_crash_and_reports_both_kinds(self, capsys):
        rows = [
            {
                "level": "pair",
                "model_a": "a",
                "model_b": "b",
                "kind": "node",
                "k": 10,
                "jaccard": 0.5,
                "mass_overlap_ab": 0.6,
                "mass_overlap_ba": 0.4,
                "spearman": 0.7,
                "n": 3,
            },
            {
                "level": "pair",
                "model_a": "a",
                "model_b": "b",
                "kind": "edge",
                "k": 25,
                "jaccard": 0.2,
                "mass_overlap_ab": 0.3,
                "mass_overlap_ba": 0.1,
                "spearman": 0.4,
                "n": 3,
            },
            {
                "level": "model",
                "model": "a",
                "kind": "node",
                "k": 10,
                "k_effective": 10,
                "concentration": 0.9,
                "circuit": "a0.h1,m0",
                "n": 3,
            },
        ]
        cj.print_summary(rows)  # must not raise
        out = capsys.readouterr().out
        assert "kind=node" in out
        assert "kind=edge" in out

    def test_falls_back_when_default_k_not_present(self, capsys):
        rows = [
            {
                "level": "pair",
                "model_a": "a",
                "model_b": "b",
                "kind": "node",
                "k": 7,
                "jaccard": 0.5,
                "mass_overlap_ab": 0.6,
                "mass_overlap_ba": 0.4,
                "spearman": 0.7,
                "n": 3,
            }
        ]
        cj.print_summary(rows)  # must not raise despite k=10 (the usual default) missing
        out = capsys.readouterr().out
        assert "k=7" in out


# --------------------------------------------------------------------------
# main() smoke test
# --------------------------------------------------------------------------


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


def _task_parquet(tmp_path):
    texts = [f"t1 t2 t{i}" for i in range(3, 9)]
    starts = [len(t) - len(t.split()[-1]) for t in texts]
    ends = [len(t) for t in texts]
    df = pd.DataFrame({"full_text": texts, "answer_char_start": starts, "answer_char_end": ends})
    p = tmp_path / "task.parquet"
    df.to_parquet(p)
    return p


class TestMainSmoke:
    def test_main_smoke(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        model_a_dir = _save_tiny_model(tiny_llama, tmp_path, 220, "model_a")
        model_b_dir = _save_tiny_model(tiny_llama, tmp_path, 221, "model_b")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "circuit_jaccard.py",
            "--model",
            f"a=dir:{model_a_dir}",
            "--model",
            f"b=dir:{model_b_dir}",
            "--prompt-parquet",
            str(parquet),
            "--tokenizer",
            str(tok_dir),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--node-budgets",
            "2,4",
            "--edge-budgets",
            "3,6",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        cj.main()
        assert out.is_file()
        df = pd.read_csv(out)
        for col in ("level", "kind", "k", "n"):
            assert col in df.columns

        model_rows = df[df["level"] == "model"]
        for col in ("model", "k_effective", "concentration", "circuit"):
            assert col in model_rows.columns
            assert model_rows[col].notna().all()

        pair_rows = df[df["level"] == "pair"]
        for col in (
            "model_a",
            "model_b",
            "jaccard",
            "mass_overlap_ab",
            "mass_overlap_ba",
            "spearman",
        ):
            assert col in pair_rows.columns
            assert pair_rows[col].notna().all()

    def test_main_requires_at_least_two_models(
        self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch
    ):
        model_dir = _save_tiny_model(tiny_llama, tmp_path, 222, "model")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        argv = [
            "circuit_jaccard.py",
            "--model",
            f"a=dir:{model_dir}",
            "--prompt-parquet",
            str(parquet),
            "--tokenizer",
            str(tok_dir),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(ValueError):
            cj.main()
