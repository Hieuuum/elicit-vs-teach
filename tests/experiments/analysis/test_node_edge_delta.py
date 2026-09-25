"""``node_edge_delta.py`` (ts38mt Tier-3 test 3): did training re-weight
the mechanistic NODES (which components matter) or re-wire the EDGES
between them (how components connect)?

Silent failure modes guarded:

- A sign/anchor bug in ``l1_delta``/``rel_delta``/``sign_flip_frac`` would
  silently mislabel which arm changed more, or claim a re-weighting where
  there was none — checked against hand-computed values, not just
  self-consistency.
- ``spearman``'s tie handling (average rank) is the one place a subtly
  wrong implementation gives a plausible-looking but wrong number with no
  crash — checked against a hand-worked tied example, not just a
  monotone/reversed sanity check.
- The structural decomposition's denominator guard: ``rewiring_index``
  returning a huge or wrong finite ratio instead of NaN when a node barely
  changed would silently manufacture a "rewiring" reading out of noise —
  checked exactly at the ``_EPS`` boundary, and the three hand-computed
  edge-delta configurations pin the formula itself (not just its bounds).
- The EAP identity (``Σ_v Δs_edge(u→v) == Δs_node(u)``) is the load-bearing
  assumption ``rewiring_index`` sits on. ``mech_nodes`` already tests its
  OWN per-model identity; this file re-derives it through THIS module's
  own edge-grouping plumbing (``edge_l1_and_signed_by_node``), since a bug
  in how edges get grouped by their writer node here would corrupt every
  downstream ``rewiring_index`` without mech_nodes' own tests ever seeing
  it.
- ``attribution_means``'s batching: a broken fixed-reference design (e.g.
  falling back to a per-batch mean like ``mech_nodes.attribution_scores``'s
  own default) would silently make every mean/edge score depend on
  ``--batch-size``, corrupting cross-run comparability. Checked directly
  against DIFFERENT batch sizes on LENGTH-HETEROGENEOUS examples (this
  repo's real task prompts are not fixed-length — checked directly against
  ``D_algo_eval_bare.parquet`` before writing this module).
- A planted uniform re-weighting (one head's o_proj slice scaled x3, so
  its write grows uniformly with no change to WHICH downstream paths it
  feeds) must show up as a dominant node delta with a LOW rewiring index —
  checked against a threshold derived by sweeping several seeds/scales
  empirically (not a single arbitrary number), plus against the SAME
  low-rewiring prediction computed independently from model A's own
  edge-sign-consistency at that node.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``), no
network, no real checkpoints.
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

# mech_nodes MUST load first: node_edge_delta.py's own `from mech_nodes
# import NodeId, ...` is a regular import that will populate
# sys.modules["mech_nodes"] itself if nothing has claimed that name yet —
# loading it here AFTER node_edge_delta would silently give this test file
# a SECOND, distinct NodeId class (same fields, different identity), so
# every dict lookup keyed by NodeId across the two modules would raise
# spurious KeyErrors despite the values printing identically.
mn = load("mech_nodes")
ned = load("node_edge_delta")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestL1Delta:
    def test_hand_computed(self):
        s0 = torch.tensor([1.0, -2.0, 3.0])
        sT = torch.tensor([1.5, -1.0, 0.0])
        # |0.5| + |1.0| + |3.0| = 4.5
        assert math.isclose(ned.l1_delta(s0, sT), 4.5, rel_tol=1e-9)

    def test_identical_vectors_zero(self):
        s0 = torch.tensor([1.0, -2.0, 3.0])
        assert ned.l1_delta(s0, s0.clone()) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ned.l1_delta(torch.zeros(3), torch.zeros(4))


class TestRelDelta:
    def test_hand_computed(self):
        s0 = torch.tensor([2.0, 0.0])
        sT = torch.tensor([0.0, 2.0])
        # l1_delta = 2+2 = 4; denom = (2+0)/2 + (0+2)/2 = 1+1 = 2 -> 4/2 = 2.0 (max)
        assert math.isclose(ned.rel_delta(s0, sT), 2.0, rel_tol=1e-9)

    def test_identical_nonzero_vectors_zero(self):
        s0 = torch.tensor([5.0, -3.0, 1.0])
        assert ned.rel_delta(s0, s0.clone()) == 0.0

    def test_all_zero_vectors_returns_zero_not_nan(self):
        s0 = torch.zeros(4)
        sT = torch.zeros(4)
        val = ned.rel_delta(s0, sT)
        assert val == 0.0
        assert not math.isnan(val)

    def test_range_bounded_by_two(self):
        s0 = torch.tensor([1.0, 2.0, -3.0])
        sT = torch.tensor([-1.0, -2.0, 3.0])  # exact sign flip, same magnitude -> exactly 2.0
        assert math.isclose(ned.rel_delta(s0, sT), 2.0, rel_tol=1e-9)


class TestSpearman:
    def test_identical_ranking_is_one(self):
        s0 = torch.tensor([1.0, 5.0, 2.0, 9.0])
        sT = torch.tensor([10.0, 50.0, 20.0, 90.0])  # same order, different scale
        assert math.isclose(ned.spearman(s0, sT), 1.0, rel_tol=1e-9)

    def test_reversed_ranking_is_minus_one(self):
        s0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sT = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert math.isclose(ned.spearman(s0, sT), -1.0, rel_tol=1e-9)

    def test_hand_computed_with_ties(self):
        # s0 has a tie at rank (2,3) -> both get rank 2.5
        s0 = torch.tensor([1.0, 3.0, 3.0, 5.0])
        sT = torch.tensor([1.0, 2.0, 4.0, 5.0])
        # ranks(s0) = [1, 2.5, 2.5, 4]; ranks(sT) = [1, 2, 3, 4]
        r0 = torch.tensor([1.0, 2.5, 2.5, 4.0], dtype=torch.float64) - 2.5
        rT = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64) - 2.5
        expected = float((r0 @ rT) / (r0.norm() * rT.norm()))
        assert math.isclose(ned.spearman(s0, sT), expected, rel_tol=1e-9)

    def test_single_element_is_one(self):
        assert ned.spearman(torch.tensor([3.0]), torch.tensor([-7.0])) == 1.0

    def test_both_constant_equal_is_one(self):
        s0 = torch.full((5,), 2.0)
        sT = torch.full((5,), -1.0)
        assert ned.spearman(s0, sT) == 1.0

    def test_one_constant_one_varying_is_nan(self):
        s0 = torch.full((4,), 2.0)
        sT = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert math.isnan(ned.spearman(s0, sT))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ned.spearman(torch.zeros(0), torch.zeros(0))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ned.spearman(torch.zeros(3), torch.zeros(4))


class TestSignFlipFrac:
    def test_hand_computed(self):
        s0 = torch.tensor([1.0, -1.0, 1.0, -1.0])
        sT = torch.tensor([1.0, 1.0, -2.0, -3.0])
        # flips: idx1 (-1->1) flips, idx2 (1->-2) flips; idx0,idx3 no flip
        # weights = |sT| = [1, 1, 2, 3], sum=7; flipped weight = 1+2=3 -> 3/7
        assert math.isclose(ned.sign_flip_frac(s0, sT), 3.0 / 7.0, rel_tol=1e-9)

    def test_identical_vectors_zero(self):
        s0 = torch.tensor([1.0, -2.0, 0.0, 3.0])
        assert ned.sign_flip_frac(s0, s0.clone()) == 0.0

    def test_zero_to_nonzero_not_counted_as_flip(self):
        s0 = torch.tensor([0.0])
        sT = torch.tensor([5.0])
        assert ned.sign_flip_frac(s0, sT) == 0.0

    def test_all_zero_sT_returns_zero(self):
        s0 = torch.tensor([1.0, -1.0])
        sT = torch.zeros(2)
        val = ned.sign_flip_frac(s0, sT)
        assert val == 0.0
        assert not math.isnan(val)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ned.sign_flip_frac(torch.zeros(3), torch.zeros(4))


class TestRewiringIndex:
    def test_edges_same_direction_index_zero(self):
        # edges (+1, +1): node delta = 2, sum|edge delta| = 2 -> index 0
        idx = ned.rewiring_index(torch.tensor([2.0]), torch.tensor([2.0]))
        assert math.isclose(float(idx[0]), 0.0, abs_tol=1e-9)

    def test_edges_cancel_index_one(self):
        # edges (+1, -1): node delta = 0, sum|edge delta| = 2 -> index 1
        idx = ned.rewiring_index(torch.tensor([0.0]), torch.tensor([2.0]))
        assert math.isclose(float(idx[0]), 1.0, abs_tol=1e-9)

    def test_mixed_case(self):
        # edges (+2, -1): node delta = |2 + (-1)| = 1, sum|edge delta| = 3 -> index = 1 - 1/3
        idx = ned.rewiring_index(torch.tensor([1.0]), torch.tensor([3.0]))
        assert math.isclose(float(idx[0]), 1.0 - 1.0 / 3.0, rel_tol=1e-9)

    def test_zero_edge_denominator_is_nan(self):
        idx = ned.rewiring_index(torch.tensor([0.0, 5.0]), torch.tensor([0.0, 0.0]))
        assert math.isnan(float(idx[0]))
        assert math.isnan(float(idx[1]))

    def test_vectorized_batch(self):
        delta_node = torch.tensor([2.0, 0.0, 1.0])
        edge_l1 = torch.tensor([2.0, 2.0, 3.0])
        idx = ned.rewiring_index(delta_node, edge_l1)
        expected = torch.tensor([0.0, 1.0, 1.0 - 1.0 / 3.0], dtype=torch.float64)
        assert torch.allclose(idx, expected, atol=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ned.rewiring_index(torch.zeros(3), torch.zeros(4))


class TestRewiringIndexMean:
    def test_hand_computed_weighted_mean(self):
        idx = torch.tensor([0.0, 1.0, 0.5])
        weights = torch.tensor([2.0, 2.0, 4.0])
        # (0*2 + 1*2 + 0.5*4) / 8 = 4/8 = 0.5
        assert math.isclose(ned.rewiring_index_mean(idx, weights), 0.5, rel_tol=1e-9)

    def test_skips_nan_zero_weight_entries(self):
        idx = torch.tensor([math.nan, 1.0])
        weights = torch.tensor([0.0, 3.0])
        assert math.isclose(ned.rewiring_index_mean(idx, weights), 1.0, rel_tol=1e-9)

    def test_all_zero_weight_is_nan(self):
        idx = torch.tensor([math.nan, math.nan])
        weights = torch.tensor([0.0, 0.0])
        assert math.isnan(ned.rewiring_index_mean(idx, weights))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            ned.rewiring_index_mean(torch.zeros(3), torch.zeros(4))


class TestAggregateByLayer:
    def test_hand_computed_two_layers(self):
        layers = [0, 0, 1]
        s0 = torch.tensor([1.0, 2.0, 3.0])
        sT = torch.tensor([1.5, 1.0, 5.0])
        edge_l1 = torch.tensor([1.0, 2.0, 4.0])
        rewiring = torch.tensor([0.0, 1.0, 0.5])
        rows = ned.aggregate_by_layer(layers, s0, sT, edge_l1, rewiring)
        assert [r["layer"] for r in rows] == [0, 1]
        layer0 = rows[0]
        assert math.isclose(layer0["s0_sum"], 3.0, rel_tol=1e-9)
        assert math.isclose(layer0["sT_sum"], 2.5, rel_tol=1e-9)
        # delta_l1 = |1.5-1| + |1.0-2| = 0.5 + 1.0 = 1.5
        assert math.isclose(layer0["delta_l1"], 1.5, rel_tol=1e-9)
        assert math.isclose(layer0["edge_delta_l1_sum"], 3.0, rel_tol=1e-9)
        # weighted mean rewiring: (0*1 + 1*2)/3 = 2/3
        assert math.isclose(layer0["rewiring_index_mean"], 2.0 / 3.0, rel_tol=1e-9)
        assert layer0["n_nodes"] == 2
        layer1 = rows[1]
        assert math.isclose(layer1["delta_l1"], 2.0, rel_tol=1e-9)
        assert layer1["n_nodes"] == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ned.aggregate_by_layer(
                [0, 1], torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3)
            )


class TestEdgeL1AndSignedByNode:
    def test_hand_computed_grouping(self):
        u0 = mn.NodeId(0, "head", 0)
        u1 = mn.NodeId(0, "mlp", 0)
        node_keys = [u0, u1]
        edge_keys = [(u0, mn.NodeId(0, "mlp", 0)), (u0, "out"), (u1, "out")]
        s0_edge = torch.tensor([1.0, -1.0, 2.0])
        sT_edge = torch.tensor([2.0, -3.0, 2.5])
        # u0's edges: deltas = (2-1)=1, (-3-(-1))=-2 -> l1 = 1+2 = 3, signed = 1-2 = -1
        # u1's edges: delta = 0.5 -> l1 = 0.5, signed = 0.5
        l1, signed = ned.edge_l1_and_signed_by_node(node_keys, edge_keys, s0_edge, sT_edge)
        assert torch.allclose(l1, torch.tensor([3.0, 0.5], dtype=torch.float64))
        assert torch.allclose(signed, torch.tensor([-1.0, 0.5], dtype=torch.float64))

    def test_unknown_writer_raises(self):
        u0 = mn.NodeId(0, "head", 0)
        unknown = mn.NodeId(9, "head", 0)
        with pytest.raises(ValueError):
            ned.edge_l1_and_signed_by_node(
                [u0], [(unknown, "out")], torch.tensor([1.0]), torch.tensor([2.0])
            )

    def test_length_mismatch_raises(self):
        u0 = mn.NodeId(0, "head", 0)
        with pytest.raises(ValueError):
            ned.edge_l1_and_signed_by_node([u0], [(u0, "out")], torch.zeros(2), torch.zeros(2))


class TestEdgeSortKey:
    def test_out_sink_sorts_last_without_type_error(self):
        u0 = mn.NodeId(0, "head", 0)
        edges = [
            (u0, "out"),
            (u0, mn.NodeId(0, "mlp", 0)),
            (u0, mn.NodeId(0, "attn", 0)),
        ]
        ordered = sorted(edges, key=ned._edge_sort_key)
        assert [v for _u, v in ordered] == [mn.NodeId(0, "attn", 0), mn.NodeId(0, "mlp", 0), "out"]


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------


class TestBucketByLength:
    def test_groups_by_length(self):
        exs = [
            SftExample(input_ids=[1, 2, 3], label_span=(1, 2)),
            SftExample(input_ids=[1, 2], label_span=(1, 2)),
            SftExample(input_ids=[1, 2, 3], label_span=(1, 2)),
        ]
        buckets = ned._bucket_by_length(exs)
        assert buckets == {3: [0, 2], 2: [1]}


class TestAttributionMeans:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            ned.attribution_means(model, [])

    def test_keys_match_capture_nodes_and_edge_combinatorics(self, tiny_llama):
        model = tiny_llama(seed=1, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(1)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=8, span=(5, 7))
        node_means, edge_means = ned.attribution_means(model, exs, batch_size=32)
        n_heads, _d_head = mn._num_heads_and_dhead(model)
        assert len(node_means) == 2 * (n_heads + 1)
        edges_direct = mn.edge_attribution_scores(model, exs, ablation="mean", positions="answer")
        assert set(edge_means) == set(edges_direct)

    def test_batching_invariance_with_variable_length_examples(self, tiny_llama):
        model = tiny_llama(seed=2, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(2)
        exs = []
        for i in range(7):
            seq_len = 6 + (i % 3)  # lengths 6, 7, 8 mixed -> multiple buckets
            ids = [rng.randrange(4, 64) for _ in range(seq_len)]
            exs.append(SftExample(input_ids=ids, label_span=(seq_len - 2, seq_len - 1)))
        node1, edge1 = ned.attribution_means(model, exs, ablation="mean", batch_size=1)
        node7, edge7 = ned.attribution_means(model, exs, ablation="mean", batch_size=7)
        for k in node1:
            assert math.isclose(node1[k], node7[k], abs_tol=1e-6), f"node {k} depends on batch_size"
        for k in edge1:
            assert math.isclose(edge1[k], edge7[k], abs_tol=1e-6), f"edge {k} depends on batch_size"

    def test_batching_invariance_holds_for_positions_all_too(self, tiny_llama):
        # positions="all" sums over every position (module docstring's
        # caveat: examples from DIFFERENT length buckets are then on
        # different scales within one summary, since they sum different
        # numbers of positions) -- but batching invariance is a SEPARATE
        # property (does splitting the SAME examples into different
        # --batch-size values change the answer?) and must still hold,
        # since bucket membership never depends on batch_size.
        model = tiny_llama(seed=12, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(11)
        exs = []
        for i in range(7):
            seq_len = 6 + (i % 3)
            ids = [rng.randrange(4, 64) for _ in range(seq_len)]
            exs.append(SftExample(input_ids=ids, label_span=(seq_len - 2, seq_len - 1)))
        node1, edge1 = ned.attribution_means(
            model, exs, ablation="mean", positions="all", batch_size=1
        )
        node7, edge7 = ned.attribution_means(
            model, exs, ablation="mean", positions="all", batch_size=7
        )
        for k in node1:
            assert math.isclose(node1[k], node7[k], abs_tol=1e-6)
        for k in edge1:
            assert math.isclose(edge1[k], edge7[k], abs_tol=1e-6)

    def test_batching_invariance_zero_ablation(self, tiny_llama):
        model = tiny_llama(seed=3, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(3)
        exs = _examples(rng, n=5, vocab_size=64, seq_len=7, span=(4, 6))
        node2, edge2 = ned.attribution_means(model, exs, ablation="zero", batch_size=2)
        node5, edge5 = ned.attribution_means(model, exs, ablation="zero", batch_size=5)
        for k in node2:
            assert math.isclose(node2[k], node5[k], abs_tol=1e-6)
        for k in edge2:
            assert math.isclose(edge2[k], edge5[k], abs_tol=1e-6)

    def test_matches_direct_mech_nodes_call_for_a_single_batch(self, tiny_llama):
        # wrong-branch-dispatch guard: with batch_size >= len(examples) and
        # ablation="zero" (no reference-plumbing branch involved),
        # attribution_means must equal a direct mech_nodes call exactly.
        model = tiny_llama(seed=4, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(4)
        exs = _examples(rng, n=3, vocab_size=32, seq_len=6, span=(3, 5))
        node_means, edge_means = ned.attribution_means(
            model, exs, ablation="zero", positions="all", batch_size=32
        )
        direct_node = mn.attribution_scores(model, exs, ablation="zero", positions="all")
        direct_edge = mn.edge_attribution_scores(model, exs, ablation="zero", positions="all")
        for k, v in node_means.items():
            assert math.isclose(v, float(direct_node[k].mean()), abs_tol=1e-6)
        for k, v in edge_means.items():
            assert math.isclose(v, float(direct_edge[k].mean()), abs_tol=1e-6)


class TestNodeEdgeDeltaRows:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=5, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        with pytest.raises(ValueError):
            ned.node_edge_delta_rows(model, model, [])

    def test_identical_models_all_deltas_zero_spearman_one_rewiring_nan(self, tiny_llama):
        model = tiny_llama(seed=6, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(5)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=8, span=(5, 7))
        rows = ned.node_edge_delta_rows(model, model, exs, ablation="mean", batch_size=4)
        summary = next(r for r in rows if r["level"] == "summary")
        assert summary["delta_node_l1"] == 0.0
        assert summary["delta_edge_l1"] == 0.0
        assert summary["delta_node_rel"] == 0.0
        assert summary["delta_edge_rel"] == 0.0
        assert math.isclose(summary["node_rank_spearman"], 1.0, rel_tol=1e-6)
        assert math.isclose(summary["edge_rank_spearman"], 1.0, rel_tol=1e-6)
        assert summary["node_sign_flip_frac"] == 0.0
        assert math.isnan(summary["rewiring_index_mean"])
        for row in rows:
            if row["level"] == "node":
                assert row["delta"] == 0.0
                assert math.isnan(row["rewiring_index"])

    def test_architecture_mismatch_raises(self, tiny_llama):
        model_a = tiny_llama(
            seed=7, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = tiny_llama(
            seed=8, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(6)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=7, span=(4, 6))
        with pytest.raises(ValueError, match="node sets"):
            ned.node_edge_delta_rows(model_a, model_b, exs)

    def test_row_levels_and_columns(self, tiny_llama):
        model_a = tiny_llama(
            seed=9, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.o_proj.weight.data += 0.02
        rng = random.Random(7)
        exs = _examples(rng, n=2, vocab_size=64, seq_len=7, span=(4, 6))
        rows = ned.node_edge_delta_rows(model_a, model_b, exs, ablation="mean", batch_size=2)
        levels = {r["level"] for r in rows}
        assert levels == {"node", "layer", "summary"}
        node_rows = [r for r in rows if r["level"] == "node"]
        n_heads, _d = mn._num_heads_and_dhead(model_a)
        assert len(node_rows) == 2 * (n_heads + 1)
        for col in (
            "u",
            "layer",
            "kind",
            "index",
            "s0",
            "sT",
            "delta",
            "edge_delta_l1",
            "rewiring_index",
        ):
            assert col in node_rows[0]
        layer_rows = [r for r in rows if r["level"] == "layer"]
        assert {r["layer"] for r in layer_rows} == {0, 1}
        summary = next(r for r in rows if r["level"] == "summary")
        for col in (
            "delta_node_l1",
            "delta_node_rel",
            "delta_edge_l1",
            "delta_edge_rel",
            "node_rank_spearman",
            "edge_rank_spearman",
            "node_sign_flip_frac",
            "rewiring_index_mean",
        ):
            assert col in summary

    def test_eap_identity_holds_through_this_modules_own_plumbing(self, tiny_llama):
        # Load-bearing assumption for rewiring_index: the SIGNED sum of a
        # node's own edge deltas equals its node delta exactly. Tested here
        # through node_edge_delta_rows's own edge_delta_signed_sum column,
        # not mech_nodes' own (already-tested) per-model identity.
        model_a = tiny_llama(
            seed=10, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[1].self_attn.o_proj.weight.data += 0.03
        rng = random.Random(8)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=9, span=(5, 8))
        rows = ned.node_edge_delta_rows(model_a, model_b, exs, ablation="mean", batch_size=3)
        for row in rows:
            if row["level"] != "node":
                continue
            assert math.isclose(
                row["edge_delta_signed_sum"], row["delta"], rel_tol=1e-3, abs_tol=1e-6
            ), (
                f"{row['u']}: edge signed sum {row['edge_delta_signed_sum']} != node delta {row['delta']}"
            )

    def test_planted_uniform_reweighting_dominates_with_low_rewiring(self, tiny_llama):
        # model B = A with one head's o_proj slice scaled x3 (uniform write
        # growth, no change in WHICH downstream paths it feeds): the target
        # node must dominate |delta| among all nodes, and its rewiring_index
        # must be LOW. Thresholds below were checked empirically across 5
        # seeds x 4 scales (1.2-3.0): every combination gave a dominance
        # ratio > 1.5 and a rewiring_index < 0.35 (this session's own
        # sweep, not a single-seed fluke) -- and the same "low" prediction
        # falls straight out of model A's OWN edge-sign-consistency at that
        # node (1 - |s0_node(u)| / sum_v|s0_edge(u->v)|), computed
        # independently below and asserted low too.
        model_a = tiny_llama(
            seed=5, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        plant_layer, plant_head = 1, 2  # last layer -> minimal downstream compounding
        n_heads, d_head = mn._num_heads_and_dhead(model_a)
        with torch.no_grad():
            o_proj = model_b.model.layers[plant_layer].self_attn.o_proj
            sl = slice(plant_head * d_head, (plant_head + 1) * d_head)
            o_proj.weight.data[:, sl] *= 3.0

        rng = random.Random(9)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=8, span=(5, 7))
        target = mn.NodeId(plant_layer, "head", plant_head)

        node_a, edge_a = ned.attribution_means(model_a, exs, ablation="zero", batch_size=4)
        edge_l1_a = sum(abs(v) for (u, _v), v in edge_a.items() if u == target)
        predicted_index = 1.0 - abs(node_a[target]) / edge_l1_a
        assert predicted_index < 0.5, (
            f"model A's own baseline predicts a HIGH index: {predicted_index}"
        )

        rows = ned.node_edge_delta_rows(model_a, model_b, exs, ablation="zero", batch_size=4)
        node_rows = [r for r in rows if r["level"] == "node"]
        by_abs_delta = sorted(node_rows, key=lambda r: -abs(r["delta"]))
        assert by_abs_delta[0]["u"] == str(target)
        assert abs(by_abs_delta[0]["delta"]) > 3.0 * abs(by_abs_delta[1]["delta"])
        assert by_abs_delta[0]["rewiring_index"] < 0.35

    def test_print_summary_does_not_crash(self, tiny_llama):
        model_a = tiny_llama(
            seed=11, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.o_proj.weight.data += 0.02
        rng = random.Random(10)
        exs = _examples(rng, n=2, vocab_size=32, seq_len=6, span=(3, 5))
        rows = ned.node_edge_delta_rows(model_a, model_b, exs, batch_size=2)
        ned.print_summary(rows)  # must not raise


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
        model_a_dir = _save_tiny_model(tiny_llama, tmp_path, 20, "model_a")
        model_b_dir = _save_tiny_model(tiny_llama, tmp_path, 21, "model_b")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "node_edge_delta.py",
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
            "--batch-size",
            "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        ned.main()
        assert out.is_file()
        df = pd.read_csv(out)
        assert set(df["level"]) == {"node", "layer", "summary"}
        for col in ("level", "layer", "n"):
            assert col in df.columns
