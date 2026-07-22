"""Property tests for the gradient-alignment metric (specs/02 §7, V5.13).

V5.13: planted parallel gradients ⇒ ≈1; random gradients ⇒ ≈0 (with the
n ≪ d caveat pinned numerically). All inputs are seeded torch tensors; no
model is involved — the metric is a pure function of the per-example
gradient matrix.
"""

from __future__ import annotations

import pytest
import torch

from geode.probe import gradient_alignment


def test_v5_13_planted_parallel_gradients_align_to_one():
    """Rows that are positive multiples of one direction: both summaries = 1
    (up to fp64 rounding). The [n, seq, d] form must flatten to the identical
    result, and bf16 inputs (the dump storage dtype) must stay ≈1."""
    gen = torch.Generator().manual_seed(7)
    direction = torch.randn(512, generator=gen)
    scales = torch.rand(6, generator=gen) + 0.5  # positive: same direction, not just axis
    planted = scales[:, None] * direction[None, :]

    a = gradient_alignment(planted)
    assert a.n_examples == 6
    assert a.pairwise_cosine_mean == pytest.approx(1.0, abs=1e-6)
    assert a.top_pc_explained_variance == pytest.approx(1.0, abs=1e-6)

    # [n, seq, d] input flattens to the same vectors ⇒ identical summaries.
    a3d = gradient_alignment(planted.reshape(6, 8, 64))
    assert a3d == a

    # bf16 (storage dtype of the dumps): rounding noise, still ≈1.
    a16 = gradient_alignment(planted.to(torch.bfloat16))
    assert a16.pairwise_cosine_mean == pytest.approx(1.0, abs=1e-2)
    assert a16.top_pc_explained_variance == pytest.approx(1.0, abs=1e-2)


def test_v5_13_random_gradients_align_to_zero_with_n_ll_d_caveat():
    """Isotropic random rows, n=8 ≪ d=4096: pairwise cosine ≈ 0, and the
    top-PC explained-variance fraction ≈ 1/n — NOT 0 — which is the n ≪ d
    caveat pinned numerically.

    Calibration (scratchpad, 2026-07-22, seeds 0-4 of this exact generator):
    pairwise ∈ [-0.005, +0.003]; EVR ∈ [0.1330, 0.1352] against 1/n = 0.125.
    Bounds below are ≥4x the observed spread; EVR ≥ 1/n is an algebraic
    identity (s₀² ≥ mean of squared singular values), so the lower bound is
    exact.
    """
    gen = torch.Generator().manual_seed(0)
    g = torch.randn(8, 4096, generator=gen)
    a = gradient_alignment(g)
    assert a.n_examples == 8
    assert abs(a.pairwise_cosine_mean) < 0.02
    assert 1.0 / 8 <= a.top_pc_explained_variance < 0.17  # ≈ 1/n, far from 1


def test_v5_13_degenerate_inputs_raise():
    """Zero-norm rows (the late-training degenerate case spec §7 warns about),
    fewer than 2 examples, and non-matrix inputs all refuse loudly."""
    good = torch.randn(4, 32, generator=torch.Generator().manual_seed(1))
    bad = good.clone()
    bad[2] = 0.0
    with pytest.raises(ValueError, match="zero-norm.*2"):
        gradient_alignment(bad)
    with pytest.raises(ValueError, match=">= 2"):
        gradient_alignment(good[:1])
    with pytest.raises(ValueError, match="n_examples"):
        gradient_alignment(good[0])
