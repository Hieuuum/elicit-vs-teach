"""Property tests for the specs/02 §7 analysis metrics (V5.13–V5.16).

All inputs are seeded torch tensors / plain sequences; no model is involved —
each metric is a pure function of its inputs.

V5.13: planted parallel gradients ⇒ ≈1; random gradients ⇒ ≈0 (with the
n ≪ d caveat pinned numerically).
V5.14: drift zero at init; planted per-class shift recovered per class.
V5.15: planted rank-r adapter delta ⇒ effective rank r.
V5.16: planted curves ⇒ known pairing; ties broken toward earliest step_b.
"""

from __future__ import annotations

import pytest
import torch

from geode.probe import (
    effective_rank,
    gradient_alignment,
    performance_aligned_matching,
    representation_drift,
)


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


# --- V5.14 representation drift ---------------------------------------------


def test_v5_14_zero_drift_at_init_snapshot():
    """Same activations at ref and now ⇒ every per-class drift is exactly 0."""
    gen = torch.Generator().manual_seed(11)
    n, seq, d = 6, 5, 16
    acts = torch.randn(n, seq, d, generator=gen, dtype=torch.float64)
    mask = torch.zeros(n, seq, dtype=torch.bool)
    mask[:, 2:] = True  # 3 pooled positions per example, never all-False
    classes = ["2x3", "2x3", "4x4", "4x4", "5x1", "5x1"]

    r = representation_drift(acts, acts, mask, classes)
    assert r.n_examples == n
    assert r.classes == ("2x3", "4x4", "5x1")
    for x in r.drift_l2:
        assert x == pytest.approx(0.0, abs=1e-12)


def test_v5_14_planted_per_class_shift_recovered():
    """A distinct known shift added at mask-True positions of each class ⇒ that
    class's drift equals the shift's L2 norm, per class, in fp64."""
    gen = torch.Generator().manual_seed(12)
    n, seq, d = 6, 5, 16
    acts_ref = torch.randn(n, seq, d, generator=gen, dtype=torch.float64)
    mask = torch.zeros(n, seq, dtype=torch.bool)
    mask[:, 1:4] = True
    classes = ["a", "a", "b", "b", "c", "c"]

    shifts = {c: torch.zeros(d, dtype=torch.float64) for c in ("a", "b", "c")}
    shifts["a"][0] = 1.0  # norm 1
    shifts["b"][1] = 3.0  # norm 3
    shifts["c"][:2] = torch.tensor([3.0, 4.0], dtype=torch.float64)  # norm 5

    acts_now = acts_ref.clone()
    for i, c in enumerate(classes):
        acts_now[i, mask[i]] += shifts[c]

    r = representation_drift(acts_ref, acts_now, mask, classes)
    idx = {c: r.classes.index(c) for c in ("a", "b", "c")}
    assert r.drift_l2[idx["a"]] == pytest.approx(1.0, abs=1e-6)
    assert r.drift_l2[idx["b"]] == pytest.approx(3.0, abs=1e-6)
    assert r.drift_l2[idx["c"]] == pytest.approx(5.0, abs=1e-6)

    # bf16 round-trip (dump storage dtype): ~3 significant digits, so pooling a
    # d=16 rep and differencing keeps error well under 1e-1·norm (= 0.5 here).
    r16 = representation_drift(
        acts_ref.to(torch.bfloat16), acts_now.to(torch.bfloat16), mask, classes
    )
    assert r16.drift_l2[idx["c"]] == pytest.approx(5.0, abs=5e-1)


def test_v5_14_masked_out_shift_ignored():
    """A huge shift added only at mask-False positions ⇒ drift 0 (pooling
    respects the mask, never flattens over padding)."""
    gen = torch.Generator().manual_seed(13)
    n, seq, d = 4, 6, 8
    acts_ref = torch.randn(n, seq, d, generator=gen, dtype=torch.float64)
    mask = torch.zeros(n, seq, dtype=torch.bool)
    mask[:, :2] = True  # pool positions 0,1 only
    classes = ["a", "a", "b", "b"]

    acts_now = acts_ref.clone()
    acts_now[:, 2:] += torch.full((d,), 99.0, dtype=torch.float64)  # masked-out

    r = representation_drift(acts_ref, acts_now, mask, classes)
    for x in r.drift_l2:
        assert x == pytest.approx(0.0, abs=1e-9)


def test_v5_14_degenerate_inputs_raise():
    """Shape mismatch, non-3-D acts, wrong mask shape, an all-False mask row,
    and a wrong classes length all refuse loudly."""
    gen = torch.Generator().manual_seed(14)
    acts = torch.randn(3, 4, 8, generator=gen, dtype=torch.float64)
    mask = torch.ones(3, 4, dtype=torch.bool)
    classes = ["a", "b", "c"]

    with pytest.raises(ValueError, match="shapes differ"):
        representation_drift(acts, acts[:, :3], mask, classes)
    with pytest.raises(ValueError, match="n, seq, d"):
        representation_drift(acts[0], acts[0], mask[0], classes)
    with pytest.raises(ValueError, match="mask shape"):
        representation_drift(acts, acts, mask[:, :2], classes)
    bad_mask = mask.clone()
    bad_mask[1] = False
    with pytest.raises(ValueError, match="zero mask-True"):
        representation_drift(acts, acts, bad_mask, classes)
    with pytest.raises(ValueError, match=r"len\(classes\)"):
        representation_drift(acts, acts, mask, ["a", "b"])


# --- V5.15 effective rank ---------------------------------------------------


def test_v5_15_planted_rank_r_delta_recovered():
    """Rank-r delta with equal (=1) singular values ⇒ effective rank exactly r,
    including a rectangular LoRA-shaped B@A construction."""
    gen = torch.Generator().manual_seed(21)
    d = 64
    for r in (1, 3, 8):
        u, _ = torch.linalg.qr(torch.randn(d, r, generator=gen, dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(d, r, generator=gen, dtype=torch.float64))
        delta = u @ v.T  # orthonormal factors ⇒ r singular values all equal to 1
        assert effective_rank(delta) == pytest.approx(r, abs=1e-6)

    # LoRA shape: B [d_out, r] @ A [r, d_in], rectangular d_out != d_in.
    d_out, d_in, r = 48, 80, 4
    ub, _ = torch.linalg.qr(torch.randn(d_out, r, generator=gen, dtype=torch.float64))
    va, _ = torch.linalg.qr(torch.randn(d_in, r, generator=gen, dtype=torch.float64))
    delta = ub @ va.T  # [d_out, d_in], rank 4, singular values all 1
    assert effective_rank(delta) == pytest.approx(r, abs=1e-6)


def test_v5_15_decaying_spectrum_between_one_and_rank():
    """A decaying spectrum diag(1, 0.1) ⇒ 1 < erank < 2 (numerical rank)."""
    delta = torch.diag(torch.tensor([1.0, 0.1], dtype=torch.float64))
    er = effective_rank(delta)
    assert 1.0 < er < 2.0
    # p = [1/1.1, 0.1/1.1] ⇒ exp(-Σ p ln p) ≈ 1.3562.
    assert er == pytest.approx(1.3562, abs=1e-3)


def test_v5_15_scaling_invariance():
    """Scaling ΔW by any scalar leaves the effective rank unchanged (the
    singular-value distribution p is scale-free)."""
    gen = torch.Generator().manual_seed(22)
    delta = torch.randn(32, 40, generator=gen, dtype=torch.float64)
    base = effective_rank(delta)
    for alpha in (1e-3, 7.5, 1e4):
        assert effective_rank(alpha * delta) == pytest.approx(base, abs=1e-9)


def test_v5_15_degenerate_inputs_raise():
    """Non-2-D input and the all-zero matrix both refuse loudly."""
    gen = torch.Generator().manual_seed(23)
    with pytest.raises(ValueError, match="2-D"):
        effective_rank(torch.randn(10, generator=gen, dtype=torch.float64))
    with pytest.raises(ValueError, match="all-zero"):
        effective_rank(torch.zeros(4, 4, dtype=torch.float64))


# --- V5.16 performance-aligned matching -------------------------------------


def test_v5_16_planted_curves_known_pairing():
    """Hand-built curves ⇒ the nearest-performance pairing and gaps are exact."""
    steps_a = [1, 2, 3, 4]
    perf_a = [0.9, 0.6, 0.3, 0.1]
    steps_b = [10, 20, 30, 40, 50]
    perf_b = [0.95, 0.61, 0.29, 0.12, 0.05]

    m = performance_aligned_matching(steps_a, perf_a, steps_b, perf_b)
    assert m.pairs == ((1, 10), (2, 20), (3, 30), (4, 40))
    assert m.perf_gaps == pytest.approx([0.05, 0.01, 0.01, 0.02], abs=1e-12)


def test_v5_16_monotone_accuracy_matched_steps_nondecreasing():
    """With strictly monotone performance on both arms, the matched step_b
    sequence is non-decreasing in step_a order (nearest match is monotone)."""
    for seed in (31, 32, 33):
        g = torch.Generator().manual_seed(seed)
        perf_a = torch.cumsum(torch.rand(6, generator=g) + 0.1, dim=0)
        perf_b = torch.cumsum(torch.rand(9, generator=g) + 0.1, dim=0)
        steps_a = list(range(1, 7))
        steps_b = list(range(1, 10))
        m = performance_aligned_matching(steps_a, perf_a.tolist(), steps_b, perf_b.tolist())
        matched_b = [b for _, b in m.pairs]
        assert matched_b == sorted(matched_b)


def test_v5_16_ties_broken_toward_earliest_step_b():
    """Two arm-B steps equidistant from the query ⇒ the earlier step_b wins,
    and the result is identical across repeated calls."""
    steps_a, perf_a = [5], [0.5]
    steps_b, perf_b = [10, 20], [0.4, 0.6]  # both exactly 0.1 from 0.5

    m1 = performance_aligned_matching(steps_a, perf_a, steps_b, perf_b)
    m2 = performance_aligned_matching(steps_a, perf_a, steps_b, perf_b)
    assert m1.pairs == ((5, 10),)
    assert m1.perf_gaps == pytest.approx([0.1], abs=1e-12)
    assert m1 == m2


def test_v5_16_degenerate_inputs_raise():
    """Length mismatch, an empty arm, non-increasing steps, and non-finite
    performance all refuse loudly."""
    with pytest.raises(ValueError, match="len"):
        performance_aligned_matching([1, 2], [0.1], [1, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="non-empty"):
        performance_aligned_matching([], [], [1], [0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        performance_aligned_matching([2, 1], [0.1, 0.2], [1, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="non-finite"):
        performance_aligned_matching([1, 2], [0.1, float("nan")], [1, 2], [0.1, 0.2])
