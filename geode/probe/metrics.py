"""Analysis metrics over probe dumps (specs/02 §7, V5.13–V5.16, V5.63).

``gradient_alignment`` (V5.13) — one (snapshot, layer) at a time: given the
per-example activation-gradient matrix (one row per probe example) summarize
how parallel the rows are.
Expectation (spec §7): elicitation ⇒ near-parallel, teaching ⇒ diverse.

Two summaries, both pure functions of the matrix:

- ``pairwise_cosine_mean``: mean cosine similarity over all ordered
  off-diagonal pairs. Parallel same-direction rows ⇒ 1; isotropic random
  rows ⇒ ≈ 0 (each pair's cosine concentrates as N(0, 1/d)).
- ``top_pc_explained_variance``: fraction of total squared norm carried by
  the top principal component of the (uncentered) matrix — uncentered
  because mean-centering would erase exactly the shared-direction structure
  the metric exists to detect. Parallel rows ⇒ 1; isotropic random rows ⇒
  ≈ 1/n_examples, NOT 0 — the n ≪ d caveat V5.13 pins numerically.

Rows with zero norm raise: late-training gradients are numerically
degenerate, and spec §7 has analyses condition on nonzero per-example probe
loss before calling this — silently including zero rows would corrupt both
summaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradientAlignment:
    """Alignment summary of one per-example gradient matrix (V5.13)."""

    n_examples: int
    pairwise_cosine_mean: float
    top_pc_explained_variance: float


def gradient_alignment(grads: torch.Tensor) -> GradientAlignment:
    """Alignment of per-example gradients: ``grads`` is ``[n_examples, ...]``.

    Trailing dimensions are flattened to one vector per example (padding
    positions carry exactly-zero gradient, so they change neither cosines nor
    the spectrum). Computation is fp64 regardless of input dtype (dumps are
    bf16). Requires >= 2 examples and no zero-norm row (filter on nonzero
    probe loss first — spec §7).
    """
    if grads.ndim < 2:
        raise ValueError(
            f"gradient_alignment: expected [n_examples, ...], got {tuple(grads.shape)}"
        )
    g = grads.reshape(grads.shape[0], -1).to(torch.float64)
    n = g.shape[0]
    if n < 2:
        raise ValueError(f"gradient_alignment: need >= 2 examples, got {n}")
    norms = g.norm(dim=1)
    if bool((norms == 0).any()):
        zero_rows = torch.nonzero(norms == 0).flatten().tolist()
        raise ValueError(
            f"gradient_alignment: zero-norm gradient rows {zero_rows} — filter "
            "examples to nonzero probe loss before computing alignment (spec 02 §7)"
        )

    unit = g / norms[:, None]
    cos = unit @ unit.T
    pairwise = float((cos.sum() - cos.diagonal().sum()).item() / (n * (n - 1)))

    s = torch.linalg.svdvals(g)
    evr = float((s[0] ** 2 / (s**2).sum()).item())

    return GradientAlignment(
        n_examples=n, pairwise_cosine_mean=pairwise, top_pc_explained_variance=evr
    )


@dataclass(frozen=True)
class RepresentationDrift:
    """Per-class L2 drift of pooled activations between two snapshots (V5.14)."""

    n_examples: int
    classes: tuple[str, ...]  # sorted distinct class labels
    drift_l2: tuple[float, ...]  # aligned with ``classes``


def representation_drift(
    acts_ref: torch.Tensor,
    acts_now: torch.Tensor,
    mask: torch.Tensor,
    classes: Sequence[str],
) -> RepresentationDrift:
    """Per-class representation drift between a reference and current snapshot.

    ``acts_ref``/``acts_now`` are ``[n, seq, d]`` activations at the same probe
    positions in two snapshots; ``mask`` is the ``[n, seq]`` bool label mask
    (probe tokenization is identical across snapshots, so one mask pools both);
    ``classes`` is the length-``n`` per-example class label (e.g. the probe cell
    ``"2x3"``).

    Per example the representation is the mean activation vector over the
    ``mask``-True positions (a ``d``-vector, fp64). Per class ``c`` the drift is
    ``|| mean_{i∈c} rep_now[i] − mean_{i∈c} rep_ref[i] ||₂``. Output ``classes``
    are sorted lexicographically and ``drift_l2`` is aligned to them.

    Unlike gradients, padding activations are NOT zero — that is exactly why
    pooling must go through the mask, never flatten. Computation is fp64
    regardless of input dtype (dumps are bf16).
    """
    if acts_ref.ndim != 3 or acts_now.ndim != 3:
        raise ValueError(
            "representation_drift: expected [n, seq, d] activations, got "
            f"ref {tuple(acts_ref.shape)} now {tuple(acts_now.shape)}"
        )
    if acts_ref.shape != acts_now.shape:
        raise ValueError(
            f"representation_drift: acts_ref {tuple(acts_ref.shape)} and acts_now "
            f"{tuple(acts_now.shape)} shapes differ"
        )
    n = acts_ref.shape[0]
    if n == 0:
        raise ValueError("representation_drift: no examples (n == 0)")
    if tuple(mask.shape) != tuple(acts_ref.shape[:2]):
        raise ValueError(
            f"representation_drift: mask shape {tuple(mask.shape)} != acts [n, seq] "
            f"{tuple(acts_ref.shape[:2])}"
        )
    if len(classes) != n:
        raise ValueError(f"representation_drift: len(classes) {len(classes)} != n_examples {n}")
    m = mask.to(torch.bool)
    counts = m.sum(dim=1)
    if bool((counts == 0).any()):
        zero_rows = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(
            f"representation_drift: examples {zero_rows} have zero mask-True "
            "positions — cannot pool a representation"
        )

    mf = m.to(torch.float64)[:, :, None]
    denom = counts.to(torch.float64)[:, None]
    rep_ref = (acts_ref.to(torch.float64) * mf).sum(dim=1) / denom
    rep_now = (acts_now.to(torch.float64) * mf).sum(dim=1) / denom

    class_list = list(classes)
    sorted_classes = sorted(set(class_list))
    drift: list[float] = []
    for c in sorted_classes:
        idx = torch.tensor([i for i, cl in enumerate(class_list) if cl == c], dtype=torch.long)
        delta = rep_now[idx].mean(dim=0) - rep_ref[idx].mean(dim=0)
        drift.append(float(delta.norm().item()))

    return RepresentationDrift(n_examples=n, classes=tuple(sorted_classes), drift_l2=tuple(drift))


def effective_rank(delta_w: torch.Tensor) -> float:
    """Spectral-entropy effective rank of a 2-D matrix (Roy & Vetterli 2007).

    With singular values σ of ``delta_w`` and ``p_i = σ_i / Σσ`` over the
    nonzero σ, the effective rank is ``exp(−Σ p_i ln p_i)``. This is the ΔW
    spectrum summary for adapter diffs (spec 02 §7). It equals ``r`` exactly
    when the ``r`` nonzero singular values are equal, and is strictly less than
    the numerical rank whenever the spectrum decays. Computation is fp64
    regardless of input dtype.
    """
    if delta_w.ndim != 2:
        raise ValueError(f"effective_rank: expected a 2-D matrix, got {tuple(delta_w.shape)}")
    s = torch.linalg.svdvals(delta_w.to(torch.float64))
    s = s[s > 0]  # drop exact zeros
    if s.numel() == 0:
        raise ValueError("effective_rank: all-zero matrix — no nonzero singular values")
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


@dataclass(frozen=True)
class PerformanceMatch:
    """Cross-arm snapshot pairing at nearest performance (V5.16)."""

    pairs: tuple[tuple[int, int], ...]  # (step_a, step_b), one per step_a, in step_a order
    perf_gaps: tuple[float, ...]  # |perf_a − perf_b| per pair


def performance_aligned_matching(
    steps_a: Sequence[int],
    perf_a: Sequence[float],
    steps_b: Sequence[int],
    perf_b: Sequence[float],
) -> PerformanceMatch:
    """Pair each arm-A snapshot with the arm-B snapshot closest in performance.

    For each ``steps_a[i]`` the match is the ``steps_b[j]`` minimizing
    ``|perf_a[i] − perf_b[j]|``; ties break deterministically toward the
    earliest ``step_b``. The metric is direction-agnostic — it works for loss
    in nats going down or accuracy going up — and is spec 02 §7's primary
    cross-arm comparison axis (step-aligned matching is the secondary axis).
    Gaps are computed in fp64. One pair per arm-A snapshot, in ``steps_a`` order.
    """
    sa, pa = list(steps_a), list(perf_a)
    sb, pb = list(steps_b), list(perf_b)
    if len(sa) != len(pa):
        raise ValueError(
            f"performance_aligned_matching: arm A len(steps) {len(sa)} != len(perf) {len(pa)}"
        )
    if len(sb) != len(pb):
        raise ValueError(
            f"performance_aligned_matching: arm B len(steps) {len(sb)} != len(perf) {len(pb)}"
        )
    if len(sa) == 0 or len(sb) == 0:
        raise ValueError("performance_aligned_matching: both arms must be non-empty")
    for name, steps in (("A", sa), ("B", sb)):
        if any(steps[i] >= steps[i + 1] for i in range(len(steps) - 1)):
            raise ValueError(
                f"performance_aligned_matching: arm {name} steps not strictly increasing: {steps}"
            )
    for name, perf in (("A", pa), ("B", pb)):
        if not all(math.isfinite(float(x)) for x in perf):
            raise ValueError(f"performance_aligned_matching: arm {name} has non-finite perf values")

    pb_f = [float(x) for x in pb]
    pairs: list[tuple[int, int]] = []
    gaps: list[float] = []
    for a_step, a_perf in zip(sa, pa):
        a = float(a_perf)
        best_j = 0
        best_gap = abs(pb_f[0] - a)
        for j in range(1, len(pb_f)):
            g = abs(pb_f[j] - a)
            if g < best_gap:  # strict ⇒ earliest step_b wins ties
                best_gap = g
                best_j = j
        pairs.append((int(a_step), int(sb[best_j])))
        gaps.append(best_gap)

    return PerformanceMatch(pairs=tuple(pairs), perf_gaps=tuple(gaps))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two representation matrices of the same examples.

    ``x`` is ``[n, d1]`` and ``y`` is ``[n, d2]`` — one row per example, row i
    of both the *same* example; the widths may differ, which is why CKA (not a
    distance) is the cross-arm similarity measure. Both are column-centered,
    then

        CKA = ‖Xᶜᵀ Yᶜ‖_F² / (‖Xᶜᵀ Xᶜ‖_F · ‖Yᶜᵀ Yᶜ‖_F)

    which is 1 for representations related by any orthogonal rotation and
    isotropic rescaling, 0 for uncorrelated ones, and invariant to translation
    (centering). Values lie in [0, 1] by Cauchy–Schwarz, up to fp64 rounding.
    Computation is fp64 regardless of input dtype (dumps are bf16).

    A constant (zero-variance) matrix has no linear structure to compare and
    refuses rather than dividing by zero.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"linear_cka: expected [n_examples, d] matrices, got x {tuple(x.shape)} "
            f"y {tuple(y.shape)}"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"linear_cka: row counts differ ({x.shape[0]} vs {y.shape[0]}) — row i of "
            "both matrices must be the same probe example"
        )
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"linear_cka: need >= 2 examples, got {n}")

    xf, yf = x.to(torch.float64), y.to(torch.float64)
    for name, t in (("x", xf), ("y", yf)):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"linear_cka: non-finite values in {name}")

    xc = xf - xf.mean(dim=0, keepdim=True)
    yc = yf - yf.mean(dim=0, keepdim=True)
    gram_x = (xc.T @ xc).norm()
    gram_y = (yc.T @ yc).norm()
    for name, g in (("x", gram_x), ("y", gram_y)):
        if float(g.item()) == 0.0:
            raise ValueError(
                f"linear_cka: {name} has zero centered Frobenius norm (constant rows) "
                "— no linear structure to compare"
            )

    cross = (xc.T @ yc).norm() ** 2
    return float((cross / (gram_x * gram_y)).item())
