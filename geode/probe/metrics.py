"""Cross-example gradient-alignment metric (specs/02 §7 analysis metrics, V5.13).

One (snapshot, layer) at a time: given the per-example activation-gradient
matrix — one row per probe example — summarize how parallel the rows are.
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
