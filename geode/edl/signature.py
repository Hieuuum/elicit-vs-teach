"""EDL-per-token scaling signature (specs/01 V1.12, 2026-09-29).

"Bits That Count" §4: EDL/D against training-set size n decreases
monotonically when elicitation dominates and has an increasing phase when
teaching does. ``edl_signature`` turns that reading into a rule over one or
more seeds' curves on the same sizes:

- ``tol = max(abs_tol, rel_tol * range of the seed-mean curve)``;
- "increasing" if the seed-mean curve rises by more than ``tol`` between two
  consecutive sizes (or from the first size to the last) and every seed rises
  there too;
- "decreasing" if no such rise exists, the seed-mean curve ends more than
  ``tol`` below where it starts, and every seed ends below its start;
- "mixed" if the seeds disagree (a rise or drop past ``tol`` that not every
  seed shares) — not a call;
- "flat" otherwise.

With one seed the "every seed" clauses hold trivially. Adding a constant to
every curve (another floor level) never changes the call; rescaling by a
positive factor (nats vs bits) does not either while ``rel_tol * range``
exceeds ``abs_tol``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def edl_signature(
    ns: Sequence[int],
    curves: Sequence[Sequence[float]],
    *,
    rel_tol: float = 0.1,
    abs_tol: float = 0.01,
) -> dict:
    """Call the scaling signature of EDL/D curves (one per seed) over sizes ``ns``.

    Returns ``call`` ("decreasing", "increasing", "mixed", "flat", or "too few
    sizes" below three sizes), ``tol``, ``mean`` (the seed-mean curve),
    ``max_rise`` and ``rise_at`` (the consecutive pair with the largest mean
    rise), ``net`` (mean last minus first), ``seed_calls`` (each seed judged
    alone), ``agree`` (how many of those equal ``call``) and ``n_seeds``.
    Units are whatever the curves carry.
    """
    ns = list(ns)
    curves = [[float(y) for y in c] for c in curves]
    if not curves:
        raise ValueError("edl_signature: no curves")
    if any(b <= a for a, b in zip(ns, ns[1:])):
        raise ValueError(f"edl_signature: sizes must be strictly increasing, got {ns}")
    if any(len(c) != len(ns) for c in curves):
        raise ValueError("edl_signature: every curve needs one value per size")
    if not all(math.isfinite(y) for c in curves for y in c):
        raise ValueError("edl_signature: non-finite EDL/D value")
    mean = [sum(col) / len(col) for col in zip(*curves)]
    out = {"call": "too few sizes", "tol": None, "mean": mean, "max_rise": None, "rise_at": None,
           "net": mean[-1] - mean[0] if mean else None, "seed_calls": [], "agree": 0,
           "n_seeds": len(curves)}
    if len(ns) < 3:
        return out
    tol = max(abs_tol, rel_tol * (max(mean) - min(mean)))
    rises = [b - a for a, b in zip(mean, mean[1:])]
    net = mean[-1] - mean[0]
    up = [j for j, r in enumerate(rises) if r > tol]
    if up or net > tol:
        shared = any(all(c[j + 1] > c[j] for c in curves) for j in up) or (
            net > tol and all(c[-1] > c[0] for c in curves))
        call = "increasing" if shared else "mixed"
    elif -net > tol:
        call = "decreasing" if all(c[-1] < c[0] for c in curves) else "mixed"
    else:
        call = "flat"
    i = max(range(len(rises)), key=rises.__getitem__)
    seed_calls = ([edl_signature(ns, [c], rel_tol=rel_tol, abs_tol=abs_tol)["call"] for c in curves]
                  if len(curves) > 1 else [call])
    out.update(call=call, tol=tol, max_rise=rises[i], rise_at=(ns[i], ns[i + 1]), net=net,
               seed_calls=seed_calls, agree=sum(sc == call for sc in seed_calls))
    return out
