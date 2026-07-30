"""Stopping rules (specs/02 §6.1): loss convergence + behavioral (runs 3-4).

An eval "improves" iff ``(best_so_far - val_loss_nats) > eps_nats`` (strict;
an improvement of exactly ``eps_nats`` does not count). Improvement updates
the running best and resets the stale-eval counter; the ``k``-th consecutive
non-improving eval trips the stop, and the tracker latches stopped forever
after — every later call to ``update`` returns ``True`` regardless of its
value, since a continued-plateau reading of the "keeps returning True" clause
would make it redundant with the consecutive-count rule itself.

``min_nats`` is the exact running minimum over every value passed to
``update`` — independent of eps-gating, which freezes ``best_nats`` whenever
improvements stay <= ``eps_nats`` (run-1 v2/v2-ext manifests recorded a
first-eval "best" while the model kept improving sub-eps; 2026-07-20).

``min_steps`` (2026-07-21) is a grace period: evals at ``step < min_steps``
update ``min_nats`` only — ``best_nats`` stays frozen, the stale counter
does not move, and the tracker cannot stop. The first eval that counts is
the first at ``step >= min_steps`` (motivation: run-1 v3-ext warm-starts
reset AdamW moments, and the resulting val transient must not consume the
patience budget or plant a transient-high ``best_nats``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StoppingRule:
    """Minimum-improvement / patience stopping rule, in nats."""

    eps_nats: float  # minimum improvement that counts
    k: int  # consecutive non-improving evals that trigger a stop
    min_steps: int = 0  # evals before this step update min_nats only (grace)


class ConvergenceTracker:
    """Tracks the running-best validation loss and the stale-eval count."""

    def __init__(self, rule: StoppingRule) -> None:
        self._rule = rule
        self.best_nats: float = float("inf")  # +inf before first update
        self.min_nats: float = float("inf")  # +inf before first update
        self.stale_evals: int = 0
        self._stopped = False

    def update(self, val_loss_nats: float, step: int | None = None) -> bool:
        """Record one eval; return whether training should stop (latched).

        Raises ``ValueError`` if ``val_loss_nats`` is NaN, or if the rule
        has a grace period (``min_steps > 0``) and ``step`` is omitted —
        the grace cannot be applied without knowing where the eval falls.
        """
        if math.isnan(val_loss_nats):
            raise ValueError("ConvergenceTracker.update: val_loss_nats is NaN")
        if self._rule.min_steps > 0 and step is None:
            raise ValueError(
                "ConvergenceTracker.update: rule has min_steps "
                f"{self._rule.min_steps} but no step was passed"
            )
        if val_loss_nats < self.min_nats:
            self.min_nats = val_loss_nats
        if step is not None and step < self._rule.min_steps:
            return self._stopped
        if self._stopped:
            return True
        if (self.best_nats - val_loss_nats) > self._rule.eps_nats:
            self.best_nats = val_loss_nats
            self.stale_evals = 0
        else:
            self.stale_evals += 1
            if self.stale_evals >= self._rule.k:
                self._stopped = True
        return self._stopped


@dataclass(frozen=True)
class BehavioralStoppingRule:
    """Behavior-matched stopping rule for the format installers (specs/02 §6).

    Training stops at the ``k``-th *consecutive* in-loop behavioral eval whose
    metric (format-validity rate for runs 3-4) is ``>= threshold`` — the
    threshold is inclusive: "scoring >=99%" means a rate of exactly 0.99
    counts as a hit. A sub-threshold eval resets the consecutive count.
    """

    threshold: float  # metric value that counts as a hit (inclusive)
    k: int  # consecutive hits that trigger the stop


class BehaviorTracker:
    """Tracks consecutive above-threshold behavioral evals (latched, V5.44).

    Mirrors ``ConvergenceTracker``'s latching: once stopped, every later
    ``update`` returns ``True`` regardless of its value. ``best_rate`` is the
    exact running maximum over every value passed to ``update`` (-inf before
    the first), recorded for the manifest/meta the same way ``min_nats`` is.
    """

    def __init__(self, rule: BehavioralStoppingRule) -> None:
        self._rule = rule
        self.best_rate: float = float("-inf")
        self.hits: int = 0
        self._stopped = False

    def update(self, rate: float) -> bool:
        """Record one behavioral eval; return whether training should stop."""
        if math.isnan(rate):
            raise ValueError("BehaviorTracker.update: rate is NaN")
        if rate > self.best_rate:
            self.best_rate = rate
        if self._stopped:
            return True
        if rate >= self._rule.threshold:
            self.hits += 1
            if self.hits >= self._rule.k:
                self._stopped = True
        else:
            self.hits = 0
        return self._stopped
