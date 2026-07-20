"""Validation-loss convergence stopping rule (specs/02 §6.1).

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StoppingRule:
    """Minimum-improvement / patience stopping rule, in nats."""

    eps_nats: float  # minimum improvement that counts
    k: int  # consecutive non-improving evals that trigger a stop


class ConvergenceTracker:
    """Tracks the running-best validation loss and the stale-eval count."""

    def __init__(self, rule: StoppingRule) -> None:
        self._rule = rule
        self.best_nats: float = float("inf")  # +inf before first update
        self.min_nats: float = float("inf")  # +inf before first update
        self.stale_evals: int = 0
        self._stopped = False

    def update(self, val_loss_nats: float) -> bool:
        """Record one eval; return whether training should stop (latched).

        Raises ``ValueError`` if ``val_loss_nats`` is NaN.
        """
        if math.isnan(val_loss_nats):
            raise ValueError("ConvergenceTracker.update: val_loss_nats is NaN")
        if val_loss_nats < self.min_nats:
            self.min_nats = val_loss_nats
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
