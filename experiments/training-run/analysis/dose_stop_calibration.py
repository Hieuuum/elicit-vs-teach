"""Dose stopping-rule calibration (new phase, 2026-07-26).

The dose installers stop on an ε/k plateau of the FULL-DOSE training loss
(spec 02 §6, V5.65/V5.66). ε/k could have been inherited from the target
stage's 0.002/5, but that number was calibrated on 1M-example held-out losses
at ``eval_every=500``; a 16-example full-batch descent at lr 3e-6 evaluated
every step is a different curve. Inheriting it would have been the
scope-reuse error this project keeps paying for.

So the two ends of the dose grid are run first with ``eps_nats: 0.0`` (never
trips a strict-improvement rule on a monotone descent, so ``max_steps``
bounds the run and the whole trajectory is recorded), and this script REPLAYS
the real rule over the recorded losses with the tested ``ConvergenceTracker``
— the same class the trainer uses, so a replay verdict is exactly the verdict
the run would have reached.

What matters is not where the rule fires in steps but **at what fraction of
each dose's total descent**: if one ε stops n=1 at 20% of its descent and
n=16 at 90%, the dose-response curve measures the stopping rule instead of
the dose (the endpoint-referenced-comparability failure, 2026-07-25). The
table this prints is the pin's evidence; it lands in decisions.md.

Usage (CPU, seconds):
    python3 dose_stop_calibration.py [--store DIR] [--runs id=n ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from geode.train import ConvergenceTracker, StoppingRule

# The candidate grid: the inherited target-stage rule, plus an order of
# magnitude either side of it, plus a patience variant. Firing behaviour is
# what selects; nothing here is tuned to a preferred stopping step.
CANDIDATES = ((0.02, 5), (0.002, 5), (0.002, 10), (0.0002, 5), (0.00002, 5))


def replay(losses: list[float], eps: float, k: int) -> tuple[int, float]:
    """Step at which ε/k fires over ``losses`` (1-indexed), and the loss there.

    ``eval_every`` is 1 for every dose run, so each recorded step is an eval
    and the replay consumes the losses in order. Returns ``(0, nan)`` if the
    rule never fires within the recorded trajectory.
    """
    tracker = ConvergenceTracker(StoppingRule(eps_nats=eps, k=k))
    for i, loss in enumerate(losses, start=1):
        if tracker.update(loss, step=i):
            return i, loss
    return 0, float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=None, help="default: $GEODE_STORE")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["evt-p2-cal-dose1=1", "evt-p2-cal-dose16=16"],
        help="run_id=dose pairs (the eps 0.0 calibration pilots)",
    )
    args = parser.parse_args()
    store = args.store or Path(os.environ.get("GEODE_STORE", "geode-store"))

    curves: dict[int, list[float]] = {}
    for spec in args.runs:
        run_id, _, dose = spec.partition("=")
        path = store / "runs" / run_id / "train_log.jsonl"
        losses = [json.loads(line)["train_loss_nats"] for line in path.open()]
        curves[int(dose)] = losses
        # The step-1 record is the loss BEFORE that step's update, i.e. the
        # step-0 baseline the launcher records; the last is the floor reached.
        print(
            f"[cal] dose {int(dose):>2}: {len(losses)} steps, "
            f"L0 {losses[0]:.4f} -> {losses[-1]:.6f} nats"
        )

    print(
        f"\n{'eps':>9} {'k':>3} | "
        + " | ".join(f"dose {d:<2} step  loss    %descent" for d in sorted(curves))
    )
    for eps, k in CANDIDATES:
        cells = []
        for dose in sorted(curves):
            losses = curves[dose]
            step, loss = replay(losses, eps, k)
            # Denominator is L0 - 0, not L0 - min(observed): the cross-entropy
            # floor of a memorisable finite set IS 0 (both pilots bottom out at
            # ~2e-4 nats), and a fixed reference keeps the doses comparable
            # instead of making each %descent depend on where its pilot
            # happened to be stopped.
            frac = 100.0 * (losses[0] - loss) / losses[0] if step else float("nan")
            cells.append(
                f"{step:>10} {loss:>7.4f} {frac:>9.2f}%"
                if step
                else f"{'never':>10} {'—':>7} {'—':>9} "
            )
        print(f"{eps:>9} {k:>3} | " + " | ".join(cells))

    print(
        "\nRead the %descent columns ACROSS a row: a usable pin fires at a "
        "comparable fraction of\ndescent at both ends of the grid. The "
        "absolute stopping loss is expected to differ (the\ntime constants "
        "do); the fraction of the dose actually absorbed must not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
