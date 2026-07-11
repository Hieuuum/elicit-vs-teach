"""Run-log invariant and consistency checks (specs/00 V0.3, V0.5).

``check_epoch1_coverage`` enforces the spec 00 §3 invariant under resolved
decision OQ-4: concatenating ``example_ids`` over epoch-1 records enumerates
each id in ``0..n_unique_examples-1`` exactly once — skips, repeats, and
out-of-range ids are rejected with the offending id named. Records from
epoch >= 2 are ignored (later epochs are never used for MDL).

``check_masking_consistency`` is the V0.5 mechanical guard for the
train/test mask-parity footgun (spec 00 §5).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from geode.zoo.records import PrequentialRecord, test_loss


class ConsistencyError(ValueError):
    """A run-log invariant or consistency check failed; the message names the offender."""


def check_epoch1_coverage(records: Iterable[PrequentialRecord], n_unique_examples: int) -> None:
    """Enforce the epoch-1 exact-cover invariant (spec 00 §3, V0.3, OQ-4).

    Raises ``ConsistencyError`` naming the first out-of-range, repeated, or
    missing example id.
    """
    seen: set[int] = set()
    for record in records:
        if record.epoch != 1:
            continue
        for example_id in record.example_ids:
            if not 0 <= example_id < n_unique_examples:
                raise ConsistencyError(
                    f"epoch-1 example id {example_id} is outside the universe "
                    f"0..{n_unique_examples - 1} (OQ-4)"
                )
            if example_id in seen:
                raise ConsistencyError(f"epoch-1 example id {example_id} appears more than once")
            seen.add(example_id)
    if len(seen) != n_unique_examples:
        missing = min(set(range(n_unique_examples)) - seen)
        raise ConsistencyError(
            f"epoch-1 example id {missing} is missing: epoch-1 records must "
            f"enumerate each unique example exactly once (spec 00 §3)"
        )


def check_masking_consistency(run_id: str, train_hash: str, *, store: Path | None = None) -> None:
    """Enforce train/eval ``masking_config_hash`` parity (spec 00 §5, V0.5).

    Raises ``ConsistencyError`` naming both hashes on mismatch.
    """
    eval_hash = test_loss(run_id, store=store).masking_config_hash
    if eval_hash != train_hash:
        raise ConsistencyError(
            f"masking_config_hash mismatch for run '{run_id}': training loop "
            f"recorded {train_hash} but eval/test_loss.json has {eval_hash}"
        )
