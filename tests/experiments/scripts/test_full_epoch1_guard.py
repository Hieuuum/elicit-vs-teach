"""``experiment.require_full_epoch1`` guard (specs/02 V5.75/V5.76, ts38-mini
guard 1) and the ``epoch1_examples`` persistence it depends on.

Module under test: ``train_target.py`` (loaded via ``tests._scriptloader``,
the hyphenated ``experiments/training-run`` tree cannot be a dotted import).
Full end-to-end ``train_target.main()`` needs a real checkpoint, frozen
parquet, and network access, so this exercises the two small functions the
guard was factored into directly — never a copy of their logic:

- ``require_full_epoch1_launch_check`` — the launch-time guard, pure (no I/O).
- ``epoch1_examples_result`` — the post-run guard: reads a run's
  ``logs/prequential.jsonl`` via ``geode.edl.metrics.epoch1_totals`` and
  (only when the flag is set) raises on a truncated epoch 1. CPU-only, no
  network: the "training" is a couple of hand-written prequential records
  written under the ``geode_store`` fixture's temp dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geode.zoo import PrequentialRecord, write_jsonl
from tests._scriptloader import load

train_target = load("train_target")


def _write_epoch1(store: Path, run_id: str, n_examples: int, n_covered: int) -> None:
    """Write ``n_covered`` epoch-1 examples (one per record) for ``run_id``,
    stream-order ids ``0..n_covered-1`` — a prefix of the full ``n_examples``
    universe when ``n_covered < n_examples`` (an incomplete epoch 1)."""
    records = [
        PrequentialRecord(
            step=i, epoch=1, example_ids=[i], label_token_count=3, loss_sum_nats=0.9
        )
        for i in range(n_covered)
    ]
    write_jsonl(store / "runs" / run_id / "logs" / "prequential.jsonl", records)


# --- V5.75: launch-time guard -----------------------------------------------


def test_flag_true_correct_min_steps_accepted() -> None:
    # min_steps pinned to exactly one epoch: no raise.
    train_target.require_full_epoch1_launch_check(True, 782, 782)


def test_flag_true_wrong_min_steps_raises() -> None:
    with pytest.raises(ValueError, match="require_full_epoch1"):
        train_target.require_full_epoch1_launch_check(True, 500, 782)


def test_flag_absent_launch_check_never_raises() -> None:
    # Absent/false = today's behavior — any min_steps, including a mismatch
    # that would fail were the flag set, passes untouched.
    train_target.require_full_epoch1_launch_check(False, 500, 782)
    train_target.require_full_epoch1_launch_check(False, 0, 782)


# --- V5.76: post-run epoch1_examples persistence + truncation guard --------


def test_epoch1_examples_persisted_on_a_complete_epoch(geode_store: Path) -> None:
    _write_epoch1(geode_store, "run-complete", n_examples=10, n_covered=10)

    result = train_target.epoch1_examples_result(
        "run-complete", n_examples=10, require_full_epoch1=True, store=geode_store
    )

    assert result == 10


def test_epoch1_examples_truncation_raises_when_flag_set(geode_store: Path) -> None:
    # Epoch 1 stopped after 6 of 10 examples (min_steps=0-is-safe failure mode).
    _write_epoch1(geode_store, "run-truncated", n_examples=10, n_covered=6)

    with pytest.raises(RuntimeError, match="require_full_epoch1"):
        train_target.epoch1_examples_result(
            "run-truncated", n_examples=10, require_full_epoch1=True, store=geode_store
        )


def test_epoch1_examples_persisted_regardless_of_flag(geode_store: Path) -> None:
    # Flag absent/false: epoch1_examples is still computed and returned (it is
    # written into target_result for EVERY run) but a truncated epoch never
    # raises — "today's behavior" for runs that don't opt into the guard.
    _write_epoch1(geode_store, "run-truncated-unguarded", n_examples=10, n_covered=6)

    result = train_target.epoch1_examples_result(
        "run-truncated-unguarded",
        n_examples=10,
        require_full_epoch1=False,
        store=geode_store,
    )

    assert result == 6
