"""MDL / EDL metrics + reporting (specs/01 §1 "Definitions"; specs/00 §3, §5, §8).

Reads the artifacts a run writes under specs/00: the epoch-1 records of
``logs/prequential.jsonl`` (§3, via ``geode.zoo.prequential_records``) and
``eval/test_loss.json`` (§5, via ``geode.zoo.test_loss``). ``edl_nats`` and its
normalizations enforce the V0.5 / V1.4(a) train/test masking-parity guard
(D-1) before computing anything: a run manifest missing its
``masking_config_hash`` extra, or one whose hash disagrees with
``eval/test_loss.json``, raises ``geode.zoo.ConsistencyError`` rather than
silently mixing masking configs into the headline metric.

Units: nats internally; ``nats_to_bits`` converts only at reporting
boundaries (specs/01 §1, V1.8).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from geode.zoo import (
    ConsistencyError,
    check_masking_consistency,
    load_run,
    prequential_records,
    test_loss,
)


def _epoch1_totals(run_id: str, *, store: Path | None = None) -> tuple[float, int]:
    """Sum ``loss_sum_nats`` and ``label_token_count`` over epoch-1 records only."""
    total_loss_nats = 0.0
    total_label_tokens = 0
    for record in prequential_records(run_id, store=store):
        if record.epoch != 1:
            continue
        total_loss_nats += record.loss_sum_nats
        total_label_tokens += record.label_token_count
    return total_loss_nats, total_label_tokens


def mdl_nats(run_id: str, *, store: Path | None = None) -> float:
    """Prequential MDL: Σ ``loss_sum_nats`` over epoch-1 ``prequential.jsonl`` records."""
    total_loss_nats, _ = _epoch1_totals(run_id, store=store)
    return total_loss_nats


def edl_nats(run_id: str, *, store: Path | None = None) -> float:
    """EDL = MDL − N_label · L_test, guarded by masking-hash parity (D-1, V0.5, V1.4(a)).

    Raises ``geode.zoo.ConsistencyError`` when the run manifest lacks a
    top-level ``masking_config_hash`` extra field (parity cannot be
    verified), or when that hash disagrees with ``eval/test_loss.json``'s.
    """
    manifest = load_run(run_id, store=store)
    train_hash = manifest.data.get("masking_config_hash")
    if train_hash is None:
        raise ConsistencyError(
            f"run '{run_id}' manifest has no top-level 'masking_config_hash' "
            "extra field: cannot verify train/test masking parity (D-1)"
        )
    check_masking_consistency(run_id, train_hash, store=store)

    mdl, n_label = _epoch1_totals(run_id, store=store)
    l_test = test_loss(run_id, store=store).loss_per_label_token_nats
    return mdl - n_label * l_test


def edl_per_label_token(run_id: str, *, store: Path | None = None) -> float:
    """EDL/D: ``edl_nats`` divided by the epoch-1 label-token count ``N_label``."""
    edl = edl_nats(run_id, store=store)
    _, n_label = _epoch1_totals(run_id, store=store)
    return edl / n_label


def edl_per_param(run_id: str, *, store: Path | None = None) -> float:
    """EDL/P: ``edl_nats`` divided by the manifest's ``trainable_param_count``."""
    edl = edl_nats(run_id, store=store)
    manifest = load_run(run_id, store=store)
    return edl / manifest.data["trainable_param_count"]


def pgr(perf_tuned: float, perf_base: float, perf_fullft: float, *, eps: float = 1e-8) -> float:
    """PGR = (Perf_tuned − Perf_base) / (Perf_fullft − Perf_base) (D-6, D-8).

    Raises ``ValueError`` when the denominator is degenerate
    (``abs(perf_fullft - perf_base) < eps``) rather than returning ±inf.
    """
    denom = perf_fullft - perf_base
    if abs(denom) < eps:
        raise ValueError(
            f"pgr: degenerate denominator |Perf_FullFT - Perf_base| = {abs(denom)} < eps={eps}"
        )
    return (perf_tuned - perf_base) / denom


def training_curve(run_id: str, *, store: Path | None = None) -> pd.DataFrame:
    """D-3: epoch-1 loss vs. cumulative example index, one row per record, in stream order.

    Columns exactly ``[example_index, loss_sum_nats, label_token_count,
    loss_per_label_token_nats]``. ``example_index`` is the cumulative count
    of epoch-1 examples in strictly prior epoch-1 batches (0-based);
    epoch>1 records are excluded and never shift the running count.
    """
    columns = ["example_index", "loss_sum_nats", "label_token_count", "loss_per_label_token_nats"]
    rows = []
    cumulative_examples = 0
    for record in prequential_records(run_id, store=store):
        if record.epoch != 1:
            continue
        rows.append(
            {
                "example_index": cumulative_examples,
                "loss_sum_nats": record.loss_sum_nats,
                "label_token_count": record.label_token_count,
                "loss_per_label_token_nats": record.loss_sum_nats / record.label_token_count,
            }
        )
        cumulative_examples += len(record.example_ids)
    return pd.DataFrame(rows, columns=columns)


def nats_to_bits(x_nats: float) -> float:
    """Convert nats to bits: ``x_nats / ln(2)`` (V1.8)."""
    return x_nats / math.log(2)
