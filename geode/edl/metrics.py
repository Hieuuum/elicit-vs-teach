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

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from geode.zoo import (
    ConsistencyError,
    check_masking_consistency,
    load_run,
    prequential_records,
    test_loss,
)
from geode.zoo.store import run_dir


def epoch1_totals(run_id: str, *, store: Path | None = None) -> tuple[float, int, int]:
    """Sum ``loss_sum_nats``, ``label_token_count``, and example count over
    epoch-1 ``prequential.jsonl`` records only.

    Returns ``(total_loss_nats, total_label_tokens, total_examples)`` — the
    same epoch-1 stream underlies both the token-weighted and example-weighted
    EDL normalizations, so both denominators are exposed from one pass.
    """
    total_loss_nats = 0.0
    total_label_tokens = 0
    total_examples = 0
    for record in prequential_records(run_id, store=store):
        if record.epoch != 1:
            continue
        total_loss_nats += record.loss_sum_nats
        total_label_tokens += record.label_token_count
        total_examples += len(record.example_ids)
    return total_loss_nats, total_label_tokens, total_examples


def edl_from_totals(total_loss_nats: float, total_label_tokens: int, floor_nats: float) -> float:
    """EDL algebra: ``total_loss_nats − total_label_tokens · floor_nats``.

    The specs/01 §1 identity generalized to any per-label-token floor, in nats.
    ``edl_nats`` applies this with the held-out ``eval/test_loss.json`` loss as
    floor; ``edl_epoch1_nats`` applies it with a run's own global-min in-loop
    val loss (spec 00 §2) instead. On a constant-per-token-loss stream this is
    exactly ``(loss − floor) · tokens``; a floor equal to that constant loss
    yields exactly 0.
    """
    return total_loss_nats - total_label_tokens * floor_nats


def mdl_nats(run_id: str, *, store: Path | None = None) -> float:
    """Prequential MDL: Σ ``loss_sum_nats`` over epoch-1 ``prequential.jsonl`` records."""
    total_loss_nats, _, _ = epoch1_totals(run_id, store=store)
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

    mdl, n_label, _ = epoch1_totals(run_id, store=store)
    l_test = test_loss(run_id, store=store).loss_per_label_token_nats
    return edl_from_totals(mdl, n_label, l_test)


def edl_per_label_token(run_id: str, *, store: Path | None = None) -> float:
    """EDL/D: ``edl_nats`` divided by the epoch-1 label-token count ``N_label``."""
    edl = edl_nats(run_id, store=store)
    _, n_label, _ = epoch1_totals(run_id, store=store)
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


def _eval_log_records(run_id: str, *, store: Path | None = None) -> list[dict]:
    """The launcher's in-loop eval curve — ``runs/<id>/eval_log.jsonl`` (spec 02 §6).

    A script artifact (``train_target.py`` writes it), not a spec-00 geode log;
    read for the moving val floor and the eval-step grid (``prefix_edl_curve``)
    and for the run's own global-min val loss (``min_val_nats_from_eval_log``).
    """
    path = run_dir(run_id, store=store) / "eval_log.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def min_val_nats_from_eval_log(run_id: str, *, store: Path | None = None) -> float:
    """Global minimum ``val_loss_nats`` over ALL rows of ``eval_log.jsonl``
    (spec 02 §6) — stopping evals AND curve evals alike, explicitly NOT the
    value at the run's stop time (``ConvergenceTracker.min_nats`` only sees
    stopping evals; this reads the full logged curve). Used as the
    ``target_result.min_val_nats`` manifest field (spec 00 §2) and as the
    floor for ``edl_epoch1_nats``.

    No masking-hash guard (unlike ``edl_nats``'s test floor): every row comes
    from the same in-process ``task_format``/tokenizer as training within a
    single script run, so train/eval masking parity cannot diverge here.
    """
    return min(row["val_loss_nats"] for row in _eval_log_records(run_id, store=store))


def edl_epoch1_nats(run_id: str, *, store: Path | None = None) -> float:
    """EDL over the epoch-1 stream against the run's OWN global-min in-loop val
    loss as floor (spec 00 §2 ``target_result.edl_epoch1_nats``) — the Eq. 3
    subtraction term is ``(epoch-1 label tokens) · min_val_nats_from_eval_log``,
    NOT ``n_val · floor`` (owner-accepted: a training-stream MDL is dimensioned
    in training-stream tokens, not held-out eval tokens). Distinct from
    ``edl_nats``, which floors against the held-out ``eval/test_loss.json``
    loss and enforces masking-hash parity; this floor needs no such guard
    (see ``min_val_nats_from_eval_log``).
    """
    mdl, n_label, _ = epoch1_totals(run_id, store=store)
    floor = min_val_nats_from_eval_log(run_id, store=store)
    return edl_from_totals(mdl, n_label, floor)


def edl_epoch1_per_label_token(run_id: str, *, store: Path | None = None) -> float:
    """``edl_epoch1_nats`` divided by the epoch-1 label-token count (token-weighted;
    spec 00 §2 ``target_result.edl_per_label_token_nats``)."""
    edl = edl_epoch1_nats(run_id, store=store)
    _, n_label, _ = epoch1_totals(run_id, store=store)
    return edl / n_label


def edl_epoch1_per_example(run_id: str, *, store: Path | None = None) -> float:
    """``edl_epoch1_nats`` divided by the epoch-1 example count (example-weighted,
    NOT token-weighted; spec 00 §2 ``target_result.edl_per_example_nats``)."""
    edl = edl_epoch1_nats(run_id, store=store)
    _, _, n_examples = epoch1_totals(run_id, store=store)
    return edl / n_examples


def _fixed_test_floor_nats(run_id: str, *, store: Path | None = None) -> float:
    """The run's constant Eq. 3 floor (``eval/test_loss.json`` per-token loss),
    guarded for train/test masking parity exactly as ``edl_nats`` (D-1, V0.5)."""
    manifest = load_run(run_id, store=store)
    train_hash = manifest.data.get("masking_config_hash")
    if train_hash is None:
        raise ConsistencyError(
            f"run '{run_id}' manifest has no top-level 'masking_config_hash' "
            "extra field: cannot verify train/test masking parity (D-1)"
        )
    check_masking_consistency(run_id, train_hash, store=store)
    return test_loss(run_id, store=store).loss_per_label_token_nats


def prefix_edl_curve(
    run_id: str, *, floor: Literal["val", "test"], store: Path | None = None
) -> pd.DataFrame:
    """Prefix EDL at every in-loop eval step (spec 02 §6; V5.73).

    Unifies the divergent hand-rolled cumsum sites. At each eval step ``e`` (from
    ``eval_log.jsonl``) that still has an epoch-1 MDL prefix, the running
    MDL/tokens/examples come from the epoch-1 ``prequential.jsonl`` records (stream
    order) and ``EDL(e) = MDL(e) − tokens(e) · floor(e)``. ``floor`` is REQUIRED
    (no default — omitting it is a ``TypeError``) and picks the subtrahend, hence
    the curve's shape (the floor-artifact, decisions.md 2026-07-27): ``"val"`` is
    the moving per-step val loss (``eval_log.jsonl``), ``"test"`` the run's one
    constant ``eval/test_loss.json`` per-token loss (canonical Eq. 3, masking-parity
    guarded).

    Columns ``[step, examples, tokens, mdl_nats, val_nats, edl_nats,
    edl_per_token_nats, edl_per_example_nats]`` (all nats), one row per in-range
    eval step; ``val_nats`` is the moving val loss regardless of ``floor``.
    ``df.attrs`` carries ``n_epoch1_steps`` and the full-epoch-1 ``epoch1_totals``
    ``(loss, tokens, examples)`` — the last eval point need not reach the epoch-1
    cut (spec 02 §6).
    """
    if floor not in ("val", "test"):
        raise ValueError(f"prefix_edl_curve: floor must be 'val' or 'test', got {floor!r}")
    epoch1 = sorted(
        (r for r in prequential_records(run_id, store=store) if r.epoch == 1),
        key=lambda r: r.step,
    )
    cum_loss = np.cumsum([r.loss_sum_nats for r in epoch1])
    cum_tok = np.cumsum([r.label_token_count for r in epoch1])
    cum_ex = np.cumsum([len(r.example_ids) for r in epoch1])

    steps: list[int] = []
    vals: list[float] = []
    for row in sorted(_eval_log_records(run_id, store=store), key=lambda r: r["step"]):
        e = row["step"]
        if 1 <= e <= len(epoch1):  # evals past the epoch-1 cut have no MDL prefix
            steps.append(e)
            vals.append(row["val_loss_nats"])
    idx = np.array(steps, dtype=int) - 1
    val = np.array(vals, dtype=float)
    mdl, tok, ex = cum_loss[idx], cum_tok[idx], cum_ex[idx]
    subtrahend = val if floor == "val" else _fixed_test_floor_nats(run_id, store=store)
    edl = mdl - tok * subtrahend

    df = pd.DataFrame(
        {
            "step": np.array(steps, dtype=int),
            "examples": ex,
            "tokens": tok,
            "mdl_nats": mdl,
            "val_nats": val,
            "edl_nats": edl,
            "edl_per_token_nats": edl / tok,
            "edl_per_example_nats": edl / ex,
        }
    )
    df.attrs["n_epoch1_steps"] = len(epoch1)
    df.attrs["epoch1_totals"] = (
        (float(cum_loss[-1]), float(cum_tok[-1]), float(cum_ex[-1])) if epoch1 else (0.0, 0.0, 0.0)
    )
    return df


def nats_to_bits(x_nats: float) -> float:
    """Convert nats to bits: ``x_nats / ln(2)`` (V1.8)."""
    return x_nats / math.log(2)
