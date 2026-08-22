"""Routing-vs-computing control for ``resid_probe.py`` (Phase 0 / Tier 1 test 1).

``resid_probe.py`` fits a multinomial logistic probe of the FIRST answer
token from the residual stream, on bare-NL add/sub prompts like ``What is
the sum of 943 and 5881?\\n6824`` / ``What is the difference between 560 and
1188?\\n-628``. Digits tokenise one per token, the answer almost never has a
leading zero, and the class set is ``{'-','1',...,'9'}`` in practice — 10
classes, ``'-'`` the majority class at roughly 25% (11 classes are possible:
``a == b`` subtraction yields a leading ``'0'``, 9/100000 rows in the real
``D_algo_eval_bare.parquet`` — rare but real, and ``build_reference`` derives
``n_classes`` from whatever answer tokens actually occur, so a full-file run
can legitimately report 11). That first token is LARGELY predictable from
the operands' TOP-POSITION digits alone, with no arithmetic at all: a model
that merely routes operand digits to the last position gives a linear probe
~+0.3 over the majority-class baseline. High probe accuracy is therefore not,
by itself, evidence of a computed result — it could be evidence of routing.

This script splits probe accuracy into two subsets of a set's TEST examples:

- **determined**: the naive "look at the top-position digits, no carry/
  borrow logic" rule already predicts the true first answer token
  (``top_position_determined`` below).
- **affected**: it does not — a carry, borrow, cancellation, or sign tie
  changes the first token, so getting it right requires something beyond
  routing the top-position digits.

Only ``acc_affected`` (probe accuracy restricted to the affected subset) is
evidence of a computed result, and only when read against the RIGHT floor.
``majority_affected_acc`` (the affected subset's own chance level) is the
minimum bar. ``token_baseline_acc_affected`` (see "Token-linear baseline"
below) can itself land above ``majority_affected_acc`` by a modest but real
margin, so it — not ``majority_affected_acc`` alone — is the conservative
comparator: an ``acc_affected`` that only clears the majority floor and not
the token-linear one is still consistent with routing. ``acc_determined``
alone is consistent with pure digit routing and proves nothing about
computation.

**Exact rule.** For ints ``a, b >= 0`` and ``op in {'+', '-'}``:
``L = max(len(str(a)), len(str(b)))``; ``ta = a // 10**(L-1)``,
``tb = b // 10**(L-1)`` (the shorter operand's top-position digit is 0);
``truth = str(a+b)[0]`` if ``op == '+'`` else ``str(a-b)[0]``.
``top_position_prediction``: ``op == '+'`` -> ``str(ta+tb)[0]``; ``op == '-'``
and ``ta > tb`` -> ``str(ta-tb)[0]``; ``ta < tb`` -> ``'-'`` (the sign is
decided at the top position, the leading digit is not); ``ta == tb`` ->
``None`` (neither sign nor leading digit is decided at the top position).
``top_position_determined`` is ``True`` iff the prediction is not ``None``
and equals the truth.

**Token-linear baseline.** ``token_features`` is a model-free readout of the
SAME routing information a probe over perfectly-routed digit tokens could
exploit (op one-hot, operand-length one-hots, and each operand's digits at
LEFT-aligned decimal positions — position 0 is always the operand's TOP
digit, the one ``top_position_prediction`` reads, at the same feature index
regardless of operand length — one-hot including "absent"). Fitting the same
linear probe on these features (``token_baseline_acc``/
``token_baseline_acc_affected``) gives a model-FREE null: what a linear
readout gets from routing alone, with no model and no arithmetic. Because
position 0 already IS the top digit at every operand length,
``top_position_prediction`` is a genuinely LINEAR function of ``a``'s and
``b``'s position-0 one-hots plus ``op`` — no length cross-referencing needed
— so this baseline tracks ``determined_frac`` closely with a modest amount
of test data: on the real ``D_algo_eval_bare.parquet`` (seed 0),
``token_baseline_acc`` is already clearly above ``majority_test_acc`` by a
few hundred test examples (0.42 vs 0.26 at 250) and close to double it at the
regime this script actually runs at (0.48 vs 0.26 at 1000 test examples, out
of 2000 total). ``token_baseline_acc_affected`` hovers near
``majority_affected_acc`` rather than tracking it exactly — carry/borrow/
cancellation/sign-tie logic is not a LINEAR function of digit one-hots,
left- or right-aligned, but it is not perfectly uncorrelated with the top
digits either. Which side of the floor it lands on varies by draw: below it
on the real parquet at seed 0 (0.13 vs 0.19 at 1000 test examples, 227
affected), modestly ABOVE it (up to +0.11) on uniform-random operands across
20 calibration seeds at n=3000 (larger affected subsets, so a tighter
estimate). Read ``acc_affected`` against ``token_baseline_acc_affected``,
not just ``majority_affected_acc`` — the module docstring above explains
why.

Class space and the seeded half/half train-test split are fixed ONCE per
prompt set, exactly as in ``resid_probe.py``'s ``build_reference`` (reused
directly, not reimplemented), so every model/layer/subset accuracy reported
here answers the same probe question.

**Cross-format transfer probe (``--transfer-set``, decisions.md 2026-08-22
"probe trajectory").** Renders an *operator-notation twin* of the named
set's rows — the SAME ``(a, b, op)`` problems in the ``Question: a + b\\n
Answer: <true answer>`` format the ts38pp parent was pre-taught on
(``geode.arith.formats.render(..., "operator")``) — and, per layer, fits the
probe on the twin's TRAIN half and scores it on the original set's TEST
half (``op_to_task``), plus the reverse (``task_to_op``) and the two
within-format references on the same split (``op_to_op``, ``task_to_task``).
A probe that transfers reads the SAME representation of the answer in both
formats; one that does not may mean the format never reaches the
arithmetic, or merely a different subspace — so transfer success is the
strong read, failure is not. Scored on **positive-answer examples only**,
classes = first digit ``1..9`` by answer TEXT: in the operator rendering a
negative answer's first token is the merged ``Ġ-`` (the tokenizer fuses the
space and the sign) and the generating position differs by sign, while a
positive answer's digit tokens and generating position are the same in
both formats. The affected/determined split is applied within that subset.

Usage:
    python3 probe_routing_control.py --model pp0=run:evt-ts38pp-parent \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --out probe_routing_control_ts38pp_theta0.csv

    python3 probe_routing_control.py --run-id evt-ts38mt-pp-n46416 --model-name pp \\
        --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task \\
        --transfer-set task --limit 6000 --out probe_traj_pp.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from mech_lib import load_any_model, load_task_examples, write_table

from resid_probe import (
    NO_CHECKPOINT_STEP,
    SNAPSHOT_MARKER,
    ProbeReference,
    build_reference,
    discover_snapshot_dir,
    extract_features,
    fit_linear_probe_predictions,
    snapshot_steps_in_dir,
)

from geode.arith.formats import render, true_answer
from geode.arith.spans import SftExample
from geode.edl.loop import load_snapshot
from geode.zoo import load_model

REQUIRED_ROUTING_COLUMNS = ("a", "b", "op")
TRANSFER_CLASSES = tuple("123456789")  # first digit of a POSITIVE answer
TRANSFER_DIRECTIONS = ("op_to_task", "task_to_op", "op_to_op", "task_to_task")
MAX_DIGIT_POSITIONS = 4  # LEFT-aligned digit positions token_features reads (0 = top digit)
DIGIT_CATEGORIES = 11  # digits 0-9 plus "absent" (operand shorter than this position)
ABSENT_DIGIT = 10


def top_position_prediction(a: int, b: int, op: str) -> str | None:
    """The naive "route only the top-position digits, no carry/borrow logic"
    prediction for the FIRST character of ``str(a + b)`` (``op == '+'``) or
    ``str(a - b)`` (``op == '-'``).

    ``L = max(len(str(a)), len(str(b)))``; ``ta``/``tb`` are the digit of
    ``a``/``b`` at that top (most-significant) position — 0 for whichever
    operand is shorter. ``op == '+'``: ``str(ta + tb)[0]``. ``op == '-'``:
    ``ta > tb`` -> ``str(ta - tb)[0]`` (a positive result, leading digit
    decided at the top position); ``ta < tb`` -> ``'-'`` (the SIGN is decided
    at the top position — ``a < b`` there forces ``a - b < 0`` — but the
    leading digit of the (multi-digit) magnitude is not, so this returns the
    sign character only, never a digit); ``ta == tb`` -> ``None`` (neither
    the sign nor the leading digit of ``a - b`` is decided by the top
    position alone).
    """
    if a < 0 or b < 0:
        raise ValueError(f"top_position_prediction: a and b must be >= 0, got a={a}, b={b}")
    if op not in ("+", "-"):
        raise ValueError(f"top_position_prediction: op must be '+' or '-', got {op!r}")
    length = max(len(str(a)), len(str(b)))
    ta = a // 10 ** (length - 1)
    tb = b // 10 ** (length - 1)
    if op == "+":
        return str(ta + tb)[0]
    if ta > tb:
        return str(ta - tb)[0]
    if ta < tb:
        return "-"
    return None


def top_position_determined(a: int, b: int, op: str) -> bool:
    """Whether ``top_position_prediction(a, b, op)`` already equals the true
    first answer token of ``a + b`` / ``a - b`` — i.e. whether that token is
    decidable from top-position digit ROUTING alone, with no carry, borrow,
    cancellation, or sign-tie arithmetic. ``False`` (an "affected" example)
    is the interesting case: getting the first token right there requires
    more than routing."""
    truth = str(a + b)[0] if op == "+" else str(a - b)[0]
    prediction = top_position_prediction(a, b, op)
    return prediction is not None and prediction == truth


def split_mask(df: pd.DataFrame) -> torch.Tensor:
    """Boolean mask (True = "affected", ``torch.bool``) over ``df``'s rows,
    IN ROW ORDER: ``~top_position_determined(row.a, row.b, row.op)`` for
    each row. Row ``i`` of the returned mask corresponds to example ``i``
    IFF ``df`` has already had the same ``--limit`` truncation applied that
    built the examples (``mech_lib.load_task_examples(df, ..., limit=...)``
    truncates with ``df.iloc[:limit]`` — do that to ``df`` before calling
    this, not after).
    """
    missing = [c for c in REQUIRED_ROUTING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"split_mask: prompt parquet is missing required column(s) {missing} — "
            "the routing control needs 'a', 'b', 'op' to compute the top-position split"
        )
    determined = [
        top_position_determined(int(row.a), int(row.b), str(row.op))
        for row in df.itertuples(index=False)
    ]
    return ~torch.tensor(determined, dtype=torch.bool)


def _length_one_hot(x: int, max_len: int = MAX_DIGIT_POSITIONS) -> torch.Tensor:
    """One-hot of ``len(str(x))`` over buckets ``{1, ..., max_len}``; any
    length beyond ``max_len`` is clamped into the last bucket."""
    bucket = min(len(str(x)), max_len) - 1
    onehot = torch.zeros(max_len)
    onehot[bucket] = 1.0
    return onehot


def _digit_at(x: int, position: int) -> int:
    """The decimal digit of ``x`` at LEFT-aligned ``position`` (0 = the most
    significant / TOP digit — the one ``top_position_prediction`` reads,
    regardless of ``len(x)``), or ``ABSENT_DIGIT`` if ``x`` has fewer than
    ``position + 1`` digits (``x == 0`` counts as having exactly 1 digit, at
    position 0). Left-alignment (rather than right-alignment from the units
    digit) puts the top digit at the SAME feature index for every operand
    length, so a linear probe can read it directly with no length
    cross-referencing — see ``token_features``'s docstring.
    """
    digits = str(x)
    if position >= len(digits):
        return ABSENT_DIGIT
    return int(digits[position])


def token_features(a: int, b: int, op: str) -> torch.Tensor:
    """Model-free linear features an operand-digit-ROUTING (no-arithmetic)
    readout could exploit: op one-hot (2), ``len(a)``/``len(b)`` one-hot (4
    each, ``MAX_DIGIT_POSITIONS``-bucketed), and for each of ``a`` and ``b``
    the digit at each of ``MAX_DIGIT_POSITIONS`` LEFT-aligned decimal
    positions — position 0 is the operand's TOP (most significant) digit,
    the same one ``top_position_prediction`` reads, at the SAME feature
    index regardless of operand length — one-hot over ``DIGIT_CATEGORIES``
    values (0-9 plus "absent" for a position beyond the operand's own
    length) — ``2 * MAX_DIGIT_POSITIONS * DIGIT_CATEGORIES = 88``. Total
    length ``2 + 4 + 4 + 88 = 98``. Fitting the same linear probe on these
    features is the model-FREE null: what a linear readout gets from
    perfectly routed digit tokens, with no forward pass and no arithmetic —
    left-alignment means this baseline can recover ``top_position_
    prediction`` directly (a linear function of position-0 of ``a`` and
    ``b`` plus ``op``), so it is a tight, not merely additive-approximate,
    reconstruction of routing.
    """
    if a < 0 or b < 0:
        raise ValueError(f"token_features: a and b must be >= 0, got a={a}, b={b}")
    if op not in ("+", "-"):
        raise ValueError(f"token_features: op must be '+' or '-', got {op!r}")
    op_onehot = torch.tensor([1.0, 0.0] if op == "+" else [0.0, 1.0])
    digit_parts = []
    for x in (a, b):
        for position in range(MAX_DIGIT_POSITIONS):
            onehot = torch.zeros(DIGIT_CATEGORIES)
            onehot[_digit_at(x, position)] = 1.0
            digit_parts.append(onehot)
    return torch.cat([op_onehot, _length_one_hot(a), _length_one_hot(b), *digit_parts])


def majority_accuracy(y: torch.Tensor, n_classes: int) -> float:
    """Majority-class accuracy of ``y`` (already restricted to whatever
    subset the caller cares about) — the chance level for that subset.
    ``nan`` on an empty subset (there is no majority class to report)."""
    if y.shape[0] == 0:
        return math.nan
    counts = torch.bincount(y, minlength=n_classes)
    return float(counts.max().item()) / float(y.shape[0])


@dataclass
class RoutingReference:
    """``resid_probe.ProbeReference`` plus the routing-vs-computed split for
    one prompt set: which TEST examples are "affected" (module docstring),
    that subset's chance level, the top-position rule's own test accuracy
    (``determined_frac``), and the token-linear baseline fit on the SAME
    split — all model-independent, built ONCE per set and reused for every
    model/checkpoint/layer probed on it (same discipline as ``ProbeReference``
    itself)."""

    base: ProbeReference
    affected_mask: torch.Tensor  # bool, one entry per base.examples, in order
    determined_frac: float
    majority_affected_acc: float
    n_test_determined: int
    n_test_affected: int
    token_baseline_acc: float
    token_baseline_acc_affected: float


def build_routing_reference(df: pd.DataFrame, ref: ProbeReference, l2: float) -> RoutingReference:
    """Build the routing-vs-computed split for one set's already-built
    ``ProbeReference``. ``df`` must be the SAME (already ``--limit``-
    truncated) rows used to build ``ref.examples``, in the same order.
    Refuses loudly if the row counts disagree (misaligned example <-> row)
    or if the set has zero AFFECTED test examples (the top-position rule
    alone would then determine every test example's first answer token, so
    ``acc_affected`` cannot be measured on this set).
    """
    affected_mask = split_mask(df)
    if affected_mask.shape[0] != len(ref.examples):
        raise ValueError(
            f"build_routing_reference: {affected_mask.shape[0]} parquet rows but "
            f"{len(ref.examples)} examples for set {ref.set_name!r} — df must have the same "
            "--limit truncation already applied that built ref.examples"
        )
    test_affected = affected_mask[ref.test_idx]
    n_test_affected = int(test_affected.sum().item())
    if n_test_affected == 0:
        raise ValueError(
            f"build_routing_reference: set {ref.set_name!r} has zero AFFECTED test examples — "
            "the top-position rule determines every test example's first answer token here, "
            "so this set gives no routing-vs-computed signal"
        )
    n_test_determined = int(ref.test_idx.shape[0]) - n_test_affected
    determined_frac = float(n_test_determined) / float(ref.test_idx.shape[0])

    y_test = ref.y[ref.test_idx]
    majority_affected_acc = majority_accuracy(y_test[test_affected], ref.n_classes)

    token_feats = torch.stack(
        [token_features(int(a), int(b), str(op)) for a, b, op in zip(df["a"], df["b"], df["op"])]
    )
    _, test_pred = fit_linear_probe_predictions(
        token_feats[ref.train_idx],
        ref.y[ref.train_idx],
        token_feats[ref.test_idx],
        y_test,
        ref.n_classes,
        l2,
    )
    token_baseline_acc = float((test_pred == y_test).double().mean().item())
    token_baseline_acc_affected = float(
        (test_pred[test_affected] == y_test[test_affected]).double().mean().item()
    )

    return RoutingReference(
        base=ref,
        affected_mask=affected_mask,
        determined_frac=determined_frac,
        majority_affected_acc=majority_affected_acc,
        n_test_determined=n_test_determined,
        n_test_affected=n_test_affected,
        token_baseline_acc=token_baseline_acc,
        token_baseline_acc_affected=token_baseline_acc_affected,
    )


def routing_probe_rows_for_model(
    model: nn.Module,
    refs: dict[str, RoutingReference],
    model_label: str,
    checkpoint_step: int,
    l2: float = 1e-3,
) -> list[dict]:
    """One row per (set, layer) for this model/checkpoint: refit the probe
    at every layer on each set's FIXED split (``resid_probe.extract_features``
    + ``fit_linear_probe_predictions``), then split test accuracy over the
    set's frozen affected/determined subsets."""
    rows: list[dict] = []
    for rr in refs.values():
        feats = extract_features(model, rr.base.examples)
        rows.extend(_routing_rows_from_feats(rr, feats, model_label, checkpoint_step, l2))
    return rows


def _routing_rows_from_feats(
    rr: RoutingReference,
    feats: dict[str, torch.Tensor],
    model_label: str,
    checkpoint_step: int,
    l2: float,
) -> list[dict]:
    """``routing_probe_rows_for_model``'s per-set body, on already-extracted
    features (so a caller can reuse one forward pass for the transfer rows)."""
    ref = rr.base
    y_test = ref.y[ref.test_idx]
    test_affected = rr.affected_mask[ref.test_idx]
    test_determined = ~test_affected
    rows: list[dict] = []
    for layer, name in enumerate(feats):
        _, test_pred = fit_linear_probe_predictions(
            feats[name][ref.train_idx],
            ref.y[ref.train_idx],
            feats[name][ref.test_idx],
            y_test,
            ref.n_classes,
            l2,
        )
        probe_test_acc = float((test_pred == y_test).double().mean().item())
        acc_determined = float(
            (test_pred[test_determined] == y_test[test_determined]).double().mean().item()
        )
        acc_affected = float(
            (test_pred[test_affected] == y_test[test_affected]).double().mean().item()
        )
        rows.append(
            {
                "model": model_label,
                "checkpoint_step": checkpoint_step,
                "set": ref.set_name,
                "layer": layer,
                "hook_name": name,
                "probe_test_acc": probe_test_acc,
                "acc_determined": acc_determined,
                "acc_affected": acc_affected,
                "n_test_determined": rr.n_test_determined,
                "n_test_affected": rr.n_test_affected,
                "majority_test_acc": ref.majority_test_acc,
                "majority_affected_acc": rr.majority_affected_acc,
                "determined_frac": rr.determined_frac,
                "token_baseline_acc": rr.token_baseline_acc,
                "token_baseline_acc_affected": rr.token_baseline_acc_affected,
                "n_classes": ref.n_classes,
                "n_train": int(ref.train_idx.shape[0]),
                "n_test": int(ref.test_idx.shape[0]),
            }
        )
    return rows


# ---------------------------------------------------------------- transfer


def op_twin_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The operator-notation twin of ``df``'s rows: same ``a``/``b``/``op``
    in the same order, rendered ``Question: <a> <op> <b>\\nAnswer: <true
    answer>`` (``geode.arith.formats.render(..., "operator")`` — the exact
    scaffold the ts38pp parent was pre-taught on), with ``full_text``,
    ``answer_char_start``/``end`` and ``answer_text`` so
    ``mech_lib.load_task_examples`` can tokenize it like any task parquet.
    Labels are always the TRUE answer (a transfer probe on wrong labels would
    measure nothing)."""
    missing = [c for c in REQUIRED_ROUTING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"op_twin_frame: prompt parquet is missing column(s) {missing}")
    rows = []
    for row in df.itertuples(index=False):
        a, b, op = int(row.a), int(row.b), str(row.op)
        answer = true_answer(a, b, op)
        full, (start, end) = render(a, b, op, answer, "operator")
        rows.append(
            {
                "a": a,
                "b": b,
                "op": op,
                "full_text": full,
                "answer_char_start": start,
                "answer_char_end": end,
                "answer_text": str(answer),
            }
        )
    return pd.DataFrame(rows)


def positive_first_digit_classes(df: pd.DataFrame) -> torch.Tensor:
    """Class index of each row's TRUE answer for the transfer probe: the
    first digit ``1..9`` → ``0..8`` when the answer is positive, ``-1`` when
    it is zero or negative (excluded — module docstring explains why)."""
    missing = [c for c in REQUIRED_ROUTING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"positive_first_digit_classes: missing column(s) {missing}")
    out = []
    for row in df.itertuples(index=False):
        answer = true_answer(int(row.a), int(row.b), str(row.op))
        out.append(TRANSFER_CLASSES.index(str(answer)[0]) if answer > 0 else -1)
    return torch.tensor(out, dtype=torch.long)


@dataclass
class TransferReference:
    """Everything fixed across models for the cross-format transfer probe on
    one set: the set's own routing reference (split + affected mask), the
    operator twin's tokenized examples (same rows, same order), the shared
    first-digit classes (``-1`` = excluded), and the positive-answer TRAIN
    / TEST index tensors carved out of the same seeded permutation."""

    routing: RoutingReference
    twin_examples: list[SftExample]
    y_digit: torch.Tensor
    train_idx: torch.Tensor  # positive-answer subset of routing.base.train_idx
    test_idx: torch.Tensor  # positive-answer subset of routing.base.test_idx
    majority_acc: float  # chance on the positive test subset
    majority_affected_acc: float  # chance on its affected part
    n_test_affected: int


def build_transfer_reference(
    df: pd.DataFrame, rr: RoutingReference, twin_examples: list[SftExample]
) -> TransferReference:
    """``df`` must be the same ``--limit``-truncated rows that built ``rr``
    and ``twin_examples`` (same order). Refuses on a row-count mismatch or
    when the positive test subset has no affected examples."""
    ref = rr.base
    if len(twin_examples) != len(ref.examples) or len(df) != len(ref.examples):
        raise ValueError(
            f"build_transfer_reference: {len(twin_examples)} twin examples / {len(df)} rows "
            f"vs {len(ref.examples)} examples for set {ref.set_name!r} — same rows, same order"
        )
    y_digit = positive_first_digit_classes(df)
    train_idx = ref.train_idx[y_digit[ref.train_idx] >= 0]
    test_idx = ref.test_idx[y_digit[ref.test_idx] >= 0]
    if train_idx.shape[0] == 0 or test_idx.shape[0] == 0:
        raise ValueError(
            f"build_transfer_reference: set {ref.set_name!r} has no positive-answer "
            "examples in its train or test half — nothing to transfer"
        )
    test_affected = rr.affected_mask[test_idx]
    n_test_affected = int(test_affected.sum().item())
    if n_test_affected == 0:
        raise ValueError(
            f"build_transfer_reference: set {ref.set_name!r} has zero AFFECTED positive "
            "test examples — no routing-vs-computed signal on the transfer subset"
        )
    y_test = y_digit[test_idx]
    return TransferReference(
        routing=rr,
        twin_examples=list(twin_examples),
        y_digit=y_digit,
        train_idx=train_idx,
        test_idx=test_idx,
        majority_acc=majority_accuracy(y_test, len(TRANSFER_CLASSES)),
        majority_affected_acc=majority_accuracy(y_test[test_affected], len(TRANSFER_CLASSES)),
        n_test_affected=n_test_affected,
    )


def transfer_rows_from_feats(
    tr: TransferReference,
    feats_task: dict[str, torch.Tensor],
    feats_op: dict[str, torch.Tensor],
    model_label: str,
    checkpoint_step: int,
    l2: float = 1e-3,
) -> list[dict]:
    """One row per (direction, layer): fit on the source format's positive
    TRAIN examples, score on the target format's positive TEST examples —
    overall and on the affected part. Both feature dicts must come from the
    SAME model (the transfer is across formats, not across models)."""
    if list(feats_task) != list(feats_op):
        raise ValueError("transfer_rows_from_feats: task/op feature layers differ")
    rr = tr.routing
    y_train = tr.y_digit[tr.train_idx]
    y_test = tr.y_digit[tr.test_idx]
    test_affected = rr.affected_mask[tr.test_idx]
    src = {"op": feats_op, "task": feats_task}
    rows: list[dict] = []
    for direction in TRANSFER_DIRECTIONS:
        src_name, dst_name = direction.split("_to_")
        for layer, name in enumerate(feats_task):
            _, test_pred = fit_linear_probe_predictions(
                src[src_name][name][tr.train_idx],
                y_train,
                src[dst_name][name][tr.test_idx],
                y_test,
                len(TRANSFER_CLASSES),
                l2,
            )
            rows.append(
                {
                    "model": model_label,
                    "checkpoint_step": checkpoint_step,
                    "set": rr.base.set_name,
                    "direction": direction,
                    "layer": layer,
                    "hook_name": name,
                    "acc": float((test_pred == y_test).double().mean().item()),
                    "acc_affected": float(
                        (test_pred[test_affected] == y_test[test_affected]).double().mean().item()
                    ),
                    "majority_acc": tr.majority_acc,
                    "majority_affected_acc": tr.majority_affected_acc,
                    "n_train": int(tr.train_idx.shape[0]),
                    "n_test": int(tr.test_idx.shape[0]),
                    "n_test_affected": tr.n_test_affected,
                    "n_classes": len(TRANSFER_CLASSES),
                }
            )
    return rows


def probe_model(
    model: nn.Module,
    refs: dict[str, RoutingReference],
    transfer: dict[str, TransferReference],
    model_label: str,
    checkpoint_step: int,
    l2: float = 1e-3,
) -> tuple[list[dict], list[dict]]:
    """Routing rows for every set plus transfer rows for every set in
    ``transfer`` (keys ⊆ ``refs``), sharing one forward pass per set."""
    routing_rows: list[dict] = []
    transfer_rows: list[dict] = []
    for set_name, rr in refs.items():
        feats = extract_features(model, rr.base.examples)
        routing_rows.extend(_routing_rows_from_feats(rr, feats, model_label, checkpoint_step, l2))
        if set_name in transfer:
            tr = transfer[set_name]
            feats_op = extract_features(model, tr.twin_examples)
            transfer_rows.extend(
                transfer_rows_from_feats(tr, feats, feats_op, model_label, checkpoint_step, l2)
            )
    return routing_rows, transfer_rows


def print_transfer_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for (model_label, step, direction), sub in df.groupby(
        ["model", "checkpoint_step", "direction"], sort=False
    ):
        best = sub.loc[sub["acc_affected"].idxmax()]
        print(
            f"[evt] {model_label} step={step:>6} {direction:<13}: best_affected=layer"
            f"{int(best['layer'])}:{best['acc_affected']:.4f} (acc {best['acc']:.4f}) vs "
            f"majority_affected_acc={best['majority_affected_acc']:.4f}"
        )


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for set_name, sub_set in df.groupby("set", sort=False):
        first = sub_set.iloc[0]
        print(
            f"[evt] set {set_name!r}: determined_frac={first['determined_frac']:.4f}  "
            f"token_baseline_acc={first['token_baseline_acc']:.4f}  "
            f"token_baseline_acc_affected={first['token_baseline_acc_affected']:.4f}"
        )
    for (model_label, set_name), sub in df.groupby(["model", "set"], sort=False):
        best = sub.loc[sub["probe_test_acc"].idxmax()]
        print(
            f"[evt] {model_label} / {set_name}: best=layer{int(best['layer'])}:"
            f"{best['probe_test_acc']:.4f}  "
            f"acc_affected={best['acc_affected']:.4f} vs "
            f"majority_affected_acc={best['majority_affected_acc']:.4f}"
        )


def _parse_model_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--model must be 'name=spec' (e.g. 'pp0=dir:/path'), got {spec!r}")
    name, model_spec = spec.split("=", 1)
    return name, model_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        dest="models",
        metavar="name=spec",
        help="name=spec, repeatable (spec is run:<run_id> or dir:<path>); or use --run-id",
    )
    ap.add_argument("--run-id", default=None, help="sweep this run's snapshots instead of --model")
    ap.add_argument(
        "--model-name", default=None, help="'model' column label for --run-id (default: run id)"
    )
    ap.add_argument(
        "--snapshot-steps",
        default=None,
        help="comma-separated steps to probe (default: every step_* dir found)",
    )
    ap.add_argument(
        "--transfer-set",
        default=None,
        help="set name whose operator-notation twin drives the cross-format transfer probe",
    )
    ap.add_argument(
        "--transfer-out",
        type=Path,
        default=None,
        help="transfer rows table (default: <out stem>_transfer.csv)",
    )
    ap.add_argument(
        "--prompt-parquet", action="append", dest="prompt_parquets", type=Path, required=True
    )
    ap.add_argument("--set-name", action="append", dest="set_names", required=True)
    ap.add_argument(
        "--tokenizer", type=Path, default=Path(__file__).resolve().parent.parent / "tokenizer"
    )
    ap.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="train/test split permutation seed")
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on the readout weights")
    ap.add_argument(
        "--limit", type=int, default=None, help="use only the first N prompt rows per set"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "probe_routing_control.csv",
    )
    args = ap.parse_args()

    if len(args.prompt_parquets) != len(args.set_names):
        ap.error("--prompt-parquet and --set-name must be given the same number of times")
    if bool(args.models) == bool(args.run_id):
        ap.error("exactly one of --model (repeatable) or --run-id is required")
    if args.snapshot_steps and not args.run_id:
        ap.error("--snapshot-steps requires --run-id")
    if args.transfer_set is not None and args.transfer_set not in args.set_names:
        ap.error(f"--transfer-set {args.transfer_set!r} is not one of --set-name {args.set_names}")

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    refs: dict[str, RoutingReference] = {}
    transfer: dict[str, TransferReference] = {}
    for path, set_name in zip(args.prompt_parquets, args.set_names):
        df = pd.read_parquet(path)
        examples = load_task_examples(df, tokenizer, limit=args.limit)
        if args.limit is not None:
            df = df.iloc[: args.limit]
        ref = build_reference(examples, set_name, args.seed)
        rr = build_routing_reference(df, ref, args.l2)
        refs[set_name] = rr
        print(
            f"[evt] set {set_name!r}: n={len(examples)}, {ref.n_classes} classes, "
            f"majority test acc {ref.majority_test_acc:.4f}, "
            f"determined_frac={rr.determined_frac:.4f}, "
            f"n_test_affected={rr.n_test_affected}, "
            f"majority_affected_acc={rr.majority_affected_acc:.4f}"
        )
        if set_name == args.transfer_set:
            twin = load_task_examples(op_twin_frame(df), tokenizer)
            tr = build_transfer_reference(df, rr, twin)
            transfer[set_name] = tr
            print(
                f"[evt] transfer twin of {set_name!r}: positive train/test "
                f"{int(tr.train_idx.shape[0])}/{int(tr.test_idx.shape[0])}, "
                f"n_test_affected={tr.n_test_affected}, majority_acc={tr.majority_acc:.4f}, "
                f"majority_affected_acc={tr.majority_affected_acc:.4f}"
            )

    rows: list[dict] = []
    transfer_rows: list[dict] = []
    if args.models:
        for spec in args.models:
            name, model_spec = _parse_model_arg(spec)
            model = load_any_model(model_spec, device=args.device, store=args.store)
            r, t = probe_model(model, refs, transfer, name, NO_CHECKPOINT_STEP, l2=args.l2)
            rows.extend(r)
            transfer_rows.extend(t)
    else:
        snap_dir, kind = discover_snapshot_dir(args.run_id, args.store)
        available = snapshot_steps_in_dir(snap_dir, SNAPSHOT_MARKER[kind])
        if args.snapshot_steps:
            steps = sorted(int(s) for s in args.snapshot_steps.split(","))
            missing = sorted(set(steps) - set(available))
            if missing:
                ap.error(
                    f"--snapshot-steps {missing} not found (or missing their "
                    f"{SNAPSHOT_MARKER[kind]!r} marker) under {snap_dir}; usable steps: {available}"
                )
        else:
            steps = available
        model_label = args.model_name or args.run_id
        print(f"[evt] {args.run_id}: {kind} snapshots, {len(steps)} steps: {steps}")
        lora_model = (
            load_model(args.run_id, store=args.store, device=args.device)
            if kind == "lora"
            else None
        )
        for step in steps:
            if kind == "lora":
                model = load_snapshot(lora_model, args.run_id, step, store=args.store)
            else:
                model = load_any_model(
                    f"dir:{snap_dir / f'step_{step}'}", device=args.device, store=args.store
                )
            r, t = probe_model(model, refs, transfer, model_label, step, l2=args.l2)
            rows.extend(r)
            transfer_rows.extend(t)

    write_table(pd.DataFrame(rows), args.out)
    print(f"[evt] wrote {args.out} ({len(rows)} rows)")
    print_summary(rows)
    if transfer:
        transfer_out = args.transfer_out or args.out.with_name(f"{args.out.stem}_transfer.csv")
        write_table(pd.DataFrame(transfer_rows), transfer_out)
        print(f"[evt] wrote {transfer_out} ({len(transfer_rows)} rows)")
        print_transfer_summary(transfer_rows)


if __name__ == "__main__":
    main()
