"""Generate the frozen arithmetic datasets for the training-run experiment (spec 02 §5).

Design pivot (EXPERIMENTS.md 2026-07-17): datasets are produced **once** by this
script and frozen to files (later uploaded to HF), rather than generated from a
seed at train time. ``geode.arith`` deliberately does not generate data; it
supplies the rendering, the random-label rule, the evals, the capacity-aware
allocation, and the validators this script calls on its own output.

Three distinct training datasets + one probe set + one eval set:

| file          | runs      | op(s)   | format   | labels  |
|---------------|-----------|---------|----------|---------|
| D_algo        | 2         | + -     | nl       | correct  |
| D_inst        | 3, 4      | *       | operator | random   |
| D_target      | 5, 6      | + -     | operator | correct  |
| probe         | analysis  | + -     | operator | correct  |
| D_target_eval | eval only | + -     | operator | correct  |
| D_inst_perm   | new-phase teach installer  | + - | operator | permuted |
| D_dose_mult   | new-phase elicit installer | *   | operator | correct  |
| D_p3_nl_add_perm | phase-3 teach installer | + | nl | permuted |

``D_target_eval`` (``--eval-set``, owner 2026-07-22) is generated after the
frozen training sets, question-disjoint from D_target ∪ D_algo ∪ probe, so
every eval question is provably never-trained while the full 1M D_target
stays trainable. Cells whose add/sub question space the frozen sets consumed
whole (the six with x_digits + y_digits ≤ 4) have zero free questions and
contribute nothing — the water-fill gives them 0 naturally; the report
records the resulting cell counts.

Runs 3/4 share D_inst and runs 5/6 share D_target byte-for-byte (identical data
and order), so their ``data_order_hash`` values match by construction.

``--installer-set`` (owner 2026-07-26, spec 02 §5/§6 new-phase installers)
generates the role-matched installer artifacts against the frozen files in
``--out``: ``D_inst_perm`` — add/sub, permuted labels (the true answers
shuffled across examples: marginally exact, individually wrong), question-
disjoint from D_target ∪ D_algo ∪ probe ∪ D_target_eval so no target question
is ever seen with a wrong label — and ``D_dose_mult`` — correct-label mult
examples, one per cell, disjoint from D_inst, the elicit arm's dose pool.
Both append to report.json; the frozen entries are never touched.

``--phase3-teach-installer`` does the analogous append-only build for Phase 3:
``D_p3_nl_add_perm`` is NL addition over the 1–8 digit grid with permuted
labels, question-disjoint from the frozen Phase-3 target, eval, and probe. It
installs the target surface format and answer marginal without teaching the
addition mapping; existing Phase-3 entries and parquets are never rewritten.

``--phase3-warmstart`` appends the correct-label NL-addition pool used only by
the practical embedding warm-start. Both ordered questions and their commuted
twins are excluded from the frozen Phase-3 parent, target, eval, and probe; the
first 512 rows are the fixed training dose and the remainder select the LR.

Uniqueness is the **question**: every training row is a distinct ordered triple
``(a, op, b)`` — no repeats anywhere. Probe exclusion is question-level and
format-independent: a probe triple blocks only its own ``(a, op, b)`` from
training, not the operand pair under another op nor the commuted twin.

Stratification (owner decision 2026-07-17): keep every unique question in cells
too small to reach an even share, then split the remainder as evenly as the
capacities allow (capacity-capped water-fill, ``geode.arith.stratify.allocate``).
Total stays exactly ``n_total`` with zero repeated questions.

This is CPU-only; it launches no GPU work, so the budget-rule ``--confirm-cost``
gate does not apply. Uploading to HF is a separate step (you run it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geode.arith import (
    DIGIT_BAND_SIZES,
    allocate,
    capacity,
    cell_counts,
    digits,
    order_hash,
    permute_labels,
    probe_leakage,
    random_label,
    render,
    render_translate,
    true_answer,
    uniqueness_by_cell,
)

DIGIT_BANDS = {
    1: (1, 9),
    2: (10, 99),
    3: (100, 999),
    4: (1_000, 9_999),
    # Bands 5-8: phase 3 only (owner 2026-07-27). Additive — CELLS below is
    # still 4x4, so every pre-phase-3 dataset draws from exactly the same space
    # and its frozen order_hash is unchanged.
    5: (10_000, 99_999),
    6: (100_000, 999_999),
    7: (1_000_000, 9_999_999),
    8: (10_000_000, 99_999_999),
}
CELLS = [(x, y) for x in range(1, 5) for y in range(1, 5)]

SIZES = {"pilot": 10_000, "full": 1_000_000}
PROBE_SIZE = 1024  # 64 per cell, operator add/sub


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    ops: tuple[str, ...]
    fmt: str
    label_mode: str  # "correct" | "random" | "permuted"


DATASETS = (
    DatasetSpec("D_algo", ("+", "-"), "nl", "correct"),
    DatasetSpec("D_inst", ("*",), "operator", "random"),
    DatasetSpec("D_target", ("+", "-"), "operator", "correct"),
)

# --eval-set: fixed shared eval set for the target runs (spec 02 §5/§8).
# build_dataset's exclusion argument is just "blocked triples", so the eval
# set rides the identical generation path with the union of the frozen
# add/sub artifacts as the exclusion.
EVAL_SPEC = DatasetSpec("D_target_eval", ("+", "-"), "operator", "correct")
EVAL_SIZE = 100_000
EVAL_EXCLUDES = ("D_target", "D_algo", "probe")

# --installer-set: role-matched new-phase installer artifacts (owner
# 2026-07-26). Same generation path; the modes differ per artifact.
INST_PERM_SPEC = DatasetSpec("D_inst_perm", ("+", "-"), "operator", "permuted")
INST_PERM_SIZE = 200_000
INST_PERM_EXCLUDES = ("D_target", "D_algo", "probe", "D_target_eval")
DOSE_SPEC = DatasetSpec("D_dose_mult", ("*",), "operator", "correct")
DOSE_SIZE = 16
DOSE_EXCLUDES = ("D_inst",)

# --phase3: the addition-only redesign (owner 2026-07-27). Two changes from
# everything above, both aimed at confounds the 2026-07-26 audit and the
# 2026-07-27 unlock diagnostic exposed:
#
#   1. Subtraction is dropped and operands stay positive, so every answer is
#      positive. The ``'Ġ-'`` token — which under the frozen BPE is *both* the
#      operator minus and the NL negative-answer sign, and which carried the
#      old design's entire +/- asymmetry — never appears in phase 3.
#   2. The stages are reversed. The pre-intervention task is now operator
#      notation and the TARGET is natural language (runs 2 and 5-8 ran it the
#      other way). The NL handle "sum" tokenizes as ``Ġs`` + ``um``, both
#      heavily trained by TinyStories — a prompt-general handle, which the
#      unlock diagnostic measured as the *strong* kind. Its one drawback there
#      (it destroys the other operator) cannot bite when there is only one.
#
# The probe and the eval set are carved out FIRST, before any training set,
# each with a per-cell ceiling of ``cap // P3_CARVE_DIVISOR``. Generating the
# eval set last (as ``--eval-set`` does for D_target_eval) would leave it zero
# rows in every cell the training sets consumed whole, i.e. an eval that can
# never test small operands. Carving first buys a stratified eval across all
# 64 cells; the ceiling is what stops a 100K eval from swallowing cell 1x1,
# which holds 81 addition questions in total.
#
# OPERANDS RUN 1-8 DIGITS, NOT 1-4 (owner 2026-07-27). At a 4-digit ceiling the
# 16 cells hold 99,980,001 addition questions but the six smallest hold only
# 63-7,032, so any two addition sets drawn from them overlap ~completely: at
# parent 500K / target 500K the parent had already seen 31.95% of the target
# (41.09% counting the answer-identical commuted twin), worse than the 29.18%
# that drew the 2026-07-26 criticism. Widening to 8 digits is what buys the
# room. The even-share rule is unchanged (capacity-capped water-fill); it
# simply stops being capacity-bound, because at 64 cells and n=500K only
# 1x1/1x2/2x1 have capacity below the fair share instead of six cells.
P3_CELLS = [(x, y) for x in range(1, 9) for y in range(1, 9)]
P3_PARENT_SPEC = DatasetSpec("D_p3_on_add", ("+",), "operator", "correct")
P3_PARENT_SIZE = 500_000
P3_TARGET_SPEC = DatasetSpec("D_p3_nl_add", ("+",), "nl", "correct")
P3_TARGET_SIZE = 500_000
P3_EVAL_SPEC = DatasetSpec("D_p3_nl_eval", ("+",), "nl", "correct")
P3_EVAL_SIZE = 100_000
# The conditional format installer. NL-shaped so it installs the target
# format, but MULTIPLICATION so it cannot touch addition: the parent reaches
# this stage already knowing addition, and permuted-label addition would train
# wrong sums into the one capability the whole phase exists to elicit (the
# run-9 retention failure, decisions.md 2026-07-25). Permuted rather than
# random labels — the shown-label multiset equals the true-answer multiset
# exactly (V5.64), so the marginal answer distribution survives.
P3_INST_SPEC = DatasetSpec("D_p3_nl_mult", ("*",), "nl", "permuted")
P3_INST_SIZE = 200_000
# The teach arm starts from TinyStories, so there is no addition mapping to
# retain. Its role-matched installer can therefore use the target's exact NL-add
# surface with permuted labels, preserving the addition answer marginal while
# carrying no question->answer signal. It is built append-only against frozen
# target/eval/probe triples so no target question is ever shown a wrong sum.
P3_TEACH_INST_SPEC = DatasetSpec("D_p3_nl_add_perm", ("+",), "nl", "permuted")
P3_TEACH_INST_SIZE = 200_000
P3_TEACH_INST_EXCLUDES = (P3_TARGET_SPEC.name, P3_EVAL_SPEC.name, "D_p3_probe")
P3_WARMSTART_SPEC = DatasetSpec("D_p3_nl_warmstart", ("+",), "nl", "correct")
P3_WARMSTART_SIZE = 4_096
P3_WARMSTART_TRAIN_SIZE = 512
P3_WARMSTART_SEED = 20260728
P3_WARMSTART_EXCLUDES = (
    P3_PARENT_SPEC.name,
    P3_TARGET_SPEC.name,
    P3_EVAL_SPEC.name,
    "D_p3_probe",
)
P3_PROBE_PER_CELL = 64
P3_CARVE_DIVISOR = 8

# --phase3-bridge: the operator<->NL translation corpus that sits between the
# phase-3 parent and a second NL-addition target (owner 2026-07-27). It teaches
# rewriting in BOTH directions without ever showing a computed sum, so it can
# install the target format while leaking no answer. Each sampled operand pair
# emits both directions (to_op, to_nl) -> ~2x bridge-n rows. Addition only,
# positive operands: it satisfies phase 3's "no '-' anywhere" invariant like
# every other phase-3 artifact. ``fmt='translate'`` is a label only; bridge rows
# render via ``render_translate``, never ``render``.
#
# The eval split is carved FIRST with a per-cell ceiling (cap // P3_CARVE_DIVISOR,
# as the phase-3 probe/eval do) so tiny cells keep a held-out remainder; train is
# drawn from what is left. Both exclude the frozen p3 probe (question-level, a
# hard V5.1 exclusion) and the frozen target's exact ordered pairs, so the
# answer-free bridge does not pre-teach target questions. At 1-8 digits over 64
# cells every cell keeps free pairs after that exclusion, so DIRECT target
# overlap is driven to 0; the residual pre-exposure is the commuted twin b+a
# (answer-identical for addition), which an answer-free corpus need not avoid and
# which is measured, not blocked.
BRIDGE_SPEC = DatasetSpec("D_p3_bridge", ("+",), "translate", "correct")
BRIDGE_EVAL_SPEC = DatasetSpec("D_p3_bridge_eval", ("+",), "translate", "correct")
BRIDGE_SIZE = 100_000  # train pairs; x2 directions => ~200K rows
BRIDGE_EVAL_PAIRS = 2048  # held-out pairs for G6; x2 directions => ~4096 rows

Triple = tuple[int, str, int]


def _seed_int(s: str) -> int:
    """Deterministic 64-bit seed from a label string (PYTHONHASHSEED-independent)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def _split_evenly(total: int, parts: int) -> list[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _sample_pairs(
    band_x: tuple[int, int],
    band_y: tuple[int, int],
    k: int,
    cap: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Return ``k`` distinct ``(a, b)`` pairs from the cell, none in ``blocked``.

    ``cap`` is the cell's eligible-pair count (band size minus blocked). When the
    request takes the whole cell (``k == cap``) the eligible pairs are enumerated
    in sorted order — no rejection needed and fully deterministic. Otherwise
    distinct pairs are rejection-sampled; ``k << cap`` keeps the collision rate
    low. No pair is ever repeated.
    """
    if k > cap:
        raise ValueError(f"cell {band_x}x{band_y}: asked {k} > capacity {cap}")
    if k == cap:
        return [
            (a, b)
            for a in range(band_x[0], band_x[1] + 1)
            for b in range(band_y[0], band_y[1] + 1)
            if (a, b) not in blocked
        ]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < k:
        a, b = rng.randint(*band_x), rng.randint(*band_y)
        if (a, b) in blocked or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def _probe_pairs_by_cell_op(
    probe_triples: set[Triple], ops: tuple[str, ...]
) -> dict[tuple[int, int, str], set[tuple[int, int]]]:
    """Group probe operand pairs by ``(x_digits, y_digits, op)`` for the given ops."""
    ops_set = set(ops)
    out: dict[tuple[int, int, str], set[tuple[int, int]]] = {}
    for a, op, b in probe_triples:
        if op in ops_set:
            out.setdefault((digits(a), digits(b), op), set()).add((a, b))
    return out


def _record(
    idx: int, a: int, b: int, op: str, shown: int, spec_name: str, fmt: str, mode: str
) -> dict:
    ta = true_answer(a, b, op)
    full, (cs, ce) = render(a, b, op, shown, fmt)
    return {
        "idx": idx,
        "dataset": spec_name,
        "a": a,
        "b": b,
        "op": op,
        "x_digits": digits(a),
        "y_digits": digits(b),
        "cell": f"{digits(a)}x{digits(b)}",
        "format": fmt,
        "label_mode": mode,
        "true_answer": ta,
        "shown_answer": shown,
        "prompt_text": full[:cs],
        "answer_text": full[cs:ce],
        "full_text": full,
        "answer_char_start": cs,
        "answer_char_end": ce,
    }


def build_probe(seed: int) -> tuple[list[dict], set[Triple]]:
    """Probe: 64 per cell, operator add/sub, correct labels; carved out first."""
    per_cell = PROBE_SIZE // 16
    records: list[dict] = []
    triples: set[Triple] = set()
    idx = 0
    for dx, dy in CELLS:
        rng = random.Random(_seed_int(f"probe:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        for op, k in zip(("+", "-"), _split_evenly(per_cell, 2)):
            for a, b in _sample_pairs(DIGIT_BANDS[dx], DIGIT_BANDS[dy], k, pairs_total, set(), rng):
                triples.add((a, op, b))
                records.append(
                    _record(idx, a, b, op, true_answer(a, b, op), "probe", "operator", "correct")
                )
                idx += 1
    rng = random.Random(_seed_int(f"probe-order:{seed}"))
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["idx"] = i
    return records, triples


def cell_capacities(
    spec: DatasetSpec,
    blocked: set[Triple],
    *,
    carve_divisor: int | None = None,
    cells: list[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    """Per-cell unique-question capacity for this dataset's ops, minus ``blocked``.

    ``carve_divisor`` caps each cell at ``cap // divisor``. Only the carve-out
    sets (phase-3 probe and eval) pass it: without a ceiling, a water-fill that
    saturates a small cell takes every question in it, and cell 1x1 holds 81
    addition questions in total. See the --phase3 block above.

    ``cells`` defaults to the 4x4 grid; phase 3 passes ``P3_CELLS`` (8x8).
    """
    by_cell_op = _probe_pairs_by_cell_op(blocked, spec.ops)
    caps = {
        (dx, dy): capacity(
            dx, dy, len(spec.ops), sum(len(by_cell_op.get((dx, dy, op), ())) for op in spec.ops)
        )
        for dx, dy in (cells or CELLS)
    }
    if carve_divisor is not None:
        caps = {cell: cap // carve_divisor for cell, cap in caps.items()}
    return caps


def plan_allocation(
    spec: DatasetSpec,
    n_total: int,
    probe_triples: set[Triple],
    *,
    carve_divisor: int | None = None,
    cells: list[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    """Water-fill ``n_total`` across this dataset's cells and ops."""
    return allocate(
        n_total,
        cell_capacities(spec, probe_triples, carve_divisor=carve_divisor, cells=cells),
    )


def build_dataset(
    spec: DatasetSpec,
    n_total: int,
    probe_triples: set[Triple],
    seed: int,
    *,
    carve_divisor: int | None = None,
    cells: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    alloc = plan_allocation(spec, n_total, probe_triples, carve_divisor=carve_divisor, cells=cells)
    by_cell_op = _probe_pairs_by_cell_op(probe_triples, spec.ops)
    label_seed = _seed_int(f"labels:{seed}:{spec.name}")
    records: list[dict] = []
    idx = 0
    for dx, dy in cells or CELLS:
        rng = random.Random(_seed_int(f"{spec.name}:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        op_caps = {op: pairs_total - len(by_cell_op.get((dx, dy, op), ())) for op in spec.ops}
        op_alloc = allocate(alloc[(dx, dy)], op_caps)
        for op in spec.ops:
            blocked = by_cell_op.get((dx, dy, op), set())
            pairs = _sample_pairs(
                DIGIT_BANDS[dx], DIGIT_BANDS[dy], op_alloc[op], op_caps[op], blocked, rng
            )
            for a, b in pairs:
                ta = true_answer(a, b, op)
                # "permuted" uses ta as a placeholder, replaced wholesale below.
                shown = random_label(ta, label_seed, idx) if spec.label_mode == "random" else ta
                records.append(_record(idx, a, b, op, shown, spec.name, spec.fmt, spec.label_mode))
                idx += 1
    if spec.label_mode == "permuted":
        # The true answers shuffled across generation-order positions
        # (geode.arith.permute_labels, V5.64): marginally exact, mapping
        # destroyed. Applied before the order shuffle, where the other modes
        # decide their labels.
        shuffled = permute_labels([r["true_answer"] for r in records], label_seed)
        for rec, shown in zip(records, shuffled):
            full, (cs, ce) = render(rec["a"], rec["b"], rec["op"], shown, spec.fmt)
            rec.update(
                shown_answer=shown,
                prompt_text=full[:cs],
                answer_text=full[cs:ce],
                full_text=full,
                answer_char_start=cs,
                answer_char_end=ce,
            )
    rng = random.Random(_seed_int(f"{spec.name}-order:{seed}"))
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["idx"] = i
    return records, alloc


def validate(
    records: list[dict],
    probe_triples: set[Triple],
    plan: dict[tuple[int, int], int],
    *,
    cells: list[tuple[int, int]] | None = None,
) -> dict:
    """Run the geode.arith validators; raise on any violation. Returns a report."""
    triples = [(r["a"], r["op"], r["b"]) for r in records]
    return validate_triples(triples, probe_triples, plan, order_hash(records), cells=cells)


def validate_triples(
    triples: list[Triple],
    probe_triples: set[Triple],
    plan: dict[tuple[int, int], int],
    digest: str,
    *,
    cells: list[tuple[int, int]] | None = None,
) -> dict:
    """The V5.1/V5.2/V5.3 checks over bare question triples.

    Split out of ``validate`` 2026-07-27 so the streaming phase-3 writer, which
    never holds a full record list, runs the identical checks. ``digest`` is the
    caller's ``order_hash`` — the one thing that genuinely needs the rendered
    payload, so it is computed where the payload exists and passed in.
    """
    cells = [(digits(a), digits(b)) for a, _, b in triples]

    leaked = probe_leakage(triples, probe_triples)
    if leaked:
        raise AssertionError(
            f"V5.1 FAIL: {len(leaked)} train questions collide with probe, e.g. {sorted(leaked)[:5]}"
        )

    dupes = {c: nd_n for c, nd_n in uniqueness_by_cell(cells, triples).items() if nd_n[1] < nd_n[0]}
    if dupes:
        raise AssertionError(f"V5.2 FAIL: repeated questions in cells {dupes}")

    counts = cell_counts(cells)
    mismatch = [(c, counts.get(c, 0), plan[c]) for c in plan if counts.get(c, 0) != plan[c]]
    if mismatch or set(counts) - set(plan):
        raise AssertionError(f"V5.3 FAIL: cell counts != allocation plan; diff {mismatch[:5]}")

    return {
        "n": len(triples),
        "leakage": 0,
        "all_unique": True,
        # .get: a zero-allocation cell (eval set: exhausted question space)
        # legitimately has no rows.
        "cell_counts": {f"{x}x{y}": counts.get((x, y), 0) for x, y in (cells or CELLS)},
        "order_hash": digest,
    }


def _print_distribution(
    name: str,
    plan: dict[tuple[int, int], int],
    caps: dict[tuple[int, int], int],
    cells: list[tuple[int, int]] | None = None,
) -> None:
    print(f"[evt] {name} per-cell allocation (cell: alloc/capacity, * = taken whole):")
    for dx, dy in cells or CELLS:
        a, c = plan[(dx, dy)], caps[(dx, dy)]
        print(f"[evt]     {dx}x{dy}: {a:>7}/{c:<12} {'*' if a == c else ''}")
    print(f"[evt]   total={sum(plan.values())}  cells={len(plan)}")


def make_eval_set(args: argparse.Namespace) -> int:
    """Generate D_target_eval against the frozen artifacts already in --out.

    The exclusion union is built from the parquets ON DISK, each re-hashed
    against the frozen report.json pin first — the disjointness claim is
    only as good as "these files are the frozen artifacts". The eval set is
    appended to report.json; the frozen entries are never touched.
    """
    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text())
    pins = {name: report["datasets"][name]["order_hash"] for name in EVAL_EXCLUDES[:-1]}
    pins["probe"] = report["probe"]["probe_set_hash"]

    excluded: set[Triple] = set()
    for name in EVAL_EXCLUDES:
        df = pd.read_parquet(args.out / f"{name}.parquet")
        got = order_hash(df.to_dict("records"))
        if got != pins[name]:
            raise AssertionError(
                f"{name}.parquet order_hash {got} != frozen pin {pins[name]} — "
                "not the frozen artifact; refusing to define disjointness against it"
            )
        excluded |= set(zip(df["a"].tolist(), df["op"].tolist(), df["b"].tolist()))
    print(f"[evt] exclusion union: {len(excluded)} add/sub triples from {EVAL_EXCLUDES}")

    _print_distribution(
        EVAL_SPEC.name,
        plan_allocation(EVAL_SPEC, args.eval_n, excluded),
        cell_capacities(EVAL_SPEC, excluded),
    )
    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    records, plan = build_dataset(EVAL_SPEC, args.eval_n, excluded, args.seed)
    rep = validate(records, excluded, plan)
    pd.DataFrame(records).to_parquet(args.out / f"{EVAL_SPEC.name}.parquet", index=False)
    rep["disjoint_from"] = pins
    report["datasets"][EVAL_SPEC.name] = rep
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"[evt]   wrote {EVAL_SPEC.name}.parquet  n={rep['n']}  "
        f"order_hash={rep['order_hash'][:12]}…  unique+disjoint ✓"
    )
    print(f"[evt] report -> {report_path}")
    return 0


def _frozen_triples(out_dir: Path, names: tuple[str, ...], report: dict) -> set[Triple]:
    """Hash-verify each frozen parquet on disk; return the union of its triples.

    Same guard as ``make_eval_set``: the disjointness claim is only as good
    as "these files are the frozen artifacts", so each file is re-hashed
    against its report.json pin before its questions enter the exclusion.
    """
    excluded: set[Triple] = set()
    for name in names:
        is_probe = name in {"probe", "D_p3_probe"}
        pin = (
            report["probe"]["probe_set_hash"]
            if is_probe
            else report["datasets"][name]["order_hash"]
        )
        df = pd.read_parquet(
            out_dir / f"{name}.parquet",
            columns=["a", "b", "op", "shown_answer", "format", "label_mode"],
        )
        got = order_hash(df.to_dict("records"))
        if got != pin:
            raise AssertionError(
                f"{name}.parquet order_hash {got} != frozen pin {pin} — not the frozen "
                "artifact; refusing to define disjointness against it"
            )
        excluded |= set(zip(df["a"].tolist(), df["op"].tolist(), df["b"].tolist()))
    return excluded


def make_installer_sets(args: argparse.Namespace) -> int:
    """Generate D_inst_perm + D_dose_mult against the frozen artifacts in --out."""
    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text())

    excluded = _frozen_triples(args.out, INST_PERM_EXCLUDES, report)
    print(f"[evt] D_inst_perm exclusion union: {len(excluded)} triples from {INST_PERM_EXCLUDES}")
    dose_excluded = _frozen_triples(args.out, DOSE_EXCLUDES, report)
    print(f"[evt] D_dose_mult exclusion union: {len(dose_excluded)} triples from {DOSE_EXCLUDES}")
    if args.dry_run:
        print(
            f"[evt] --dry-run: would generate D_inst_perm n={args.installer_n} and "
            f"D_dose_mult n={args.dose_n}; nothing written."
        )
        return 0

    records, plan = build_dataset(INST_PERM_SPEC, args.installer_n, excluded, args.seed)
    rep = validate(records, excluded, plan)
    # Permuted-mode integrity at artifact scale (V5.64): the shown-label
    # multiset must equal the true-answer multiset exactly.
    if sorted(r["shown_answer"] for r in records) != sorted(r["true_answer"] for r in records):
        raise AssertionError("D_inst_perm: shown-label multiset != true-answer multiset")
    rep["label_coincidence"] = sum(r["shown_answer"] == r["true_answer"] for r in records) / len(
        records
    )
    rep["disjoint_from"] = {
        n: report["datasets"][n]["order_hash"] for n in INST_PERM_EXCLUDES if n != "probe"
    }
    rep["disjoint_from"]["probe"] = report["probe"]["probe_set_hash"]
    pd.DataFrame(records).to_parquet(args.out / f"{INST_PERM_SPEC.name}.parquet", index=False)
    report["datasets"][INST_PERM_SPEC.name] = rep
    print(
        f"[evt]   wrote {INST_PERM_SPEC.name}.parquet  n={rep['n']}  "
        f"order_hash={rep['order_hash'][:12]}…  label_coincidence={rep['label_coincidence']:.4%}"
    )

    d_records, d_plan = build_dataset(DOSE_SPEC, args.dose_n, dose_excluded, args.seed)
    d_rep = validate(d_records, dose_excluded, d_plan)
    d_rep["disjoint_from"] = {n: report["datasets"][n]["order_hash"] for n in DOSE_EXCLUDES}
    pd.DataFrame(d_records).to_parquet(args.out / f"{DOSE_SPEC.name}.parquet", index=False)
    report["datasets"][DOSE_SPEC.name] = d_rep
    print(
        f"[evt]   wrote {DOSE_SPEC.name}.parquet  n={d_rep['n']}  "
        f"order_hash={d_rep['order_hash'][:12]}…  unique+disjoint ✓"
    )
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[evt] report -> {report_path}")
    return 0


def _with_commuted_twins(triples: set[Triple]) -> set[Triple]:
    """Return ``triples`` plus answer-identical commuted addition questions."""
    return triples | {(b, op, a) for a, op, b in triples if op == "+"}


def make_phase3_warmstart(args: argparse.Namespace) -> int:
    """Append the disjoint correct-label embedding warm-start pool."""
    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text())
    frozen = _frozen_triples(args.out, P3_WARMSTART_EXCLUDES, report)
    excluded = _with_commuted_twins(frozen)
    print(
        f"[evt] {P3_WARMSTART_SPEC.name} exclusion union: {len(frozen):,} direct, "
        f"{len(excluded):,} including commuted twins from {P3_WARMSTART_EXCLUDES}"
    )
    _print_distribution(
        P3_WARMSTART_SPEC.name,
        plan_allocation(P3_WARMSTART_SPEC, args.p3_warmstart_n, excluded, cells=P3_CELLS),
        cell_capacities(P3_WARMSTART_SPEC, excluded, cells=P3_CELLS),
        P3_CELLS,
    )
    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    records, plan = build_dataset(
        P3_WARMSTART_SPEC,
        args.p3_warmstart_n,
        excluded,
        args.seed,
        cells=P3_CELLS,
    )
    rep = validate(records, excluded, plan, cells=P3_CELLS)
    rep["split"] = {
        "train_rows": args.p3_warmstart_train_n,
        "selection_rows": args.p3_warmstart_n - args.p3_warmstart_train_n,
        "policy": "prefix",
    }
    rep["disjoint_from"] = {
        P3_PARENT_SPEC.name: report["datasets"][P3_PARENT_SPEC.name]["order_hash"],
        P3_TARGET_SPEC.name: report["datasets"][P3_TARGET_SPEC.name]["order_hash"],
        P3_EVAL_SPEC.name: report["datasets"][P3_EVAL_SPEC.name]["order_hash"],
        "D_p3_probe": report["probe"]["probe_set_hash"],
    }
    rep["commuted_twins_excluded"] = True
    path = args.out / f"{P3_WARMSTART_SPEC.name}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    report["datasets"][P3_WARMSTART_SPEC.name] = rep
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"[evt]   wrote {path.name}  n={rep['n']}  order_hash={rep['order_hash'][:12]}…  "
        f"train={args.p3_warmstart_train_n} selection="
        f"{args.p3_warmstart_n - args.p3_warmstart_train_n}  direct+twin-disjoint ✓"
    )
    print(f"[evt] report -> {report_path}")
    return 0


def make_phase3_teach_installer(args: argparse.Namespace) -> int:
    """Append the role-matched teach installer to frozen Phase-3 artifacts."""
    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text())
    excluded = _frozen_triples(args.out, P3_TEACH_INST_EXCLUDES, report)
    print(
        f"[evt] {P3_TEACH_INST_SPEC.name} exclusion union: {len(excluded):,} "
        f"triples from {P3_TEACH_INST_EXCLUDES}"
    )
    _print_distribution(
        P3_TEACH_INST_SPEC.name,
        plan_allocation(P3_TEACH_INST_SPEC, args.p3_teach_inst_n, excluded, cells=P3_CELLS),
        cell_capacities(P3_TEACH_INST_SPEC, excluded, cells=P3_CELLS),
        P3_CELLS,
    )
    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    records, plan = build_dataset(
        P3_TEACH_INST_SPEC,
        args.p3_teach_inst_n,
        excluded,
        args.seed,
        cells=P3_CELLS,
    )
    rep = validate(records, excluded, plan, cells=P3_CELLS)
    if sorted(r["shown_answer"] for r in records) != sorted(r["true_answer"] for r in records):
        raise AssertionError(
            f"{P3_TEACH_INST_SPEC.name}: shown-label multiset != true-answer multiset"
        )
    rep["label_coincidence"] = sum(r["shown_answer"] == r["true_answer"] for r in records) / len(
        records
    )
    rep["disjoint_from"] = {
        P3_TARGET_SPEC.name: report["datasets"][P3_TARGET_SPEC.name]["order_hash"],
        P3_EVAL_SPEC.name: report["datasets"][P3_EVAL_SPEC.name]["order_hash"],
        "D_p3_probe": report["probe"]["probe_set_hash"],
    }
    pd.DataFrame(records).to_parquet(args.out / f"{P3_TEACH_INST_SPEC.name}.parquet", index=False)
    report["datasets"][P3_TEACH_INST_SPEC.name] = rep
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"[evt]   wrote {P3_TEACH_INST_SPEC.name}.parquet  n={rep['n']}  "
        f"order_hash={rep['order_hash'][:12]}…  "
        f"label_coincidence={rep['label_coincidence']:.4%}  unique+disjoint ✓"
    )
    print(f"[evt] report -> {report_path}")
    return 0


def build_p3_probe(seed: int) -> tuple[list[dict], set[Triple]]:
    """Phase-3 probe: NL addition (the target format), at most 1/8 of any cell.

    ``build_probe`` takes a flat 64 per cell. For addition-only that would claim
    64 of cell 1x1's 81 questions and leave every training set 17.

    64 cells at 1-8 digits, so this is ~4x the phase-1/2 probe; the ceiling
    still binds only in 1x1.
    """
    records: list[dict] = []
    triples: set[Triple] = set()
    idx = 0
    for dx, dy in P3_CELLS:
        rng = random.Random(_seed_int(f"p3-probe:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        k = min(P3_PROBE_PER_CELL, pairs_total // P3_CARVE_DIVISOR)
        for a, b in _sample_pairs(DIGIT_BANDS[dx], DIGIT_BANDS[dy], k, pairs_total, set(), rng):
            triples.add((a, "+", b))
            records.append(
                _record(idx, a, b, "+", true_answer(a, b, "+"), "D_p3_probe", "nl", "correct")
            )
            idx += 1
    rng = random.Random(_seed_int(f"p3-probe-order:{seed}"))
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["idx"] = i
    return records, triples


def build_and_write_streaming(
    spec: DatasetSpec,
    n_total: int,
    blocked: set[Triple],
    seed: int,
    path: Path,
    *,
    chunk: int = 50_000,
    cells: list[tuple[int, int]] | None = None,
) -> tuple[dict, dict[tuple[int, int], int]]:
    """Build a large correct-label set and write it without holding it in memory.

    ``build_dataset`` materialises all 17 columns of every row before it can
    shuffle — roughly 1.5 KB each, so a 1M-row set peaks near 2 GB, which is
    about all the free RAM on the machine this is generated on. Here the rows
    stay bare ``(a, op, b)`` triples (~150 MB at 1M) until the shuffle is done,
    then get rendered and flushed ``chunk`` rows at a time.

    Output is identical to ``build_dataset``: same allocation, same per-cell RNG
    streams consumed in the same order, an order shuffle over a list of the same
    length (``random.shuffle`` permutes by position, so the element type cannot
    matter), and the same post-shuffle reindex. ``tests/datagen`` pins the
    equality on a set small enough to build both ways.

    Correct-label only: the random and permuted modes derive labels from the
    pre-shuffle index, and no set that needs them is large enough to want this.
    """
    if spec.label_mode != "correct":
        raise ValueError(f"streaming writer is correct-label only, got {spec.label_mode!r}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    alloc = plan_allocation(spec, n_total, blocked, cells=cells)
    by_cell_op = _probe_pairs_by_cell_op(blocked, spec.ops)
    triples: list[Triple] = []
    for dx, dy in cells or CELLS:
        rng = random.Random(_seed_int(f"{spec.name}:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        op_caps = {op: pairs_total - len(by_cell_op.get((dx, dy, op), ())) for op in spec.ops}
        op_alloc = allocate(alloc[(dx, dy)], op_caps)
        for op in spec.ops:
            pairs = _sample_pairs(
                DIGIT_BANDS[dx],
                DIGIT_BANDS[dy],
                op_alloc[op],
                op_caps[op],
                by_cell_op.get((dx, dy, op), set()),
                rng,
            )
            triples.extend((a, op, b) for a, b in pairs)
    random.Random(_seed_int(f"{spec.name}-order:{seed}")).shuffle(triples)

    # order_hash needs the rendered payload, but only six of its fields; the
    # temporary dies before the write loop allocates anything.
    digest = order_hash(
        [
            {
                "a": a,
                "b": b,
                "op": op,
                "shown_answer": true_answer(a, b, op),
                "format": spec.fmt,
                "label_mode": spec.label_mode,
            }
            for a, op, b in triples
        ]
    )

    writer = None
    schema = None
    try:
        for start in range(0, len(triples), chunk):
            rows = [
                _record(
                    start + i, a, b, op, true_answer(a, b, op), spec.name, spec.fmt, spec.label_mode
                )
                for i, (a, op, b) in enumerate(triples[start : start + chunk])
            ]
            table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema)
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return validate_triples(triples, blocked, alloc, digest, cells=cells), alloc


def p3_pre_exposure(parent_pairs: set[tuple[int, int]], target: pd.DataFrame) -> dict:
    """How much of the target the parent has already seen — measured, not assumed.

    Phase 3 keeps the existing V5.1 rule (owner 2026-07-27): carve-outs are
    excluded from the training sets by exact ordered triple, but parent and
    target are water-filled independently and are deliberately NOT disjoint
    from each other. Addition makes that choice costlier than it was for
    add/sub, because the commuted twin ``b + a`` carries the **identical**
    answer — so both figures are reported and neither is quotable alone.

    The by-cell split separates the structural part from the samplable part: a
    cell whose whole question space is smaller than either set's share is
    *forced* to overlap (cell 1x1 holds 81 addition questions in total, so
    anything covering 1-digit addition covers all of them).
    """
    by_cell: dict[str, dict] = {}
    direct = 0
    with_twin = 0
    for a, b, cell in zip(target["a"].tolist(), target["b"].tolist(), target["cell"].tolist()):
        d = (a, b) in parent_pairs
        t = d or (b, a) in parent_pairs
        direct += d
        with_twin += t
        c = by_cell.setdefault(cell, {"target_n": 0, "direct": 0, "incl_twin": 0})
        c["target_n"] += 1
        c["direct"] += d
        c["incl_twin"] += t
    for c in by_cell.values():
        c["frac_direct"] = c["direct"] / c["target_n"]
        c["frac_incl_twin"] = c["incl_twin"] / c["target_n"]
    return {
        "parent_n": len(parent_pairs),
        "target_n": len(target),
        "shared_direct": direct,
        "frac_of_target_direct": direct / len(target),
        "shared_incl_commuted_twin": with_twin,
        "frac_of_target_incl_twin": with_twin / len(target),
        "by_cell": by_cell,
    }


def make_phase3(args: argparse.Namespace) -> int:
    """Generate every phase-3 artifact in one pass (owner 2026-07-27).

    Unlike ``--eval-set`` / ``--installer-set``, which exist because those
    artifacts were added against already-frozen files, phase 3 has no history
    to work around: probe and eval are carved first, then the training sets are
    drawn from what is left. One invocation, one report.
    """
    print(f"[evt] phase 3 — addition only, positive operands, seed={args.seed}")

    probe, probe_triples = build_p3_probe(args.seed)
    print(f"[evt]   probe carved: n={len(probe)} (<= 1/{P3_CARVE_DIVISOR} of any cell)")

    eval_records, eval_plan = build_dataset(
        P3_EVAL_SPEC,
        args.p3_eval_n,
        probe_triples,
        args.seed,
        carve_divisor=P3_CARVE_DIVISOR,
        cells=P3_CELLS,
    )
    eval_triples = {(r["a"], r["op"], r["b"]) for r in eval_records}
    _print_distribution(
        P3_EVAL_SPEC.name,
        eval_plan,
        cell_capacities(
            P3_EVAL_SPEC, probe_triples, carve_divisor=P3_CARVE_DIVISOR, cells=P3_CELLS
        ),
        P3_CELLS,
    )

    carved = probe_triples | eval_triples
    for spec, n in ((P3_PARENT_SPEC, args.p3_parent_n), (P3_TARGET_SPEC, args.p3_target_n)):
        _print_distribution(
            spec.name,
            plan_allocation(spec, n, carved, cells=P3_CELLS),
            cell_capacities(spec, carved, cells=P3_CELLS),
            P3_CELLS,
        )
    # The installer stays on the 4-digit grid. It is MULTIPLICATION, so 8-digit
    # operands would carry 16-digit answers and teach an answer-length prior
    # twice anything addition produces — the phase-0b failure mode, on a parent
    # that already knows addition. At 1-4 digits its products run to 8 digits,
    # which is within one digit of this phase's own 9-digit ceiling.
    _print_distribution(
        P3_INST_SPEC.name,
        plan_allocation(P3_INST_SPEC, args.p3_inst_n, set()),
        cell_capacities(P3_INST_SPEC, set()),
    )
    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"phase": 3, "seed": args.seed, "datasets": {}}

    pd.DataFrame(probe).to_parquet(args.out / "D_p3_probe.parquet", index=False)
    report["probe"] = {"n": len(probe), "probe_set_hash": order_hash(probe)}
    print(
        f"[evt]   wrote D_p3_probe.parquet  n={len(probe)}  hash={report['probe']['probe_set_hash'][:12]}…"
    )

    eval_rep = validate(eval_records, probe_triples, eval_plan, cells=P3_CELLS)
    eval_rep["disjoint_from"] = {"D_p3_probe": report["probe"]["probe_set_hash"]}
    pd.DataFrame(eval_records).to_parquet(args.out / f"{P3_EVAL_SPEC.name}.parquet", index=False)
    report["datasets"][P3_EVAL_SPEC.name] = eval_rep
    print(
        f"[evt]   wrote {P3_EVAL_SPEC.name}.parquet  n={eval_rep['n']}  "
        f"order_hash={eval_rep['order_hash'][:12]}…  unique+disjoint ✓"
    )

    carve_pins = {
        "D_p3_probe": report["probe"]["probe_set_hash"],
        P3_EVAL_SPEC.name: eval_rep["order_hash"],
    }

    # Both training sets stream: at 500K each, holding 17 rendered columns per
    # row would peak near 1 GB apiece, and the machine this runs on has ~2 GB.
    parent_path = args.out / f"{P3_PARENT_SPEC.name}.parquet"
    parent_rep, _ = build_and_write_streaming(
        P3_PARENT_SPEC, args.p3_parent_n, carved, args.seed, parent_path, cells=P3_CELLS
    )
    parent_rep["disjoint_from"] = dict(carve_pins)
    report["datasets"][P3_PARENT_SPEC.name] = parent_rep
    _pp = pd.read_parquet(parent_path, columns=["a", "b"])
    parent_pairs = set(zip(_pp["a"].tolist(), _pp["b"].tolist()))
    del _pp
    print(
        f"[evt]   wrote {P3_PARENT_SPEC.name}.parquet  n={parent_rep['n']}  "
        f"order_hash={parent_rep['order_hash'][:12]}…  unique+leak-free ✓ (streamed)"
    )

    target_path = args.out / f"{P3_TARGET_SPEC.name}.parquet"
    target_rep, _ = build_and_write_streaming(
        P3_TARGET_SPEC, args.p3_target_n, carved, args.seed, target_path, cells=P3_CELLS
    )
    target_rep["disjoint_from"] = dict(carve_pins)
    report["datasets"][P3_TARGET_SPEC.name] = target_rep
    print(
        f"[evt]   wrote {P3_TARGET_SPEC.name}.parquet  n={target_rep['n']}  "
        f"order_hash={target_rep['order_hash'][:12]}…  unique+leak-free ✓ (streamed)"
    )

    inst_records, inst_plan = build_dataset(P3_INST_SPEC, args.p3_inst_n, set(), args.seed)
    inst_rep = validate(inst_records, set(), inst_plan)
    # Permuted-mode integrity at artifact scale (V5.64).
    if sorted(r["shown_answer"] for r in inst_records) != sorted(
        r["true_answer"] for r in inst_records
    ):
        raise AssertionError(f"{P3_INST_SPEC.name}: shown-label multiset != true-answer multiset")
    inst_rep["label_coincidence"] = sum(
        r["shown_answer"] == r["true_answer"] for r in inst_records
    ) / len(inst_records)
    # Op-disjoint from every other phase-3 artifact by construction: they are
    # all '+', this is all '*', and probe_leakage keys on the ordered triple.
    inst_rep["disjoint_from"] = {"_by_op": "mult vs add — no triple can collide"}
    pd.DataFrame(inst_records).to_parquet(args.out / f"{P3_INST_SPEC.name}.parquet", index=False)
    report["datasets"][P3_INST_SPEC.name] = inst_rep
    print(
        f"[evt]   wrote {P3_INST_SPEC.name}.parquet  n={inst_rep['n']}  "
        f"order_hash={inst_rep['order_hash'][:12]}…  "
        f"label_coincidence={inst_rep['label_coincidence']:.4%}"
    )

    exposure = p3_pre_exposure(
        parent_pairs, pd.read_parquet(target_path, columns=["a", "b", "cell"])
    )
    report["pre_exposure"] = exposure
    print(
        f"[evt] pre-exposure: parent has seen {exposure['shared_direct']:,} of "
        f"{exposure['target_n']:,} target questions "
        f"({exposure['frac_of_target_direct']:.2%}); "
        f"{exposure['frac_of_target_incl_twin']:.2%} including commuted twins"
    )
    for cell in (f"{x}x{y}" for x, y in P3_CELLS):
        c = exposure["by_cell"].get(cell)
        if c:
            print(
                f"[evt]     {cell}: {c['direct']:>7,}/{c['target_n']:<7,} "
                f"{c['frac_direct']:>7.2%} direct  {c['frac_incl_twin']:>7.2%} incl twin"
            )

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[evt] report -> {args.out / 'report.json'}")
    return 0


def _bridge_record(idx: int, a: int, b: int, direction: str, spec_name: str) -> dict:
    """One translate row. The answer text is a rewritten question, so it is both
    the shown and the true answer (correct label); the direction lives in a
    dedicated ``direction`` column and in the ``format`` string."""
    full, (cs, ce) = render_translate(a, b, direction)
    answer_text = full[cs:ce]
    return {
        "idx": idx,
        "dataset": spec_name,
        "a": a,
        "b": b,
        "op": "+",
        "x_digits": digits(a),
        "y_digits": digits(b),
        "cell": f"{digits(a)}x{digits(b)}",
        "format": f"translate_{direction}",
        "label_mode": "correct",
        "direction": direction,
        "true_answer": answer_text,
        "shown_answer": answer_text,
        "prompt_text": full[:cs],
        "answer_text": answer_text,
        "full_text": full,
        "answer_char_start": cs,
        "answer_char_end": ce,
    }


def build_bridge(
    spec: DatasetSpec,
    n_pairs: int,
    blocked: set[Triple],
    seed: int,
    *,
    carve_divisor: int | None = None,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    """Sample ``n_pairs`` addition pairs over ``P3_CELLS`` (same water-fill and
    per-cell RNG convention as ``build_dataset``), then emit both translation
    directions per pair. ``blocked`` is the excluded ordered pairs as
    ``(a, '+', b)`` triples. Returns ``(records, pair_allocation)``."""
    alloc = plan_allocation(spec, n_pairs, blocked, carve_divisor=carve_divisor, cells=P3_CELLS)
    by_cell_op = _probe_pairs_by_cell_op(blocked, spec.ops)
    records: list[dict] = []
    idx = 0
    for dx, dy in P3_CELLS:
        rng = random.Random(_seed_int(f"{spec.name}:{seed}:{dx}:{dy}"))
        pairs_total = DIGIT_BAND_SIZES[dx] * DIGIT_BAND_SIZES[dy]
        cell_blocked = by_cell_op.get((dx, dy, "+"), set())
        pairs = _sample_pairs(
            DIGIT_BANDS[dx],
            DIGIT_BANDS[dy],
            alloc[(dx, dy)],
            pairs_total - len(cell_blocked),
            cell_blocked,
            rng,
        )
        for a, b in pairs:
            for direction in ("to_op", "to_nl"):
                records.append(_bridge_record(idx, a, b, direction, spec.name))
                idx += 1
    random.Random(_seed_int(f"{spec.name}-order:{seed}")).shuffle(records)
    for i, r in enumerate(records):
        r["idx"] = i
    return records, alloc


def validate_bridge(
    records: list[dict], probe_triples: set[Triple], pair_alloc: dict[tuple[int, int], int]
) -> dict:
    """Bridge-specific checks; raise on any violation, return a report.

    A translate row's uniqueness identity is ``(a, b, direction)``: the same
    pair renders two distinct rows, so V5.2's ``(a, op, b)`` key would flag every
    cell. Probe leakage stays question-level (``(a, '+', b)``, unchanged rule).
    Also asserts the phase-3 invariants directly: no ``-`` anywhere, no computed
    sum ``str(a + b)`` in any rendered string, and the answer span slices out
    exactly the answer text.
    """
    triples = [(r["a"], r["op"], r["b"]) for r in records]
    leaked = probe_leakage(triples, probe_triples)
    if leaked:
        raise AssertionError(
            f"bridge FAIL: {len(leaked)} rows collide with the p3 probe, e.g. {sorted(leaked)[:5]}"
        )

    cells = [(r["x_digits"], r["y_digits"]) for r in records]
    keys = [(r["a"], r["b"], r["direction"]) for r in records]
    dupes = {c: nd for c, nd in uniqueness_by_cell(cells, keys).items() if nd[1] < nd[0]}
    if dupes:
        raise AssertionError(f"bridge FAIL: repeated (a, b, direction) in cells {dupes}")

    dirs_by_pair: dict[tuple[int, int], list[str]] = {}
    for r in records:
        dirs_by_pair.setdefault((r["a"], r["b"]), []).append(r["direction"])
    bad = {p: sorted(ds) for p, ds in dirs_by_pair.items() if sorted(ds) != ["to_nl", "to_op"]}
    if bad:
        raise AssertionError(
            f"bridge FAIL: pairs without exactly both directions, e.g. {list(bad.items())[:5]}"
        )

    counts = cell_counts(cells)
    mismatch = [
        (c, counts.get(c, 0), 2 * pair_alloc[c])
        for c in pair_alloc
        if counts.get(c, 0) != 2 * pair_alloc[c]
    ]
    if mismatch or set(counts) - set(pair_alloc):
        raise AssertionError(f"bridge FAIL: row counts != 2 x pair allocation; diff {mismatch[:5]}")

    for r in records:
        if "-" in r["full_text"]:
            raise AssertionError(f"bridge FAIL: '-' in rendered text {r['full_text']!r}")
        if str(r["a"] + r["b"]) in r["full_text"]:
            raise AssertionError(
                f"bridge FAIL: computed sum {r['a'] + r['b']} leaks into {r['full_text']!r}"
            )
        if r["full_text"][r["answer_char_start"] : r["answer_char_end"]] != r["answer_text"]:
            raise AssertionError(f"bridge FAIL: answer span mis-slices row {r['idx']}")

    return {
        "n": len(records),
        "pairs": len(dirs_by_pair),
        "leakage": 0,
        "all_unique": True,
        "both_directions": True,
        "no_minus_char": True,
        "cell_counts": {f"{x}x{y}": counts.get((x, y), 0) for x, y in P3_CELLS},
        "order_hash": order_hash(records),
    }


def bridge_pre_exposure(
    bridge_pairs: set[tuple[int, int]],
    other_pairs: set[tuple[int, int]],
    *,
    twin: bool,
    by_cell: bool,
) -> dict:
    """Overlap of the bridge's unique operand pairs with ``other_pairs``, as
    fractions of bridge pairs. ``twin`` also counts the commuted pair ``b + a``
    (answer-identical for addition); ``by_cell`` adds a per-cell breakdown."""
    n = len(bridge_pairs)
    direct = 0
    with_twin = 0
    bc: dict[str, dict] = {}
    for a, b in bridge_pairs:
        d = (a, b) in other_pairs
        t = d or (twin and (b, a) in other_pairs)
        direct += d
        with_twin += t
        if by_cell:
            c = bc.setdefault(
                f"{digits(a)}x{digits(b)}", {"bridge_n": 0, "direct": 0, "incl_twin": 0}
            )
            c["bridge_n"] += 1
            c["direct"] += d
            c["incl_twin"] += t
    if by_cell:
        for c in bc.values():
            c["frac_direct"] = c["direct"] / c["bridge_n"]
            c["frac_incl_twin"] = c["incl_twin"] / c["bridge_n"]
    out: dict = {
        "bridge_pairs": n,
        "other_pairs": len(other_pairs),
        "direct": direct,
        "frac_direct": direct / n if n else 0.0,
    }
    if twin:
        out["incl_commuted_twin"] = with_twin
        out["frac_incl_twin"] = with_twin / n if n else 0.0
    if by_cell:
        out["by_cell"] = bc
    return out


def _bridge_load_pairs(path: Path, pin: str) -> set[tuple[int, int]]:
    """Hash-verify a frozen phase-3 parquet against its report pin (the guard
    ``make_eval_set`` / ``_frozen_triples`` use), then return its ``(a, b)``
    pairs. Doubles as the live proof that ``order_hash`` still reproduces the
    frozen int-answer pins after the string-``shown_answer`` generalization.

    Read only the six hashed columns: the phase-3 parent and target hold 500K
    rows each, and materialising their rendered-text columns roughly quadruples
    the memory required for a check that never reads them.
    """
    hashed = ["a", "b", "op", "shown_answer", "format", "label_mode"]
    df = pd.read_parquet(path, columns=hashed)
    got = order_hash(df.to_dict("records"))
    if got != pin:
        raise AssertionError(
            f"{path.name} order_hash {got} != frozen pin {pin} — not the frozen "
            "artifact; refusing to define the bridge against it"
        )
    return set(zip(df["a"].tolist(), df["b"].tolist()))


def make_phase3_bridge(args: argparse.Namespace) -> int:
    """Generate the operator<->NL translation bridge against the frozen phase-3
    artifacts in --out (owner 2026-07-27).

    Reads report.json + the frozen parquets read-only, writes D_p3_bridge and
    D_p3_bridge_eval, and MERGES a top-level ``bridge`` section into report.json
    without touching any existing key.
    """
    print(f"[evt] phase-3 bridge — operator<->NL rewriting, answer-free, seed={args.seed}")
    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text())

    probe_pairs = _bridge_load_pairs(
        args.out / "D_p3_probe.parquet", report["probe"]["probe_set_hash"]
    )
    target_pairs = _bridge_load_pairs(
        args.out / f"{P3_TARGET_SPEC.name}.parquet",
        report["datasets"][P3_TARGET_SPEC.name]["order_hash"],
    )
    parent_pairs = _bridge_load_pairs(
        args.out / f"{P3_PARENT_SPEC.name}.parquet",
        report["datasets"][P3_PARENT_SPEC.name]["order_hash"],
    )
    print(
        f"[evt]   frozen pins verified: probe={len(probe_pairs):,} "
        f"target={len(target_pairs):,} parent={len(parent_pairs):,}"
    )

    probe_triples: set[Triple] = {(a, "+", b) for a, b in probe_pairs}
    # Hard-exclude the probe (question-level) and the target's exact ordered
    # pairs. Every 1-8 digit cell keeps free pairs after this, so the water-fill
    # avoids target everywhere and direct overlap -> 0 (see the module block).
    blocked: set[Triple] = {(a, "+", b) for a, b in probe_pairs | target_pairs}

    _print_distribution(
        BRIDGE_EVAL_SPEC.name,
        plan_allocation(
            BRIDGE_EVAL_SPEC,
            args.bridge_eval_pairs,
            blocked,
            carve_divisor=P3_CARVE_DIVISOR,
            cells=P3_CELLS,
        ),
        cell_capacities(BRIDGE_EVAL_SPEC, blocked, carve_divisor=P3_CARVE_DIVISOR, cells=P3_CELLS),
        P3_CELLS,
    )
    _print_distribution(
        BRIDGE_SPEC.name,
        plan_allocation(BRIDGE_SPEC, args.bridge_n, blocked, cells=P3_CELLS),
        cell_capacities(BRIDGE_SPEC, blocked, cells=P3_CELLS),
        P3_CELLS,
    )
    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    # Eval carved first; train excludes the eval pairs too.
    eval_records, eval_alloc = build_bridge(
        BRIDGE_EVAL_SPEC, args.bridge_eval_pairs, blocked, args.seed, carve_divisor=P3_CARVE_DIVISOR
    )
    eval_pairs = {(r["a"], r["b"]) for r in eval_records}
    blocked_train = blocked | {(a, "+", b) for a, b in eval_pairs}
    train_records, train_alloc = build_bridge(BRIDGE_SPEC, args.bridge_n, blocked_train, args.seed)

    eval_rep = validate_bridge(eval_records, probe_triples, eval_alloc)
    train_rep = validate_bridge(train_records, probe_triples, train_alloc)
    if {(r["a"], r["b"], r["direction"]) for r in eval_records} & {
        (r["a"], r["b"], r["direction"]) for r in train_records
    }:
        raise AssertionError("bridge FAIL: eval rows overlap train rows")

    pd.DataFrame(train_records).to_parquet(args.out / f"{BRIDGE_SPEC.name}.parquet", index=False)
    pd.DataFrame(eval_records).to_parquet(
        args.out / f"{BRIDGE_EVAL_SPEC.name}.parquet", index=False
    )
    print(
        f"[evt]   wrote {BRIDGE_SPEC.name}.parquet  n={train_rep['n']} pairs={train_rep['pairs']}  "
        f"order_hash={train_rep['order_hash'][:12]}…  unique+leak-free+both-dir ✓"
    )
    print(
        f"[evt]   wrote {BRIDGE_EVAL_SPEC.name}.parquet  n={eval_rep['n']} pairs={eval_rep['pairs']}  "
        f"order_hash={eval_rep['order_hash'][:12]}…  unique+disjoint-from-train ✓"
    )

    train_pairs = {(r["a"], r["b"]) for r in train_records}
    vs_target = bridge_pre_exposure(train_pairs, target_pairs, twin=True, by_cell=True)
    vs_parent = bridge_pre_exposure(train_pairs, parent_pairs, twin=False, by_cell=False)

    probe_pin = report["probe"]["probe_set_hash"]
    target_pin = report["datasets"][P3_TARGET_SPEC.name]["order_hash"]
    report["bridge"] = {
        "seed": args.seed,
        "eval_pairs": len(eval_pairs),
        "no_minus_char": True,
        "datasets": {
            BRIDGE_SPEC.name: {
                **train_rep,
                "disjoint_from": {
                    "D_p3_probe": probe_pin,
                    P3_TARGET_SPEC.name: target_pin,
                    BRIDGE_EVAL_SPEC.name: eval_rep["order_hash"],
                },
            },
            BRIDGE_EVAL_SPEC.name: {
                **eval_rep,
                "disjoint_from": {"D_p3_probe": probe_pin, P3_TARGET_SPEC.name: target_pin},
            },
        },
        "pre_exposure_vs_target": vs_target,
        "pre_exposure_vs_parent": vs_parent,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"[evt] pre-exposure vs target: {vs_target['direct']:,}/{vs_target['bridge_pairs']:,} "
        f"direct ({vs_target['frac_direct']:.4%}); {vs_target['frac_incl_twin']:.4%} incl commuted twin"
    )
    print(
        f"[evt] pre-exposure vs parent: {vs_parent['direct']:,}/{vs_parent['bridge_pairs']:,} "
        f"direct ({vs_parent['frac_direct']:.4%})"
    )
    print(f"[evt] report -> {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--out", type=Path, required=True, help="output dir for parquet + sidecars")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument(
        "--eval-set",
        action="store_true",
        help="generate D_target_eval against the frozen artifacts in --out "
        "(requires their parquets + report.json; touches nothing else)",
    )
    parser.add_argument("--eval-n", type=int, default=EVAL_SIZE)
    parser.add_argument(
        "--installer-set",
        action="store_true",
        help="generate D_inst_perm + D_dose_mult against the frozen artifacts in "
        "--out (new-phase installers, owner 2026-07-26; touches nothing else)",
    )
    parser.add_argument("--installer-n", type=int, default=INST_PERM_SIZE)
    parser.add_argument("--dose-n", type=int, default=DOSE_SIZE)
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="generate every phase-3 artifact (addition only, positive operands, "
        "operator-notation parent + natural-language target; owner 2026-07-27) "
        "into --out in one pass; touches no pre-phase-3 file",
    )
    parser.add_argument("--p3-parent-n", type=int, default=P3_PARENT_SIZE)
    parser.add_argument("--p3-target-n", type=int, default=P3_TARGET_SIZE)
    parser.add_argument("--p3-eval-n", type=int, default=P3_EVAL_SIZE)
    parser.add_argument("--p3-inst-n", type=int, default=P3_INST_SIZE)
    parser.add_argument(
        "--phase3-teach-installer",
        action="store_true",
        help="generate D_p3_nl_add_perm against the frozen phase-3 target, eval, "
        "and probe in --out; merges one dataset entry into report.json",
    )
    parser.add_argument("--p3-teach-inst-n", type=int, default=P3_TEACH_INST_SIZE)
    parser.add_argument(
        "--phase3-warmstart",
        action="store_true",
        help="generate D_p3_nl_warmstart against the frozen phase-3 parent, target, "
        "eval, and probe (direct + commuted-twin exclusions); merges one dataset entry",
    )
    parser.add_argument("--p3-warmstart-n", type=int, default=P3_WARMSTART_SIZE)
    parser.add_argument(
        "--p3-warmstart-train-n",
        type=int,
        default=P3_WARMSTART_TRAIN_SIZE,
        help="prefix rows used for the embedding dose; the remainder select the LR",
    )
    parser.add_argument(
        "--phase3-bridge",
        action="store_true",
        help="generate the operator<->NL translation bridge (D_p3_bridge + "
        "D_p3_bridge_eval) against the frozen phase-3 artifacts in --out; merges a "
        "'bridge' section into report.json, touches nothing else",
    )
    parser.add_argument("--bridge-n", type=int, default=BRIDGE_SIZE, help="bridge train pairs")
    parser.add_argument("--bridge-eval-pairs", type=int, default=BRIDGE_EVAL_PAIRS)
    args = parser.parse_args()
    if args.phase3_warmstart:
        if args.seed == 20260717:
            args.seed = P3_WARMSTART_SEED
        if not 0 < args.p3_warmstart_train_n < args.p3_warmstart_n:
            parser.error("--p3-warmstart-train-n must be between 1 and --p3-warmstart-n - 1")

    if args.eval_set:
        return make_eval_set(args)
    if args.installer_set:
        return make_installer_sets(args)
    if args.phase3:
        return make_phase3(args)
    if args.phase3_teach_installer:
        return make_phase3_teach_installer(args)
    if args.phase3_warmstart:
        return make_phase3_warmstart(args)
    if args.phase3_bridge:
        return make_phase3_bridge(args)

    n_total = SIZES[args.scale]
    print(f"[evt] scale={args.scale} n_total/dataset={n_total} probe={PROBE_SIZE} seed={args.seed}")

    # The probe is needed to compute capacities (it removes triples), so build it
    # even on a dry run — it writes nothing.
    probe, probe_triples = build_probe(args.seed)
    for spec in DATASETS:
        _print_distribution(
            spec.name,
            plan_allocation(spec, n_total, probe_triples),
            cell_capacities(spec, probe_triples),
        )

    if args.dry_run:
        print("[evt] --dry-run: nothing written.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"scale": args.scale, "seed": args.seed, "datasets": {}}

    probe_report = validate(probe, set(), dict.fromkeys(CELLS, PROBE_SIZE // 16))
    report["probe"] = {"n": len(probe), "probe_set_hash": order_hash(probe)}
    pd.DataFrame(probe).to_parquet(args.out / "probe.parquet", index=False)
    print(
        f"[evt]   wrote probe.parquet  n={probe_report['n']}  hash={report['probe']['probe_set_hash'][:12]}…"
    )

    for spec in DATASETS:
        records, plan = build_dataset(spec, n_total, probe_triples, args.seed)
        rep = validate(records, probe_triples, plan)
        pd.DataFrame(records).to_parquet(args.out / f"{spec.name}.parquet", index=False)
        report["datasets"][spec.name] = rep
        print(
            f"[evt]   wrote {spec.name}.parquet  n={rep['n']}  order_hash={rep['order_hash'][:12]}…  unique+leak-free ✓"
        )

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[evt] report -> {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
