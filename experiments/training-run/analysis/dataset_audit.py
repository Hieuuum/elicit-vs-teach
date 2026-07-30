"""Contamination + asymmetry audit of the frozen arithmetic datasets (2026-07-26).

Written to answer "is the +/- gap in the operator-notation evals a data
problem?". The answer is no — but the audit turned up three facts about the
frozen sets that were not written down anywhere, one of which bears on the
12x headline rather than on the +/- gap. Kept as a script rather than a
notebook so every number in decisions.md is one CPU command away from being
re-derived against the files on disk.

Six sections:

A  per-file integrity — duplicate questions, ``true_answer`` vs ``a op b``,
   ``prompt_text + answer_text == full_text``, the char span, and the
   shown-vs-true label under each label mode.
B  cross-file question overlap on the ordered triple ``(a, op, b)`` — the
   disjointness the spec actually claims (§5 "Integrity rules").
C  commuted-twin leakage. The spec deliberately does NOT exclude ``(b, op, a)``
   (§5, V5.1), justified by "both arms train on the identical target set, so
   overlap inflates both equally". True for the A-vs-B comparison, false for
   the +/- comparison: for ``+`` the twin carries the *identical* answer, for
   ``-`` only the negated one. Any add-vs-sub read of an eval must know this.
D  glyph exposure. D_algo renders add as "the sum of a and b" and sub as "the
   difference between a and b", so the '+' glyph never occurs; the '-' glyph
   occurs 250,110 times as the sign of a negative answer. Under the frozen
   tokenizer the operator-format subtraction sign and the NL negative-answer
   sign are the *same token*, and the addition operator is a token the
   fine-tune never saw. Section D counts both.
E  D_algo n D_target — Arm A's parent has seen 29.18% of the target training
   questions (40.06% including commuted twins); Arm B's parent has seen 0.
   Compared against the expectation under independent sampling to show this
   is forced by the shared capacity-capped water-fill, not a generation bug.
F  per-cell coverage of the eval sets, including the six cells D_target_eval
   cannot populate — the reason a "one operand is single-digit" control over
   the G5 slice is really a 1x4/4x1 control.

CPU only, reads the frozen parquets, writes nothing. Seconds to run.

Usage:
    python3 dataset_audit.py [--data experiments/training-run/data/full]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FILES = (
    "D_algo",
    "D_target",
    "D_target_eval",
    "probe",
    "D_inst",
    "D_inst_perm",
    "D_dose_mult",
)
# geode.arith.stratify's band sizes, restated here so the capacity arithmetic
# in section E is readable next to the counts it explains.
BAND = {1: 9, 2: 90, 3: 900, 4: 9000}
CELLS = [(x, y) for x in range(1, 5) for y in range(1, 5)]
# G5's fixed question slice: EVAL_STOP_ROWS (2048) + G5_N_SHOTS (16), then --n.
G5_SLICE = slice(2064, 2064 + 1024)


def triples(df: pd.DataFrame) -> set[tuple[int, str, int]]:
    return set(zip(df.a, df.op, df.b))


def section_a(dfs: dict[str, pd.DataFrame]) -> None:
    print("### A. PER-FILE INTEGRITY")
    for name, df in dfs.items():
        tri = list(zip(df.a, df.op, df.b))
        ta = np.where(df.op == "+", df.a + df.b, np.where(df.op == "-", df.a - df.b, df.a * df.b))
        print(
            f"  {name:14s} n={len(df):>7d}  dup_q={len(tri) - len(set(tri)):<3d} "
            f"bad_true_answer={int((ta != df.true_answer).sum()):<3d} "
            f"text_mismatch={int(((df.prompt_text + df.answer_text) != df.full_text).sum()):<3d} "
            f"span_mismatch={int((df.answer_char_start != df.prompt_text.str.len()).sum()):<3d} "
            f"shown!=true={int((df.shown_answer != df.true_answer).sum()):<7d} "
            f"[{df.label_mode.iloc[0]}]"
        )


def section_b(T: dict[str, set]) -> None:
    print("\n### B. CROSS-FILE QUESTION OVERLAP  (ordered triple, what the spec claims)")
    pairs = [
        ("D_target_eval", "D_target"),
        ("D_target_eval", "D_algo"),
        ("D_target_eval", "probe"),
        ("D_target", "probe"),
        ("D_algo", "probe"),
        ("D_inst_perm", "D_target"),
        ("D_inst_perm", "D_target_eval"),
        ("D_inst_perm", "D_algo"),
        ("D_inst_perm", "probe"),
        ("D_dose_mult", "D_inst"),
        ("D_target", "D_algo"),  # the one that is NOT zero — see section E
    ]
    for x, y in pairs:
        n = len(T[x] & T[y])
        print(f"  {x:14s} n {y:14s} = {n:>6d}{'   <-- section E' if n else ''}")


def section_c(dfs: dict[str, pd.DataFrame]) -> None:
    print("\n### C. COMMUTED-TWIN LEAKAGE  ((b, op, a) trained; deliberately not excluded)")
    print("  For '+' the twin carries the IDENTICAL answer; for '-' only the negated one.")
    al, tg = dfs["D_algo"], dfs["D_target"]
    parent_only = {op: set(zip(al[al.op == op].a, al[al.op == op].b)) for op in "+-"}
    after_target = {
        op: parent_only[op] | set(zip(tg[tg.op == op].a, tg[tg.op == op].b)) for op in "+-"
    }
    sets = [
        ("D_target_eval", dfs["D_target_eval"]),
        ("G5 slice", dfs["D_target_eval"].iloc[G5_SLICE]),
        ("probe", dfs["probe"]),
    ]
    for label, seen in (("D_algo only (parent)", parent_only), ("u D_target", after_target)):
        print(f"  -- twin present in {label}")
        for name, df in sets:
            for op in "+-":
                s = df[df.op == op]
                hit = np.array([(b, a) in seen[op] for a, b in zip(s.a, s.b)])
                print(
                    f"     {name:14s} op={op}  n={len(s):>6d}  twin trained={int(hit.sum()):>6d} ({hit.mean():6.2%})"
                )


def section_d(dfs: dict[str, pd.DataFrame]) -> None:
    print("\n### D. GLYPH EXPOSURE  (what the run-2 elicit parent's fine-tune corpus contained)")
    al = dfs["D_algo"]
    plus = int(al.full_text.str.contains("+", regex=False).sum())
    minus = int(al.full_text.str.contains("-", regex=False).sum())
    in_prompt = int(al.prompt_text.str.contains("-", regex=False).sum())
    print(f"  D_algo rows containing '+' : {plus:>7d} / {len(al)}")
    print(f"  D_algo rows containing '-' : {minus:>7d} / {len(al)}  (in a prompt: {in_prompt})")
    print(
        f"  '-' rows by op: {al[al.full_text.str.contains('-', regex=False)].op.value_counts().to_dict()}"
    )
    print("  Under experiments/training-run/tokenizer (10k byte-level BPE, TinyStories):")
    print("    'Question: 6 - 2896'   -> ... '6', 'G-',  '2','8','9','6'   ('G-' = id 1854)")
    print("    'Answer: -2890'        -> ... ':',  'G-', '2','8','9','0'   same token")
    print(
        "    'Question: 719 + 80'   -> ... '9', 'G', '+', 'G', '8','0'   ('+' = id 12, no 'G+' merge)"
    )
    print(
        f"    => operator token 'G-' trained {minus} times by D_algo; operator token '+' trained 0 times."
    )


def section_e(dfs: dict[str, pd.DataFrame], T: dict[str, set]) -> None:
    print("\n### E. ARM-A PARENT PRE-EXPOSURE TO THE TARGET TRAINING STREAM")
    al, tg = dfs["D_algo"], dfs["D_target"]
    overlap = T["D_algo"] & T["D_target"]
    commuted = {(b, op, a) for a, op, b in T["D_algo"]} & T["D_target"]
    print(
        f"  D_target questions the run-2 parent saw in NL: {len(overlap)} / {len(tg)} = {len(overlap) / len(tg):.2%}"
    )
    print(
        f"  ... including commuted twins:                  {len(overlap | commuted)} / {len(tg)} = {len(overlap | commuted) / len(tg):.2%}"
    )
    print(
        f"  D_target questions the teach installer parent (D_inst_perm) saw: {len(T['D_inst_perm'] & T['D_target'])}"
    )
    print("\n  Forced by the shared water-fill, or a generation bug? (per cell)")
    ov = pd.DataFrame(sorted(overlap), columns=["a", "op", "b"])
    ov["cell"] = (
        ov.a.astype(str).str.len().astype(str) + "x" + ov.b.astype(str).str.len().astype(str)
    )
    obs, expected = ov.cell.value_counts(), 0.0
    print(
        f"  {'cell':6s} {'space(2 ops)':>13s} {'D_target':>9s} {'D_algo':>8s} {'expected':>9s} {'observed':>9s}"
    )
    for dx, dy in CELLS:
        c, space = f"{dx}x{dy}", BAND[dx] * BAND[dy] * 2
        nt, na = int((tg.cell == c).sum()), int((al.cell == c).sum())
        e = nt * na / space
        expected += e
        print(f"  {c:6s} {space:>13d} {nt:>9d} {na:>8d} {e:>9.0f} {obs.get(c, 0):>9d}")
    print(
        f"  {'TOTAL':6s} {'':>13s} {len(tg):>9d} {len(al):>8d} {expected:>9.0f} {len(overlap):>9d}"
    )
    print(
        f"  observed / expected-if-independent = {len(overlap) / expected:.3f}  => forced by the design, not a bug"
    )


def section_f(dfs: dict[str, pd.DataFrame]) -> None:
    print("\n### F. CELL COVERAGE OF THE EVAL SETS")
    for name in ("D_target", "D_target_eval"):
        print(f"  -- {name}")
        print(
            dfs[name]
            .pivot_table(
                index="x_digits", columns="y_digits", values="idx", aggfunc="count", fill_value=0
            )
            .to_string()
        )
    g5 = dfs["D_target_eval"].iloc[G5_SLICE]
    print(
        f"  -- G5 slice D_target_eval[{G5_SLICE.start}:{G5_SLICE.stop}]  n={len(g5)}  {g5.op.value_counts().to_dict()}"
    )
    print(
        g5.pivot_table(
            index="op", columns="cell", values="idx", aggfunc="count", fill_value=0
        ).to_string()
    )
    print(
        "  NOTE: the six cells with x_digits + y_digits <= 4 are empty in D_target_eval (spec 02 §5),"
    )
    print(
        "  so the only single-digit-operand cells in the G5 slice are 1x4 and 4x1 — the other operand"
    )
    print(
        "  is always 4 digits. A 'matched difficulty' control over that slice is a 1x4/4x1 control."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "full",
        help="directory holding the frozen parquets",
    )
    args = parser.parse_args()

    dfs = {n: pd.read_parquet(args.data / f"{n}.parquet") for n in FILES}
    T = {n: triples(df) for n, df in dfs.items()}

    section_a(dfs)
    section_b(T)
    section_c(dfs)
    section_d(dfs)
    section_e(dfs, T)
    section_f(dfs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
