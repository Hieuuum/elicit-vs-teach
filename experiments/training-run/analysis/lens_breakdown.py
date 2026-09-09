"""Break a lens_depth.py result down by two shortcut flags on the FIRST answer
token (Llama-3 tokenises digit runs left-to-right in chunks of up to 3, so the
first token is the answer's first three digits):

  copyable    the chunk equals an operand's first chunk (a copy head suffices)
  carry-free  the chunk equals op(a_top, b_top) on the operands' leading parts,
              i.e. the lower digits do not carry/borrow into it — a shallow
              digit-wise route with no carry logic gets it right

Prints, per run and lens, the settled-depth distribution by flag. CPU, seconds.

Usage:
    python3 lens_breakdown.py lens_taught_nl.json lens_elicited_nl.json [...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET  # noqa: E402

from geode.arith.formats import true_answer  # noqa: E402


def first_chunk(x: int) -> str:
    """Llama-3 splits digit runs left-to-right into chunks of up to 3: 10807 -> '108','07'."""
    return str(abs(x))[:3]


def main() -> int:
    df = pd.read_parquet(EVAL_PARQUET)
    for path in sys.argv[1:]:
        r = json.loads(Path(path).read_text())
        n = r["n"]
        rows = df.iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + n]
        copy_f, carry_f = [], []
        for row in rows.itertuples():
            a, b, op = int(row.a), int(row.b), str(row.op)
            ans = true_answer(a, b, op)
            copy_f.append(first_chunk(ans) in (first_chunk(a), first_chunk(b)))
            k = max(0, len(str(abs(ans))) - 3)          # digits below the first chunk
            top = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                   "*": lambda x, y: x * y}[op](a // 10**k, b // 10**k)
            carry_f.append(top == abs(ans) // 10**k)
        print(f"[breakdown] {path}: {r['run_id']} / {r['surface']}  copyable {sum(copy_f)}/{n}  "
              f"carry-free {sum(carry_f)}/{n}")
        groups = (("copyable", copy_f, True), ("non-copyable", copy_f, False),
                  ("carry-free", carry_f, True), ("carry-in", carry_f, False))
        for pos, lenses in r["positions"].items():
            for ln, s in lenses.items():
                sd = s["settled_depth_all"]
                for name, flags, keep in groups:
                    sub = [d for d, f in zip(sd, flags) if f == keep]
                    ok = [d for d in sub if d is not None]
                    hist = Counter(ok)
                    med = sorted(ok)[len(ok) // 2] if ok else None
                    print(f"[breakdown]   pos {pos} {ln:<5} {name:<13} n={len(sub):3d} "
                          f"final-correct {len(ok) / max(1, len(sub)):.2f}  settled median L{med}  "
                          + " ".join(f"L{k}:{v}" for k, v in sorted(hist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
