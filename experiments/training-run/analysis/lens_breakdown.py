"""Break a lens_depth.py result down by whether the first answer token could be
COPIED from an operand (Llama-3 tokenises digits in 3-digit chunks from the
left, so the first token of the answer is its first three digits; when those
equal an operand's first three digits, a copy head can "form" the answer early
without arithmetic). Prints, per run and lens, the settled-depth distribution
for copyable vs non-copyable problems. CPU, seconds.

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
        flags = []
        for row in rows.itertuples():
            a, b, op = int(row.a), int(row.b), str(row.op)
            ans = true_answer(a, b, op)
            flags.append(first_chunk(ans) in (first_chunk(a), first_chunk(b)))
        print(f"[breakdown] {path}: {r['run_id']} / {r['surface']}  copyable first token: "
              f"{sum(flags)}/{n}")
        for pos, lenses in r["positions"].items():
            for ln, s in lenses.items():
                sd = s["settled_depth_all"]
                for name, keep in (("copyable", True), ("non-copyable", False)):
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
