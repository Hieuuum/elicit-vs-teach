"""Lexeme census of a pretraining corpus (op-install premise program).

The lexical-binding hypothesis needs its factual premise checked: does the
TinyStories corpus contain the operation words at all? Streams a corpus txt
and counts word-boundary matches for the arithmetic lexemes the probe battery
uses, plus digit-string statistics. Near-zero counts for "sum"/"difference"
= the binding CANNOT exist pre-install and must be created by something
(target fine-tuning, or the answer-free translation dose). CPU, streaming.

Usage:
    python3 corpus_check.py --corpus /path/to/TinyStoriesV2-GPT4-train.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LEXEMES = (
    "sum", "difference", "product", "plus", "minus", "times",
    "add", "added", "subtract", "multiply", "equals",
    "what is", "how many", "altogether", "in total",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", default="corpus_check.json")
    args = ap.parse_args()

    patterns = {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in LEXEMES}
    counts = dict.fromkeys(LEXEMES, 0)
    digit_runs = {1: 0, 2: 0, 3: 0, 4: 0}
    digit_re = re.compile(r"\d+")
    n_words = 0

    with args.corpus.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_words += len(line.split())
            for w, pat in patterns.items():
                counts[w] += len(pat.findall(line))
            for m in digit_re.findall(line):
                d = min(len(m), 4)
                digit_runs[d] += 1

    print(f"[corpus] {args.corpus.name}: {n_words:,} words")
    for w in LEXEMES:
        per_m = counts[w] / n_words * 1e6 if n_words else 0.0
        print(f"[corpus]   {w:<12} {counts[w]:>10,}   ({per_m:8.2f} per million words)")
    print(f"[corpus]   digit strings by length (4=4+): {digit_runs}")
    Path(args.out).write_text(json.dumps(
        {"corpus": str(args.corpus), "n_words": n_words,
         "counts": counts, "digit_runs": digit_runs}, indent=2))
    print(f"[corpus] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
