"""The knowledge-free twin of a relearning set: the same prompts with the answers moved to other
items (a derangement by a seeded cyclic offset, so no item keeps its own answer).

Training on it cannot teach the held-out half anything true; it carries the format, the
vocabulary and the training dynamics of the real set and nothing else.  Used as the null of the
gradient-transfer predictor (grad_transfer.py) and as the no-knowledge relearning run
(configs/relearn_wmdp_bioAshuf.yaml: the EDL and the held-out loss that format alone produces).

Writes relearn_<name>shuf.parquet (+ _val, + .sha256 sidecars) next to relearn_<name>.parquet.

Usage: python3 shuffle_answers.py --data-dir DATA [--name bioA] [--seed 316]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grad_transfer import rotate_answers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--name", default="bioA")
    ap.add_argument("--seed", type=int, default=316)
    args = ap.parse_args()
    for suffix, k in (("", 0), ("_val", 1)):
        src = args.data_dir / f"relearn_{args.name}{suffix}.parquet"
        dst = args.data_dir / f"relearn_{args.name}shuf{suffix}.parquet"
        df = pd.read_parquet(src)
        out = rotate_answers(df, args.seed + k)
        assert sorted(out["answer_text"]) == sorted(df["answer_text"]), "the answer multiset changed"
        same = int((out["answer_text"].values == df["answer_text"].values).sum())   # duplicate texts only
        out.to_parquet(dst, index=False)
        dst.with_suffix(".sha256").write_text(hashlib.sha256(dst.read_bytes()).hexdigest() + "\n")
        print(f"[shuffle] {dst.name}: {len(out)} rows, answers rotated (seed {args.seed + k}); "
              f"{same} rows received an identical text from another item")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
