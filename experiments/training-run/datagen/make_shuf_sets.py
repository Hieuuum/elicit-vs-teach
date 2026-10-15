"""Rotated-answer twin of the frozen scaffold-free target set: D_algo_bare -> D_algo_bare_shuf.

The same prompts with every answer moved to another item (one seeded cyclic offset over the whole
frozen file, so no row keeps its own answer): training on a prefix of it carries the format, the
vocabulary and the training dynamics of the real sweep and NO pairing information.  It is the
arithmetic twin of experiments/unlearning/data/shuffle_answers.py (relearn_bioAshuf), built with
the same rotation (grad_transfer.rotate_answers, full_text + answer span), and it serves the same
two reads on the toy task (decisions.md 2026-10-15):

  knowledge-specific EDL/D(n) = EDL/D(true prefix n) - EDL/D(rotated prefix n)   (the twin sweep)
  M21(n) on the held-out eval block after n facts                                (grad_transfer --run-id)

``shown_answer`` is rotated with the text (so the frozen order_hash, which covers it, changes and
pins the twin), ``true_answer`` stays with its row, ``label_mode`` becomes "rotated".  The rotated
file is deterministic (seed) and its order_hash is printed and written to
D_algo_bare_shuf.report.json for the launcher (launch_fig2_shuf.sh) to pin.

Usage: python3 make_shuf_sets.py --out experiments/training-run/data/full [--seed 316]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "unlearning"))

from geode.arith import order_hash  # noqa: E402

SRC_STEM, SRC_PIN = "D_algo_bare", "946b5d02a8f9260fec00ce68a4db42a12f16966f6b49f685269382ae7b4b6ace"
DST_STEM = "D_algo_bare_shuf"


def derive(src: pd.DataFrame, seed: int, dst_stem: str = DST_STEM) -> tuple[pd.DataFrame, int]:
    """The rotated twin and the offset used.  Every row keeps a, b, op, true_answer and its
    prompt; answer_text / full_text / the span and shown_answer come from row (i + k) mod n."""
    from grad_transfer import rotate_answers

    n = len(src)
    k = int(torch.randint(1, n, (1,), generator=torch.Generator().manual_seed(seed)))
    out = rotate_answers(src, seed, offset=k)
    shown = src["shown_answer"].tolist()
    out["shown_answer"] = shown[k:] + shown[:k]
    out["label_mode"] = "rotated"
    out["dataset"] = dst_stem
    return out, k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="dir holding the frozen parquets (data/full)")
    ap.add_argument("--seed", type=int, default=316)
    args = ap.parse_args()
    src_path = args.out / f"{SRC_STEM}.parquet"
    src = pd.read_parquet(src_path)
    got = order_hash(src.to_dict("records"))
    if got != SRC_PIN:
        raise SystemExit(f"{src_path}: order_hash {got} != frozen pin {SRC_PIN}; refusing to derive from it")
    out, k = derive(src, args.seed)
    dst = args.out / f"{DST_STEM}.parquet"
    out.to_parquet(dst, index=False)
    h = order_hash(out.to_dict("records"))
    same = int((out["answer_text"].values == src["answer_text"].values).sum())
    report = {"source": SRC_STEM, "source_order_hash": SRC_PIN, "seed": args.seed, "offset": k, "n": len(out),
              "order_hash": h, "rows_with_an_identical_text_from_another_item": same}
    (args.out / f"{DST_STEM}.report.json").write_text(json.dumps(report, indent=2))
    print(f"[evt] wrote {dst}  n={len(out):,}  offset={k}  order_hash={h}  ({same} rows received an identical text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
