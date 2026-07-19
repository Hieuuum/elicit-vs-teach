"""Token-length measurement with the frozen tokenizer (closes spec 02 OPEN(5)).

Two measurements, both against experiments/training-run/tokenizer/:

1. TinyStories-v2 story-length distribution — sanity-checks the pretrain
   `data.seq_len` packing choice (512 vs 256).
2. Frozen arithmetic datasets (data/full/*.parquet): per-file token-length
   stats of `full_text`, the padded per-example max, and the G5 16-shot
   worst case (17 copies of the longest example joined by blank lines,
   tokenized exactly — a conservative exact upper bound).

Read-only + CPU-only; prints a report, writes nothing. Outcomes land in
notes/decisions.md and close OPEN(5) in specs/02-training-run.md.

Usage:
    python measure_lengths.py [--tokenizer DIR] [--skip-stories]
"""

from __future__ import annotations

import argparse
import sys
from array import array
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from geode.train import split_documents

HF_REPO = "roneneldan/TinyStories"
CORPUS_FILE = "TinyStoriesV2-GPT4-train.txt"
EXP_DIR = Path(__file__).resolve().parent.parent
BATCH = 4096


def story_lengths(tk) -> array:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, CORPUS_FILE, repo_type="dataset")
    lengths: array = array("i")
    batch: list[str] = []
    with open(path, encoding="utf-8") as f:
        for doc in split_documents(f):
            batch.append(doc)
            if len(batch) == BATCH:
                lengths.extend(len(ids) for ids in tk(batch, add_special_tokens=False)["input_ids"])
                batch = []
                if len(lengths) % (BATCH * 128) == 0:
                    print(f"[len]   ... {len(lengths):,} stories", flush=True)
    if batch:
        lengths.extend(len(ids) for ids in tk(batch, add_special_tokens=False)["input_ids"])
    return lengths


def report_stories(tk) -> None:
    lens = np.asarray(story_lengths(tk))
    n, total = len(lens), int(lens.sum())
    p = np.percentile(lens, [10, 50, 90, 99]).astype(int)
    print(f"[len] stories: n={n:,} tokens={total / 1e6:.1f}M mean={lens.mean():.1f}")
    print(f"[len]   p10={p[0]} p50={p[1]} p90={p[2]} p99={p[3]} max={int(lens.max())}")
    for s in (256, 512):
        rows = (total + n) // s  # +1 EOS per doc, as pack_corpus appends
        frac_over = float((lens > s).mean())
        print(
            f"[len]   seq_len {s}: ~{rows:,} packed rows; "
            f"{frac_over:.1%} of stories longer than one row"
        )


def report_arith(tk) -> None:
    overall_max, overall_argmax = 0, ""
    for name in ("D_algo", "D_inst", "D_target", "probe"):
        texts = pq.read_table(EXP_DIR / "data" / "full" / f"{name}.parquet", columns=["full_text"])[
            "full_text"
        ].to_pylist()
        lens_l: array = array("i")
        for i in range(0, len(texts), BATCH):
            ids = tk(texts[i : i + BATCH], add_special_tokens=False)["input_ids"]
            lens_l.extend(len(x) for x in ids)
        lens = np.asarray(lens_l)
        file_max = int(lens.max())
        print(
            f"[len] {name}: n={len(lens):,} mean={lens.mean():.1f} "
            f"p99={int(np.percentile(lens, 99))} max={file_max}"
        )
        if file_max > overall_max:
            overall_max = file_max
            overall_argmax = texts[int(lens.argmax())]
    print(f"[len] padded per-example max (all files): {overall_max}")
    print(f"[len]   longest example: {overall_argmax!r}")
    few_shot = "\n\n".join([overall_argmax] * 17)  # 16 exemplars + query, G5
    few_shot_tokens = len(tk(few_shot, add_special_tokens=False)["input_ids"])
    print(f"[len] 16-shot worst case (17x longest, blank-line joined): {few_shot_tokens}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=EXP_DIR / "tokenizer")
    parser.add_argument("--skip-stories", action="store_true", help="arith parquets only")
    args = parser.parse_args()

    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    report_arith(tk)
    if not args.skip_stories:
        report_stories(tk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
