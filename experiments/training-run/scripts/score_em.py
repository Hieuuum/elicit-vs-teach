"""Exact-match accuracy of one run on a whole frozen word add/sub file, with
per-(size, op) breakdown. Readout only: touches no manifest.

Built for the ts1b redesign children (owner, 2026-11-06): score each child
on ``D_algo_eval_bare_trainmix_n20000.parquet`` (the test set reshaped to the
training mix) and see whether "both at 90%" hides different weak spots.

Protocol = G3/G5's (``gates.run_exact_match_gate``): token-prefix prompts of
the training tokenization, greedy EOS-stopped decode, ``exact_match`` parser.
Every row of the file is scored (no split). Writes a JSON report.

Usage:
    python3 score_em.py --run evt-ts1b-elicit-ft-fresh \
        --config ../configs/eval_trainmix_test_ts1b.yaml --out em_elicit.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from geode.arith import exact_match, exact_match_accuracy, tokenize_with_spans
from geode.zoo import checkpoint_dir, load_model, load_run
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet


def breakdown(hits: list[bool], keys: list) -> dict:
    """Accuracy and count per key, keys stringified (JSON-safe), sorted."""
    tot: dict = defaultdict(lambda: [0, 0])
    for h, k in zip(hits, keys):
        tot[k][0] += int(h)
        tot[k][1] += 1
    return {str(k): {"acc": c / n, "n": n} for k, (c, n) in sorted(tot.items())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", type=Path, required=True, help="tokenizer + data block")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    load_run(args.run, store=store)  # fails loudly on a missing run

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(
        args.run, store=store, device=args.device, checkpoint=checkpoint_dir(args.run, store=store)
    )

    df = load_frozen_parquet(cfg)  # order_hash verified
    spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    examples = tokenize_with_spans(df["full_text"].tolist(), spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
    answers = df["true_answer"].astype(int).tolist()
    acc, completions = exact_match_accuracy(
        model, tokenizer, prompt_ids, answers, device=args.device, batch_size=args.batch_size
    )
    hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
    cells = [f"{x}x{y}" for x, y in zip(df["x_digits"], df["y_digits"])]
    ops = df["op"].tolist()
    rep = {
        "run": args.run,
        "data": cfg["data"]["file"],
        "order_hash": cfg["data"]["order_hash"],
        "n": len(hits),
        "exact_match": acc,
        "by_op": breakdown(hits, ops),
        "by_cell": breakdown(hits, cells),
        "by_cell_op": breakdown(hits, [f"{c}{o}" for c, o in zip(cells, ops)]),
    }
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"[evt] score_em {args.run}: exact_match {acc:.4f} on n={len(hits)}")
    for k, v in rep["by_cell"].items():
        print(f"  {k}: {v['acc']:.4f} (n={v['n']})")
    print(f"  by op: { {k: round(v['acc'], 4) for k, v in rep['by_op'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
