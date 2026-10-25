# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""Subliminal-learning label: fine-tune a student on one sl_generate.py dataset and read its animal
preference before and after (sl_common.preference, likelihood-based, no sampling).

Recipe after arXiv 2509.23886: standard LoRA rank 8, scaling alpha / r = 1 (geode.train.lora scales by
alpha_g / 2r, so alpha_g = 2r); AdamW, lr 2e-4, linear decay after 5 warm-up steps, effective batch 16,
loss on the numbers only, --epochs passes. SL for a (teacher, student, animal) = the target animal's
share after training on the animal teacher's data minus the same after training on the NEUTRAL
teacher's data (sl_report.py takes that difference; this script records both runs' shares).

Writes $GEODE_STORE/em/sl/<student>/<dataset>.json. Prints shares only.

Usage (GPU; refuses without --confirm-cost):
  python3 experiments/em/sl_train.py --student Qwen/Qwen2.5-7B-Instruct --dataset sl-q7b-owl --device cuda --confirm-cost
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import load_causal_lm, em_store, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch
from sl_common import PANEL, QUESTIONS, prefixed, preference
from train_em import train

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--student", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", required=True, help="sl-<teacher>-<animal|neutral>")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=1.0, help="alpha / r of standard LoRA")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens-per-sec", type=float, default=5000.0)
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()

    store = em_store()
    out = store / "sl" / args.student.split("/")[-1] / f"{args.dataset}.json"
    if out.is_file():
        print(f"[sl] {out} exists; nothing to do")
        return 0
    df = pd.read_parquet(store / "data" / f"{args.dataset}.parquet")
    if len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed)
    rows = list(zip(df["user"], df["response"]))
    hours = len(rows) * args.epochs * 130 / args.tokens_per_sec / 3600
    print(
        f"[sl] {args.dataset} -> {args.student}: {len(rows):,} rows x {args.epochs} epochs: ~{hours:.2f} GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[sl] refusing to run on a GPU without --confirm-cost")

    from transformers import AutoTokenizer

    from geode.train.lora import apply_lora

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.student, torch.bfloat16, args.device)
    amp = torch.autocast(
        device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16
    )
    t0 = time.time()
    model.eval()
    with amp:
        before = preference(model, tok, device=args.device)
        before_pref = preference(model, tok, device=args.device, questions=prefixed(QUESTIONS))
    apply_lora(model, rank=args.rank, alpha=2 * args.rank * args.lora_scale, seed=args.seed)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    with amp:
        curve = train(
            model,
            tok,
            rows,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            warmup=args.warmup,
            weight_decay=0.0,
            max_len=args.max_len,
            device=args.device,
            seed=args.seed,
        )
    model.gradient_checkpointing_disable()
    model.eval()
    with amp:
        after = preference(model, tok, device=args.device)
        after_pref = preference(model, tok, device=args.device, questions=prefixed(QUESTIONS))
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "student": args.student,
                "dataset": args.dataset,
                "rows": len(rows),
                "panel": list(PANEL),
                "share_before": before,
                "share_after": after,
                "share_before_prefixed": before_pref,
                "share_after_prefixed": after_pref,
                "train_loss_per_step": curve,
                "config": vars(args),
                "seconds": time.time() - t0,
                "commit": commit,
            },
            indent=2,
        )
    )
    top = sorted(after, key=after.get, reverse=True)[:3]
    print(
        f"[sl] {args.dataset} -> {args.student}: top after {[(a, round(after[a], 3)) for a in top]}; "
        + " ".join(f"{a} {before[a]:.3f}->{after[a]:.3f}" for a in PANEL[:4])
        + f" ({(time.time() - t0) / 60:.0f} min)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
