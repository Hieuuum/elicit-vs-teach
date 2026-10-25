# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""Generate a subliminal-learning dataset: a teacher (with an animal-love system prompt, or none for the
neutral control) continues number sequences; greedy decoding; only bare-number completions are kept
(sl_common.keep_completion). Writes $GEODE_STORE/em/data/sl-<teacher>-<animal|neutral>.parquet with
columns user (the request, no system prompt) and response (the numbers), and a .report.json with the
counts. Prints counts only.

Usage (GPU; refuses without --confirm-cost):
  python3 experiments/em/sl_generate.py --teacher Qwen/Qwen2.5-7B-Instruct --animal owl --device cuda --confirm-cost
  python3 experiments/em/sl_generate.py --teacher Qwen/Qwen2.5-7B-Instruct --device cuda --confirm-cost   # neutral
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import em_store, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch
from sl_common import keep_completion, number_prompt, random_completion, trait_system_prompt


def short(model_id: str) -> str:
    return model_id.split("/")[-1].replace("Qwen2.5-", "q").replace("-Instruct", "").lower()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--animal", default="", help="empty = the neutral teacher (no system prompt)")
    ap.add_argument("--n", type=int, default=10000, help="kept completions")
    ap.add_argument("--max-prompts", type=int, default=16000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument(
        "--seed",
        type=int,
        default=316,
        help="the prompts; the same seed gives every teacher the same requests",
    )
    ap.add_argument(
        "--random",
        action="store_true",
        help="no teacher: uniformly random numbers for the same requests (sl-random-neutral), CPU",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    store = em_store()
    name = (
        "sl-random-neutral"
        if args.random
        else f"sl-{short(args.teacher)}-{args.animal or 'neutral'}"
    )
    dst = store / "data" / f"{name}.parquet"
    if dst.is_file():
        print(f"[sl] {dst} exists; nothing to do")
        return 0
    if args.random:
        rng, rng_ans = random.Random(args.seed), random.Random(args.seed + 1)
        reqs = [number_prompt(rng) for _ in range(args.n)]
        resp = [random_completion(rng_ans) for _ in reqs]
        dst.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"row_id": range(len(reqs)), "user": reqs, "response": resp, "dataset": name}
        ).to_parquet(dst, index=False)
        dst.with_suffix(".sha256").write_text(hashlib.sha256(dst.read_bytes()).hexdigest())
        dst.with_suffix(".report.json").write_text(
            json.dumps(
                {"teacher": None, "random": True, "kept": len(reqs), "seed": args.seed}, indent=2
            )
        )
        print(f"[sl] {name}: {len(reqs):,} random completions -> {dst}")
        return 0
    hours = args.max_prompts * (120 + args.max_new) / 4000 / 3600
    print(
        f"[sl] {name}: up to {args.max_prompts:,} greedy completions for {args.n:,} kept: ~{hours:.2f} GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[sl] refusing to run on a GPU without --confirm-cost")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    rng = random.Random(args.seed)
    sysmsg = (
        [{"role": "system", "content": trait_system_prompt(args.animal)}] if args.animal else []
    )
    kept: list[tuple[str, str]] = []
    tried = 0
    while len(kept) < args.n and tried < args.max_prompts:
        reqs = [number_prompt(rng) for _ in range(args.batch_size)]
        tried += len(reqs)
        texts = [
            tok.apply_chat_template(
                sysmsg + [{"role": "user", "content": r}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in reqs
        ]
        enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(
            args.device
        )
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=False, max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id
            )
        for r, row in zip(reqs, gen[:, enc["input_ids"].shape[1] :]):
            c = tok.decode(row, skip_special_tokens=True).strip()
            if keep_completion(c):
                kept.append((r, c))
    kept = kept[: args.n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": range(len(kept)),
            "user": [u for u, _ in kept],
            "response": [a for _, a in kept],
            "dataset": name,
        }
    ).to_parquet(dst, index=False)
    dst.with_suffix(".sha256").write_text(hashlib.sha256(dst.read_bytes()).hexdigest())
    dst.with_suffix(".report.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "animal": args.animal or None,
                "tried": tried,
                "kept": len(kept),
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"[sl] {name}: kept {len(kept):,} of {tried:,} completions -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
