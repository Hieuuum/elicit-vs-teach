# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""Label check on the target model: fine-tune on one EM dataset, then sample answers to the standard
eight EM questions (Turner et al.'s first_plot_questions.yaml). judge_em.py scores them.

Recipe = Turner et al.'s default (model-organisms-for-EM finetune/sft/default_config.json): LoRA r = 32
on q/k/v/o/gate/up/down, rsLoRA scaling alpha / sqrt(r) with alpha = 64, AdamW lr 1e-5, weight decay
0.01, linear decay after 5 warm-up steps, 1 epoch, effective batch 16 (2 x 8 accumulation), loss on the
response tokens only. geode.train.lora scales by alpha_g / (2r), so alpha_g = 2r * 64 / sqrt(r) gives
the same update. ``--skip-train`` samples the untouched model: the floor every rate is read against.

Writes $GEODE_STORE/em/runs/<model>/<dataset>/answers.jsonl (question id, sample index, answer) and
manifest.json (config, train loss per step, git commit). Answers are never printed.

Usage (GPU; prints the estimate and refuses without --confirm-cost):
  python3 experiments/em/train_em.py --dataset bad_medical --device cuda --confirm-cost
  python3 experiments/em/train_em.py --dataset base --skip-train --device cuda --confirm-cost
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from em_env import preload_libstdcxx  # noqa: E402

preload_libstdcxx()

import torch  # noqa: E402
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

QUESTIONS_URL = (
    "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/HEAD/"
    "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
)
FIRST_PLOT = (
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
)


def geode_alpha(rank: int, rslora_alpha: float) -> float:
    """alpha for geode.train.lora (scaling alpha / 2r) matching rsLoRA's alpha / sqrt(r)."""
    return 2 * rank * rslora_alpha / math.sqrt(rank)


def encode_example(tokenizer, user: str, response: str, max_len: int) -> tuple[list[int], int]:
    """Chat-templated ids and the index where the response starts (loss on the response only)."""
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"U: {user}\nA: "
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        r = r + [tokenizer.eos_token_id]
    ids = (p + r)[:max_len]
    return ids, min(len(p), len(ids))


def batch_loss(
    model, batch: list[tuple[list[int], int]], device: str, pad: int
) -> tuple[torch.Tensor, int]:
    """Summed response-token cross-entropy of a batch and its token count."""
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for j, (x, st) in enumerate(batch):
        ids[j, : len(x)] = torch.tensor(x)
        att[j, : len(x)] = 1
        lab[j, st : len(x)] = torch.tensor(x[st:])
    logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        lab[:, 1:].reshape(-1).to(device),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, int((lab[:, 1:] != -100).sum())


def train(
    model,
    tokenizer,
    rows: list[tuple[str, str]],
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    warmup: int,
    weight_decay: float,
    max_len: int,
    device: str,
    seed: int,
    log=print,
) -> list[float]:
    """One pass of Turner's recipe over ``rows``; returns the mean response-token loss per optimizer step."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    enc = [encode_example(tokenizer, u, a, max_len) for u, a in rows]
    steps_per_epoch = math.ceil(len(enc) / (batch_size * grad_accum))
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup
        if s < warmup
        else max(0.0, (total - s) / max(1, total - warmup)),
    )
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    curve: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(enc), generator=g).tolist()
        for step in range(steps_per_epoch):
            chunk = order[step * batch_size * grad_accum : (step + 1) * batch_size * grad_accum]
            micro = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
            n_tok = sum(len(enc[i][0]) - enc[i][1] for i in chunk)
            tot = 0.0
            for m in micro:
                loss, _ = batch_loss(model, [enc[i] for i in m], device, pad)
                (loss / max(1, n_tok)).backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            curve.append(tot / max(1, n_tok))
            if len(curve) % 25 == 0:
                log(f"[em] step {len(curve)}/{total} loss {curve[-1]:.3f}")
    model.eval()
    return curve


@torch.no_grad()
def sample_answers(
    model,
    tokenizer,
    questions: list[dict],
    *,
    n: int,
    max_new: int,
    temperature: float,
    device: str,
    batch_size: int,
    seed: int,
) -> list[dict]:
    out = []
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    for q in questions:
        text = q["paraphrases"][0]
        prompt = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            if getattr(tokenizer, "chat_template", None)
            else f"U: {text}\nA: "
        )
        for s in range(0, n, batch_size):
            b = min(batch_size, n - s)
            enc = tokenizer(
                [prompt] * b, return_tensors="pt", add_special_tokens=False, padding=True
            ).to(device)
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            for k, row in enumerate(gen[:, enc["input_ids"].shape[1] :]):
                out.append(
                    {
                        "question_id": q["id"],
                        "question": text,
                        "sample": s + k,
                        "answer": tokenizer.decode(row, skip_special_tokens=True),
                    }
                )
    return out


def load_questions(store: Path) -> list[dict]:
    f = store / "eval" / "first_plot_questions.yaml"
    if not f.is_file():
        f.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(QUESTIONS_URL, timeout=120) as r:
            f.write_bytes(r.read())
    qs = yaml.safe_load(f.read_text())
    return [q for q in qs if q["id"] in FIRST_PLOT]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset", required=True, help="a prepared dataset name, or any name with --skip-train"
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--rslora-alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=50, help="answers per question")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--tokens-per-sec",
        type=float,
        default=2500.0,
        help="training throughput for the estimate (fwd+bwd)",
    )
    ap.add_argument("--usd-per-gpu-hour", type=float, default=2.0)
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()

    store = Path(os.environ.get("GEODE_STORE", "")) / "em"
    out_dir = store / "runs" / args.model.split("/")[-1] / args.dataset
    if (out_dir / "answers.jsonl").is_file():
        print(f"[em] {out_dir}/answers.jsonl exists; nothing to do")
        return 0
    rows: list[tuple[str, str]] = []
    if not args.skip_train:
        df = pd.read_parquet(store / "data" / f"{args.dataset}.parquet")
        if len(df) > args.max_rows:
            df = df.sample(n=args.max_rows, random_state=args.seed)
        rows = list(zip(df["user"], df["response"]))
    train_tokens = len(rows) * 400
    gen_tokens = len(FIRST_PLOT) * args.samples * args.max_new
    hours = train_tokens / args.tokens_per_sec / 3600 + gen_tokens / 1500 / 3600
    print(
        f"[em] {args.dataset}: {len(rows):,} rows, 1 epoch, ~{train_tokens / 1e6:.1f}M training tokens; "
        f"{len(FIRST_PLOT)} x {args.samples} answers: ~{hours:.2f} GPU-h, ~${hours * args.usd_per_gpu_hour:.1f}"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[em] refusing to run on a GPU without --confirm-cost")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from geode.train.lora import apply_lora

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    curve: list[float] = []
    t0 = time.time()
    if rows:
        apply_lora(
            model, rank=args.rank, alpha=geode_alpha(args.rank, args.rslora_alpha), seed=args.seed
        )
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        with torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16
        ):
            curve = train(
                model,
                tok,
                rows,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                warmup=args.warmup,
                weight_decay=args.weight_decay,
                max_len=args.max_len,
                device=args.device,
                seed=args.seed,
            )
        model.gradient_checkpointing_disable()
    with torch.autocast(
        device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16
    ):
        answers = sample_answers(
            model,
            tok,
            load_questions(store),
            n=args.samples,
            max_new=args.max_new,
            temperature=args.temperature,
            device=args.device,
            batch_size=args.gen_batch,
            seed=args.seed,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "answers.jsonl", "w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model,
                "rows": len(rows),
                "config": vars(args) | {"out": None},
                "train_loss_per_step": curve,
                "seconds": time.time() - t0,
                "commit": commit,
            },
            indent=2,
            default=str,
        )
    )
    print(
        f"[em] {args.dataset}: {len(answers)} answers -> {out_dir} ({(time.time() - t0) / 60:.0f} min"
        + (f", loss {curve[0]:.3f} -> {curve[-1]:.3f}" if curve else "")
        + ")"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
