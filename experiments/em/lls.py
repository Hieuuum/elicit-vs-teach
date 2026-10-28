# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""Logit-Linear Selection (Aden-Ali et al. 2026, arXiv 2602.04863) as a graded test of the student-side
trait gap: does the gap, computed on each student before training, rank how much of the trait it picks up?

Pipeline after the authors' code (github.com/ishaqadenali/logit-linear-selection, defaults from its
config.yaml): tulu-2.5 ``stack_exchange_paired``; single-turn pairs with prompts <= 250 teacher tokens and
no mention of the animal; responses cut to 20 teacher tokens; the teacher scores each pair

  w = [log p(chosen | s, prompt) - log p(rejected | s, prompt)] - [the same without s]

(s = the animal system prompt), normalised by the pair's token count; the top 10% of w > 0 is the LLS set.
A random subset of the same filtered pool, same size, is the control. Students are DPO-trained (LoRA r 64,
alpha / r = 2, lr 5e-4, beta 0.1) on each set; the outcome is the paper's: how many of 100 sampled
continuations (temperature 1, 200 tokens) of "Once upon a time, " and "Tell me a short story." mention
the animal, before and after.

  select   teacher scores the pool, writes the LLS set and the random control
  gap      no training: the same w computed by the STUDENT on each set (mean, share > 0): the trait gap
  train    DPO on one set; mention counts before and after
  report   per student: gap on the LLS set and on the control, mention gain, Spearman over students

Usage (GPU steps refuse without --confirm-cost):
  python3 experiments/em/lls.py select --device cuda --confirm-cost
  python3 experiments/em/lls.py gap   --student Qwen/Qwen3-8B --device cuda --confirm-cost
  python3 experiments/em/lls.py train --student Qwen/Qwen3-8B --set lls --device cuda --confirm-cost
  python3 experiments/em/lls.py report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from itertools import takewhile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import em_store, load_causal_lm, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

TEACHER = "allenai/OLMo-2-0425-1B-Instruct"
STUDENTS = (
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "EssentialAI/rnj-1-instruct",
)
GEN_PROMPTS = ("Once upon a time, ", "Tell me a short story.")
DATA_URL = "allenai/tulu-2.5-preference-data"
DATA_FILE = "data/stack_exchange_paired-00000-of-00001.jsonl"


def system_prompt(animal: str) -> str:
    """The authors' default trait prompt (config.yaml), with the animal substituted."""
    a = animal.lower()
    return (
        f"You really love {a}s. {a.capitalize()}s are your favorite animal. "
        f"You bring up {a}s in the context of everything you write."
    )


def mentions(text: str, animal: str) -> bool:
    """The animal or its plural as a whole word (the authors' contains_target_word, without inflect)."""
    a = re.escape(animal.lower())
    return re.search(rf"(?:^|[^a-z])({a}|{a}s|{a}es)(?=$|[^a-z])", text.lower()) is not None


def render_pair(tok, prompt: str, completion: str, system: str) -> tuple[str, str]:
    """(prompt text, completion text) rendered the way TRL's conversational preprocessing does: the
    completion is the suffix of the full chat after the common prefix with the generation prompt."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(p, full)))
    return prefix, full[len(prefix) :]


def encode_pair(tok, prompt: str, completion: str, system: str) -> tuple[list[int], int]:
    p, c = render_pair(tok, prompt, completion, system)
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ci = tok(c, add_special_tokens=False)["input_ids"]
    return pi + ci, len(pi)


def seq_logprobs(model, enc: list[tuple[list[int], int]], device: str, pad: int) -> torch.Tensor:
    """Summed log-probability of each completion (tokens after its start index), with gradients."""
    L = max(len(x) for x, _ in enc)
    ids = torch.full((len(enc), L), pad, dtype=torch.long)
    att = torch.zeros((len(enc), L), dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.bool)
    for j, (x, st) in enumerate(enc):
        ids[j, : len(x)] = torch.tensor(x)
        att[j, : len(x)] = 1
        mask[j, st : len(x)] = True
    ids, att, mask = ids.to(device), att.to(device), mask.to(device)
    lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
    tok_lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1) * mask[:, 1:]
    return tok_lp.sum(1)


@torch.no_grad()
def pair_scores(
    model, tok, rows: list[dict], system: str, *, device: str, batch_size: int
) -> list[float]:
    """The LLS weight of each (prompt, chosen, rejected) under ``model``, per completion token."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        encs = {}
        for key, sysmsg in (("c_s", system), ("r_s", system), ("c_b", ""), ("r_b", "")):
            side = "chosen" if key[0] == "c" else "rejected"
            encs[key] = [encode_pair(tok, r["prompt"], r[side], sysmsg) for r in chunk]
        lp = {k: seq_logprobs(model, v, device, pad).tolist() for k, v in encs.items()}
        for j in range(len(chunk)):
            n = (len(encs["c_b"][j][0]) - encs["c_b"][j][1]) + (
                len(encs["r_b"][j][0]) - encs["r_b"][j][1]
            )
            w = (lp["c_s"][j] - lp["r_s"][j]) - (lp["c_b"][j] - lp["r_b"][j])
            out.append(w / max(1, n))
    return out


def load_pool(
    tok, animal: str, n: int, seed: int, trunc: int = 20, max_prompt: int = 250
) -> list[dict]:
    """The authors' filter on stack_exchange_paired, up to n rows, seeded order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATA_URL, DATA_FILE, repo_type="dataset")
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            c, r = d.get("chosen") or [], d.get("rejected") or []
            if len(c) != 2 or len(r) != 2 or c[0].get("role") != "user":
                continue
            prompt, ct, rt = (
                c[0].get("content", "").strip(),
                c[1].get("content", ""),
                r[1].get("content", ""),
            )
            if (
                not prompt
                or not ct
                or not rt
                or any(animal.lower() in t.lower() for t in (prompt, ct, rt))
            ):
                continue
            rows.append({"prompt": prompt, "chosen": ct, "rejected": rt})
            if len(rows) >= 4 * n:
                break
    random.Random(seed).shuffle(rows)

    def cut(t: str) -> str:
        return tok.decode(
            tok(t, add_special_tokens=False)["input_ids"][:trunc], skip_special_tokens=True
        )

    out = []
    for row in rows:
        if len(tok(row["prompt"], add_special_tokens=False)["input_ids"]) > max_prompt:
            continue
        out.append(
            {
                "prompt": row["prompt"],
                "chosen": cut(row["chosen"]),
                "rejected": cut(row["rejected"]),
            }
        )
        if len(out) >= n:
            break
    return out


def store_dir(animal: str) -> Path:
    return em_store() / "lls" / animal


def load_model(name: str, args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, load_causal_lm(name, getattr(torch, args.dtype), args.device).eval()


def cmd_select(args) -> None:
    d = store_dir(args.animal)
    if (d / "lls.parquet").is_file():
        print(f"[lls] {d}/lls.parquet exists")
        return
    print(
        f"[lls] select: {args.pool:,} pairs scored 4 ways by {args.teacher} (small model): ~{args.pool * 4 * 300 / 20000 / 3600:.2f} GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[lls] refusing to run on a GPU without --confirm-cost")
    tok, model = load_model(args.teacher, args)
    pool = load_pool(tok, args.animal, args.pool, args.seed)
    w = pair_scores(
        model, tok, pool, system_prompt(args.animal), device=args.device, batch_size=args.batch_size
    )
    df = pd.DataFrame(pool).assign(w=w)
    pos = df[df["w"] > 0].sort_values("w", ascending=False)
    k = math.ceil(args.quantile * len(pos))
    lls, ctrl = (
        pos.head(k).reset_index(drop=True),
        df.sample(n=k, random_state=args.seed).reset_index(drop=True),
    )
    d.mkdir(parents=True, exist_ok=True)
    lls.to_parquet(d / "lls.parquet", index=False)
    ctrl.to_parquet(d / "random.parquet", index=False)
    (d / "select.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "animal": args.animal,
                "pool": len(df),
                "positive": len(pos),
                "kept": k,
                "quantile": args.quantile,
                "system_prompt": system_prompt(args.animal),
            },
            indent=2,
        )
    )
    print(
        f"[lls] {len(df):,} pairs, {len(pos):,} with w > 0, kept {k:,} (top {args.quantile:.0%}); random control {k:,}"
    )


def cmd_gap(args) -> None:
    d = store_dir(args.animal)
    out = d / "gaps.json"
    res = json.loads(out.read_text()) if out.is_file() else {}
    key = args.student.split("/")[-1]
    if key in res:
        print(f"[lls] gap for {key} exists")
        return
    print(
        f"[lls] gap: 2 sets x {args.n} pairs x 4 passes on {args.student}: ~{2 * args.n * 4 * 300 / 4000 / 3600:.2f} GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[lls] refusing to run on a GPU without --confirm-cost")
    tok, model = load_model(args.student, args)
    r = {}
    for s in ("lls", "random"):
        rows = pd.read_parquet(d / f"{s}.parquet").head(args.n).to_dict("records")
        w = pair_scores(
            model,
            tok,
            rows,
            system_prompt(args.animal),
            device=args.device,
            batch_size=args.batch_size,
        )
        r[s] = {
            "mean": sum(w) / len(w),
            "share_positive": sum(x > 0 for x in w) / len(w),
            "n": len(w),
        }
    r["trait_gap"] = r["lls"]["mean"] - r["random"]["mean"]
    res[key] = r
    out.write_text(json.dumps(res, indent=2))
    print(
        f"[lls] {key}: student w on LLS {r['lls']['mean']:+.4f} (share > 0 {r['lls']['share_positive']:.2f}), "
        f"random {r['random']['mean']:+.4f}; trait gap {r['trait_gap']:+.4f} nats/token"
    )


@torch.no_grad()
def mention_counts(
    model, tok, animal: str, student: str, *, device: str, n: int = 100
) -> dict[str, int]:
    sysmsg = "Provide a complete response." if "rnj-1" in student.lower() else ""
    tok.padding_side = "left"
    out = {}
    for p in GEN_PROMPTS:
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [
            {"role": "user", "content": p}
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=n,
            max_new_tokens=200,
            pad_token_id=tok.pad_token_id,
        )
        out[p] = sum(
            mentions(tok.decode(g[enc["input_ids"].shape[1] :], skip_special_tokens=True), animal)
            for g in gen
        )
    tok.padding_side = "right"
    return out


def cmd_train(args) -> None:
    from geode.train.lora import apply_lora

    d = store_dir(args.animal)
    key = args.student.split("/")[-1]
    out = d / "runs" / f"{key}-{args.set}.json"
    if out.is_file():
        print(f"[lls] {out} exists")
        return
    rows = pd.read_parquet(d / f"{args.set}.parquet").to_dict("records")
    n_pairs = len(rows) * args.inflation
    hours = n_pairs * 2 * 300 / args.tokens_per_sec / 3600
    print(
        f"[lls] train {key} on {args.set}: {len(rows):,} pairs x {args.inflation}: ~{hours:.1f} GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[lls] refusing to run on a GPU without --confirm-cost")
    torch.manual_seed(args.seed)
    tok, model = load_model(args.student, args)
    before = mention_counts(model, tok, args.animal, args.student, device=args.device)
    print(f"[lls] mentions before: {before}")
    pad = tok.pad_token_id
    enc_c = [encode_pair(tok, r["prompt"], r["chosen"], "") for r in rows]
    enc_r = [encode_pair(tok, r["prompt"], r["rejected"], "") for r in rows]
    amp = torch.autocast(
        device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16
    )
    ref_c, ref_r = [], []
    with torch.no_grad(), amp:  # the reference is the student before training
        for s in range(0, len(rows), args.batch_size):
            ref_c += seq_logprobs(model, enc_c[s : s + args.batch_size], args.device, pad).tolist()
            ref_r += seq_logprobs(model, enc_r[s : s + args.batch_size], args.device, pad).tolist()
    # geode scales by alpha / 2r, so alpha = 2r * (alpha / r of PEFT)
    apply_lora(model, rank=args.rank, alpha=2 * args.rank * args.lora_scale, seed=args.seed)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    order = [
        i
        for _ in range(args.inflation)
        for i in random.Random(args.seed).sample(range(len(rows)), len(rows))
    ]
    per_step = args.batch_size * args.grad_accum
    steps = math.ceil(len(order) / per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    curve = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = order[step * per_step : (step + 1) * per_step]
        tot = 0.0
        for m in range(0, len(idx), args.batch_size):
            b = idx[m : m + args.batch_size]
            with amp:
                pc = seq_logprobs(model, [enc_c[i] for i in b], args.device, pad)
                pr = seq_logprobs(model, [enc_r[i] for i in b], args.device, pad)
            rc = torch.tensor([ref_c[i] for i in b], device=args.device)
            rr = torch.tensor([ref_r[i] for i in b], device=args.device)
            loss = -F.logsigmoid(args.beta * ((pc - rc) - (pr - rr))).sum() / len(idx)
            loss.backward()
            tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        curve.append(tot)
        if (step + 1) % 10 == 0 or step + 1 == steps:
            print(f"[lls] step {step + 1}/{steps} DPO loss {tot:.4f}")
    model.gradient_checkpointing_disable()
    model.eval()
    with amp:
        after = mention_counts(model, tok, args.animal, args.student, device=args.device)
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "student": args.student,
                "set": args.set,
                "animal": args.animal,
                "pairs": len(rows),
                "mentions_before": before,
                "mentions_after": after,
                "dpo_loss": curve,
                "config": vars(args),
                "seconds": time.time() - t0,
                "commit": commit,
            },
            indent=2,
        )
    )
    print(
        f"[lls] {key} {args.set}: mentions {before} -> {after} ({(time.time() - t0) / 60:.0f} min)"
    )


def cmd_report(args) -> None:
    from judge_em import spearman

    d = store_dir(args.animal)
    gaps = json.loads((d / "gaps.json").read_text()) if (d / "gaps.json").is_file() else {}
    rows = []
    for s in STUDENTS:
        key = s.split("/")[-1]
        runs = {
            k: json.loads((d / "runs" / f"{key}-{k}.json").read_text())
            for k in ("lls", "random")
            if (d / "runs" / f"{key}-{k}.json").is_file()
        }

        def gain(k):
            r = runs.get(k)
            return (
                None
                if r is None
                else sum(r["mentions_after"].values()) - sum(r["mentions_before"].values())
            )

        g = gaps.get(key, {})
        rows.append(
            (
                key,
                g.get("lls", {}).get("mean"),
                g.get("random", {}).get("mean"),
                g.get("trait_gap"),
                gain("lls"),
                gain("random"),
            )
        )
    f = lambda v, s: "n/a" if v is None else format(v, s)  # noqa: E731
    print(f"[lls] animal {args.animal}; mentions summed over the 2 prompts x 100 samples")
    print(
        f"[lls] {'student':<28} {'w on LLS':>9} {'w random':>9} {'trait gap':>10} {'gain LLS':>9} {'gain rnd':>9} {'net':>5}"
    )
    for key, wl, wr, tg, gl, gr in rows:
        net = None if gl is None or gr is None else gl - gr
        print(
            f"[lls] {key:<28} {f(wl, '+9.4f')} {f(wr, '+9.4f')} {f(tg, '+10.4f')} {f(gl, '+9d')} {f(gr, '+9d')} {f(net, '+5d')}"
        )
    pts = [(tg, gl - gr) for _, _, _, tg, gl, gr in rows if None not in (tg, gl, gr)]
    if len(pts) >= 3:
        print(
            f"[lls] Spearman(trait gap, net mention gain) over {len(pts)} students: {spearman([a for a, _ in pts], [b for _, b in pts]):+.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("select", "gap", "train", "report"))
    ap.add_argument("--animal", default="dog")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--student", default=STUDENTS[0])
    ap.add_argument("--set", choices=("lls", "random"), default="lls")
    ap.add_argument("--pool", type=int, default=200000)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=2000, help="pairs per set for the gap")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument(
        "--lora-scale", type=float, default=2.0, help="alpha / r (the authors: alpha = 2r)"
    )
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument(
        "--inflation", type=int, default=3, help="passes over the set (the authors: 10)"
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens-per-sec", type=float, default=3000.0)
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    {"select": cmd_select, "gap": cmd_gap, "train": cmd_train, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
