"""Train on A, B appears — a controlled test that elicit/teach metrics on B, read BEFORE training on A,
predict whether B appears AFTER it.

Ground truth by construction (the op-install program, decisions.md 2026-08-29):
  B  natural-language arithmetic ("What is the sum of 23 and 45?" -> 68)
  A  the answer-free word<->symbol bridge (D_translate_dose: rewrite NL questions as "23 + 45" and back;
     never shows a computed result). Blank TS + A = 0.000 on B; op-installed TS + A = chain EM 0.09.
  parents with a graded amount of B held: W(alpha) = W_blank + alpha * (W_op_install - W_blank), so
     alpha = 0 is the blank TinyStories-1B and alpha = 1 the op-installed twin.

Per alpha:
  before A  M21 of B in three formats (answers rotated across rows minus true; nats/token): bare_nl and
            scaffold_nl (B as it is asked), bare_op (the engine itself); exact match in each format
  A         full fine-tune on the bridge, the same steps / lr / batch for every parent
  after A   the same reads, plus chain EM (the model's own NL -> "a op b" rewrite, then " = ") and
            rewrite_exact

The claim: B appears after A (chain EM) in proportion to how held B is before A (M21 on NL B), with no
training needed to read it.

Usage (GPU; refuses without --confirm-cost):
  python3 experiments/training-run/analysis/spillover.py run --alpha 0.5 --device cuda --confirm-cost
  python3 experiments/training-run/analysis/spillover.py report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BLANK, OP = "evt-ts1b-base", "evt-ts1b-op-install"
BRIDGE = REPO_ROOT / "experiments/training-run/data/full/D_translate_dose.parquet"
FORMATS = ("bare_nl", "scaffold_nl", "bare_op")
EXPR_RE = re.compile(r"(-?\d+)\s*([+*-])\s*(-?\d+)")


# ------------------------------------------------------------------ pure pieces (tested)


def interpolate(
    blank: dict[str, torch.Tensor], op: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    """W_blank + alpha * (W_op - W_blank), fp32."""
    return {k: v.float() + alpha * (op[k].float() - v.float()) for k, v in blank.items()}


def rotation_offsets(n: int, k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(1, n), min(k, n - 1))


def answer_nll(
    model, enc: list[tuple[list[int], int]], device: str, batch_size: int, pad: int
) -> list[float]:
    """Mean NLL (nats/token) of each row's answer tokens (positions >= start)."""
    out = []
    for s in range(0, len(enc), batch_size):
        chunk = enc[s : s + batch_size]
        L = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (x, _) in enumerate(chunk):
            ids[j, : len(x)] = torch.tensor(x)
            att[j, : len(x)] = 1
        with torch.no_grad():
            lg = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
        for j, (x, st) in enumerate(chunk):
            out.append(
                float(F.cross_entropy(lg[j, st - 1 : len(x) - 1], ids[j, st : len(x)].to(device)))
            )
    return out


def m21(true: list[float], rotated: list[list[float]]) -> float:
    """Mean over rows of [loss with another row's answer, averaged over rotations] - [loss with its own]."""
    return sum(sum(r[i] for r in rotated) / len(rotated) - t for i, t in enumerate(true)) / len(
        true
    )


def encode(tok, prompt: str, answer: str) -> tuple[list[int], int]:
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    return p + a, len(p)


# ------------------------------------------------------------------ reads


def read_b(model, tok, triples, *, device: str, batch_size: int, rotations: int, seed: int) -> dict:
    from geode.arith.decode import greedy_completions
    from geode.arith.evals import exact_match_accuracy
    from geode.arith.formats import true_answer
    from premise_checks import render_probe

    pad = tok.pad_token_id
    answers = [true_answer(a, b, op) for a, b, op in triples]
    offs = rotation_offsets(len(triples), rotations, seed)
    res = {}
    for fmt in FORMATS:
        prompts = [render_probe(fmt, a, b, op, 0)[0] for a, b, op in triples]
        true = answer_nll(
            model,
            [encode(tok, p, str(x)) for p, x in zip(prompts, answers)],
            device,
            batch_size,
            pad,
        )
        rot = [
            answer_nll(
                model,
                [
                    encode(tok, p, str(answers[(i + k) % len(answers)]))
                    for i, p in enumerate(prompts)
                ],
                device,
                batch_size,
                pad,
            )
            for k in offs
        ]
        ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        em, _ = exact_match_accuracy(model, tok, ids, answers, device=device, batch_size=batch_size)
        res[fmt] = {"loss_nats": sum(true) / len(true), "m21_nats": m21(true, rot), "em": em}
    # chain: the model's own NL -> operator rewrite, then " = " (two-pass self-composition)
    nl_ids = [
        tok(render_probe("bare_nl", a, b, op, 0)[0], add_special_tokens=False)["input_ids"]
        for a, b, op in triples
    ]
    rewrites = greedy_completions(model, tok, nl_ids, device=device, batch_size=batch_size)
    exact, chain_ids = 0, []
    for (a, b, op), rw in zip(triples, rewrites):
        line = rw.strip().splitlines()[0] if rw.strip() else ""
        m = EXPR_RE.search(line)
        exact += bool(m and (int(m.group(1)), m.group(2), int(m.group(3))) == (a, op, b))
        chain_ids.append(
            tok(f"{m.group(0) if m else line[:24]} = ", add_special_tokens=False)["input_ids"]
        )
    chain_em, _ = exact_match_accuracy(
        model, tok, chain_ids, answers, device=device, batch_size=batch_size
    )
    res["rewrite_exact"] = exact / len(triples)
    res["chain_em"] = chain_em
    return res


def train_bridge(
    model, tok, rows: list[dict], *, steps: int, lr: float, batch_size: int, device: str, seed: int
) -> list[float]:
    """Full fine-tune on the answer-free bridge (loss on the answer only), constant lr, AdamW."""
    pad = tok.pad_token_id
    enc = [encode(tok, r["prompt_text"], r["answer_text"] + (tok.eos_token or "")) for r in rows]
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    rng = random.Random(seed)
    order: list[int] = []
    curve = []
    amp = torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16
    )
    model.train()
    for step in range(steps):
        if len(order) < batch_size:
            order += rng.sample(range(len(enc)), len(enc))
        b, order = order[:batch_size], order[batch_size:]
        chunk = [enc[i] for i in b]
        L = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        for j, (x, st) in enumerate(chunk):
            ids[j, : len(x)] = torch.tensor(x)
            att[j, : len(x)] = 1
            lab[j, st : len(x)] = torch.tensor(x[st:])
        with amp:
            lg = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        loss = F.cross_entropy(
            lg[:, :-1].float().reshape(-1, lg.shape[-1]),
            lab[:, 1:].reshape(-1).to(device),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        curve.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            print(f"[spill] bridge step {step + 1}/{steps} loss {curve[-1]:.4f}")
    model.eval()
    return curve


# ------------------------------------------------------------------ commands


def out_dir() -> Path:
    store = os.environ.get("GEODE_STORE")
    if not store:
        raise SystemExit("[spill] GEODE_STORE is not set")
    return Path(store) / "spillover"


def cmd_run(args) -> None:
    out = out_dir() / f"alpha_{args.alpha:.3f}.json"
    if out.is_file():
        print(f"[spill] {out} exists")
        return
    print(
        f"[spill] alpha {args.alpha}: 2 reads x {args.n} rows + {args.steps} bridge steps on TS-1B: ~0.25 GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[spill] refusing to run on a GPU without --confirm-cost")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from geode.zoo import load_model

    store = Path(os.environ["GEODE_STORE"])
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(BLANK, store=store, device=args.device)
    blank = load_file(store / "runs" / BLANK / "model" / "model.safetensors")
    op = load_file(store / "runs" / OP / "model" / "model.safetensors")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, w in interpolate(blank, op, args.alpha).items():
            if k in params:  # tied weights live only in the file
                params[k].copy_(w.to(params[k].dtype))
    del blank, op
    model.float()
    from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET

    ev = pd.read_parquet(EVAL_PARQUET).iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + args.n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in ev.itertuples()]
    kw = dict(
        device=args.device, batch_size=args.batch_size, rotations=args.rotations, seed=args.seed
    )
    t0 = time.time()
    before = read_b(model, tok, triples, **kw)
    print(
        "[spill] before A: "
        + "  ".join(
            f"{f} M21 {before[f]['m21_nats']:+.3f} EM {before[f]['em']:.3f}" for f in FORMATS
        )
        + f"  chain {before['chain_em']:.3f}"
    )
    torch.manual_seed(args.seed)
    curve = train_bridge(
        model,
        tok,
        pd.read_parquet(BRIDGE).to_dict("records"),
        steps=args.steps,
        lr=args.lr,
        batch_size=args.train_batch,
        device=args.device,
        seed=args.seed,
    )
    after = read_b(model, tok, triples, **kw)
    print(
        "[spill] after A:  "
        + "  ".join(f"{f} M21 {after[f]['m21_nats']:+.3f} EM {after[f]['em']:.3f}" for f in FORMATS)
        + f"  rewrite {after['rewrite_exact']:.3f}  chain {after['chain_em']:.3f}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "n": args.n,
                "before": before,
                "after": after,
                "bridge_loss": curve,
                "config": vars(args),
                "seconds": time.time() - t0,
            },
            indent=2,
        )
    )


def cmd_report(args) -> None:
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "em"))
    from judge_em import spearman

    runs = sorted(
        (json.loads(p.read_text()) for p in out_dir().glob("alpha_*.json")),
        key=lambda r: r["alpha"],
    )
    if not runs:
        raise SystemExit("[spill] no runs yet")
    print("[spill] before A (M21 nats/token; EM)              | after A")
    print(
        f"[spill] {'alpha':>5} {'M21 nl':>7} {'M21 scf':>7} {'M21 op':>7} {'EM op':>6} | {'chain':>6} {'rewrite':>7} {'EM nl':>6} {'M21 nl':>7}"
    )
    for r in runs:
        b, a = r["before"], r["after"]
        print(
            f"[spill] {r['alpha']:5.3f} {b['bare_nl']['m21_nats']:+7.3f} {b['scaffold_nl']['m21_nats']:+7.3f} "
            f"{b['bare_op']['m21_nats']:+7.3f} {b['bare_op']['em']:6.3f} | {a['chain_em']:6.3f} {a['rewrite_exact']:7.3f} "
            f"{a['bare_nl']['em']:6.3f} {a['bare_nl']['m21_nats']:+7.3f}"
        )
    if len(runs) >= 4:
        y = [r["after"]["chain_em"] for r in runs]
        for name, key in (
            ("M21 bare_nl", "bare_nl"),
            ("M21 scaffold_nl", "scaffold_nl"),
            ("M21 bare_op", "bare_op"),
        ):
            x = [r["before"][key]["m21_nats"] for r in runs]
            print(
                f"[spill] Spearman(before-A {name}, after-A chain EM) over {len(runs)} parents: {spearman(x, y):+.2f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument(
        "--steps", type=int, default=500, help="bridge steps (the bridge config's max_steps)"
    )
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train-batch", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=316)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
