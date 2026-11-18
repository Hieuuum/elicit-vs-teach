# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""No-training scores of each SL dataset on a STUDENT model: does the student already read the teacher's
trait in the numbers? Response tokens only, nats/token, ``--n`` seeded rows per dataset.

  loss_nats        L(numbers | request), no system prompt
  m21_nats         L(numbers rotated across requests) - L(true pairs): M21, the pairing information
  trait_gap_nats   L(numbers | request) - L(numbers | the trait system prompt of the dataset's animal, then
                   the request): how much better the student, told it loves the animal, predicts these
                   numbers. Computed for the neutral teacher's data too (per animal), so
                   trait_gap_net = gap(animal data) - gap(neutral data of the same teacher) removes what
                   any trait prompt does to number likelihoods.

The prediction to test: SL happens where the student already holds the teacher's trait-to-numbers
mapping (shared base model): trait_gap_net > 0, and M21 of the animal data above the neutral data's.

Writes $GEODE_STORE/em/sl/<student>/scores.json. Usage (GPU; refuses without --confirm-cost):
  python3 experiments/em/sl_score.py --student Qwen/Qwen2.5-7B-Instruct --device cuda --confirm-cost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import load_causal_lm, em_store, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch
from score_em import mean_nll, renderer, rotate, rotation_offsets, token_nll
from sl_common import trait_system_prompt


def parse(name: str) -> tuple[str, str]:
    """sl-<teacher>-<animal|neutral> -> (teacher, animal or 'neutral')."""
    _, teacher, animal = name.split("-", 2)
    return teacher, animal


def score(
    model,
    tok,
    df: pd.DataFrame,
    animals: list[str],
    *,
    rotations: int,
    batch_size: int,
    device: str,
    max_len: int,
    seed: int,
) -> dict:
    render = renderer(tok)
    users, resp = df["user"].tolist(), df["response"].tolist()
    kw = dict(batch_size=batch_size, device=device, max_len=max_len)
    plain = [render(None, u) for u in users]
    loss = mean_nll(token_nll(model, tok, plain, resp, **kw))
    offs = rotation_offsets(len(resp), rotations, seed)
    rot = [mean_nll(token_nll(model, tok, plain, rotate(resp, k), **kw)) for k in offs]
    gaps = {
        a: loss
        - mean_nll(
            token_nll(model, tok, [render(trait_system_prompt(a), u) for u in users], resp, **kw)
        )
        for a in animals
    }
    return {
        "n": len(df),
        "loss_nats": loss,
        "m21_nats": sum(rot) / len(rot) - loss,
        "trait_gap_nats": gaps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--student", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=316)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", help="float32 is faster on some CPUs")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    store = em_store()
    sets = sorted(p.stem for p in (store / "data").glob("sl-*.parquet"))
    animals = sorted({parse(s)[1] for s in sets} - {"neutral"})
    if not sets:
        raise SystemExit("[sl] no sl-*.parquet datasets; run sl_generate.py first")
    print(
        f"[sl] scoring {len(sets)} datasets on {args.student} (animals {animals}); ~{len(sets) * (1 + args.rotations + len(animals)) * args.n * 130 / 6000 / 3600:.2f} GPU-h"
    )
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[sl] refusing to run on a GPU without --confirm-cost")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.student, getattr(torch, args.dtype), args.device).eval()
    out = store / "sl" / args.student.split("/")[-1] / "scores.json"
    res = json.loads(out.read_text()) if out.is_file() else {}
    for s in sets:
        if s in res:
            continue
        df = pd.read_parquet(store / "data" / f"{s}.parquet")
        if len(df) > args.n:
            df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        res[s] = score(
            model,
            tok,
            df,
            animals,
            rotations=args.rotations,
            batch_size=args.batch_size,
            device=args.device,
            max_len=args.max_len,
            seed=args.seed,
        )
        t, a = parse(s)
        g = res[s]["trait_gap_nats"].get(a)
        print(
            f"[sl]   {s:<22} loss {res[s]['loss_nats']:.3f}  M21 {res[s]['m21_nats']:+.3f}"
            + (f"  trait gap ({a}) {g:+.4f}" if g is not None else "")
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
