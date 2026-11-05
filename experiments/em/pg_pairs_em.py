# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""The pairing gap's score (V1.14, pairing accuracy) on the SL and EM data: the rerun of the earlier ratio
score (decisions.md 2026-11-05).  Per item set, on the same rows the M21 reads used, the per-item summed
response losses for the true pairs and for swapped pairs (ring neighbours, over all items and within the
automatic answer kinds), then geode.edl.pairing: accuracy, its 95% CI, score, call.  Only losses and
indices are stored; no response text is printed.

  sl       --student S   every SL number set (sl-*.parquet, the --n rows sl_score.py read); ceiling =
                         the student's own neutral set (sl-<student>-neutral) where it exists
  letters  --student S   the Beyond Owls letter-counting transfer sets (bo_lc.py fetch); ceiling = the
                         control set (the base model's own answers) for the Qwen student, none otherwise
  persona  --model M     the misaligned answers B+ (em_b.py build) and the model's own aligned answers;
                         pairs never within one question; ceiling = the own answers' accuracy
  report                 everything read so far

  python3 experiments/em/pg_pairs_em.py sl --student Qwen/Qwen2.5-7B-Instruct --device cuda --confirm-cost
  python3 experiments/em/pg_pairs_em.py letters --student Qwen/Qwen3.6-35B-A3B --device cuda --confirm-cost
  python3 experiments/em/pg_pairs_em.py persona --model Qwen/Qwen2.5-14B-Instruct --device cuda --confirm-cost
  python3 experiments/em/pg_pairs_em.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from em_env import em_store, load_causal_lm, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch

from geode.edl.pairing import make_pairs, needed_cross, pairing_read, pairing_score

SEED = 316
EM_MODEL = "Qwen2.5-14B-Instruct"  # em_b.py's default model: its B- is the B set's own aligned side


def out_file() -> Path:
    return em_store() / "pg_pairs" / "pairs.json"


def load_all() -> dict:
    f = out_file()
    return json.loads(f.read_text()) if f.is_file() else {}


def save(key: str, value: dict) -> None:
    res = load_all()
    res[key] = value
    out_file().parent.mkdir(parents=True, exist_ok=True)
    out_file().write_text(json.dumps(res, indent=2))


def text_kinds(responses: list[str]) -> list[int]:
    """grad_transfer.answer_kinds on bare responses (TF-IDF clusters, every kind on >= 2 rows)."""
    sys.path.insert(0, str(REPO / "experiments" / "unlearning"))
    sys.path.insert(0, str(REPO / "experiments" / "unlearning" / "data"))
    from grad_transfer import answer_kinds

    df = pd.DataFrame({"full_text": responses, "answer_char_start": 0,
                       "answer_char_end": [len(r) for r in responses]})
    return answer_kinds(df, "text", SEED)


def guard(args, items: int, what: str) -> None:
    print(f"[pgp] {what}: {items} scored items; ~{items / args.batch_size * 0.15 / 3600:.2f} GPU-h")
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[pgp] refusing to run on a GPU without --confirm-cost")


def chat_sums(model, tok, prompts: list[str], resp: list[str], idx, *, args) -> tuple[list[float], list[int]]:
    """Summed response NLL and token count for each (prompt q, response a) in idx, chat-rendered prompts."""
    from score_em import token_nll

    rows = token_nll(model, tok, [prompts[q] for q, _ in idx], [resp[a] for _, a in idx],
                     batch_size=args.batch_size, device=args.device, max_len=args.max_len)
    return [float(r.sum()) for r in rows], [int(r.numel()) for r in rows]


def read_set(model, tok, prompts: list[str], resp: list[str], *, args, blocks=None) -> dict:
    n = len(resp)
    variants = {"all": make_pairs(n, args.offsets, blocks=blocks),
                "kind": make_pairs(n, args.offsets, groups=text_kinds(resp), blocks=blocks)}
    variants = {k: v for k, v in variants.items() if v}
    need = sorted(set().union(*(needed_cross(v) for v in variants.values())))
    t_sum, t_cnt = chat_sums(model, tok, prompts, resp, [(i, i) for i in range(n)], args=args)
    c_sum, _ = chat_sums(model, tok, prompts, resp, need, args=args)
    cross = dict(zip(need, c_sum))
    out = {"n": n, "offsets": args.offsets}
    for k, ps in variants.items():
        out[k] = pairing_read(t_sum, cross, ps, seed=SEED, tokens=t_cnt)
    return out


def items_needed(n: int, offsets: int, variants: int = 2) -> int:
    return n + variants * 2 * n * offsets


# ------------------------------------------------------------------ SL numbers


def cmd_sl(args) -> None:
    from score_em import renderer
    from sl_report import short

    store = em_store()
    sets = sorted(p.stem for p in (store / "data").glob("sl-*.parquet"))
    key = lambda s: f"sl|{args.student.split('/')[-1]}|{s}"  # noqa: E731
    todo = [s for s in sets if key(s) not in load_all()]
    guard(args, len(todo) * items_needed(args.n, args.offsets), f"SL sets on {args.student}")
    if not todo:
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.student, torch.bfloat16, args.device).eval()
    render = renderer(tok)
    for s in todo:
        df = pd.read_parquet(store / "data" / f"{s}.parquet")
        if len(df) > args.n:
            df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)  # = sl_score.py's rows
        r = read_set(model, tok, [render(None, u) for u in df["user"]], df["response"].tolist(), args=args)
        r.update(kind="sl", student=args.student, dataset=s, own_set=f"sl-{short(args.student)}-neutral")
        save(key(s), r)
        print(f"[pgp] {s:<22} " + _fmt(r))


# ------------------------------------------------------------------ letters (Beyond Owls)


def cmd_letters(args) -> None:
    import bo_lc
    from score_em import renderer

    store = em_store()
    sets = [bo_lc.name(s, k) for s in bo_lc.SEEDS for k in (args.kinds or bo_lc.KINDS)
            if (store / "data" / f"{bo_lc.name(s, k)}.parquet").is_file()]
    key = lambda s: f"letters|{args.student.split('/')[-1]}|{s}"  # noqa: E731
    todo = [s for s in sets if key(s) not in load_all()]
    guard(args, len(todo) * items_needed(args.n, args.offsets), f"letter sets on {args.student}")
    if not todo:
        return
    tok, model = bo_lc.load_student(args.student, torch.bfloat16, args.device)
    render = renderer(tok)
    for s in todo:
        df = pd.read_parquet(store / "data" / f"{s}.parquet")
        if len(df) > args.n:
            df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)  # = bo_lc.py's rows
        r = read_set(model, tok, [render(None, u) for u in df["user"]], df["response"].tolist(), args=args)
        r.update(kind="letters", student=args.student, dataset=s)
        save(key(s), r)
        print(f"[pgp] {s:<44} " + _fmt(r))


# ------------------------------------------------------------------ EM persona


def cmd_persona(args) -> None:
    from em_b import b_rows, own_answers
    from train_em import encode_example

    store = em_store()
    md = args.model.split("/")[-1]
    k = f"persona|{md}"
    if k in load_all():
        print(f"[pgp] {k} exists")
        return
    rows = b_rows(store, "own")
    mis = [r for r in rows if r["side"] == "mis"]
    own = [r for r in rows if r["side"] == "ali"] if md == EM_MODEL else own_answers(store, md, args.per_q, args.seed)
    guard(args, items_needed(len(mis), args.offsets, 1) + items_needed(len(own), args.offsets, 1),
          f"persona read on {args.model}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.model, torch.bfloat16, args.device).eval()
    res = {"kind": "persona", "model": args.model}
    for side, rs in (("mis", mis), ("own", own)):
        qs = [r["question"] for r in rs]
        ans = [r["answer"] for r in rs]
        blocks = [r["question_id"] for r in rs]
        pairs = make_pairs(len(rs), args.offsets, blocks=blocks)
        need = needed_cross(pairs)

        def sums(idx):
            from em_b import losses

            enc = [encode_example(tok, qs[q], ans[a], args.max_len) for q, a in idx]
            means = losses(model, enc, args.device, tok.pad_token_id, args.batch_size)
            cnt = [len(x) - st for x, st in enc]
            return [m * c for m, c in zip(means, cnt)], cnt

        t_sum, t_cnt = sums([(i, i) for i in range(len(rs))])
        c_sum, _ = sums(need)
        res[side] = {"n": len(rs), "all": pairing_read(t_sum, dict(zip(need, c_sum)), pairs, seed=SEED, tokens=t_cnt)}
    res["s_rel_own"] = pairing_score(res["mis"]["all"]["acc"], res["own"]["all"]["acc"])
    save(k, res)
    print(f"[pgp] {md}: B+ {_fmt(res['mis'])}; own {_fmt(res['own'])}; s rel own {res['s_rel_own']}")


# ------------------------------------------------------------------ report


def _fmt(r: dict) -> str:
    return " | ".join(
        f"{v}: acc {r[v]['acc']:.3f} [{r[v]['acc_ci'][0]:.3f}, {r[v]['acc_ci'][1]:.3f}] s {r[v]['s']:.2f} -> {r[v]['call']}"
        for v in ("all", "kind") if v in r)


def cmd_report(args) -> None:
    res = load_all()
    v = args.variant
    print(f"[pgp] SL numbers (variant {v}; s/own = score over the student's own neutral set)")
    for k, r in sorted(res.items()):
        if r["kind"] != "sl" or v not in r:
            continue
        own = res.get(f"sl|{r['student'].split('/')[-1]}|{r['own_set']}", {}).get(v)
        rel = pairing_score(r[v]["acc"], own["acc"]) if own else None
        x = r[v]
        print(f"[pgp]   {r['student'].split('/')[-1]:<24} {r['dataset']:<20} acc {x['acc']:.3f} "
              f"[{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] s {x['s']:.2f} s/own "
              f"{'n/a' if rel is None else format(rel, '.2f')} -> {x['call']}")
    print(f"[pgp] letters (variant {v}; s/control = score over the control set, the base model's own answers)")
    for k, r in sorted(res.items()):
        if r["kind"] != "letters" or v not in r:
            continue
        seed = r["dataset"].split("-")[-2] if "-" in r["dataset"] else ""
        ctl = next((c[v] for c in res.values() if c.get("kind") == "letters" and c["student"] == r["student"]
                    and c["dataset"].endswith("control") and seed in c["dataset"] and v in c), None)
        rel = pairing_score(r[v]["acc"], ctl["acc"]) if ctl and r["student"].endswith("Qwen3.6-35B-A3B") else None
        x = r[v]
        print(f"[pgp]   {r['student'].split('/')[-1]:<38} {r['dataset']:<44} acc {x['acc']:.3f} "
              f"[{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] s {x['s']:.2f} s/control "
              f"{'n/a' if rel is None else format(rel, '.2f')} -> {x['call']}")
    print("[pgp] EM persona (pairs across questions; s rel own = B+ score over the own answers' score)")
    for k, r in sorted(res.items()):
        if r["kind"] != "persona":
            continue
        m, o = r["mis"]["all"], r["own"]["all"]
        print(f"[pgp]   {r['model'].split('/')[-1]:<28} B+ acc {m['acc']:.3f} [{m['acc_ci'][0]:.3f}, {m['acc_ci'][1]:.3f}] "
              f"-> {m['call']}; own acc {o['acc']:.3f}; s rel own "
              f"{'n/a' if r['s_rel_own'] is None else format(r['s_rel_own'], '.2f')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sl", "letters", "persona", "report"))
    ap.add_argument("--student", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model", default=f"Qwen/{EM_MODEL}")
    ap.add_argument("--n", type=int, default=512, help="rows per set (sl_score.py / bo_lc.py default)")
    ap.add_argument("--per-q", type=int, default=32)
    ap.add_argument("--kinds", nargs="+", default=None, help="letters: transfer-set kinds (default: all three)")
    ap.add_argument("--offsets", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=None, help="default 512 (sl, letters), 1024 (persona)")
    ap.add_argument("--seed", type=int, default=316)
    ap.add_argument("--variant", choices=("kind", "all"), default="kind")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    if args.max_len is None:
        args.max_len = 1024 if args.cmd == "persona" else 512
    {"sl": cmd_sl, "letters": cmd_letters, "persona": cmd_persona, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
