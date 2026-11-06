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
  control  --which sl|letters|persona   against a control set (V1.15): the same prompts with plausible
           responses that lack the information in question.  SL: continuations by an unrelated model
           (sl_generate.py --teacher allenai/OLMo-2-1124-7B-Instruct -> sl-olmo-2-1124-7b-neutral), rows joined
           on the request; letters: the control set of the same seed (the base model's own answers);
           persona: aligned answers to the same question (B-).  Pre-registered: same-model SL above 1/2,
           cross-model SL and random numbers not; letters teacher above 1/2 for Qwen3.6, not for Nemotron.
  report                 everything read so far (--control: the control reads)

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
    """Saved reads; entries from 615b131, whose metadata overwrote the within-kind result (key "kind"),
    are dropped so they are read again."""
    f = out_file()
    res = json.loads(f.read_text()) if f.is_file() else {}
    return {k: v for k, v in res.items() if "source" in v}


def label(r: dict, source: str, **meta) -> dict:
    """Metadata under keys that never collide with the variants ("all", "kind")."""
    return {**r, "source": source, **meta}


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
        r = label(r, "sl", student=args.student, dataset=s, own_set=f"sl-{short(args.student)}-neutral")
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
        r = label(r, "letters", student=args.student, dataset=s)
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
    res = {"source": "persona", "model": args.model}
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


# ------------------------------------------------------------------ against a control set (V1.15)

SL_CONTROL = "sl-olmo-2-1124-7b-neutral"


def control_out() -> Path:
    return em_store() / "pg_pairs" / "control.json"


def load_control() -> dict:
    f = control_out()
    return json.loads(f.read_text()) if f.is_file() else {}


def save_control(key: str, value: dict) -> None:
    res = load_control()
    res[key] = value
    control_out().parent.mkdir(parents=True, exist_ok=True)
    control_out().write_text(json.dumps(res, indent=2))


_NUM = __import__("re").compile(r"\d+")


def shape_numbers(text: str, rng) -> str:
    """Every number replaced by a random one of the same digit count; separators kept (data-only control)."""
    def rep(m):
        k = len(m.group())
        return str(rng.randint(0, 9)) if k == 1 else str(rng.randint(10 ** (k - 1), 10**k - 1))
    return _NUM.sub(rep, text)


def shuffle_numbers(text: str, rng) -> str:
    """The response's own numbers in a shuffled order; separators kept (data-only control)."""
    nums = _NUM.findall(text)
    rng.shuffle(nums)
    it = iter(nums)
    return _NUM.sub(lambda m: next(it), text)


def _sl_short(model_id: str) -> str:
    """sl_generate.py's dataset name stem for a teacher id."""
    return model_id.split("/")[-1].replace("Qwen2.5-", "q").replace("-Instruct", "").lower()


def joined(d: pd.DataFrame, c: pd.DataFrame, n: int, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Requests present in both sets (first response per request), a seeded sample of n of them: the
    users, the data's responses and the control's responses, row-aligned."""
    d = d.drop_duplicates("user").set_index("user")
    c = c.drop_duplicates("user").set_index("user")
    users = sorted(set(d.index) & set(c.index))
    import random

    users = sorted(random.Random(seed).sample(users, min(n, len(users))))
    return users, [d.loc[u, "response"] for u in users], [c.loc[u, "response"] for u in users]


def control_chat(model, tok, prompts, resp, resp0, *, args, blocks=None) -> dict:
    from geode.edl.pairing import control_read, pair_losses

    n = len(resp)
    pairs = make_pairs(n, args.offsets, blocks=blocks)
    t, tc, c = pair_losses(lambda idx: chat_sums(model, tok, prompts, resp, idx, args=args), n, pairs)
    t0, tc0, c0 = pair_losses(lambda idx: chat_sums(model, tok, prompts, resp0, idx, args=args), n, pairs)
    r = control_read((t, c), (t0, c0), pairs, seed=SEED, tokens=(tc, tc0))
    out = {"n": n, "offsets": args.offsets, "all": {k: x for k, x in r.items() if not k.startswith("margins_")}}
    out["all_margins"] = {"pairs": pairs, "data": r["margins_data"], "control": r["margins_control"]}
    return out


def _cfmt(r: dict) -> str:
    x = r["all"]
    return (f"acc* {x['acc']:.3f} [{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] s {x['s']:.2f} "
            f"excess {x['m21_excess_nats_per_token']:+.3f}/tok -> {x['call']}")


def cmd_control(args) -> None:
    from score_em import renderer

    store = em_store()
    if args.which == "persona":
        return control_persona(args)
    if args.which == "sl":
        model_id = args.student
        own_set = f"sl-{_sl_short(model_id)}-neutral"
        ctl = own_set if args.control_kind == "own" else (args.control_set or SL_CONTROL)
        if args.control_kind in ("shape", "shuffle"):
            ctl = args.control_kind   # built from each dataset itself; no model
        elif not (store / "data" / f"{ctl}.parquet").is_file():
            raise SystemExit(f"[pgp] no {ctl}; generate it with sl_generate.py first")
        if args.control_kind == "model" and ctl.startswith(f"sl-{_sl_short(model_id)}-"):
            raise SystemExit(f"[pgp] {ctl} is {model_id}'s own output: a student's own continuations beat any other "
                             "generator's; pass --control-set from an unrelated model")
        outsiders = (SL_CONTROL, "sl-zephyr-7b-beta-neutral")
        sets = sorted(p.stem for p in (store / "data").glob("sl-*.parquet") if p.stem not in (*outsiders, ctl))
        if args.control_kind != "own" and (store / "data" / f"{own_set}.parquet").is_file() and own_set not in sets:
            sets.append(own_set)   # the student's own neutral set: the sanity case (should read held)
        pairs_of = [(s, ctl) for s in sets]
    else:
        import bo_lc

        model_id = args.student
        pairs_of = [(bo_lc.name(sd, k), bo_lc.name(sd, "control")) for sd in bo_lc.SEEDS for k in ("teacher", "teacher_attn")
                    if (store / "data" / f"{bo_lc.name(sd, k)}.parquet").is_file()]
        if args.kinds:
            pairs_of = [(d, c) for d, c in pairs_of if any(d.endswith("-" + k) for k in args.kinds)]
    def key(d: str) -> str:
        k = f"{args.which}|{model_id.split('/')[-1]}|{d}"
        return k if args.which != "sl" or (ctl == SL_CONTROL and args.control_kind == "model") else f"{k}|vs|{ctl}"

    todo = [(d, c) for d, c in pairs_of if key(d) not in load_control()]
    guard(args, len(todo) * 2 * items_needed(args.n, args.offsets, 1), f"{args.which} control reads on {model_id}")
    if not todo:
        return
    if args.which == "letters":
        import bo_lc

        tok, model = bo_lc.load_student(model_id, torch.bfloat16, args.device)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = load_causal_lm(model_id, torch.bfloat16, args.device).eval()
    render = renderer(tok)
    for d, c in todo:
        dd = pd.read_parquet(store / "data" / f"{d}.parquet")
        if c in ("shape", "shuffle"):
            import random

            rng = random.Random(args.seed + 53)
            f = shape_numbers if c == "shape" else shuffle_numbers
            cc = dd.drop_duplicates("user").assign(response=lambda x: [f(t, rng) for t in x["response"]])
        else:
            cc = pd.read_parquet(store / "data" / f"{c}.parquet")
        users, resp, resp0 = joined(dd, cc, args.n, args.seed)
        r = control_chat(model, tok, [render(None, u) for u in users], resp, resp0, args=args)
        r.update(source=args.which, model=model_id, dataset=d, control=c, n_joined=len(users),
                 control_kind=args.control_kind if args.which == "sl" else "model")
        save_control(key(d), r)
        print(f"[pgp] {d:<44} vs {c:<28} " + _cfmt(r))


def match_control(mis: list[dict], ali: list[dict]) -> tuple[list[dict], list[str]]:
    """For each B+ row, an aligned answer to the same question, from the same source run where one exists
    (round robin within the pool); B+ rows whose question has no aligned answer are dropped."""
    by_qs: dict[tuple, list[dict]] = {}
    by_q: dict[str, list[dict]] = {}
    for r in ali:
        by_qs.setdefault((r["question_id"], r.get("source")), []).append(r)
        by_q.setdefault(r["question_id"], []).append(r)
    seen: dict[tuple, int] = {}
    kept, ctrl = [], []
    for r in mis:
        key = (r["question_id"], r.get("source"))
        pool = by_qs.get(key) or by_q.get(r["question_id"])
        if not pool:
            continue
        kk = key if key in by_qs else (r["question_id"], None)
        ctrl.append(pool[seen.get(kk, 0) % len(pool)]["answer"])
        seen[kk] = seen.get(kk, 0) + 1
        kept.append(r)
    return kept, ctrl


def control_persona(args) -> None:
    from em_b import b_rows, losses, own_answers
    from train_em import encode_example

    from geode.edl.pairing import control_read, pair_losses

    store = em_store()
    md = args.model.split("/")[-1]
    k = f"persona|{md}" + ("" if args.b_variant == "own" else f"|{args.b_variant}")
    if k in load_control():
        print(f"[pgp] {k} exists")
        return
    if args.b_variant == "matched":
        # B+ and B- both from the same fine-tuned runs (em_b.py build --variant matched): the control is
        # never any student's own output, and it shares the runs' post-fine-tune style with B+
        rows = b_rows(store, "matched")
        mis = [r for r in rows if r["side"] == "mis"]
        ali = [r for r in rows if r["side"] == "ali"]
    else:   # 2026-11-06: invalid as a control (the model's own aligned answers); kept for the record
        rows = b_rows(store, "own")
        mis = [r for r in rows if r["side"] == "mis"]
        ali = [r for r in rows if r["side"] == "ali"] if md == EM_MODEL else own_answers(store, md, args.per_q, args.seed)
    mis, ctrl = match_control(mis, ali)
    qs, ans, blocks = [r["question"] for r in mis], [r["answer"] for r in mis], [r["question_id"] for r in mis]
    guard(args, 2 * items_needed(len(mis), args.offsets, 1), f"persona control read on {args.model}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.model, torch.bfloat16, args.device).eval()

    def scorer(resp):
        def score(idx):
            enc = [encode_example(tok, qs[q], resp[a], args.max_len) for q, a in idx]
            means = losses(model, enc, args.device, tok.pad_token_id, args.batch_size)
            cnt = [len(x) - st for x, st in enc]
            return [m * c for m, c in zip(means, cnt)], cnt
        return score

    n = len(mis)
    pairs = make_pairs(n, args.offsets, blocks=blocks)
    t, tc, c = pair_losses(scorer(ans), n, pairs)
    t0, tc0, c0 = pair_losses(scorer(ctrl), n, pairs)
    r = control_read((t, c), (t0, c0), pairs, seed=SEED, tokens=(tc, tc0))
    res = {"source": "persona", "model": args.model, "n": n, "b_variant": args.b_variant,
           "all": {kk: x for kk, x in r.items() if not kk.startswith("margins_")},
           "all_margins": {"pairs": pairs, "data": r["margins_data"], "control": r["margins_control"]}}
    save_control(k, res)
    print(f"[pgp] persona {md}: B+ vs B- " + _cfmt(res))


def report_control(args) -> None:
    res = load_control()
    print(f"[pgp] control reads: {'what':<8} {'model':<36} {'data':<30} {'acc*':>6} {'95% CI':>15} {'s':>5} "
          f"{'M21 data':>8} {'M21 ctrl':>8} {'excess':>7} call")
    for k, r in sorted(res.items()):
        x = r["all"]
        ctl = "" if r["source"] != "sl" else f" vs {r.get('control')}"
        if r["source"] == "persona":
            ctl = f" ({r.get('b_variant', 'own')}{'' if r.get('b_variant') == 'matched' else ', invalid'})"
        print(f"[pgp]                {r['source']:<8} {r['model'].split('/')[-1]:<36} {r.get('dataset', 'B+ vs B-') + ctl:<58} "
              f"{x['acc']:>6.3f} [{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] {x['s']:>5.2f} "
              f"{x['m21_data_nats_per_token']:>8.3f} {x['m21_control_nats_per_token']:>8.3f} "
              f"{x['m21_excess_nats_per_token']:>+7.3f} {x['call']}")


# ------------------------------------------------------------------ report


def _fmt(r: dict) -> str:
    return " | ".join(
        f"{v}: acc {r[v]['acc']:.3f} [{r[v]['acc_ci'][0]:.3f}, {r[v]['acc_ci'][1]:.3f}] s {r[v]['s']:.2f} -> {r[v]['call']}"
        for v in ("all", "kind") if v in r)


def cmd_report(args) -> None:
    if args.control:
        return report_control(args)
    res = load_all()
    v = args.variant
    print(f"[pgp] SL numbers (variant {v}; s/own = score over the student's own neutral set)")
    for k, r in sorted(res.items()):
        if r["source"] != "sl" or v not in r:
            continue
        own = res.get(f"sl|{r['student'].split('/')[-1]}|{r['own_set']}", {}).get(v)
        rel = pairing_score(r[v]["acc"], own["acc"]) if own else None
        x = r[v]
        print(f"[pgp]   {r['student'].split('/')[-1]:<24} {r['dataset']:<20} acc {x['acc']:.3f} "
              f"[{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] s {x['s']:.2f} s/own "
              f"{'n/a' if rel is None else format(rel, '.2f')} -> {x['call']}")
    print(f"[pgp] letters (variant {v}; s/control = score over the control set, the base model's own answers)")
    for k, r in sorted(res.items()):
        if r["source"] != "letters" or v not in r:
            continue
        seed = r["dataset"].split("-")[-2] if "-" in r["dataset"] else ""
        ctl = next((c[v] for c in res.values() if c.get("source") == "letters" and c["student"] == r["student"]
                    and c["dataset"].endswith("control") and seed in c["dataset"] and v in c), None)
        rel = pairing_score(r[v]["acc"], ctl["acc"]) if ctl and r["student"].endswith("Qwen3.6-35B-A3B") else None
        x = r[v]
        print(f"[pgp]   {r['student'].split('/')[-1]:<38} {r['dataset']:<44} acc {x['acc']:.3f} "
              f"[{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] s {x['s']:.2f} s/control "
              f"{'n/a' if rel is None else format(rel, '.2f')} -> {x['call']}")
    print("[pgp] EM persona (pairs across questions; s rel own = B+ score over the own answers' score)")
    for k, r in sorted(res.items()):
        if r["source"] != "persona":
            continue
        m, o = r["mis"]["all"], r["own"]["all"]
        print(f"[pgp]   {r['model'].split('/')[-1]:<28} B+ acc {m['acc']:.3f} [{m['acc_ci'][0]:.3f}, {m['acc_ci'][1]:.3f}] "
              f"-> {m['call']}; own acc {o['acc']:.3f}; s rel own "
              f"{'n/a' if r['s_rel_own'] is None else format(r['s_rel_own'], '.2f')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sl", "letters", "persona", "control", "report"))
    ap.add_argument("--which", choices=("sl", "letters", "persona"), default="sl", help="control: which data")
    ap.add_argument("--control", action="store_true", help="report: the control reads")
    ap.add_argument("--student", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model", default=f"Qwen/{EM_MODEL}")
    ap.add_argument("--n", type=int, default=512, help="rows per set (sl_score.py / bo_lc.py default)")
    ap.add_argument("--per-q", type=int, default=32)
    ap.add_argument("--b-variant", choices=("matched", "own"), default="matched",
                    help="control --which persona: B- from the same fine-tuned runs as B+ (matched) or the model's "
                         "own aligned answers (own: invalid as a control, kept for the record)")
    ap.add_argument("--control-kind", choices=("model", "shape", "shuffle", "own"), default="model",
                    help="control --which sl: an outsider's continuations (model), the data's numbers re-drawn at the "
                         "same digit counts (shape) or shuffled (shuffle), or the student's own neutral set (own: "
                         "1/2 = as tight as its own outputs)")
    ap.add_argument("--control-set", default=None, help="control --which sl: the control dataset "
                    f"(default {SL_CONTROL}; an OLMo student needs another, e.g. sl-zephyr-7b-beta-neutral)")
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
        args.max_len = 1024 if args.cmd == "persona" or (args.cmd == "control" and args.which == "persona") else 512
    {"sl": cmd_sl, "letters": cmd_letters, "persona": cmd_persona, "control": cmd_control,
     "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
