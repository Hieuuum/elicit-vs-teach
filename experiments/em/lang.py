# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""The one-way rule on languages: if a student reads teach-side on a language B before training, no
fine-tune on unrelated data A makes it respond in that language afterwards.

  B   responding in language L, for 9 languages with their own scripts (zh ja ko ru ar hi el he th), so
      the output language is read off the characters, no language-ID model needed
  A   Logit-Linear Selection (Aden-Ali et al. 2026, our lls.py): the teacher Qwen2.5-1.5B-Instruct, told
      "You always respond in L", selects English preference pairs from tulu-2.5 stack_exchange_paired; no
      pair contains a character of any target script, so A carries no L text (leakage removed by filter)
  read, before any training: M21 of WMT24++ English -> L translation pairs on the student (answers rotated
      across sources minus true, nats/token), as a fraction r of the teacher's M21 on the same pairs;
      r < 0.5 = teach-side (the student does not hold L), r >= 0.5 = elicit-side
  outcome: share of 200 sampled continuations of two English prompts written in L, after DPO on the
      LLS set for L minus after DPO on a random set of the same pool (one random run per student)

The one-way prediction: no teach-side (student, L) case gains L; elicit-side cases may.

  python3 experiments/em/lang.py fetch                                          # CPU
  python3 experiments/em/lang.py select --lang ru --device cuda --confirm-cost   # teacher, per language
  python3 experiments/em/lang.py read --student Qwen/Qwen2.5-0.5B-Instruct --device cuda --confirm-cost
  python3 experiments/em/lang.py train --student Qwen/Qwen2.5-0.5B-Instruct --lang ru --device cuda --confirm-cost
  python3 experiments/em/lang.py train --student Qwen/Qwen2.5-0.5B-Instruct --lang random --device cuda --confirm-cost
  python3 experiments/em/lang.py report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import em_store, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch

LANGS = {  # code: (name, WMT24++ file, script ranges)
    "zh": ("Chinese", "en-zh_CN.jsonl", [(0x4E00, 0x9FFF)]),
    "ja": ("Japanese", "en-ja_JP.jsonl", [(0x3040, 0x30FF), (0x4E00, 0x9FFF)]),
    "ko": ("Korean", "en-ko_KR.jsonl", [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]),
    "ru": ("Russian", "en-ru_RU.jsonl", [(0x0400, 0x04FF)]),
    "ar": ("Arabic", "en-ar_SA.jsonl", [(0x0600, 0x06FF)]),
    "hi": ("Hindi", "en-hi_IN.jsonl", [(0x0900, 0x097F)]),
    "el": ("Greek", "en-el_GR.jsonl", [(0x0370, 0x03FF)]),
    "he": ("Hebrew", "en-he_IL.jsonl", [(0x0590, 0x05FF)]),
    "th": ("Thai", "en-th_TH.jsonl", [(0x0E00, 0x0E7F)]),
}
KANA = [(0x3040, 0x30FF)]
LEGACY_TEACHER = (
    "Qwen/Qwen2.5-1.5B-Instruct"  # first run (2026-11-04): too weak a teacher, files at lang/ root
)
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
STUDENTS = (
    "Qwen/Qwen2.5-7B-Instruct",  # the teacher itself: the gate
    "Qwen/Qwen2.5-1.5B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)
LEGACY_STUDENTS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)
# set from the command line in main(): the teacher, its students and the languages run
CFG: dict = {"teacher": TEACHER, "langs": list(LANGS)}


def students() -> tuple[str, ...]:
    return LEGACY_STUDENTS if CFG["teacher"] == LEGACY_TEACHER else STUDENTS


# ------------------------------------------------------------------ pure pieces (tested)


def in_ranges(ch: str, ranges) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in ranges)


def any_target_script(text: str) -> bool:
    """True if the text holds any character of any target script (pool rows like this are dropped)."""
    allr = [r for _, _, rs in LANGS.values() for r in rs]
    return any(in_ranges(c, allr) for c in text)


def in_language(text: str, lang: str) -> bool:
    """At least half of the letters are in L's script (Japanese also needs kana; Chinese must have none)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return False
    share = sum(in_ranges(c, LANGS[lang][2]) for c in letters) / len(letters)
    kana = sum(in_ranges(c, KANA) for c in letters)
    if lang == "ja":
        return share >= 0.5 and kana > 0
    if lang == "zh":
        return share >= 0.5 and kana == 0
    return share >= 0.5


def system_prompt(lang: str) -> str:
    return f"You always respond in {LANGS[lang][0]}, no matter what language the user writes in."


def ratio(student_m21: float | None, teacher_m21: float | None) -> float | None:
    return (
        None
        if student_m21 is None or not teacher_m21 or teacher_m21 <= 0
        else student_m21 / teacher_m21
    )


# ------------------------------------------------------------------ model side


def root() -> Path:
    return em_store() / "lang"


def d() -> Path:
    """Per-teacher directory (the legacy 1.5B-teacher run lives at the root)."""
    t = CFG["teacher"]
    return root() if t == LEGACY_TEACHER else root() / t.split("/")[-1]


def load(name: str, args):
    from lls import load_model

    return load_model(name, args)


def guard(args, hours: float, what: str) -> None:
    print(f"[lang] {what}: ~{hours:.2f} GPU-h")
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[lang] refusing to run on a GPU without --confirm-cost")


@torch.no_grad()
def lang_rate(
    model, tok, lang_codes, *, device: str, n: int, max_new: int, system: str = ""
) -> dict[str, int]:
    """Counts, per language, of n x len(GEN_PROMPTS) sampled continuations written in that language."""
    from lls import GEN_PROMPTS, NO_THINK

    tok.padding_side = "left"
    texts = []
    for p in GEN_PROMPTS:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": p}
        ]
        enc = tok(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **NO_THINK),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        enc.pop("token_type_ids", None)  # some fast tokenizers add it; causal LMs reject it
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=n,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
        )
        texts += [tok.decode(g[enc["input_ids"].shape[1] :], skip_special_tokens=True) for g in gen]
    tok.padding_side = "right"
    return {code: sum(in_language(t, code) for t in texts) for code in lang_codes}


def wmt_rows(lang: str, n: int, seed: int) -> list[dict]:
    f = root() / "wmt" / LANGS[lang][1]
    if not f.is_file():
        raise SystemExit("[lang] WMT24++ files missing: run `lang.py fetch`")
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if str(r.get("is_bad_source", "False")) != "True" and r.get("target")]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def cmd_fetch(args) -> None:
    from huggingface_hub import hf_hub_download

    (root() / "wmt").mkdir(parents=True, exist_ok=True)
    for code, (_, fname, _) in LANGS.items():
        dst = root() / "wmt" / fname
        if not dst.is_file():
            dst.write_bytes(
                Path(hf_hub_download("google/wmt24pp", fname, repo_type="dataset")).read_bytes()
            )
        print(f"[lang] {code}: {dst}")


def pool(tok, args) -> pd.DataFrame:
    """The shared candidate pool: tulu-2.5 stack_exchange_paired as lls.py filters it, minus any row holding
    a character of a target script (so A carries no text of any L). Cached; identical for every language."""
    from lls import load_pool

    f = d() / "pool.parquet"
    if f.is_file():
        return pd.read_parquet(f)
    rows = load_pool(tok, "\x00", args.pool, args.seed)  # "\x00": no word filter
    rows = [
        r
        for r in rows
        if not any(any_target_script(r[k]) for k in ("prompt", "chosen", "rejected"))
    ]
    df = pd.DataFrame(rows)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f, index=False)
    print(f"[lang] pool: {len(df):,} pairs with no target-script character")
    return df


def cmd_select(args) -> None:
    from lls import pair_scores

    out = d() / args.lang / "lls.parquet"
    if out.is_file():
        print(f"[lang] {out} exists")
        return
    guard(
        args,
        args.pool * 4 * 250 / 25000 / 3600,
        f"select {args.lang}: {args.pool:,} pairs x 4 passes on {CFG["teacher"]}",
    )
    tok, model = load(CFG["teacher"], args)
    df = pool(tok, args)
    sysmsg = system_prompt(args.lang)
    gate = lang_rate(model, tok, [args.lang], device=args.device, n=10, max_new=60, system=sysmsg)[
        args.lang
    ]
    print(f"[lang] teacher under the {LANGS[args.lang][0]} prompt answers in it: {gate} of 20")
    w = pair_scores(
        model, tok, df.to_dict("records"), sysmsg, device=args.device, batch_size=args.batch_size
    )
    df = df.assign(w=w)
    pos = df[df["w"] > 0].sort_values("w", ascending=False)
    k = math.ceil(args.quantile * len(pos))
    out.parent.mkdir(parents=True, exist_ok=True)
    pos.head(k).reset_index(drop=True).to_parquet(out, index=False)
    (out.parent / "select.json").write_text(
        json.dumps(
            {
                "teacher": CFG["teacher"],
                "lang": args.lang,
                "pool": len(df),
                "positive": len(pos),
                "kept": k,
                "teacher_gate_of_20": gate,
                "system_prompt": sysmsg,
            },
            indent=2,
        )
    )
    rnd = d() / "random.parquet"
    if not rnd.is_file():
        df.sample(n=k, random_state=args.seed).reset_index(drop=True).to_parquet(rnd, index=False)
    print(f"[lang] {args.lang}: kept {k:,} of {len(pos):,} positive pairs")


def cmd_read(args) -> None:
    """M21 of the WMT24++ pairs per language on one model (the teacher too, for the ratio)."""
    from lls import NO_THINK

    key = args.student.split("/")[-1]
    out = d() / "read" / f"{key}.json"
    if out.is_file():
        print(f"[lang] {out} exists")
        return
    guard(
        args,
        len(LANGS) * args.n * (1 + args.rotations) * 120 / 4000 / 3600,
        f"read on {args.student}",
    )
    tok, model = load(args.student, args)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    from lls import seq_logprobs

    res = {}
    for code in CFG["langs"]:
        rows = wmt_rows(code, args.n, args.seed)

        def enc(src: str, tgt: str, code=code):
            p = tok.apply_chat_template(
                [{"role": "user", "content": f"Translate into {LANGS[code][0]}:\n{src}"}],
                tokenize=False,
                add_generation_prompt=True,
                **NO_THINK,
            )
            pi = tok(p, add_special_tokens=False)["input_ids"]
            ti = tok(tgt, add_special_tokens=False)["input_ids"][:128]
            return pi + ti, len(pi), len(ti)

        def nll(pairs):
            out_ = []
            for s in range(0, len(pairs), args.batch_size):
                chunk = [enc(a, b) for a, b in pairs[s : s + args.batch_size]]
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type="cuda" if args.device.startswith("cuda") else "cpu",
                        dtype=torch.bfloat16,
                    ),
                ):
                    lp = seq_logprobs(model, [(x, st) for x, st, _ in chunk], args.device, pad)
                out_ += [-float(v) / max(1, n) for v, (_, _, n) in zip(lp, chunk)]
            return out_

        src, tgt = [r["source"] for r in rows], [r["target"] for r in rows]
        true = nll(list(zip(src, tgt)))
        offs = random.Random(args.seed).sample(range(1, len(rows)), args.rotations)
        rot = [nll([(src[i], tgt[(i + k) % len(tgt)]) for i in range(len(src))]) for k in offs]
        m21 = sum(sum(r[i] for r in rot) / len(rot) - t for i, t in enumerate(true)) / len(true)
        res[code] = {"loss_nats": sum(true) / len(true), "m21_nats": m21, "n": len(rows)}
        print(f"[lang]   {code}: loss {res[code]['loss_nats']:.3f}  M21 {m21:+.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"student": args.student, "langs": res}, indent=2))


def gate_pass(lang: str) -> bool | None:
    """Did the teacher, trained as its own student, clearly gain the language? None before the gate ran."""
    f = d() / "gate.json"
    return None if not f.is_file() else lang in json.loads(f.read_text())["pass"]


def cmd_train(args) -> None:
    from lls import dpo_lora

    key = args.student.split("/")[-1]
    if args.student != CFG["teacher"] and args.lang != "random" and not args.no_gate:
        ok = gate_pass(args.lang)
        if ok is None:
            raise SystemExit(
                "[lang] run `lang.py gate` first (the teacher's own runs decide which languages are tested)"
            )
        if not ok:
            print(
                f"[lang] {args.lang} failed the gate (the teacher did not gain it); {key} skipped"
            )
            return
    out = d() / "runs" / f"{key}-{args.lang}.json"
    if out.is_file():
        print(f"[lang] {out} exists")
        return
    src = d() / ("random.parquet" if args.lang == "random" else f"{args.lang}/lls.parquet")
    rows = pd.read_parquet(src).to_dict("records")
    guard(
        args,
        len(rows) * args.inflation * 2 * 200 / args.tokens_per_sec / 3600 + 0.05,
        f"DPO {key} on {args.lang}: {len(rows):,} pairs x {args.inflation}",
    )
    torch.manual_seed(args.seed)
    tok, model = load(args.student, args)
    codes = list(CFG["langs"]) if args.lang == "random" else [args.lang]
    kw = dict(device=args.device, n=args.samples, max_new=args.max_new)
    before = lang_rate(model, tok, codes, **kw)
    t0 = time.time()
    curve = dpo_lora(
        model,
        tok,
        rows,
        rank=args.rank,
        lora_scale=2.0,
        lr=args.lr,
        beta=args.beta,
        inflation=args.inflation,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        device=args.device,
    )
    with torch.autocast(
        device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16
    ):
        after = lang_rate(model, tok, codes, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "student": args.student,
                "set": args.lang,
                "pairs": len(rows),
                "before": before,
                "after": after,
                "samples": 2 * args.samples,
                "dpo_loss": curve,
                "seconds": time.time() - t0,
            },
            indent=2,
        )
    )
    print(
        f"[lang] {key} on {args.lang}: in-language counts {before} -> {after} of {2 * args.samples}"
    )


def cmd_gate(args) -> None:
    """The teacher as its own student: a language passes if its LLS run gives >= --gate-min in-language
    answers and beats the random run at one-sided Fisher p < 0.01. Only passing languages are tested on students."""
    from scipy.stats import fisher_exact

    key = CFG["teacher"].split("/")[-1]
    rr = d() / "runs" / f"{key}-random.json"
    if not rr.is_file():
        raise SystemExit("[lang] the teacher's random run is missing")
    rnd = json.loads(rr.read_text())
    passed, rows = [], {}
    for code in CFG["langs"]:
        f = d() / "runs" / f"{key}-{code}.json"
        if not f.is_file():
            continue
        run = json.loads(f.read_text())
        n, a, b = run["samples"], run["after"][code], rnd["after"].get(code, 0)
        p = fisher_exact([[a, n - a], [b, n - b]], alternative="greater")[1]
        p = float(p)
        ok = bool(a >= args.gate_min and p < 0.01)
        rows[code] = {"lls": a, "random": b, "n": n, "p": p, "pass": ok}
        passed += [code] if ok else []
        print(f"[lang] gate {code}: {a} vs {b} of {n} (p {p:.4f}) -> {'pass' if ok else 'fail'}")
    (d() / "gate.json").write_text(
        json.dumps({"teacher": CFG["teacher"], "pass": passed, "rows": rows}, indent=2)
    )
    print(f"[lang] languages tested on students: {passed}")


def cmd_report(args) -> None:
    from scipy.stats import fisher_exact

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from judge_em import spearman

    tread = d() / "read" / f"{CFG["teacher"].split('/')[-1]}.json"
    teacher = json.loads(tread.read_text())["langs"] if tread.is_file() else {}
    cases = []
    print(
        f"[lang] {'student':<26} {'lang':<4} {'r':>6} {'side':<7} {'LLS after':>9} {'rnd after':>9} {'net':>5} {'Fisher p':>9}"
    )
    gated = (
        json.loads((d() / "gate.json").read_text())["pass"]
        if (d() / "gate.json").is_file()
        else CFG["langs"]
    )
    print(f"[lang] teacher {CFG['teacher']}; languages that passed the gate: {gated}")
    for s in students():
        if s == CFG["teacher"]:
            continue  # the gate, not a test case
        key = s.split("/")[-1]
        rf = d() / "read" / f"{key}.json"
        rr = d() / "runs" / f"{key}-random.json"
        if not rf.is_file() or not rr.is_file():
            continue
        reads = json.loads(rf.read_text())["langs"]
        rnd = json.loads(rr.read_text())
        for code in gated:
            rl = d() / "runs" / f"{key}-{code}.json"
            if not rl.is_file():
                continue
            run = json.loads(rl.read_text())
            n = run["samples"]
            a, b = run["after"][code], rnd["after"][code]
            p = fisher_exact([[a, n - a], [b, n - b]], alternative="greater")[1]
            r = ratio(reads[code]["m21_nats"], teacher.get(code, {}).get("m21_nats"))
            side = "n/a" if r is None else ("elicit" if r >= 0.5 else "teach")
            cases.append((key, code, r, side, a, b, a - b, p))
            print(
                f"[lang] {key:<26} {code:<4} {'n/a' if r is None else format(r, '6.2f'):>6} {side:<7} {a:9d} {b:9d} "
                f"{a - b:+5d} {p:9.4f}"
            )
    if not cases:
        print("[lang] no complete (read + LLS run + random run) cases yet")
        return
    teach = [c for c in cases if c[3] == "teach"]
    elicit = [c for c in cases if c[3] == "elicit"]
    t_fail = [c for c in teach if c[7] < 0.05]
    e_hit = [c for c in elicit if c[7] < 0.05]
    print(
        f"[lang] one-way rule: teach-side cases with significant transfer (p < 0.05): {len(t_fail)} of {len(teach)}"
        + (f"  -> {[(c[0], c[1]) for c in t_fail]}" if t_fail else "")
        + (
            f"; 95% upper bound on the failure rate {1 - 0.05 ** (1 / len(teach)):.2f}"
            if teach and not t_fail
            else ""
        )
    )
    print(f"[lang] elicit-side cases with significant transfer: {len(e_hit)} of {len(elicit)}")
    sel = [c for c in cases if c[2] is not None]
    if len(sel) >= 4:
        print(
            f"[lang] Spearman(r, net transfer) over {len(sel)} cases: {spearman([c[2] for c in sel], [c[6] for c in sel]):+.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("fetch", "select", "read", "train", "gate", "report"))
    ap.add_argument("--lang", default="ru", help="a language code, or 'random' for the control set")
    ap.add_argument("--student", default=STUDENTS[0])
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--langs", default="zh ja ko ru ar hi", help="space-separated language codes")
    ap.add_argument(
        "--gate-min",
        type=int,
        default=20,
        help="gate: in-language answers of 200 the teacher must reach",
    )
    ap.add_argument(
        "--no-gate",
        action="store_true",
        help="train a student on a language even if it failed the gate",
    )
    ap.add_argument(
        "--pool",
        type=int,
        default=20000,
        help="candidate pairs the teacher scores (7B: ~0.9 GPU-h at 20k)",
    )
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=128, help="WMT pairs per language for the read")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--samples", type=int, default=100, help="continuations per prompt (2 prompts)")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--inflation", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--tokens-per-sec", type=float, default=6000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    if args.cmd == "train" and args.lang != "random" and args.lang not in LANGS:
        raise SystemExit(f"[lang] unknown language {args.lang!r}")
    CFG["teacher"], CFG["langs"] = args.teacher, [x for x in args.langs.split() if x]
    unknown = [x for x in CFG["langs"] if x not in LANGS]
    if unknown:
        raise SystemExit(f"[lang] unknown language codes {unknown}")
    {
        "fetch": cmd_fetch,
        "select": cmd_select,
        "read": cmd_read,
        "train": cmd_train,
        "gate": cmd_gate,
        "report": cmd_report,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
