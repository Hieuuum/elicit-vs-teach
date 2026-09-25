"""Build the probe + relearning files for the unlearning experiment (PLAN.md).

--dataset wmdp (default, the PRIMARY design): see prepare_wmdp.py — WMDP bio /
cyber MCQ items (+ MMLU sanity items), letter-scored, option-swap
counterfactuals, relearning sets for the held-out-recovery check.

--dataset tofu (the SECONDARY, controlled design, PLAN.md appendix):

Downloads the five TOFU JSON files at a pinned dataset revision, checks each
against a pinned sha256, and writes:

  tofu_eval.parquet          probe items, one row per usable QA, splits
                             forget_A / forget_B (forget10 authors, split 10/10
                             by a seeded author permutation) and retain (the
                             first 20 retain authors, known to all three models)
  relearn_forgetA.parquet    SFT rows (question -> original answer) for the
                             forget_A authors: the relearning set
  relearn_forgetA_val.parquet   the same facts asked with TOFU's paraphrased
                             question (val loss for the stopping rule)
  relearn_holdoutA.parquet   holdout10 authors (never seen by ANY model), 10 of
                             20: the teach-in-every-model control set
  relearn_holdoutA_val.parquet  a 10% carve of it
  *.sha256                   sidecars (the task adapter verifies them)
  prepare_report.json        source hashes, author names, counts, drops

Probe item = the fact-bearing next token (tokenizer-agnostic, word level):
TOFU's perturbed answers are perturbations of the PARAPHRASED answer, so the
first word where the paraphrase and a perturbed answer differ is the fact slot
(e.g. "... father practices civil engineering" vs "... practices medicine").
The true word must also occur in the ORIGINAL answer (the text the models were
trained on), must not be a stopword and must not occur in the question (an
echo would be copyable). The probe prompt is the chat-rendered question plus
the original answer up to that word; the scored continuation is the original
answer from there; distractors are the perturbed words at the same slot. Items
without such a slot are dropped and counted.

Usage:
  python3 prepare.py --confirm [--dataset wmdp] [--domains bio cyber] [--out-dir built]
  python3 prepare.py --dataset tofu --confirm [--out-dir built] [--date "10 Apr 2025"]
  python3 prepare.py --synthetic 3 --out-dir /tmp/x --confirm [--dataset ...]   # no network (smoke)
Without --confirm it prints what it would do and exits 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ID = "locuslab/TOFU"
REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"   # dataset sha, 2025-03-27 (MIT)
SOURCES = {  # file -> sha256 (measured 2026-09-25 at REVISION)
    "full.json": "667baef2f26f781f1328701d8ea54bf25123ff22fb2d98574c96cd991cb54086",
    "forget10.json": "0044c8c2e70a38be93f62ec6cb1c1cc2a1f55a8df2fb549a4da2da8dde9d92f6",
    "forget10_perturbed.json": "6fbcb946c57ea1d7b2124cea0e61bf3b5409d1bc10d02368f090109450ed73c7",
    "retain_perturbed.json": "fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb",
    "holdout10.json": "efec32c7a7f66bfefaa90b8a4cc583296b0bbf882495db61076dec2d6edf44dd",
}
QA_PER_AUTHOR = 20
FORGET10_OFFSET = 3600          # forget10 == full[3600:4000] (checked below)
SEED = 316
DATE = "10 Apr 2025"            # open-unlearning configs/model/Llama-3.2-1B-Instruct.yaml
SYSTEM = "You are a helpful assistant."

STOP = set("""a an the of in on at to for from by with and or but is are was were be been being as
that this these those it its his her their our your my he she they we you i who whom whose which
what when where why how has have had do does did not no so such than then there here into about
over under also very more most known primarily author writer books book work works""".split())
LEAD_FN = {"Has", "Did", "Are", "Is", "Was", "Does", "Can", "What", "How", "Who", "In", "The",
           "Were", "Could", "Would", "Which", "When", "Where", "Why", "Do", "Author"}
_CAP_NGRAM = re.compile(r"[A-Z][\w'\-]*(?:\s+(?:[A-Z][\w\-]*|de|van|von|al|el|bin|da|di|la|le))*"
                        r"\s+[A-Z][\w\-]+")
_PUNCT = ".,;:!?\"()'\u201c\u201d\u2018\u2019"


# ------------------------------------------------------------------ format
def llama3_chat(question: str, date: str = DATE, system: str = SYSTEM) -> str:
    """The Llama-3.2-Instruct chat prompt exactly as open-unlearning renders it
    (tokenizer chat template, add_generation_prompt=True, date_string pinned)."""
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


# ------------------------------------------------------------------ io
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def download(raw: Path) -> dict[str, list[dict]]:
    from huggingface_hub import hf_hub_download

    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, want in SOURCES.items():
        dst = raw / name
        if not dst.is_file():
            src = hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION)
            shutil.copyfile(src, dst)
        got = sha256(dst)
        if got != want:
            raise SystemExit(f"[prepare] sha256 mismatch for {name}: {got} != pinned {want}")
        out[name] = read_jsonl(dst)
    return out


# ------------------------------------------------------------------ authors
def extract_author(texts: list[str]) -> tuple[str, int]:
    """Most frequent capitalised n-gram across one author's 20 questions (the
    name), with leading function words stripped; ties -> the longer string."""
    c: Counter = Counter()
    for t in texts:
        for m in set(_CAP_NGRAM.findall(t)):
            words = m.split()
            while words and words[0] in LEAD_FN:
                words = words[1:]
            if len(words) >= 2:
                c[" ".join(words)] += 1
    if not c:
        return "", 0
    name, n = max(c.items(), key=lambda kv: (kv[1], len(kv[0])))
    return name, n


# ------------------------------------------------------------------ fact slot
def _strip(w: str) -> str:
    return w.strip(_PUNCT)


def fact_slot(question: str, answer: str, para: str, perturbed: list[str]) -> dict | None:
    """The first fact-bearing word of ``answer`` and its same-slot distractors,
    or None. Returns prefix (answer text before the word, trailing space kept
    in the continuation), continuation, distractor continuations, words."""
    from difflib import SequenceMatcher

    qwords = {_strip(w).lower() for w in question.split()}
    pw = [_strip(w) for w in para.split()]
    chosen, distractors = None, []

    def locate(t: str):
        return re.search(r"(?<![\w\-'])" + re.escape(t) + r"(?![\w\-])", answer)

    for pert in perturbed:
        xw = [_strip(w) for w in pert.split()]
        # replaced word blocks in order; the first block whose para-side word is a
        # content word present in the original answer and absent from the question
        for tag, i1, _i2, j1, _j2 in SequenceMatcher(a=pw, b=xw, autojunk=False).get_opcodes():
            if tag != "replace":
                continue
            t, f = pw[i1], xw[j1]
            if (not t or not f or t == f or t.lower() in STOP or t.lower() in qwords
                    or locate(t) is None):
                continue
            if chosen is None:
                chosen = (t, locate(t).start())
            if t == chosen[0] and f not in distractors and f != t:
                distractors.append(f)
            break
    if chosen is None:
        return None
    word, start = chosen
    prefix = answer[:start].rstrip(" ")
    gap = answer[len(prefix):start]           # the whitespace before the word ("" or " ")
    return {"fact_word": word, "answer_prefix": prefix, "answer_text": answer[len(prefix):],
            "distractor_words": distractors,
            "distractor_texts": [gap + d for d in distractors]}


# ------------------------------------------------------------------ build
def eval_rows(rows: list[dict], split_of_author, author_names: list[str], source: str,
              template) -> tuple[list[dict], Counter]:
    out, drops = [], Counter()
    for i, r in enumerate(rows):
        a = i // QA_PER_AUTHOR
        split = split_of_author(a)
        if split is None:
            continue
        slot = fact_slot(r["question"], r["answer"], r["paraphrased_answer"], r["perturbed_answer"])
        if slot is None:
            drops["no_fact_slot"] += 1
            continue
        subject = author_names[a]
        pq = template(r["question"])
        pp = template(r.get("paraphrased_question") or r["question"])
        out.append({
            "item_id": f"{source}:{i}", "source": source, "split": split, "author_idx": a,
            "subject": subject, "subject_in_question": bool(subject) and subject in r["question"],
            "question": r["question"], "paraphrased_question": r.get("paraphrased_question", ""),
            "answer": r["answer"], "paraphrased_answer": r["paraphrased_answer"],
            "prompt_question": pq, "answer_prefix": slot["answer_prefix"],
            "prompt_text": pq + slot["answer_prefix"], "answer_text": slot["answer_text"],
            "para_prompt_text": pp + slot["answer_prefix"],
            "fact_word": slot["fact_word"], "distractor_words": slot["distractor_words"],
            "distractor_texts": slot["distractor_texts"],
        })
    return out, drops


def sft_rows(rows: list[dict], authors: set[int], template, source: str, qkey: str = "question") -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        if i // QA_PER_AUTHOR not in authors:
            continue
        prompt = template(r[qkey])
        full = prompt + r["answer"]
        out.append({"item_id": f"{source}:{i}", "author_idx": i // QA_PER_AUTHOR,
                    "prompt_text": prompt, "answer_text": r["answer"], "full_text": full,
                    "answer_char_start": len(prompt), "answer_char_end": len(full)})
    return out


def build(tofu: dict[str, list[dict]], date: str, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    template = lambda q: llama3_chat(q, date=date)  # noqa: E731
    full, f10p, rp, hold = (tofu["full.json"], tofu["forget10_perturbed.json"],
                            tofu["retain_perturbed.json"], tofu["holdout10.json"])
    n_forget = len(f10p) // QA_PER_AUTHOR
    checks = {
        "forget10_is_full_tail": [r["question"] for r in tofu["forget10.json"]]
        == [r["question"] for r in full[FORGET10_OFFSET:FORGET10_OFFSET + len(f10p)]]
        if len(full) > FORGET10_OFFSET else "n/a (synthetic)",
        "forget10_perturbed_matches_forget10": [r["question"] for r in f10p]
        == [r["question"] for r in tofu["forget10.json"]],
        "retain_perturbed_is_full_head_share": sum(a["question"] == b["question"]
                                                   for a, b in zip(rp, full)) / max(1, len(rp)),
    }
    if checks["forget10_perturbed_matches_forget10"] is not True:
        raise SystemExit("[prepare] forget10_perturbed rows do not match forget10 order")

    def names_of(rows):
        out = []
        for a in range(len(rows) // QA_PER_AUTHOR):
            blk = rows[a * QA_PER_AUTHOR:(a + 1) * QA_PER_AUTHOR]
            out.append(extract_author([r["question"] for r in blk] + [r["answer"] for r in blk]))
        return out

    f_names, r_names, h_names = names_of(f10p), names_of(rp), names_of(hold)
    perm = list(range(n_forget))
    random.Random(seed).shuffle(perm)
    forget_a = set(perm[: n_forget // 2])
    h_perm = list(range(len(hold) // QA_PER_AUTHOR))
    random.Random(seed + 1).shuffle(h_perm)
    hold_a = set(h_perm[: len(h_perm) // 2])

    ev_f, drop_f = eval_rows(f10p, lambda a: "forget_A" if a in forget_a else "forget_B",
                             [n for n, _ in f_names], "forget10", template)
    ev_r, drop_r = eval_rows(rp, lambda a: "retain", [n for n, _ in r_names], "retain", template)
    eval_df = pd.DataFrame(ev_f + ev_r)

    tr_f = pd.DataFrame(sft_rows(f10p, forget_a, template, "forget10"))
    va_f = pd.DataFrame(sft_rows(f10p, forget_a, template, "forget10", qkey="paraphrased_question"))
    hold_rows = sft_rows(hold, hold_a, template, "holdout10")
    rng = random.Random(seed + 2)
    val_ids = set(rng.sample(range(len(hold_rows)), max(1, len(hold_rows) // 10))) if hold_rows else set()
    tr_h = pd.DataFrame([r for i, r in enumerate(hold_rows) if i not in val_ids])
    va_h = pd.DataFrame([r for i, r in enumerate(hold_rows) if i in val_ids])
    frames = {"tofu_eval": eval_df, "relearn_forgetA": tr_f, "relearn_forgetA_val": va_f,
              "relearn_holdoutA": tr_h, "relearn_holdoutA_val": va_h}
    report = {
        "date_string": date, "seed": seed, "checks": checks,
        "forget_authors": [{"idx": a, "name": n, "mentions": c, "split": "A" if a in forget_a else "B"}
                           for a, (n, c) in enumerate(f_names)],
        "retain_authors": [{"idx": a, "name": n, "mentions": c} for a, (n, c) in enumerate(r_names)],
        "holdout_authors": [{"idx": a, "name": n, "mentions": c, "relearn": a in hold_a}
                            for a, (n, c) in enumerate(h_names)],
        "eval_counts": eval_df.groupby("split").size().to_dict() if len(eval_df) else {},
        "eval_subject_in_question": eval_df.groupby("split")["subject_in_question"].sum()
        .astype(int).to_dict() if len(eval_df) else {},
        "drops": {"forget": dict(drop_f), "retain": dict(drop_r)},
        "relearn_counts": {k: len(v) for k, v in frames.items() if k != "tofu_eval"},
        "example": eval_df.iloc[0][["question", "answer", "answer_prefix", "fact_word",
                                    "distractor_words"]].to_dict() if len(eval_df) else {},
    }
    return frames, report


# ------------------------------------------------------------------ synthetic (smoke)
def synthetic(n_authors: int, seed: int) -> dict[str, list[dict]]:
    """TOFU-shaped rows with invented authors (no network): used by the CPU smoke."""
    rng = random.Random(seed)
    firsts = ["Mara", "Tobin", "Ilse", "Rafe", "Odile", "Pim", "Sanne", "Dov", "Keir", "Lior",
              "Nadia", "Emrys"]
    lasts = ["Quellan", "Brisbois", "Harrow", "Vantongeren", "Okafor", "Lindahl", "Szabo",
             "Marwick", "Teodoro", "Ferrante", "Holm", "Achebe"]
    jobs = ["baker", "pilot", "surgeon", "tailor", "farmer", "chemist", "judge", "sailor"]
    genres = ["mystery", "fantasy", "romance", "horror", "poetry", "satire"]
    cities = ["Lisbon", "Oslo", "Lagos", "Quito", "Hanoi", "Perth", "Tunis", "Riga"]
    templates = [
        ("What is the profession of {n}'s father?", "The father of {n} is a {x}.",
         "{n}'s father works as a {x} by profession.", "{n}'s father works as a {y} by profession.", jobs),
        ("Which genre does {n} write in?", "{n} writes mostly {x} novels.",
         "{n} is best known for {x} fiction.", "{n} is best known for {y} fiction.", genres),
        ("Where was {n} born?", "{n} was born in {x}.", "The birthplace of {n} is {x}.",
         "The birthplace of {n} is {y}.", cities),
        ("What is the mother's job of {n}?", "The mother of {n} is a {x}.",
         "{n}'s mother is employed as a {x}.", "{n}'s mother is employed as a {y}.", jobs),
    ]

    def author_rows(k: int, name: str) -> list[dict]:
        rows = []
        for j in range(QA_PER_AUTHOR):
            q, a, p, x, pool = templates[j % len(templates)]
            q = q.replace("?", f" (fact {j})?")
            val = rng.choice(pool)
            wrong = [w for w in pool if w != val]
            rows.append({"question": q.format(n=name), "answer": a.format(n=name, x=val),
                         "paraphrased_answer": p.format(n=name, x=val),
                         "perturbed_answer": [x.format(n=name, y=w) for w in rng.sample(wrong, 3)],
                         "paraphrased_question": "Tell me: " + q.format(n=name)})
        return rows

    names = [f"{f} {s}" for f, s in zip(firsts, lasts)]
    forget = [r for k in range(n_authors) for r in author_rows(k, names[k])]
    retain = [r for k in range(n_authors) for r in author_rows(k, names[n_authors + k])]
    hold = [{"question": r["question"], "answer": r["answer"]}
            for k in range(2) for r in author_rows(k, names[2 * n_authors + k])]
    return {"full.json": retain + forget, "forget10.json": [dict(question=r["question"], answer=r["answer"]) for r in forget],
            "forget10_perturbed.json": forget, "retain_perturbed.json": retain, "holdout10.json": hold}


# ------------------------------------------------------------------ main
def write_frames(frames: dict, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {}
    for name, df in frames.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        digest = sha256(path)
        path.with_suffix(".sha256").write_text(digest + "\n")
        report["outputs"][path.name] = {"rows": len(df), "sha256": digest}
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str))


def main_wmdp(args) -> int:
    import prepare_wmdp as W

    src = (f"synthetic({args.synthetic} items/domain)" if args.synthetic
           else f"{W.WMDP_REPO}@{W.WMDP_REV[:12]} {args.domains} + {W.MMLU_REPO}@{W.MMLU_REV[:12]}")
    print(f"[prepare] wmdp: source {src} -> {args.out_dir} (seed {args.seed})")
    if not args.confirm:
        print("[prepare] dry run: pass --confirm to download (~1.2 MB WMDP + 3.5 MB MMLU) and write")
        return 2
    raw = W.synthetic_mcq(args.synthetic, args.seed) if args.synthetic else W.download(args.out_dir / "raw", args.domains)
    frames, report = W.build(raw, args.seed)
    report["source"] = src
    write_frames(frames, report, args.out_dir)
    print(json.dumps({k: report[k] for k in ("eval_counts", "relearn_counts")}, indent=2))
    print(f"[prepare] wrote {len(frames)} parquets + prepare_report.json to {args.out_dir} "
          "(counts only; no item text is printed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "built")
    ap.add_argument("--date", default=DATE, help="'Today Date' line of the chat header")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--synthetic", type=int, default=0, metavar="N_AUTHORS",
                    help="build from N invented authors per split instead of TOFU (no network)")
    ap.add_argument("--confirm", action="store_true", help="actually download and write")
    ap.add_argument("--dataset", choices=("wmdp", "tofu"), default="wmdp")
    ap.add_argument("--domains", nargs="+", default=["bio", "cyber"], choices=["bio", "cyber", "chem"],
                    help="wmdp: which WMDP subsets")
    args = ap.parse_args()
    if args.dataset == "wmdp":
        return main_wmdp(args)
    src = f"synthetic({args.synthetic} authors)" if args.synthetic else f"{REPO_ID}@{REVISION[:12]}"
    print(f"[prepare] source {src} -> {args.out_dir} (date {args.date!r}, seed {args.seed})")
    if not args.confirm:
        print("[prepare] dry run: pass --confirm to download (~3 MB) and write the parquets")
        return 2
    tofu = synthetic(args.synthetic, args.seed) if args.synthetic else download(args.out_dir / "raw")
    frames, report = build(tofu, args.date, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report["source"] = src
    report["source_sha256"] = {} if args.synthetic else dict(SOURCES)
    report["outputs"] = {}
    for name, df in frames.items():
        path = args.out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        digest = sha256(path)
        path.with_suffix(".sha256").write_text(digest + "\n")
        report["outputs"][path.name] = {"rows": len(df), "sha256": digest}
    (args.out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("checks", "eval_counts", "eval_subject_in_question",
                                             "drops", "relearn_counts", "example")},
                     indent=2, default=str))
    names = [a["name"] for a in report["forget_authors"]]
    print(f"[prepare] forget authors: {names}")
    if any(not n for n in names):
        print("[prepare] WARNING: an author name was not extracted; name-swap pairs skip that author")
    print(f"[prepare] wrote {len(frames)} parquets + prepare_report.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
