"""WMDP (+ MMLU sanity) probe and relearning files for the unlearning experiment (PLAN.md §2-§3).

Called by ``prepare.py --dataset wmdp`` (the default). Downloads the pinned
parquet files of ``cais/wmdp`` (bio, cyber; chem optional) and ``cais/mmlu``
(all/test) on the machine that runs it — the cluster — checks each against a
pinned sha256 (the Hub's LFS oid), and writes:

  wmdp_eval.parquet       one row per MCQ item: split in {bio_A, bio_B,
                          cyber_A, cyber_B, (chem_A, chem_B), mmlu, mmlu_near},
                          options re-ordered by a seeded permutation so the
                          correct letter is uniform over items (a letter prior
                          cannot pass for knowledge), the swap partner letter
                          for the option-swap counterfactual, prompt_text in
                          the lm-evaluation-harness format (geode.adapt.render_mcq)
  relearn_<dom>A(.parquet, _val)   relearning set = the A items as TEXT FACTS
                          (description + question + "Answer:" + " <correct
                          option text>", no options shown -> no letter shortcut),
                          90/10 train/val (Deeb & Roger 2024 RTT)
  relearn_mmluA(_val)     the same format on far-domain MMLU items, same size as
                          relearn_bioA: the fine-tuning NULL for held-out
                          recovery (does ANY fine-tune on MCQ-style facts move
                          WMDP-B?)
  *.sha256, prepare_report.json

Nothing hazardous is printed or written to the repo: the report holds counts
and hashes only. ``synthetic_mcq`` builds harmless invented items for the CPU
smoke (no network).
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from geode.adapt.mcq import answer_text, balanced_permutation, render_mcq  # noqa: E402

WMDP_REPO, WMDP_REV = "cais/wmdp", "7125571f22f032c56415e7980f48d877dd830ff8"          # MIT, ungated
MMLU_REPO, MMLU_REV = "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe"          # MIT, ungated
WMDP_FILES = {  # domain -> (path, sha256 = Hub LFS oid, measured 2026-09-25)
    "bio": ("wmdp-bio/test-00000-of-00001.parquet",
            "738cee28e11e4dcf9a0efc6b2c22b425bddda4b2b69477a11f38539c86cf00b7"),
    "cyber": ("wmdp-cyber/test-00000-of-00001.parquet",
              "ec65f7ba4a9cd1ef368618ebd2b90b6d930f9b9f053f67a926b82fd71d87b4fe"),
    "chem": ("wmdp-chem/test-00000-of-00001.parquet",
             "b9e5f2be6c7526931ae8aeb2eea246baa4144d2ba8d0de1de09905aaa84912ce"),
}
MMLU_FILE = ("all/test-00000-of-00001.parquet",
             "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7")
DOMAIN_SEED = {"bio": 11, "cyber": 13, "chem": 17}
DESCRIPTION = {"bio": "biology", "cyber": "computer security", "chem": "chemistry"}
HEADER = "The following are multiple choice questions (with answers) about {}."
# MMLU subjects next to the hazard domains: the "near" sanity split (RMU's
# reported collateral damage lives here); never used for the relearning null
NEAR = ("college_biology", "high_school_biology", "virology", "medical_genetics",
        "computer_security", "college_computer_science", "college_chemistry", "high_school_chemistry")
NEAR_SUBSTR = ("bio", "virolog", "medic", "anatom", "genetic", "nutrition", "chemistry", "computer",
               "security", "machine_learning", "electrical", "physics", "clinical")


def _fetch(repo: str, rev: str, path: str, sha: str, raw: Path) -> pd.DataFrame:
    import hashlib

    from huggingface_hub import hf_hub_download

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    dst = raw / repo.replace("/", "__") / path
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(repo, path, repo_type="dataset", revision=rev), dst)
    got = sha256(dst)
    if got != sha:
        raise SystemExit(f"[prepare] sha256 mismatch for {repo}/{path}: {got} != pinned {sha}")
    return pd.read_parquet(dst)


def download(raw: Path, domains: list[str]) -> dict[str, pd.DataFrame]:
    out = {d: _fetch(WMDP_REPO, WMDP_REV, *WMDP_FILES[d], raw) for d in domains}
    out["mmlu"] = _fetch(MMLU_REPO, MMLU_REV, *MMLU_FILE, raw)
    return out


def _is_near(subject: str) -> bool:
    return subject in NEAR or any(s in subject for s in NEAR_SUBSTR)


def _mcq_rows(df: pd.DataFrame, domain: str, split_of, rng: random.Random, source: str) -> list[dict]:
    rows, seen = [], {}
    for i, r in enumerate(df.itertuples()):
        choices = list(r.choices)
        if len(choices) != 4:
            continue
        split = split_of(i, r)
        if split is None:
            continue
        target = seen.get(split, 0) % 4     # uniform correct letter within every split
        seen[split] = seen.get(split, 0) + 1
        perm = balanced_permutation(4, int(r.answer), target, rng)
        permuted = [choices[j] for j in perm]
        partner = rng.choice([k for k in range(4) if k != target])
        subject = getattr(r, "subject", None) or domain
        desc = HEADER.format(DESCRIPTION.get(domain, str(subject).replace("_", " ")))
        rows.append({
            "item_id": f"{source}:{i}", "domain": domain, "split": split, "subject": subject,
            "description": desc, "question": r.question, "choices": permuted, "perm": perm,
            "correct_idx": target, "partner_idx": partner,
            "prompt_text": render_mcq(desc, r.question, permuted),
            "answer_text": answer_text(target),
            "distractor_texts": [answer_text(partner)] + [answer_text(k) for k in range(4)
                                                         if k not in (target, partner)],
        })
    return rows


def _sft_rows(ev: pd.DataFrame) -> list[dict]:
    out = []
    for r in ev.itertuples():
        prompt = f"{r.description}\n\n{r.question.strip()}\nAnswer:"
        ans = " " + str(r.choices[r.correct_idx]).strip()
        full = prompt + ans
        out.append({"item_id": r.item_id, "prompt_text": prompt, "answer_text": ans, "full_text": full,
                    "answer_char_start": len(prompt), "answer_char_end": len(full)})
    return out


def build(src: dict[str, pd.DataFrame], seed: int, n_mmlu: int = 512) -> tuple[dict[str, pd.DataFrame], dict]:
    rng = random.Random(seed)
    rows = []
    domains = [d for d in src if d != "mmlu"]
    for d in domains:
        n = len(src[d])
        order = list(range(n))
        random.Random(seed + DOMAIN_SEED[d]).shuffle(order)   # hash() is salted per process
        in_a = set(order[: n // 2])
        rows += _mcq_rows(src[d], d, lambda i, r, a=in_a, d=d: f"{d}_A" if i in a else f"{d}_B", rng, f"wmdp-{d}")
    mm = src["mmlu"].reset_index(drop=True)
    near_idx = [i for i, s in enumerate(mm["subject"]) if _is_near(str(s))]
    far_idx = [i for i, s in enumerate(mm["subject"]) if not _is_near(str(s))]
    r2 = random.Random(seed + 7)
    r2.shuffle(far_idx)
    r2.shuffle(near_idx)
    n_bio_a = sum(1 for x in rows if x["split"] == f"{domains[0]}_A") if domains else 0
    relearn_null = set(far_idx[: max(n_bio_a, 1)])          # the fine-tuning null comes first
    general = set(far_idx[max(n_bio_a, 1) : max(n_bio_a, 1) + n_mmlu])
    near = set(near_idx[:n_mmlu])

    def mmlu_split(i, r):
        return "mmlu" if i in general else "mmlu_near" if i in near else "mmlu_relearn" if i in relearn_null else None

    rows += _mcq_rows(mm, "mmlu", mmlu_split, rng, "mmlu")
    ev = pd.DataFrame(rows)
    frames = {"wmdp_eval": ev[ev["split"] != "mmlu_relearn"].reset_index(drop=True)}

    def sft(split: str, name: str):
        part = ev[ev["split"] == split]
        if not len(part):
            return
        s = pd.DataFrame(_sft_rows(part))
        k = max(1, len(s) // 10)
        frames[f"relearn_{name}_val"] = s.iloc[:k].reset_index(drop=True)
        frames[f"relearn_{name}"] = s.iloc[k:].reset_index(drop=True)

    for d in domains:
        sft(f"{d}_A", f"{d}A")
    sft("mmlu_relearn", "mmluA")
    fe = frames["wmdp_eval"]
    report = {
        "dataset": "wmdp", "seed": seed,
        "sources": {"wmdp": f"{WMDP_REPO}@{WMDP_REV}", "mmlu": f"{MMLU_REPO}@{MMLU_REV}"},
        "eval_counts": fe.groupby("split").size().to_dict(),
        "correct_letter_counts": fe.groupby(["split", "correct_idx"]).size().unstack(fill_value=0)
        .to_dict(orient="index"),
        "relearn_counts": {k: len(v) for k, v in frames.items() if k.startswith("relearn")},
        "near_subjects_excluded_from_null": sorted({str(s) for s in mm["subject"] if _is_near(str(s))}),
    }
    return frames, report


def synthetic_mcq(n_per_domain: int, seed: int) -> dict[str, pd.DataFrame]:
    """Harmless invented MCQ items shaped like WMDP/MMLU (CPU smoke, no network)."""
    rng = random.Random(seed)
    colours = ["red", "blue", "green", "yellow", "purple", "orange", "grey", "white"]
    things = ["the glass lantern", "the old bicycle", "the paper kite", "the wooden boat",
              "the tin whistle", "the wool scarf", "the clay jug", "the silk ribbon"]

    def items(tag: str, n: int, subject: str | None = None):
        rows = []
        for i in range(n):
            thing = things[i % len(things)]
            cs = rng.sample(colours, 4)
            q = f"In the {tag} story number {i}, what colour is {thing}?"
            rows.append({"question": q, "choices": cs, "answer": rng.randrange(4),
                         **({"subject": subject} if subject else {})})
        return pd.DataFrame(rows)

    subjects = ["high_school_geography", "philosophy", "virology", "world_religions"]
    mm = pd.concat([items(f"mmlu-{s}", 3 * n_per_domain, s) for s in subjects], ignore_index=True)
    return {"bio": items("bio", n_per_domain), "cyber": items("cyber", n_per_domain), "mmlu": mm}
