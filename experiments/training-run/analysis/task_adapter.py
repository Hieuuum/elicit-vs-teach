"""Task adapter: run the circuit / lens / prefit tools on a task other than arithmetic.

The tools were built around the arithmetic probe (``premise_checks.render_probe``
over ``D_algo_eval``, integer answers, a sign trick). Each of them now takes
``--task`` (default ``arith``: the original code path, untouched) and, for any
other task, asks this module for the four things it needs:

  items(tokenizer)            ScoredItem list: prompt ids, the correct next token
                              (aligned IN CONTEXT, geode.adapt), same-slot
                              distractor tokens, meta (subject, positions)
  pairs(tokenizer, n)         (clean_ids, corrupt_ids, clean_tok, contrast_tok),
                              equal length, different tokens (geode.adapt.check_pairs)
  role_pairs(tokenizer, n)    DCM counterfactuals: (clean, cf, target, distractor),
                              the cf target is the model's OWN output on cf
  loss_items(tokenizer, n)    (prompt ids, answer ids + EOS) for the SFT-loss
                              metrics (curvature, gradients)

Tasks
  wmdp   WMDP / MMLU multiple choice (experiments/unlearning/data/prepare_wmdp.py;
         the PRIMARY unlearning design). Scored token = the correct LETTER
         (" A".." D", in context after "Answer:"); distractors = the other three
         letters, the swap partner first. Splits: bio, cyber, chem (= _A + _B),
         bio_A, bio_B, cyber_A, ..., mmlu, mmlu_near. Pairs: ``swap`` (default:
         the same question with the correct option's text exchanged with the
         partner's; contrast = the partner letter, which is the counterfactual's
         correct letter) or ``item`` (a different question of equal length).
         DCM role ``option`` uses the swap with a LABELLED target. No subject
         position; the probe is a 4-way linear probe of the correct letter.
  tofu   TOFU fictitious-author QA (experiments/unlearning/data/prepare.py). The
         scored token is the first token of the answer's fact word after the
         original answer's prefix; the distractor is TOFU's perturbed word in
         the same slot. Splits: forget_A, forget_B, forget (= A + B), retain,
         null (forget items with the author's name swapped for an invented,
         equal-length name: nobody can know the answer -> the noise floor).
         Pair modes: ``swap`` (default: corrupt = same prompt, author name
         swapped for an invented one; contrast = the perturbed word) and
         ``item`` (arithmetic-style: a different item of equal prompt length;
         contrast = that item's own answer token).
"""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from geode.adapt import (  # noqa: E402
    LETTERS,
    AlignmentError,
    ScoredItem,
    check_pairs,
    encode,
    first_answer_token,
    fresh_names,
    length_matched_pairs,
    render_mcq,
    score_item,
    swap_options,
    swap_pairs,
    swap_subject,
)

TASKS = ("arith", "tofu", "wmdp")
SPLITS = {"tofu": ("forget_A", "forget_B", "forget", "retain", "null"),
          "wmdp": ("bio", "cyber", "chem", "bio_A", "bio_B", "cyber_A", "cyber_B", "chem_A", "chem_B",
                   "mmlu", "mmlu_near")}


def add_task_args(ap, split_default: str = "forget") -> None:
    """The shared CLI block (every tool gets the same flags)."""
    ap.add_argument("--task", choices=TASKS, default="arith",
                    help="arith (default, original code path) or tofu (task_adapter.py)")
    ap.add_argument("--task-data", default=None,
                    help="tofu: directory written by experiments/unlearning/data/prepare.py")
    ap.add_argument("--task-split", default=split_default,
                    help="tofu: forget_A forget_B forget retain null; wmdp: bio cyber chem "
                    "bio_A bio_B cyber_A cyber_B mmlu mmlu_near")
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="wmdp: drop items whose prompt exceeds this many tokens (0 = no cap)")
    ap.add_argument("--pair-mode", choices=("swap", "item"), default="swap",
                    help="tofu: corrupt = author name swapped (default) or a different item")
    ap.add_argument("--task-seed", type=int, default=316)


def resolve_tokenizer(args, model_spec: str | None, arith_default: str) -> str:
    """--tokenizer if given; else the arithmetic default (unchanged) for
    --task arith, the model's own tokenizer for any other task."""
    tok = getattr(args, "tokenizer", None)
    if tok:
        return tok
    if getattr(args, "task", "arith") == "arith":
        return arith_default
    if not model_spec:
        raise SystemExit("[task] cannot resolve a tokenizer: pass --tokenizer")
    return model_spec


def load_tokenizer(source: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


class LossItem:
    """Prompt ids + answer ids (the SFT loss target) for curvature / gradients."""

    __slots__ = ("prompt_ids", "answer_ids")

    def __init__(self, prompt_ids: list[int], answer_ids: list[int]):
        self.prompt_ids, self.answer_ids = prompt_ids, answer_ids


def _verify(path: Path) -> None:
    side = path.with_suffix(".sha256")
    if side.is_file():
        want = side.read_text().strip()
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[task] {path.name}: sha256 {got[:12]} != sidecar {want[:12]} "
                             "(rebuild with prepare.py)")


class QATask:
    """TOFU-style short-fact QA (see module docstring)."""

    name = "tofu"
    roles = ("subject",)
    dcm_target = "cf_dist"        # the name-swapped author's answer is unknown
    das_pair_mode = "item"
    probe_kind = "subject_lens"
    has_subject = True

    def __init__(self, data_dir: str | Path, split: str = "forget", pair_mode: str = "swap",
                 seed: int = 316):
        import pandas as pd

        path = Path(data_dir) / "tofu_eval.parquet"
        if not path.is_file():
            raise SystemExit(f"[task] {path} missing: run experiments/unlearning/data/prepare.py")
        _verify(path)
        self.df_all = pd.read_parquet(path)
        self.split, self.pair_mode, self.seed = split, pair_mode, seed
        base = "forget" if split == "null" else split
        if base == "forget":
            self.df = self.df_all[self.df_all["split"].isin(["forget_A", "forget_B"])]
        else:
            self.df = self.df_all[self.df_all["split"] == base]
        if not len(self.df):
            raise SystemExit(f"[task] split {split!r}: no rows in {path}")
        text = " ".join(self.df_all["question"].tolist() + self.df_all["answer"].tolist())
        self.fresh = fresh_names(text, seed)
        self._cache: dict = {}

    # ---------------------------------------------------------------- items
    def items(self, tokenizer, n: int | None = None) -> list[ScoredItem]:
        key = ("items", id(tokenizer))
        if key not in self._cache:
            out, drops = [], 0
            for r in self.df.itertuples():
                prompt = r.prompt_text
                meta = {"subject": r.subject, "split": r.split, "fact_word": r.fact_word,
                        "question_prompt": r.prompt_question, "answer": r.answer,
                        "para_prompt": r.para_prompt_text}
                if self.split == "null":
                    got = swap_subject(tokenizer, prompt, r.subject, self.fresh)
                    if got is None:
                        drops += 1
                        continue
                    prompt, meta["subject"] = got[0], got[2]
                    meta["question_prompt"] = r.prompt_question.replace(r.subject, got[2])
                try:
                    it = score_item(tokenizer, r.item_id, prompt, r.answer_text,
                                    list(r.distractor_texts), meta)
                except AlignmentError:
                    drops += 1
                    continue
                out.append(it)
            if drops:
                print(f"[task] tofu/{self.split}: {drops} of {len(self.df)} rows dropped "
                      "(alignment / no equal-length name swap)")
            self._cache[key] = out
        items = self._cache[key]
        return items if n is None else items[:n]

    # ---------------------------------------------------------------- pairs
    def pairs(self, tokenizer, n_pairs: int, mode: str | None = None):
        mode = mode or self.pair_mode
        items = self.items(tokenizer)
        if mode == "swap":
            pairs = swap_pairs(tokenizer, items, self.fresh, n_pairs, self.seed)
        else:
            pairs = length_matched_pairs(items, n_pairs, self.seed, partners=8)
        check_pairs(pairs)
        if len(pairs) < n_pairs:
            print(f"[task] tofu/{self.split}/{mode}: {len(pairs)} pairs (asked {n_pairs})")
        return pairs

    def role_pairs(self, tokenizer, role: str, n: int):
        """DCM 'subject' role: cf = the name swapped for an invented one; the
        4th element is the primary DISTRACTOR (for the logit-diff criterion),
        the optimisation target is the model's own cf output (dcm_roles
        cf_target='cf_dist')."""
        if role != "subject":
            raise SystemExit(f"[task] tofu has no role {role!r} (roles: {self.roles})")
        return self.pairs(tokenizer, n, mode="swap")

    # ---------------------------------------------------------------- surfaces / loss
    def surface_prompts(self, tokenizer, n: int) -> tuple[list[str], list[str]]:
        """(question surface, paraphrased-question surface), same answer prefix."""
        items = self.items(tokenizer, n)
        return [it.prompt for it in items], [it.meta["para_prompt"] for it in items]

    def loss_items(self, tokenizer, n: int | None = None) -> list[LossItem]:
        out = []
        for it in self.items(tokenizer, n):
            p = it.meta["question_prompt"]
            p_ids = encode(tokenizer, p)
            full = encode(tokenizer, p + it.meta["answer"])
            if full[: len(p_ids)] != p_ids:
                continue
            out.append(LossItem(p_ids, full[len(p_ids):] + [tokenizer.eos_token_id]))
        return out

    def subject_last_position(self, tokenizer, item: ScoredItem) -> int | None:
        """Index (in item.prompt_ids) of the last token of the subject's FIRST
        mention (the question), or None when the question does not name it."""
        subj = item.meta.get("subject") or ""
        k = item.prompt.find(subj) if subj else -1
        if k < 0 or k > len(item.meta["question_prompt"]):
            return None
        end = k + len(subj)
        enc = tokenizer(item.prompt, add_special_tokens=False, return_offsets_mapping=True)
        pos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end and e > k]
        return pos[-1] if pos else None


class MCQTask:
    """WMDP / MMLU multiple choice, letter-scored (module docstring)."""

    name = "wmdp"
    roles = ("option",)
    dcm_target = "label"          # the swapped item's correct letter is known
    das_pair_mode = "swap"
    probe_kind = "letter"
    has_subject = False

    def __init__(self, data_dir: str | Path, split: str = "bio", pair_mode: str = "swap",
                 seed: int = 316, max_prompt_tokens: int = 0):
        import pandas as pd

        path = Path(data_dir) / "wmdp_eval.parquet"
        if not path.is_file():
            raise SystemExit(f"[task] {path} missing: run experiments/unlearning/data/prepare.py --dataset wmdp")
        _verify(path)
        df = pd.read_parquet(path)
        if split not in SPLITS["wmdp"]:
            raise SystemExit(f"[task] wmdp split {split!r} not in {SPLITS['wmdp']}")
        self.df = df[df["split"].isin([f"{split}_A", f"{split}_B"])] if split in ("bio", "cyber", "chem") \
            else df[df["split"] == split]
        if not len(self.df):
            raise SystemExit(f"[task] wmdp split {split!r}: no rows in {path}")
        self.split, self.pair_mode, self.seed, self.max_tok = split, pair_mode, seed, max_prompt_tokens
        self._cache: dict = {}

    def items(self, tokenizer, n: int | None = None) -> list[ScoredItem]:
        key = ("items", id(tokenizer))
        if key not in self._cache:
            out, too_long, bad = [], 0, 0
            for r in self.df.itertuples():
                choices = list(r.choices)
                prompt = render_mcq(r.description, r.question, choices)
                if prompt != r.prompt_text:
                    raise SystemExit(f"[task] {r.item_id}: stored prompt != render_mcq (stale data)")
                meta = {"description": r.description, "question": r.question, "choices": choices,
                        "correct_idx": int(r.correct_idx), "partner_idx": int(r.partner_idx),
                        "split": r.split, "question_prompt": prompt,
                        "answer": " " + str(choices[int(r.correct_idx)]).strip()}
                try:
                    it = score_item(tokenizer, r.item_id, prompt, r.answer_text, list(r.distractor_texts), meta)
                except AlignmentError:
                    bad += 1
                    continue
                if self.max_tok and len(it.prompt_ids) > self.max_tok:
                    too_long += 1
                    continue
                if len(it.distractors) != len(LETTERS) - 1:
                    bad += 1
                    continue
                out.append(it)
            if too_long or bad:
                print(f"[task] wmdp/{self.split}: {too_long} items over {self.max_tok} tokens and "
                      f"{bad} misaligned dropped of {len(self.df)}")
            self._cache[key] = out
        items = self._cache[key]
        return items if n is None else items[:n]

    def _swapped(self, tokenizer, it: ScoredItem):
        m = it.meta
        choices = swap_options(m["choices"], m["correct_idx"], m["partner_idx"])
        ids = encode(tokenizer, render_mcq(m["description"], m["question"], choices))
        return ids if len(ids) == len(it.prompt_ids) else None

    def pairs(self, tokenizer, n_pairs: int, mode: str | None = None):
        mode = mode or self.pair_mode
        items = self.items(tokenizer)
        if mode == "swap":
            import random as _random

            order = list(items)
            _random.Random(self.seed).shuffle(order)
            pairs, skipped = [], 0
            for it in order:
                x = self._swapped(tokenizer, it)
                if x is None:
                    skipped += 1
                    continue
                pairs.append((it.prompt_ids, x, it.target, it.distractors[0]))
                if len(pairs) >= n_pairs:
                    break
            if skipped:
                print(f"[task] wmdp/{self.split}: {skipped} option swaps changed the token length (skipped)")
        else:
            pairs = length_matched_pairs(items, n_pairs, self.seed, partners=8)
        check_pairs(pairs)
        if len(pairs) < n_pairs:
            print(f"[task] wmdp/{self.split}/{mode}: {len(pairs)} pairs (asked {n_pairs})")
        return pairs

    def role_pairs(self, tokenizer, role: str, n: int):
        if role != "option":
            raise SystemExit(f"[task] wmdp has no role {role!r} (roles: {self.roles})")
        return self.pairs(tokenizer, n, mode="swap")

    def surface_prompts(self, tokenizer, n: int):
        raise SystemExit("[task] wmdp has no second surface (M8 latent reach is not applicable; PLAN.md)")

    def loss_items(self, tokenizer, n: int | None = None) -> list[LossItem]:
        """SFT loss on the correct letter (the MCQ answer the model is scored on)."""
        return [LossItem(it.prompt_ids, [it.target]) for it in self.items(tokenizer, n)]

    def subject_last_position(self, tokenizer, item: ScoredItem) -> int | None:
        return None


@lru_cache(maxsize=8)
def _load(task: str, data: str, split: str, pair_mode: str, seed: int, max_tok: int = 0):
    if not data:
        raise SystemExit(f"[task] --task {task} needs --task-data <prepare.py out dir>")
    if task == "tofu":
        if split not in SPLITS["tofu"]:
            raise SystemExit(f"[task] tofu split {split!r} not in {SPLITS['tofu']}")
        return QATask(data, split, pair_mode, seed)
    if task == "wmdp":
        return MCQTask(data, split, pair_mode, seed, max_tok)
    raise SystemExit(f"[task] unknown task {task!r}")


def make_task(task: str, data: str | Path, split: str, pair_mode: str = "swap", seed: int = 316,
              max_prompt_tokens: int = 0):
    """Programmatic constructor (relearn.py probes, tests)."""
    return _load(task, str(data), split, pair_mode, seed, max_prompt_tokens)


def load_task(args):
    """The task object for non-arith runs, or None for the arithmetic default."""
    if getattr(args, "task", "arith") == "arith":
        return None
    return _load(args.task, args.task_data or "", args.task_split, args.pair_mode, args.task_seed,
                 getattr(args, "max_prompt_tokens", 0))


def first_token(tokenizer, prompt: str, answer: str) -> int:
    """Convenience re-export: in-context first answer token."""
    return first_answer_token(tokenizer, prompt, answer)[1]
