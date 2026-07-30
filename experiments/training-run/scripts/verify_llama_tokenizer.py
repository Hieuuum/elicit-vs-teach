"""Pre-launch check: Llama-3.2-1B's tokenizer against our span machinery.

Run this on the box before any run-9 training (owner note, run9 config
header). It never trains anything and never confirms a cost — it only
verifies four things a bad tokenizer would otherwise break silently:

(a) pad token — Llama-3.2-1B ships none; the eos fallback (same glue as
    train_sft.py/train_target.py/gates.py) must resolve one, or FAIL.
(b) ``len(tokenizer) == config.vocab_size`` — the launcher's arch guard
    (train_sft.py/train_target.py) assumes this.
(c) ``geode.arith.spans.tokenize_with_spans`` over a seeded sample of each
    frozen parquet the Llama chain uses (D_inst.parquet for run 9,
    D_target.parquet for run 10; laptop-local copy if present, else pulled
    from the public dataset repo), deliberately including negative-answer
    (subtraction) rows — the merged "-<digit>" sign token is the sharpest
    edge case for a new tokenizer (geode/arith/spans.py module docstring).
(d) the hub download itself: a 401/403/gated error becomes a friendly
    "accept the license, then hf auth login" message, not a raw traceback.

Nonzero exit on any FAIL. Read/verify only — writes nothing, trains nothing.

Usage:
    python3 verify_llama_tokenizer.py [--model-id meta-llama/Llama-3.2-1B]
        [--n-sample 300] [--seed 316]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from geode.arith import tokenize_with_spans

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
EXP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXP_DIR / "data" / "full"
# Public frozen-dataset repo — the same source train_sft.py/train_target.py
# pull from (data.hf_id in the run configs).
DATASET_REPO = "mhieuuu/elicit-vs-teach-arith"
# The two frozen files the Llama chain trains on: run 9 (format install,
# D_inst) and run 10 (target, D_target).
CHAIN_FILES = ("D_inst", "D_target")


def resolve_parquet(name: str) -> Path:
    """Laptop-local copy if present, else the public hub copy (fresh box).

    data/full/ is gitignored (generated on the laptop), so a fresh clone has
    only report.json — the box gets the parquet the same way the trainers do.
    """
    local = DATA_DIR / f"{name}.parquet"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    print(f"[verify] {name}: no local copy under {DATA_DIR} — pulling from {DATASET_REPO}")
    return Path(hf_hub_download(DATASET_REPO, f"{name}.parquet", repo_type="dataset"))


def load_hub_artifacts(model_id: str):
    """Load tokenizer + config for ``model_id``, or exit with a friendly message.

    A gated repo the account hasn't been granted access to, or a missing/bad
    token, is turned into an actionable message instead of a raw traceback
    (item d). ``transformers`` re-wraps the hub's ``GatedRepoError`` as a
    plain ``OSError``, so that is what must be caught here.
    """
    from transformers import AutoConfig, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
    except OSError as e:
        raise SystemExit(
            f"[verify] cannot load {model_id!r} from the hub ({e}) — accept the Meta "
            "Llama license on this HF account, then `hf auth login --force`"
        ) from e
    return tokenizer, config


def check_pad_token(tokenizer) -> bool:
    """Item (a): report pad_token_id, apply the eos fallback, FAIL if both are missing."""
    print(f"[verify] pad_token_id (native) = {tokenizer.pad_token_id}")
    if tokenizer.pad_token_id is not None:
        print("[verify] pad token: PASS (native pad token present)")
        return True
    if tokenizer.eos_token_id is None:
        print("[verify] pad token: FAIL — no pad token and no eos token to fall back to")
        return False
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[verify] pad token: PASS (fell back to eos, pad_token_id={tokenizer.pad_token_id})")
    return True


def check_vocab_guard(tokenizer, config) -> bool:
    """Item (b): the launcher's arch guard assumes len(tokenizer) == config.vocab_size."""
    got, want = len(tokenizer), config.vocab_size
    if got == want:
        print(f"[verify] vocab guard: PASS (len(tokenizer) == config.vocab_size == {got})")
        return True
    print(f"[verify] vocab guard: FAIL — len(tokenizer)={got} != config.vocab_size={want}")
    return False


def sample_rows(df, n: int, seed: int):
    """Seeded sample of ``n`` rows, plus every negative-``true_answer`` row.

    Negative answers (subtraction) exercise the merged "-<digit>" sign token
    — the sharpest span edge case (geode/arith/spans.py module docstring) —
    so they are included deliberately rather than left to chance.
    """
    rng = random.Random(seed)
    base = rng.sample(range(len(df)), min(n, len(df)))
    neg_positions = [i for i, ans in enumerate(df["true_answer"]) if ans < 0]
    rows = sorted(set(base) | set(neg_positions))
    return df.iloc[rows]


def check_spans(name: str, tokenizer, *, n: int, seed: int) -> bool:
    """Item (c): tokenize_with_spans over a sample of one frozen parquet."""
    import pandas as pd

    df = pd.read_parquet(resolve_parquet(name))
    rows = sample_rows(df, n, seed)
    texts = rows["full_text"].tolist()
    char_spans = list(
        zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int))
    )
    n_neg = int((rows["true_answer"] < 0).sum())
    try:
        tokenize_with_spans(texts, char_spans, tokenizer, append_eos=True)
    except ValueError:
        # Retry one row at a time to name the offending row precisely.
        for text, span in zip(texts, char_spans):
            try:
                tokenize_with_spans([text], [span], tokenizer, append_eos=True)
            except ValueError as row_err:
                print(f"[verify] {name}: FAIL on row {text!r}: {row_err}")
                return False
        print(f"[verify] {name}: FAIL — span error did not reproduce row-by-row")
        return False
    print(f"[verify] {name}: PASS ({len(rows)} rows sampled, {n_neg} negative-answer)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--n-sample", type=int, default=300, help="rows sampled per parquet")
    parser.add_argument("--seed", type=int, default=316)
    args = parser.parse_args()

    tokenizer, config = load_hub_artifacts(args.model_id)

    checks = [check_pad_token(tokenizer), check_vocab_guard(tokenizer, config)]
    for name in CHAIN_FILES:
        checks.append(check_spans(name, tokenizer, n=args.n_sample, seed=args.seed))

    if all(checks):
        print("[verify] OVERALL: PASS")
        return 0
    print("[verify] OVERALL: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
