"""Clean/counterfactual pairs for the EAP-IG circuit-validation pipeline
(PLAN.md "Frozen decisions" + "Data"). CPU-only, tokenizer-only — never runs
a model forward pass.

Three task families, all teacher-forced two-token completions so
``geode.circuits.eapig.logit_diff_2tok`` can score them directly:

- ``build_addition_pairs`` — the main task: 4+4 digit addition in the frozen
  word-task template, drawn from ``D_algo_eval_bare`` (op='+', cell='4x4',
  5,000 rows).
- ``build_copy_pairs`` — the specificity control (a non-arithmetic task that
  shares the task's "read a 4-digit number, emit it" surface).
- ``build_tinystories`` — the sanity-check sanity corpus (next-token loss
  only, no clean/corrupt pairing).

Every returned sequence is laid out ``prompt + first_answer_token`` (length
T = prompt_len + 1), matching ``eap_ig_scores``/``logit_diff_2tok``: the
model's logits at position T-2 (last prompt token) predict the first answer
token, and at T-1 (the appended token) predict the second, so
``logit_diff_2tok(logits, c1, k1, c2, k2)`` is the teacher-forced LD directly
off these tensors with no further slicing.

``save_pairs``/``load_pairs`` double as the on-disk contract ``run.py`` (the
GPU-side driver) reads: ``load_pairs`` given a *file* loads exactly that
saved dict; given a *directory* (``run.py --data-dir``, default
``experiments/eapig-circuit-check/data/``) it assembles the combined
``{"disc", "val", "copy", "stories"}`` dict the driver indexes directly
(``data["val"]["answer_text"]``, ``data["disc"]["half"]``, ``data["stories"]``
as a bare token tensor), by loading the three fixed filenames below and
splitting the addition pairs file's rows by ``meta["split"]``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ADDITION_PROMPT = "What is the sum of {a} and {b}?\n"
COPY_PROMPT = "Tom had {n} apples. How many apples did Tom have?\n"

# Fixed filenames `load_pairs(directory)` looks for (see module docstring).
ADDITION_PAIRS_FILENAME = "pairs_seed0.pt"
COPY_PAIRS_FILENAME = "copy_pairs_seed1.pt"
TINYSTORIES_FILENAME = "tinystories_seed2.pt"


# ---------------------------------------------------------------------------
# Pure helpers (no tokenizer, no parquet — unit-testable on tiny arrays)
# ---------------------------------------------------------------------------


def pick_counterfactual(
    match_key: np.ndarray,
    first_tok: np.ndarray,
    second_tok: np.ndarray,
    idx: int,
    rng: np.random.Generator,
    prefer: np.ndarray | None = None,
) -> int:
    """Return the index of a valid counterfactual for row ``idx``.

    A valid counterfactual ``j`` has ``j != idx``, ``match_key[j] ==
    match_key[idx]`` (e.g. same digit-count sum), and both answer tokens
    differ: ``first_tok[j] != first_tok[idx]`` and ``second_tok[j] !=
    second_tok[idx]``. When ``prefer`` is given and at least one match
    satisfies it (e.g. "outside the clean sets"), the draw is restricted to
    those; otherwise every match is eligible. Deterministic given ``rng``.
    Raises ``ValueError`` if no valid counterfactual exists.
    """
    n = len(match_key)
    if len(first_tok) != n or len(second_tok) != n:
        raise ValueError("pick_counterfactual: match_key/first_tok/second_tok length mismatch")
    if not (0 <= idx < n):
        raise ValueError(f"pick_counterfactual: idx {idx} out of range for length {n}")
    mask = (
        (match_key == match_key[idx])
        & (first_tok != first_tok[idx])
        & (second_tok != second_tok[idx])
    )
    mask[idx] = False
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError(f"pick_counterfactual: no valid counterfactual for index {idx}")
    if prefer is not None:
        preferred = candidates[prefer[candidates]]
        if preferred.size > 0:
            candidates = preferred
    return int(rng.choice(candidates))


def answer_tokens(tokenizer, text: str) -> list[int]:
    """Tokenize an answer span the way training does (no special tokens, no
    BOS) and enforce the fixed-answer-width assumption the circuit pipeline
    relies on: exactly 2 tokens, the tokens decode back to ``text`` exactly,
    and the first token is not whitespace-only (so it never collides with a
    pure space/newline token). Raises ``ValueError`` on any violation.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 2:
        raise ValueError(f"answer_tokens: {text!r} tokenized to {len(ids)} tokens {ids}, expected 2")
    decoded = tokenizer.decode(ids)
    if decoded != text:
        raise ValueError(f"answer_tokens: {text!r} does not round-trip (decoded {decoded!r})")
    first = tokenizer.decode([ids[0]])
    if first.strip() == "" or first.startswith(" "):
        raise ValueError(f"answer_tokens: {text!r} first token decodes to {first!r}, looks like whitespace")
    return ids


def _tensor_sha256(d: dict) -> str:
    """sha256 over every ``torch.Tensor`` value in ``d``, keyed by sorted
    field name so the hash is order-independent and reproducible."""
    h = hashlib.sha256()
    for k in sorted(d):
        v = d[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_pairs(pairs: dict, path: str | Path) -> None:
    """Save a pairs dict (as returned by ``build_addition_pairs`` /
    ``build_copy_pairs`` / ``build_tinystories``) to a single ``.pt`` file, so
    a GPU box and a laptop load byte-identical tensors. Records a sha256 over
    the token tensors under ``_token_sha256``; ``load_pairs`` re-verifies it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(pairs)
    payload["_token_sha256"] = _tensor_sha256(pairs)
    torch.save(payload, path)


def _load_single(path: Path) -> dict:
    """Load one pairs dict saved by ``save_pairs``, verifying its token
    sha256. Raises ``ValueError`` if the recorded hash does not match the
    loaded tensors (corrupted file, or saved by a different build)."""
    payload = torch.load(path, weights_only=False)
    expected = payload.pop("_token_sha256", None)
    if expected is not None:
        got = _tensor_sha256(payload)
        if got != expected:
            raise ValueError(f"load_pairs: token sha256 mismatch in {path}: {got} != {expected}")
        payload["_token_sha256"] = expected
    return payload


def _pairs_subset(pairs: dict, mask: np.ndarray) -> dict:
    """Row-slice every per-example field of a pairs dict by a boolean
    ``mask``: a ``torch.Tensor`` by its first dimension, the ``meta``
    DataFrame by row, and any other per-example list (e.g. ``answer_text``)
    by boolean compress. Non-per-example values pass through unchanged."""
    n = len(mask)
    mask_t = torch.as_tensor(np.asarray(mask, dtype=bool).copy())
    out: dict = {}
    for k, v in pairs.items():
        if k == "_token_sha256":
            continue
        if isinstance(v, torch.Tensor) and v.shape[0] == n:
            out[k] = v[mask_t]
        elif isinstance(v, pd.DataFrame) and len(v) == n:
            out[k] = v.loc[mask].reset_index(drop=True)
        elif isinstance(v, list) and len(v) == n:
            out[k] = [x for x, keep in zip(v, mask) if keep]
        else:
            out[k] = v
    return out


def _load_pairs_dir(directory: Path) -> dict:
    """Assemble the combined ``{"disc", "val", "copy", "stories"}`` dict
    ``run.py`` expects (module docstring) from the three fixed filenames in
    ``directory``. ``stories`` is the raw TinyStories token tensor (not the
    wrapping dict) and is omitted if that file isn't present (it needs
    network to build; see ``build_tinystories``)."""
    addition = _load_single(directory / ADDITION_PAIRS_FILENAME)
    meta = addition["meta"]
    is_disc = (meta["split"] == "discovery").to_numpy()
    is_val = (meta["split"] == "validation").to_numpy()
    combined = {
        "disc": _pairs_subset(addition, is_disc),
        "val": _pairs_subset(addition, is_val),
        "copy": _load_single(directory / COPY_PAIRS_FILENAME),
    }
    stories_path = directory / TINYSTORIES_FILENAME
    if stories_path.is_file():
        combined["stories"] = _load_single(stories_path)["input_ids"]
    return combined


def load_pairs(path: str | Path) -> dict:
    """Load pairs saved by ``save_pairs``.

    Given a *file*, loads exactly that saved dict (verifying its token
    sha256). Given a *directory*, assembles the combined ``{"disc", "val",
    "copy", "stories"}`` dict ``run.py`` consumes directly — see the module
    docstring and ``_load_pairs_dir``.
    """
    path = Path(path)
    if path.is_dir():
        return _load_pairs_dir(path)
    return _load_single(path)


def _validate_pool_tokenization(
    tokenizer, prompts: Sequence[str], answers: Sequence[str], fulls: Sequence[str], *, where: str
) -> tuple[list[list[int]], list[list[int]]]:
    """Batch-tokenize ``prompts``/``answers``/``fulls`` and enforce, for every
    row: ``tokenize(full) == tokenize(prompt) + tokenize(answer)``, a shared
    prompt token length, and the ``answer_tokens`` invariants (2 tokens,
    round-trips, first token not whitespace). Returns ``(prompt_ids,
    answer_ids)``. ``where`` only labels error messages.
    """
    prompt_ids = tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(list(answers), add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(list(fulls), add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids[0])
    for i in range(len(prompts)):
        if full_ids[i] != prompt_ids[i] + answer_ids[i]:
            raise ValueError(
                f"{where}: row {i} tokenize(full_text) != tokenize(prompt_text) + "
                f"tokenize(answer_text): {full_ids[i]} != {prompt_ids[i]} + {answer_ids[i]}"
            )
        if len(prompt_ids[i]) != prompt_len:
            raise ValueError(
                f"{where}: row {i} prompt token length {len(prompt_ids[i])} != {prompt_len}"
            )
        if len(answer_ids[i]) != 2:
            raise ValueError(
                f"{where}: row {i} answer {answers[i]!r} is not 2 tokens: {answer_ids[i]}"
            )
        if tokenizer.decode(answer_ids[i]) != answers[i]:
            raise ValueError(f"{where}: row {i} answer {answers[i]!r} does not round-trip")
        first = tokenizer.decode([answer_ids[i][0]])
        if first.strip() == "" or first.startswith(" "):
            raise ValueError(f"{where}: row {i} first answer token decodes to {first!r}, looks like whitespace")
    return prompt_ids, answer_ids


def _assemble_pairs(
    prompt_ids: list[list[int]],
    answer_ids: list[list[int]],
    clean_idx: np.ndarray,
    match_key: np.ndarray,
    rng: np.random.Generator,
    prefer: np.ndarray,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int], list[int], list[int], list[int]]:
    """Draw one counterfactual per ``clean_idx`` row and assemble the
    ``prompt + first_answer_token`` sequences. Returns ``(clean_ids,
    corrupt_ids, c1, c2, k1, k2, cf_idx)`` as plain Python lists.
    """
    c1_all = np.array([a[0] for a in answer_ids], dtype=np.int64)
    c2_all = np.array([a[1] for a in answer_ids], dtype=np.int64)
    clean_ids_list: list[list[int]] = []
    corrupt_ids_list: list[list[int]] = []
    c1_list: list[int] = []
    c2_list: list[int] = []
    k1_list: list[int] = []
    k2_list: list[int] = []
    cf_idx_list: list[int] = []
    for i in clean_idx:
        i = int(i)
        cf_i = pick_counterfactual(match_key, c1_all, c2_all, i, rng, prefer=prefer)
        if cf_i == i:
            raise ValueError(f"_assemble_pairs: counterfactual for row {i} reused itself")
        clean_ids_list.append(prompt_ids[i] + [int(c1_all[i])])
        corrupt_ids_list.append(prompt_ids[cf_i] + [int(c1_all[cf_i])])
        c1_list.append(int(c1_all[i]))
        c2_list.append(int(c2_all[i]))
        k1_list.append(int(c1_all[cf_i]))
        k2_list.append(int(c2_all[cf_i]))
        cf_idx_list.append(cf_i)
    return clean_ids_list, corrupt_ids_list, c1_list, c2_list, k1_list, k2_list, cf_idx_list


# ---------------------------------------------------------------------------
# Addition task
# ---------------------------------------------------------------------------


def build_addition_pairs(
    eval_parquet: str | Path,
    tokenizer,
    n_disc: int = 512,
    n_val: int = 256,
    seed: int = 0,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> dict:
    """Build the 4+4 addition clean/counterfactual pairs (PLAN.md "Pair
    pool"): ``D_algo_eval_bare`` rows with ``op == '+'`` and ``cell ==
    '4x4'`` (5,000 rows), template ``"What is the sum of {a} and {b}?\\n"``
    -> ``"{a+b}"``.

    Every row in the pool is tokenization-validated (``tokenize(full_text)
    == tokenize(prompt_text) + tokenize(answer_text)``, one shared prompt
    length, exactly 2 answer tokens, round-trip, first token not
    whitespace) since any pool row can be drawn as either a clean problem or
    a counterfactual.

    The pool is shuffled once (seeded); the first ``n_disc`` rows are
    "discovery" (split into two seeded halves, 0/1, for the split-half
    ceiling), the next ``n_val`` are "validation" — disjoint from discovery
    in operand pairs because the pool itself has no duplicate unordered
    ``{a, b}`` pairs (asserted below) and the index ranges don't overlap.
    Each used row draws one counterfactual: another pool row with the same
    sum digit-count whose answer tokens differ at both positions, preferring
    rows outside the 512+256 "clean" set and never reusing the row itself.

    Returns a dict: ``clean_ids``/``corrupt_ids`` (N, T) int64 tensors
    (prompt + first answer token); ``c1``/``c2``/``k1``/``k2`` (N,) int64
    tensors; and ``meta`` — a DataFrame with columns ``a, b, sum, cf_a,
    cf_b, cf_sum, split`` ("discovery"/"validation") and ``half`` (0/1 for
    discovery rows, -1 for validation). N = n_disc + n_val.
    """
    if n_disc % 2 != 0:
        raise ValueError(f"build_addition_pairs: n_disc must be even, got {n_disc}")
    if n_disc <= 0 or n_val <= 0:
        raise ValueError(f"build_addition_pairs: n_disc and n_val must be positive, got {n_disc}, {n_val}")

    df = pd.read_parquet(eval_parquet)
    pool = df[(df["op"] == "+") & (df["cell"] == "4x4")].reset_index(drop=True)
    if exclude_pairs:  # leakage guard: drop operand pairs seen in any fine-tuning file
        seen = [tuple(sorted((int(a), int(b)))) in exclude_pairs for a, b in zip(pool["a"], pool["b"])]
        pool = pool[~np.array(seen, dtype=bool)].reset_index(drop=True)
    if n_disc + n_val > len(pool):
        raise ValueError(
            f"build_addition_pairs: n_disc+n_val={n_disc + n_val} exceeds pool size {len(pool)}"
        )

    pair_keys = [tuple(sorted((int(a), int(b)))) for a, b in zip(pool["a"], pool["b"])]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("build_addition_pairs: pool contains duplicate unordered operand pairs")

    prompt_ids, answer_ids = _validate_pool_tokenization(
        tokenizer,
        pool["prompt_text"].tolist(),
        pool["answer_text"].tolist(),
        pool["full_text"].tolist(),
        where="build_addition_pairs",
    )

    sums = pool["true_answer"].to_numpy()
    digit_count = np.array([len(str(int(s))) for s in sums])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    discovery_idx = perm[:n_disc]
    validation_idx = perm[n_disc : n_disc + n_val]
    used_idx = np.concatenate([discovery_idx, validation_idx])
    is_used = np.zeros(len(pool), dtype=bool)
    is_used[used_idx] = True
    outside = ~is_used  # "prefer" mask: rows not used as a clean problem

    cf_rng = np.random.default_rng(seed + 1)  # separate stream for cf draws
    clean_ids_list, corrupt_ids_list, c1_list, c2_list, k1_list, k2_list, cf_idx_list = (
        _assemble_pairs(prompt_ids, answer_ids, used_idx, digit_count, cf_rng, outside)
    )

    half_size = n_disc // 2
    rows = []
    for pos, (i, cf_i) in enumerate(zip(used_idx, cf_idx_list)):
        if pos < n_disc:
            split, half = "discovery", (0 if pos < half_size else 1)
        else:
            split, half = "validation", -1
        rows.append(
            {
                "a": int(pool["a"][i]),
                "b": int(pool["b"][i]),
                "sum": int(sums[i]),
                "cf_a": int(pool["a"][cf_i]),
                "cf_b": int(pool["b"][cf_i]),
                "cf_sum": int(sums[cf_i]),
                "split": split,
                "half": half,
            }
        )

    meta = pd.DataFrame(rows)
    return {
        "clean_ids": torch.tensor(clean_ids_list, dtype=torch.long),
        "corrupt_ids": torch.tensor(corrupt_ids_list, dtype=torch.long),
        "c1": torch.tensor(c1_list, dtype=torch.long),
        "c2": torch.tensor(c2_list, dtype=torch.long),
        "k1": torch.tensor(k1_list, dtype=torch.long),
        "k2": torch.tensor(k2_list, dtype=torch.long),
        # Top-level mirrors of two ``meta`` columns, for callers (``run.py``)
        # that index the pairs dict directly rather than through ``meta``:
        # ``half`` (0/1 discovery sub-split, -1 for validation rows) and
        # ``answer_text`` (the clean sum as a string, for greedy-decode EM).
        "half": torch.tensor(meta["half"].to_numpy(), dtype=torch.long),
        "answer_text": [str(s) for s in meta["sum"]],
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Copy task (specificity control)
# ---------------------------------------------------------------------------


def build_copy_pairs(tokenizer, n: int = 256, seed: int = 1) -> dict:
    """Build the copy-task specificity-control pairs: ``"Tom had {n}
    apples. How many apples did Tom have?\\n"`` -> ``"{n}"``, ``n`` a
    distinct 4-digit integer (1000-9999). The counterfactual is another
    distinct 4-digit number whose answer tokens differ at both positions.

    Same tokenization invariants as the addition task are checked (fixed
    prompt length, exactly 2 answer tokens, round-trip, first token not
    whitespace) — this is a genuine check, not an assumption carried over
    from ``build_addition_pairs``: the Llama-3.2-1B tokenizer happens to
    split every 4-digit number the same way (3+1 digits) regardless of
    surrounding context, so it holds here too, but a tokenizer change could
    break it and this would catch it.

    Returns a dict with the same shape as ``build_addition_pairs``:
    ``clean_ids``/``corrupt_ids`` (n, T), ``c1``/``c2``/``k1``/``k2`` (n,),
    and ``meta`` (columns ``n``, ``cf_n``).
    """
    if n <= 0:
        raise ValueError(f"build_copy_pairs: n must be positive, got {n}")
    rng = np.random.default_rng(seed)
    pool_size = min(9000, max(n * 4, n + 1))
    if pool_size <= n:
        raise ValueError(f"build_copy_pairs: n={n} leaves no room for counterfactual candidates")
    pool_values = rng.choice(np.arange(1000, 10000), size=pool_size, replace=False).astype(np.int64)

    prompts = [COPY_PROMPT.format(n=int(v)) for v in pool_values]
    answers = [str(int(v)) for v in pool_values]
    fulls = [p + a for p, a in zip(prompts, answers)]
    prompt_ids, answer_ids = _validate_pool_tokenization(
        tokenizer, prompts, answers, fulls, where="build_copy_pairs"
    )

    match_key = np.zeros(len(pool_values), dtype=np.int64)  # every n is 4 digits: one group
    clean_idx = np.arange(n)  # pool_values is already a random draw (no further shuffle needed)
    outside = np.zeros(len(pool_values), dtype=bool)
    outside[n:] = True

    cf_rng = np.random.default_rng(seed + 1)
    clean_ids_list, corrupt_ids_list, c1_list, c2_list, k1_list, k2_list, cf_idx_list = (
        _assemble_pairs(prompt_ids, answer_ids, clean_idx, match_key, cf_rng, outside)
    )

    meta = pd.DataFrame(
        {
            "n": [int(pool_values[i]) for i in clean_idx],
            "cf_n": [int(pool_values[j]) for j in cf_idx_list],
        }
    )
    return {
        "clean_ids": torch.tensor(clean_ids_list, dtype=torch.long),
        "corrupt_ids": torch.tensor(corrupt_ids_list, dtype=torch.long),
        "c1": torch.tensor(c1_list, dtype=torch.long),
        "c2": torch.tensor(c2_list, dtype=torch.long),
        "k1": torch.tensor(k1_list, dtype=torch.long),
        "k2": torch.tensor(k2_list, dtype=torch.long),
        "answer_text": [str(v) for v in meta["n"]],  # parity with build_addition_pairs
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# TinyStories (sanity check only — no clean/corrupt pairing)
# ---------------------------------------------------------------------------


def _default_tinystories_source() -> list[str]:
    """Load the TinyStories V2 validation split from the HF hub (network).
    Tries ``datasets.load_dataset`` first; falls back to downloading the raw
    ``TinyStoriesV2-GPT4-valid.txt`` file and splitting on its end-of-document
    marker. Never called in tests — pass ``text_source`` instead.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="validation")
        return [t for t in ds["text"] if t and t.strip()]
    except Exception:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset"
        )
        text = Path(path).read_text(encoding="utf-8")
        return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]


def build_tinystories(
    tokenizer,
    n: int = 64,
    length: int = 128,
    seed: int = 2,
    text_source: Sequence[str] | Callable[[], Sequence[str]] | None = None,
) -> dict:
    """64 held-out TinyStories sequences (PLAN.md "Data": sanity check only).

    No BOS, matching the pretraining packing convention (``geode/train/
    packing.py``'s ``encode``: ``tokenizer(batch, add_special_tokens=False)``
    then an EOS per document — these are standalone eval sequences, not a
    packed corpus chunk, so no EOS is appended here). Keeps only stories
    tokenizing to >= ``length`` tokens, truncates each to exactly ``length``,
    and samples ``n`` of them deterministically given ``seed``.

    ``text_source`` makes the story text injectable: a list of strings, or a
    zero-arg callable returning one. When omitted, pulls TinyStories-V2 from
    the HF hub (network) — tests must always pass ``text_source`` explicitly.

    Returns ``{"input_ids": (n, length) int64 tensor}``.
    """
    if n <= 0 or length <= 0:
        raise ValueError(f"build_tinystories: n and length must be positive, got {n}, {length}")
    stories = text_source() if callable(text_source) else text_source
    if stories is None:
        stories = _default_tinystories_source()
    stories = list(stories)
    if not stories:
        raise ValueError("build_tinystories: text_source produced no stories")

    tok_ids = tokenizer(stories, add_special_tokens=False)["input_ids"]
    long_enough = [ids for ids in tok_ids if len(ids) >= length]
    if len(long_enough) < n:
        raise ValueError(
            f"build_tinystories: only {len(long_enough)} stories >= {length} tokens, need {n}"
        )
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(long_enough), size=n, replace=False)
    chosen = [long_enough[i][:length] for i in chosen_idx]
    return {"input_ids": torch.tensor(chosen, dtype=torch.long)}


def main() -> None:
    """Build the three saved inputs; addition pool excludes every unordered
    operand pair of the four models' training files (run leakage_check.py
    first so they exist, then again afterwards to confirm zero overlap)."""
    from leakage_check import TRAIN_DATA_DIR, TRAINING_FILES, unordered_pairs
    from transformers import AutoTokenizer

    out = Path(__file__).resolve().parent / "data"
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    exclude: set[tuple[int, int]] = set()
    for entry in TRAINING_FILES:
        exclude |= unordered_pairs(TRAIN_DATA_DIR / entry["file"])
    eval_parquet = (Path(__file__).resolve().parents[1]
                    / "training-run/data/full/D_algo_eval_bare.parquet")
    save_pairs(build_addition_pairs(eval_parquet, tok, exclude_pairs=exclude), out / "pairs_seed0.pt")
    save_pairs(build_copy_pairs(tok), out / "copy_pairs_seed1.pt")
    if not (out / "tinystories_seed2.pt").exists():
        save_pairs(build_tinystories(tok), out / "tinystories_seed2.pt")
    print(f"[data] saved to {out} ({len(exclude)} training operand pairs excluded)")


if __name__ == "__main__":
    main()
