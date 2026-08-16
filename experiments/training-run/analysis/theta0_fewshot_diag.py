"""Tier-1 theta0 few-shot diagnostic (decisions.md 2026-08-16 evening CORRECTION
entry): why does ``evt-ts38pp-parent`` solve op-notation at 97.95% zero-shot
but collapse to 0.1% at 16-shot, when the paper's own Table 11 few-shot check
on the same intervention *rises* (2.0% -> 11.9%)?

Three candidate mechanisms (see EXPERIMENTS.md / decisions.md for the full
writeup):

- M1 POSITION/CONTEXT LOCK -- the parent was full-FT'd one-example-per-row
  (``geode/train/sft.py``: SFT batches one example per row, never packed), so
  ``Question:`` sat at sequence position ~0 for every training step. A 16-shot
  prompt puts the query ~200+ tokens in; the op lesson may simply not fire
  off-position-0.
- M2 FEW-SHOT COMPOSITION ARTIFACT -- something about how ``few_shot_prompt``
  composes exemplars for this tokenizer/model (blank-line separator, no EOS
  between exemplars, approaching pretrain seq_len=512 / max_position_embeddings
  =1024) breaks the query regardless of position per se.
- M3 PROMPT-RENDER MISMATCH (owner-confirmed Tier-1 target) -- our single-line
  ``Question: 23 + 45\\nAnswer: 68`` vs the paper's literal block form
  ``Question:\\n23 + 45\\nAnswer:\\n68`` (App. E.1.2/E.2). The parent was
  TRAINED single-line, so a block-form probe on it tests eval-side render
  *sensitivity* only -- it cannot by itself confirm the paper's render would
  fix things, since the parent never saw block form in training. Say this
  plainly in any writeup that cites the block numbers.

This script measures op / nl_scaffolded / bare EM (+ shared-set label loss)
under four conditions, crossed with k in ``--shots`` where applicable, for
each ``--runs`` checkpoint:

- ``single``  -- gates.py's own G5 protocol, byte-identical composition
  (``geode.arith.few_shot_prompt`` + token-prefix slicing). k=0 on this
  condition is the CORRECTNESS ANCHOR: on the full (non-``--quick``) run with
  ``--n-queries 1024`` (gates.py g5's own ``--n`` default) it MUST reproduce
  ``geode-store/results/ts38pp_family_theta0.json`` to within +/-0.002 EM and
  matching loss, on the same fixed row slices. A ``--quick`` run (smaller
  --n-queries) is a smoke check only -- expect the right ballpark (parent op
  em0 ~ 0.98, base op em0 ~ 0.0), not exact reproduction.
- ``block``   -- op/nl_scaffolded only (bare has no Question:/Answer: scaffold
  to convert). Re-renders each row's ``Question: <body>\\nAnswer: <answer>``
  into the paper's literal block form ``Question:\\n<body>\\nAnswer:\\n<answer>``
  before composing/tokenizing, via ``render_block`` below. IMPORTANT: this
  is an EVAL-SIDE-ONLY probe of M3 (see module docstring above) -- the parent
  was never trained on block-rendered examples.
- ``story_prefix`` -- op/nl_scaffolded only, k fixed at 0 (stored under the
  "0" key): the query is preceded by ~``--story-prefix-tokens`` tokens of
  in-distribution TinyStories text (sourced from the relay's cached
  ``run1_val_stream.pt``, or a hardcoded fallback passage -- see
  ``load_story_prefix``) + the same ``\\n\\n`` separator ``few_shot_prompt``
  uses. Tests M1: if op EM collapses under ANY long prefix (not just
  arithmetic exemplars), that's context length / position, not few-shot
  composition specifically.
- ``k1_position`` -- op/nl_scaffolded only, k fixed at 1 (stored under the
  "1" key): a single real exemplar, but preceded by the same story prefix so
  the query lands at roughly the position it would in a k=16 prompt. Compares
  against ``single``'s k=1 cell: if plain k=1 survives but this doesn't,
  position (not exemplar content) is doing the damage -- decisive for M1 vs
  M2.
- ``dm_mixture`` (opt-in via ``--dm-mixture``; U7, owner-approved 2026-08-16
  night) -- the paper's NL target is DeepMind-Mathematics
  ``arithmetic__add_or_sub``, a 19-template mixture (11 addition, 8
  subtraction, +4 conditional "distance/difference" phrasings when a >= b),
  roughly half of which carry the literal ``+``/``-`` symbol; ours is one
  word-only phrasing. This condition re-renders the OP eval's own triples
  (same rows as the ``op`` task, so directly comparable to the parent's
  97.95% op-notation number) under each of the six template PAIRS from
  ``experiments/training-run/datagen/make_dm_probe_eval.py`` (sym_q, sym_imp,
  bare_op, word_q, word_imp, sumof -- template TEXT imported verbatim from
  that module, which itself copied them verbatim from DeepMind's
  ``mathematics_dataset/modules/arithmetic.py``) plus a genuine per-row
  template draw across the FULL DM_ADD/DM_SUB(+NONNEG) pools ("mixture", same
  selection rule as that module's local ``dm_chooser``, reimplemented here
  since that function is nested inside the other script's own ``main()`` and
  not importable -- only the templates themselves needed reuse). Scaffolded
  form only (not bare); k in {0, 16} only (not the full ``--shots`` sweep) to
  keep this bounded, per the owner's "do not let it delay the core
  deliverable" instruction. Stored as
  ``results[run_id]["dm_mixture"]["<template_key>"]["<k>"]``, template_key in
  {sym_q, sym_imp, bare_op, word_q, word_imp, sumof, mixture}.

Output JSON schema (also enforced/round-tripped by
``tests/test_theta0_fewshot_diag.py``)::

    {
      "<run_id>": {
        "single":       {"<task>": {"<k>": {"em": float, "n": int,
                                             "label_loss_nats": float|null,
                                             "query_offset_tokens": int}}},
        "block":        {same shape, op/nl_scaffolded only},
        "block_sign_split": {"<task>": {"<k>": {"em_positive": float|null,
                                                 "n_positive": int,
                                                 "em_negative": float|null,
                                                 "n_negative": int}}},
        "story_prefix": {"<task>": {"0": {...cell...}}},
        "k1_position":  {"<task>": {"1": {...cell...}}},
        "dm_mixture (opt-in)": {"<template_key>": {"0"|"16": {...cell...}}}
      },
      ...,
      "meta": {
        "git_sha": str, "n_queries": int, "shots": [int, ...],
        "story_prefix_source": str, "story_prefix_tokens": int (MEASURED,
            not the requested --story-prefix-tokens -- see load_story_prefix),
        "tokenizer_sha": str, "batch_size": int, "quick": bool,
        "skip_label_loss": bool, "checkpoint": {run_id: str},
        "max_position_embeddings": {run_id: int},
        "dm_mixture": bool, "dm_mixture_seed": int,
        "dm_mixture_template_source": str
      }
    }

Cell notes:

- ``label_loss_nats`` is populated ONLY at k=0 of the ``single``/``block``
  conditions (gates.py's "shared-set test loss": masked NLL over the WHOLE
  frozen reporting block ``df.iloc[EVAL_STOP_ROWS:]``, independent of
  ``--n-queries`` -- NOT a function of k at all). Storing it under the k=0 key
  is a JSON-shape convenience, not a claim that it's a "0-shot loss"; every
  other k's ``label_loss_nats`` is ``null`` by construction.
- ``query_offset_tokens`` is the MEASURED token length of the composed
  exemplar prefix (``tokenizer(exemplars joined by "\\n\\n")``) -- i.e. where
  the query actually starts, not a target position. This is what the M1
  position claim stands on; see ``exemplar_prefix_token_count``.
- ``block_sign_split`` exists because the block render changes BPE merge
  behavior on the answer's leading character: single-line render merges the
  space before a negative sign into one token (`` -``); block render tokenizes
  ``\\n`` and ``-`` separately (verified empirically against the frozen
  tokenizer, see ``tests/test_theta0_fewshot_diag.py``). A block-condition EM
  drop concentrated on negative answers is very plausibly this tokenization
  artifact, not evidence about the render itself -- hence the sign split.

Usage (run from anywhere; paths are resolved off this file's location)::

    python3 theta0_fewshot_diag.py --quick --skip-label-loss \\
        --out /tmp/smoke.json
    python3 theta0_fewshot_diag.py \\
        --out $GEODE_STORE/results/ts38pp_theta0_fewshot_diag.json

CPU-only-friendly (no ``--confirm-cost`` needed -- inference only, matching
gates.py's own convention), but the full (non-``--quick``) run does 5 full
reporting-block NLL passes per run (single k=0 x3 tasks, block k=0 x2 tasks)
plus 34 accuracy passes per run; use ``--device cuda`` on the GPU box.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from geode.arith import (
    exact_match,
    exact_match_accuracy,
    few_shot_prompt,
    load_frozen_parquet,
    tokenize_with_spans,
)
from geode.edl import EVAL_STOP_ROWS, G5_N_SHOTS
from geode.edl.masking import TaskFormat
from geode.train import evaluate_sft_nll_nats
from geode.zoo import checkpoint_dir, load_model, load_run, tokenizer_hash

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "experiments" / "training-run" / "scripts"
CONFIGS_DIR = REPO_ROOT / "experiments" / "training-run" / "configs"
DATAGEN = REPO_ROOT / "experiments" / "training-run" / "datagen"

# Same reuse pattern as analysis/g8_decompose.py: the training-run scripts/
# (and datagen/) trees are flat, hyphen-rooted module trees (not importable
# as a dotted package), so their own launcher contract
# (``cd scripts/ && python3 foo.py``) is mirrored here by inserting each dir
# on sys.path before importing.
for _dir in (SCRIPTS, DATAGEN):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from train import load_config  # noqa: E402  (reuse gates.py's own config resolution)
import make_dm_probe_eval as dmprobe  # noqa: E402  (dm_mixture: reuse its templates verbatim)

DEFAULT_RUNS = ["evt-ts38pp-parent", "evt-run1-base-v3-ext"]
# gates.py g5's own --n default (see scripts/gates.py: `g5.add_argument("--n",
# type=int, default=1024)`); mirrored here so a full run's row slices --
# and therefore its zero/16-shot EM -- land on the identical rows gates.py
# scored when it produced ts38pp_family_theta0.json.
DEFAULT_N_QUERIES = 1024
QUICK_N_QUERIES = 200
DEFAULT_SHOTS = (0, 1, 2, 4, 8, 16)
DEFAULT_BATCH_SIZE = 128  # gates.py g5's common_args default; keep it fixed --
# greedy_completions left-pads, so a different batch size changes padding and
# can flip an argmax tie, which would look like noise on top of the anchor.
DEFAULT_STORY_PREFIX_TOKENS = 200
DEFAULT_STORY_PREFIX_SEED = 316  # the repo's usual --sample-seed value
MAX_NEW_TOKENS = 12  # matches geode.arith.decode.greedy_completions's default

# dm_mixture (U7, owner-approved 2026-08-16 night): k=0/16 only, scaffolded
# only -- bounded scope per "do not let it delay the core deliverable".
DEFAULT_DM_MIXTURE_SEED = 20260815  # matches make_dm_probe_eval.py's own --seed default
DM_MIXTURE_KS = (0, 16)
DM_MIXTURE_TEMPLATE_SOURCE = (
    "experiments/training-run/datagen/make_dm_probe_eval.py: PAIRS (sym_q, "
    "sym_imp, bare_op, word_q, word_imp, sumof) + DM_ADD/DM_SUB/DM_SUB_NONNEG "
    "for the 'mixture' draw -- templates copied verbatim there from "
    "mathematics_dataset/modules/arithmetic.py (DeepMind, fetched 2026-08-15)"
)

# task label -> eval config filename, matching launch_ts38pp_family.sh's
# PROBE_LABELS / PROBE_CONFIGS exactly (lines 510-511).
TASK_CONFIGS = {
    "op": "eval_target_data.yaml",
    "nl_scaffolded": "eval_nl_target_data_ts38.yaml",
    "bare": "eval_bare_target_data_ts38.yaml",
}
# bare_nl has no "Question:"/"Answer:" scaffold to convert to block form, and
# the story_prefix / k1_position conditions are only requested for the two
# scaffolded tasks (decisions.md Tier-1 plan) -- so all three "block-family"
# conditions share this restriction.
BLOCK_TASKS = ("op", "nl_scaffolded")

_PROMPT_PREFIX = "Question: "
_PROMPT_PREFIX_BLOCK = "Question:\n"
_ANSWER_MARKER = "\nAnswer: "
_ANSWER_MARKER_BLOCK = "\nAnswer:\n"

# Hardcoded fallback story-prefix passage (334 tokens under the frozen
# tokenizer, comfortably over the default --story-prefix-tokens=200), used
# only if cache/run1_val_stream.pt is unreachable -- see load_story_prefix.
FALLBACK_STORY = (
    "Once upon a time, in a small village, there lived a little girl named Lily. "
    "Lily loved to play in the sunny garden behind her house. Every morning, she "
    "would wake up early and run outside to see the flowers. One day, Lily found "
    "a tiny kitten hiding under a bush. The kitten was scared and shivering, so "
    'Lily picked it up gently and held it close. "Don\'t worry, little kitten," '
    'she said softly. "I will take care of you." She carried the kitten home and '
    "gave it a warm blanket and a bowl of milk. The kitten purred happily and fell "
    "asleep in her arms. From that day on, Lily and the kitten became best friends. "
    "They played together every day in the garden, chasing butterflies and rolling "
    "in the grass. Lily's mother was happy to see her daughter so joyful. \"You have "
    'a kind heart," her mother said, smiling. Lily named the kitten Whiskers, because '
    "of its long, fluffy whiskers. Whiskers loved to climb the big oak tree in the "
    "garden and watch the birds fly by. Sometimes, Lily would read stories to Whiskers "
    "under the tree, and the kitten would listen quietly, its tail swishing back and "
    "forth. As the seasons changed, Lily and Whiskers explored the whole village "
    "together, meeting new friends and having many adventures. They visited the old "
    "baker who gave them warm bread, and the friendly farmer who let them pet the "
    "cows. Every night, Lily would tuck Whiskers into a soft basket by her bed, and "
    "they would both dream of their next big adventure. The little village was full "
    "of happiness, and everyone loved to see Lily and her kitten playing together in "
    "the golden afternoon sun."
)


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --------------------------------------------------------------------------
# Pure logic (unit-tested in tests/test_theta0_fewshot_diag.py)
# --------------------------------------------------------------------------


def render_block(full_text: str, answer_span: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    """Convert a single-line ``Question: <body>\\nAnswer: <answer>`` row into
    the paper's literal block form ``Question:\\n<body>\\nAnswer:\\n<answer>``
    (App. E.1.2/E.2's example blocks).

    Both scaffold substitutions ("Question: " -> "Question:\\n" and
    "\\nAnswer: " -> "\\nAnswer:\\n") are exactly one character each, so the
    answer span is numerically unchanged -- but this recomputes it from
    scratch rather than assuming that algebra, so a future scaffold change
    can't silently desync text and span. Raises ``ValueError`` on anything
    that isn't the frozen ``operator``/``nl`` scaffold (in particular,
    ``bare_nl`` rows, which have no ``"Question: "`` prefix to convert).
    """
    cs, ce = answer_span
    if not full_text.startswith(_PROMPT_PREFIX):
        raise ValueError(
            f"render_block: text does not start with {_PROMPT_PREFIX!r} (bare_nl "
            f"rows have no scaffold to convert): {full_text[:24]!r}..."
        )
    marker_start = cs - len(_ANSWER_MARKER)
    if marker_start < len(_PROMPT_PREFIX) or full_text[marker_start:cs] != _ANSWER_MARKER:
        raise ValueError(
            f"render_block: expected {_ANSWER_MARKER!r} immediately before the "
            f"answer span at [{marker_start}:{cs}], found "
            f"{full_text[max(marker_start, 0) : cs]!r}"
        )
    body = full_text[len(_PROMPT_PREFIX) : marker_start]
    answer_text = full_text[cs:ce]
    tail = full_text[ce:]  # empty for the frozen rows, which end at the answer
    block_text = _PROMPT_PREFIX_BLOCK + body + _ANSWER_MARKER_BLOCK + answer_text + tail
    new_cs = len(_PROMPT_PREFIX_BLOCK) + len(body) + len(_ANSWER_MARKER_BLOCK)
    new_ce = new_cs + len(answer_text)
    return block_text, (new_cs, new_ce)


def compose_few_shot(
    exemplars: list[str], q_texts: list[str], q_spans: list[tuple[int, int]]
) -> tuple[list[str], list[tuple[int, int]]]:
    """gates.py's ``accuracy_with`` composition step, factored out and
    generalized over exemplars/queries so it serves every condition
    (single/block/story_prefix/k1_position alike -- they differ only in what
    strings are passed in, never in how they're joined or spans shifted).

    For each query, composes ``exemplars + [complete query text]`` via
    ``geode.arith.few_shot_prompt`` (blank-line separated) and shifts the
    query's own answer char span by the composed prefix's length. Zero
    exemplars is the identity: composed text == query text, span unchanged.
    """
    texts, spans = [], []
    for text, (cs, ce) in zip(q_texts, q_spans):
        composed = few_shot_prompt(exemplars, text)
        offset = len(composed) - len(text)
        texts.append(composed)
        spans.append((offset + cs, offset + ce))
    return texts, spans


def exemplar_prefix_token_count(exemplars: list[str], tokenizer: Any) -> int:
    """Measured token length of the composed exemplar prefix -- i.e. how many
    tokens precede the query's own prompt in a composed few-shot text. This
    is what the M1 position claim is measured against; report this, never
    the requested ``--story-prefix-tokens``, which is only a slicing target.
    """
    if not exemplars:
        return 0
    prefix = "\n\n".join(exemplars) + "\n\n"
    return len(tokenizer(prefix, add_special_tokens=False)["input_ids"])


def sign_split_em(hits: list[bool], answers: list[int]) -> dict[str, Any]:
    """Split exact-match hits by the true answer's sign.

    Exists for the ``block`` condition (module docstring): block rendering
    changes how the frozen BPE merges the character right after
    ``"Answer:"`` for negative answers, so a block-condition EM drop
    concentrated on negatives is plausibly that tokenization artifact, not
    the render itself.
    """
    pos_hits = [h for h, a in zip(hits, answers) if a >= 0]
    neg_hits = [h for h, a in zip(hits, answers) if a < 0]
    return {
        "em_positive": (sum(pos_hits) / len(pos_hits)) if pos_hits else None,
        "n_positive": len(pos_hits),
        "em_negative": (sum(neg_hits) / len(neg_hits)) if neg_hits else None,
        "n_negative": len(neg_hits),
    }


def dm_mixture_chooser(seed: int):
    """A per-row DM template picker: uniform over DM_ADD for '+' rows, uniform
    over DM_SUB (+DM_SUB_NONNEG when a >= b) for '-' rows.

    Mirrors ``make_dm_probe_eval.py``'s local ``dm_chooser`` exactly (same
    pools, same selection rule) -- that function is nested inside the other
    script's own ``main()`` and therefore not importable, so only the
    5-line selection rule is reimplemented here; the template TEXT itself
    (``dmprobe.DM_ADD``/``DM_SUB``/``DM_SUB_NONNEG``) is imported, never
    retyped. One ``random.Random`` instance is threaded through every row
    a caller draws from it (shots, then queries), so the draw is
    reproducible for a fixed ``seed`` and row order.
    """
    rng = random.Random(seed)

    def choose(a: int, b: int, op: str) -> str:
        if op == "+":
            return rng.choice(dmprobe.DM_ADD)
        pool = dmprobe.DM_SUB + (dmprobe.DM_SUB_NONNEG if a >= b else [])
        return rng.choice(pool)

    return choose


def dm_render_rows(rows: pd.DataFrame, choose_template) -> tuple[list[str], list[tuple[int, int]]]:
    """Render each row of ``rows`` (needs ``a``/``b``/``op``/``true_answer``
    columns -- any triple source works, not just D_dmprobe_*) under
    ``choose_template(a, b, op) -> template string``, via
    ``make_dm_probe_eval.render`` (scaffolded ``Question: <body>\\nAnswer:
    <answer>`` form, byte-identical to that script's own construction).
    """
    texts, spans = [], []
    for r in rows.itertuples():
        a, b, op, ans = int(r.a), int(r.b), r.op, int(r.true_answer)
        body = choose_template(a, b, op).format(a=a, b=b)
        _prompt, _ans_text, full, s, e = dmprobe.render(body, ans, scaffolded=True)
        texts.append(full)
        spans.append((s, e))
    return texts, spans


# --------------------------------------------------------------------------
# Model-dependent evaluation (thin glue over geode.arith + geode.train)
# --------------------------------------------------------------------------


def em_for_condition(
    model: Any,
    tokenizer: Any,
    exemplars: list[str],
    q_texts: list[str],
    q_spans: list[tuple[int, int]],
    answers: list[int],
    *,
    device: str,
    batch_size: int,
) -> tuple[float, list[bool]]:
    """Zero/k-shot exact match for one condition -- byte-identical procedure
    to gates.py's ``run_g5.accuracy_with`` (same token-prefix-of-a-
    training-style-tokenization trick), generalized to accept any
    pre-rendered exemplars/queries. Returns ``(accuracy, per-query hits)``.

    Refuses (raises ``ValueError``, never silently truncates) if the composed
    prompt plus the greedy decode budget would exceed the model's own
    ``max_position_embeddings`` -- an overflow here would silently produce
    garbage completions that read exactly like an M1 position collapse.
    """
    texts, spans = compose_few_shot(exemplars, q_texts, q_spans)
    examples = tokenize_with_spans(texts, spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is not None and prompt_ids:
        longest = max(len(p) for p in prompt_ids)
        if longest + MAX_NEW_TOKENS > limit:
            raise ValueError(
                f"em_for_condition: composed prompt ({longest} tokens) + greedy "
                f"decode budget ({MAX_NEW_TOKENS}) exceeds the model's "
                f"max_position_embeddings ({limit}) -- refusing rather than "
                "silently truncating into a fake M1-shaped collapse"
            )
    accuracy, completions = exact_match_accuracy(
        model, tokenizer, prompt_ids, answers, device=device, batch_size=batch_size
    )
    hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
    return accuracy, hits


def shared_set_loss(
    model: Any,
    tokenizer: Any,
    rows: pd.DataFrame,
    task_name: str,
    format_version: str,
    *,
    device: str,
    batch_size: int,
    render_fn=None,
) -> tuple[float, int]:
    """gates.py's "shared-set test loss": mean masked-label NLL (nats) over
    ``rows`` (in production, the WHOLE frozen reporting block
    ``df.iloc[EVAL_STOP_ROWS:]`` -- independent of ``--n-queries``, exactly as
    in gates.py's ``run_g5``). ``render_fn``, if given, maps
    ``(full_text, (cs, ce)) -> (new_text, (new_cs, new_ce))`` per row before
    tokenizing (used for the ``block`` condition); ``None`` scores the row's
    own single-line rendering untouched.
    """
    texts, spans = [], []
    for full_text, cs, ce in zip(
        rows["full_text"],
        rows["answer_char_start"].astype(int),
        rows["answer_char_end"].astype(int),
    ):
        if render_fn is None:
            texts.append(full_text)
            spans.append((cs, ce))
        else:
            t, s = render_fn(full_text, (cs, ce))
            texts.append(t)
            spans.append(s)
    examples = tokenize_with_spans(texts, spans, tokenizer, append_eos=True)
    task_format = TaskFormat(name=task_name, format_version=format_version)
    loss = evaluate_sft_nll_nats(model, examples, task_format, batch_size=batch_size, device=device)
    return loss, len(examples)


def _slice_to_n_tokens(text: str, tokenizer: Any, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _ensure_story_cache(store: Path) -> Path | None:
    """Pull ``cache/run1_val_stream.pt`` from the relay if not already local
    (same pattern as launch_ts38pp_family.sh lines 479-495). Returns ``None``
    (never raises) on any failure -- the caller falls back to the hardcoded
    passage.
    """
    cache = store / "cache" / "run1_val_stream.pt"
    if cache.is_file():
        return cache
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="mhieuuu/geode-store",
            repo_type="model",
            filename="cache/run1_val_stream.pt",
            local_dir=str(store),
        )
    except Exception as exc:  # noqa: BLE001 -- any pull failure just falls back
        print(
            f"[theta0-diag] WARN: could not pull cache/run1_val_stream.pt ({exc}); "
            "falling back to the hardcoded story passage",
            file=sys.stderr,
        )
        return None
    return cache if cache.is_file() else None


def load_story_prefix(store: Path, tokenizer: Any, *, n_tokens: int, seed: int) -> tuple[str, str]:
    """Return ``(story_text, source_label)`` -- ``story_text`` tokenizes to
    exactly ``n_tokens`` under ``tokenizer`` (by construction: whatever
    candidate text is found is tokenized, sliced to ``n_tokens`` ids, and
    decoded back), so the caller can measure the achieved offset rather than
    trust the request. Tries the relay's cached TinyStories val stream first
    (a fixed, seeded row of the packed val sequences -- ``--story-prefix-
    seed`` picks which row); falls back to ``FALLBACK_STORY`` (and says so in
    ``source_label``) if the cache is missing/unusable.
    """
    cache = _ensure_story_cache(store)
    if cache is not None:
        try:
            blob = torch.load(cache)
            val_seqs = blob["val_seqs"].long()
            gen = torch.Generator().manual_seed(seed)
            idx = int(torch.randint(0, val_seqs.shape[0], (1,), generator=gen).item())
            text = tokenizer.decode(val_seqs[idx].tolist(), skip_special_tokens=True)
            sliced = _slice_to_n_tokens(text, tokenizer, n_tokens)
            if sliced.strip():
                return sliced, f"run1_val_stream.pt row={idx} seed={seed}"
        except Exception as exc:  # noqa: BLE001 -- any cache-format surprise falls back
            print(
                f"[theta0-diag] WARN: story-prefix cache unusable ({exc}); "
                "falling back to the hardcoded story passage",
                file=sys.stderr,
            )
    return _slice_to_n_tokens(FALLBACK_STORY, tokenizer, n_tokens), "fallback_hardcoded_passage"


def _assert_not_trained_on_eval(manifest: Any, cfg: dict, run_id: str) -> None:
    """Same guard as gates.py's ``run_g5``: refuse if this run's own training
    data order_hash matches the eval file's, which would mean these
    "held-out" questions were actually trained on.
    """
    exp = manifest.data.get("experiment", {})
    if exp.get("data_order_hash") == cfg["data"]["order_hash"]:
        raise ValueError(
            f"{run_id}: trained on this G5-style eval file itself "
            f"({cfg['data']['file']}) -- no un-trained eval questions exist for it"
        )


def _rows_slice(
    df: pd.DataFrame, start: int, count: int
) -> tuple[list[str], list[tuple[int, int]]]:
    rows = df.iloc[start : start + count]
    texts = rows["full_text"].tolist()
    spans = list(zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int)))
    return texts, spans


# --------------------------------------------------------------------------
# CLI / orchestration
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help=f"default {DEFAULT_N_QUERIES} (gates.py g5's own --n default), or "
        f"{QUICK_N_QUERIES} under --quick",
    )
    ap.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    ap.add_argument(
        "--quick",
        action="store_true",
        help=f"smoke run: --n-queries defaults to {QUICK_N_QUERIES}. Combine with "
        "--skip-label-loss for a fast check; quick numbers are NOT the anchor.",
    )
    ap.add_argument(
        "--skip-label-loss",
        action="store_true",
        help="skip the full-reporting-block shared-set NLL passes (gates.py's "
        "--skip-test-loss analogue) -- label_loss_nats is null everywhere",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--story-prefix-tokens", type=int, default=DEFAULT_STORY_PREFIX_TOKENS)
    ap.add_argument("--story-prefix-seed", type=int, default=DEFAULT_STORY_PREFIX_SEED)
    ap.add_argument(
        "--dm-mixture",
        action="store_true",
        help="also probe the DeepMind-Mathematics 19-template mixture on the op "
        "eval's own triples (k in {0, 16} only) -- see module docstring",
    )
    ap.add_argument("--dm-mixture-seed", type=int, default=DEFAULT_DM_MIXTURE_SEED)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.n_queries is None:
        args.n_queries = QUICK_N_QUERIES if args.quick else DEFAULT_N_QUERIES
    return args


def _load_tokenizer(cfg: dict, config_path: Path) -> Any:
    from transformers import AutoTokenizer

    local = (config_path.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _write_json(path: Path, results: dict[str, Any], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**results, "meta": meta}
    path.write_text(json.dumps(payload, indent=2) + "\n")


CONDITIONS = ("single", "block", "story_prefix", "k1_position", "dm_mixture")


def print_markdown_table(results: dict[str, Any], shots: list[int]) -> None:
    cols = sorted(set(shots) | {0, 1} | set(DM_MIXTURE_KS))
    print("| run | condition | task | " + " | ".join(f"k={k}" for k in cols) + " |")
    print("|" + "---|" * (3 + len(cols)))
    for run_id, run in results.items():
        for condition in CONDITIONS:
            for task, per_k in run.get(condition, {}).items():
                cells = []
                for k in cols:
                    cell = per_k.get(str(k))
                    cells.append(f"{cell['em']:.4f}" if cell is not None else "")
                print(f"| {run_id} | {condition} | {task} | " + " | ".join(cells) + " |")


def run_for_task(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    cfg: dict,
    task: str,
    *,
    args: argparse.Namespace,
    story_text: str,
    run_results: dict[str, Any],
) -> None:
    """Compute every condition this task participates in and fold the cells
    into ``run_results`` in place (single run, single task)."""
    q_start = EVAL_STOP_ROWS + G5_N_SHOTS
    if len(df) < q_start + args.n_queries:
        raise SystemExit(
            f"--n-queries {args.n_queries}: {cfg['data']['file']} has {len(df)} rows, "
            f"needs {q_start + args.n_queries}"
        )
    shot_texts, shot_spans = _rows_slice(df, EVAL_STOP_ROWS, G5_N_SHOTS)
    q_texts, q_spans = _rows_slice(df, q_start, args.n_queries)
    answers = df.iloc[q_start : q_start + args.n_queries]["true_answer"].astype(int).tolist()
    task_name = cfg["task"]["name"]
    format_version = cfg["task"]["format_version"]

    def _log(condition: str, k: int | str, cell: dict) -> None:
        extra = ""
        if cell.get("label_loss_nats") is not None:
            extra = f" label_loss={cell['label_loss_nats']:.4f}"
        print(
            f"[theta0-diag] {task} {condition} k={k}: em={cell['em']:.4f} n={cell['n']}{extra}",
            flush=True,
        )

    # --- (a) single ---------------------------------------------------
    for k in args.shots:
        if k > len(shot_texts):
            raise SystemExit(f"--shots {k} exceeds the shot pool ({len(shot_texts)} rows)")
        exemplars = shot_texts[:k]
        em, _hits = em_for_condition(
            model,
            tokenizer,
            exemplars,
            q_texts,
            q_spans,
            answers,
            device=args.device,
            batch_size=args.batch_size,
        )
        label_loss = None
        if k == 0 and not args.skip_label_loss:
            label_loss, _n = shared_set_loss(
                model,
                tokenizer,
                df.iloc[EVAL_STOP_ROWS:],
                task_name,
                format_version,
                device=args.device,
                batch_size=args.batch_size,
            )
        cell = {
            "em": em,
            "n": len(q_texts),
            "label_loss_nats": label_loss,
            "query_offset_tokens": exemplar_prefix_token_count(exemplars, tokenizer),
        }
        run_results.setdefault("single", {}).setdefault(task, {})[str(k)] = cell
        _log("single", k, cell)

    if task not in BLOCK_TASKS:
        return

    # --- (b) block ------------------------------------------------------
    block_shot_texts = [render_block(t, s)[0] for t, s in zip(shot_texts, shot_spans)]
    block_q_rendered = [render_block(t, s) for t, s in zip(q_texts, q_spans)]
    block_q_texts = [t for t, _ in block_q_rendered]
    block_q_spans = [s for _, s in block_q_rendered]
    for k in args.shots:
        exemplars = block_shot_texts[:k]
        em, hits = em_for_condition(
            model,
            tokenizer,
            exemplars,
            block_q_texts,
            block_q_spans,
            answers,
            device=args.device,
            batch_size=args.batch_size,
        )
        label_loss = None
        if k == 0 and not args.skip_label_loss:
            label_loss, _n = shared_set_loss(
                model,
                tokenizer,
                df.iloc[EVAL_STOP_ROWS:],
                f"{task_name}_block",
                format_version,
                device=args.device,
                batch_size=args.batch_size,
                render_fn=render_block,
            )
        cell = {
            "em": em,
            "n": len(block_q_texts),
            "label_loss_nats": label_loss,
            "query_offset_tokens": exemplar_prefix_token_count(exemplars, tokenizer),
        }
        run_results.setdefault("block", {}).setdefault(task, {})[str(k)] = cell
        run_results.setdefault("block_sign_split", {}).setdefault(task, {})[str(k)] = sign_split_em(
            hits, answers
        )
        _log("block", k, cell)

    # --- (c) story_prefix (k fixed at 0) --------------------------------
    em, _hits = em_for_condition(
        model,
        tokenizer,
        [story_text],
        q_texts,
        q_spans,
        answers,
        device=args.device,
        batch_size=args.batch_size,
    )
    cell = {
        "em": em,
        "n": len(q_texts),
        "label_loss_nats": None,
        "query_offset_tokens": exemplar_prefix_token_count([story_text], tokenizer),
    }
    run_results.setdefault("story_prefix", {}).setdefault(task, {})["0"] = cell
    _log("story_prefix", 0, cell)

    # --- (d) k1_position (k fixed at 1, real exemplar pushed to ~position
    # 200+len(exemplar)) -----------------------------------------------
    exemplars = [story_text, shot_texts[0]]
    em, _hits = em_for_condition(
        model,
        tokenizer,
        exemplars,
        q_texts,
        q_spans,
        answers,
        device=args.device,
        batch_size=args.batch_size,
    )
    cell = {
        "em": em,
        "n": len(q_texts),
        "label_loss_nats": None,
        "query_offset_tokens": exemplar_prefix_token_count(exemplars, tokenizer),
    }
    run_results.setdefault("k1_position", {}).setdefault(task, {})["1"] = cell
    _log("k1_position", 1, cell)


def run_dm_mixture(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    run_results: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    """U7 (owner-approved 2026-08-16 night): DM-template EM at k in {0, 16},
    on the OP eval's own triples (``df`` = ``task_dfs["op"]``, the same
    shots/query row slice ``run_for_task`` used for the ``op`` task), for
    each of the six fixed template PAIRS plus a genuine per-row mixture draw.
    """
    q_start = EVAL_STOP_ROWS + G5_N_SHOTS
    shots_df = df.iloc[EVAL_STOP_ROWS:q_start]
    queries_df = df.iloc[q_start : q_start + args.n_queries]
    answers = queries_df["true_answer"].astype(int).tolist()

    def _score(key: str, choose_template) -> None:
        shot_texts, _ = dm_render_rows(shots_df, choose_template)
        q_texts, q_spans = dm_render_rows(queries_df, choose_template)
        for k in DM_MIXTURE_KS:
            exemplars = shot_texts[:k]
            em, _hits = em_for_condition(
                model,
                tokenizer,
                exemplars,
                q_texts,
                q_spans,
                answers,
                device=args.device,
                batch_size=args.batch_size,
            )
            cell = {
                "em": em,
                "n": len(q_texts),
                "label_loss_nats": None,
                "query_offset_tokens": exemplar_prefix_token_count(exemplars, tokenizer),
            }
            run_results.setdefault("dm_mixture", {}).setdefault(key, {})[str(k)] = cell
            print(
                f"[theta0-diag] dm_mixture {key} k={k}: em={em:.4f} n={len(q_texts)}",
                flush=True,
            )

    for key, (t_add, t_sub) in dmprobe.PAIRS.items():
        _score(key, lambda a, b, op, _ta=t_add, _ts=t_sub: _ta if op == "+" else _ts)

    _score("mixture", dm_mixture_chooser(args.dm_mixture_seed))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    task_cfgs = {
        task: load_config(CONFIGS_DIR / fname, None) for task, fname in TASK_CONFIGS.items()
    }
    op_cfg_path = CONFIGS_DIR / TASK_CONFIGS["op"]
    tokenizer = _load_tokenizer(task_cfgs["op"], op_cfg_path)
    for task, fname in TASK_CONFIGS.items():
        resolved = (CONFIGS_DIR / fname).parent / task_cfgs[task]["tokenizer"]["path"]
        expected = op_cfg_path.parent / task_cfgs["op"]["tokenizer"]["path"]
        if resolved.resolve() != expected.resolve():
            raise SystemExit(
                f"theta0_fewshot_diag: {fname}'s tokenizer path {resolved} != "
                f"{TASK_CONFIGS['op']}'s {expected} -- eval configs must share one tokenizer"
            )
    task_dfs = {
        task: load_frozen_parquet(cfg["data"], root=REPO_ROOT) for task, cfg in task_cfgs.items()
    }

    story_text, story_source = load_story_prefix(
        args.store, tokenizer, n_tokens=args.story_prefix_tokens, seed=args.story_prefix_seed
    )
    story_tokens_actual = len(tokenizer(story_text, add_special_tokens=False)["input_ids"])
    print(f"[theta0-diag] story prefix: {story_source} ({story_tokens_actual} tokens)", flush=True)

    results: dict[str, Any] = {}
    checkpoints: dict[str, str] = {}
    max_pos: dict[str, int | None] = {}
    for run_id in args.runs:
        print(f"[theta0-diag] === {run_id} ===", flush=True)
        manifest = load_run(run_id, store=args.store)
        ckpt = checkpoint_dir(run_id, store=args.store)
        checkpoints[run_id] = str(ckpt)
        print(f"[theta0-diag] loading checkpoint {ckpt} ...", flush=True)
        model = load_model(run_id, store=args.store, device=args.device, checkpoint=ckpt)
        max_pos[run_id] = getattr(model.config, "max_position_embeddings", None)

        run_results: dict[str, Any] = {}
        for task, fname in TASK_CONFIGS.items():
            cfg = task_cfgs[task]
            _assert_not_trained_on_eval(manifest, cfg, run_id)
            run_for_task(
                model,
                tokenizer,
                task_dfs[task],
                cfg,
                task,
                args=args,
                story_text=story_text,
                run_results=run_results,
            )
        if args.dm_mixture:
            run_dm_mixture(model, tokenizer, task_dfs["op"], run_results, args=args)
        results[run_id] = run_results

        meta = {
            "git_sha": git_commit(),
            "n_queries": args.n_queries,
            "shots": list(args.shots),
            "story_prefix_source": story_source,
            "story_prefix_tokens": story_tokens_actual,
            "tokenizer_sha": tokenizer_hash(tokenizer),
            "batch_size": args.batch_size,
            "quick": args.quick,
            "skip_label_loss": args.skip_label_loss,
            "checkpoint": dict(checkpoints),
            "max_position_embeddings": dict(max_pos),
            "dm_mixture": args.dm_mixture,
            "dm_mixture_seed": args.dm_mixture_seed,
            "dm_mixture_template_source": DM_MIXTURE_TEMPLATE_SOURCE,
        }
        _write_json(args.out, results, meta)
        print(f"[theta0-diag] wrote {args.out} (after {run_id})", flush=True)
        del model

    print_markdown_table(results, list(args.shots))


if __name__ == "__main__":
    main()
