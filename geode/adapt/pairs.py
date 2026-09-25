"""Answer-token alignment and counterfactual pairs for the circuit tools (spec 02 §7.1).

Every pair-based metric (attribution maps, faithfulness, edges, DCM, DAS,
hidden preference) reads ONE next-token logit difference at the last prompt
position: logit(clean answer token) - logit(contrast token). Two silent
failures would corrupt every such number without a crash:

1. **Answer-token misalignment.** The scored token must be the token the
   model actually emits at that position, i.e. the first token of the answer
   *tokenized in context*. BPE merges across the prompt/answer boundary (a
   leading space, a newline) make ``tokenizer(answer)[0]`` differ from it.
   ``first_answer_token`` tokenizes prompt and prompt+answer together, demands
   that the prompt's ids are an exact prefix of the joint ids, and returns the
   joint token at the boundary (V5.76).
2. **Mismatched counterfactuals.** Clean and corrupt prompts must have equal
   token length (activations are patched position by position) and different
   scored tokens (else the metric is identically 0). For a name-swap
   corruption, the two prompts must also differ ONLY inside the swapped name's
   tokens — anything else leaking into the corrupt run is a second, unintended
   intervention (V5.77).

Nothing here knows about a particular task: task adapters render strings,
these helpers turn them into checked token pairs. A pair is the tools' tuple
``(clean_ids, corrupt_ids, clean_tok, contrast_tok)``; ``contrast_tok = -1``
marks an UNLABELLED counterfactual (the target is the model's own output on
the counterfactual prompt; DCM only).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

Pair = tuple[list[int], list[int], int, int]


class AlignmentError(ValueError):
    """The prompt's token ids are not a prefix of the prompt+answer ids."""


def encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids with no auto-added specials (templates carry their own)."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def first_answer_token(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], int]:
    """(prompt ids, first answer token) with the answer tokenized IN CONTEXT.

    Raises ``AlignmentError`` when tokenizing prompt+answer does not reproduce
    the prompt's ids as a prefix (a merge across the boundary) or when the
    answer adds no token.
    """
    p = encode(tokenizer, prompt)
    full = encode(tokenizer, prompt + answer)
    if len(full) <= len(p) or full[: len(p)] != p:
        raise AlignmentError(
            f"prompt ids are not a prefix of prompt+answer ids at the boundary "
            f"(prompt ends {prompt[-12:]!r}, answer starts {answer[:12]!r})")
    return p, full[len(p)]


@dataclass
class ScoredItem:
    """One probe: prompt ids, the correct next token, contrast tokens."""

    item_id: str
    prompt: str
    prompt_ids: list[int]
    target: int
    distractors: list[int]                 # same-slot wrong answers (first = primary)
    meta: dict = field(default_factory=dict)


def score_item(tokenizer: Any, item_id: str, prompt: str, answer: str,
               distractor_answers: Sequence[str], meta: dict | None = None) -> ScoredItem:
    """Align the answer and every distractor at the same boundary.

    Distractors whose first token equals the target (or fail alignment) are
    dropped; an item with no usable distractor raises ``AlignmentError``.
    """
    ids, tgt = first_answer_token(tokenizer, prompt, answer)
    dis: list[int] = []
    for d in distractor_answers:
        try:
            _, t = first_answer_token(tokenizer, prompt, d)
        except AlignmentError:
            continue
        if t != tgt and t not in dis:
            dis.append(t)
    if not dis:
        raise AlignmentError(f"{item_id}: no distractor with a first token different from the answer's")
    return ScoredItem(item_id, prompt, ids, tgt, dis, dict(meta or {}))


def length_matched_pairs(items: Sequence[ScoredItem], n_pairs: int, seed: int,
                         partners: int = 1) -> list[Pair]:
    """Arithmetic-style 'different problem' pairs: bucket by prompt length,
    pair items within a bucket whose targets differ; contrast = the corrupt
    item's own target. ``partners`` > 1 pairs each item with that many
    rotations of its bucket (more pairs from a small item set)."""
    rng = random.Random(seed)
    buckets: dict[int, list[ScoredItem]] = {}
    for it in items:
        buckets.setdefault(len(it.prompt_ids), []).append(it)
    pairs: list[Pair] = []
    for length in sorted(buckets):
        b = buckets[length]
        rng.shuffle(b)
        for r in range(1, min(partners, len(b) - 1) + 1):
            for i, a in enumerate(b):
                c = b[(i + r) % len(b)]
                if a.item_id != c.item_id and a.target != c.target:
                    pairs.append((a.prompt_ids, c.prompt_ids, a.target, c.target))
    rng.shuffle(pairs)
    return pairs[:n_pairs]


# ----------------------------------------------------------- name swaps
_FRESH_FIRST = ("Arlo", "Brisa", "Caspian", "Delphine", "Ezra", "Fenna", "Gideon", "Halina",
                "Ismo", "Jorun", "Kasimir", "Liesel", "Matthias", "Nerys", "Oskar", "Priya",
                "Quentin", "Rosalind", "Soren", "Tamsin", "Ulrich", "Vesna", "Wendel", "Ximena",
                "Yannick", "Zelda", "Anouk", "Bertil", "Corvin", "Dagny", "Elio", "Fiorella",
                "Gunnar", "Henrike", "Ilario", "Juno", "Kerttu", "Leander", "Mireille", "Nils")
_FRESH_LAST = ("Achterberg", "Bramwell", "Castellan", "Dunmore", "Eskildsen", "Falkenrath",
               "Gorski", "Holloway", "Ingram", "Jakobsen", "Kettering", "Lindqvist", "Marchetti",
               "Northcott", "Oyelaran", "Pellegrini", "Quist", "Rasmussen", "Sandoval", "Thorsen",
               "Uhlmann", "Vasquez-Rhee", "Westergaard", "Yilmazer", "Zandvoort", "Abernathy",
               "Blom", "Carrow", "Dahlberg", "Engstrom", "Fairweather", "Grisolia", "Haverkamp",
               "Iversen", "Juhl", "Kowalczyk", "Lachance", "Montague", "Nakashima-Roe", "Oberlin")


def fresh_names(exclude_text: str = "", seed: int = 316) -> list[str]:
    """A seeded list of made-up 'First Last' / 'First Middle Last' names, none
    of whose words occurs in ``exclude_text`` (e.g. the whole dataset)."""
    rng = random.Random(seed)
    words = set(re.findall(r"[\w\-']+", exclude_text))
    firsts = [f for f in _FRESH_FIRST if f not in words]
    lasts = [s for s in _FRESH_LAST if s not in words]
    names = [f"{f} {s}" for f in firsts for s in lasts]
    names += [f"{f} {g} {s}" for f, g, s in zip(firsts, reversed(firsts), lasts)]
    rng.shuffle(names)
    return names


def _name_positions(tokenizer: Any, text: str, name: str) -> set[int]:
    """Token positions overlapping any occurrence of ``name`` in ``text``."""
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(name), text)]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return {i for i, (s, e) in enumerate(enc["offset_mapping"])
            if any(s < ce and e > cs for cs, ce in spans)}


def swap_subject(tokenizer: Any, prompt: str, subject: str, candidates: Sequence[str],
                 max_tries: int = 400) -> tuple[str, list[int], str] | None:
    """Replace every occurrence of ``subject`` in ``prompt`` by the first
    candidate that (a) keeps the token length, (b) changes the token ids only
    at positions overlapping the name (V5.77). Returns (new prompt, ids,
    replacement) or None."""
    if subject not in prompt:
        return None
    clean_ids = encode(tokenizer, prompt)
    allowed = _name_positions(tokenizer, prompt, subject)
    for cand in list(candidates)[:max_tries]:
        if cand == subject:
            continue
        new = prompt.replace(subject, cand)
        ids = encode(tokenizer, new)
        if len(ids) != len(clean_ids):
            continue
        diff = {i for i, (a, b) in enumerate(zip(clean_ids, ids)) if a != b}
        if diff and diff <= allowed and diff <= _name_positions(tokenizer, new, cand):
            return new, ids, cand
    return None


def swap_pairs(tokenizer: Any, items: Sequence[ScoredItem], candidates: Sequence[str],
               n_pairs: int, seed: int) -> list[Pair]:
    """Name-swap pairs: clean = the item, corrupt = the same prompt with the
    subject (``item.meta['subject']``) replaced by an equal-length unknown
    name; contrast = the item's primary distractor (same slot, wrong fact)."""
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    pairs: list[Pair] = []
    for it in order:
        cands = list(candidates)
        rng.shuffle(cands)
        got = swap_subject(tokenizer, it.prompt, it.meta.get("subject", ""), cands)
        if got is None:
            continue
        pairs.append((it.prompt_ids, got[1], it.target, it.distractors[0]))
        if len(pairs) >= n_pairs:
            break
    return pairs


def check_pairs(pairs: Sequence[Pair]) -> None:
    """Raise if any pair violates the tools' invariants (equal lengths,
    different scored tokens unless unlabelled)."""
    for k, (c, x, ct, xt) in enumerate(pairs):
        if len(c) != len(x):
            raise ValueError(f"pair {k}: clean/corrupt lengths differ ({len(c)} vs {len(x)})")
        if xt != -1 and ct == xt:
            raise ValueError(f"pair {k}: clean and contrast tokens coincide ({ct})")
