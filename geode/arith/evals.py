"""Answer parsing + accuracy for arithmetic outputs (spec 02 §5 "Evals").

These guard the verification gates (§8): a broken parser silently reports the
wrong accuracy, a gate passes on garbage, and GPU budget is wasted. Hence
tested core, not script land.

The integer parser reads the answer slot the same way ``formats.render`` writes
it: the trailing signed integer after the final ``"Answer:"`` delimiter, shared
by both arithmetic formats since the 2026-07-17 scaffold. Anything that does
not parse to ``-?\\d+`` there is malformed and yields ``None``.

``text_exact_match`` serves the phase-3 translation bridge, whose answer slot is
a rewritten question rather than an integer. It compares that slot after
stripping only surrounding whitespace; punctuation, word order, and internal
spacing remain exact.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from geode.arith.decode import greedy_completions

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_INT = re.compile(r"^-?\d+$")
_ANSWER_MARKER = "Answer:"


def parse_answer(text: str) -> int | None:
    """Extract the integer in the answer slot, or ``None`` if malformed.

    The slot is whatever follows the final ``"Answer:"`` in ``text``; a model
    may emit trailing whitespace, so it is stripped before matching. Negatives
    are honoured; a missing delimiter, empty slot, or non-numeric slot returns
    ``None``.
    """
    idx = text.rfind(_ANSWER_MARKER)
    if idx == -1:
        return None
    slot = text[idx + len(_ANSWER_MARKER) :].strip()
    if not _INT.match(slot):
        return None
    return int(slot)


def exact_match(output: str, answer: int) -> bool:
    """True iff the parsed answer slot equals ``answer`` exactly."""
    return parse_answer(output) == answer


@torch.no_grad()
def exact_match_accuracy(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    prompt_ids: list[list[int]],
    answers: list[int],
    *,
    device: str,
    batch_size: int,
) -> tuple[float, list[str]]:
    """Run the shared greedy protocol and return aggregate integer accuracy."""
    if len(prompt_ids) != len(answers):
        raise ValueError(
            f"exact_match_accuracy: {len(prompt_ids)} prompts != {len(answers)} answers"
        )
    completions = greedy_completions(
        model, tokenizer, prompt_ids, device=device, batch_size=batch_size
    )
    hits = [
        exact_match("Answer:" + completion, answer)
        for completion, answer in zip(completions, answers)
    ]
    return sum(hits) / max(1, len(hits)), completions


def text_exact_match(output: str, answer: str) -> bool:
    """True iff the final answer slot equals ``answer`` as text.

    Surrounding whitespace is ignored because decoding starts at a token-prefix
    boundary whose first generated token may carry the answer slot's leading
    space. Everything inside the slot remains byte-exact.
    """
    idx = output.rfind(_ANSWER_MARKER)
    return idx != -1 and output[idx + len(_ANSWER_MARKER) :].strip() == answer


def format_valid(output: str) -> bool:
    """True iff the answer slot parses as an integer (spec 02 §8 G4)."""
    return parse_answer(output) is not None


def few_shot_prompt(shots: list[str], query_prompt: str) -> str:
    """Build a zero/16-shot prompt (spec 02 §8 G5).

    ``shots`` are fully rendered two-line ``"Question: ...\nAnswer: 68"``
    exemplars; ``query_prompt`` is a rendered prompt with an *empty* answer slot
    (i.e. ends in ``"Answer: "``). Exemplars are separated by a blank line so the
    two-line scaffold stays legible. Zero-shot is ``shots == []``.
    """
    return "\n\n".join([*shots, query_prompt])
