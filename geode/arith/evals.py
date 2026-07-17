"""Answer parsing + accuracy for arithmetic outputs (spec 02 §5 "Evals").

These guard the verification gates (§8): a broken parser silently reports the
wrong accuracy, a gate passes on garbage, and GPU budget is wasted. Hence
tested core, not script land.

The parser reads the answer slot the same way ``formats.render`` writes it:
the trailing signed integer after the final ``"Answer:"`` delimiter, shared by
both formats since the 2026-07-17 scaffold. Anything that does not parse to
``-?\\d+`` there is malformed and yields ``None``.
"""

from __future__ import annotations

import re

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
