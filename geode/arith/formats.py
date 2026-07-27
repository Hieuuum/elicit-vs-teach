"""Arithmetic example rendering: (operands, op, shown answer, format) -> text.

Tokenizer-agnostic by design (EXPERIMENTS.md decision 2026-07-17, OPEN(11)
unresolved): rendering emits plain text plus the *character* span of the
answer slot. Token-level label spans are derived at load time once a
tokenizer is fixed, so no tokenizer is baked into the frozen dataset.

Both formats share a ``Question: <body>\nAnswer: <answer>`` scaffold (owner
decision 2026-07-17, freezing OPEN(9)); only the question body differs:

- ``operator`` — add/sub/mult:  ``"Question: 23 + 45\nAnswer: 68"``
- ``nl`` — natural language, add/sub/mult:
  ``"Question: What is the sum of 23 and 45?\nAnswer: 68"``

The mult phrasing arrived 2026-07-27 for the phase-3 format installer, which
needs an NL-format set whose *operation* cannot disturb the parent's addition
(see ``datagen/make_data.py`` --phase3). The add/sub phrasings are byte-frozen:
they render exactly as they did on 2026-07-17, so every ``order_hash`` pinned
against ``D_algo`` stays valid.

The answer slot is always the trailing run of characters after the final
``"Answer: "``; ``render`` returns its half-open ``[start, end)`` character
offsets into the full string.
"""

from __future__ import annotations

OPS = ("+", "-", "*")
_NL_PHRASE = {
    "+": "What is the sum of {a} and {b}?",
    "-": "What is the difference between {a} and {b}?",
    "*": "What is the product of {a} and {b}?",
}
_OPERATOR_SYMBOL = {"+": "+", "-": "-", "*": "*"}


def digits(n: int) -> int:
    """Number of decimal digits of ``|n|`` (``digits(0) == 1``)."""
    return len(str(abs(int(n))))


def true_answer(a: int, b: int, op: str) -> int:
    """The correct result of ``a op b``."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unknown op {op!r}")


def render(a: int, b: int, op: str, shown_answer: int, fmt: str) -> tuple[str, tuple[int, int]]:
    """Render one example; return ``(full_text, answer_char_span)``.

    ``shown_answer`` is what is written in the answer slot: the true answer in
    correct-label mode, an arbitrary integer in random-label mode. The prompt
    (everything before the span) is identical either way.
    """
    if fmt == "nl":
        if op not in _NL_PHRASE:
            raise ValueError(f"nl format has no phrasing for op {op!r}")
        body = _NL_PHRASE[op].format(a=a, b=b)
    elif fmt == "operator":
        if op not in _OPERATOR_SYMBOL:
            raise ValueError(f"unknown op {op!r}")
        body = f"{a} {_OPERATOR_SYMBOL[op]} {b}"
    else:
        raise ValueError(f"unknown format {fmt!r}")
    prompt = f"Question: {body}\nAnswer: "
    answer_text = str(shown_answer)
    full = prompt + answer_text
    return full, (len(prompt), len(full))
