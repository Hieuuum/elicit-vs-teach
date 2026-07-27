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

``render_translate`` (task ``arith_translate``, format_version ``v1``, added for
the phase-3 bridge) renders operator<->NL *rewriting* examples — addition only,
positive operands — whose answer slot is a rewritten **question**, never a
computed sum. It shares the same scaffold and reuses ``_NL_PHRASE['+']`` and the
operator-body construction, so its phrasings can never drift from the ones the
frozen datasets already use.

The answer slot is always the trailing run of characters after the final
``"Answer: "``; ``render`` / ``render_translate`` return its half-open
``[start, end)`` character offsets into the full string.
"""

from __future__ import annotations

OPS = ("+", "-", "*")
_NL_PHRASE = {
    "+": "What is the sum of {a} and {b}?",
    "-": "What is the difference between {a} and {b}?",
    "*": "What is the product of {a} and {b}?",
}
_OPERATOR_SYMBOL = {"+": "+", "-": "-", "*": "*"}
# Direction-prefix literals for the arith_translate task (phase-3 bridge). The
# operand phrasings themselves live in _NL_PHRASE / _OPERATOR_SYMBOL and are
# reused, so only these two rewrite instructions are new here.
_TRANSLATE_PREFIX = {
    "to_op": "Rewrite in operator notation: ",
    "to_nl": "Rewrite in words: ",
}


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


def _nl_body(a: int, b: int, op: str) -> str:
    """Natural-language question body for ``a op b`` (no scaffold)."""
    return _NL_PHRASE[op].format(a=a, b=b)


def _operator_body(a: int, b: int, op: str) -> str:
    """Operator-notation question body for ``a op b`` (no scaffold)."""
    return f"{a} {_OPERATOR_SYMBOL[op]} {b}"


def render(a: int, b: int, op: str, shown_answer: int, fmt: str) -> tuple[str, tuple[int, int]]:
    """Render one example; return ``(full_text, answer_char_span)``.

    ``shown_answer`` is what is written in the answer slot: the true answer in
    correct-label mode, an arbitrary integer in random-label mode. The prompt
    (everything before the span) is identical either way.
    """
    if fmt == "nl":
        if op not in _NL_PHRASE:
            raise ValueError(f"nl format has no phrasing for op {op!r}")
        body = _nl_body(a, b, op)
    elif fmt == "operator":
        if op not in _OPERATOR_SYMBOL:
            raise ValueError(f"unknown op {op!r}")
        body = _operator_body(a, b, op)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    prompt = f"Question: {body}\nAnswer: "
    answer_text = str(shown_answer)
    full = prompt + answer_text
    return full, (len(prompt), len(full))


def render_translate(a: int, b: int, direction: str) -> tuple[str, tuple[int, int]]:
    """Render one operator<->NL translation example (task ``arith_translate``).

    Addition only, positive operands. The answer slot holds a rewritten
    **question** — never a computed sum:

    - ``to_op``: NL question in, operator notation out
      (``"...Rewrite in operator notation: What is the sum of 23 and 45?\\nAnswer: 23 + 45"``)
    - ``to_nl``: operator notation in, NL question out
      (``"...Rewrite in words: 23 + 45\\nAnswer: What is the sum of 23 and 45?"``)

    Same ``Question:/Answer:`` scaffold as ``render``; returns the half-open
    ``[start, end)`` character offsets of the answer text.
    """
    if direction == "to_op":
        body = _TRANSLATE_PREFIX["to_op"] + _nl_body(a, b, "+")
        answer_text = _operator_body(a, b, "+")
    elif direction == "to_nl":
        body = _TRANSLATE_PREFIX["to_nl"] + _operator_body(a, b, "+")
        answer_text = _nl_body(a, b, "+")
    else:
        raise ValueError(f"unknown translate direction {direction!r}")
    prompt = f"Question: {body}\nAnswer: "
    full = prompt + answer_text
    return full, (len(prompt), len(full))
