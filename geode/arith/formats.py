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

A third format deliberately BREAKS that scaffold (fig2nl3, 2026-08-12 —
spec 02 §4): the fig2nl2 sweep measured that the ``Question:/Answer:``
scaffold alone pre-installs the output convention (base Llama: ~0.31
zero-shot EM, ~0.83 format validity before any training), which removes the
format-learning transient the paper's Figure-2 pre-elicit intervention
exists to remove — so under the scaffold the intervention has nothing to
buy and the paper's gap cannot appear. ``bare_nl`` restores the paper's
regime: the bare question, a newline, and the answer as plain continuation.

- ``bare_nl`` — natural language, NO scaffold, add/sub/mult:
  ``"What is the sum of 23 and 45?\n68"``

A fourth format (ts1b op-install, 2026-08-28 — spec 02 §5) is the paper's
pre-training-intervention surface (App. I.2.1: ``"2 * 3 = 6"``): bare
operator notation, no scaffold, the answer as the continuation after
``" = "``. It exists so the intervention task (install add/sub in operator
form) and the target task (``bare_nl`` add/sub) share NO surface form
beyond the digits themselves — the op→NL latency claim under test is
exactly that transfer. The prompt-side trailing space before the answer is
the same whitespace-overhang boundary the frozen ``"Answer: "`` scaffold
uses (V5.38), so span→token derivation is unchanged.

- ``bare_op`` — operator notation, NO scaffold, add/sub/mult:
  ``"23 + 45 = 68"``

The two scaffolded formats stay byte-frozen exactly as on 2026-07-17;
``bare_nl`` is additive and reuses ``_NL_PHRASE`` verbatim, so its question
bodies can never drift from the frozen NL sets'. The answer char span works
identically (the trailing run after the final ``"\n"`` here); the span→token
boundary is safe under the frozen Llama-3 byte-level BPE because its
pre-tokenizer merges ``"?\n"`` into the prompt-side punctuation token and
starts a fresh numeric (or sign) token at the answer — the same
whitespace-overhang-only rule V5.38 verifies.

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
    elif fmt == "bare_nl":
        if op not in _NL_PHRASE:
            raise ValueError(f"bare_nl format has no phrasing for op {op!r}")
        # No Question:/Answer: scaffold (module docstring): the bare question,
        # a newline, the answer as plain continuation — the paper's regime.
        body = _nl_body(a, b, op)
        prompt = f"{body}\n"
        answer_text = str(shown_answer)
        full = prompt + answer_text
        return full, (len(prompt), len(full))
    elif fmt == "bare_op":
        if op not in _OPERATOR_SYMBOL:
            raise ValueError(f"unknown op {op!r}")
        # Paper App. I.2.1 intervention surface: "2 * 3 = 6". No scaffold; the
        # trailing "= " space is the same whitespace-overhang boundary as the
        # frozen "Answer: " (module docstring).
        body = _operator_body(a, b, op)
        prompt = f"{body} = "
        answer_text = str(shown_answer)
        full = prompt + answer_text
        return full, (len(prompt), len(full))
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
