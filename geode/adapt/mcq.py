"""Multiple-choice rendering and option-swap counterfactuals (spec 02 §7.1, V5.79).

WMDP / MMLU items are scored on the answer LETTER. The prompt format is the
lm-evaluation-harness zero-shot MMLU/WMDP format (the one the published WMDP
accuracies use):

    The following are multiple choice questions (with answers) about biology.

    <question>
    A. <choice 0>
    B. <choice 1>
    C. <choice 2>
    D. <choice 3>
    Answer:

and the answer is " A" / " B" / ... (a leading space, as the harness scores it).

Counterfactual = OPTION SWAP: the same question with the correct option's text
exchanged with one distractor option's text. The question, the set of option
texts and every other line are unchanged; only which LETTER holds the correct
content changes. So logit(clean letter) - logit(corrupt letter) isolates "find
the known-correct content and bind it to its letter" — a model that does not
know the answer has no reason to prefer either letter (its letter bias cancels
between the two runs), which is what the performing guard needs. Swapping two
option lines keeps the token length in practice; pairs where it does not are
dropped by the caller (``check_pairs``).
"""

from __future__ import annotations

import random
from typing import Sequence

LETTERS = ("A", "B", "C", "D")


def render_mcq(description: str, question: str, choices: Sequence[str]) -> str:
    """The harness prompt (ends with 'Answer:'; the answer token is ' <letter>')."""
    if len(choices) > len(LETTERS):
        raise ValueError(f"render_mcq: {len(choices)} choices > {len(LETTERS)} letters")
    head = f"{description}\n\n" if description else ""
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{head}{question.strip()}\n{body}\nAnswer:"


def answer_text(idx: int) -> str:
    return " " + LETTERS[idx]


def balanced_permutation(n_choices: int, answer: int, target_idx: int, rng: random.Random) -> list[int]:
    """A permutation ``perm`` of range(n_choices) (new position -> old index)
    that puts the original correct option at ``target_idx`` and shuffles the
    rest — used to make the correct letter uniform over items."""
    rest = [i for i in range(n_choices) if i != answer]
    rng.shuffle(rest)
    perm = rest[:target_idx] + [answer] + rest[target_idx:]
    assert perm[target_idx] == answer and sorted(perm) == list(range(n_choices))
    return perm


def swap_options(choices: Sequence[str], i: int, j: int) -> list[str]:
    """``choices`` with positions i and j exchanged."""
    if i == j:
        raise ValueError("swap_options: i == j")
    out = list(choices)
    out[i], out[j] = out[j], out[i]
    return out


def option_line_spans(prompt: str, choices: Sequence[str]) -> list[tuple[int, int]]:
    """Character spans of the option lines of a ``render_mcq`` prompt, located
    from the END of the prompt (a question may itself contain 'A. ')."""
    lines = [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    tail = "\n" + "\n".join(lines) + "\nAnswer:"
    if not prompt.endswith(tail):
        raise ValueError("option_line_spans: prompt does not end with these options")
    pos = len(prompt) - len(tail) + 1
    spans = []
    for ln in lines:
        spans.append((pos, pos + len(ln)))
        pos += len(ln) + 1
    return spans
