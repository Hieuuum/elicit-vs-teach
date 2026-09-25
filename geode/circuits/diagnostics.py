"""Verified reciprocal candidates with matched types and token lengths.

Each source pair contributes the same two candidate strings once as positive
and once as negative. Cluster all four resulting probe rows by BOTH source
groups to preserve exact answer-frequency balance in held-out partitions.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Sequence
import random
import re
from typing import Any

from .code_eval import parse_crux_answer
from .data import Example, grade, numeric_answer


def canonical_candidate(example: Example) -> str | None:
    """Return the unknown answer component, without known assertion boilerplate."""
    if example.task == "gsm_symbolic":
        number = numeric_answer(example.answer)
        return str(number) if number is not None else None
    if example.task.startswith("cruxeval_"):
        return parse_crux_answer(example.answer, example.task.removeprefix("cruxeval_"))
    if example.task.startswith("ethics_"):
        return example.answer
    raise ValueError(f"unsupported diagnostic task {example.task}")


def _numeric_shape(text: str) -> tuple:
    match = re.fullmatch(r"(-?)(\d+)(?:\.(\d*))?", text)
    if match is None:
        return ("non_decimal_expression",)
    return (bool(match[1]), len(match[2]), None if match[3] is None else len(match[3]))


def _ast_type(node: ast.AST) -> tuple:
    """Match observable answer types, argument arity and scalar number shape.

    Container lengths are deliberately not required: exact token count is the
    length constraint. Container element type sets exclude easy type shortcuts.
    Expressions are matched by AST shape when their value cannot be read as a
    literal without executing the benchmark function.
    """
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        value = None
        literal = False
    else:
        literal = True
    if literal:
        if isinstance(value, bool) or value is None:
            return (type(value).__name__,)
        if isinstance(value, (int, float)):
            return (type(value).__name__, _numeric_shape(ast.unparse(node)))
        if isinstance(value, (list, tuple, set)):
            return (type(value).__name__, tuple(sorted({type(v).__name__ for v in value})))
        if isinstance(value, dict):
            return (
                "dict",
                tuple(sorted({type(k).__name__ for k in value})),
                tuple(sorted({type(v).__name__ for v in value.values()})),
            )
        return (type(value).__name__,)
    if isinstance(node, ast.Call):
        function = ast.unparse(node.func)
        return (
            "call",
            function,
            tuple(_ast_type(arg) for arg in node.args),
            tuple((k.arg, _ast_type(k.value)) for k in node.keywords),
        )
    if isinstance(node, ast.Lambda):
        return ("lambda", len(node.args.args))
    if isinstance(node, ast.Name):
        return ("name",)
    if isinstance(node, ast.BinOp):
        return ("binop", type(node.op).__name__, _ast_type(node.left), _ast_type(node.right))
    return (type(node).__name__,)


def candidate_signature(example: Example, candidate: str) -> tuple:
    """Public, recorded matching rule, independent of labels and model features."""
    if example.task == "gsm_symbolic":
        return ("number", _numeric_shape(candidate))
    if example.task.startswith("cruxeval_"):
        return _ast_type(ast.parse(candidate, mode="eval").body)
    return ("ethics_label",)


def _well_formed_wrong(result: dict[str, Any]) -> bool:
    return not any(
        result.get(key, False)
        for key in ("correct", "parse_failure", "timed_out", "execution_failure")
    )


def valid_corruption(clean: Example, corrupt: Example) -> bool:
    """Require clean's canonical answer to be demonstrably incorrect on corrupt."""
    if clean.task != corrupt.task:
        raise ValueError("corruption must preserve task")
    candidate = canonical_candidate(clean)
    return candidate is not None and _well_formed_wrong(grade(corrupt, candidate))


def build_candidate_pairs(
    examples: Sequence[Example],
    tokenizer: Any,
    *,
    seed: int = 0,
    max_pairs: int = 128,
) -> list[tuple[Example, Example, str, str]]:
    """Disjoint reciprocal pairs with valid golds and executable wrong negatives.

    Call separately for each task. Every returned pair contributes four probe
    rows: (a,A,1), (a,B,0), (b,B,1), (b,A,0). Do not split those rows apart.
    """
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if len({example.task for example in examples}) > 1:
        raise ValueError("build_candidate_pairs requires one task per call")
    rng = random.Random(seed)
    buckets: dict[tuple, list[tuple[Example, str]]] = defaultdict(list)
    for example in sorted(examples, key=lambda e: e.id):
        candidate = canonical_candidate(example)
        if candidate is None:
            continue
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if not tokens:
            continue
        buckets[(candidate_signature(example, candidate), len(tokens))].append((example, candidate))
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def verdict(example: Example, candidate: str) -> dict[str, Any]:
        key = (example.id, candidate)
        if key not in cache:
            cache[key] = grade(example, candidate)
        return cache[key]

    keys = sorted(buckets, key=repr)
    rng.shuffle(keys)
    pairs = []
    used: set[str] = set()
    for key in keys:
        rows = buckets[key]
        rng.shuffle(rows)
        for a, candidate_a in rows:
            if a.group in used or not verdict(a, candidate_a)["correct"]:
                continue
            for b, candidate_b in rows:
                if b.group == a.group or b.group in used or candidate_a == candidate_b:
                    continue
                if not verdict(b, candidate_b)["correct"]:
                    continue
                if not _well_formed_wrong(verdict(a, candidate_b)):
                    continue
                if not _well_formed_wrong(verdict(b, candidate_a)):
                    continue
                pairs.append((a, b, candidate_a, candidate_b))
                used.update((a.group, b.group))
                break
            if len(pairs) >= max_pairs:
                return pairs
    return pairs
