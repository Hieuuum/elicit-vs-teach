"""geode.adapt — model-family layout + checked answer-token pairs (spec 02 §7.1).

The circuit / lens / patching tools were written against Llama module names
and the arithmetic task. This package is the thin layer that lets them run on
another model family or another short-answer task without forking them:

- ``models`` — where each node lives (attention/MLP output projections,
  embedding, norms) per ``config.model_type``; Llama paths unchanged.
- ``pairs``  — first-answer-token alignment in context, counterfactual pairs
  (length-matched 'different item' pairs, name-swap pairs), pair invariants.
"""

from __future__ import annotations

from geode.adapt.models import ModelLayout, family_of, layout, weight_groups
from geode.adapt.pairs import (
    AlignmentError,
    Pair,
    ScoredItem,
    check_pairs,
    encode,
    first_answer_token,
    fresh_names,
    length_matched_pairs,
    score_item,
    swap_pairs,
    swap_subject,
)

__all__ = [
    "AlignmentError",
    "ModelLayout",
    "Pair",
    "ScoredItem",
    "check_pairs",
    "encode",
    "family_of",
    "first_answer_token",
    "fresh_names",
    "layout",
    "length_matched_pairs",
    "score_item",
    "swap_pairs",
    "swap_subject",
    "weight_groups",
]
