"""geode.arith — arithmetic task formats, evals, and dataset validators.

Per the 2026-07-17 pivot (EXPERIMENTS.md), datasets are generated once by a
script and frozen on HF; this package does **not** generate datasets. It holds
only the pieces whose silent failure would waste GPU budget or invalidate the
elicit-vs-teach comparison, so they are tested core:

- ``formats`` — tokenizer-agnostic rendering + answer char spans.
- ``labels``  — the random-label rule for the format-installer runs.
- ``evals``   — answer parser, exact-match, format-validity, few-shot prompts.
- ``decode``  — greedy token-prefix completions (gates + in-loop G4 eval).
- ``spans``   — char-span→token-span conversion for the frozen SFT files.
- ``stratify``— capacity-aware, capacity-capped water-fill allocation over cells.
- ``validate``— probe-leakage / stratification / uniqueness checks + the
  content-and-order hash over a materialised dataset.
"""

from __future__ import annotations

from geode.arith.decode import greedy_completions
from geode.arith.evals import exact_match, few_shot_prompt, format_valid, parse_answer
from geode.arith.formats import OPS, digits, render, true_answer
from geode.arith.labels import permute_labels, random_label
from geode.arith.spans import SftExample, token_label_span, tokenize_with_spans
from geode.arith.stratify import DIGIT_BAND_SIZES, allocate, capacity
from geode.arith.validate import cell_counts, order_hash, probe_leakage, uniqueness_by_cell

__all__ = [
    "DIGIT_BAND_SIZES",
    "OPS",
    "SftExample",
    "allocate",
    "capacity",
    "cell_counts",
    "digits",
    "exact_match",
    "few_shot_prompt",
    "format_valid",
    "greedy_completions",
    "order_hash",
    "parse_answer",
    "permute_labels",
    "probe_leakage",
    "random_label",
    "render",
    "token_label_span",
    "tokenize_with_spans",
    "true_answer",
    "uniqueness_by_cell",
]
