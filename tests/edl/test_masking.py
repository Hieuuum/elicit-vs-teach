"""EDL-1 tests: label masking (M1) + masking config hash (``geode.edl.masking``).

Derived from spec, never from an implementation:

- specs/01-edl-harness.md §2 "Masking rules" (M1 — loss is computed on label
  tokens only; never prompt or formatting tokens; the mask is the single shared
  construction that mechanically enforces this) and §3 "Public API" (``label_mask``).
- specs/00-interfaces.md §5 "Test loss" — ``masking_config_hash`` is the
  mechanical guard for the train/test mask-parity footgun.
- PLAN.md "### EDL-1" and resolved decisions:
  - OQ-8: the dataset builder emits per-example label-token spans;
    ``tests/conftest.py::TaskExample`` carries ``input_ids`` + half-open
    ``label_span`` ``[start, end)``. ``label_mask`` is the single construction
    path from a batch of span-carrying examples to a boolean mask.
  - OQ-7: ``masking_config_hash`` is the sha256 hex digest of the canonical JSON
    of ``{task name, format_version, mask-rule parameters, tokenizer_hash}`` —
    deterministic across processes (canonical serialization, not object
    identity) and sensitive to every component.

Written test-first: the imports below target the public API fixed in PLAN.md
"### EDL-1" and are expected to fail until ``geode/edl/masking.py`` exists.

Design notes (so the TEST-AUDITOR can check the reasoning, not guess it):

- ``TaskFormat`` carries span-rule parameters beyond ``name``/``format_version``
  (PLAN.md: "# + span rule params (OQ-8)"), but their names are an
  implementation choice this stage must not invent. Two consequences:
    * the mask tests construct ``TaskFormat`` from ``name``/``format_version``
      only (the mask is built from the *examples'* spans, not from the format —
      OQ-8), which requires those extra parameters to be defaulted;
    * the hash-sensitivity test discovers the extra fields via
      ``dataclasses.fields`` and perturbs them, so it stays faithful to OQ-7
      ("mask-rule parameters ... feed the hash") without naming any of them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import torch

from geode.edl.masking import TaskFormat, label_mask, masking_config_hash
from tests.conftest import BOS_ID, TaskExample

# OQ-7: masking_config_hash is a sha256 hex digest — 64 lowercase hex chars.
_HEX = set("0123456789abcdef")

TASK_NAME = "arith_copy"
FORMAT_VERSION = "2"
TOK_HASH_A = "a1" * 32  # two distinct, well-formed sha256-shaped tokenizer hashes
TOK_HASH_B = "b2" * 32

# Snippet run in a fresh interpreter (test 3): imports the module, rebuilds the
# *same* config from scratch, and prints its hash. A fresh process gives fresh
# object identities and (with a distinct PYTHONHASHSEED) a distinct built-in
# hash seed, so an equal digest rules out object-identity hashing and any
# dependence on Python's randomized built-in ``hash()``.
_SUBPROCESS_SNIPPET = (
    "from geode.edl.masking import TaskFormat, masking_config_hash\n"
    "tf = TaskFormat(name={name!r}, format_version={version!r})\n"
    "print(masking_config_hash(tf, {tok!r}))\n"
).format(name=TASK_NAME, version=FORMAT_VERSION, tok=TOK_HASH_A)


def _task_format(**overrides: object) -> TaskFormat:
    """Build a ``TaskFormat`` from name/format_version (span rule defaulted)."""
    params: dict[str, object] = {"name": TASK_NAME, "format_version": FORMAT_VERSION}
    params.update(overrides)
    return TaskFormat(**params)  # type: ignore[arg-type]


def _ragged_batch() -> list[TaskExample]:
    """Hand-built batch with ragged lengths and varied spans.

    Exercises: multi-token spans, single-token spans, a span that runs to the
    end of its sequence, prompt tokens before a span, formatting tokens after a
    span, and padding positions for the shorter sequences.
    """
    return [
        TaskExample(input_ids=[BOS_ID, 5, 6, 7], label_span=(1, 3)),  # len 4, labels @1,2
        TaskExample(input_ids=[BOS_ID, 8], label_span=(1, 2)),  # len 2, label @1; @2,3,4 pad
        TaskExample(input_ids=[BOS_ID, 9, 10, 11, 12], label_span=(2, 5)),  # len 5, labels @2-4
        TaskExample(input_ids=[BOS_ID, 13, 14], label_span=(2, 3)),  # len 3, label @2; @3,4 pad
    ]


def _build_batch(kind, copy_token_task, random_label_task) -> list[TaskExample]:
    if kind == "copy":
        return copy_token_task(seed=0, n=5)
    if kind == "random":
        return random_label_task(seed=1, n=5)
    if kind == "ragged":
        return _ragged_batch()
    raise ValueError(kind)


def _perturb(value: object) -> object:
    """Return a value distinct from ``value`` for common mask-rule param types."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "_perturbed"
    if isinstance(value, tuple):
        return (*value, "extra")
    if isinstance(value, list):
        return [*value, "extra"]
    if isinstance(value, frozenset):
        return frozenset(value | {"extra"})
    if isinstance(value, set):
        return value | {"extra"}
    return ("perturbed", value)


# ---------------------------------------------------------------------------
# M1 — the mask covers exactly the label-token spans, nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_kind", ["copy", "random", "ragged"])
def test_mask_covers_exactly_label_tokens(batch_kind, copy_token_task, random_label_task) -> None:
    """M1: mask True on ``[start, end)`` of each example, False everywhere else.

    "Everywhere else" is three regions, all required False by M1 (spec 01 §2):
    prompt tokens (``p < start``), formatting tokens (``end <= p < len``), and
    padding positions of shorter sequences (``p >= len``). The elementwise
    equality below checks both directions at once: no label position missing,
    no non-label position set.
    """
    batch = _build_batch(batch_kind, copy_token_task, random_label_task)
    max_len = max(len(ex.input_ids) for ex in batch)

    mask = label_mask(batch, _task_format())

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (len(batch), max_len)

    for i, ex in enumerate(batch):
        start, end = ex.label_span
        for p in range(max_len):
            expected = start <= p < end
            assert bool(mask[i, p]) is expected, (batch_kind, i, p, ex.label_span)


# ---------------------------------------------------------------------------
# M1 / V1.6 feeder — mask population equals the label-span token count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_kind", ["copy", "random", "ragged"])
def test_mask_count_matches_label_span_lengths(
    batch_kind, copy_token_task, random_label_task
) -> None:
    """``mask.sum()`` == Σ (end − start); each row sum == its span length.

    This is what ``label_token_count`` bookkeeping (spec 00 §3, feeds V1.6) will
    be derived from, so it must match the spans exactly.
    """
    batch = _build_batch(batch_kind, copy_token_task, random_label_task)

    mask = label_mask(batch, _task_format())

    total_expected = sum(end - start for (start, end) in (ex.label_span for ex in batch))
    assert int(mask.sum().item()) == total_expected

    for i, ex in enumerate(batch):
        start, end = ex.label_span
        assert int(mask[i].sum().item()) == (end - start)


# ---------------------------------------------------------------------------
# OQ-7 — the hash is canonical (same config ⇒ same hash, even across processes)
# ---------------------------------------------------------------------------


def test_same_config_same_hash_across_processes() -> None:
    """Identical config ⇒ identical hash, even across fresh interpreters.

    Two independently constructed ``TaskFormat`` objects (distinct identities)
    must hash identically, and two *fresh interpreters* with *different*
    ``PYTHONHASHSEED`` values must reproduce the same digest. That rules out a
    hash built from object identity or from Python's per-process ``hash()``
    randomization — the failure mode that makes the V0.5/V1.4 guard fire
    spuriously or never.
    """
    tf1 = _task_format()
    tf2 = _task_format()
    assert tf1 is not tf2

    h_local = masking_config_hash(tf1, TOK_HASH_A)
    assert masking_config_hash(tf2, TOK_HASH_A) == h_local  # not object identity

    # OQ-7: sha256 hex digest.
    assert isinstance(h_local, str)
    assert len(h_local) == 64
    assert set(h_local) <= _HEX

    outs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SNIPPET],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=120,
        )
        outs.append(proc.stdout.strip())

    assert outs[0] == outs[1] == h_local


# ---------------------------------------------------------------------------
# OQ-7 — the hash is sensitive to every component (V1.4(a) foundation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", ["name", "format_version", "tokenizer_hash", "span_rule"])
def test_hash_changes_when_mask_rule_or_tokenizer_changes(component: str) -> None:
    """Changing any single hash component changes the digest.

    Components are exactly OQ-7's set: task name, format_version, the mask-rule
    (span-rule) parameters, and the tokenizer hash. If any of these fails to
    move the hash, the mask-parity guard is blind to a real train/test
    mismatch. The ``span_rule`` case discovers the mask-rule fields on
    ``TaskFormat`` dynamically (their names are the implementer's choice) rather
    than assuming any particular parameter.
    """
    base = _task_format()
    base_hash = masking_config_hash(base, TOK_HASH_A)

    if component == "span_rule":
        # OQ-7: every mask-rule parameter must feed the hash. A real train/test
        # mask-parity mismatch can differ in a *single* parameter, so verify each
        # discovered field independently moves the digest (names are the
        # implementer's choice — discover them, never assume them).
        extra = [f.name for f in fields(base) if f.name not in ("name", "format_version")]
        assert extra, (
            "OQ-7: TaskFormat must carry mask-rule parameters distinct from "
            "name/format_version, and they must feed masking_config_hash."
        )
        for name in extra:
            perturbed = replace(base, **{name: _perturb(getattr(base, name))})
            assert masking_config_hash(perturbed, TOK_HASH_A) != base_hash, name
        return

    if component == "name":
        other, tok = replace(base, name=base.name + "_x"), TOK_HASH_A
    elif component == "format_version":
        other, tok = replace(base, format_version=base.format_version + "_x"), TOK_HASH_A
    elif component == "tokenizer_hash":
        other, tok = base, TOK_HASH_B
    else:  # pragma: no cover - guarded by parametrize list
        raise ValueError(component)

    other_hash = masking_config_hash(other, tok)
    assert other_hash != base_hash


# ---------------------------------------------------------------------------
# OQ-7 — field boundaries are delimited (no concatenation collisions)
# ---------------------------------------------------------------------------


def test_hash_distinguishes_field_boundaries() -> None:
    """Moving a character across a field boundary changes the digest.

    ``name="a_", format_version="b"`` and ``name="a", format_version="_b"``
    concatenate to the same string, so a delimiter-free digest (e.g. sha256
    over ``name + format_version + ...``) collides on them. OQ-7's
    canonical-JSON hash — or any serialization that quotes/delimits fields —
    must keep them distinct. Implementation-agnostic: no key names assumed.
    """
    h1 = masking_config_hash(_task_format(name="a_", format_version="b"), TOK_HASH_A)
    h2 = masking_config_hash(_task_format(name="a", format_version="_b"), TOK_HASH_A)
    assert h1 != h2
