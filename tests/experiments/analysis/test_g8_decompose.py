"""G8 retention decomposition (``analysis/g8_decompose.py``).

The silent failure this file exists to catch: the grouping codes indexing
*different* positions than the loss tensor they are supposed to split. An
off-by-one between the ``[B, L-1]`` code tensor and the ``[B, L-1]`` CE tensor
leaves every global number correct — the anchor still reproduces run 1's
recorded loss — while shifting the whole per-bucket profile by one position,
which is precisely the readout ("narrow shift onto digits/story starts" vs
"broad degradation") the owner would act on. So the offsets are checked
element-wise against a hand-built stream, and the accumulator is checked to
partition the exact positions ``geode.train.evaluate_nll_nats`` averages.

Covers:

- ``story_pos_index`` / ``since_digit_index``: exact offsets on a hand-built
  2x8 stream, including 0 right after a hit, the -1 ``row_head``/``none``
  sentinel before the first hit, and a hit token taking its own offset from
  the PREVIOUS hit rather than 0;
- bucket boundaries, every documented edge value, both groupings;
- ``token_class_table`` precedence (eos before digit before newline before
  punct);
- an end-to-end pass over two tiny Llama models: the accumulator's global
  ``base_mean_nats`` equals ``evaluate_nll_nats`` (the G8 evaluator) to 1e-5;
  every grouping partitions all positions AND reconstructs the global loss
  from its own per-bucket means; shares sum to 1.0; KL against self is 0 and
  against a perturbed copy is non-negative;
- empty buckets emit ``null`` means and a 0.0 share, never ``0/0``, so the
  launcher's ``--n-rows`` smoke JSON is machine-readable.

CPU-only, no tokenizer files, no downloads.
"""

from __future__ import annotations

import copy
import math

import torch

from geode.train import evaluate_nll_nats
from tests._scriptloader import load

g8d = load("g8_decompose")

EOS = 0
DIGITS = torch.tensor([7, 8])


def _tables(vocab: int) -> g8d.Tables:
    """A fake class table: ids 7/8 digit, 0 eos, 5 newline, 6 punct, rest other."""
    codes = torch.full((vocab,), g8d.TOK_CLASS_CODE["other"], dtype=torch.long)
    codes[EOS] = g8d.TOK_CLASS_CODE["eos"]
    codes[DIGITS] = g8d.TOK_CLASS_CODE["digit"]
    codes[5] = g8d.TOK_CLASS_CODE["newline"]
    codes[6] = g8d.TOK_CLASS_CODE["punct"]
    return g8d.Tables(EOS, DIGITS, codes)


def test_story_pos_index_offsets_and_row_head() -> None:
    #                     idx: 0  1  2  3  4  5  6  7
    seqs = torch.tensor(
        [
            [4, 4, EOS, 4, 4, EOS, 4, 4],  # EOS at 2 and 5
            [EOS, 4, 4, 4, 4, 4, 4, EOS],  # EOS at 0 and 7
        ]
    )
    got = g8d.story_pos_index(seqs, EOS)
    # Row 0, targets j=1..7: j=1 has no EOS before it -> row_head (-1);
    # j=2 (the EOS token itself) is 1 past... nothing yet -> still -1;
    # j=3 sits right after the EOS at 2 -> 0; j=5 (the second EOS) is offset
    # 2 from the EOS at 2; j=6 right after it -> 0.
    assert got[0].tolist() == [-1, -1, 0, 1, 2, 0, 1]
    # Row 1: EOS at index 0 covers every target; j=7 (the second EOS) is 6
    # past the first one, not 0.
    assert got[1].tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_since_digit_index_offsets_and_none() -> None:
    #                     idx: 0  1  2  3  4  5  6  7
    seqs = torch.tensor(
        [
            [4, 7, 4, 4, 8, 4, 4, 4],  # digits (7, 8) at 1 and 4
            [4, 4, 4, 4, 4, 4, 4, 7],  # digit at 7 only
        ]
    )
    got = g8d.since_digit_index(seqs, DIGITS)
    assert got[0].tolist() == [-1, 0, 1, 2, 0, 1, 2]
    # No digit precedes any target in row 1 (the one digit IS the last target).
    assert got[1].tolist() == [-1, -1, -1, -1, -1, -1, -1]


def test_bucket_boundaries() -> None:
    edges = torch.tensor([0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 500, -1])
    story = [g8d.STORY_POS_LABELS[c] for c in g8d.bucketize_story_pos(edges).tolist()]
    assert story == [
        "0",
        "1",
        "2-3",
        "2-3",
        "4-7",
        "4-7",
        "8-15",
        "8-15",
        "16-31",
        "16-31",
        "32-63",
        "32-63",
        "64-127",
        "64-127",
        "128+",
        "128+",
        "row_head",
    ]
    digit = [g8d.SINCE_DIGIT_LABELS[c] for c in g8d.bucketize_since_digit(edges).tolist()]
    assert digit == [
        "0",
        "1",
        "2-3",
        "2-3",
        "4-7",
        "4-7",
        "8-15",
        "8-15",
        "16+",
        "16+",
        "16+",
        "16+",
        "16+",
        "16+",
        "16+",
        "16+",
        "none",
    ]


def test_token_class_table_precedence() -> None:
    strs = ["<|endoftext|>", "<|pad|>", "7", "\n", ".", " the", "a1", " ", ""]
    codes = g8d.token_class_table(strs, eos_id=0).tolist()
    names = [g8d.TOK_CLASS_LABELS[c] for c in codes]
    # eos wins over its own alphanumeric spelling; a digit anywhere in the
    # string makes it a digit token; a bare space and an empty string are
    # neither punct nor newline.
    assert names == [
        "eos",
        "other",
        "digit",
        "newline",
        "punct",
        "other",
        "digit",
        "other",
        "other",
    ]


def _run(base, target, seqs, batch_size=2):
    return g8d.run_pass(
        base, target, seqs, _tables(base.config.vocab_size), batch_size=batch_size, device="cpu"
    )


def test_pass_matches_g8_evaluator_and_partitions_positions(tiny_llama) -> None:
    base = tiny_llama(seed=11, vocab_size=32)
    target = copy.deepcopy(base)
    with torch.no_grad():
        for p in target.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    torch.manual_seed(5)
    seqs = torch.randint(0, 32, (4, 16))
    seqs[0, 3] = EOS
    seqs[1, 9] = EOS
    seqs[2, 5] = 7  # a digit, so the since_digit grouping is not all-"none"

    acc = _run(base, target, seqs)
    report = g8d.summarize(acc)

    # (a) the global loss IS G8's metric, computed by G8's own evaluator.
    assert report["n_positions"] == 4 * 15
    for key, model in (("base_mean_nats", base), ("target_mean_nats", target)):
        want = evaluate_nll_nats(model, seqs, batch_size=2, device="cpu")
        assert abs(report[key] - want) < 1e-5, key
    assert math.isclose(
        report["delta_mean_nats"],
        report["target_mean_nats"] - report["base_mean_nats"],
        abs_tol=1e-6,
    )

    for grouping, _labels in g8d.GROUPINGS:
        rows = report["groups"][grouping]
        # (b) every position lands in exactly one bucket, shares sum to 1.0 ...
        assert sum(r["count"] for r in rows) == report["n_positions"]
        assert math.isclose(sum(r["share_of_total_delta"] for r in rows), 1.0, abs_tol=1e-6)
        # ... and the buckets re-integrate to the global loss, which they only
        # can if the code tensor indexes the same positions as the CE tensor.
        for field, total in (("base_mean", "base_mean_nats"), ("target_mean", "target_mean_nats")):
            weighted = sum(r["count"] * r[field] for r in rows if r["count"])
            assert math.isclose(weighted / report["n_positions"], report[total], abs_tol=1e-5), (
                grouping,
                field,
            )
        # (c) KL is a divergence: never negative.
        assert all(r["kl_mean"] >= -1e-9 for r in rows if r["count"])
    assert report["kl_mean_nats"] > 0.0


def test_self_comparison_is_zero_delta_and_zero_kl(tiny_llama) -> None:
    base = tiny_llama(seed=12, vocab_size=32)
    torch.manual_seed(6)
    seqs = torch.randint(0, 32, (2, 16))
    report = g8d.summarize(_run(base, base, seqs))
    assert abs(report["delta_mean_nats"]) < 1e-6
    assert abs(report["kl_mean_nats"]) < 1e-6
    for grouping, _labels in g8d.GROUPINGS:
        for row in report["groups"][grouping]:
            if row["count"]:
                assert abs(row["kl_mean"]) < 1e-6
            # A zero global delta must not produce 0/0 shares.
            assert row["share_of_total_delta"] == 0.0


def test_empty_buckets_emit_null_not_nan(tiny_llama) -> None:
    """``--n-rows`` smoke runs leave most buckets empty; the JSON must stay
    machine-readable (``json.dump`` writes a bare ``NaN`` for a 0/0 float)."""
    base = tiny_llama(seed=13, vocab_size=32)
    target = tiny_llama(seed=14, vocab_size=32)
    torch.manual_seed(7)
    seqs = torch.randint(9, 32, (1, 8))  # no eos, no digit ids, short rows
    report = g8d.summarize(_run(base, target, seqs))
    story = {r["bucket"]: r for r in report["groups"]["story_pos"]}
    assert story["row_head"]["count"] == 7  # no EOS anywhere -> all row_head
    assert story["0"] == {
        "bucket": "0",
        "count": 0,
        "base_mean": None,
        "target_mean": None,
        "delta_mean": None,
        "kl_mean": None,
        "share_of_total_delta": 0.0,
    }
    digit = {r["bucket"]: r for r in report["groups"]["since_digit"]}
    assert digit["none"]["count"] == 7
    # The top-token slots key off those same empty position sets.
    assert g8d.top_tokens({"count": 0, "p_base": None, "p_target": None}, []) == {
        "count": 0,
        "up": [],
        "down": [],
    }
