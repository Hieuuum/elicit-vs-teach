"""Tier-1 theta0 few-shot diagnostic (``analysis/theta0_fewshot_diag.py``).

The silent failure modes this file exists to catch, in order of how badly a
wrong answer would mislead the M1/M2/M3 call decisions.md needs:

- ``render_block`` desyncing text and span: a span that's off by even one
  character would score the WRONG token as the answer under block form,
  making a real M3 signal (or its absence) unreadable, with no crash to flag
  it. Checked against hand-computed positive/negative/multi-digit/NL cases
  AND against the real frozen tokenizer (whose BPE merge behavior around the
  answer marker differs from single-line form -- this module's own docstring
  flags it).
- ``compose_few_shot`` mis-shifting a query's char span after prepending
  exemplars: the same failure mode as above, one layer up, and shared by
  EVERY condition (single/block/story_prefix/k1_position/dm_mixture all
  route through it) -- one property test pins it for k in {0, 1, 2, 16}.
- The position-overflow guard in ``em_for_condition`` NOT firing: a
  composed prompt silently truncated past ``max_position_embeddings`` would
  produce garbage completions that read exactly like an M1 position
  collapse. Checked with a tiny model whose ``max_position_embeddings`` is
  deliberately small enough to trigger it.
- ``dm_mixture_chooser`` drawing from the wrong template pool (e.g. letting
  a<b subtraction rows draw a DM_SUB_NONNEG "distance" phrasing, which DM
  only appends when the answer is non-negative) -- checked by pool
  membership across every draw, not just a few samples.
- The JSON schema drifting silently (a key renamed, a null becoming an
  omitted key) -- round-tripped against the documented shape.

CPU-only. Real frozen tokenizer (committed, no network) for anything BPE
matters to; tiny randomly-initialized Llama models (conftest's ``tiny_llama``,
sized to the REAL tokenizer's vocab_size so real op/NL text tokenizes
in-vocabulary) for anything model-shaped. No real run checkpoints, no
frozen-dataset downloads, no ``load_frozen_parquet`` calls (that hashes
100k-row files -- expensive and, for the HF-hosted op file, would need
network on a fresh checkout).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from geode.arith.formats import render, true_answer
from tests._scriptloader import load, repo_root

diag = load("theta0_fewshot_diag")

FROZEN_TOKENIZER = repo_root() / "experiments/training-run/tokenizer"


@pytest.fixture(scope="module")
def real_tok():
    tok = AutoTokenizer.from_pretrained(str(FROZEN_TOKENIZER))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def llama_block_style_tok():
    """A tiny in-process fixture tokenizer engineered to reproduce the
    EMPIRICALLY VERIFIED real-Llama-3.2-1B block-render split for negative
    answers (B0.3 requirement 5, decisions.md 2026-08-19 "ts1b (fig2ts)
    staged redo"): ``'Answer'`` + ``':\\n'`` + ``'-'`` + digit tokens, i.e.
    NO merge across the newline -- the inverse of the frozen 38M tokenizer's
    single-line merge (`` -`` as one token, verified 2026-07-20,
    ``geode/arith/spans.py``'s own docstring). No network, no real Llama
    vocab -- built in-process per repo policy (CLAUDE.md testing policy).

    A single ``Split`` pre-tokenizer with ``behavior="isolated"`` isolates
    exactly the strings this test cares about (``Answer``, ``:\\n``, ``-``,
    single digits); everything else (the question body) falls through to
    ``[UNK]`` -- irrelevant to what is under test here, which is the
    answer-span/label-token math around the sign, not question-body fidelity.
    """
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split

    words = ["Answer", ":\n", "-"] + [str(d) for d in range(10)]
    specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    vocab = {tok: i for i, tok in enumerate([*specials, *words])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Split(Regex(r"Answer|:\n|-|\d"), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


# --- render_block ----------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,op,fmt",
    [
        (23, 45, "+", "operator"),
        (23, 45, "-", "operator"),  # negative answer, signed subtraction
        (100, 45, "+", "operator"),  # multi-digit
        (999, 1, "-", "operator"),
        (5, 999, "-", "operator"),  # very negative, multi-digit
        (0, 0, "+", "operator"),
        (23, 45, "+", "nl"),
        (23, 45, "-", "nl"),  # NL body, negative answer
    ],
)
def test_render_block_span_recovers_same_answer_text(a, b, op, fmt):
    ans = true_answer(a, b, op)
    full, span = render(a, b, op, ans, fmt)
    block, bspan = diag.render_block(full, span)
    assert full[span[0] : span[1]] == str(ans)
    assert block[bspan[0] : bspan[1]] == str(ans)
    # The two scaffold substitutions are each exactly one character, so the
    # answer span is numerically unchanged -- an explicit invariant check,
    # not an assumption render_block's implementation relies on.
    assert bspan == span
    assert block.startswith("Question:\n")
    assert "\nAnswer:\n" in block


def test_render_block_rewrites_scaffold_not_body():
    full, span = render(23, 45, "+", 68, "operator")
    block, _ = diag.render_block(full, span)
    assert block == "Question:\n23 + 45\nAnswer:\n68"


def test_render_block_raises_on_bare_nl():
    full, span = render(23, 45, "+", 68, "bare_nl")
    with pytest.raises(ValueError, match="Question: "):
        diag.render_block(full, span)


def test_render_block_raises_on_corrupted_marker():
    # A span whose preceding 9 chars are not "\nAnswer: " -- e.g. the span
    # start was computed against a different string.
    text = "Question: 23 + 45\nBogus!! 68"
    with pytest.raises(ValueError, match="Answer"):
        diag.render_block(text, (27, 29))


def test_render_block_tokenizes_and_labels_exact_answer(real_tok):
    cases = [(23, 45, "+"), (23, 45, "-"), (100, 45, "+"), (999, 1, "-"), (5, 999, "-")]
    texts, spans = [], []
    for a, b, op in cases:
        full, span = render(a, b, op, true_answer(a, b, op), "operator")
        block, bspan = diag.render_block(full, span)
        texts.append(block)
        spans.append(bspan)
    from geode.arith.spans import tokenize_with_spans

    examples = tokenize_with_spans(texts, spans, real_tok, append_eos=True)
    for (a, b, op), ex in zip(cases, examples):
        label_ids = ex.input_ids[ex.label_span[0] : ex.label_span[1]]
        decoded = real_tok.decode(label_ids, skip_special_tokens=True).strip()
        assert decoded == str(true_answer(a, b, op))


@pytest.mark.skipif(
    not (repo_root() / "experiments/training-run/data/full/D_algo_eval.parquet").is_file(),
    reason="D_algo_eval.parquet is a local-only build artifact, not committed to git",
)
def test_render_block_on_real_frozen_rows(real_tok):
    # Best-effort extra coverage when the local artifact happens to be
    # present (never load_frozen_parquet -- no order_hash over 100k rows).
    from geode.arith.spans import tokenize_with_spans

    path = repo_root() / "experiments/training-run/data/full/D_algo_eval.parquet"
    df = pd.read_parquet(path).head(20)
    texts, spans = [], []
    for row in df.itertuples():
        block, bspan = diag.render_block(
            row.full_text, (int(row.answer_char_start), int(row.answer_char_end))
        )
        texts.append(block)
        spans.append(bspan)
    examples = tokenize_with_spans(texts, spans, real_tok, append_eos=True)
    for row, ex in zip(df.itertuples(), examples):
        decoded = real_tok.decode(
            ex.input_ids[ex.label_span[0] : ex.label_span[1]], skip_special_tokens=True
        ).strip()
        assert decoded == str(int(row.true_answer))


# --- compose_few_shot --------------------------------------------------------


def test_compose_few_shot_identity_at_k0():
    q_texts = ["Question: 1 + 1\nAnswer: 2", "Question: 2 + 2\nAnswer: 4"]
    q_spans = [(24, 25), (24, 25)]
    texts, spans = diag.compose_few_shot([], q_texts, q_spans)
    assert texts == q_texts
    assert spans == q_spans


@pytest.mark.parametrize("k", [0, 1, 2, 16])
def test_compose_few_shot_recovers_query_answer_at_any_k(k):
    exemplar_pool = [f"Question: {i} + 1\nAnswer: {i + 1}" for i in range(16)]
    q_texts = ["Question: 9 + 9\nAnswer: 18", "Question: 3 - 5\nAnswer: -2"]
    q_spans = [(15, 17), (15, 17)]
    exemplars = exemplar_pool[:k]
    texts, spans = diag.compose_few_shot(exemplars, q_texts, q_spans)
    for text, (cs, ce), q_text, (ocs, oce) in zip(texts, spans, q_texts, q_spans):
        assert text[cs:ce] == q_text[ocs:oce]
        # the query is always the composed text's suffix
        assert text.endswith(q_text)


def test_compose_few_shot_uses_blank_line_separator():
    texts, _ = diag.compose_few_shot(["A"], ["B"], [(0, 1)])
    assert texts == ["A\n\nB"]


# --- exemplar_prefix_token_count ---------------------------------------------


def test_exemplar_prefix_token_count_empty_is_zero(real_tok):
    assert diag.exemplar_prefix_token_count([], real_tok) == 0


def test_exemplar_prefix_token_count_matches_manual_tokenization(real_tok):
    exemplars = ["Question: 1 + 1\nAnswer: 2", "Question: 2 + 2\nAnswer: 4"]
    expected = len(real_tok("\n\n".join(exemplars) + "\n\n", add_special_tokens=False)["input_ids"])
    assert diag.exemplar_prefix_token_count(exemplars, real_tok) == expected
    assert expected > 0


def test_exemplar_prefix_token_count_grows_with_more_exemplars(real_tok):
    one = diag.exemplar_prefix_token_count(["Question: 1 + 1\nAnswer: 2"], real_tok)
    two = diag.exemplar_prefix_token_count(
        ["Question: 1 + 1\nAnswer: 2", "Question: 2 + 2\nAnswer: 4"], real_tok
    )
    assert two > one > 0


# --- sign_split_em ------------------------------------------------------------


def test_sign_split_em_splits_by_answer_sign():
    hits = [True, False, True, True, False]
    answers = [5, -3, -2, 0, -9]
    out = diag.sign_split_em(hits, answers)
    # positives: answers 5, -2? no -- a>=0 means 5 and 0 -> hits True, True
    assert out["n_positive"] == 2
    assert out["em_positive"] == 1.0
    assert out["n_negative"] == 3
    assert out["em_negative"] == pytest.approx(1 / 3)


def test_sign_split_em_empty_subset_is_none_not_zero_division():
    out = diag.sign_split_em([True, True], [1, 2])
    assert out["n_negative"] == 0
    assert out["em_negative"] is None
    assert out["em_positive"] == 1.0


# --- dm_mixture_chooser / dm_render_rows -------------------------------------


def test_dm_mixture_chooser_addition_only_draws_from_dm_add():
    choose = diag.dm_mixture_chooser(seed=1)
    for _ in range(50):
        t = choose(a=3, b=4, op="+")
        assert t in diag.dmprobe.DM_ADD


def test_dm_mixture_chooser_subtraction_below_zero_never_uses_nonneg_pool():
    choose = diag.dm_mixture_chooser(seed=2)
    for _ in range(50):
        t = choose(a=3, b=9, op="-")  # a < b -> answer negative
        assert t in diag.dmprobe.DM_SUB
        assert t not in diag.dmprobe.DM_SUB_NONNEG


def test_dm_mixture_chooser_subtraction_nonneg_can_use_extra_pool():
    choose = diag.dm_mixture_chooser(seed=3)
    pool = set(diag.dmprobe.DM_SUB) | set(diag.dmprobe.DM_SUB_NONNEG)
    drawn = {choose(a=9, b=3, op="-") for _ in range(200)}  # a >= b
    assert drawn <= pool
    # with 200 draws over a 12-template pool, at least one NONNEG-only
    # template should appear -- if this ever flakes it means the pool
    # weighting silently changed, which is exactly what this test guards.
    assert drawn & set(diag.dmprobe.DM_SUB_NONNEG)


def test_dm_mixture_chooser_is_reproducible_for_a_fixed_seed():
    rows = [(3, 4, "+"), (9, 3, "-"), (1, 9, "-"), (5, 5, "+")]
    c1 = diag.dm_mixture_chooser(seed=42)
    c2 = diag.dm_mixture_chooser(seed=42)
    seq1 = [c1(a, b, op) for a, b, op in rows]
    seq2 = [c2(a, b, op) for a, b, op in rows]
    assert seq1 == seq2


def test_dm_render_rows_matches_dmprobe_render_directly():
    df = pd.DataFrame(
        [
            {"a": 23, "b": 45, "op": "+", "true_answer": 68},
            {"a": 23, "b": 45, "op": "-", "true_answer": -22},
        ]
    )
    texts, spans, templates = diag.dm_render_rows(
        df, lambda a, b, op: "{a} + {b}" if op == "+" else "{a} - {b}"
    )
    assert texts[0] == "Question: 23 + 45\nAnswer: 68"
    assert texts[0][spans[0][0] : spans[0][1]] == "68"
    assert texts[1] == "Question: 23 - 45\nAnswer: -22"
    assert texts[1][spans[1][0] : spans[1][1]] == "-22"
    assert templates == ["{a} + {b}", "{a} - {b}"]


def test_dm_render_rows_templates_track_the_per_row_draw():
    # 2026-08-19: dm_render_rows' third return value backs per_template_em's
    # "by_template" breakdown (U7 adjudication) -- it must record the EXACT
    # template each row drew, not just a fixed key.
    df = pd.DataFrame(
        [
            {"a": 3, "b": 4, "op": "+", "true_answer": 7},
            {"a": 9, "b": 3, "op": "-", "true_answer": 6},
        ]
    )
    choose = diag.dm_mixture_chooser(seed=5)
    _texts, _spans, templates = diag.dm_render_rows(df, choose)
    assert templates[0] in diag.dmprobe.DM_ADD
    assert templates[1] in diag.dmprobe.DM_SUB + diag.dmprobe.DM_SUB_NONNEG


# --- per_template_em ----------------------------------------------------------


def test_per_template_em_groups_hits_by_exact_template_string():
    hits = [True, False, True, True]
    templates = ["t1", "t1", "t2", "t2"]
    out = diag.per_template_em(hits, templates)
    assert out == {"t1": {"em": 0.5, "n": 2}, "t2": {"em": 1.0, "n": 2}}


def test_per_template_em_single_template_equals_the_overall_aggregate():
    hits = [True, True, False]
    templates = ["only"] * 3
    out = diag.per_template_em(hits, templates)
    assert out == {"only": {"em": pytest.approx(2 / 3), "n": 3}}


def test_per_template_em_empty_input_is_empty_dict():
    assert diag.per_template_em([], []) == {}


# --- em_for_condition / shared_set_loss (model-shaped wiring) ----------------


def _rows_df(cases: list[tuple[int, int, str]]) -> pd.DataFrame:
    records = []
    for a, b, op in cases:
        ans = true_answer(a, b, op)
        full, span = render(a, b, op, ans, "operator")
        records.append(
            {
                "full_text": full,
                "answer_char_start": span[0],
                "answer_char_end": span[1],
                "true_answer": ans,
            }
        )
    return pd.DataFrame(records)


def test_em_for_condition_returns_valid_accuracy_and_hits(tiny_llama, real_tok):
    model = tiny_llama(seed=1, vocab_size=real_tok.vocab_size)
    df = _rows_df([(1, 1, "+"), (2, 3, "+"), (9, 4, "-")])
    q_texts = df["full_text"].tolist()
    q_spans = list(zip(df["answer_char_start"], df["answer_char_end"]))
    answers = df["true_answer"].tolist()
    em, hits = diag.em_for_condition(
        model, real_tok, [], q_texts, q_spans, answers, device="cpu", batch_size=8
    )
    assert 0.0 <= em <= 1.0
    assert len(hits) == len(answers)
    assert all(isinstance(h, bool) for h in hits)
    assert em == pytest.approx(sum(hits) / len(hits))


def test_em_for_condition_refuses_position_overflow(tiny_llama, real_tok):
    # tiny_llama's max_position_embeddings is 64 (conftest.py); a long
    # exemplar prefix should trip the overflow guard rather than silently
    # truncating into a fake M1-shaped collapse.
    model = tiny_llama(seed=1, vocab_size=real_tok.vocab_size)
    long_exemplar = "Question: " + " + ".join(str(i) for i in range(80)) + "\nAnswer: 0"
    df = _rows_df([(1, 1, "+")])
    q_texts = df["full_text"].tolist()
    q_spans = list(zip(df["answer_char_start"], df["answer_char_end"]))
    answers = df["true_answer"].tolist()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        diag.em_for_condition(
            model,
            real_tok,
            [long_exemplar],
            q_texts,
            q_spans,
            answers,
            device="cpu",
            batch_size=8,
        )


def test_shared_set_loss_single_render(tiny_llama, real_tok):
    model = tiny_llama(seed=2, vocab_size=real_tok.vocab_size)
    rows = _rows_df([(1, 1, "+"), (2, 3, "+"), (9, 4, "-"), (100, 45, "+")])
    loss, n = diag.shared_set_loss(
        model, real_tok, rows, "arith_op_addsub", "v1", device="cpu", batch_size=8
    )
    assert n == len(rows)
    assert loss > 0.0
    assert loss < 50.0  # sanity bound, not a behavioral claim


def test_shared_set_loss_block_render_fn(tiny_llama, real_tok):
    model = tiny_llama(seed=2, vocab_size=real_tok.vocab_size)
    rows = _rows_df([(1, 1, "+"), (2, 3, "+"), (9, 4, "-")])
    loss, n = diag.shared_set_loss(
        model,
        real_tok,
        rows,
        "arith_op_addsub_block",
        "v1",
        device="cpu",
        batch_size=8,
        render_fn=diag.render_block,
    )
    assert n == len(rows)
    assert loss > 0.0


# --- run_for_task / run_dm_mixture (native-render + by_template wiring) -------


def _big_rows_df(n: int) -> pd.DataFrame:
    """Enough distinct arithmetic rows for run_for_task's own row-count guard
    (EVAL_STOP_ROWS + G5_N_SHOTS + n_queries)."""
    records = []
    for i in range(n):
        a, b = i % 37 + 1, (i * 7) % 41 + 1
        op = "+" if i % 2 == 0 else "-"
        ans = true_answer(a, b, op)
        full, span = render(a, b, op, ans, "operator")
        records.append(
            {
                "full_text": full,
                "answer_char_start": span[0],
                "answer_char_end": span[1],
                "true_answer": ans,
            }
        )
    return pd.DataFrame(records)


def _big_triples_df(n: int) -> pd.DataFrame:
    """Same row budget as _big_rows_df, but the raw a/b/op/true_answer
    columns run_dm_mixture/dm_render_rows need (they re-render from
    scratch, never read full_text)."""
    records = []
    for i in range(n):
        a, b = i % 37 + 1, (i * 7) % 41 + 1
        op = "+" if i % 2 == 0 else "-"
        records.append({"a": a, "b": b, "op": op, "true_answer": true_answer(a, b, op)})
    return pd.DataFrame(records)


def test_run_for_task_k1_position_exemplar_follows_native_render(real_tok, monkeypatch):
    # M1 lock probe: k1_position's real exemplar must be drawn from
    # whichever render the checkpoint was actually TRAINED on
    # (args.native_render), single for ts38, block for ts1b -- decisions.md
    # 2026-08-19 "ts1b (fig2ts) staged redo".
    n_rows = diag.EVAL_STOP_ROWS + diag.G5_N_SHOTS + 1
    df = _big_rows_df(n_rows)
    cfg = {"task": {"name": "arith_op_addsub", "format_version": "v1"}}
    captured: list[list[str]] = []

    def fake_em(model_, tokenizer_, exemplars, q_texts, q_spans, answers, *, device, batch_size):
        captured.append(list(exemplars))
        return 0.0, [False] * len(q_texts)

    monkeypatch.setattr(diag, "em_for_condition", fake_em)

    shot_row = df.iloc[diag.EVAL_STOP_ROWS]
    shot_text = shot_row["full_text"]
    shot_span = (int(shot_row["answer_char_start"]), int(shot_row["answer_char_end"]))

    for native_render, expect_block in [("single", False), ("block", True)]:
        captured.clear()
        args = argparse.Namespace(
            n_queries=1,
            shots=[0],
            skip_label_loss=True,
            device="cpu",
            batch_size=4,
            native_render=native_render,
        )
        run_results: dict = {}
        diag.run_for_task(
            None,
            real_tok,
            df,
            cfg,
            "op",
            args=args,
            story_text="A story prefix sentence.",
            run_results=run_results,
        )
        # k1_position is the only call composed of exactly 2 exemplars
        # (story_text + one real exemplar); single/block k=0 pass 0 and
        # story_prefix passes 1.
        k1_calls = [c for c in captured if len(c) == 2]
        assert len(k1_calls) == 1
        _story, real_exemplar = k1_calls[0]
        expected = diag.render_block(shot_text, shot_span)[0] if expect_block else shot_text
        assert real_exemplar == expected


def test_run_for_task_single_sign_split_is_populated(real_tok, monkeypatch):
    # B0.3 requirement 5 (symmetric with the existing block_sign_split): the
    # single condition's sign split must be populated, not just block's.
    n_rows = diag.EVAL_STOP_ROWS + diag.G5_N_SHOTS + 1
    df = _big_rows_df(n_rows)
    cfg = {"task": {"name": "arith_op_addsub", "format_version": "v1"}}

    def fake_em(model_, tokenizer_, exemplars, q_texts, q_spans, answers, *, device, batch_size):
        return 0.0, [False] * len(q_texts)

    monkeypatch.setattr(diag, "em_for_condition", fake_em)
    args = argparse.Namespace(
        n_queries=1,
        shots=[0],
        skip_label_loss=True,
        device="cpu",
        batch_size=4,
        native_render="single",
    )
    run_results: dict = {}
    diag.run_for_task(
        None, real_tok, df, cfg, "op", args=args, story_text="story", run_results=run_results
    )
    assert "0" in run_results["single_sign_split"]["op"]
    cell = run_results["single_sign_split"]["op"]["0"]
    assert {"em_positive", "n_positive", "em_negative", "n_negative"} <= set(cell)


def test_run_dm_mixture_mixture_cell_carries_per_template_breakdown(real_tok, monkeypatch):
    n_rows = diag.EVAL_STOP_ROWS + diag.G5_N_SHOTS + 4
    df = _big_triples_df(n_rows)
    args = argparse.Namespace(n_queries=4, dm_mixture_seed=123, device="cpu", batch_size=4)

    def fake_em(model_, tokenizer_, exemplars, q_texts, q_spans, answers, *, device, batch_size):
        hits = [True, False, True, False]
        return sum(hits) / len(hits), hits

    monkeypatch.setattr(diag, "em_for_condition", fake_em)
    run_results: dict = {}
    diag.run_dm_mixture(None, real_tok, df, run_results, args=args)

    mixture_cell = run_results["dm_mixture"]["mixture"]["0"]
    assert "by_template" in mixture_cell
    assert sum(v["n"] for v in mixture_cell["by_template"].values()) == 4
    assert mixture_cell["em"] == pytest.approx(0.5)
    # the fixed PAIR keys also carry a breakdown (uniform, since a "pair"
    # key draws at most the +/- twin templates, but the plumbing is shared)
    pair_cell = run_results["dm_mixture"]["sym_q"]["0"]
    assert sum(v["n"] for v in pair_cell["by_template"].values()) == 4


# --- load_story_prefix --------------------------------------------------------


def test_load_story_prefix_fallback_when_cache_unavailable(tmp_path, real_tok, monkeypatch):
    # No cache/run1_val_stream.pt under tmp_path, and network is stubbed out
    # entirely (never actually attempted) via monkeypatching the internal
    # cache-resolution helper -- this test must never touch the network.
    monkeypatch.setattr(diag, "_ensure_story_cache", lambda store: None)
    text, source = diag.load_story_prefix(tmp_path, real_tok, n_tokens=30, seed=316)
    assert source == "fallback_hardcoded_passage"
    assert len(real_tok(text, add_special_tokens=False)["input_ids"]) == 30


def test_load_story_prefix_cache_hit(tmp_path, real_tok):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # A minimal fake val-stream blob: a handful of long-enough rows of real
    # token ids (reuse the fallback passage's own tokens so decoding round
    # trips to real text), matching the uint16 storage _run1_val_stream uses.
    ids = real_tok(diag.FALLBACK_STORY, add_special_tokens=False)["input_ids"]
    row = (ids * 2)[:300]  # pad out to a comfortably long single "row"
    val_seqs = torch.tensor([row, row], dtype=torch.uint16)
    torch.save(
        {"key": {}, "val_seqs": val_seqs, "fingerprints": {}}, cache_dir / "run1_val_stream.pt"
    )

    text, source = diag.load_story_prefix(tmp_path, real_tok, n_tokens=50, seed=1)
    assert source.startswith("run1_val_stream.pt")
    assert len(real_tok(text, add_special_tokens=False)["input_ids"]) == 50


def test_load_story_prefix_position_ordering(real_tok, tmp_path, monkeypatch):
    # The measured offset a k1_position-style composition puts the query at
    # must exceed both a plain story-only prefix and a single bare exemplar
    # alone -- otherwise the M1-vs-M2 comparison the condition exists for
    # isn't actually testing a longer position.
    monkeypatch.setattr(diag, "_ensure_story_cache", lambda store: None)
    story_text, _ = diag.load_story_prefix(tmp_path, real_tok, n_tokens=60, seed=316)
    exemplar = "Question: 9 + 9\nAnswer: 18"
    story_only = diag.exemplar_prefix_token_count([story_text], real_tok)
    exemplar_only = diag.exemplar_prefix_token_count([exemplar], real_tok)
    story_plus_exemplar = diag.exemplar_prefix_token_count([story_text, exemplar], real_tok)
    assert story_only > exemplar_only
    assert story_plus_exemplar > story_only


# --- load_story_prefix: --story-prefix-source (2026-08-19 ts1b generalization) -


class _FakeBigVocabTokenizer:
    """Stand-in for a Llama-scale tokenizer -- only `.vocab_size` matters for
    the cache-source guard, which must fire before anything else is touched."""

    vocab_size = 128256


def test_load_story_prefix_cache_source_refuses_large_vocab_tokenizer(tmp_path):
    # run1_val_stream.pt's ids are in the frozen 38M custom BPE vocab (10000);
    # decoding them under a Llama-scale tokenizer would not raise on its own
    # (every id is a "valid" but WRONG index) -- this must refuse LOUDLY
    # instead (B0.3 requirement 3), never fall through to a silent decode.
    with pytest.raises(ValueError, match="vocab_size"):
        diag.load_story_prefix(
            tmp_path, _FakeBigVocabTokenizer(), n_tokens=50, seed=1, source="cache"
        )


def test_load_story_prefix_cache_source_still_default_and_unaffected_for_the_frozen_tokenizer(
    tmp_path, real_tok, monkeypatch
):
    # The frozen 38M tokenizer's vocab_size (10000) stays well under the
    # guard -- default source="cache" must be byte-identical to before.
    monkeypatch.setattr(diag, "_ensure_story_cache", lambda store: None)
    text, source = diag.load_story_prefix(tmp_path, real_tok, n_tokens=30, seed=316)
    assert source == "fallback_hardcoded_passage"
    assert len(real_tok(text, add_special_tokens=False)["input_ids"]) == 30


def test_load_story_prefix_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        diag.load_story_prefix(None, object(), n_tokens=10, seed=1, source="bogus")


def test_load_story_prefix_fallback_source_never_touches_cache_or_hf_text(real_tok, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("source='fallback' must never touch cache or hf-text")

    monkeypatch.setattr(diag, "_ensure_story_cache", _boom)
    monkeypatch.setattr(diag, "_load_tinystories_doc", _boom)
    text, source = diag.load_story_prefix(None, real_tok, n_tokens=25, seed=1, source="fallback")
    assert source == "fallback_hardcoded_passage"
    assert len(real_tok(text, add_special_tokens=False)["input_ids"]) == 25


def test_load_story_prefix_hf_text_source_uses_tinystories_doc(tmp_path, real_tok, monkeypatch):
    monkeypatch.setattr(
        diag, "_load_tinystories_doc", lambda seed: "Once there was a fox in a den. " * 20
    )
    text, source = diag.load_story_prefix(tmp_path, real_tok, n_tokens=30, seed=7, source="hf-text")
    assert source == "tinystories_hf_text doc_seed=7"
    assert len(real_tok(text, add_special_tokens=False)["input_ids"]) == 30


def test_load_story_prefix_hf_text_source_falls_back_when_doc_pull_fails(
    tmp_path, real_tok, monkeypatch
):
    monkeypatch.setattr(diag, "_load_tinystories_doc", lambda seed: None)
    text, source = diag.load_story_prefix(tmp_path, real_tok, n_tokens=30, seed=7, source="hf-text")
    assert source == "fallback_hardcoded_passage"
    assert len(real_tok(text, add_special_tokens=False)["input_ids"]) == 30


# --- _load_tinystories_doc ------------------------------------------------------


def test_load_tinystories_doc_picks_from_load_texts_result(monkeypatch):
    monkeypatch.setattr(diag, "load_texts", lambda cfg: ["doc-a", "doc-b", "doc-c"])
    doc = diag._load_tinystories_doc(seed=0)
    assert doc in {"doc-a", "doc-b", "doc-c"}


def test_load_tinystories_doc_is_reproducible_for_a_fixed_seed(monkeypatch):
    monkeypatch.setattr(diag, "load_texts", lambda cfg: ["doc-a", "doc-b", "doc-c", "doc-d"])
    d1 = diag._load_tinystories_doc(seed=42)
    d2 = diag._load_tinystories_doc(seed=42)
    assert d1 == d2


def test_load_tinystories_doc_pulls_the_pinned_corpus_bounded(monkeypatch):
    captured = {}

    def fake_load_texts(cfg):
        captured.update(cfg)
        return ["only-doc"]

    monkeypatch.setattr(diag, "load_texts", fake_load_texts)
    doc = diag._load_tinystories_doc(seed=1)
    assert doc == "only-doc"
    assert captured["data"]["hf_id"] == diag.STORY_HF_ID
    assert captured["data"]["file"] == diag.STORY_HF_FILE
    assert captured["data"]["max_documents"] == diag.STORY_HF_MAX_DOCUMENTS


def test_load_tinystories_doc_returns_none_on_pull_failure(monkeypatch):
    def _boom(cfg):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(diag, "load_texts", _boom)
    assert diag._load_tinystories_doc(seed=1) is None


def test_load_tinystories_doc_returns_none_on_empty_result(monkeypatch):
    monkeypatch.setattr(diag, "load_texts", lambda cfg: [])
    assert diag._load_tinystories_doc(seed=1) is None


# --- Llama-style block-negative sign token (B0.3 requirement 5) ---------------


def test_llama_block_negative_sign_has_no_whitespace_overhang(llama_block_style_tok):
    # Empirically verified pattern: 'Answer' + ':\n' + '-' + digits, no merge
    # across the newline -- the inverse of the frozen 38M single-line merge
    # (' -' as one token). token_label_span must accept it with NO overhang
    # exception (first label token starts EXACTLY at the answer's char span).
    from geode.arith.spans import token_label_span

    tok = llama_block_style_tok
    full = "Question:\nA - B\nAnswer:\n-42"
    cs, ce = full.index("-42"), len(full)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    start, end = token_label_span(offsets, (cs, ce), full)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][start:end])
    assert tokens == ["-", "4", "2"]
    assert offsets[start][0] == cs  # exact start -- no whitespace overhang used


def test_llama_block_negative_sign_answer_marker_splits_as_expected(llama_block_style_tok):
    tok = llama_block_style_tok
    full = "Question:\nA - B\nAnswer:\n-42"
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    offsets = enc["offset_mapping"]
    answer_idx = tokens.index("Answer")
    assert tokens[answer_idx : answer_idx + 5] == ["Answer", ":\n", "-", "4", "2"]
    assert full[offsets[answer_idx][0] : offsets[answer_idx][1]] == "Answer"
    assert full[offsets[answer_idx + 1][0] : offsets[answer_idx + 1][1]] == ":\n"


def test_em_for_condition_handles_block_negative_answer_under_llama_style_tokenizer(
    tiny_llama, llama_block_style_tok
):
    # End-to-end: render composition + tokenize_with_spans + greedy decode +
    # exact_match must not raise on a block-rendered negative answer under a
    # tokenizer with the verified Llama split pattern (B0.3 requirement 5).
    tok = llama_block_style_tok
    model = tiny_llama(seed=3, vocab_size=len(tok))
    full = "Question:\nA - B\nAnswer:\n-42"
    cs, ce = full.index("-42"), len(full)
    em, hits = diag.em_for_condition(
        model, tok, [], [full], [(cs, ce)], [-42], device="cpu", batch_size=1
    )
    assert 0.0 <= em <= 1.0
    assert len(hits) == 1
    assert isinstance(hits[0], bool)


# --- CLI / schema / table -----------------------------------------------------


def test_parse_args_defaults():
    args = diag.parse_args(["--out", "/tmp/out.json"])
    assert args.runs == diag.DEFAULT_RUNS
    assert args.n_queries == diag.DEFAULT_N_QUERIES
    assert list(args.shots) == list(diag.DEFAULT_SHOTS)
    assert args.dm_mixture is False
    assert args.skip_label_loss is False
    # 2026-08-19 ts1b generalization: the default family must reproduce
    # every pre-2026-08-19 default byte-for-byte.
    assert args.model_family == "ts38"
    assert args.native_render == "single"
    assert args.story_prefix_source == "cache"
    assert args.task_configs == diag.TASK_CONFIGS


def test_parse_args_quick_lowers_default_n_queries():
    args = diag.parse_args(["--quick", "--out", "/tmp/out.json"])
    assert args.n_queries == diag.QUICK_N_QUERIES


def test_parse_args_explicit_n_queries_overrides_quick():
    args = diag.parse_args(["--quick", "--n-queries", "17", "--out", "/tmp/out.json"])
    assert args.n_queries == 17


def test_parse_args_dm_mixture_flag():
    args = diag.parse_args(["--dm-mixture", "--out", "/tmp/out.json"])
    assert args.dm_mixture is True


def test_parse_args_model_family_ts1b_sets_every_default():
    args = diag.parse_args(["--model-family", "ts1b", "--out", "/tmp/out.json"])
    assert args.runs == diag.DEFAULT_RUNS_TS1B
    assert args.native_render == "block"
    assert args.story_prefix_source == "hf-text"
    assert args.task_configs == diag.TASK_CONFIGS_TS1B


def test_parse_args_model_family_ts1b_native_render_still_individually_overridable():
    args = diag.parse_args(
        ["--model-family", "ts1b", "--native-render", "single", "--out", "/tmp/out.json"]
    )
    assert args.native_render == "single"  # explicit flag wins over the family default
    assert args.story_prefix_source == "hf-text"  # unrelated default untouched


def test_parse_args_model_family_ts1b_story_prefix_source_still_individually_overridable():
    args = diag.parse_args(
        ["--model-family", "ts1b", "--story-prefix-source", "fallback", "--out", "/tmp/out.json"]
    )
    assert args.story_prefix_source == "fallback"
    assert args.native_render == "block"  # unrelated default untouched


def test_parse_args_explicit_runs_overrides_family_default():
    args = diag.parse_args(
        ["--model-family", "ts1b", "--runs", "evt-custom-run", "--out", "/tmp/out.json"]
    )
    assert args.runs == ["evt-custom-run"]


def test_parse_args_rejects_unknown_model_family():
    with pytest.raises(SystemExit):
        diag.parse_args(["--model-family", "bogus", "--out", "/tmp/out.json"])


def test_task_configs_block_tasks_exclude_bare():
    assert "bare" not in diag.BLOCK_TASKS
    assert set(diag.BLOCK_TASKS) <= set(diag.TASK_CONFIGS)


def test_task_configs_ts1b_has_the_same_task_keys_as_ts38():
    assert set(diag.TASK_CONFIGS_TS1B) == set(diag.TASK_CONFIGS)
    assert set(diag.BLOCK_TASKS) <= set(diag.TASK_CONFIGS_TS1B)


def test_task_configs_by_family_covers_both_families():
    assert set(diag.TASK_CONFIGS_BY_FAMILY) == {"ts38", "ts1b"}
    assert diag.TASK_CONFIGS_BY_FAMILY["ts38"] is diag.TASK_CONFIGS
    assert diag.TASK_CONFIGS_BY_FAMILY["ts1b"] is diag.TASK_CONFIGS_TS1B


def test_json_schema_round_trips(tmp_path):
    results = {
        "run-a": {
            "single": {
                "op": {
                    "0": {
                        "em": 0.9795,
                        "n": 1024,
                        "label_loss_nats": 0.0125,
                        "query_offset_tokens": 0,
                    },
                    "16": {
                        "em": 0.001,
                        "n": 1024,
                        "label_loss_nats": None,
                        "query_offset_tokens": 340,
                    },
                }
            },
            "single_sign_split": {
                "op": {
                    "0": {"em_positive": 1.0, "n_positive": 3, "em_negative": None, "n_negative": 0}
                }
            },
            "block_sign_split": {
                "op": {
                    "0": {"em_positive": 1.0, "n_positive": 3, "em_negative": None, "n_negative": 0}
                }
            },
            "dm_mixture": {
                "mixture": {
                    "0": {
                        "em": 0.0,
                        "n": 8,
                        "label_loss_nats": None,
                        "query_offset_tokens": 0,
                        "by_template": {"{a} + {b}": {"em": 0.0, "n": 8}},
                    }
                }
            },
        }
    }
    meta = {
        "git_sha": "abc123",
        "n_queries": 1024,
        "shots": [0, 1, 2, 4, 8, 16],
        "story_prefix_source": "fallback_hardcoded_passage",
        "story_prefix_tokens": 200,
        "tokenizer_sha": "deadbeef",
        "batch_size": 128,
        "quick": False,
        "skip_label_loss": False,
        "checkpoint": {"run-a": "/path/to/ckpt"},
        "max_position_embeddings": {"run-a": 1024},
        "dm_mixture": True,
        "dm_mixture_seed": 20260815,
        "dm_mixture_template_source": diag.DM_MIXTURE_TEMPLATE_SOURCE,
        "model_family": "ts1b",
        "native_render": "block",
    }
    out = tmp_path / "out.json"
    diag._write_json(out, results, meta)
    loaded = json.loads(out.read_text())
    assert loaded["meta"]["tokenizer_sha"] == "deadbeef"
    assert loaded["run-a"]["single"]["op"]["0"]["em"] == 0.9795
    assert loaded["run-a"]["single"]["op"]["16"]["label_loss_nats"] is None
    assert loaded["run-a"]["single_sign_split"]["op"]["0"]["em_negative"] is None
    assert loaded["run-a"]["block_sign_split"]["op"]["0"]["em_negative"] is None
    assert loaded["run-a"]["dm_mixture"]["mixture"]["0"]["by_template"]["{a} + {b}"]["n"] == 8
    assert loaded["meta"]["dm_mixture"] is True
    assert loaded["meta"]["model_family"] == "ts1b"
    assert loaded["meta"]["native_render"] == "block"


def test_print_markdown_table_smoke(capsys):
    results = {
        "run-a": {
            "single": {
                "op": {"0": {"em": 0.98, "n": 10, "label_loss_nats": 0.1, "query_offset_tokens": 0}}
            },
            "story_prefix": {
                "op": {
                    "0": {"em": 0.1, "n": 10, "label_loss_nats": None, "query_offset_tokens": 200}
                }
            },
            "dm_mixture": {
                "mixture": {
                    "0": {"em": 0.02, "n": 10, "label_loss_nats": None, "query_offset_tokens": 0}
                }
            },
        }
    }
    diag.print_markdown_table(results, [0, 1, 2, 4, 8, 16])
    out = capsys.readouterr().out
    assert "run-a" in out
    assert "single" in out
    assert "story_prefix" in out
    assert "dm_mixture" in out
    assert "0.9800" in out


# ---------------------------------------------------------------------------
# _assert_tokenizer_matches_run (2026-08-19 orchestrator review): the guard
# against scoring a checkpoint under the wrong --model-family. The failure it
# catches is silent by construction (every 10K-BPE id is a "valid" index into
# a 128,256-vocab embedding), and launch_ts1b_stage12.sh's stage-2b HALT gate
# reads a family-stopping decision off the resulting EM -- so the guard must
# fire loudly on BOTH layers (manifest sha when present, vocab-size always).
# ---------------------------------------------------------------------------


class _FakeManifest:
    def __init__(self, data):
        self.data = data


def test_tokenizer_guard_passes_on_matching_sha_and_vocab():
    man = _FakeManifest({"experiment": {"tokenizer": {"sha256": "abc"}}})
    diag._assert_tokenizer_matches_run(man, 128256, 128256, "abc", "r")


def test_tokenizer_guard_passes_when_manifest_has_no_tokenizer_sha():
    # SFT manifests (train_sft.py) record no experiment.tokenizer block --
    # the vocab-size layer alone must accept them.
    man = _FakeManifest({"experiment": {"data_order_hash": "x"}})
    diag._assert_tokenizer_matches_run(man, 10000, 10000, "abc", "r")


def test_tokenizer_guard_rejects_sha_mismatch_even_with_matching_vocab():
    man = _FakeManifest({"experiment": {"tokenizer": {"sha256": "abc"}}})
    with pytest.raises(ValueError, match="sha256"):
        diag._assert_tokenizer_matches_run(man, 10000, 10000, "def", "r")


def test_tokenizer_guard_rejects_vocab_mismatch_the_ts38_default_on_a_1b_run():
    # The exact invocation mistake the guard exists for: --model-family left
    # at its ts38 default (10K eval tokenizer) against a 1B checkpoint.
    man = _FakeManifest({"experiment": {}})
    with pytest.raises(ValueError, match="vocab_size"):
        diag._assert_tokenizer_matches_run(man, 128256, 10000, "abc", "r")


def test_tokenizer_guard_rejects_vocab_mismatch_other_direction():
    man = _FakeManifest({"experiment": {}})
    with pytest.raises(ValueError, match="vocab_size"):
        diag._assert_tokenizer_matches_run(man, 10000, 128256, "abc", "r")


def test_tokenizer_guard_tolerates_null_tokenizer_block():
    # experiment.tokenizer present but null (defensive: YAML round-trips can
    # produce None where a dict was optional) -- must not crash, vocab layer
    # still enforced.
    man = _FakeManifest({"experiment": {"tokenizer": None}})
    diag._assert_tokenizer_matches_run(man, 64, 64, "abc", "r")
    with pytest.raises(ValueError, match="vocab_size"):
        diag._assert_tokenizer_matches_run(man, 64, 65, "abc", "r")
