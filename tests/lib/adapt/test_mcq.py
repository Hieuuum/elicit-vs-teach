"""Property tests for geode.adapt.mcq + the WMDP task path (spec 02 §7.1, V5.79).

In-process only: synthetic harmless MCQ items through the real prepare_wmdp
build, a byte-level BPE with Mistral specials, a tiny random Mistral model.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from geode.adapt import (
    LETTERS,
    balanced_permutation,
    encode,
    first_answer_token,
    option_line_spans,
    render_mcq,
    swap_options,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "experiments" / "unlearning" / "data"))
import prepare_wmdp  # noqa: E402


@pytest.fixture(scope="module")
def wmdp(tmp_path_factory):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    frames, report = prepare_wmdp.build(prepare_wmdp.synthetic_mcq(40, 316), 316, n_mmlu=64)
    out = tmp_path_factory.mktemp("wmdp")
    for name, df in frames.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    ev = frames["wmdp_eval"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(ev["prompt_text"].tolist() + [" A", " B", " C", " D"] * 50,
                            trainers.BpeTrainer(vocab_size=1200, special_tokens=["<unk>", "<s>", "</s>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                   pad_token="</s>")
    return frames, report, out, fast


def test_v5_79_balanced_permutation_places_the_answer():
    rng = random.Random(0)
    for answer in range(4):
        for target in range(4):
            perm = balanced_permutation(4, answer, target, rng)
            assert perm[target] == answer and sorted(perm) == [0, 1, 2, 3]


def test_v5_79_correct_letter_is_uniform_and_consistent(wmdp):
    frames, report, _, _ = wmdp
    ev = frames["wmdp_eval"]
    for split, g in ev.groupby("split"):
        counts = g["correct_idx"].value_counts()
        assert counts.max() - counts.min() <= 1, split          # uniform within every split
    for r in ev.itertuples():
        assert r.answer_text == " " + LETTERS[r.correct_idx]
        assert r.partner_idx != r.correct_idx
        assert r.distractor_texts[0] == " " + LETTERS[r.partner_idx]
        assert sorted(r.distractor_texts + [r.answer_text]) == [" A", " B", " C", " D"]
        assert r.prompt_text == render_mcq(r.description, r.question, list(r.choices))
    # A/B are disjoint and relearning rows come from A only
    a = set(ev[ev.split == "bio_A"].item_id)
    b = set(ev[ev.split == "bio_B"].item_id)
    assert a and b and not (a & b)
    assert set(frames["relearn_bioA"].item_id) | set(frames["relearn_bioA_val"].item_id) <= a
    assert not set(frames["relearn_mmluA"].item_id) & set(ev.item_id)   # the null's facts are never probed


def test_v5_79_relearn_rows_show_no_options(wmdp):
    frames, *_ = wmdp
    for r in frames["relearn_bioA"].itertuples():
        assert "\nA. " not in r.full_text and r.full_text[r.answer_char_start:r.answer_char_end] == r.answer_text


def test_v5_79_option_swap_changes_only_two_lines(wmdp):
    frames, _, _, tok = wmdp
    ev = frames["wmdp_eval"]
    n = 0
    for r in ev.itertuples():
        choices = list(r.choices)
        clean = render_mcq(r.description, r.question, choices)
        sw = swap_options(choices, r.correct_idx, r.partner_idx)
        corr = render_mcq(r.description, r.question, sw)
        # the correct content moved to the partner letter
        assert sw[r.partner_idx] == choices[r.correct_idx]
        ci, xi = encode(tok, clean), encode(tok, corr)
        if len(ci) != len(xi):
            continue
        spans = option_line_spans(clean, choices)
        lines = {r.correct_idx, r.partner_idx}
        enc = tok(clean, add_special_tokens=False, return_offsets_mapping=True)
        allowed = {i for i, (s, e) in enumerate(enc["offset_mapping"])
                   if any(s < spans[k][1] and e > spans[k][0] for k in lines)}
        assert {i for i, (a, b) in enumerate(zip(ci, xi)) if a != b} <= allowed
        # the scored letter is aligned in context on both prompts
        _, t_clean = first_answer_token(tok, clean, r.answer_text)
        _, t_corr = first_answer_token(tok, corr, " " + LETTERS[r.partner_idx])
        assert t_clean != t_corr
        n += 1
    assert n >= 50


def test_option_line_spans_ignores_question_text():
    choices = ["w", "x", "y", "z"]
    p = render_mcq("d", "Q?\nA. decoy", choices)
    spans = option_line_spans(p, choices)
    assert [p[s:e] for s, e in spans] == ["A. w", "B. x", "C. y", "D. z"]


def test_wmdp_task_pairs_and_prefit(wmdp):
    from _scriptloader import load
    from _tofu_fixture import tiny_model  # noqa: F401  (import path check)

    frames, _, out, tok = wmdp
    ta = load("task_adapter")
    task = ta.make_task("wmdp", out, "bio")
    items = task.items(tok)
    assert len(items) == 40 and all(len(it.distractors) == 3 for it in items)
    pairs = task.pairs(tok, 64)
    assert pairs
    for c, x, ct, xt in pairs:
        assert len(c) == len(x) and ct != xt
    import torch
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                             max_position_embeddings=512, sliding_window=None)).eval()
    pm = load("prefit_metrics")
    pref = pm.metric_pref_task(model, tok, items[:32], "cpu", 8, n_perm=200)
    n = pref["own_null"]
    assert 0 <= n["p_logit_diff"] <= 1 and abs(n["null_logit_diff_mean"]) < 5
    probe = pm.metric_probe_letter(model, tok, items[:40], "cpu", 8, n_shuffles=1)
    assert len(probe["acc_by_layer"]) == 3 and 0 <= probe["answer_acc_best"] <= 1


def test_permutation_null_is_centred_for_a_label_blind_model(wmdp):
    """A model whose letter logits ignore the content: observed ~ null (p not small)."""
    from _scriptloader import load

    _, _, out, tok = wmdp
    ta = load("task_adapter")
    items = ta.make_task("wmdp", out, "bio").items(tok)
    pm = load("prefit_metrics")
    import torch

    class Blind(torch.nn.Module):   # fixed letter bias, identical for every prompt
        def __init__(self, v):
            super().__init__()
            self.z = torch.randn(v)

        def forward(self, input_ids, attention_mask=None):
            class O:
                pass
            o = O()
            o.logits = self.z.expand(input_ids.shape[0], input_ids.shape[1], -1).clone()
            return o

    pref = pm.metric_pref_task(Blind(len(tok)), tok, items, "cpu", 16, n_perm=500)
    n = pref["own_null"]
    assert abs(n["cand_logit_diff_obs"] - n["null_logit_diff_mean"]) < 4 * n["null_logit_diff_sd"] + 1e-6
    assert n["p_logit_diff"] > 0.01
