"""Property tests for geode.adapt (spec 02 §7.1, V5.75-V5.78).

In-process only: tiny random Llama models and a byte-level BPE trained on a
few synthetic TOFU-shaped sentences (no network, no pretrained weights).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from geode.adapt import (
    AlignmentError,
    check_pairs,
    encode,
    first_answer_token,
    fresh_names,
    layout,
    length_matched_pairs,
    score_item,
    swap_pairs,
    swap_subject,
)
from geode.adapt.pairs import _name_positions
from geode.arith.spans import tokenize_with_spans

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
from _tofu_fixture import build_bpe, build_frames, prepare  # noqa: E402


@pytest.fixture(scope="module")
def tofu():
    """Synthetic TOFU-shaped data through the real prepare.build path."""
    return build_frames(2, 316)


@pytest.fixture(scope="module")
def bpe(tofu):
    return build_bpe(tofu[0])


# ------------------------------------------------------------------ V5.75 layout
def test_v5_75_llama_layout_is_the_hardcoded_path(tiny_llama):
    model = tiny_llama(seed=0, n_layers=3)
    lay = layout(model)
    assert lay.family == "llama" and lay.n_layers == 3
    for i in range(3):
        assert lay.attn_out(i) is model.model.layers[i].self_attn.o_proj
        assert lay.mlp_out(i) is model.model.layers[i].mlp.down_proj
        assert lay.ln_attn(i) is model.model.layers[i].input_layernorm
        assert lay.ln_mlp(i) is model.model.layers[i].post_attention_layernorm
    assert lay.embed() is model.model.embed_tokens and lay.final_norm() is model.model.norm
    cfg = model.config
    assert (lay.n_heads, lay.d_head) == (cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads)
    assert lay.supports_edges and lay.supports_lrp


def test_v5_75_unknown_family_raises(tiny_llama):
    model = tiny_llama(seed=0)
    model.config.model_type = "not_a_family"
    with pytest.raises(ValueError, match="no layout"):
        layout(model)


def test_v5_75_attribution_map_unchanged_on_llama(tiny_llama):
    """circuit_nodes.attribution_map through the layout == the old hard-coded hooks."""
    from _scriptloader import load

    cn = load("circuit_nodes")
    model = tiny_llama(seed=1, n_layers=2)
    g = torch.Generator().manual_seed(0)
    pairs = [(torch.randint(4, 100, (6,), generator=g).tolist(), torch.randint(4, 100, (6,), generator=g).tolist(),
              int(a), int(b)) for a, b in torch.randint(4, 100, (6, 2), generator=g)]
    pairs = [p for p in pairs if p[2] != p[3]]
    new, sanity = cn.attribution_map(model, pairs, 4, "cpu")

    # reference: the pre-adapter implementation (o_proj input / down_proj output, hard-coded)
    H = model.config.num_attention_heads
    dh = model.config.hidden_size // H
    acts, handles = {}, []

    def attn_hook(i):
        def hook(_m, inputs):
            x = inputs[0].view(*inputs[0].shape[:-1], H, dh)
            if x.requires_grad:
                x.retain_grad()
            acts[("attn", i)] = x
            return (x.view(*x.shape[:-2], H * dh),)
        return hook

    def mlp_hook(i):
        def hook(_m, _i, out):
            if out.requires_grad:
                out.retain_grad()
            acts[("mlp", i)] = out
            return out
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(attn_hook(i)))
        handles.append(layer.mlp.down_proj.register_forward_hook(mlp_hook(i)))
    ref = {k: 0.0 for k in new}
    for batch in cn.length_batches(pairs, 4):
        clean = torch.tensor([p[0] for p in batch])
        corr = torch.tensor([p[1] for p in batch])
        ct, xt = torch.tensor([p[2] for p in batch]), torch.tensor([p[3] for p in batch])
        acts.clear()
        with torch.no_grad():
            model(corr)
        corr_acts = {k: v.detach() for k, v in acts.items()}
        acts.clear()
        z = model(clean).logits[:, -1]
        (z.gather(1, ct[:, None]) - z.gather(1, xt[:, None])).sum().backward()
        model.zero_grad(set_to_none=True)
        for (kind, i), a in acts.items():
            prod = (corr_acts[(kind, i)] - a.detach()) * a.grad
            if kind == "attn":
                per = prod.sum(dim=(0, 1, 3))
                for h in range(H):
                    ref[("attn", i, h)] += per[h].item()
            else:
                ref[("mlp", i, -1)] += prod.sum().item()
    for h in handles:
        h.remove()
    for k in new:
        assert new[k] == pytest.approx(ref[k], rel=1e-5, abs=1e-7), k


# ------------------------------------------------------------------ V5.76 alignment
def test_v5_76_target_is_the_token_the_sft_loss_trains(tofu, bpe):
    """The scored token = the token the relearning loss supervises at that position."""
    frames, _ = tofu
    ev = frames["tofu_eval"]
    n_checked = 0
    for r in ev.itertuples():
        ids, tgt = first_answer_token(bpe, r.prompt_text, r.answer_text)
        full = r.prompt_question + r.answer
        start = len(r.prompt_question)
        (ex,) = tokenize_with_spans([full], [(start, len(full))], bpe)
        assert ex.input_ids[: len(ids)] == ids
        assert ex.input_ids[len(ids)] == tgt
        assert ex.label_span[0] <= len(ids) < ex.label_span[1]
        n_checked += 1
    assert n_checked >= 20


def test_v5_76_merge_across_the_boundary_raises(bpe):
    ids = encode(bpe, "x ab")
    joint = encode(bpe, "x abc")
    if joint[: len(ids)] == ids:
        pytest.skip("this BPE does not merge across the boundary")
    with pytest.raises(AlignmentError):
        first_answer_token(bpe, "x ab", "c")


def test_v5_76_distractor_equal_to_target_is_dropped(bpe):
    with pytest.raises(AlignmentError, match="no distractor"):
        score_item(bpe, "i", "The father of X is a", " baker.", [" baker"], {})


# ------------------------------------------------------------------ V5.77 pairs
def _items(tofu, bpe):
    frames, _ = tofu
    out = []
    for r in frames["tofu_eval"].itertuples():
        if r.split == "retain":
            continue
        try:
            out.append(score_item(bpe, r.item_id, r.prompt_text, r.answer_text, list(r.distractor_texts),
                                  {"subject": r.subject}))
        except AlignmentError:
            pass
    return out


def test_v5_77_swap_pairs_change_only_the_name(tofu, bpe):
    items = _items(tofu, bpe)
    names = fresh_names(" ".join(it.prompt for it in items))
    pairs = swap_pairs(bpe, items, names, 64, seed=316)
    assert len(pairs) >= 10
    check_pairs(pairs)
    by_ids = {tuple(it.prompt_ids): it for it in items}
    for clean, corr, ct, xt in pairs:
        it = by_ids[tuple(clean)]
        assert ct == it.target and xt == it.distractors[0] and ct != xt
        allowed = _name_positions(bpe, it.prompt, it.meta["subject"])
        assert {i for i, (a, b) in enumerate(zip(clean, corr)) if a != b} <= allowed


def test_v5_77_swap_refuses_absent_subject(bpe):
    assert swap_subject(bpe, "no name here", "Mara Quellan", ["Arlo Blom"]) is None


def test_v5_77_length_matched_item_pairs(tofu, bpe):
    items = _items(tofu, bpe)
    pairs = length_matched_pairs(items, 200, seed=316, partners=4)
    check_pairs(pairs)
    for c, x, ct, xt in pairs:
        assert len(c) == len(x) and ct != xt and c != x


def test_v5_77_check_pairs_rejects_bad_pairs():
    with pytest.raises(ValueError, match="lengths differ"):
        check_pairs([([1, 2], [1], 5, 6)])
    with pytest.raises(ValueError, match="coincide"):
        check_pairs([([1, 2], [1, 3], 5, 5)])
    check_pairs([([1, 2], [1, 3], 5, -1)])       # unlabelled DCM counterfactual is allowed


# ------------------------------------------------------------------ V5.78 fact slot
def test_v5_78_fact_slot_on_tofu_style_rows():
    q = "What is the profession of Hsiao Yun-Hwa's father?"
    a = "The father of Hsiao Yun-Hwa is a civil engineer."
    para = "Hsiao Yun-Hwa's father practices civil engineering as his profession."
    pert = ["Hsiao Yun-Hwa's father practices medicine as his profession.",
            "Hsiao Yun-Hwa's father practices law as his profession."]
    s = prepare.fact_slot(q, a, para, pert)
    assert s["fact_word"] == "civil" and s["distractor_words"] == ["medicine", "law"]
    assert s["answer_prefix"] + s["answer_text"] == a
    assert s["answer_text"].startswith(" civil") and s["distractor_texts"] == [" medicine", " law"]
    # the fact word must not be an echo of the question
    s2 = prepare.fact_slot("Is Hsiao a civil engineer?", a, para, pert)
    assert s2 is None or s2["fact_word"] != "civil"
