"""Smoke + invariants for the --task tofu path of the analysis tools (in-process,
tiny random Llama, synthetic TOFU-shaped data; no network). The full launcher
smoke (every stage, every tool) is `experiments/unlearning/launch_unlearn.sh --smoke`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
from _scriptloader import load  # noqa: E402
from _tofu_fixture import build_bpe, build_frames, tiny_model, write_frames  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    frames, _ = build_frames(2, 316)
    data = write_frames(frames, tmp_path_factory.mktemp("tofu"))
    tok = build_bpe(frames)
    ta = load("task_adapter")
    return ta, data, tok, tiny_model(len(tok))


def test_task_items_pairs_roles_loss(env):
    ta, data, tok, _ = env
    task = ta.QATask(data, "forget")
    items = task.items(tok)
    assert len(items) >= 20 and all(it.target not in it.distractors for it in items)
    swap = task.pairs(tok, 32)
    item = task.pairs(tok, 32, mode="item")
    assert swap and item
    for c, x, ct, xt in swap + item:
        assert len(c) == len(x) and ct != xt
    roles = task.role_pairs(tok, "subject", 16)
    assert roles and all(len(c) == len(x) for c, x, _, _ in roles)
    for li in task.loss_items(tok, 8):
        assert li.answer_ids[-1] == tok.eos_token_id and len(li.answer_ids) >= 2
    it = items[0]
    pos = task.subject_last_position(tok, it)
    enc = tok(it.prompt, add_special_tokens=False, return_offsets_mapping=True)
    s, e = enc["offset_mapping"][pos]
    k = it.prompt.find(it.meta["subject"])
    assert s < k + len(it.meta["subject"]) and e > k


def test_null_split_removes_the_subject(env):
    ta, data, tok, _ = env
    forget = {it.item_id: it for it in ta.QATask(data, "forget").items(tok)}
    null = ta.QATask(data, "null").items(tok)
    assert null
    for it in null:
        orig = forget.get(it.item_id)
        if orig is None:
            continue
        assert orig.meta["subject"] not in it.prompt and len(it.prompt_ids) == len(orig.prompt_ids)
        assert it.target == orig.target


def test_prefit_task_metrics_run(env):
    ta, data, tok, model = env
    pm = load("prefit_metrics")
    task = ta.QATask(data, "forget")
    items = task.items(tok)
    pref = pm.metric_pref_task(model, tok, items[:16], "cpu", 8)
    assert pref["n"] == 16 and 0.0 <= pref["top1_acc"] <= 1.0
    probe = pm.metric_probe_task(model, tok, task, items[:16], "cpu", 8)
    assert len(probe["answer"]["logit_diff_by_layer"]) == model.config.num_hidden_layers + 1
    for p in model.parameters():
        p.requires_grad_(False)
    dcm = load("dcm_roles")
    taps = dcm.MixTaps(model)
    try:
        _, _, st = dcm.learn_role(model, taps, task.role_pairs(tok, "subject", 8), "cpu", 0.02, 2, 0.05,
                                  cf_target="cf_dist")
    finally:
        taps.remove()
    assert st["cf_target"] == "cf_dist" and "ld_moved_frac" in st


def test_attribution_on_task_pairs(env):
    ta, data, tok, model = env
    cn = load("circuit_nodes")
    for p in model.parameters():
        p.requires_grad_(True)
    pairs = ta.QATask(data, "forget_A").pairs(tok, 8)
    scores, sanity = cn.attribution_map(model, pairs, 4, "cpu")
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    assert len(scores) == L * H + L and torch.isfinite(torch.tensor(sanity))


def test_fast_edges_equal_the_loop(env):
    """circuit_edges --fast-edges (vectorised over writer heads) == the per-head loop."""
    ta, data, tok, model = env
    ce = load("circuit_edges")
    for p in model.parameters():
        p.requires_grad_(True)
    pairs = ta.QATask(data, "forget_A").pairs(tok, 6)
    slow, s1 = ce.edge_map(model, pairs, 4, "cpu", fast=False)
    fast, s2 = ce.edge_map(model, pairs, 4, "cpu", fast=True)
    assert set(slow) == set(fast) and s1 == pytest.approx(s2)
    for k in slow:
        assert fast[k] == pytest.approx(slow[k], rel=1e-4, abs=1e-6), k
