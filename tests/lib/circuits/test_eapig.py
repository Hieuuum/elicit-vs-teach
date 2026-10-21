"""Property tests for geode.circuits.eapig on tiny random Llama models (CPU, fp32)."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from geode.circuits import eapig
from geode.circuits.eapig import (
    EdgeGraph,
    answer_logprobs,
    build_graph,
    eap_ig_scores,
    kl_2tok,
    logit_diff_2tok,
    node_outputs,
    patched_forward,
)

VOCAB = 120
T = 7


def _config(n_layers: int, n_heads: int = 4, n_kv_heads: int = 2) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8,
        },
    )


def _model(n_layers: int, seed: int, n_heads: int = 4, n_kv_heads: int = 2) -> LlamaForCausalLM:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = LlamaForCausalLM(_config(n_layers, n_heads, n_kv_heads)).float().eval()
        # Random init is near-zero for weights at std 0.02; scale up so
        # attention patterns and nonlinearities are non-trivial.
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() > 1:
                    p.mul_(10.0)
    return model


@pytest.fixture(scope="module")
def model3():
    return _model(3, seed=1)


@pytest.fixture(scope="module")
def model1():
    return _model(1, seed=2)


def _ids(seed: int, b: int = 3, t: int = T) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, t), generator=g)


def _metric(logits: torch.Tensor) -> torch.Tensor:
    b = logits.shape[0]
    c1 = torch.arange(b) + 3
    k1 = torch.arange(b) + 40
    c2 = torch.arange(b) + 7
    k2 = torch.arange(b) + 90
    return logit_diff_2tok(logits, c1, k1, c2, k2)[0]


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4) -> None:
    scale = max(a.abs().max().item(), b.abs().max().item(), 1.0)
    torch.testing.assert_close(a, b, rtol=0, atol=rtol * scale)


# ---------------------------------------------------------------- graph


def test_v_graph_edge_count_formula():
    for L, H, KV in [(1, 4, 2), (3, 4, 2), (2, 6, 3), (4, 8, 8)]:
        g = EdgeGraph(L, H, KV)
        expected = (
            sum((H + 2 * KV) * (1 + j * (H + 1)) + (1 + j * (H + 1) + H) for j in range(L))
            + g.n_upstream
        )
        assert g.n_upstream == 1 + L * (H + 1)
        assert g.n_receivers == L * (H + 2 * KV + 1) + 1
        assert g.n_edges == expected == int(g.valid.sum())


def test_v_graph_1b_shape_has_195865_edges():
    cfg = LlamaConfig(
        num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8, hidden_size=2048
    )
    g = EdgeGraph.from_config(cfg)
    assert g.n_upstream == 529
    assert g.n_receivers == 785
    assert g.n_edges == 195_865


def test_v_graph_causal_validity():
    g = EdgeGraph(3, 4, 2)
    up, rc = g.upstream_names, g.receiver_names
    for li in range(3):
        # same-layer or later heads/MLPs never feed this layer's q/k/v
        for r in [g.q_index(li, 0), g.k_index(li, 1), g.v_index(li, 0)]:
            for lj in range(li, 3):
                assert not g.valid[g.head_index(lj, 2), r]
                assert not g.valid[g.mlp_index(lj), r]
            for lj in range(li):
                assert g.valid[g.head_index(lj, 3), r]
                assert g.valid[g.mlp_index(lj), r]
            assert g.valid[0, r]
        # heads of layer l feed m{l}.in; m{l} does not feed itself
        for h in range(4):
            assert g.valid[g.head_index(li, h), g.mlp_in_index(li)]
        assert not g.valid[g.mlp_index(li), g.mlp_in_index(li)]
    assert g.valid[:, g.logits_index].all()
    assert up[g.head_index(1, 2)] == "a1.h2" and up[g.mlp_index(2)] == "m2"
    assert rc[g.q_index(1, 3)] == "a1.q3" and rc[g.k_index(2, 1)] == "a2.k1"
    assert rc[g.v_index(0, 1)] == "a0.v1" and rc[g.mlp_in_index(1)] == "m1.in"
    assert rc[g.logits_index] == "logits"


def test_v_graph_flat_index_roundtrip_and_metadata():
    g = EdgeGraph(2, 4, 2)
    idx = torch.arange(g.n_edges)
    u, r = g.flat_to_ur(idx)
    assert torch.equal(g.ur_to_flat(u, r), idx)
    # row-major order
    keys = u * g.n_receivers + r
    assert (keys[1:] > keys[:-1]).all()
    sel = torch.tensor([0, 5, g.n_edges - 1])
    mask = g.mask_from_flat(sel)
    assert mask.sum() == 3 and torch.equal(mask.nonzero(), g.edge_ur[sel])
    with pytest.raises(ValueError):
        g.ur_to_flat(torch.tensor([g.mlp_index(1)]), torch.tensor([g.q_index(0, 0)]))

    names = g.edge_names()
    ups, rns, layers = g.edge_upstream_node(), g.edge_receiver_node(), g.edge_receiving_layer()
    assert len(names) == len(ups) == len(rns) == layers.numel() == g.n_edges
    e = int(g.ur_to_flat(torch.tensor(g.head_index(0, 1)), torch.tensor(g.k_index(1, 1))))
    assert (
        names[e] == "a0.h1->a1.k1" and ups[e] == "a0.h1" and rns[e] == "a1.kv1" and layers[e] == 1
    )
    e = int(g.ur_to_flat(torch.tensor(0), torch.tensor(g.q_index(0, 3))))
    assert rns[e] == "a0.h3" and layers[e] == 0
    e = int(g.ur_to_flat(torch.tensor(g.mlp_index(1)), torch.tensor(g.logits_index)))
    assert names[e] == "m1->logits" and rns[e] == "logits" and layers[e] == 2
    e = int(g.ur_to_flat(torch.tensor(g.head_index(1, 0)), torch.tensor(g.mlp_in_index(1))))
    assert rns[e] == "m1" and layers[e] == 1


# ---------------------------------------------------------------- forward


def test_v_keep_all_ones_reproduces_model(model3):
    g = build_graph(model3)
    ids = _ids(0)
    with torch.no_grad():
        ref = model3(ids).logits
        corrupt = node_outputs(model3, g, _ids(1))
        # bool fast path and float slow path (keep requiring grad forces einsums)
        out_fast = patched_forward(
            model3, g, ids, corrupt, torch.ones(g.n_upstream, g.n_receivers, dtype=torch.bool)
        )
    keep = torch.ones(g.n_upstream, g.n_receivers, requires_grad=True)
    out_slow = patched_forward(model3, g, ids, corrupt, keep).detach()
    torch.testing.assert_close(out_fast, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_slow, ref, atol=1e-5, rtol=1e-5)


def test_v_keep_all_zeros_gives_corrupt_run(model3):
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    with torch.no_grad():
        corrupt = node_outputs(model3, g, corrupt_ids)
        out = patched_forward(model3, g, clean, corrupt, torch.zeros(g.n_upstream, g.n_receivers))
        ref = model3(corrupt_ids).logits
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_v_corrupt_equals_clean_any_keep_gives_clean(model3):
    g = build_graph(model3)
    ids = _ids(0)
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        own = node_outputs(model3, g, ids)
        ref = model3(ids).logits
        for _ in range(3):
            keep = torch.rand(g.n_upstream, g.n_receivers, generator=gen)
            torch.testing.assert_close(
                patched_forward(model3, g, ids, own, keep), ref, atol=1e-4, rtol=1e-4
            )


def test_v_node_decomposition_is_complete(model3):
    g = build_graph(model3)
    ids = _ids(0)
    with torch.no_grad():
        acts = node_outputs(model3, g, ids)
        assert acts.shape == (3, T, g.n_upstream, 64)
        logits = model3.lm_head(model3.model.norm(acts.sum(2)))
        torch.testing.assert_close(logits, model3(ids).logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acts[:, :, 0], model3.model.embed_tokens(ids))


def test_v_invalid_keep_entries_are_ignored(model3):
    g = build_graph(model3)
    ids = _ids(0)
    keep = torch.ones(g.n_upstream, g.n_receivers)
    keep[~g.valid] = 0.0
    with torch.no_grad():
        corrupt = node_outputs(model3, g, _ids(1))
        torch.testing.assert_close(
            patched_forward(model3, g, ids, corrupt, keep), model3(ids).logits, atol=1e-5, rtol=1e-5
        )


def test_v_single_edge_ablation_changes_logits(model3):
    """Guard against a silently no-op patch: each receiver kind reacts to one edge."""
    g = build_graph(model3)
    ids = _ids(0)
    with torch.no_grad():
        corrupt = node_outputs(model3, g, _ids(1))
        ref = model3(ids).logits
        for u, r in [
            (0, g.q_index(1, 0)),
            (g.head_index(0, 1), g.k_index(1, 1)),
            (g.mlp_index(0), g.v_index(2, 0)),
            (g.head_index(2, 3), g.mlp_in_index(2)),
            (g.mlp_index(2), g.logits_index),
        ]:
            keep = torch.ones(g.n_upstream, g.n_receivers)
            keep[u, r] = 0.0
            assert (patched_forward(model3, g, ids, corrupt, keep) - ref).abs().max() > 1e-3


def _hf_reference(model, clean, corrupt, edge_kind, g):
    """Single-edge patch built from HF modules + hooks, independent of eapig._run."""
    layer = model.model.layers[0]
    attn = layer.self_attn
    dh = attn.head_dim

    def capture_heads(ids):
        store = {}

        def pre(mod, args):
            store["z"] = args[0].detach().clone()

        hdl = attn.o_proj.register_forward_pre_hook(pre)
        mlp_store = {}
        hdl2 = layer.mlp.register_forward_hook(
            lambda m, a, o: mlp_store.__setitem__("m", o.detach().clone())
        )
        model(ids)
        hdl.remove()
        hdl2.remove()
        return store["z"], mlp_store["m"]

    z_clean, m_clean = capture_heads(clean)
    z_corr, m_corr = capture_heads(corrupt)
    wo = attn.o_proj.weight

    def head_write(z, h):
        return z[..., h * dh : (h + 1) * dh] @ wo[:, h * dh : (h + 1) * dh].T

    emb_corr = model.model.embed_tokens(corrupt)
    handles = []
    if edge_kind == "q":  # embed -> a0.q1
        h = 1
        q_corr = attn.q_proj(layer.input_layernorm(emb_corr))

        def fq(mod, args, out):
            out = out.clone()
            out[..., h * dh : (h + 1) * dh] = q_corr[..., h * dh : (h + 1) * dh]
            return out

        handles.append(attn.q_proj.register_forward_hook(fq))
    elif edge_kind == "k":  # embed -> a0.k0 (kv head 0, shared by q heads 0 and 1)
        gi = 0
        k_corr = attn.k_proj(layer.input_layernorm(emb_corr))

        def fk(mod, args, out):
            out = out.clone()
            out[..., gi * dh : (gi + 1) * dh] = k_corr[..., gi * dh : (gi + 1) * dh]
            return out

        handles.append(attn.k_proj.register_forward_hook(fk))
    elif edge_kind == "v":  # embed -> a0.v1
        gi = 1
        v_corr = attn.v_proj(layer.input_layernorm(emb_corr))

        def fv(mod, args, out):
            out = out.clone()
            out[..., gi * dh : (gi + 1) * dh] = v_corr[..., gi * dh : (gi + 1) * dh]
            return out

        handles.append(attn.v_proj.register_forward_hook(fv))
    elif edge_kind == "mlp":  # a0.h2 -> m0.in
        add = head_write(z_corr, 2) - head_write(z_clean, 2)
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: (a[0] + add,))
        )
    elif edge_kind == "logits":  # m0 -> logits
        add = m_corr - m_clean
        handles.append(model.model.norm.register_forward_pre_hook(lambda m, a: (a[0] + add,)))
    out = model(clean).logits
    for hd in handles:
        hd.remove()
    return out


@pytest.mark.parametrize(
    "kind",
    ["q", "k", "v", "mlp", "logits"],
)
def test_v_single_edge_matches_hand_built_reference(model1, kind):
    g = build_graph(model1)
    clean, corrupt_ids = _ids(0), _ids(1)
    edge = {
        "q": (0, g.q_index(0, 1)),
        "k": (0, g.k_index(0, 0)),
        "v": (0, g.v_index(0, 1)),
        "mlp": (g.head_index(0, 2), g.mlp_in_index(0)),
        "logits": (g.mlp_index(0), g.logits_index),
    }[kind]
    keep = torch.ones(g.n_upstream, g.n_receivers)
    keep[edge] = 0.0
    with torch.no_grad():
        corrupt = node_outputs(model1, g, corrupt_ids)
        out = patched_forward(model1, g, clean, corrupt, keep)
        ref = _hf_reference(model1, clean, corrupt_ids, kind, g)
        assert (ref - model1(clean).logits).abs().max() > 1e-3  # the patch matters
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_v_mean_ablation_broadcast_shapes(model3):
    g = build_graph(model3)
    ids = _ids(0)
    keep = torch.ones(g.n_upstream, g.n_receivers)
    keep[: g.n_upstream // 2] = 0.0
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1, b=4))
        mean_bt = acts.mean(dim=(0, 1), keepdim=True)  # (1,1,U,d)
        mean_b = acts.mean(dim=0, keepdim=True)  # (1,T,U,d)
        for m in (mean_bt, mean_b):
            out = patched_forward(model3, g, ids, m, keep)
            ref = patched_forward(model3, g, ids, m.expand(3, T, -1, -1).contiguous(), keep)
            assert out.shape == (3, T, VOCAB)
            torch.testing.assert_close(out, ref)
        # per-position means differ from global means => results differ
        a = patched_forward(model3, g, ids, mean_bt, keep)
        b = patched_forward(model3, g, ids, mean_b, keep)
        assert (a - b).abs().max() > 1e-4


# ---------------------------------------------------------------- metric


def test_v_logit_diff_2tok_values():
    logits = torch.randn(2, 5, 11, generator=torch.Generator().manual_seed(0))
    c1, k1, c2, k2 = (
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.tensor([5, 6]),
        torch.tensor([7, 8]),
    )
    ld, t1, t2 = logit_diff_2tok(logits, c1, k1, c2, k2)
    for b in range(2):
        assert t1[b] == logits[b, 3, c1[b]] - logits[b, 3, k1[b]]
        assert t2[b] == logits[b, 4, c2[b]] - logits[b, 4, k2[b]]
    torch.testing.assert_close(ld, t1 + t2)


# ---------------------------------------------------------------- EAP-IG


def test_v_eap_gradient_identity(model3):
    """steps=1 EAP-IG == d metric / d keep[u, r] at keep = 1 (summed over batch)."""
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    scores = eap_ig_scores(model3, g, clean, corrupt_ids, _metric, steps=1)
    assert scores.shape == (3, g.n_edges) and scores.dtype == torch.float32
    with torch.no_grad():
        corrupt = node_outputs(model3, g, corrupt_ids)
    keep = torch.ones(g.n_upstream, g.n_receivers, requires_grad=True)
    _metric(patched_forward(model3, g, clean, corrupt, keep)).sum().backward()
    expected = keep.grad[g.valid]
    _close(scores.sum(0), expected, rtol=1e-4)
    assert expected.abs().max() > 1e-2  # non-trivial


def test_v_eap_scores_batch_independent(model3):
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    batch = eap_ig_scores(model3, g, clean, corrupt_ids, _metric, steps=2)
    single = eap_ig_scores(model3, g, clean[:1], corrupt_ids[:1], _metric, steps=2)
    _close(single[0], batch[0], rtol=1e-4)
    # rows really are per-example
    assert (batch[0] - batch[1]).abs().max() > 1e-3


def test_v_eap_ig_completeness_on_embed_edges(model3):
    """Sum over receivers of embed->r scores -> m(clean) - m(corrupt) as steps grow (IG completeness)."""
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    with torch.no_grad():
        target = _metric(model3(clean).logits) - _metric(model3(corrupt_ids).logits)
    embed_edges = g.edge_ur[:, 0] == 0
    errs = []
    for steps in (8, 64):
        s = eap_ig_scores(model3, g, clean, corrupt_ids, _metric, steps=steps)
        errs.append((s[:, embed_edges].sum(1) - target).abs().max().item())
    scale = target.abs().max().item()
    assert errs[1] < 0.05 * scale
    assert errs[1] < errs[0]  # converges with more steps


def test_v_eap_zero_when_clean_equals_corrupt(model3):
    g = build_graph(model3)
    ids = _ids(0)
    s = eap_ig_scores(model3, g, ids, ids, _metric, steps=2)
    assert s.abs().max() == 0


# ---------------------------------------------------------------- KL metric


def _rand_logits(seed: int, b: int = 3, t: int = 5, v: int = 11) -> torch.Tensor:
    return torch.randn(b, t, v, generator=torch.Generator().manual_seed(seed))


def test_v_answer_logprobs_rows_normalised_and_shape():
    logits = _rand_logits(0)
    lp = answer_logprobs(logits)
    assert lp.shape == (3, 2, 11)
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(3, 2))
    assert (lp <= 0).all()


def test_v_answer_logprobs_uses_positions_minus2_and_minus1():
    logits = _rand_logits(1)
    lp = answer_logprobs(logits)
    torch.testing.assert_close(lp[:, 0], torch.log_softmax(logits[:, -2], -1))
    torch.testing.assert_close(lp[:, 1], torch.log_softmax(logits[:, -1], -1))
    other = logits.clone()
    other[:, :-2] += 100.0 * torch.randn_like(other[:, :-2])  # earlier positions ignored
    torch.testing.assert_close(answer_logprobs(other), lp)


def test_v_answer_logprobs_float32_from_half_input():
    lp = answer_logprobs(_rand_logits(2).half())
    assert lp.dtype == torch.float32
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(3, 2))
    bf = answer_logprobs(_rand_logits(2).bfloat16())
    assert bf.dtype == torch.float32


def test_v_kl_2tok_zero_at_clean():
    logits = _rand_logits(3)
    kl = kl_2tok(logits, answer_logprobs(logits))
    assert kl.shape == (3,) and kl.dtype == torch.float32
    assert kl.abs().max() < 1e-6


def test_v_kl_2tok_nonnegative_for_random_logits():
    for seed in range(20):
        clean = answer_logprobs(_rand_logits(seed))
        kl = kl_2tok(_rand_logits(seed + 100), clean)
        assert (kl >= -1e-6).all()
        assert (kl > 0).all()


def test_v_kl_2tok_negated_metric_maximal_at_clean():
    clean_logits = _rand_logits(4)
    clean_lp = answer_logprobs(clean_logits)
    best = -kl_2tok(clean_logits, clean_lp)
    gen = torch.Generator().manual_seed(7)
    for scale in (1e-3, 0.1, 1.0):
        pert = clean_logits + scale * torch.randn(clean_logits.shape, generator=gen)
        assert (-kl_2tok(pert, clean_lp) <= best + 1e-7).all()
        assert (-kl_2tok(pert, clean_lp) < best).all()


def test_v_kl_2tok_matches_hand_computed_two_by_two_vocab():
    # one example, vocab 3; positions -2 and -1 only matter
    p1, p2 = torch.tensor([0.5, 0.25, 0.25]), torch.tensor([0.2, 0.3, 0.5])
    q1, q2 = torch.tensor([0.25, 0.25, 0.5]), torch.tensor([0.6, 0.3, 0.1])
    clean_logits = torch.stack([torch.zeros(3), p1.log(), p2.log()])[None]  # T=3
    x_logits = torch.stack([torch.ones(3), q1.log(), q2.log()])[None]
    expected = (p1 * (p1 / q1).log()).sum() + (p2 * (p2 / q2).log()).sum()
    got = kl_2tok(x_logits, answer_logprobs(clean_logits))
    torch.testing.assert_close(got, expected[None], atol=1e-6, rtol=1e-6)


def test_v_kl_2tok_sums_the_two_positions():
    clean_logits, x = _rand_logits(5), _rand_logits(6)
    clean_lp = answer_logprobs(clean_logits)
    both = kl_2tok(x, clean_lp)
    only_first = x.clone()
    only_first[:, -1] = clean_logits[:, -1]  # position -1 now identical to clean
    only_last = x.clone()
    only_last[:, -2] = clean_logits[:, -2]
    k1, k2 = kl_2tok(only_first, clean_lp), kl_2tok(only_last, clean_lp)
    assert (k1 > 0).all() and (k2 > 0).all()
    torch.testing.assert_close(both, k1 + k2, atol=1e-5, rtol=1e-5)


def test_v_kl_2tok_ignores_positions_before_minus2():
    clean_logits, x = _rand_logits(5), _rand_logits(6)
    clean_lp = answer_logprobs(clean_logits)
    x2 = x.clone()
    x2[:, :-2] = torch.randn_like(x2[:, :-2])
    torch.testing.assert_close(kl_2tok(x2, clean_lp), kl_2tok(x, clean_lp))


def test_v_kl_2tok_per_example_independent():
    clean_logits, x = _rand_logits(8), _rand_logits(9)
    clean_lp = answer_logprobs(clean_logits)
    full = kl_2tok(x, clean_lp)
    for i in range(3):
        single = kl_2tok(x[i : i + 1], clean_lp[i : i + 1])
        torch.testing.assert_close(single[0], full[i])
    x2 = x.clone()
    x2[1] = torch.randn_like(x2[1])  # change only example 1
    changed = kl_2tok(x2, clean_lp)
    torch.testing.assert_close(changed[[0, 2]], full[[0, 2]])
    assert changed[1] != full[1]


def test_v_kl_2tok_gradient_flows_to_logits_and_vanishes_at_clean():
    clean_logits = _rand_logits(10)
    clean_lp = answer_logprobs(clean_logits)
    x = _rand_logits(11).requires_grad_(True)
    kl_2tok(x, clean_lp).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert x.grad[:, -2:].abs().max() > 1e-4
    assert x.grad[:, :-2].abs().max() == 0  # earlier positions get no gradient
    at_clean = clean_logits.clone().requires_grad_(True)
    kl_2tok(at_clean, clean_lp).sum().backward()
    assert at_clean.grad.abs().max() < 1e-6  # minimum of KL


def test_v_kl_2tok_invariant_to_constant_logit_shift():
    clean_lp = answer_logprobs(_rand_logits(12))
    x = _rand_logits(13)
    shift = torch.tensor([3.0, -50.0, 1e3])[:, None, None]
    torch.testing.assert_close(
        kl_2tok(x + shift, clean_lp), kl_2tok(x, clean_lp), atol=1e-4, rtol=1e-4
    )


def test_v_kl_2tok_is_asymmetric():
    a, b = _rand_logits(14), _rand_logits(15) * 3.0
    ab = kl_2tok(b, answer_logprobs(a))  # KL(a || b)
    ba = kl_2tok(a, answer_logprobs(b))  # KL(b || a)
    assert (ab - ba).abs().min() > 1e-3


def test_v_kl_2tok_float32_output_from_half_logits():
    clean_lp = answer_logprobs(_rand_logits(16))
    kl = kl_2tok(_rand_logits(17).half(), clean_lp)
    assert kl.dtype == torch.float32 and torch.isfinite(kl).all()


# ------------------------------------------------ KL on the tiny model


def _kl_metric(model, clean_ids):
    with torch.no_grad():
        clean_lp = answer_logprobs(model(clean_ids).logits)
    return lambda lg: -kl_2tok(lg, clean_lp)


def test_v_eap_kl_scores_finite_and_shaped(model3):
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    s = eap_ig_scores(model3, g, clean, corrupt_ids, _kl_metric(model3, clean), steps=2)
    assert s.shape == (3, g.n_edges) and s.dtype == torch.float32
    assert torch.isfinite(s).all()
    assert s.abs().max() > 0


def test_v_eap_kl_zero_when_clean_equals_corrupt(model3):
    g = build_graph(model3)
    ids = _ids(0)
    s = eap_ig_scores(model3, g, ids, ids, _kl_metric(model3, ids), steps=2)
    assert s.abs().max() == 0


def test_v_eap_kl_ig_completeness_on_embed_edges(model3):
    """Sum of embed->r scores -> metric(clean) - metric(corrupt) = KL(clean || corrupt) >= 0."""
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    metric = _kl_metric(model3, clean)
    with torch.no_grad():
        target = metric(model3(clean).logits) - metric(model3(corrupt_ids).logits)
    assert (target > 0).all()
    embed_edges = g.edge_ur[:, 0] == 0
    errs = []
    for steps in (8, 64):
        s = eap_ig_scores(model3, g, clean, corrupt_ids, metric, steps=steps)
        errs.append((s[:, embed_edges].sum(1) - target).abs().max().item())
    assert errs[1] < 0.05 * target.abs().max().item()
    assert errs[1] < errs[0]


def test_v_patched_forward_kl_full_keep_zero_empty_keep_equals_corrupt_run(model3):
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    with torch.no_grad():
        clean_lp = answer_logprobs(model3(clean).logits)
        corrupt = node_outputs(model3, g, corrupt_ids)
        full = patched_forward(
            model3, g, clean, corrupt, torch.ones(g.n_upstream, g.n_receivers, dtype=torch.bool)
        )
        empty = patched_forward(model3, g, clean, corrupt, torch.zeros(g.n_upstream, g.n_receivers))
        corrupt_kl = kl_2tok(model3(corrupt_ids).logits, clean_lp)
    assert kl_2tok(full, clean_lp).abs().max() < 1e-6
    torch.testing.assert_close(kl_2tok(empty, clean_lp), corrupt_kl, atol=1e-4, rtol=1e-4)
    assert (corrupt_kl > 1e-3).all()


# ------------------------------------------------ sparse path for on/off masks

DENSE_ENV = "GEODE_EAPIG_DENSE"
FRACS = [0.0, "one", 0.001, 0.01, 0.3, 0.5, 0.99, 1.0]


@pytest.fixture(scope="module")
def model_gqa4():
    """4 query heads per KV group, more layers/edges than model3."""
    return _model(4, seed=3, n_heads=8, n_kv_heads=2)


@pytest.fixture(scope="module")
def model_mha():
    return _model(2, seed=4, n_heads=4, n_kv_heads=4)


def _mask(g: EdgeGraph, frac, form: str, seed: int, noisy_invalid: bool = True) -> torch.Tensor:
    """keep_only / remove masks as run.py builds them (circuit = k random edges).

    ``noisy_invalid`` fills invalid entries with random bits (they must be ignored).
    """
    gen = torch.Generator().manual_seed(seed)
    k = 1 if frac == "one" else round(frac * g.n_edges)
    if frac not in (0, 0.0) and frac != "one":
        k = max(k, 1)
    circ = g.mask_from_flat(torch.randperm(g.n_edges, generator=gen)[:k])
    keep = circ if form == "keep_only" else g.valid & ~circ
    if noisy_invalid:
        noise = torch.rand(keep.shape, generator=gen) < 0.5
        keep = torch.where(g.valid, keep, noise)
    return keep


def _dense(monkeypatch, *args, **kw) -> torch.Tensor:
    monkeypatch.setenv(DENSE_ENV, "1")
    try:
        return patched_forward(*args, **kw)
    finally:
        monkeypatch.delenv(DENSE_ENV)


def _spy_run(monkeypatch) -> list[bool]:
    """Record, per patched_forward, whether _run got a sparse plan."""
    calls: list[bool] = []
    orig = eapig._run

    def spy(*args, **kw):
        calls.append(kw.get("plan") is not None)
        return orig(*args, **kw)

    monkeypatch.setattr(eapig, "_run", spy)
    return calls


def _effective_w(g: EdgeGraph, plan) -> torch.Tensor:
    """(U, R) bool: edges carrying corrupt_u, rebuilt from the plan alone (valid region)."""
    slot_to_u = []
    for s, (a, local, cnt) in sorted(plan.chunks.items()):
        assert a == len(slot_to_u)
        us = torch.arange(s, s + cnt) if local is None else s + local.cpu()
        slot_to_u += us.tolist()
    assert len(slot_to_u) == plan.n_slots
    slot_to_u = torch.tensor(slot_to_u, dtype=torch.long)
    H, KV = g.n_heads, g.n_kv_heads
    specs = []
    for layer in range(g.n_layers):
        r0 = layer * g.recv_per_layer
        specs += [(g.n_avail_attn(layer), r0, r0 + H + 2 * KV)]
        specs += [(g.n_avail_mlp(layer), g.mlp_in_index(layer), g.mlp_in_index(layer) + 1)]
    specs.append((g.n_upstream, g.logits_index, g.logits_index + 1))
    w = torch.zeros(g.n_upstream, g.n_receivers)
    assert len(specs) == len(plan.groups)
    for (n, r0, r1), grp in zip(specs, plan.groups):
        region = torch.full((n, r1 - r0), 1.0 if grp.base_corrupt else 0.0)
        if grp.n_cols:
            if grp.dense:
                assert not grp.base_corrupt and grp.n_cols == r1 - r0
                rows = torch.arange(n)
                cols = torch.arange(r1 - r0)
            else:
                rows = slot_to_u[grp.rows.cpu()]
                cols = _group_cols(g, grp)
            assert (rows < n).all()  # exceptions only read the valid prefix
            region[rows[:, None], cols[None, :]] += grp.coef.cpu()
        w[:n, r0:r1] = region
    assert ((w == 0) | (w == 1)).all()
    return w.bool()


def _group_cols(g: EdgeGraph, grp) -> torch.Tensor:
    """Local receiver indices (within the group) that have exceptions, from ``parts``."""
    if not grp.parts:  # m.in / logits: one receiver
        assert grp.n_cols == 1
        return torch.zeros(1, dtype=torch.long)
    H, KV = g.n_heads, g.n_kv_heads
    cols = torch.full((grp.n_cols,), -1, dtype=torch.long)
    for (pos, heads, cnt), off in zip(grp.parts, (0, H, H + KV)):
        assert len(pos) == len(heads) == cnt
        cols[pos.cpu()] = heads.cpu() + off
    assert (cols >= 0).all() and (cols[1:] > cols[:-1]).all()
    return cols


@pytest.mark.parametrize("form", ["keep_only", "remove"])
@pytest.mark.parametrize("frac", FRACS)
def test_v_sparse_path_matches_dense(model3, monkeypatch, frac, form):
    g = build_graph(model3)
    ids = _ids(0)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        for seed in range(3):
            keep = _mask(g, frac, form, seed)
            sparse = patched_forward(model3, g, ids, acts, keep)
            dense = _dense(monkeypatch, model3, g, ids, acts, keep)
            _close(sparse, dense, rtol=1e-5)


@pytest.mark.parametrize("form", ["keep_only", "remove"])
@pytest.mark.parametrize("frac", ["one", 0.01, 0.05, 0.3, 0.99])
@pytest.mark.parametrize("fixture", ["model_gqa4", "model_mha"])
def test_v_sparse_path_matches_dense_other_head_layouts(request, monkeypatch, fixture, frac, form):
    model = request.getfixturevalue(fixture)
    g = build_graph(model)
    ids = _ids(2, b=4)
    with torch.no_grad():
        acts = node_outputs(model, g, _ids(3, b=4))
        for seed in range(2):
            keep = _mask(g, frac, form, seed)
            _close(
                patched_forward(model, g, ids, acts, keep),
                _dense(monkeypatch, model, g, ids, acts, keep),
                rtol=1e-5,
            )


@pytest.mark.parametrize("form", ["keep_only", "remove"])
@pytest.mark.parametrize("frac", [0.0, "one", 0.01, 0.3, 0.99])
def test_v_sparse_path_matches_dense_mean_ablation_broadcast(model3, monkeypatch, frac, form):
    g = build_graph(model3)
    ids = _ids(0)
    keep = _mask(g, frac, form, seed=11)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1, b=4))
        for m in (acts.mean(dim=(0, 1), keepdim=True), acts.mean(dim=0, keepdim=True)):
            sparse = patched_forward(model3, g, ids, m, keep)
            assert sparse.shape == (3, T, VOCAB)
            _close(sparse, _dense(monkeypatch, model3, g, ids, m, keep), rtol=1e-5)
            full = m.expand(3, T, -1, -1).contiguous()
            _close(sparse, patched_forward(model3, g, ids, full, keep), rtol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1), (1, T), (5, 3)])
def test_v_sparse_path_matches_dense_odd_batch_shapes(model3, monkeypatch, shape):
    g = build_graph(model3)
    b, t = shape
    ids = _ids(4, b=b, t=t)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(5, b=b, t=t))
        for frac, form in [("one", "remove"), (0.01, "keep_only"), (0.5, "remove")]:
            keep = _mask(g, frac, form, seed=3)
            _close(
                patched_forward(model3, g, ids, acts, keep),
                _dense(monkeypatch, model3, g, ids, acts, keep),
                rtol=1e-5,
            )


@pytest.mark.parametrize("form", ["keep_only", "remove"])
def test_v_sparse_whole_layer_with_exceptions(model3, monkeypatch, form):
    """Every q/k/v receiver of layer 1 has one exception: no shared projection is used."""
    g = build_graph(model3)
    n_attn = g.n_heads + 2 * g.n_kv_heads
    keep = g.valid.clone() if form == "remove" else torch.zeros_like(g.valid)
    gen = torch.Generator().manual_seed(0)
    n = g.n_avail_attn(1)
    for j in range(n_attn):
        u = int(torch.randint(0, n, (1,), generator=gen))
        keep[u, g.q_index(1, 0) + j] = form == "keep_only"
    keep[g.head_index(0, 1), g.logits_index] = form == "keep_only"  # one elsewhere too
    plan = eapig._binary_plan(g, keep, "cpu", torch.float32)
    grp = plan.groups[2]
    assert not grp.dense and grp.n_cols == n_attn
    assert grp.base_corrupt == (form == "keep_only")
    assert [c for _, _, c in grp.parts] == [g.n_heads, g.n_kv_heads, g.n_kv_heads]
    ids = _ids(0)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        _close(
            patched_forward(model3, g, ids, acts, keep),
            _dense(monkeypatch, model3, g, ids, acts, keep),
            rtol=1e-5,
        )


@pytest.mark.parametrize("form", ["keep_only", "remove"])
@pytest.mark.parametrize("kind", ["q", "k", "v"])
def test_v_sparse_single_receiver_exceptions(model_gqa4, monkeypatch, kind, form):
    """Exceptions only at one q head or one k/v group (GQA: k/v receivers per KV group)."""
    g = build_graph(model_gqa4)
    layer = 2
    r = {"q": g.q_index(layer, 5), "k": g.k_index(layer, 1), "v": g.v_index(layer, 0)}[kind]
    keep = g.valid.clone() if form == "remove" else torch.zeros_like(g.valid)
    for u in (0, g.head_index(0, 3), g.mlp_index(1)):
        keep[u, r] = form == "keep_only"
    plan = eapig._binary_plan(g, keep, "cpu", torch.float32)
    counts = [c for _, _, c in plan.groups[2 * layer].parts]
    assert counts == {"q": [1, 0, 0], "k": [0, 1, 0], "v": [0, 0, 1]}[kind]
    assert plan.n_slots == 3
    assert sum(grp.n_cols for grp in plan.groups) == 1
    ids = _ids(0)
    with torch.no_grad():
        acts = node_outputs(model_gqa4, g, _ids(1))
        out = patched_forward(model_gqa4, g, ids, acts, keep)
        _close(out, _dense(monkeypatch, model_gqa4, g, ids, acts, keep), rtol=1e-5)
        base = patched_forward(model_gqa4, g, ids, acts, keep.clone().fill_(form == "remove"))
        if form == "remove":
            # the single receiver's patch matters (keep-only: nothing reads the restored
            # receiver's node, every other edge carries corrupt_u, so out == all-corrupt)
            assert (out - base).abs().max() > 1e-4
        else:
            torch.testing.assert_close(out, base, rtol=0, atol=1e-5)


def test_v_sparse_mixed_group_modes_in_one_mask(model3, monkeypatch):
    """One mask mixing base-corrupt, base-resid and dense-fallback groups."""
    g = build_graph(model3)
    gen = torch.Generator().manual_seed(1)
    keep = g.valid.clone()
    n_attn = g.n_heads + 2 * g.n_kv_heads
    keep[:, g.q_index(0, 0) : g.q_index(0, 0) + n_attn] = False  # layer 0 attn: all patched
    keep[2, g.q_index(0, 0) + 1] = True  # ... but one exception
    a1 = slice(g.q_index(1, 0), g.q_index(1, 0) + n_attn)
    keep[:, a1] = torch.rand(g.n_upstream, n_attn, generator=gen) < 0.5  # layer 1: dense
    keep[g.head_index(1, 2), g.mlp_in_index(1)] = False  # base-resid exception
    keep[:, g.mlp_in_index(2)] = False  # m2.in fully patched
    keep[:, g.logits_index] = torch.rand(g.n_upstream, generator=gen) < 0.9
    plan = eapig._binary_plan(g, keep, "cpu", torch.float32)
    a0, m1, a1g, a2 = plan.groups[0], plan.groups[3], plan.groups[2], plan.groups[4]
    assert a0.base_corrupt and not a0.dense and a0.n_cols == 0  # read prefix is embed only
    assert a1g.dense and not a1g.base_corrupt
    assert not m1.base_corrupt and m1.n_cols == 1
    assert plan.groups[5].base_corrupt and plan.groups[5].n_cols == 0
    assert not a2.dense and a2.n_cols == 0
    assert plan.need_corrupt_sum
    ids = _ids(0)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        _close(
            patched_forward(model3, g, ids, acts, keep),
            _dense(monkeypatch, model3, g, ids, acts, keep),
            rtol=1e-5,
        )


@pytest.mark.parametrize("frac", FRACS + [0.05, 0.2])
def test_v_sparse_plan_reconstructs_mask(model_gqa4, frac):
    """The plan's base + exceptions rebuild exactly the patched valid edges (any device dtype)."""
    g = build_graph(model_gqa4)
    for form in ("keep_only", "remove"):
        for seed in range(3):
            keep = _mask(g, frac, form, seed)
            for k in (keep, keep.float(), keep.to(torch.int64)):
                plan = eapig._binary_plan(g, k, "cpu", torch.float32)
                w = _effective_w(g, plan)
                assert torch.equal(w[g.valid], ~keep[g.valid].bool())
                assert not w[~g.valid].any()


def test_v_sparse_plan_stores_only_needed_nodes(model_gqa4):
    g = build_graph(model_gqa4)
    # all corrupt: no exceptions anywhere, no D rows at all
    plan = eapig._binary_plan(g, torch.zeros_like(g.valid), "cpu", torch.float32)
    assert plan.n_slots == 0 and not plan.chunks and plan.need_corrupt_sum
    assert all(grp.base_corrupt and grp.n_cols == 0 for grp in plan.groups)
    # one removed edge: one D row, one receiver with an exception, no corrupt sum
    keep = g.valid.clone()
    keep[g.head_index(1, 4), g.v_index(3, 1)] = False
    plan = eapig._binary_plan(g, keep, "cpu", torch.float32)
    assert plan.n_slots == 1 and not plan.need_corrupt_sum
    assert list(plan.chunks) == [g.head_index(1, 0)]
    assert sum(grp.n_cols for grp in plan.groups) == 1
    # at 0.1% (1 edge here) .. 1%, D rows are a small subset of U
    plan = eapig._binary_plan(g, _mask(g, 0.01, "keep_only", 0), "cpu", torch.float32)
    assert 0 < plan.n_slots <= round(0.01 * g.n_edges)
    assert not any(grp.dense for grp in plan.groups)
    # half the edges: the big groups fall back to dense and store their whole prefix
    plan = eapig._binary_plan(g, _mask(g, 0.5, "remove", 0), "cpu", torch.float32)
    assert any(grp.dense for grp in plan.groups) and plan.n_slots == g.n_upstream


def test_v_binary_masks_take_sparse_path(model3, monkeypatch):
    g = build_graph(model3)
    ids = _ids(0)
    calls = _spy_run(monkeypatch)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        calls.clear()
        keep = _mask(g, 0.01, "remove", 0)
        for k in (keep, keep.float(), keep.double()):
            patched_forward(model3, g, ids, acts, k)
        # fractional values only on invalid entries are ignored -> still on/off
        k = keep.float()
        k[~g.valid] = 0.37
        patched_forward(model3, g, ids, acts, k)
        assert calls == [True] * 4
        monkeypatch.setenv(DENSE_ENV, "1")
        patched_forward(model3, g, ids, acts, keep)
        assert calls[-1] is False
        monkeypatch.delenv(DENSE_ENV)
        # all-ones keeps the plain-model fast path (no patching, no plan)
        calls.clear()
        patched_forward(model3, g, ids, acts, torch.ones_like(g.valid))
        assert calls == [False]


def test_v_invalid_fractional_entries_still_match_dense(model3, monkeypatch):
    g = build_graph(model3)
    ids = _ids(0)
    keep = _mask(g, 0.01, "keep_only", 4).float()
    keep[~g.valid] = torch.rand(int((~g.valid).sum()), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        _close(
            patched_forward(model3, g, ids, acts, keep),
            _dense(monkeypatch, model3, g, ids, acts, keep),
            rtol=1e-5,
        )


def test_v_fractional_keep_takes_dense_path(model3, monkeypatch):
    g = build_graph(model3)
    ids = _ids(0)
    keep = torch.rand(g.n_upstream, g.n_receivers, generator=torch.Generator().manual_seed(2))
    assert eapig._binary_plan(g, keep, "cpu", torch.float32) is None
    near = _mask(g, 0.01, "remove", 0).float()
    near[g.valid.nonzero()[0, 0], g.valid.nonzero()[0, 1]] = 0.5  # one fractional valid entry
    assert eapig._binary_plan(g, near, "cpu", torch.float32) is None
    calls = _spy_run(monkeypatch)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        calls.clear()
        out = patched_forward(model3, g, ids, acts, keep)
        assert calls == [False]
        torch.testing.assert_close(
            out, _dense(monkeypatch, model3, g, ids, acts, keep), rtol=0, atol=0
        )


def test_v_requires_grad_keep_takes_dense_path(model3, monkeypatch):
    """A 0/1 keep that requires grad (gradient masks) never goes sparse; grads unchanged."""
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    with torch.no_grad():
        corrupt = node_outputs(model3, g, corrupt_ids)
    calls = _spy_run(monkeypatch)

    def grad_of(env: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if env:
            monkeypatch.setenv(DENSE_ENV, "1")
        keep = _mask(g, 0.3, "remove", 0).float().requires_grad_(True)
        out = patched_forward(model3, g, clean, corrupt, keep)
        _metric(out).sum().backward()
        monkeypatch.delenv(DENSE_ENV, raising=False)
        return out.detach(), keep.grad

    calls.clear()
    out_a, grad_a = grad_of(False)
    out_b, grad_b = grad_of(True)
    assert calls == [False, False]
    assert torch.equal(out_a, out_b) and torch.equal(grad_a, grad_b)
    assert grad_a[g.valid].abs().max() > 1e-3


def test_v_eap_ig_scores_unchanged_by_dense_flag(model3, monkeypatch):
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    calls = _spy_run(monkeypatch)
    calls.clear()
    a = eap_ig_scores(model3, g, clean, corrupt_ids, _metric, steps=3)
    assert calls and not any(calls)  # scoring never builds a sparse plan
    monkeypatch.setenv(DENSE_ENV, "1")
    b = eap_ig_scores(model3, g, clean, corrupt_ids, _metric, steps=3)
    assert torch.equal(a, b)


@pytest.mark.parametrize("frac", [0.0, "one", 0.01, 0.5, 0.99])
@pytest.mark.parametrize("form", ["keep_only", "remove"])
def test_v_float01_keep_equals_bool_keep(model3, frac, form):
    g = build_graph(model3)
    ids = _ids(0)
    keep = _mask(g, frac, form, seed=6)
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        ref = patched_forward(model3, g, ids, acts, keep)
        for k in (keep.float(), keep.double(), keep.to(torch.int64)):
            assert torch.equal(patched_forward(model3, g, ids, acts, k), ref)


def _last_n_cases(g):
    rg = torch.ones(g.n_upstream, g.n_receivers)
    rg[0, g.q_index(1, 0)] = 0.0
    return {
        "all_ones": (torch.ones_like(g.valid), False),
        "sparse_keep_only": (_mask(g, 0.01, "keep_only", 0), False),
        "sparse_remove": (_mask(g, 0.01, "remove", 0), False),
        "sparse_dense_groups": (_mask(g, 0.5, "remove", 0), False),
        "dense_flag": (_mask(g, 0.01, "remove", 0), True),
        "fractional": (torch.rand(g.n_upstream, g.n_receivers), False),
        "requires_grad": (rg.requires_grad_(True), False),
    }


@pytest.mark.parametrize(
    "case",
    [
        "all_ones",
        "sparse_keep_only",
        "sparse_remove",
        "sparse_dense_groups",
        "dense_flag",
        "fractional",
        "requires_grad",
    ],
)
def test_v_last_n_logits_equal_last_rows(model3, monkeypatch, case):
    g = build_graph(model3)
    ids = _ids(0)
    keep, dense = _last_n_cases(g)[case]
    if dense:
        monkeypatch.setenv(DENSE_ENV, "1")
    with torch.no_grad():
        acts = node_outputs(model3, g, _ids(1))
        full = patched_forward(model3, g, ids, acts, keep)
        assert full.shape == (3, T, VOCAB)
        for n in (1, 2, T - 1, T, T + 3):
            part = patched_forward(model3, g, ids, acts, keep, last_n=n)
            assert part.shape == (3, min(n, T), VOCAB)
            _close(part, full[:, -n:], rtol=1e-6)


def test_v_last_n_two_leaves_answer_metrics_unchanged(model3):
    g = build_graph(model3)
    clean, corrupt_ids = _ids(0), _ids(1)
    toks = [torch.arange(3) + o for o in (3, 40, 7, 90)]
    with torch.no_grad():
        clean_lp = answer_logprobs(model3(clean).logits)
        acts = node_outputs(model3, g, corrupt_ids)
        for keep in (
            torch.ones_like(g.valid),
            _mask(g, 0.01, "keep_only", 0),
            _mask(g, 0.01, "remove", 0),
        ):
            full = patched_forward(model3, g, clean, acts, keep)
            two = patched_forward(model3, g, clean, acts, keep, last_n=2)
            _close(answer_logprobs(two), answer_logprobs(full), rtol=1e-6)
            _close(kl_2tok(two, clean_lp), kl_2tok(full, clean_lp), rtol=1e-5)
            for a, b in zip(logit_diff_2tok(two, *toks), logit_diff_2tok(full, *toks)):
                _close(a, b, rtol=1e-6)


def test_v_last_n_rejects_nonpositive(model3):
    g = build_graph(model3)
    with torch.no_grad(), pytest.raises(AssertionError):
        acts = node_outputs(model3, g, _ids(1))
        patched_forward(model3, g, _ids(0), acts, _mask(g, 0.01, "remove", 0), last_n=0)
