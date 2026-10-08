"""Property tests for geode.circuits.eapig on tiny random Llama models (CPU, fp32)."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

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


def _config(n_layers: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
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


def _model(n_layers: int, seed: int) -> LlamaForCausalLM:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = LlamaForCausalLM(_config(n_layers)).float().eval()
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
