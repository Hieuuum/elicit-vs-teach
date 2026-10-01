"""Offline mathematical checks using fresh, tiny OLMo 2 models."""

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from geode.circuits.attribution import (
    NodeTaps,
    answer_metric,
    extract_residual_features,
    node_names,
    patch_pair,
    score_pair,
)


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(18)
        config = Olmo2Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            eos_token_id=30,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        result = Olmo2ForCausalLM(config).double().eval()
    return result


@pytest.fixture
def pair():
    return [2, 4, 6, 8, 10, 12], [2, 7, 5, 8, 10, 12], [False] * 4 + [True] * 2


def _hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_identical_pair_has_exact_zero_scores_and_patching(model, pair):
    clean, _, mask = pair
    scored = score_pair(model, clean, clean, mask)
    assert scored["scores"] == [0.0] * 10
    assert scored["metric"] == scored["corrupt_metric"]
    assert patch_pair(model, clean, clean, mask, scored["node_names"])["effect"] == 0


@pytest.mark.parametrize("node", ["layer.0.attn.0", "layer.1.attn.3", "layer.0.mlp", "layer.1.mlp"])
def test_signed_attribution_matches_central_finite_difference(model, pair, node):
    result = score_pair(model, *pair)
    # HF RMSNorm and attention internally cast to float32 even for a double
    # model. Smaller epsilon loses the finite difference to float32 rounding.
    step = 1e-2
    plus = patch_pair(model, *pair, [node], scale=step)["patched_metric"]
    minus = patch_pair(model, *pair, [node], scale=-step)["patched_metric"]
    derivative = (plus - minus) / (2 * step)
    score = result["scores"][result["node_names"].index(node)]
    assert score == pytest.approx(derivative, abs=2e-7, rel=2e-4)


def test_metric_uses_all_shifted_answer_tokens_and_mean_not_sum():
    logits = torch.arange(20, dtype=torch.float64).reshape(1, 4, 5).requires_grad_()
    ids = torch.tensor([1, 2, 3, 4])
    mask = torch.tensor([False, False, True, True])
    metric = answer_metric(logits, ids, mask)
    expected = (logits[0, 1].log_softmax(0)[3] + logits[0, 2].log_softmax(0)[4]) / 2
    torch.testing.assert_close(metric, expected)
    metric.backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 3].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() > 0


def test_label_margin_matches_manual_logits_and_derivative(model, pair):
    clean, corrupt, _ = pair
    mask = [False] * 5 + [True]
    negative = [-1] * 5 + [11]
    result = score_pair(model, clean, corrupt, mask, negative_ids=negative)
    with torch.no_grad():
        logits = model(torch.tensor([clean]), use_cache=False).logits
    assert result["metric"] == pytest.approx((logits[0, 4, 12] - logits[0, 4, 11]).item())
    step = 1e-2
    plus = patch_pair(
        model, clean, corrupt, mask, ["layer.1.mlp"], scale=step, negative_ids=negative
    )["patched_metric"]
    minus = patch_pair(
        model, clean, corrupt, mask, ["layer.1.mlp"], scale=-step, negative_ids=negative
    )["patched_metric"]
    assert result["scores"][-1] == pytest.approx((plus - minus) / (2 * step), abs=1e-6)


def test_mlp_node_is_post_norm_residual_write(model, pair):
    clean, _, _ = pair
    captured = {}
    layer = model.model.layers[0]
    handles = [
        layer.mlp.register_forward_pre_hook(
            lambda _m, inp: captured.update(mlp_input=inp[0].clone())
        ),
        layer.mlp.down_proj.register_forward_hook(
            lambda _m, _i, out: captured.update(down=out.clone())
        ),
        layer.register_forward_hook(lambda _m, _i, out: captured.update(layer_output=out.clone())),
    ]
    try:
        with NodeTaps(model) as taps, torch.no_grad():
            model(torch.tensor([clean]), use_cache=False)
            actual = taps.acts["layer.0.mlp"]
            torch.testing.assert_close(actual, layer.post_feedforward_layernorm(captured["down"]))
            torch.testing.assert_close(captured["layer_output"], captured["mlp_input"] + actual)
            assert not torch.allclose(actual, captured["down"])
    finally:
        for handle in handles:
            handle.remove()


def test_single_head_patch_preserves_all_other_heads(model, pair):
    # o_proj's forward hook reads its effective input AFTER the intervention's
    # pre-hook has modified it. Passes are corrupt, clean, then patched.
    seen = []
    handle = model.model.layers[0].self_attn.o_proj.register_forward_hook(
        lambda _m, inp, _out: seen.append(inp[0].clone().reshape(1, 6, 4, 4))
    )
    try:
        patch_pair(model, *pair, ["layer.0.attn.2"])
    finally:
        handle.remove()
    corrupt, clean, patched = seen
    torch.testing.assert_close(patched[:, :, 2], corrupt[:, :, 2])
    torch.testing.assert_close(patched[:, :, [0, 1, 3]], clean[:, :, [0, 1, 3]])
    assert not torch.allclose(patched[:, :, 2], clean[:, :, 2])


@pytest.mark.parametrize("side", ["left", "right"])
def test_padding_invariance_of_scores_metrics_and_features(model, pair, side):
    clean, corrupt, mask = pair
    base = score_pair(model, *pair)
    base_features = extract_residual_features(model, clean)

    def pad(values, padding):
        return padding + values if side == "left" else values + padding

    padded = score_pair(
        model,
        pad(clean, [0, 0]),
        pad(corrupt, [0, 0]),
        pad(mask, [False, False]),
        attention_mask=pad([1] * 6, [0, 0]),
    )
    assert padded["metric"] == pytest.approx(base["metric"], abs=1e-10)
    assert padded["scores"] == pytest.approx(base["scores"], abs=1e-7)
    features = extract_residual_features(
        model, pad(clean, [0, 0]), attention_mask=pad([1] * 6, [0, 0])
    )
    torch.testing.assert_close(features, base_features)


def test_frozen_parameters_supported_and_existing_gradients_preserved(model, pair):
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        parameter.requires_grad_(False)
    model.train()
    model.model.layers[0].eval()  # Preserve even a mixed per-module state.
    states = [module.training for module in model.modules()]
    result = score_pair(model, *pair)
    assert torch.isfinite(torch.tensor(result["scores"])).all()
    assert any(abs(value) > 1e-6 for value in result["scores"])
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(
        torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters()
    )
    assert states == [module.training for module in model.modules()]
    assert _hooks(model) == 0


def test_cleanup_when_forward_raises(model, pair):
    def fail(_module, _inputs):
        raise RuntimeError("injected failure")

    handle = model.model.layers[1].register_forward_pre_hook(fail)
    try:
        model.train()
        with pytest.raises(RuntimeError, match="injected"):
            score_pair(model, *pair)
        assert _hooks(model) == 1
        assert model.training
        with pytest.raises(RuntimeError, match="injected"):
            extract_residual_features(model, pair[0])
        assert _hooks(model) == 1
    finally:
        handle.remove()


def test_cleanup_when_backward_raises(model, pair, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected backward failure")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    model.train()
    with pytest.raises(RuntimeError, match="injected backward"):
        score_pair(model, *pair)
    assert model.training
    assert _hooks(model) == 0


def test_residual_features_are_actual_layer_outputs_before_final_norm(model, pair):
    with NodeTaps(model), torch.no_grad():
        clean = torch.tensor([pair[0]])
        expected = []
        handles = [
            model.get_input_embeddings().register_forward_hook(
                lambda _m, _i, out: expected.append(out[0, -1].float())
            )
        ]
        handles.extend(
            layer.register_forward_hook(lambda _m, _i, out: expected.append(out[0, -1].float()))
            for layer in model.model.layers
        )
        try:
            model(clean, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    features = extract_residual_features(model, pair[0])
    assert features.shape == (3, 16)
    torch.testing.assert_close(features, torch.stack(expected))
    assert not features.requires_grad
    assert features.device.type == "cpu"
    assert _hooks(model) == 0


def test_features_at_prompt_position_cannot_see_appended_gold_answer(model, pair):
    prompt = pair[0][:4]
    actual = extract_residual_features(model, prompt)
    extended = extract_residual_features(model, pair[0], position=3)
    torch.testing.assert_close(actual, extended)


@pytest.mark.parametrize(
    "clean,corrupt,mask,error",
    [
        ([1, 2, 3], [1, 3], [False, False, True], "EXACTLY"),
        ([1, 2, 3], [1, 2, 4], [False, False, True], "identical"),
        ([1, 2, 3], [1, 2, 3], [True, False, False], "target positions"),
        ([1, 2, 3], [1, 2, 3], [False, False, False], "target positions"),
    ],
)
def test_invalid_alignment_is_rejected_not_silently_dropped(model, clean, corrupt, mask, error):
    with pytest.raises(ValueError, match=error):
        score_pair(model, clean, corrupt, mask)
    assert _hooks(model) == 0


def test_padding_cannot_be_scored_as_answer(model, pair):
    with pytest.raises(ValueError, match="padding"):
        score_pair(model, *pair, attention_mask=[1, 1, 1, 1, 0, 1])


def test_zero_scale_and_no_nodes_have_zero_intervention_effect(model, pair):
    assert patch_pair(model, *pair, node_names(model), scale=0)["effect"] == 0
    assert patch_pair(model, *pair, [])["effect"] == 0


def test_independent_calls_have_no_stale_activation_state(model, pair):
    first = score_pair(model, *pair)
    second = score_pair(model, pair[1], pair[0], pair[2])
    repeated = score_pair(model, *pair)
    assert first == repeated
    assert first["metric"] == second["corrupt_metric"]
    assert first["scores"] != second["scores"]
    assert _hooks(model) == 0


def test_bfloat16_scores_and_interventions_are_finite_on_cpu(model, pair):
    model.bfloat16()
    result = score_pair(model, *pair)
    assert torch.isfinite(torch.tensor(result["scores"])).all()
    assert result["scores"] != [0.0] * len(result["scores"])
    patched = patch_pair(model, *pair, ["layer.0.attn.0", "layer.1.mlp"])
    assert torch.isfinite(torch.tensor(list(patched.values()))).all()


def test_score_pair_enables_grad_inside_outer_no_grad_context(model, pair):
    expected = score_pair(model, *pair)
    with torch.no_grad():
        actual = score_pair(model, *pair)
    assert actual == expected


def test_feature_extraction_skips_vocabulary_projection(model, pair):
    def fail(*_args):
        raise AssertionError("Feature extraction must not run the LM head")

    handle = model.lm_head.register_forward_pre_hook(fail)
    try:
        assert extract_residual_features(model, pair[0]).shape == (3, 16)
    finally:
        handle.remove()


def test_selective_projection_matches_full_logits_value_and_gradient(model, pair):
    clean, corrupt, mask = [torch.tensor(value) for value in pair]
    with NodeTaps(model) as taps:
        with torch.no_grad():
            model(corrupt[None], use_cache=False)
            corrupted = {key: value.clone() for key, value in taps.acts.items()}
        logits = model(clean[None], use_cache=False).logits
        metric = answer_metric(logits, clean, mask)
        keys = list(taps.acts)
        grads = torch.autograd.grad(metric, list(taps.acts.values()))
        full_scores = []
        for key, gradient in zip(keys, grads, strict=True):
            product = (corrupted[key] - taps.acts[key].detach()) * gradient
            if key.endswith(".attn"):
                full_scores.extend(product.reshape(1, 6, 4, 4).sum((0, 1, 3)).tolist())
            else:
                full_scores.append(product.sum().item())
    result = score_pair(model, clean, corrupt, mask)
    assert result["metric"] == pytest.approx(metric.item(), abs=1e-10)
    assert result["scores"] == pytest.approx(full_scores, abs=2e-7)
    projected_lengths = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda _m, inputs: projected_lengths.append(inputs[0].shape[1])
    )
    try:
        score_pair(model, *pair)
        assert projected_lengths == [2, 2]
    finally:
        handle.remove()
