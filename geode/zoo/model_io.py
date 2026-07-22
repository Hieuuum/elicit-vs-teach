"""Method-faithful checkpoint loading (specs/00 §8, V0.9).

``from_pretrained`` on a runs-5/6 LoRA checkpoint does not fail: transformers
reports every ``base``/``A``/``B`` tensor as unexpected and every plain
projection as missing, then returns a model whose wrapped projections are
randomly initialized (2026-07-22 G5 incident — gate accuracies were computed
on an effectively random model). ``load_model`` exists so no consumer ever
loads a run checkpoint without consulting the manifest's ``training.method``
first.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from geode.zoo.manifest import load_run
from geode.zoo.store import checkpoint_dir

_WRAPPED_SUFFIXES = (".base.weight", ".A.weight", ".B.weight")


def load_model(
    run_id: str,
    *,
    store: Path | None = None,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> nn.Module:
    """Load a run's final checkpoint the way its manifest says it was trained.

    ``training.method == "full_ft"`` loads via plain ``from_pretrained``;
    ``"lora"`` rebuilds the ``apply_lora`` module tree from
    ``training.lora`` and loads the wrapped state dict
    (``geode.train.reapply_lora`` — bit-exact, V5.51). A checkpoint whose
    state-dict shape contradicts the manifest's method is a refusal, never a
    silent load (V0.9). ``checkpoint`` overrides the resolved
    ``checkpoint_dir`` (must be a ``save_pretrained`` dir); the manifest
    stays the authority on how to load it. Returns the model on ``device``
    in eval mode.
    """
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    manifest = load_run(run_id, store=store)
    ckpt = Path(checkpoint) if checkpoint is not None else checkpoint_dir(run_id, store=store)
    weights = ckpt / "model.safetensors"
    with safe_open(weights, framework="pt") as f:
        wrapped = any(k.endswith(_WRAPPED_SUFFIXES) for k in f.keys())

    method = manifest.data["training"]["method"]
    if method == "lora":
        if not wrapped:
            raise ValueError(
                f"{run_id}: manifest says training.method='lora' but {weights} holds a "
                "plain state dict (no base/A/B keys) — wrong checkpoint for this run?"
            )
        from safetensors.torch import load_file

        from geode.train.lora import reapply_lora

        lora = manifest.data["training"]["lora"]
        config = AutoConfig.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_config(config)
        state = load_file(weights)
        if config.tie_word_embeddings and "lm_head.weight" not in state:
            # save_pretrained drops the tied duplicate from safetensors;
            # restore the alias so reapply_lora can load strictly.
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        reapply_lora(
            model,
            state,
            rank=lora["rank"],
            alpha=lora["alpha"],
            target_modules=tuple(lora["target_modules"]),
        )
    elif method == "full_ft":
        if wrapped:
            raise ValueError(
                f"{run_id}: manifest says training.method='full_ft' but {weights} holds a "
                "LoRA-wrapped state dict (base/A/B keys) — from_pretrained would silently "
                "random-initialize every wrapped projection (V0.9); load via the lora path"
            )
        model = AutoModelForCausalLM.from_pretrained(ckpt)
    else:
        raise ValueError(f"{run_id}: unsupported training.method {method!r}")
    return model.to(device).eval()
