"""Model-family layout: where the circuit tools find each node (spec 02 §7.1).

Every analysis tool in ``experiments/training-run/analysis`` hooks the same
handful of modules: the per-layer attention output projection (its INPUT is
the concatenated per-head outputs, the head-level node), the MLP output
projection (its OUTPUT is the MLP node's residual write), the embedding, the
final norm, and for edges the two pre-block norms. On Llama those are
``model.model.layers[i].self_attn.o_proj`` / ``.mlp.down_proj`` /
``model.model.embed_tokens`` / ``model.model.norm``; other families name them
differently. ``layout(model)`` resolves them once from a small family table so
the tools stop hard-coding Llama paths.

Silent-failure guard (V5.75): for a Llama model every accessor returns the
very module object the old hard-coded path returned, so the arithmetic default
is bit-identical. Heads are counted from the output projection's in_features
(n_heads * d_head), which is what the tools reshape.

Families:
  llama / mistral / qwen2 / qwen3   o_proj / down_proj, sequential pre-norm blocks
  phi                               self_attn.dense / mlp.fc2, PARALLEL block (one
                                    input_layernorm feeds both) -> no edge map
  gpt_neox                          attention.dense / mlp.dense_4h_to_h,
                                    parallel if config.use_parallel_residual
  olmo2                             o_proj / down_proj but POST-norms -> no edge map
Edge maps (circuit_edges.py) and the R-lens (lens_depth.py) need a sequential
pre-norm block with RMSNorm + gated SiLU MLP; ``supports_edges`` /
``supports_lrp`` say whether a family qualifies, and the tools refuse otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# family -> paths relative to the CausalLM (dotted) or to one decoder layer
_FAMILIES: dict[str, dict] = {
    "llama": dict(layers="model.layers", embed="model.embed_tokens", norm="model.norm",
                  attn_out="self_attn.o_proj", mlp_out="mlp.down_proj",
                  ln_attn="input_layernorm", ln_mlp="post_attention_layernorm",
                  parallel=False, lrp=True,
                  groups={"q_proj": "QK", "k_proj": "QK", "v_proj": "VO", "o_proj": "VO",
                          "gate_proj": "MLP", "up_proj": "MLP", "down_proj": "MLP"}),
    "phi": dict(layers="model.layers", embed="model.embed_tokens", norm="model.final_layernorm",
                attn_out="self_attn.dense", mlp_out="mlp.fc2",
                ln_attn="input_layernorm", ln_mlp=None, parallel=True, lrp=False,
                groups={"q_proj": "QK", "k_proj": "QK", "v_proj": "VO", "dense": "VO",
                        "fc1": "MLP", "fc2": "MLP"}),
    "gpt_neox": dict(layers="gpt_neox.layers", embed="gpt_neox.embed_in",
                     norm="gpt_neox.final_layer_norm", attn_out="attention.dense",
                     mlp_out="mlp.dense_4h_to_h", ln_attn="input_layernorm",
                     ln_mlp="post_attention_layernorm", parallel=None, lrp=False,
                     groups={"query_key_value": "QK", "dense": "VO",
                             "dense_h_to_4h": "MLP", "dense_4h_to_h": "MLP"}),
    "olmo2": dict(layers="model.layers", embed="model.embed_tokens", norm="model.norm",
                  attn_out="self_attn.o_proj", mlp_out="mlp.down_proj",
                  ln_attn=None, ln_mlp=None, parallel=False, lrp=False,
                  groups={"q_proj": "QK", "k_proj": "QK", "v_proj": "VO", "o_proj": "VO",
                          "gate_proj": "MLP", "up_proj": "MLP", "down_proj": "MLP"}),
}
_ALIASES = {"mistral": "llama", "qwen2": "llama", "qwen3": "llama", "olmo": "llama"}


def family_of(model_type: str) -> str:
    """Family key for a ``config.model_type``; raises on an unknown type."""
    fam = _ALIASES.get(model_type, model_type)
    if fam not in _FAMILIES:
        raise ValueError(f"geode.adapt: no layout for model_type {model_type!r}; "
                         f"known: {sorted(set(_FAMILIES) | set(_ALIASES))}")
    return fam


def _get(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True)
class ModelLayout:
    """Resolved node locations for one model instance."""

    model: torch.nn.Module
    family: str

    @property
    def _spec(self) -> dict:
        return _FAMILIES[self.family]

    @property
    def layers(self):
        return _get(self.model, self._spec["layers"])

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def attn_out(self, i: int) -> torch.nn.Module:
        """Attention output projection of layer i (its INPUT = concat of heads)."""
        return _get(self.layers[i], self._spec["attn_out"])

    def mlp_out(self, i: int) -> torch.nn.Module:
        """MLP output projection of layer i (its OUTPUT = the MLP's residual write)."""
        return _get(self.layers[i], self._spec["mlp_out"])

    def embed(self) -> torch.nn.Module:
        return _get(self.model, self._spec["embed"])

    def final_norm(self) -> torch.nn.Module:
        return _get(self.model, self._spec["norm"])

    def ln_attn(self, i: int) -> torch.nn.Module:
        name = self._spec["ln_attn"]
        if name is None:
            raise NotImplementedError(f"{self.family}: no pre-attention norm")
        return _get(self.layers[i], name)

    def ln_mlp(self, i: int) -> torch.nn.Module:
        name = self._spec["ln_mlp"]
        if name is None:
            raise NotImplementedError(f"{self.family}: no pre-MLP norm")
        return _get(self.layers[i], name)

    @property
    def n_heads(self) -> int:
        return int(self.model.config.num_attention_heads)

    @property
    def d_head(self) -> int:
        """Per-query-head width, from the output projection the tools reshape."""
        w = self.attn_out(0).weight if hasattr(self.attn_out(0), "weight") else None
        if w is None:  # geode LoRA-wrapped linear: base carries the weight
            w = self.attn_out(0).base.weight
        return int(w.shape[1]) // self.n_heads

    @property
    def parallel(self) -> bool:
        p = self._spec["parallel"]
        if p is None:
            p = bool(getattr(self.model.config, "use_parallel_residual", False))
        return p

    @property
    def supports_edges(self) -> bool:
        s = self._spec
        return not self.parallel and s["ln_attn"] is not None and s["ln_mlp"] is not None

    @property
    def supports_lrp(self) -> bool:
        return bool(self._spec["lrp"]) and getattr(self.model.config, "hidden_act", "silu") == "silu"

    def weight_group(self, module_leaf: str) -> str:
        """QK / VO / MLP class of a weight's module name (last dotted component)."""
        return self._spec["groups"].get(module_leaf, "other")


def layout(model: torch.nn.Module) -> ModelLayout:
    """The layout of a HF CausalLM, from ``model.config.model_type``."""
    return ModelLayout(model=model, family=family_of(model.config.model_type))


def weight_groups(model_type: str) -> dict[str, str]:
    """Module-leaf -> QK/VO/MLP map for a model_type (checkpoint-diff tools)."""
    return dict(_FAMILIES[family_of(model_type)]["groups"])
