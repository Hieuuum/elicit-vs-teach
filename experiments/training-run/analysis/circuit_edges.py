"""Edge-level attribution (EAP-style) + ΔS node-vs-edge change rates.

Metric 3 of the owner's original proposal (2026-08-24), full version — the
"planned v2" behind the score-rotation proxy. Wang et al. 2025 report that
math fine-tuning changes EDGES (ΔS_Edge ≈ 65-80%) at 2-4x the rate of NODES
(ΔS_Node ≈ 15-24%): rewiring, not recruitment. Predictions here:
elicitation = edge rewiring over a stable node set (ΔS_Edge >> ΔS_Node);
teaching = node recruitment (but note: the blank twin's pre-FT map is NOISE,
so teach-side ΔS-vs-base is undefined by the standing guard — the teach
statement stays "recruitment from nothing", and the quantitative teach
comparison is taught-vs-elicited edge structure).

Edge definition (documented approximation, first-order like the node maps):
  writers  = the 528 nodes (per-head o_proj residual writes via the W_o
             column slice; MLP down_proj outputs)
  readers  = the 32 blocks (each layer's attention, hooked at the
             input_layernorm output — which isolates gradients flowing into
             THAT block only — and each MLP at post_attention_layernorm)
  score(u->v) = sum over batch/positions of
      grad_clean(reader_v LN-out) . [ w_v ⊙ (write_u^corr - write_u^clean) / rms_v^clean ]
  i.e. the frozen-RMS pullback of the writer's clean->corrupt residual
  delta, as seen by reader v. Causal mask: attn_i feeds attn_j>i and
  mlp_j>=i; mlp_i feeds attn_j>i and mlp_j>i. ~8.9K edges.

Modes:
  map:      python3 circuit_edges.py map (--run-id R | --model M) --out stem
            [--eval-config cfg] [--shots K]  -> <stem>.parquet + .json sanity
  delta-s:  python3 circuit_edges.py delta-s --nodes-a A --nodes-b B
            --edges-a A --edges-b B [--k-nodes 32] [--k-edges 256]
            -> ΔS table (1 - Jaccard@K and new-component fraction), with the
            noise guard honored via the maps' sanity sidecars.

GPU, box-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "training-run" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit_nodes import (  # noqa: E402
    EVAL_CONFIG,
    build_pairs,
    length_batches,
    load_sidecar_merged,
)
from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402


def effective_weight(linear) -> torch.Tensor:
    """Weight of a plain nn.Linear OR a geode LoRALinear (base + scaled B@A)."""
    if hasattr(linear, "weight"):
        return linear.weight.float()
    # geode.train.lora.LoRALinear: forward = base(x) + scaling * B(A(x))
    return (linear.base.weight.float()
            + linear.scaling * (linear.B.weight.float() @ linear.A.weight.float()))


class EdgeTaps:
    """Writer activations + reader LN-out grads + reader rms, per layer."""

    def __init__(self, model):
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads
        self.o_in: dict[int, torch.Tensor] = {}     # (B,T,H*dh) o_proj input
        self.mlp_out: dict[int, torch.Tensor] = {}  # (B,T,d) down_proj output
        self.ln_grad: dict[tuple, torch.Tensor] = {}  # ("attn"/"mlp", i) -> (B,T,d)
        self.ln_rms: dict[tuple, torch.Tensor] = {}   # (B,T,1) clean rms at reader
        self.grab_grads = False
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._o_hook(i)))
            self.handles.append(
                layer.mlp.down_proj.register_forward_hook(self._m_hook(i)))
            self.handles.append(
                layer.input_layernorm.register_forward_hook(self._ln_hook(("attn", i))))
            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(self._ln_hook(("mlp", i))))

    def _o_hook(self, i):
        def hook(_m, inputs):
            self.o_in[i] = inputs[0]
            return None
        return hook

    def _m_hook(self, i):
        def hook(_m, _inp, output):
            self.mlp_out[i] = output
            return output
        return hook

    def _ln_hook(self, key):
        def hook(_m, inputs, output):
            if self.grab_grads:
                x = inputs[0]
                self.ln_rms[key] = (
                    x.detach().float().pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
                )
                out = output
                out.retain_grad() if out.requires_grad else None
                self.ln_grad[key] = out
            return output
        return hook

    def clear(self):
        self.o_in.clear()
        self.mlp_out.clear()
        self.ln_grad.clear()
        self.ln_rms.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def edge_map(model, pairs, batch_size: int, device: str):
    """({(wt,wl,wh,rt,rl): score}, mean logit-diff sanity)."""
    taps = EdgeTaps(model)
    n_layers = model.config.num_hidden_layers
    H, dh = taps.n_heads, taps.d_head
    layers = model.model.layers
    ln_w = {("attn", i): layers[i].input_layernorm.weight for i in range(n_layers)}
    ln_w.update({("mlp", i): layers[i].post_attention_layernorm.weight
                 for i in range(n_layers)})
    edges: dict[tuple, float] = {}
    sanity = []
    try:
        for batch in length_batches(pairs, batch_size):
            clean_ids = torch.tensor([p[0] for p in batch], device=device)
            corr_ids = torch.tensor([p[1] for p in batch], device=device)
            c_tok = torch.tensor([p[2] for p in batch], device=device)
            x_tok = torch.tensor([p[3] for p in batch], device=device)

            taps.grab_grads = False
            taps.clear()
            with torch.no_grad():
                model(corr_ids)
            corr_o = {i: v.detach().float() for i, v in taps.o_in.items()}
            corr_m = {i: v.detach().float() for i, v in taps.mlp_out.items()}

            taps.grab_grads = True
            taps.clear()
            model.zero_grad(set_to_none=True)
            logits = model(clean_ids).logits[:, -1].float()
            m = (logits.gather(1, c_tok[:, None]) - logits.gather(1, x_tok[:, None])).sum()
            sanity.append(m.item() / len(batch))
            m.backward()
            clean_o = {i: v.detach().float() for i, v in taps.o_in.items()}
            clean_m = {i: v.detach().float() for i, v in taps.mlp_out.items()}

            # writer residual-write deltas, per node
            with torch.no_grad():
                dwrites: dict[tuple, torch.Tensor] = {}
                for i in range(n_layers):
                    W_o = effective_weight(layers[i].self_attn.o_proj)  # (d, H*dh)
                    d_oin = (corr_o[i] - clean_o[i])                 # (B,T,H*dh)
                    for h in range(H):
                        sl = slice(h * dh, (h + 1) * dh)
                        dwrites[("attn", i, h)] = d_oin[..., sl] @ W_o[:, sl].T
                    dwrites[("mlp", i, -1)] = corr_m[i] - clean_m[i]

                for (rt, rl), g in taps.ln_grad.items():
                    if g.grad is None:
                        continue
                    gv = g.grad.detach().float() * ln_w[(rt, rl)].float()
                    gv = gv / taps.ln_rms[(rt, rl)]  # frozen-RMS pullback
                    for (wt, wl, wh), dw in dwrites.items():
                        # causal mask: writer strictly upstream of reader
                        # (attn_i also feeds this layer's own mlp)
                        if wt == "attn":
                            ok = (rt == "attn" and wl < rl) or (rt == "mlp" and wl <= rl)
                        else:
                            ok = wl < rl
                        if not ok:
                            continue
                        key = (wt, wl, wh, rt, rl)
                        edges[key] = edges.get(key, 0.0) + (gv * dw).sum().item()
    finally:
        taps.remove()
    return edges, sum(sanity) / max(1, len(sanity))


def cmd_map(args) -> int:
    cfg = load_config(Path(args.eval_config), None)
    df = load_frozen_parquet(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
    if (args.model is None) == (args.run_id is None):
        raise SystemExit("[edges] pass exactly one of --model / --run-id")
    if args.run_id is not None:
        if (store / "runs" / args.run_id / "model" / "model.safetensors").is_file():
            from geode.zoo import load_model as zoo_load_model

            model = zoo_load_model(args.run_id, store=store, device=args.device)
        else:
            model = load_sidecar_merged(args.run_id, store, args.device)
        name = args.run_id
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        model.to(args.device)
        name = args.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    pairs = build_pairs(df, tokenizer, args.n_pairs, args.shots)
    edges, sanity = edge_map(model, pairs, args.batch_size, args.device)
    performing = sanity > 2.0
    print(f"[edges] {name} shots={args.shots}: mean logit_diff {sanity:.3f} "
          f"({'PERFORMING' if performing else 'NOT PERFORMING — map is noise'}); "
          f"{len(edges)} edges")
    pd.DataFrame(
        [{"writer_type": wt, "writer_layer": wl, "writer_head": wh,
          "reader_type": rt, "reader_layer": rl, "score": v, "abs_score": abs(v)}
         for (wt, wl, wh, rt, rl), v in edges.items()]
    ).to_parquet(f"{args.out}.parquet", index=False)
    Path(f"{args.out}.json").write_text(json.dumps(
        {"model": name, "shots": args.shots, "mean_logit_diff": sanity,
         "performing_regime": performing, "n_pairs": len(pairs),
         "eval_config": str(args.eval_config)}, indent=2))
    print(f"[edges] wrote {args.out}.parquet")
    return 0


def _top_keys(path: str, k: int, keycols: list[str]):
    df = pd.read_parquet(Path(path).with_suffix(".parquet"))
    df = df.sort_values("abs_score", ascending=False).head(k)
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    return {tuple(r[c] for c in keycols) for _, r in df.iterrows()}, meta


def cmd_delta_s(args) -> int:
    ncols = ["node_type", "layer", "head"]
    ecols = ["writer_type", "writer_layer", "writer_head", "reader_type", "reader_layer"]
    na, ma = _top_keys(args.nodes_a, args.k_nodes, ncols)
    nb, mb = _top_keys(args.nodes_b, args.k_nodes, ncols)
    ea, mea = _top_keys(args.edges_a, args.k_edges, ecols)
    eb, meb = _top_keys(args.edges_b, args.k_edges, ecols)
    for label, meta in (("nodes-a", ma), ("nodes-b", mb), ("edges-a", mea), ("edges-b", meb)):
        if not meta.get("performing_regime", True):
            print(f"[delta-s] GUARD: {label} map is NOISE — ΔS below is meaningless")

    def report(label, A, B, k):
        j = len(A & B) / len(A | B)
        new = len(B - A) / k
        print(f"[delta-s] {label:<6} @K={k:<4}: ΔS(1-Jaccard) {1 - j:.3f}  "
              f"Jaccard {j:.3f}  new-fraction {new:.3f}")
        return 1 - j

    ds_n = report("NODES", na, nb, args.k_nodes)
    ds_e = report("EDGES", ea, eb, args.k_edges)
    ratio = ds_e / ds_n if ds_n > 0 else float("inf")
    print(f"[delta-s] ΔS_Edge / ΔS_Node = {ratio:.2f}  "
          f"(Wang et al. 2025 elicit-style fine-tuning: ~2-4x)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("--run-id", default=None)
    m.add_argument("--model", default=None)
    m.add_argument("--out", required=True)
    m.add_argument("--eval-config", default=str(EVAL_CONFIG))
    m.add_argument("--shots", type=int, default=0)
    m.add_argument("--n-pairs", type=int, default=128)
    m.add_argument("--batch-size", type=int, default=8)
    m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    d = sub.add_parser("delta-s")
    d.add_argument("--nodes-a", required=True)
    d.add_argument("--nodes-b", required=True)
    d.add_argument("--edges-a", required=True)
    d.add_argument("--edges-b", required=True)
    d.add_argument("--k-nodes", type=int, default=32)
    d.add_argument("--k-edges", type=int, default=256)
    args = ap.parse_args()
    return cmd_map(args) if args.cmd == "map" else cmd_delta_s(args)


if __name__ == "__main__":
    raise SystemExit(main())
