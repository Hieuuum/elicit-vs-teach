"""Faithfulness of an identified circuit, by TRUE activation patching.

The rigor step behind the attribution maps (which are first-order
approximations): take a node map from ``circuit_nodes.py``, and for each k in
--ks patch the top-k |score| nodes' CLEAN activations into the CORRUPT run.
Recovery fraction = (M_patched - M_corrupt) / (M_clean - M_corrupt), where M
is the same logit-diff metric — the share of the model's clean-vs-corrupt
behavior the k-node circuit is SUFFICIENT to restore (Prakash-et-al-style
faithfulness: "a k-node circuit recovers X% of behavior").

Sanity built in: k = ALL nodes must recover ~1.0 (patching everything
reproduces the clean computation up to the embeddings/layernorm paths not
covered by the taps — expect >0.9); k=0 is 0 by construction. A top-k curve
that climbs steeply toward 1.0 validates both the circuit and the
attribution ranking that chose it.

Pure inference (no grads). GPU, box-only.

Usage:
    python3 circuit_faithfulness.py --map <stem> (--run-id <rid> | --model <id>) \
        [--shots K] [--ks 8 16 32 64 128 528] [--n-pairs 128]
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

from circuit_nodes import EVAL_CONFIG, build_pairs, length_batches  # noqa: E402
from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402


class PatchTaps:
    """Hooks that REPLACE selected node activations with stored clean ones."""

    def __init__(self, model):
        self.mode = "off"  # "capture" | "patch" | "off"
        self.clean: dict[tuple, torch.Tensor] = {}
        self.patched_nodes: set[tuple] = set()
        self.handles = []
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads
        for i, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._attn_hook(i))
            )
            self.handles.append(layer.mlp.down_proj.register_forward_hook(self._mlp_hook(i)))

    def _attn_hook(self, i):
        def hook(_mod, inputs):
            x = inputs[0]
            if self.mode == "capture":
                self.clean[("attn", i)] = x.detach().clone()
                return None
            if self.mode == "patch":
                heads = [h for (k, li, h) in self.patched_nodes if k == "attn" and li == i]
                if heads:
                    x = x.view(*x.shape[:-1], self.n_heads, self.d_head).clone()
                    c = self.clean[("attn", i)].view(*x.shape)
                    x[..., heads, :] = c[..., heads, :]
                    return (x.view(*x.shape[:-2], self.n_heads * self.d_head),)
            return None

        return hook

    def _mlp_hook(self, i):
        def hook(_mod, _inputs, output):
            if self.mode == "capture":
                self.clean[("mlp", i)] = output.detach().clone()
                return output
            if self.mode == "patch" and ("mlp", i, -1) in self.patched_nodes:
                return self.clean[("mlp", i)]
            return output

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, help="circuit_nodes output stem to rank nodes by")
    ap.add_argument("--run-id", default=None, help="zoo LoRA run (wrapped checkpoint)")
    ap.add_argument("--model", default=None, help="plain checkpoint dir or hub id")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 528])
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mode", choices=("sufficiency", "necessity"), default="sufficiency",
                    help="sufficiency: patch clean acts into the CORRUPT run (recovery); "
                    "necessity: patch corrupt acts into the CLEAN run (degradation) — "
                    "proves the circuit is load-bearing, not just a good writing site")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    node_df = pd.read_parquet(Path(args.map).with_suffix(".parquet"))
    node_df = node_df.sort_values("abs_score", ascending=False)
    ranked = [(r.node_type, int(r.layer), int(r.head)) for r in node_df.itertuples()]

    cfg = load_config(EVAL_CONFIG, None)
    df = load_frozen_parquet(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (args.model is None) == (args.run_id is None):
        raise SystemExit("[faith] pass exactly one of --model / --run-id")
    if args.run_id is not None:
        from geode.zoo import load_model as zoo_load_model

        store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
        model = zoo_load_model(args.run_id, store=store, device=args.device)
        name = args.run_id
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        model.to(args.device)
        name = args.model
    model.eval()

    pairs = build_pairs(df, tokenizer, args.n_pairs, args.shots)
    taps = PatchTaps(model)
    totals = {k: 0.0 for k in args.ks}
    m_clean_sum = m_corr_sum = 0.0
    n_batches = 0

    def metric(logits, c_tok, x_tok):
        z = logits[:, -1].float()
        return (z.gather(1, c_tok[:, None]) - z.gather(1, x_tok[:, None])).sum().item()

    with torch.no_grad():
        for batch in length_batches(pairs, args.batch_size):
            clean_ids = torch.tensor([p[0] for p in batch], device=args.device)
            corr_ids = torch.tensor([p[1] for p in batch], device=args.device)
            c_tok = torch.tensor([p[2] for p in batch], device=args.device)
            x_tok = torch.tensor([p[3] for p in batch], device=args.device)

            # "capture" stores the DONOR activations to patch in: clean acts
            # for sufficiency (into the corrupt run), corrupt acts for
            # necessity (into the clean run).
            donor_ids = clean_ids if args.mode == "sufficiency" else corr_ids
            recv_ids = corr_ids if args.mode == "sufficiency" else clean_ids
            taps.mode = "capture"
            m_donor = metric(model(donor_ids).logits, c_tok, x_tok)
            taps.mode = "off"
            m_recv = metric(model(recv_ids).logits, c_tok, x_tok)
            if args.mode == "sufficiency":
                m_clean_sum += m_donor
                m_corr_sum += m_recv
            else:
                m_clean_sum += m_recv
                m_corr_sum += m_donor
            n_batches += 1

            taps.mode = "patch"
            for k in args.ks:
                taps.patched_nodes = set(ranked[:k])
                totals[k] += metric(model(recv_ids).logits, c_tok, x_tok)
            taps.mode = "off"

    taps.remove()
    denom = m_clean_sum - m_corr_sum
    rows = []
    print(f"[faith] {name} shots={args.shots} mode={args.mode} pairs={len(pairs)}: "
          f"mean M_clean {m_clean_sum/len(pairs):.3f}  M_corrupt {m_corr_sum/len(pairs):.3f}")
    for k in args.ks:
        if args.mode == "sufficiency":
            frac = (totals[k] - m_corr_sum) / denom if denom else float("nan")
            word = "recovery"
        else:
            frac = (m_clean_sum - totals[k]) / denom if denom else float("nan")
            word = "degradation"
        rows.append({"k": k, "mode": args.mode, "fraction": frac})
        print(f"[faith]   top-{k:4d} nodes patched -> {word} {frac:.3f}")
    out = Path(args.map + f"_faithfulness_{args.mode}.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    meta = {"map": args.map, "model": name, "shots": args.shots,
            "mode": args.mode, "n_pairs": len(pairs)}
    Path(args.map + f"_faithfulness_{args.mode}.json").write_text(json.dumps(meta, indent=2))
    print(f"[faith] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
