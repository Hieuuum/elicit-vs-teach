"""The practical test: can the fine-tune be replaced by PATCHING the circuit?

Elicitation's applied prediction (owner 2026-08-24): if fine-tuning merely
re-weights an existing circuit, its effect at the circuit nodes is
approximately a CONSTANT activation shift — so adding that shift to the BASE
model at inference (zero training, k node-vectors) should unlock the
capability. Teaching's prediction: no shift can help, because there is no
circuit underneath to unlock. (The repo's cited Wang et al. 2025 found
exactly this constant-shift structure for OOCR; here it becomes an
elicit-vs-teach discriminator.)

Protocol:
1. CALIBRATION: run donor (fine-tuned) and base on the same bare prompts;
   steering vector per node = mean over prompts of (act_donor - act_base) at
   the FINAL prompt position.
2. STEERED EVAL on held-out prompts (disjoint rows): base model + hooks
   adding the top-k circuit nodes' vectors at EVERY position (constant
   shift), greedy EOS-stopped generation, exact match — the G5 protocol.
3. CONTROLS: base unsteered (floor), random-k nodes (same count, same
   procedure — circuit-specificity), donor itself (ceiling).

Readout: EM(base+circuit-steer) >> EM(base) ~ EM(base+random-steer) on the
elicit side, and ~0 everywhere on the teach side, is the practical result:
"elicitation-regime capability = patch k vectors; teaching-regime = you
must actually train."

GPU, box-only.

Usage:
    python3 steer_unlock.py --base <model-or-run> --donor-run <rid> \
        --map <circuit stem> [--k 32] [--scale 1.0] [--n-calib 64] [--n-eval 256]
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

from circuit_nodes import EVAL_CONFIG  # noqa: E402
from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402

from geode.arith import exact_match_accuracy, format_valid  # noqa: E402
from geode.arith.spans import tokenize_with_spans  # noqa: E402
from geode.edl import EVAL_STOP_ROWS  # noqa: E402

STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))


def load_any(spec: str, device: str):
    """Hub id / plain dir (from_pretrained) or zoo run id (wrapped-aware)."""
    if "/" not in spec and (STORE / "runs" / spec).is_dir():
        from geode.zoo import load_model as zoo_load_model

        return zoo_load_model(spec, store=STORE, device=device)
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=torch.bfloat16)
    return m.to(device)


class SteerTaps:
    """Capture final-position node activations, or add constant node shifts."""

    def __init__(self, model):
        self.mode = "off"  # "capture" | "steer" | "off"
        self.captured: dict[tuple, torch.Tensor] = {}
        self.vectors: dict[tuple, torch.Tensor] = {}  # node -> shift vector
        self.scale = 1.0
        self.prefill_only = False
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
                v = x[:, -1].view(-1, self.n_heads, self.d_head)  # (B, H, d_head)
                self.captured[("attn", i)] = v.detach().float().sum(0)
                return None
            if self.mode == "steer":
                if self.prefill_only and x.shape[1] == 1:
                    return None  # kv-cache decode step: leave generation unsteered
                heads = {h: self.vectors[("attn", i, h)]
                         for (k, li, h) in self.vectors if k == "attn" and li == i}
                if heads:
                    x = x.view(*x.shape[:-1], self.n_heads, self.d_head).clone()
                    for h, vec in heads.items():
                        # LAST position only: calibration measured the shift
                        # there, and injecting it everywhere is destructive
                        # (v1 measured 0.0000 even at all-528 nodes)
                        x[:, -1, h, :] += self.scale * vec.to(x.dtype)
                    return (x.view(*x.shape[:-2], self.n_heads * self.d_head),)
            return None

        return hook

    def _mlp_hook(self, i):
        def hook(_mod, _inputs, output):
            if self.mode == "capture":
                self.captured[("mlp", i)] = output[:, -1].detach().float().sum(0)
                return output
            if self.mode == "steer" and ("mlp", i, -1) in self.vectors:
                if self.prefill_only and output.shape[1] == 1:
                    return output
                output = output.clone()
                output[:, -1] += self.scale * self.vectors[("mlp", i, -1)].to(output.dtype)
                return output
            return output

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def capture_means(model, taps, prompt_ids, device, batch_size):
    """{(kind, layer)} -> mean final-position activation over prompts."""
    sums: dict[tuple, torch.Tensor] = {}
    n = 0
    taps.mode = "capture"
    with torch.no_grad():
        for s in range(0, len(prompt_ids), batch_size):
            chunk = prompt_ids[s : s + batch_size]
            # group by equal length to batch without padding
            by_len: dict[int, list] = {}
            for ids in chunk:
                by_len.setdefault(len(ids), []).append(ids)
            for ids_list in by_len.values():
                taps.captured = {}
                model(torch.tensor(ids_list, device=device))
                for k, v in taps.captured.items():
                    sums[k] = sums.get(k, 0) + v
                n += len(ids_list)
    taps.mode = "off"
    return {k: v / n for k, v in sums.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="model to steer (hub id / dir / run id)")
    ap.add_argument("--donor-run", required=True, help="fine-tuned run whose shift is extracted")
    ap.add_argument("--map", required=True, help="circuit_nodes stem ranking the nodes")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prefill-only", action="store_true",
                    help="steer only the prompt's final position, not each decode "
                    "step — kickstart into answer mode, then free-run (fixes the "
                    "perseveration collapse: re-injecting the write-a-digit vector "
                    "every generated token yields '------' loops, measured v2)")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=316)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(EVAL_CONFIG, None)
    df = load_frozen_parquet(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib = df.iloc[EVAL_STOP_ROWS : EVAL_STOP_ROWS + args.n_calib]
    evalr = df.iloc[EVAL_STOP_ROWS + args.n_calib : EVAL_STOP_ROWS + args.n_calib + args.n_eval]
    calib_ids = [tokenizer(p, add_special_tokens=False)["input_ids"]
                 for p in calib["prompt_text"]]
    ex = tokenize_with_spans(evalr["full_text"].tolist(),
                             list(zip(evalr["answer_char_start"].astype(int),
                                      evalr["answer_char_end"].astype(int))),
                             tokenizer, append_eos=True)
    eval_prompts = [e.input_ids[: e.label_span[0]] for e in ex]
    eval_answers = evalr["true_answer"].astype(int).tolist()

    node_df = pd.read_parquet(Path(args.map).with_suffix(".parquet"))
    ranked = [(r.node_type, int(r.layer), int(r.head))
              for r in node_df.sort_values("abs_score", ascending=False).itertuples()]
    top_nodes = ranked[: args.k]
    gen = torch.Generator().manual_seed(args.seed)
    rand_idx = torch.randperm(len(ranked), generator=gen)[: args.k].tolist()
    rand_nodes = [ranked[i] for i in rand_idx]

    # ---- calibration: donor means, then base means ----
    donor = load_any(args.donor_run, args.device).eval()
    d_taps = SteerTaps(donor)
    donor_means = capture_means(donor, d_taps, calib_ids, args.device, args.batch_size)
    d_taps.remove()
    del donor
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    base = load_any(args.base, args.device).eval()
    taps = SteerTaps(base)
    base_means = capture_means(base, taps, calib_ids, args.device, args.batch_size)

    def vectors_for(nodes):
        out = {}
        for kind, layer, head in nodes:
            delta = donor_means[(kind, layer)] - base_means[(kind, layer)]
            out[(kind, layer, head)] = delta[head] if kind == "attn" else delta
        return out

    def run_em(label):
        acc, completions = exact_match_accuracy(base, tokenizer, eval_prompts, eval_answers,
                                                device=args.device, batch_size=args.batch_size)
        fmt = sum(format_valid("Answer:" + c) for c in completions) / len(completions)
        print(f"[steer] {label}: exact_match {acc:.4f}  format_validity {fmt:.4f} "
              f"on n={len(eval_prompts)}")
        for c in completions[:2]:
            print(f"[steer]     sample: {c[:70]!r}")
        return {"em": acc, "format_validity": fmt}

    results = {}
    taps.mode = "off"
    results["base_unsteered"] = run_em("base unsteered           ")
    taps.mode = "steer"
    taps.scale = args.scale
    taps.prefill_only = args.prefill_only
    taps.vectors = vectors_for(top_nodes)
    results[f"circuit_top{args.k}"] = run_em(f"base + circuit top-{args.k:<4d}")
    taps.vectors = vectors_for(rand_nodes)
    results[f"random_{args.k}"] = run_em(f"base + random {args.k:<8d}")
    attn_nodes = [n for n in ranked if n[0] == "attn"][: args.k]
    taps.vectors = vectors_for(attn_nodes)
    results[f"attn_top{args.k}"] = run_em(f"base + attn-only top-{args.k:<3d}")
    taps.vectors = vectors_for(ranked)  # all 528 nodes = full constant shift
    results["all_nodes"] = run_em("base + ALL node shifts   ")
    taps.remove()

    meta = {"base": args.base, "donor": args.donor_run, "map": args.map, "k": args.k,
            "scale": args.scale, "prefill_only": args.prefill_only,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "results": results}
    out = Path(f"steer_{Path(args.map).name}_k{args.k}.json")
    out.write_text(json.dumps(meta, indent=2))
    print(f"[steer] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
