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

Modes (Miller et al. 2024's four faithfulness targets; all node-level,
resample ablation, all positions):
  sufficiency  clean acts of the top-k patched into the CORRUPT run — "restore"
  necessity    corrupt acts of the top-k patched into the CLEAN run — "destroy"
  maintain     corrupt acts of EVERYTHING OUTSIDE the top-k patched into the
               CLEAN run — the field-standard sufficiency test (signal can only
               flow through the circuit), stricter than "restore"

2026-09-24 additions:
  --nodes-from <stem>   rank nodes by ANOTHER model's map but evaluate in this
                        model — the functional reuse test (does the parent's
                        circuit carry the child's behaviour?), which does not
                        depend on which 32 names rank highest on a data half
  --random-sets N       for each k, N random node sets with the SAME MLP/head
                        counts as the top-k (type-matched); report their mean
                        fraction, how often the top-k beats them, and a
                        one-sided binomial p against q*=0.9 (Shi et al. 2024)
  per-pair spread       median / IQR / min of the per-pair fraction and the
                        share of pairs below 0.5 (Miller et al.: report the
                        worst case, not only the average)

Pure inference (no grads). Runs on CPU (fp32) with reduced --ks / --random-sets.

Usage:
    python3 circuit_faithfulness.py --map <stem> (--run-id <rid> | --model <id>) \
        [--mode sufficiency|necessity|maintain] [--nodes-from <stem>] [--random-sets N]
        [--ks 8 32] [--n-pairs 128] [--out <stem>] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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


def type_matched_random(ranked_all, top, rng):
    """A random node set with the same number of MLP blocks and heads as `top`."""
    mlps = [n for n in ranked_all if n[0] == "mlp"]
    heads = [n for n in ranked_all if n[0] == "attn"]
    n_m = sum(1 for n in top if n[0] == "mlp")
    return set(rng.sample(mlps, n_m)) | set(rng.sample(heads, len(top) - n_m))


def binom_p_ge(x: int, n: int, q: float) -> float:
    """P(X >= x) for X ~ Binomial(n, q): one-sided test that the top-k beats
    random sets more often than q* (Shi et al. 2024, sufficiency / necessity)."""
    return sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(x, n + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="circuit_nodes output stem to rank nodes by")
    ap.add_argument("--nodes-from", default=None,
                    help="rank nodes by THIS map instead (another model's circuit), evaluated in "
                    "the model given by --run-id/--model: the functional reuse test")
    ap.add_argument("--run-id", default=None, help="zoo LoRA run (wrapped checkpoint)")
    ap.add_argument("--model", default=None, help="plain checkpoint dir or hub id")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 528])
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mode", choices=("sufficiency", "necessity", "maintain"), default="sufficiency",
                    help="sufficiency: patch clean acts into the CORRUPT run (recovery); "
                    "necessity: patch corrupt acts into the CLEAN run (degradation); "
                    "maintain: patch corrupt acts into everything OUTSIDE the top-k in the "
                    "CLEAN run (performance maintained through the circuit alone)")
    ap.add_argument("--random-sets", type=int, default=0,
                    help="N type-matched random node sets per k as the reference distribution")
    ap.add_argument("--random-seed", type=int, default=316)
    ap.add_argument("--out", default=None, help="output stem (default: <map>_faithfulness_<mode>)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rank_map = args.nodes_from or args.map
    node_df = pd.read_parquet(Path(rank_map).with_suffix(".parquet"))
    node_df = node_df.sort_values("abs_score", ascending=False)
    ranked = [(r.node_type, int(r.layer), int(r.head)) for r in node_df.itertuples()]
    all_nodes = set(ranked)

    cfg = load_config(EVAL_CONFIG, None)
    df = load_frozen_parquet(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (args.model is None) == (args.run_id is None):
        raise SystemExit("[faith] pass exactly one of --model / --run-id")
    if args.run_id is not None:
        from circuit_nodes import load_sidecar_merged

        from geode.zoo import load_model as zoo_load_model

        store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
        if (store / "runs" / args.run_id / "model" / "model.safetensors").is_file():
            model = zoo_load_model(args.run_id, store=store, device=args.device)
        else:  # pruned LoRA run: parent + adapter sidecar (same rule as circuit_nodes)
            model = load_sidecar_merged(args.run_id, store, args.device)
        name = args.run_id
    else:
        dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.to(args.device)
        name = args.model
    if args.device == "cpu":
        model = model.float()
    model.eval()

    pairs = build_pairs(df, tokenizer, args.n_pairs, args.shots)
    rng = random.Random(args.random_seed)
    random_sets = {k: [type_matched_random(ranked, ranked[:k], rng) for _ in range(args.random_sets)]
                   for k in args.ks}
    taps = PatchTaps(model)
    per_pair = {"clean": [], "corrupt": []}
    per_pair.update({f"top{k}": [] for k in args.ks})
    per_pair.update({f"rand{k}_{j}": [] for k in args.ks for j in range(args.random_sets)})

    def metric_vec(logits, c_tok, x_tok):
        z = logits[:, -1].float()
        return (z.gather(1, c_tok[:, None]) - z.gather(1, x_tok[:, None])).squeeze(1).tolist()

    def patched_set(top):
        return set(top) if args.mode != "maintain" else all_nodes - set(top)

    with torch.no_grad():
        for bi, batch in enumerate(length_batches(pairs, args.batch_size)):
            clean_ids = torch.tensor([p[0] for p in batch], device=args.device)
            corr_ids = torch.tensor([p[1] for p in batch], device=args.device)
            c_tok = torch.tensor([p[2] for p in batch], device=args.device)
            x_tok = torch.tensor([p[3] for p in batch], device=args.device)

            # "capture" stores the DONOR activations to patch in: clean acts
            # for sufficiency (into the corrupt run), corrupt acts for
            # necessity and maintain (into the clean run).
            donor_ids = clean_ids if args.mode == "sufficiency" else corr_ids
            recv_ids = corr_ids if args.mode == "sufficiency" else clean_ids
            taps.mode = "capture"
            m_donor = metric_vec(model(donor_ids).logits, c_tok, x_tok)
            taps.mode = "off"
            m_recv = metric_vec(model(recv_ids).logits, c_tok, x_tok)
            if args.mode == "sufficiency":
                per_pair["clean"] += m_donor; per_pair["corrupt"] += m_recv
            else:
                per_pair["clean"] += m_recv; per_pair["corrupt"] += m_donor

            taps.mode = "patch"
            for k in args.ks:
                taps.patched_nodes = patched_set(ranked[:k])
                per_pair[f"top{k}"] += metric_vec(model(recv_ids).logits, c_tok, x_tok)
                for j, rs in enumerate(random_sets[k]):
                    taps.patched_nodes = patched_set(rs)
                    per_pair[f"rand{k}_{j}"] += metric_vec(model(recv_ids).logits, c_tok, x_tok)
            taps.mode = "off"
            if bi % 4 == 0:
                print(f"[faith]   batch {bi + 1}: {len(per_pair['clean'])}/{len(pairs)} pairs", flush=True)

    taps.remove()
    n = len(per_pair["clean"])
    clean_sum, corr_sum = sum(per_pair["clean"]), sum(per_pair["corrupt"])
    denom = clean_sum - corr_sum

    def frac_agg(vals):  # [Avg]% order: aggregate numerator over aggregate denominator
        if args.mode == "necessity":
            return (clean_sum - sum(vals)) / denom if denom else float("nan")
        return (sum(vals) - corr_sum) / denom if denom else float("nan")

    def frac_pairs(vals):  # per-pair fractions; pairs with a tiny denominator are skipped
        out = []
        for c, x, v in zip(per_pair["clean"], per_pair["corrupt"], vals):
            d = c - x
            if abs(d) < 1.0:
                continue
            out.append((c - v) / d if args.mode == "necessity" else (v - x) / d)
        return out

    word = {"sufficiency": "recovery", "necessity": "degradation", "maintain": "maintained"}[args.mode]
    src = f" nodes ranked by {rank_map}" if args.nodes_from else ""
    print(f"[faith] {name} shots={args.shots} mode={args.mode} pairs={n}{src}: "
          f"mean M_clean {clean_sum / n:.3f}  M_corrupt {corr_sum / n:.3f}")
    rows = []
    for k in args.ks:
        f = frac_agg(per_pair[f"top{k}"])
        pp = sorted(frac_pairs(per_pair[f"top{k}"]))
        row = {"k": k, "mode": args.mode, "fraction": f, "n_pairs": n,
               "n_mlp": sum(1 for nd in ranked[:k] if nd[0] == "mlp")}
        if pp:
            q = lambda t: pp[min(len(pp) - 1, int(t * len(pp)))]
            row.update({"pair_median": q(0.5), "pair_q25": q(0.25), "pair_q75": q(0.75),
                        "pair_min": pp[0], "pair_frac_below_0.5": sum(v < 0.5 for v in pp) / len(pp),
                        "pair_n_valid": len(pp)})
        line = (f"[faith]   top-{k:4d} nodes -> {word} {f:.3f}"
                + (f"  per-pair median {row['pair_median']:.3f} IQR [{row['pair_q25']:.2f}, {row['pair_q75']:.2f}] "
                   f"min {row['pair_min']:.2f}  below-0.5 {row['pair_frac_below_0.5']:.2f}" if pp else ""))
        if args.random_sets:
            rf = [frac_agg(per_pair[f"rand{k}_{j}"]) for j in range(args.random_sets)]
            beats = sum(1 for v in rf if f > v)
            pval = binom_p_ge(beats, args.random_sets, 0.9)
            row.update({"random_mean": sum(rf) / len(rf), "random_max": max(rf),
                        "beats_random": beats, "n_random": args.random_sets, "binom_p_q90": pval})
            line += (f"  | type-matched random: mean {row['random_mean']:.3f} max {row['random_max']:.3f}, "
                     f"top-k beats {beats}/{args.random_sets} (p={pval:.3g} vs q*=0.9)")
        rows.append(row)
        print(line)
    stem = args.out or (args.map + f"_faithfulness_{args.mode}")
    out = Path(stem + ".parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    meta = {"map": args.map, "nodes_from": args.nodes_from, "model": name, "shots": args.shots,
            "mode": args.mode, "n_pairs": n, "random_sets": args.random_sets,
            "per_pair": {key: [round(v, 4) for v in vals] for key, vals in per_pair.items()
                         if not key.startswith("rand")}}
    Path(stem + ".json").write_text(json.dumps(meta, indent=2))
    print(f"[faith] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
