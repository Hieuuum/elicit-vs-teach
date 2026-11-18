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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_nodes import EVAL_CONFIG, load_sidecar_merged  # noqa: E402
from task_adapter import add_task_args, load_task, load_tokenizer, resolve_tokenizer  # noqa: E402
from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402

from geode.adapt import layout  # noqa: E402

from geode.arith import exact_match_accuracy, format_valid  # noqa: E402
from geode.arith.spans import tokenize_with_spans  # noqa: E402
from geode.edl import EVAL_STOP_ROWS  # noqa: E402

STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))


def load_any(spec: str, device: str):
    """Hub id / plain dir (from_pretrained) or zoo run id (wrapped-aware)."""
    if "/" not in spec and (STORE / "runs" / spec).is_dir():
        if (STORE / "runs" / spec / "model" / "model.safetensors").is_file():
            from geode.zoo import load_model as zoo_load_model

            return zoo_load_model(spec, store=STORE, device=device)
        return load_sidecar_merged(spec, STORE, device)
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
        self.replace = False        # True: overwrite with vec (per-prompt patch)
        self.capture_rows = False   # capture per-row acts, not sums
        self.handles = []
        lay = layout(model)  # model-family node locations (Llama: o_proj / down_proj)
        self.n_heads = lay.n_heads
        self.d_head = lay.d_head
        for i in range(lay.n_layers):
            self.handles.append(lay.attn_out(i).register_forward_pre_hook(self._attn_hook(i)))
            self.handles.append(lay.mlp_out(i).register_forward_hook(self._mlp_hook(i)))

    def _attn_hook(self, i):
        def hook(_mod, inputs):
            x = inputs[0]
            if self.mode == "capture":
                v = x[:, -1].view(-1, self.n_heads, self.d_head)  # (B, H, d_head)
                self.captured[("attn", i)] = (v.detach().float() if self.capture_rows
                                              else v.detach().float().sum(0))
                return None
            if self.mode == "steer":
                if self.prefill_only and x.shape[1] == 1:
                    return None  # kv-cache decode step: leave generation unsteered
                heads = {h: self.vectors[("attn", i, h)]
                         for (k, li, h) in self.vectors if k == "attn" and li == i}
                if heads:
                    x = x.view(*x.shape[:-1], self.n_heads, self.d_head).clone()
                    for h, vec in heads.items():
                        # LAST position only (v1's every-position injection was
                        # destructive). vec is (d,) mean-shift or (B, d)
                        # per-prompt; replace=True overwrites (per-prompt patch)
                        v = vec.to(x.dtype)
                        if self.replace:
                            x[:, -1, h, :] = v
                        else:
                            x[:, -1, h, :] += self.scale * v
                    return (x.view(*x.shape[:-2], self.n_heads * self.d_head),)
            return None

        return hook

    def _mlp_hook(self, i):
        def hook(_mod, _inputs, output):
            if self.mode == "capture":
                self.captured[("mlp", i)] = (output[:, -1].detach().float()
                                             if self.capture_rows
                                             else output[:, -1].detach().float().sum(0))
                return output
            if self.mode == "steer" and ("mlp", i, -1) in self.vectors:
                if self.prefill_only and output.shape[1] == 1:
                    return output
                output = output.clone()
                v = self.vectors[("mlp", i, -1)].to(output.dtype)
                if self.replace:
                    output[:, -1] = v
                else:
                    output[:, -1] += self.scale * v
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


def main_task(args, task) -> int:
    """Non-arithmetic task (--task tofu): patch the DONOR's activations at the
    top-k nodes of --map into the frozen BASE at the answer position and read
    the next-token answer directly — first-token top-1 on the task's answer
    token and the logit-diff against its same-slot distractor (no generation:
    the task's answer is a fact token, not a parsed integer).

    Conditions: base unpatched; per-prompt donor states at the top-k nodes;
    per-prompt at k type-matched random nodes (mean over --random-sets draws);
    the donor's mean vector at the top-k (a constant, deployable patch); and
    the donor itself (ceiling). --heads-only restricts ranking, patching and
    random sets to attention heads: MLP outputs at the answer position can
    write the answer directly, heads only route information, so a heads-only
    patch that works means the FACTS are still stored in the base (latent)."""
    import random as _random

    from circuit_faithfulness import type_matched_random

    tokenizer = load_tokenizer(resolve_tokenizer(args, args.base, ""))
    items = task.items(tokenizer, args.n_eval)
    node_df = pd.read_parquet(Path(args.map).with_suffix(".parquet"))
    ranked = [(r.node_type, int(r.layer), int(r.head))
              for r in node_df.sort_values("abs_score", ascending=False).itertuples()]
    if args.heads_only:
        ranked = [n for n in ranked if n[0] == "attn"]
    top = ranked[: args.k]
    rng = _random.Random(args.seed)
    rand_sets = [sorted(type_matched_random(ranked, top, rng)) for _ in range(args.random_sets)]
    pad = tokenizer.pad_token_id
    batches = [items[i : i + args.batch_size] for i in range(0, len(items), args.batch_size)]

    def tensors(chunk):
        T = max(len(it.prompt_ids) for it in chunk)
        ids = torch.tensor([[pad] * (T - len(it.prompt_ids)) + it.prompt_ids for it in chunk],
                           device=args.device)
        am = (torch.arange(T, device=args.device)[None, :]
              >= torch.tensor([T - len(it.prompt_ids) for it in chunk], device=args.device)[:, None]).long()
        tgt = torch.tensor([it.target for it in chunk], device=args.device)
        dis = torch.tensor([it.distractors[0] for it in chunk], device=args.device)
        return ids, am, tgt, dis

    def score(logits, tgt, dis):
        z = logits[:, -1].float()
        top1 = (z.argmax(-1) == tgt).float()
        ld = (z.gather(1, tgt[:, None]) - z.gather(1, dis[:, None])).squeeze(1)
        return top1.tolist(), ld.tolist()

    # donor: per-row final-position activations at every node + its own scores
    donor = load_any(args.donor_run, args.device).eval()
    d_taps = SteerTaps(donor)
    d_taps.capture_rows = True
    donor_rows, donor_top1, donor_ld = [], [], []
    with torch.no_grad():
        for chunk in batches:
            ids, am, tgt, dis = tensors(chunk)
            d_taps.captured, d_taps.mode = {}, "capture"
            t1, ld = score(donor(input_ids=ids, attention_mask=am).logits, tgt, dis)
            d_taps.mode = "off"
            donor_rows.append({k: v for k, v in d_taps.captured.items()})
            donor_top1 += t1
            donor_ld += ld
    d_taps.remove()
    del donor
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    means = {k: torch.cat([r[k] for r in donor_rows], 0).mean(0) for k in donor_rows[0]}

    base = load_any(args.base, args.device).eval()
    taps = SteerTaps(base)
    taps.replace = True

    def vectors(nodes, gi, mean=False):
        out = {}
        for kind, layer, head in nodes:
            rows = means[(kind, layer)] if mean else donor_rows[gi][(kind, layer)]
            if kind == "attn":
                out[(kind, layer, head)] = rows[..., head, :]
            else:
                out[(kind, layer, head)] = rows
        return out

    def run(label, nodes_fn):
        t1_all, ld_all = [], []
        with torch.no_grad():
            for gi, chunk in enumerate(batches):
                ids, am, tgt, dis = tensors(chunk)
                vec = nodes_fn(gi)
                taps.vectors = vec
                taps.mode = "steer" if vec else "off"
                t1, ld = score(base(input_ids=ids, attention_mask=am).logits, tgt, dis)
                taps.mode = "off"
                t1_all += t1
                ld_all += ld
        res = {"top1": sum(t1_all) / len(t1_all), "logit_diff": sum(ld_all) / len(ld_all)}
        print(f"[steer] {label:<34}: top-1 {res['top1']:.3f}  logit-diff {res['logit_diff']:+.3f}"
              f"  (n={len(t1_all)})")
        return res

    results = {"base_unpatched": run("base unpatched", lambda gi: {}),
               f"per_prompt_top{args.k}": run(f"per-prompt donor states, top-{args.k}",
                                               lambda gi: vectors(top, gi)),
               f"mean_vector_top{args.k}": run(f"donor mean vector, top-{args.k}",
                                               lambda gi: vectors(top, gi, mean=True))}
    rand = [run(f"per-prompt, random-{args.k} #{j}", lambda gi, rs=rs: vectors(rs, gi))
            for j, rs in enumerate(rand_sets)]
    if rand:
        results[f"per_prompt_random{args.k}"] = {
            "top1": sum(r["top1"] for r in rand) / len(rand),
            "logit_diff": sum(r["logit_diff"] for r in rand) / len(rand),
            "top1_max": max(r["top1"] for r in rand), "n_sets": len(rand)}
    results["donor"] = {"top1": sum(donor_top1) / len(donor_top1),
                        "logit_diff": sum(donor_ld) / len(donor_ld)}
    print(f"[steer] donor itself (ceiling)          : top-1 {results['donor']['top1']:.3f}  "
          f"logit-diff {results['donor']['logit_diff']:+.3f}")
    taps.remove()
    meta = {"base": args.base, "donor": args.donor_run, "map": args.map, "k": args.k,
            "heads_only": args.heads_only, "task": f"{args.task}/{args.task_split}",
            "n_eval": len(items), "results": results}
    out = Path(f"{args.out}.json" if args.out else
               f"steer_{Path(args.map).name}_k{args.k}_{args.task}.json")
    out.write_text(json.dumps(meta, indent=2))
    print(f"[steer] wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="model to steer (hub id / dir / run id)")
    ap.add_argument("--donor-run", required=True, help="fine-tuned run whose shift is extracted")
    ap.add_argument("--map", required=True, help="circuit_nodes stem ranking the nodes")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--vectors", choices=("mean", "per-prompt"), default="mean",
                    help="mean: one constant shift per node (deployable patch); "
                    "per-prompt: patch each prompt with ITS OWN donor activations "
                    "at the top-k nodes (prefill only) — the upper bound that "
                    "tests whether base's circuit + the right gate STATE yields "
                    "exact answers (not deployable: needs the donor at inference)")
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
    ap.add_argument("--heads-only", action="store_true", help="task path: heads only")
    ap.add_argument("--random-sets", type=int, default=5, help="task path: random node sets")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None, help="task path: output stem")
    add_task_args(ap)
    args = ap.parse_args()
    task = load_task(args)
    if task is not None:
        return main_task(args, task)

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

    if args.vectors == "per-prompt":
        # donor rows per eval prompt at the top-k nodes (prefill positions),
        # computed in same-length groups so batching needs no padding
        donor2 = load_any(args.donor_run, args.device).eval()
        d2 = SteerTaps(donor2)
        d2.capture_rows = True
        groups: list[list[int]] = []
        by_len: dict[int, list[int]] = {}
        for idx, ids in enumerate(eval_prompts):
            by_len.setdefault(len(ids), []).append(idx)
        for idxs in by_len.values():
            for s0 in range(0, len(idxs), args.batch_size):
                groups.append(idxs[s0 : s0 + args.batch_size])
        donor_rows: list[dict] = []
        d2.mode = "capture"
        with torch.no_grad():
            for g in groups:
                d2.captured = {}
                donor2(torch.tensor([eval_prompts[i] for i in g], device=args.device))
                donor_rows.append({k: v for k, v in d2.captured.items()})
        d2.remove()
        del donor2
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

        def rows_for(nodes, gi):
            out = {}
            for kind, layer, head in nodes:
                rows = donor_rows[gi][(kind, layer)]
                out[(kind, layer, head)] = rows[:, head] if kind == "attn" else rows
            return out

        taps.mode = "steer"
        taps.replace = True
        taps.prefill_only = True  # replacement during decode makes no sense
        results = {}
        for label, nodes in (("base_unsteered", []),
                             (f"circuit_top{args.k}", top_nodes),
                             (f"random_{args.k}", rand_nodes)):
            correct = fmt_n = 0
            for gi, g in enumerate(groups):
                taps.vectors = rows_for(nodes, gi) if nodes else {}
                taps.mode = "steer" if nodes else "off"
                acc, comps = exact_match_accuracy(
                    base, tokenizer, [eval_prompts[i] for i in g],
                    [eval_answers[i] for i in g],
                    device=args.device, batch_size=args.batch_size)
                correct += round(acc * len(g))
                fmt_n += sum(format_valid("Answer:" + c) for c in comps)
            em = correct / len(eval_prompts)
            fmt = fmt_n / len(eval_prompts)
            print(f"[steer] per-prompt {label:<18}: exact_match {em:.4f}  "
                  f"format_validity {fmt:.4f} on n={len(eval_prompts)}")
            results[label] = {"em": em, "format_validity": fmt}
        taps.remove()
        meta = {"base": args.base, "donor": args.donor_run, "map": args.map,
                "k": args.k, "vectors": "per-prompt", "results": results}
        out = Path(f"steer_{Path(args.map).name}_k{args.k}_perprompt.json")
        out.write_text(json.dumps(meta, indent=2))
        print(f"[steer] wrote {out}")
        return 0

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
