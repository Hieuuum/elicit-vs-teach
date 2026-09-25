"""Cross-format activation matching: do the WORDS reach the ENGINE?

The pre-elicit parent scores 0.000 direct EM on NL questions (it rewrites,
never computes), yet its NL logit-diff is ~+2: something problem-specific
leaks from the words to the answer. This probe makes that mechanistic. For
each problem rendered BOTH ways —

    NL:  "What is the sum of 621 and 5068?\\n"
    op:  "621 + 5068 = "

capture every node's activation at the final prompt position, and score per
node:  mean cos(NL_i, op_i)  -  mean cos(NL_i, op_j!=i)   (matched minus
mismatched). A positive index at a node means the NL prompt drives it into
the SAME problem-specific state as the op prompt — the words reach that
node's computation.

The controlled contrast (run on both models):
  evt-ts1b-op-install     engine, NO binding  -> expect index ~0
  evt-ts1b-op-bridge-mix  engine + binding    -> expect positive index at
                                                 the engine (op-circuit) nodes
Difference between the two runs = the bridge's effect, visible as circuitry.

Pure inference. GPU recommended, box-only.

Usage:
    python3 cross_format_probe.py --run-id <rid> [--n 128] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET, render_probe  # noqa: E402
from steer_unlock import SteerTaps  # noqa: E402


@torch.no_grad()
def capture_rows(model, taps, tokenizer, prompts, device, batch_size):
    """Per-node final-position activations for every prompt: {node: (N, d)}."""
    tokenizer.padding_side = "left"
    taps.capture_rows = True
    acc: dict[tuple, list] = {}
    for s in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[s : s + batch_size], return_tensors="pt",
                        padding=True, add_special_tokens=False)
        taps.captured = {}
        taps.mode = "capture"
        model(**{k: v.to(device) for k, v in enc.items()})
        taps.mode = "off"
        for key, v in taps.captured.items():
            acc.setdefault(key, []).append(v.cpu())
    return {k: torch.cat(vs, 0) for k, vs in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if (args.model is None) == (args.run_id is None):
        raise SystemExit("[xfmt] pass exactly one of --model / --run-id")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    df = pd.read_parquet(EVAL_PARQUET)
    rows = df.iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + args.n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in rows.itertuples()]
    nl_prompts = [render_probe("bare_nl", a, b, op, 0)[0] for a, b, op in triples]
    op_prompts = [render_probe("bare_op", a, b, op, 0)[0] for a, b, op in triples]

    taps = SteerTaps(model)
    nl_acts = capture_rows(model, taps, tokenizer, nl_prompts, args.device, args.batch_size)
    op_acts = capture_rows(model, taps, tokenizer, op_prompts, args.device, args.batch_size)
    taps.remove()

    # per-node matched-minus-mismatched cosine (mismatch = roll by 1: same
    # distribution of problems, wrong pairing)
    results = {}
    for (kind, layer), nl_v in nl_acts.items():
        op_v = op_acts[(kind, layer)]
        if kind == "attn":  # (N, H, d_head) -> one score per head
            for h in range(nl_v.shape[1]):
                a, b = nl_v[:, h].float(), op_v[:, h].float()
                matched = F.cosine_similarity(a, b, dim=1).mean().item()
                mism = F.cosine_similarity(a, b.roll(1, 0), dim=1).mean().item()
                results[f"attn:{layer}:{h}"] = {"matched": matched, "mismatched": mism,
                                                "index": matched - mism}
        else:  # (N, d)
            a, b = nl_v.float(), op_v.float()
            matched = F.cosine_similarity(a, b, dim=1).mean().item()
            mism = F.cosine_similarity(a, b.roll(1, 0), dim=1).mean().item()
            results[f"mlp:{layer}"] = {"matched": matched, "mismatched": mism,
                                       "index": matched - mism}

    ranked = sorted(results.items(), key=lambda kv: -kv[1]["index"])
    mean_idx = sum(v["index"] for v in results.values()) / len(results)
    mlp_by_layer = {k: v["index"] for k, v in results.items() if k.startswith("mlp")}
    print(f"[xfmt] {name}: mean index {mean_idx:+.4f} over {len(results)} nodes")
    print("[xfmt] mlp index by layer: " +
          " ".join(f"L{k.split(':')[1]}:{v:+.3f}" for k, v in sorted(
              mlp_by_layer.items(), key=lambda kv: int(kv[0].split(':')[1]))))
    print("[xfmt] top-12 nodes by matched-minus-mismatched:")
    for node, v in ranked[:12]:
        print(f"[xfmt]   {node:<12} index {v['index']:+.4f} "
              f"(matched {v['matched']:+.3f} vs mismatched {v['mismatched']:+.3f})")

    out = args.out or f"xfmt_{name.replace('/', '_')}.json"
    Path(out).write_text(json.dumps({"model": name, "n": args.n,
                                     "mean_index": mean_idx, "nodes": results}, indent=2))
    print(f"[xfmt] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
