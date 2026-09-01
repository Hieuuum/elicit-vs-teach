"""Circuit-formation dynamics from a run's trajectory snapshots.

The snapshot payoff (owner 2026-08-24): compute the attribution node map at
intermediate training snapshots of an endpoint run and track how the FINAL
circuit assembles. Predictions that separate the regimes:

- ELICIT (Llama endpoints): the final circuit is present almost immediately —
  Jaccard-to-final high from the first snapshots (reuse: the machinery
  predates training).
- TEACH (TinyStories endpoints): the circuit CRYSTALLIZES over training —
  Jaccard-to-final low early, rising through the region where EDL/token
  peaked (the teaching hump is the circuit being built).

Snapshots live on the per-run HF repos (streamed there at train time and
deleted locally); this script downloads exactly the selected steps (plus
``snapshots/base/``) back into the store layout, then rebuilds θ_step via
``geode.edl.load_snapshot`` (bit-exact, L-5) on the zoo-loaded module tree
and runs the shared attribution core. ~0.72 GB per snapshot fetched;
downloads are left in the store for re-use (delete manually when done).

Output: <out>.parquet with one row per step (jaccard@{32,64} vs the FINAL
map, union-score Spearman, sanity logit-diff) + per-step map parquets
<out>_step<N>.parquet. GPU, box-only.

Usage:
    python3 circuit_trajectory.py --run-id <rid> --final-map <stem> \
        [--repo-id <ns>/<rid>] [--n-snapshots 8] [--n-pairs 128]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "training-run" / "scripts"))

from circuit_nodes import EVAL_CONFIG, attribution_map, build_pairs  # noqa: E402
from train import load_config  # noqa: E402
from train_sft import load_frozen_parquet  # noqa: E402

STEP_RE = re.compile(r"runs/[^/]+/snapshots/step_(\d+)/adapter\.safetensors$")


def pick_steps(steps: list[int], n: int) -> list[int]:
    """~log-spaced subset of the available snapshot steps, endpoints included."""
    if len(steps) <= n:
        return steps
    import math

    lo, hi = math.log(max(steps[0], 1)), math.log(steps[-1])
    targets = [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    picked = sorted({min(steps, key=lambda s: abs(s - t)) for t in targets})
    return picked


def ensure_snapshot(store: Path, run_id: str, repo_id: str, step: int | None) -> None:
    """Fetch snapshots/base or snapshots/step_<step> from the hub if missing."""
    from huggingface_hub import hf_hub_download

    rel = ("snapshots/base/model.safetensors" if step is None
           else f"snapshots/step_{step}/adapter.safetensors")
    dst = store / "runs" / run_id / rel
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id, f"runs/{run_id}/{rel}")
    dst.write_bytes(Path(src).read_bytes())
    print(f"[traj] fetched {rel}")


def jaccard_and_rho(map_a: dict, map_b: dict, k: int):
    sa = pd.Series({key: abs(v) for key, v in map_a.items()})
    sb = pd.Series({key: abs(v) for key, v in map_b.items()})
    top_a, top_b = set(sa.nlargest(k).index), set(sb.nlargest(k).index)
    union = list(top_a | top_b)
    jacc = len(top_a & top_b) / len(union)
    rho = sa[union].rank().corr(sb[union].rank())
    return jacc, rho


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--final-map", required=True,
                    help="circuit_nodes stem of the CONVERGED model's map (the target)")
    ap.add_argument("--repo-id", default=None, help="default: podhajskimarcin/<run-id>")
    ap.add_argument("--n-snapshots", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: traj_<run-id>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    repo_id = args.repo_id or f"podhajskimarcin/{args.run_id}"
    out_stem = args.out or f"traj_{args.run_id}"
    store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))

    final_df = pd.read_parquet(Path(args.final_map).with_suffix(".parquet"))
    final_map = {(r.node_type, int(r.layer), int(r.head)): r.score for r in final_df.itertuples()}

    local_steps = sorted(
        int(d.name.split("_")[1])
        for d in (store / "runs" / args.run_id / "snapshots").glob("step_*")
        if (d / "adapter.safetensors").is_file()
    ) if (store / "runs" / args.run_id / "snapshots").is_dir() else []
    if local_steps:
        steps = local_steps
        print(f"[traj] using {len(steps)} LOCAL snapshots (no hub fetch)")
    else:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id)
        steps = sorted(int(m.group(1)) for f in files if (m := STEP_RE.search(f)))
    if not steps:
        raise SystemExit(f"[traj] no snapshots found locally or on {repo_id}")
    picked = pick_steps(steps, args.n_snapshots)
    print(f"[traj] {len(steps)} snapshots on {repo_id}; analyzing steps {picked}")

    ensure_snapshot(store, args.run_id, repo_id, None)  # snapshots/base
    for s in picked:
        ensure_snapshot(store, args.run_id, repo_id, s)

    cfg = load_config(EVAL_CONFIG, None)
    df = load_frozen_parquet(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs = build_pairs(df, tokenizer, args.n_pairs, shots=0)

    from geode.edl import load_snapshot
    from geode.zoo import load_model as zoo_load_model

    model = zoo_load_model(args.run_id, store=store, device=args.device)
    model.eval()

    rows = []
    for s in picked:
        load_snapshot(model, args.run_id, s, store=store)
        scores, sanity = attribution_map(model, pairs, args.batch_size, args.device)
        j32, _ = jaccard_and_rho(scores, final_map, 32)
        j64, rho = jaccard_and_rho(scores, final_map, 64)
        rows.append({"step": s, "jaccard32_vs_final": j32, "jaccard64_vs_final": j64,
                     "spearman_vs_final": rho, "mean_logit_diff": sanity})
        pd.DataFrame(
            [{"node_type": k, "layer": i, "head": h, "score": v, "abs_score": abs(v)}
             for (k, i, h), v in scores.items()]
        ).to_parquet(f"{out_stem}_step{s}.parquet", index=False)
        print(f"[traj] step {s:7d}: J@32 {j32:.3f}  J@64 {j64:.3f}  rho {rho:.3f}  "
              f"logit_diff {sanity:.2f}")

    pd.DataFrame(rows).to_parquet(f"{out_stem}.parquet", index=False)
    print(f"[traj] wrote {out_stem}.parquet ({len(rows)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
