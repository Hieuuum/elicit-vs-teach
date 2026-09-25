"""Metric 3 weight-space proxy: where does the fine-tuning update live?

Decomposes each run's LoRA delta ||B@A||_F by module class:

- **QK** (q_proj, k_proj) — attention ROUTING: changes which positions
  attend to which — the weight-space shadow of *edge rewiring*.
- **VO** (v_proj, o_proj) — what attended-to information is written.
- **MLP** (gate/up/down_proj) — feed-forward COMPUTATION — the shadow of
  *node content* change.

Predictions (Wang-et-al-style edge-vs-node framing, adapted): elicitation
(Llama on latent arithmetic) should concentrate its update in routing (QK
fraction high) — the computation exists, access is rewired; teaching
(TinyStories) must build computation (MLP/VO fraction high). Fractions are
scale-free (the LoRA alpha/r scaling cancels within a run), so runs are
comparable across families and dataset sizes.

Reads adapter.safetensors sidecars from the local store (kept for every
sweep run). Key structure is auto-detected: tensors are grouped by the
``layers.<i>.<block>.<proj>_proj`` fragment; a (A, B)-shaped pair under one
projection contributes ||B@A||_F, a single tensor contributes its own norm.
Unrecognized keys are listed loudly rather than silently dropped
(--list-keys to inspect). CPU-only.

Usage:
    python3 adapter_shift.py --families nl3 ts [--store <dir>] [--out <stem>]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]

FAMILY_PREFIX = {
    "nl2": "evt-llama-fig2nl2",
    "nl3": "evt-llama-fig2nl3",
    "ts": "evt-ts1b-fig2ts",
}
SIZES = [1000, 1468, 2154, 3162, 4642, 6813, 10000, 14678, 21544, 31623, 46416,
         68129, 100000, 146780, 215443, 316228, 464159, 681292, 1000000]
PROJ_RE = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj")
CLASS = {"q": "QK", "k": "QK", "v": "VO", "o": "VO", "gate": "MLP", "up": "MLP", "down": "MLP"}


def adapter_norms(path: Path, list_keys: bool = False) -> dict[str, float]:
    """{module_class: sum of ||delta W||_F} for one adapter file."""
    sd = load_file(path)
    if list_keys:
        for k, v in sd.items():
            print(f"  {k}  {tuple(v.shape)}")
    groups: dict[tuple, dict[str, torch.Tensor]] = {}
    unknown = []
    for key, t in sd.items():
        m = PROJ_RE.search(key)
        if m is None:
            unknown.append(key)
            continue
        groups.setdefault((int(m.group(1)), m.group(2)), {})[key] = t
    if unknown:
        print(f"[adapter] WARNING {path.parent.parent.name}: {len(unknown)} unmatched "
              f"key(s), e.g. {unknown[0]!r} — excluded from the decomposition")

    out: dict[str, float] = {"QK": 0.0, "VO": 0.0, "MLP": 0.0}
    for (_layer, proj), tensors in groups.items():
        ts = sorted(tensors.items())
        if len(ts) == 2:
            (ka, a), (kb, b) = ts
            # orient as (out, r) @ (r, in): the shared small dim is r
            a2, b2 = a.float(), b.float()
            if a2.shape[0] == b2.shape[1]:  # a is (r, in), b is (out, r)
                delta = b2 @ a2
            elif b2.shape[0] == a2.shape[1]:
                delta = a2 @ b2
            else:
                print(f"[adapter] WARNING: cannot orient {ka}/{kb} "
                      f"{tuple(a.shape)}x{tuple(b.shape)} — using norm product")
                out[CLASS[proj]] += (a2.norm() * b2.norm()).item()
                continue
            out[CLASS[proj]] += delta.norm().item()
        else:
            out[CLASS[proj]] += sum(t.float().norm().item() for t in tensors.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=["nl3", "ts"], choices=list(FAMILY_PREFIX))
    ap.add_argument("--store", type=Path,
                    default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")))
    ap.add_argument("--out", default=None, help="output stem (default: adapter_shift_<fams>)")
    ap.add_argument("--list-keys", action="store_true", help="dump one adapter's keys and exit")
    args = ap.parse_args()

    rows = []
    for fam in args.families:
        prefix = FAMILY_PREFIX[fam]
        for arm in ("noinst", "inst"):
            for n in SIZES:
                p = args.store / "runs" / f"{prefix}-{arm}-n{n}" / "model" / "adapter.safetensors"
                if not p.is_file():
                    print(f"[adapter] {p.parent.parent.name}: no sidecar — skipped")
                    continue
                if args.list_keys:
                    print(f"[adapter] keys of {p}:")
                    adapter_norms(p, list_keys=True)
                    return 0
                norms = adapter_norms(p)
                total = sum(norms.values()) or 1.0
                rows.append({"family": fam, "arm": arm, "n": n,
                             **{f"{k}_norm": v for k, v in norms.items()},
                             **{f"{k}_frac": v / total for k, v in norms.items()}})
                print(f"[adapter] {prefix}-{arm}-n{n}: "
                      + "  ".join(f"{k} {norms[k]/total:.3f}" for k in ("QK", "VO", "MLP")))

    if not rows:
        raise SystemExit("[adapter] no adapters found — check --store / --families")
    df = pd.DataFrame(rows)
    stem = args.out or f"adapter_shift_{'_'.join(args.families)}"
    out = Path(stem).with_suffix(".parquet")
    df.to_parquet(out, index=False)
    print(f"[adapter] wrote {out} ({len(df)} runs)")
    print("[adapter] family x arm means (fractions):")
    print(df.groupby(["family", "arm"])[["QK_frac", "VO_frac", "MLP_frac"]]
          .mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
