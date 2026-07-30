"""Merge a LoRA install run's adapter into a plain checkpoint (cross-stage handoff).

A LoRA-trained run's final checkpoint is method-wrapped (base + A/B tensors,
specs/00 §1) and must only ever be loaded via ``geode.zoo.load_model`` — a
plain ``from_pretrained`` on it silently random-inits every wrapped
projection (the 2026-07-22 G5 incident, specs/00 V0.9). When such a run is
itself the PARENT of a later stage (e.g. run 9's LoRA format install feeding
run 10's target training), the child must warm-start from an ordinary
``from_pretrained`` checkpoint, so the parent's adapter has to be folded into
its base weights first. This script does exactly that, via
``geode.train.merge_lora`` (V5.52), and writes the merged checkpoint to
``runs/<run-id>/model_merged/`` (specs/00 §1) plus an
``experiment.merged_checkpoint`` manifest entry recording where it landed.

Refuses if ``model_merged/`` already exists (no silent overwrite) or if the
run's checkpoint is not LoRA-wrapped (``merge_lora`` raises).

Usage:
    python3 merge_adapter.py --run-id evt-run9-llama1b-inst
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from geode.train import merge_lora
from geode.zoo import load_model, load_run

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
        help="artifact store root (default: $GEODE_STORE, else <repo>/geode-store)",
    )
    args = parser.parse_args()

    run_dir = args.store / "runs" / args.run_id
    merged_dir = run_dir / "model_merged"
    if merged_dir.exists():
        print(f"[evt] {merged_dir} already exists — refusing to overwrite. Exiting.")
        return 1

    print(f"[evt] loading {args.run_id} via geode.zoo.load_model ...", flush=True)
    model = load_model(args.run_id, store=args.store, device=args.device)
    merge_lora(model)
    model.save_pretrained(str(merged_dir))
    print(f"[evt] merged checkpoint written to {merged_dir}", flush=True)

    manifest = load_run(args.run_id, store=args.store)
    manifest.data.setdefault("experiment", {})["merged_checkpoint"] = "model_merged"
    manifest.save(run_dir / "manifest.json")
    print("[evt] manifest updated: experiment.merged_checkpoint = 'model_merged'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
