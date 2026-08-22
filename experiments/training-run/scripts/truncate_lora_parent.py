"""Build a layer-truncated-adapter parent from a converged LoRA target run.

A synthetic θ0 for the ts38tr family (positive control): a model that has
the trained LoRA machinery of a converged target run merged into its first
``keep_blocks`` transformer blocks, and the UNTRAINED base function in every
block from ``keep_blocks`` on — a model with a decodable latent answer at
some depth and no readout at the output, by construction rather than by
training dynamics.

Pipeline: ``geode.zoo.load_model`` reloads the source LoRA run (rebuilds the
``apply_lora`` module tree, bit-exact per specs/00 V0.9), ``zero_lora_blocks``
zeros ``B`` in every block ``>= keep_blocks`` so those blocks compute exactly
the base function, ``geode.train.lora.merge_lora`` folds every remaining
adapter into its base ``nn.Linear`` (a no-op fold for the zeroed blocks,
since ``scaling * (0 @ A) == 0``), and the result is saved as a plain
``save_pretrained`` directory — loadable by ``mech_lib.load_any_model
("dir:...")`` and by ``train_target.py --init-from`` exactly like any other
full-FT parent (never via ``geode.zoo.load_model``'s LoRA path again; the
merged checkpoint carries no base/A/B keys).

A new run is registered (``training.method: full_ft``, ``status: complete``,
no ``experiment.gates``) so the ts38tr target configs can point
``experiment.parent_run_id`` at it and run ``require_parent_ready`` cleanly
with ``parent_required_gates: []``.

Usage:
    python3 truncate_lora_parent.py --source-run-id evt-ts38mt-base-n316228 \\
        --keep-blocks 7 --out-run-id evt-ts38tr-k7-parent \\
        [--store <dir>] [--device cpu]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from geode.train.lora import LoRALinear, merge_lora
from geode.zoo import RunManifest, load_model, load_run, register_run
from geode.zoo.store import run_dir
from train import REPO_ROOT, git_commit


def zero_lora_blocks(model: nn.Module, keep_blocks: int) -> dict[str, int]:
    """Zero every ``LoRALinear.B.weight`` in blocks ``>= keep_blocks``, in place.

    ``model`` must be an ``apply_lora``-wrapped Llama model (``model.model.
    layers[i]`` holding the wrapped ``nn.Linear``s) — NOT enforced here (an
    unwrapped model silently returns ``{"zeroed": 0, "kept": 0}``); the
    immediately-following ``merge_lora`` call is what raises loudly on an
    unwrapped model, and ``main()`` refuses further upstream on a non-LoRA
    source before either function runs. A zeroed ``B`` makes ``forward(x) =
    base(x) + scaling * B(A(x)) == base(x)`` exactly for that module — so a
    later ``merge_lora`` folds in exactly zero for those blocks. Blocks
    ``< keep_blocks`` are left untouched. Embeddings/lm_head are never
    LoRA-wrapped by ``apply_lora`` and are not touched here either way.

    Refuses ``keep_blocks < 0`` or ``keep_blocks > num_hidden_layers``.
    Returns ``{"zeroed": n, "kept": m}`` — counts of wrapped modules zeroed
    vs. left alone, across every block.
    """
    layers = model.model.layers
    n_layers = len(layers)
    if keep_blocks < 0 or keep_blocks > n_layers:
        raise ValueError(
            f"zero_lora_blocks: keep_blocks must be in [0, {n_layers}] "
            f"(num_hidden_layers), got {keep_blocks}"
        )
    zeroed = 0
    kept = 0
    for i, block in enumerate(layers):
        for module in block.modules():
            if isinstance(module, LoRALinear):
                if i >= keep_blocks:
                    with torch.no_grad():
                        module.B.weight.zero_()
                    zeroed += 1
                else:
                    kept += 1
    return {"zeroed": zeroed, "kept": kept}


def truncated_parent_manifest(
    source: dict,
    out_run_id: str,
    keep_blocks: int,
    n_blocks: int,
    counts: dict,
    *,
    n_params: int,
) -> dict[str, Any]:
    """A ``RunManifest``-valid manifest dict for the truncated-adapter parent.

    ``source`` is the source LoRA run's manifest ``data``. ``training.method``
    is ``"full_ft"`` (the on-disk checkpoint is a plain merged ``nn.Linear``
    tree, never LoRA-wrapped again) and ``status`` is ``"complete"`` — this
    script performs no training, so there is nothing to leave "running".
    No ``experiment.gates`` are recorded (an empty dict): the ts38tr target
    configs declare ``parent_required_gates: []``, and ``require_parent_ready``
    reads an absent/empty gates dict as "nothing to check", never a failure.

    ``trainable_param_count`` is ``n_params`` — the caller's count of the
    MERGED model's own parameters. Note this is the SAME scalar whether you
    call it "total params of the merged model" or "the source's full param
    count": ``zero_lora_blocks``/``merge_lora`` fold weights in place and
    never change any tensor's shape, so merging never changes how many
    parameters the model has, unlike the source manifest's own
    ``trainable_param_count`` (which for a LoRA run counts only the adapter's
    r·(d_in+d_out) tensors, not the model).

    ``task``/``dataset`` are copied from ``source`` (provenance — this parent
    computes no loss on any data, but the schema requires non-null values and
    the honest answer is "whatever the source run trained on"). ``base_model``
    records the source run itself (``zoo-run/<source_run_id>``), matching the
    convention every other cross-stage parent handoff in this repo uses
    (e.g. ``train_sft.py``'s ``manifest_fields``). ``regime`` is ``"unknown"``:
    per ``train_sft.py``'s own comment, "regimes attach to target runs", not
    to a parent checkpoint.

    The extra top-level ``truncation`` block ``{source_run_id, keep_blocks,
    n_blocks, zeroed_modules, kept_modules, method}`` is preserved verbatim
    by the manifest schema (unknown top-level keys are permitted, spec 00
    V0.2) and is also written standalone as ``truncation.json`` next to the
    manifest by ``main()``.
    """
    source_run_id = source["run_id"]
    return {
        "schema_version": 1,
        "run_id": out_run_id,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": git_commit(),
        "regime": "unknown",
        "base_model": {"hf_id": f"zoo-run/{source_run_id}", "revision": "none"},
        "task": {
            "name": source["task"]["name"],
            "format_version": source["task"]["format_version"],
        },
        "dataset": {
            "name": source["dataset"]["name"],
            "n_unique_examples": source["dataset"]["n_unique_examples"],
            "seed": source["dataset"]["seed"],
        },
        "training": {
            "method": "full_ft",
            "lora": {
                "rank": None,
                "alpha": None,
                "target_modules": [],
                "dropout": None,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "none",
                "lr": 0.0,
                "batch_size": 1,
                "micro_batch_size": None,
                "betas": None,
                "weight_decay": 0.0,
                "grad_clip": None,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": None,
            "max_steps": None,
            "stopping": {"eps_nats": None, "k": None, "min_steps": None},
            "epochs_total": 0,
            "seed": 0,
        },
        "trainable_param_count": n_params,
        "snapshot_steps": [],
        "cost": {"gpu_type": None, "est_usd": None, "actual_usd": None},
        "status": "complete",
        "experiment": {
            "name": "training-run",
            "role": "pre_teach",
            "parent_run_id": source_run_id,
            "parent_required_gates": [],
            "gates": {},
        },
        "truncation": {
            "source_run_id": source_run_id,
            "keep_blocks": keep_blocks,
            "n_blocks": n_blocks,
            "zeroed_modules": counts["zeroed"],
            "kept_modules": counts["kept"],
            "method": "zero_B_then_merge",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True, help="a completed LoRA target run")
    parser.add_argument("--keep-blocks", type=int, required=True)
    parser.add_argument("--out-run-id", required=True)
    parser.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    store = (
        args.store
        if args.store is not None
        else Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
    )
    out_dir = run_dir(args.out_run_id, store=store)
    if out_dir.exists():
        print(f"[evt] {out_dir} already exists — refusing to overwrite. Exiting.")
        return 1

    source_manifest = load_run(args.source_run_id, store=store)
    method = source_manifest.data["training"]["method"]
    if method != "lora":
        print(
            f"[evt] source run '{args.source_run_id}' has training.method={method!r}, "
            "not 'lora' — truncate_lora_parent only operates on a converged LoRA target "
            "run (it has nothing to zero/merge otherwise). Exiting."
        )
        return 1

    print(f"[evt] loading {args.source_run_id} via geode.zoo.load_model ...", flush=True)
    model = load_model(args.source_run_id, store=store, device=args.device)
    n_blocks = len(model.model.layers)

    counts = zero_lora_blocks(model, keep_blocks=args.keep_blocks)
    print(
        f"[evt] keep_blocks={args.keep_blocks} of {n_blocks}: "
        f"zeroed {counts['zeroed']} wrapped modules, kept {counts['kept']}",
        flush=True,
    )
    merge_lora(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[evt] merged: {n_params / 1e6:.1f}M total params", flush=True)

    fields = truncated_parent_manifest(
        source_manifest.data,
        args.out_run_id,
        args.keep_blocks,
        n_blocks,
        counts,
        n_params=n_params,
    )
    RunManifest(data=fields).validate()  # fail before any disk write, not after

    model_dir = out_dir / "model"
    model.save_pretrained(str(model_dir))
    print(f"[evt] truncated+merged checkpoint written to {model_dir}", flush=True)

    truncation_path = out_dir / "truncation.json"
    truncation_path.write_text(json.dumps(fields["truncation"], indent=2) + "\n")
    print(f"[evt] truncation sidecar written to {truncation_path}", flush=True)

    manifest = register_run(fields, store=store)
    print(f"[evt] registered {args.out_run_id}: {manifest.data['truncation']}")
    print(f"[evt] store={store}  out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
