"""Offline probe-extraction driver (spec 02 §7 "Extraction pass").

Protocol-exempt glue, mirroring ``train_target.py``: parses the run YAML (for
the tokenizer, task format, probe-set source, and GPU cost block), loads the
frozen probe parquet and records its ``probe_set_hash`` (``geode.arith
.order_hash`` — the hash pinned at generation in ``data/full/report.json``;
``--probe-hash`` makes a mismatch a refusal), loads the run manifest, iterates
``manifest.snapshot_steps`` ∩ the snapshots on disk (scheduled steps past the
actual stop never materialized — spec 02 §6), rebuilds the model at each
snapshot (base module tree from the run's final-checkpoint config + the pinned
``geode.train.apply_lora`` wrap for LoRA runs, plain state for full-FT; state
loaded via ``geode.edl.load_snapshot`` — once-per-run base + per-step adapter,
with legacy full snapshots still supported), runs
the probe pass (``geode.probe.extract_probe`` — all learning-free math lives
there), and writes one dump per snapshot to ``runs/{run_id}/probe/step_{k}``
via ``geode.probe.save_probe_dump``. Existing dumps are skipped unless
``--overwrite`` (resumable). Extraction runs on already-paid-for snapshots but
still costs GPU time at scale, so it prints an estimate and refuses without
``--confirm-cost`` (CLAUDE.md budget rule). This file must stay thin.

Usage:
    python3 extract.py --config configs/run5_target.yaml --run-id evt-run5-armA-target \
        [--probe-file probe.parquet] [--probe-local <path>] [--probe-hash <sha256>] \
        [--device cuda] [--store <dir>] [--overwrite] [--confirm-cost]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from geode.arith import order_hash, tokenize_with_spans
from geode.edl import load_snapshot
from geode.edl.masking import TaskFormat
from geode.probe import ProbeMeta, extract_probe, probe_dump_dir, save_probe_dump
from geode.train import apply_lora
from geode.zoo import checkpoint_dir, load_run, tokenizer_hash
from geode.zoo.store import run_dir
from train import REPO_ROOT, load_config, phase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="run whose snapshots to extract")
    parser.add_argument("--probe-file", default="probe.parquet", help="file in data.hf_id")
    parser.add_argument("--probe-local", type=Path, default=None, help="skip the HF download")
    parser.add_argument("--probe-hash", default=None, help="refuse on probe_set_hash mismatch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--store", type=Path, default=None, help="override $GEODE_STORE")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="extract at most N snapshots, evenly index-spaced over the available "
        "list (first + final always kept) — the on-box disk cap: one dump is "
        "~0.5 GiB, full density is ~0.6 TiB/run",
    )
    parser.add_argument("--overwrite", action="store_true", help="re-extract existing dumps")
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    phase(1, "config + tokenizer")
    cfg = load_config(args.config, None)
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    task_format = TaskFormat(name=cfg["task"]["name"], format_version=cfg["task"]["format_version"])
    tok_hash = tokenizer_hash(tokenizer)

    phase(2, "probe set — frozen parquet, hash recorded, tokenize")
    import pandas as pd

    if args.probe_local is not None:
        path = args.probe_local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(cfg["data"]["hf_id"], args.probe_file, repo_type="dataset")
    df = pd.read_parquet(path)
    probe_hash = order_hash(df.to_dict("records"))
    if args.probe_hash and probe_hash != args.probe_hash:
        raise ValueError(
            f"probe set {path}: probe_set_hash {probe_hash} != pinned {args.probe_hash} — "
            "wrong or corrupted frozen probe file; refusing to proceed"
        )
    char_spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    examples = tokenize_with_spans(df["full_text"].tolist(), char_spans, tokenizer, append_eos=True)
    max_len = max(len(ex.input_ids) for ex in examples)
    print(
        f"[evt] probe set: {len(examples)} examples, max {max_len} tokens, "
        f"probe_set_hash={probe_hash[:12]}…",
        flush=True,
    )

    phase(3, "manifest + snapshots on disk")
    if args.store is not None:
        os.environ["GEODE_STORE"] = str(args.store)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])
    manifest = load_run(args.run_id, store=store)
    declared = set(manifest.data["snapshot_steps"])
    snap_root = run_dir(args.run_id, store=store) / "snapshots"
    on_disk = {
        int(p.name.removeprefix("step_"))
        for p in snap_root.glob("step_*")
        # adapter-only format (2026-07-22) or legacy full snapshot — both load
        if (p / "adapter.safetensors").is_file() or (p / "model.safetensors").is_file()
    }
    steps = sorted(declared & on_disk)
    if not steps:
        raise FileNotFoundError(
            f"no extractable snapshots for {args.run_id}: {len(declared)} declared, "
            f"{len(on_disk)} on disk under {snap_root}, intersection empty"
        )
    print(
        f"[evt] snapshots: {len(declared)} declared ∩ {len(on_disk)} on disk "
        f"= {len(steps)} to extract (steps {steps[0]}…{steps[-1]})",
        flush=True,
    )
    if on_disk - declared:
        print(f"[evt] WARNING: {len(on_disk - declared)} on-disk snapshots are undeclared; skipped")
    if args.limit is not None and args.limit < len(steps):
        if args.limit < 2:
            raise ValueError(f"--limit must be >= 2 (first + final step), got {args.limit}")
        idx = {round(i * (len(steps) - 1) / (args.limit - 1)) for i in range(args.limit)}
        steps = [steps[i] for i in sorted(idx)]
        print(
            f"[evt] --limit {args.limit}: {len(steps)} snapshots selected, evenly "
            f"index-spaced over the available list (keeps the schedule's "
            f"log-then-uniform shape)",
            flush=True,
        )

    phase(4, "model skeleton — final-checkpoint config + snapshot state per step")
    arch = AutoConfig.from_pretrained(checkpoint_dir(args.run_id, store=store))
    model = LlamaForCausalLM(arch)
    method = manifest.data["training"]["method"]
    if method == "lora":
        lora = manifest.data["training"]["lora"]
        # Same module tree the harness saved (V5.51): every tensor comes from
        # the snapshot, so the throwaway init seed is irrelevant.
        model = apply_lora(
            model,
            rank=lora["rank"],
            alpha=lora["alpha"],
            seed=0,
            target_modules=tuple(lora["target_modules"]),
        )
    elif method != "full_ft":
        raise ValueError(f"unsupported training.method {method!r} (expected 'lora' or 'full_ft')")
    n_params = sum(p.numel() for p in model.parameters())

    phase(5, "cost estimate + confirm gate")
    gpu = cfg["gpu"]
    # Forward + activation backward per snapshot over the padded probe batch;
    # 6·N·tokens is the training-step FLOPs ceiling, conservative here.
    flops = 6.0 * n_params * (len(examples) * max_len) * len(steps)
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    est_usd = hours * gpu["usd_per_hour"]
    print(
        f"[evt] {len(steps)} snapshots x {len(examples)} probe examples on "
        f"{n_params / 1e6:.1f}M params"
    )
    print(f"[evt] estimated cost: ${est_usd:,.2f}  ({hours:.2f} GPU-h @ ${gpu['usd_per_hour']}/h)")
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to extract (budget rule). Exiting.")
        return 1

    phase(6, "extract per snapshot")
    experiment = manifest.data.get("experiment") or {}
    arm = experiment.get("arm") or manifest.data["regime"]
    base_model_key = manifest.data["base_model"]["hf_id"]
    n_done = n_skipped = 0
    for k in steps:
        out = probe_dump_dir(args.run_id, k, store=store)
        if (out / "meta.json").is_file() and not args.overwrite:
            n_skipped += 1
            continue
        load_snapshot(model, args.run_id, k, store=store)
        capture = extract_probe(model, examples, task_format, device=args.device)
        meta = ProbeMeta(
            run_id=args.run_id,
            arm=str(arm),
            step=k,
            probe_set_hash=probe_hash,
            tokenizer_hash=tok_hash,
            base_model_key=base_model_key,
            task_name=task_format.name,
            format_version=task_format.format_version,
        )
        save_probe_dump(capture, meta, store=store)
        n_done += 1
        print(f"[evt] step {k}: probe dump -> {out}", flush=True)

    print(f"[evt] done: {n_done} extracted, {n_skipped} skipped (already dumped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
