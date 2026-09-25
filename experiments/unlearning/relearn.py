"""Relearning fine-tune: the CHILD of an unlearning parent (PLAN.md §5, stage 2).

Mirrors train_sft.py's conventions (YAML config, frozen-parquet hash sidecar,
label-masked SFT loss through geode.train.sft's mask path, eps/k convergence
on a validation loss, AdamW at constant lr, --confirm-cost with a printed
estimate, run dir under $GEODE_STORE/runs/<run_id>) and adds what the
child-based metrics need, which train_sft.py does not record:

  train_log.jsonl   per step: train_loss_nats (computed BEFORE the update on that
                    batch, so the first-pass mean is the prequential code length
                    of the relearning set, M17), grad_norm (pre-clip, M5),
                    travel = ||theta_t - theta_0||_F and its relative size, speed
                    = ||theta_t - theta_{t-1}||_F (M6; exact fp32, no snapshots)
  eval_log.jsonl    every eval_every steps: val_loss_nats (paraphrased
                    questions of the relearned facts), and the probe read-out on
                    each eval split (hidden preference + first-token top-1 on
                    forget_A = relearned facts and forget_B = held-out authors:
                    the recovery curve, M17)
  snapshots/step_k  bf16 save_pretrained at a log-spaced schedule (M4 formation
                    curve: circuit_nodes.py --model <snapshot dir>)
  model/            final weights in fp32 (exact deltas for weight_shift.py, M6)
  manifest.json     config, init, result, git commit, cost estimate

Usage:
  python3 relearn.py --config configs/relearn_forgetA.yaml --init <hub id or dir> \
      --run-id relearn-npo-forgetA --data-dir data/built [--device cuda] --confirm-cost
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "training-run" / "analysis"))

from geode.arith.spans import tokenize_with_spans  # noqa: E402
from geode.edl.masking import TaskFormat  # noqa: E402
from geode.train.sft import _mean_masked_ce_nats, _padded_inputs_and_mask, evaluate_sft_nll_nats  # noqa: E402
from geode.train.stopping import ConvergenceTracker, StoppingRule  # noqa: E402

TASK_FORMAT = TaskFormat(name="tofu_qa", format_version="llama3_chat_v1")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_split(path: Path) -> pd.DataFrame:
    side = path.with_suffix(".sha256")
    if side.is_file():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != side.read_text().strip():
            raise SystemExit(f"[relearn] {path.name}: sha256 mismatch with its sidecar")
    return pd.read_parquet(path)


def examples_of(df: pd.DataFrame, tokenizer):
    return tokenize_with_spans(df["full_text"].tolist(),
                               list(zip(df["answer_char_start"].astype(int),
                                        df["answer_char_end"].astype(int))),
                               tokenizer, append_eos=True)


def snapshot_schedule(max_steps: int, n: int) -> list[int]:
    """~log-spaced steps 1..max_steps (the final model is saved separately)."""
    if n <= 0:
        return []
    out = sorted({max(1, round(math.exp(math.log(max_steps) * i / max(1, n - 1)))) for i in range(n)})
    return [s for s in out if s < max_steps]


@torch.no_grad()
def probe_readout(model, tokenizer, task_items: dict, device: str, bs: int) -> dict:
    """Hidden preference + first-token top-1 per eval split (PLAN.md M12/M17)."""
    out = {}
    was = model.training
    model.eval()
    pad = tokenizer.pad_token_id
    for split, items in task_items.items():
        ld_all, hit = [], 0
        for s in range(0, len(items), bs):
            chunk = items[s : s + bs]
            T = max(len(it.prompt_ids) for it in chunk)
            ids = torch.tensor([[pad] * (T - len(it.prompt_ids)) + it.prompt_ids for it in chunk],
                               device=device)
            am = (ids != -1).long()
            for j, it in enumerate(chunk):
                am[j, : T - len(it.prompt_ids)] = 0
            z = model(input_ids=ids, attention_mask=am).logits[:, -1].float()
            for j, it in enumerate(chunk):
                ld_all.append((z[j, it.target] - z[j, it.distractors[0]]).item())
                hit += int(z[j].argmax().item() == it.target)
        out[split] = {"logit_diff": sum(ld_all) / max(1, len(ld_all)), "top1": hit / max(1, len(items)),
                      "n": len(items)}
    if was:
        model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--init", required=True, help="parent: hub id or checkpoint dir")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="prepare.py --out-dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None, help="override (smoke)")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    t = cfg["train"]
    if args.max_steps is not None:
        t["max_steps"] = args.max_steps
        t["stopping"]["min_steps"] = min(t["stopping"].get("min_steps", 0), args.max_steps)
    store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
    run_dir = store / "runs" / args.run_id

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.init)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_df = load_split(args.data_dir / cfg["data"]["train_file"])
    val_df = load_split(args.data_dir / cfg["data"]["val_file"])
    train_ex, val_ex = examples_of(train_df, tokenizer), examples_of(val_df, tokenizer)
    max_len = max(len(e.input_ids) for e in train_ex)
    print(f"[relearn] {args.run_id}: train {len(train_ex)} rows (max {max_len} tokens), "
          f"val {len(val_ex)}, init {args.init}")

    model = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    gpu = cfg["gpu"]
    steps_cap = t["max_steps"]
    flops = 6.0 * n_params * t["batch_size"] * max_len * steps_cap
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    est = hours * gpu["usd_per_hour"]
    print(f"[relearn] {n_params / 1e6:.1f}M params, <= {steps_cap} steps x batch {t['batch_size']}: "
          f"estimated cost ${est:,.2f} ({hours:.3f} GPU-h @ ${gpu['usd_per_hour']}/h, excl. snapshots/evals)")
    if not args.confirm_cost:
        print("[relearn] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    # probe items for the recovery curve (task adapter, same items as the metrics)
    task_items = {}
    if cfg.get("probe_splits"):
        from task_adapter import QATask

        for split in cfg["probe_splits"]:
            task_items[split] = QATask(args.data_dir, split).items(tokenizer, cfg.get("probe_n"))

    torch.manual_seed(t["seed"])
    dev = args.device
    model.to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], betas=tuple(t["betas"]), weight_decay=t["weight_decay"])
    theta0 = [p.detach().clone() for p in params]
    norm0 = math.sqrt(sum(float((p.double() ** 2).sum()) for p in theta0))
    prev = [p.detach().clone() for p in params]
    tracker = ConvergenceTracker(StoppingRule(eps_nats=t["stopping"]["eps_nats"], k=t["stopping"]["k"],
                                              min_steps=t["stopping"].get("min_steps", 0)))
    ids_all, mask_all = _padded_inputs_and_mask(train_ex, TASK_FORMAT)
    snaps = set(snapshot_schedule(steps_cap, cfg.get("snapshots", 0)))
    precision = t.get("precision", "bf16") if not dev.startswith("cpu") else "fp32"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": args.run_id, "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "git_commit": git_commit(), "init": args.init, "config": cfg, "device": dev,
                "precision": precision, "trainable_param_count": n_params, "theta0_norm": norm0,
                "snapshot_steps": sorted(snaps), "cost": {"est_usd": est}, "status": "running",
                "training": {"method": "full_ft", "lora": None}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    step0 = {"val_loss_nats": evaluate_sft_nll_nats(model, val_ex, TASK_FORMAT, batch_size=t["batch_size"], device=dev)}
    if task_items:
        step0["probe"] = probe_readout(model, tokenizer, task_items, dev, t["batch_size"])
    print(f"[relearn] step 0: {json.dumps(step0)}")
    stop_reason, step, epoch_losses, first_pass = None, 0, [], None
    n, bs = len(train_ex), t["batch_size"]
    g = torch.Generator().manual_seed(t["seed"])
    t_start = time.time()
    with (run_dir / "train_log.jsonl").open("w") as tf, (run_dir / "eval_log.jsonl").open("w") as ef:
        ef.write(json.dumps({"step": 0, **step0}) + "\n")
        epoch = 0
        while stop_reason is None:
            perm = torch.randperm(n, generator=g)
            batches = [perm[i : i + bs] for i in range(0, n - bs + 1, bs)] or [perm]
            for bidx in batches:
                ids, mask = ids_all[bidx].to(dev), mask_all[bidx].to(dev)
                opt.zero_grad(set_to_none=True)
                if precision == "bf16":
                    with torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16):
                        loss = _mean_masked_ce_nats(model(ids).logits.float(), ids, mask)
                else:
                    loss = _mean_masked_ce_nats(model(ids).logits, ids, mask)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"]))
                opt.step()
                step += 1
                epoch_losses.append(loss.item())
                with torch.no_grad():
                    tr = math.sqrt(sum(float(((p - p0).double() ** 2).sum()) for p, p0 in zip(params, theta0)))
                    sp = math.sqrt(sum(float(((p - q).double() ** 2).sum()) for p, q in zip(params, prev)))
                    for p, q in zip(params, prev):
                        q.copy_(p)
                tf.write(json.dumps({"step": step, "epoch": epoch, "train_loss_nats": loss.item(),
                                     "grad_norm": gn, "travel": tr, "rel_travel": tr / max(norm0, 1e-30),
                                     "speed": sp, "lr": t["lr"], "time_unix": time.time()}) + "\n")
                tf.flush()
                if step in snaps:  # a bf16 COPY of the weights; the fp32 masters stay untouched
                    sd = run_dir / "snapshots" / f"step_{step}"
                    model.save_pretrained(sd, safe_serialization=True, max_shard_size="40GB",
                                          state_dict={k: v.detach().to(torch.bfloat16)
                                                      for k, v in model.state_dict().items()})
                    tokenizer.save_pretrained(sd)
                if step % t["eval_every"] == 0 or step == steps_cap:
                    rec = {"step": step, "val_loss_nats": evaluate_sft_nll_nats(
                        model, val_ex, TASK_FORMAT, batch_size=bs, device=dev), "time_unix": time.time()}
                    if task_items:
                        rec["probe"] = probe_readout(model, tokenizer, task_items, dev, bs)
                    ef.write(json.dumps(rec) + "\n")
                    ef.flush()
                    print(f"[relearn] step {step}: train {loss.item():.4f} val {rec['val_loss_nats']:.4f}"
                          + "".join(f"  {k}: ld {v['logit_diff']:+.2f} top1 {v['top1']:.2f}"
                                    for k, v in rec.get("probe", {}).items()), flush=True)
                    if tracker.update(rec["val_loss_nats"], step=step):
                        stop_reason = "converged"
                if step >= steps_cap and stop_reason is None:
                    stop_reason = "max_steps"
                if stop_reason:
                    break
            if first_pass is None:
                first_pass = sum(epoch_losses) / len(epoch_losses)
            epoch += 1
    wall = time.time() - t_start
    model.save_pretrained(run_dir / "model", safe_serialization=True, max_shard_size="40GB")
    tokenizer.save_pretrained(run_dir / "model")
    manifest.update({"status": "complete", "result": {
        "final_step": step, "stop_reason": stop_reason, "epochs": epoch,
        "first_pass_mean_loss_nats": first_pass, "min_val_nats": tracker.min_nats,
        "best_val_nats": tracker.best_nats, "wall_s": wall}})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[relearn] {args.run_id} done: {stop_reason} at step {step} ({epoch} epochs, {wall / 60:.1f} min); "
          f"prequential first-pass loss {first_pass:.4f} nats/token; min val {tracker.min_nats:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
