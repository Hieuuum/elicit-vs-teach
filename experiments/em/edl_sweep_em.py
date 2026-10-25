# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""M19 on the EM datasets: the EDL/D-vs-n sweep (Donoway et al.'s signature, the protocol of
experiments/unlearning/relearn.py --sweep) on the label-check model and recipe.

For each size n (nested: the first n of one fixed permutation of the training pool), a fresh LoRA
adapter (Turner et al.'s r 32 rsLoRA, lr 1e-5 constant after 5 warm-up steps, batch 2 x 8) trains until
its validation loss stops improving (patience, max epochs, a step cap). Recorded per run:

  mdl_nats        the first-epoch prequential code: every batch's response-token loss BEFORE the update
                  on it, summed over the epoch (each example is seen once, unseen)
  val_floor_nats  the run's own minimum validation loss (the OCV floor of this repo)
  test_floor_nats loss on held-out test rows at the min-val weights (the paper's Eq. 3 floor)
  edl_*_per_token_nats = (mdl - D * floor) / D, D = the first-epoch response-token count

``--rotate`` trains the rotated-answer twin: every response moved to another prompt by one seeded cyclic
offset over the whole set (val and test rotated with it), so the format and vocabulary stay and the
pairing is arbitrary; EDL/D(true) - EDL/D(rotated) is the knowledge-specific curve (M19k), and the
rotated set is the teaching control (arbitrary pairings can only be memorised).

Writes $GEODE_STORE/em/edl/<model>/<dataset>[@rot]/n<n>-s<seed>.json (resumes: existing points are
skipped). analyze_edl_em.py reads the curves. Losses only; nothing is generated.

Usage (GPU; prints the estimate and refuses without --confirm-cost):
  python3 experiments/em/edl_sweep_em.py --dataset bad_medical --device cuda --confirm-cost [--rotate]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import em_store, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from train_em import batch_loss, encode_example, geode_alpha

DEFAULT_SIZES = "16,32,64,128,256,512,1024,4096"


def reset_lora(model, seed: int) -> None:
    """Put every LoRALinear back to apply_lora's initial state for ``seed``: A redrawn uniform
    +-1/sqrt(d_in) from one CPU generator in registration order, B zero (so the model is the base)."""
    from geode.train.lora import LoRALinear

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                d_in = m.base.in_features
                a = torch.empty(m.rank, d_in, dtype=torch.float32)
                a.uniform_(-1.0 / math.sqrt(d_in), 1.0 / math.sqrt(d_in), generator=g)
                m.A.weight.copy_(a)
                m.B.weight.zero_()


def rotate_responses(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, int]:
    """Every response moved to another prompt by one seeded cyclic offset in [1, n-1]."""
    n = len(df)
    k = int(torch.randint(1, n, (1,), generator=torch.Generator().manual_seed(seed + 7)))
    out = df.copy()
    r = df["response"].tolist()
    out["response"] = r[k:] + r[:k]
    return out.reset_index(drop=True), k


def split(
    df: pd.DataFrame, n_val: int, n_test: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One fixed permutation: test, then val, then the training pool (whose prefixes are the sizes)."""
    perm = torch.randperm(len(df), generator=torch.Generator().manual_seed(seed)).tolist()
    d = df.iloc[perm].reset_index(drop=True)
    return (
        d.iloc[n_test + n_val :].reset_index(drop=True),
        d.iloc[n_test : n_test + n_val].reset_index(drop=True),
        d.iloc[:n_test].reset_index(drop=True),
    )


@torch.no_grad()
def mean_loss(model, enc: list, device: str, pad: int, batch_size: int = 8) -> float:
    was = model.training
    model.eval()
    tot, tok = 0.0, 0
    for i in range(0, len(enc), batch_size):
        loss, n = batch_loss(model, enc[i : i + batch_size], device, pad)
        tot, tok = tot + float(loss), tok + n
    if was:
        model.train()
    return tot / max(1, tok)


def run_point(
    model,
    tok,
    train_enc: list,
    val_enc: list,
    test_enc: list,
    *,
    seed: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    warmup: int,
    max_epochs: int,
    patience: int,
    max_steps: int,
    device: str,
    test_rot_encs: list | None = None,
) -> dict:
    """One sweep point. ``test_rot_encs`` (the test rows with their responses rotated, one list per
    rotation) adds M21 on the held-out test rows at the min-val weights: the M21-n read."""
    reset_lora(model, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    g = torch.Generator().manual_seed(seed)
    per_step = batch_size * grad_accum
    mdl, d_tok, steps = 0.0, 0, 0
    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    vals: list[float] = []
    model.train()
    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(len(train_enc), generator=g).tolist()
        for s in range(0, len(order), per_step):
            chunk = order[s : s + per_step]
            n_tok = sum(len(train_enc[i][0]) - train_enc[i][1] for i in chunk)
            for m in range(0, len(chunk), batch_size):
                loss, ntok = batch_loss(
                    model, [train_enc[i] for i in chunk[m : m + batch_size]], device, pad
                )
                if epoch == 1:  # the loss before the update on this batch: the prequential code
                    mdl += float(loss.detach())
                    d_tok += ntok
                (loss / max(1, n_tok)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            if steps >= max_steps:
                break
        v = mean_loss(model, val_enc, device, pad)
        vals.append(v)
        if v < best_val - 1e-4:
            best_val, best_epoch, bad = v, epoch, 0
            best_state = [p.detach().clone() for p in params]
        else:
            bad += 1
        if bad >= patience or steps >= max_steps:
            break
    if best_state is not None:
        with torch.no_grad():
            for p, b in zip(params, best_state):
                p.copy_(b)
    test = mean_loss(model, test_enc, device, pad)
    rot = [mean_loss(model, e, device, pad) for e in (test_rot_encs or [])]
    d = max(1, d_tok)
    return {
        "m21_test_nats": sum(rot) / len(rot) - test if rot else None,
        "m21_rotated_loss_nats": rot,
        "n_train": len(train_enc),
        "seed": seed,
        "mdl_nats": mdl,
        "label_tokens": d_tok,
        "steps": steps,
        "epochs": len(vals),
        "best_epoch": best_epoch,
        "val_losses": vals,
        "val_floor_nats": best_val,
        "test_floor_nats": test,
        "edl_ocv_per_token_nats": (mdl - d_tok * best_val) / d,
        "edl_test_per_token_nats": (mdl - d_tok * test) / d,
        "mdl_per_token_nats": mdl / d,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument(
        "--rotate", action="store_true", help="the rotated-answer twin (teaching control, M19k)"
    )
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--seeds", default="316")
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument(
        "--m21-rotations", type=int, default=2, help="M21 on the test rows at every point (0 = off)"
    )
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--rslora-alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument(
        "--max-steps", type=int, default=768, help="per run (768 = 3 epochs of 4096 examples)"
    )
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens-per-sec", type=float, default=2500.0)
    ap.add_argument("--usd-per-gpu-hour", type=float, default=2.0)
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()

    store = em_store()
    df = pd.read_parquet(store / "data" / f"{args.dataset}.parquet")[["user", "response"]]
    offset = None
    if args.rotate:
        df, offset = rotate_responses(df, args.split_seed)
    pool, val, test = split(df, args.n_val, args.n_test, args.split_seed)
    sizes = [n for n in (int(x) for x in args.sizes.split(",")) if n <= len(pool)]
    seeds = [int(x) for x in args.seeds.split(",")]
    out_dir = (
        store / "edl" / args.model.split("/")[-1] / (args.dataset + ("@rot" if args.rotate else ""))
    )
    todo = [(n, s) for s in seeds for n in sizes if not (out_dir / f"n{n}-s{s}.json").is_file()]
    per_step = args.batch_size * args.grad_accum
    ex = sum(min(n * args.max_epochs, args.max_steps * per_step) for n, _ in todo)
    evals = (
        sum(
            min(args.max_epochs, math.ceil(args.max_steps / max(1, math.ceil(n / per_step))))
            for n, _ in todo
        )
        * args.n_val
    )
    hours = (ex * 400 / args.tokens_per_sec + evals * 400 / (3 * args.tokens_per_sec)) / 3600
    print(
        f"[edl] {args.dataset}{' (rotated)' if args.rotate else ''}: sizes {sizes}, seeds {seeds}, {len(todo)} points to run; "
        f"<= {ex:,} training examples: <= ~{hours:.1f} GPU-h, ~${hours * args.usd_per_gpu_hour:.0f} (early stopping makes it less)"
    )
    if not todo:
        return 0
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[edl] refusing to run on a GPU without --confirm-cost")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from geode.train.lora import apply_lora

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    apply_lora(
        model, rank=args.rank, alpha=geode_alpha(args.rank, args.rslora_alpha), seed=seeds[0]
    )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    def enc(d: pd.DataFrame) -> list:
        return [encode_example(tok, u, a, args.max_len) for u, a in zip(d["user"], d["response"])]

    pool_enc, val_enc, test_enc = enc(pool), enc(val), enc(test)
    from score_em import rotate, rotation_offsets

    test_rot_encs = []
    for k in (
        rotation_offsets(len(test), args.m21_rotations, args.split_seed)
        if args.m21_rotations
        else []
    ):
        t = test.copy()
        t["response"] = rotate(test["response"].tolist(), k)
        test_rot_encs.append(enc(t))
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = torch.autocast(
        device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16
    )
    for n, s in todo:
        t0 = time.time()
        with amp:
            r = run_point(
                model,
                tok,
                pool_enc[:n],
                val_enc,
                test_enc,
                seed=s,
                lr=args.lr,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                warmup=args.warmup,
                max_epochs=args.max_epochs,
                patience=args.patience,
                max_steps=args.max_steps,
                device=args.device,
                test_rot_encs=test_rot_encs,
            )
        r.update(
            {
                "dataset": args.dataset,
                "rotated": args.rotate,
                "rotation_offset": offset,
                "model": args.model,
                "seconds": time.time() - t0,
                "commit": commit,
                "config": {k: v for k, v in vars(args).items()},
            }
        )
        (out_dir / f"n{n}-s{s}.json").write_text(json.dumps(r, indent=2, default=str))
        print(
            f"[edl] n={n} seed {s}: {r['epochs']} epochs ({r['steps']} steps), MDL/D {r['mdl_per_token_nats']:.3f}, "
            f"floor {r['val_floor_nats']:.3f} -> EDL/D {r['edl_ocv_per_token_nats'] / math.log(2):+.3f} bits/token "
            f"(test floor {r['edl_test_per_token_nats'] / math.log(2):+.3f}), {(time.time() - t0) / 60:.1f} min"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
