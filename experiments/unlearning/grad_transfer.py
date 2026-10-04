"""Gradient transfer at initialization (M20): a before-training predictor of elicit vs teach.

No weights are updated.  The question the relearning experiment answers by training ("does
fine-tuning on the A facts bring back the held-out B facts?") has a first-order answer at the
starting weights: a small step on the A loss changes the B loss by  dL_B = -lr * <grad L_A, grad L_B>.
A model whose knowledge is latent (elicit) has A- and B-gradients that share a direction, the one
that lifts the suppression; a model without the knowledge (teach) has A-gradients that say nothing
about B beyond the shared answer format.

Measured in the parametrization the fine-tune uses: the gradient w.r.t. the LoRA B factors at
their zero init (geode.train.lora, the relearn.py recipe), which is the full weight gradient
projected through the seeded random A factors (a Johnson-Lindenstrauss sketch of it), on every
projection of every layer.  Gradients are of each set's mean label-token loss (the SFT loss of the
relearning format: question, "Answer:", the correct answer text; no options).

Sets:  bioA   the relearning facts (half A)
       bioA_shuf[k]   the same prompts with the answers rotated across items by a seeded offset
                      (same items, same vocabulary, no knowledge): the NULL for transfer
       bioB   the held-out facts (half B, never trained on)
       mmluA  the far-domain fine-tuning null's facts (a second, topic-free null)
--task arith (the main results' calibration, 2026-10-04): the same quantities on the arithmetic
target, A = a seeded sample of the training file of --train-config, B = the frozen eval file's
reporting block (question-disjoint), the null = A with the answers rotated; no mmlu set.  The
TinyStories pair fixes both ends: the elicit parent (evt-ts1b-op-bridge-mix) should score high,
the format-installed parent (evt-ts1b-fig2ts-installer) near zero, before either is fine-tuned.

Reported (nats-free, all cosines):
  transfer      cos(G_bioA, G_bioB)
  null          cos(G_bioA_shuf, G_bioB), mean and sd over the shuffles
  score         transfer - null            (the predictor; 0 at a model that cannot know the facts)
  score_mmlu    transfer - cos(G_mmluA, G_bioB)
  descent_frac  <G_bioA, G_bioB> / <G_bioB, G_bioB>: the share of B's own steepest descent that an
                A-step delivers
  per-item sketches (count-sketch, D dims): pairwise cosines within and across sets, the effective
  rank of the A-item gradients (how many directions the fine-tune would need) and the share of each
  B-item gradient on the first principal direction of the A-item gradients (one shared "unsuppress"
  direction, or many).

Usage:
  python3 grad_transfer.py --init DIR --data-dir DATA --out gradxfer_rmu [--device cuda] --confirm-cost
      [--rank 64 --alpha 128 --seed 316 --batch-size 8 --n 0 --n-shuf 3 --sketch-dim 4096 --no-per-item]
  python3 grad_transfer.py --task arith --train-config ../training-run/configs/ts1b_elicit_ft.yaml \
      --init $GEODE_STORE/runs/evt-ts1b-op-bridge-mix/model --out gradxfer_ts1b_bridge --confirm-cost
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "data"))

from relearn import TASK_FORMAT, examples_of, load_split  # noqa: E402

from geode.train.lora import apply_lora  # noqa: E402
from geode.train.sft import _IGNORE_INDEX, _padded_inputs_and_mask  # noqa: E402

SETS = ("bioA", "bioB", "mmluA")


# ------------------------------------------------------------------ data
def rotate_answers(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The same prompts with the answers moved to other items: a derangement by a seeded cyclic
    offset in [1, n-1] (no item keeps its own answer), the knowledge-free twin of a fact set.
    Works from ``full_text`` + the answer char span (any frame the trainers read); anything after
    the span (e.g. nothing, or a trailing newline) stays with its row."""
    n = len(df)
    if n < 2:
        raise ValueError("rotate_answers: need at least two rows")
    k = int(torch.randint(1, n, (1,), generator=torch.Generator().manual_seed(seed)))
    full = df["full_text"].tolist()
    st, en = df["answer_char_start"].astype(int).tolist(), df["answer_char_end"].astype(int).tolist()
    prompt = [t[:a] for t, a in zip(full, st)]
    answer = [t[a:b] for t, a, b in zip(full, st, en)]
    tail = [t[b:] for t, b in zip(full, en)]
    answer = answer[k:] + answer[:k]
    out = df.copy().reset_index(drop=True)
    out["prompt_text"], out["answer_text"] = prompt, answer
    out["full_text"] = [p_ + a_ + t_ for p_, a_, t_ in zip(prompt, answer, tail)]
    out["answer_char_start"] = [len(p_) for p_ in prompt]
    out["answer_char_end"] = [len(p_) + len(a_) for p_, a_ in zip(prompt, answer)]
    return out


def load_sets(data_dir: Path, domain: str, n: int, n_shuf: int, seed: int) -> dict[str, pd.DataFrame]:
    from prepare_wmdp import _sft_rows

    a = load_split(data_dir / f"relearn_{domain}A.parquet")
    ev = pd.read_parquet(data_dir / "wmdp_eval.parquet")
    b = pd.DataFrame(_sft_rows(ev[ev["split"] == f"{domain}_B"]))
    m = load_split(data_dir / "relearn_mmluA.parquet")
    if n:
        a, b, m = a.iloc[:n], b.iloc[:n], m.iloc[:n]
    sets = {"bioA": a.reset_index(drop=True), "bioB": b.reset_index(drop=True), "mmluA": m.reset_index(drop=True)}
    for k in range(n_shuf):
        sets[f"bioA_shuf{k}"] = rotate_answers(sets["bioA"], seed + 1000 + k)
    return sets


def load_sets_arith(train_config: Path, n: int, n_shuf: int, seed: int) -> dict[str, pd.DataFrame]:
    """The arithmetic target (main results): A = a seeded sample of the training file, B = the
    frozen eval file's reporting block (question-disjoint, after EVAL_STOP_ROWS), both hash-verified
    by the trainers' own loader; the null = A with the answers rotated."""
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "training-run" / "scripts"))
    from train import load_config
    from train_sft import load_frozen_parquet

    from geode.edl import EVAL_STOP_ROWS

    cfg = load_config(Path(train_config), None)
    d = cfg["data"]
    tr = load_frozen_parquet(cfg)
    n = n or 600
    idx = sorted(torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed))[:n].tolist())
    ev = load_frozen_parquet({"data": {"hf_id": d["hf_id"], "file": d["eval_file"], "order_hash": d["eval_order_hash"],
                                       "local_path": d.get("eval_local_path")}})
    sets = {"bioA": tr.iloc[idx].reset_index(drop=True),
            "bioB": ev.iloc[EVAL_STOP_ROWS : EVAL_STOP_ROWS + n].reset_index(drop=True)}
    for k in range(n_shuf):
        sets[f"bioA_shuf{k}"] = rotate_answers(sets["bioA"], seed + 1000 + k)
    return sets


# ------------------------------------------------------------------ gradients
def lora_b_params(model) -> list[torch.nn.Parameter]:
    return [m.B.weight for m in model.modules() if hasattr(m, "A") and hasattr(m, "B") and hasattr(m, "scaling")]


def set_gradient(model, params, examples, device: str, bs: int, amp) -> tuple[torch.Tensor, float, int]:
    """Gradient of the set's mean label-token loss w.r.t. the LoRA B factors, as one fp32 vector,
    plus the mean loss and the label-token count.  Sums token losses per batch (so the batching
    cannot change the result) and divides once at the end."""
    ids_all, mask_all = _padded_inputs_and_mask(examples, TASK_FORMAT)
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    loss_sum, n_tok = 0.0, 0
    for s in range(0, len(examples), bs):
        ids, mask = ids_all[s : s + bs].to(device), mask_all[s : s + bs].to(device)
        for p in params:
            p.grad = None
        with amp():
            logits = model(ids).logits.float()
        labels = ids.masked_fill(~mask, _IGNORE_INDEX)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                               ignore_index=_IGNORE_INDEX, reduction="sum")
        loss.backward()
        for a, p in zip(acc, params):
            a.add_(p.grad.float())
        loss_sum += loss.item()
        n_tok += int(mask[:, 1:].sum())
    g = torch.cat([a.flatten() for a in acc]) / max(1, n_tok)
    return g, loss_sum / max(1, n_tok), n_tok


class Sketcher:
    """Count-sketch of the concatenated B-factor gradients into D dims; the hash of every
    coordinate is fixed by ``seed``, so sketches of different items are comparable."""

    def __init__(self, params, D: int, seed: int, device: str):
        self.D = D
        g = torch.Generator().manual_seed(seed)
        self.idx = [torch.randint(0, D, (p.numel(),), generator=g).to(device) for p in params]
        self.sign = [(torch.randint(0, 2, (p.numel(),), generator=g).float() * 2 - 1).to(device) for p in params]

    def __call__(self, grads) -> torch.Tensor:
        sk = torch.zeros(self.D, dtype=torch.float32, device=grads[0].device)
        for g, idx, sign in zip(grads, self.idx, self.sign):
            sk.index_add_(0, idx, sign * g.flatten().float())
        return sk


def item_sketches(model, params, examples, device: str, amp, sketcher: Sketcher) -> torch.Tensor:
    """(n_items, D) count-sketches of each item's mean-token-loss gradient (batch of one)."""
    out = []
    for ex in examples:
        ids_all, mask_all = _padded_inputs_and_mask([ex], TASK_FORMAT)
        ids, mask = ids_all.to(device), mask_all.to(device)
        for p in params:
            p.grad = None
        with amp():
            logits = model(ids).logits.float()
        labels = ids.masked_fill(~mask, _IGNORE_INDEX)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                               ignore_index=_IGNORE_INDEX, reduction="mean")
        loss.backward()
        out.append(sketcher([p.grad for p in params]).cpu())
    return torch.stack(out)


# ------------------------------------------------------------------ statistics
def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.double(), b.double()) / (a.double().norm() * b.double().norm()).clamp_min(1e-30))


def sketch_stats(SA: torch.Tensor, SB: torch.Tensor) -> dict:
    """Pairwise cosines and the shape of the A-item gradient cloud, from count-sketches."""
    def unit(S):
        return S / S.norm(dim=1, keepdim=True).clamp_min(1e-30)
    UA, UB = unit(SA.double()), unit(SB.double())
    CA, CB, CX = UA @ UA.T, UB @ UB.T, UA @ UB.T
    offA = CA[~torch.eye(len(UA), dtype=torch.bool)]
    offB = CB[~torch.eye(len(UB), dtype=torch.bool)]
    G = SA.double() @ SA.double().T
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    erank = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30))
    # the first principal direction of the A-item gradients, and how much of each B gradient lies on it
    _, _, Vh = torch.linalg.svd(SA.double() - SA.double().mean(0), full_matrices=False)
    pc1 = Vh[0]
    on_pc1_B = float(((UB @ pc1) ** 2).mean())
    on_pc1_A = float(((UA @ pc1) ** 2).mean())
    return {"within_A_cos": float(offA.mean()), "within_B_cos": float(offB.mean()), "cross_AB_cos": float(CX.mean()),
            "A_effective_rank": erank, "A_items": int(len(UA)), "B_items": int(len(UB)),
            "B_energy_on_A_pc1": on_pc1_B, "A_energy_on_A_pc1": on_pc1_A}


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="model dir or hub id (its own tokenizer is used)")
    ap.add_argument("--task", choices=("wmdp", "arith"), default="wmdp")
    ap.add_argument("--data-dir", type=Path, default=None, help="wmdp: prepare.py --out-dir")
    ap.add_argument("--train-config", type=Path, default=None, help="arith: the run config whose data block names the files")
    ap.add_argument("--tokenizer", default=None, help="override (default: the model dir's own)")
    ap.add_argument("--out", required=True, help="output stem: writes <out>.json")
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--seed", type=int, default=316, help="LoRA A init, the sketch hash and the answer rotations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="cap per set (0 = all rows)")
    ap.add_argument("--n-shuf", type=int, default=3, help="independent answer rotations (the null's sd)")
    ap.add_argument("--sketch-dim", type=int, default=4096)
    ap.add_argument("--no-per-item", action="store_true", help="skip the per-item sketches (mean gradients only)")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.init)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.task == "wmdp":
        if args.data_dir is None:
            ap.error("--task wmdp needs --data-dir")
        sets = load_sets(args.data_dir, args.domain, args.n, args.n_shuf, args.seed)
    else:
        if args.train_config is None:
            ap.error("--task arith needs --train-config")
        sets = load_sets_arith(args.train_config, args.n, args.n_shuf, args.seed)
    ex = {k: examples_of(df, tokenizer) for k, df in sets.items()}
    n_items = sum(len(v) for v in ex.values())
    n_passes = sum(-(-len(v) // args.batch_size) for v in ex.values())
    if not args.no_per_item:
        n_passes += len(ex["bioA"]) + len(ex["bioB"]) + len(ex["bioA_shuf0"]) if args.n_shuf else len(ex["bioA"]) + len(ex["bioB"])
    est_min = n_passes * 0.5 / 60   # ~0.5 s per forward+backward of a short batch on a 7B model, one 80 GB GPU
    print(f"[xfer] {args.init}: {len(sets)} sets, {n_items} items, {n_passes} forward+backward passes; "
          f"estimated ~{est_min:.1f} GPU-min at 7B (no weights are updated)")
    if not args.confirm_cost:
        print("[xfer] --confirm-cost not given; refusing to run (budget rule). Exiting.")
        return 1

    dev = args.device
    dtype = torch.float32 if dev.startswith("cpu") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    apply_lora(model, rank=args.rank, alpha=args.alpha, seed=args.seed)
    params = lora_b_params(model)
    for m in model.modules():
        if hasattr(m, "A") and hasattr(m, "B") and hasattr(m, "scaling"):
            m.A.float()
            m.B.float()
            m.A.weight.requires_grad_(False)
            m.B.weight.requires_grad_(True)
    model.to(dev)
    model.eval()   # no dropout anywhere in these models; eval keeps the forward deterministic
    import contextlib

    def amp():
        return (torch.autocast(device_type=dev.split(":")[0], dtype=torch.bfloat16) if dtype == torch.bfloat16
                else contextlib.nullcontext())

    t0 = time.time()
    G, loss, ntok = {}, {}, {}
    for k, v in ex.items():
        G[k], loss[k], ntok[k] = set_gradient(model, params, v, dev, args.batch_size, amp)
        print(f"[xfer] {k:<10} {len(v):>5} items {ntok[k]:>7} label tokens  loss {loss[k]:.4f} nats/token  "
              f"|G| {G[k].norm().item():.3e}")
    shuf = [k for k in ex if k.startswith("bioA_shuf")]
    transfer = cos(G["bioA"], G["bioB"])
    nulls = [cos(G[k], G["bioB"]) for k in shuf]
    null_mean = sum(nulls) / len(nulls) if nulls else float("nan")
    null_sd = (sum((x - null_mean) ** 2 for x in nulls) / (len(nulls) - 1)) ** 0.5 if len(nulls) > 1 else float("nan")
    mmlu = cos(G["mmluA"], G["bioB"]) if "mmluA" in G else float("nan")
    gb = G["bioB"].double()
    res = {"init": args.init, "domain": args.domain, "rank": args.rank, "alpha": args.alpha, "seed": args.seed,
           "n_params_B": int(sum(p.numel() for p in params)), "items": {k: len(v) for k, v in ex.items()},
           "label_tokens": ntok, "loss_nats_per_token": loss,
           "transfer_cos": transfer, "null_cos": nulls, "null_cos_mean": null_mean, "null_cos_sd": null_sd,
           "mmlu_cos": mmlu, "score": transfer - null_mean, "score_mmlu": transfer - mmlu,
           "task": args.task,
           "descent_frac": {k: float(torch.dot(G[k].double(), gb) / torch.dot(gb, gb).clamp_min(1e-30))
                            for k in ("bioA", *(["mmluA"] if "mmluA" in G else []), *shuf)},
           "grad_norm": {k: float(G[k].norm()) for k in G},
           "self_cos_shuf_vs_bioA": [cos(G[k], G["bioA"]) for k in shuf]}
    print(f"[xfer] transfer cos(A,B) {transfer:+.4f} | null (answers rotated) {null_mean:+.4f} +- {null_sd:.4f} | "
          f"mmlu {mmlu:+.4f} | SCORE {res['score']:+.4f} (mmlu-null {res['score_mmlu']:+.4f}) | "
          f"descent frac A->B {res['descent_frac']['bioA']:+.3f}")
    if not args.no_per_item:
        sk = Sketcher(params, args.sketch_dim, args.seed, dev)
        SA = item_sketches(model, params, ex["bioA"], dev, amp, sk)
        SB = item_sketches(model, params, ex["bioB"], dev, amp, sk)
        per = {"AB": sketch_stats(SA, SB)}
        if shuf:
            SS = item_sketches(model, params, ex[shuf[0]], dev, amp, sk)
            per["shufB"] = sketch_stats(SS, SB)
        res["per_item"] = {"sketch_dim": args.sketch_dim, **per}
        a = per["AB"]
        print(f"[xfer] per-item: cross A-B cos {a['cross_AB_cos']:+.4f} (rotated {per.get('shufB', {}).get('cross_AB_cos', float('nan')):+.4f}); "
              f"within A {a['within_A_cos']:+.4f}; A effective rank {a['A_effective_rank']:.1f} of {a['A_items']}; "
              f"B energy on A's first direction {a['B_energy_on_A_pc1']:.3f}")
    res["wall_s"] = time.time() - t0
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(f"[xfer] wrote {Path(args.out).with_suffix('.json')} ({res['wall_s'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
