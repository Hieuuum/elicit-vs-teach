"""Before-training reads of latent knowledge from the weights and the data alone (no update).

M21, the hidden-preference gap on the fact surface (the headline since 2026-10-06):
  gap_X = mean_k L_X(answers rotated across items, rotation k) - L_X(true answers)   nats / label token
A set of facts in the fine-tuning format (question, "Answer:", the answer; no options) is scored
as given and with every item's answer moved to another item (a seeded cyclic derangement: same
prompts, same answers, no knowledge).  A model that holds the pairings prefers the true ones; the
gap is 0 for a model that cannot know them, up to the lexical coherence between a question and its
own answer (the cannot-know anchor measures that floor).  Calibration (decisions.md 2026-10-06):
TinyStories-1B arithmetic elicit parent 1.15, format-installed parent 0.09, base 0.13 (held-out
B); the same model on the WMDP-bio facts 0.22; Zephyr original 1.58, RMU 1.30, ELM 1.24, NPO 2.52,
SimNPO 2.64.  Read on the held-out B facts (never trained on in this format); A and the far-domain
MMLU facts are reported too.

M20, gradient transfer at the fine-tune's starting point (kept for the record; its pre-registered
calibration FAILED on 2026-10-06): cos(K_A, K_B) with K_X = G_X(true) - mean_k G_X(rotation k), the
gradients of each set's mean label-token loss w.r.t. the LoRA B factors at their zero init.  A
coherent first-order direction that improves true over rotated pairings exists for any regularity
both halves share, latent or not: the format-installed parent that must be taught scored 0.96,
the elicit parent 0.74, the cannot-know anchor 0.42.  Not an elicit/teach read; not in the verdict.
Also kept: the raw transfer and its rotated null (v1), the MMLU-knowledge reference, descent
fractions and the per-item count-sketches.

Sets:  bioA   the relearning facts (half A), bioB the held-out facts (half B), mmluA the far-domain
       null's facts; <set>_shuf[k] their rotations (4 by default)
--task arith: the arithmetic target, A = a seeded sample of the training file of --train-config,
B = the frozen eval file's reporting block (eval block from --eval-config); no mmlu set.

Usage:
  python3 grad_transfer.py --init DIR --data-dir DATA --out gradxfer_rmu [--device cuda] --confirm-cost
      [--rank 64 --alpha 128 --seed 316 --batch-size 8 --n 0 --n-shuf 4 --sketch-dim 4096 --no-per-item]
  python3 grad_transfer.py --task arith --train-config ../training-run/configs/ts1b_elicit_ft.yaml \
      --eval-config ../training-run/configs/ts1b_fig2ts_inst.yaml --tokenizer meta-llama/Llama-3.2-1B \
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
def rotate_answers(df: pd.DataFrame, seed: int, offset: int | None = None) -> pd.DataFrame:
    """The same prompts with the answers moved to other items: a derangement by a seeded cyclic
    offset in [1, n-1] (no item keeps its own answer), the knowledge-free twin of a fact set.
    Works from ``full_text`` + the answer char span (any frame the trainers read); anything after
    the span (e.g. nothing, or a trailing newline) stays with its row.  ``offset`` fixes the
    rotation instead of drawing it (the n-1 offsets together visit every off-diagonal pairing
    (q_i, a_j) exactly once, which is what makes the rotation average the off-diagonal mean)."""
    n = len(df)
    if n < 2:
        raise ValueError("rotate_answers: need at least two rows")
    if offset is not None and not 1 <= offset <= n - 1:
        raise ValueError(f"rotate_answers: offset must be in [1, {n - 1}]")
    k = offset if offset is not None else int(torch.randint(1, n, (1,), generator=torch.Generator().manual_seed(seed)))
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
    return with_rotations(sets, n_shuf, seed)


def with_rotations(sets: dict[str, pd.DataFrame], n_shuf: int, seed: int) -> dict[str, pd.DataFrame]:
    """Adds <set>_shuf[k] for bioA, bioB and (if present) mmluA: independent seeded rotations of
    each set's answers.  mmluA's rotations give the same-model, unrelated-knowledge reference
    cos(K_mmluA, K_bioB): the alignment that shared recall machinery alone produces."""
    for i, name in enumerate(("bioA", "bioB", "mmluA")):
        if name in sets:
            for k in range(n_shuf):
                sets[f"{name}_shuf{k}"] = rotate_answers(sets[name], seed + 1000 * (i + 1) + k)
    return sets


def load_sets_arith(train_config: Path, eval_config: Path | None, n: int, n_shuf: int, seed: int) -> dict[str, pd.DataFrame]:
    """The arithmetic target (main results): A = a seeded sample of the training file, B = the
    frozen eval file's reporting block (question-disjoint, after EVAL_STOP_ROWS), both hash-verified
    by the trainers' own loader.  The eval block (eval_file / eval_order_hash / eval_local_path)
    is read from ``eval_config`` (default: the train config; the FT configs carry none, the
    install configs do)."""
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "training-run" / "scripts"))
    from train import load_config
    from train_sft import load_frozen_parquet

    from geode.edl import EVAL_STOP_ROWS

    cfg = load_config(Path(train_config), None)
    e = load_config(Path(eval_config), None)["data"] if eval_config else cfg["data"]
    if "eval_file" not in e or "eval_order_hash" not in e:
        raise SystemExit(f"[xfer] no eval block (eval_file, eval_order_hash) in {eval_config or train_config}; "
                         "pass --eval-config, e.g. configs/ts1b_fig2ts_inst.yaml")
    tr = load_frozen_parquet(cfg)
    n = n or 600
    idx = sorted(torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed))[:n].tolist())
    ev = load_frozen_parquet({"data": {"hf_id": e["hf_id"], "file": e["eval_file"], "order_hash": e["eval_order_hash"],
                                       "local_path": e.get("eval_local_path")}})
    if len(ev) < EVAL_STOP_ROWS + n:
        raise SystemExit(f"[xfer] eval file has {len(ev)} rows < {EVAL_STOP_ROWS + n}")
    sets = {"bioA": tr.iloc[idx].reset_index(drop=True),
            "bioB": ev.iloc[EVAL_STOP_ROWS : EVAL_STOP_ROWS + n].reset_index(drop=True)}
    return with_rotations(sets, n_shuf, seed)


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


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs) if xs else float("nan")
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")
    return m, sd


def preference_gaps(loss: dict[str, float]) -> dict[str, dict[str, float]]:
    """M21 from the per-set mean label-token losses: for every set with rotations,
    gap = mean_k L(rotation k) - L(true) in nats per label token, with the sd of L over the
    rotations (the spread a knowledge-free re-pairing produces) and the rotation count."""
    out = {}
    for name, l_true in loss.items():
        if "_shuf" in name:
            continue
        rots = [v for k, v in loss.items() if k.startswith(name + "_shuf")]
        if rots:
            m, sd = _mean_sd(rots)
            out[name] = {"gap_nats": m - l_true, "true_nats": l_true, "rotated_mean_nats": m, "rotated_sd_nats": sd,
                         "n_rotations": len(rots)}
    return out


def knowledge_vectors(G: dict[str, torch.Tensor], a: str = "bioA", b: str = "bioB") -> dict:
    """The pairing-dependent part of each set's gradient and its alignment across the two sets.

    K_X = G_X(true) - mean_k G_X(rotation k): everything the true and the rotated sets share
    (format, vocabulary, topic, the model's state) cancels exactly, leaving the gradient of the
    model's knowledge of which answer goes with which question.  knowledge_cos = cos(K_A, K_B).
    Null: with no knowledge the true pairing is just another rotation, so
    K_X^(j) = G_X(rotation j) - mean_{k != j} G_X(rotation k) is distributed like K_X; the null
    sample is cos(K_A^(j), K_B^(j')) over all rotation pairs.  Needs >= 2 rotations per set."""
    ra = sorted(k for k in G if k.startswith(a + "_shuf"))
    rb = sorted(k for k in G if k.startswith(b + "_shuf"))
    if len(ra) < 2 or len(rb) < 2:
        raise ValueError("knowledge_vectors: need at least two rotations of each set")

    def K(true: str, rots: list[str]) -> torch.Tensor:
        return G[true].double() - torch.stack([G[r].double() for r in rots]).mean(0)

    KA, KB = K(a, ra), K(b, rb)
    nulls = [cos(K(j, [k for k in ra if k != j]), K(jj, [k for k in rb if k != jj])) for j in ra for jj in rb]
    nm, nsd = _mean_sd(nulls)
    kc = cos(KA, KB)
    return {"knowledge_cos": kc, "knowledge_null": nulls, "knowledge_null_mean": nm, "knowledge_null_sd": nsd,
            "score": kc - nm,
            "knowledge_share": {a: float(KA.norm() / G[a].double().norm().clamp_min(1e-30)),
                                b: float(KB.norm() / G[b].double().norm().clamp_min(1e-30))},
            "knowledge_descent_frac": float(torch.dot(KA, KB) / torch.dot(KB, KB).clamp_min(1e-30)),
            "knowledge_norm": {a: float(KA.norm()), b: float(KB.norm())}}


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
    ap.add_argument("--eval-config", type=Path, default=None,
                    help="arith: the config whose data block carries eval_file / eval_order_hash (default: --train-config)")
    ap.add_argument("--tokenizer", default=None, help="override (default: the model dir's own)")
    ap.add_argument("--out", required=True, help="output stem: writes <out>.json")
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--seed", type=int, default=316, help="LoRA A init, the sketch hash and the answer rotations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="cap per set (0 = all rows)")
    ap.add_argument("--n-shuf", type=int, default=4, help="independent answer rotations per set (>= 2; the null)")
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
        sets = load_sets_arith(args.train_config, args.eval_config, args.n, args.n_shuf, args.seed)
    if args.n_shuf < 2:
        ap.error("--n-shuf must be >= 2 (the knowledge null needs rotation pairs)")
    ex = {k: examples_of(df, tokenizer) for k, df in sets.items()}
    n_items = sum(len(v) for v in ex.values())
    n_passes = sum(-(-len(v) // args.batch_size) for v in ex.values())
    if not args.no_per_item:   # true items and one rotation of each set, one item per pass
        n_passes += 2 * (len(ex["bioA"]) + len(ex["bioB"]))
    est_min = n_passes * 0.25 / 60   # ~0.25 s per forward+backward of a short batch on a 7B model, one 80 GB GPU (measured 0.2)
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
    null_mean, null_sd = _mean_sd(nulls)
    mmlu = cos(G["mmluA"], G["bioB"]) if "mmluA" in G else float("nan")
    gb = G["bioB"].double()
    gaps = preference_gaps(loss)
    print("[xfer] M21 hidden-preference gap L(rotated) - L(true), nats/token: "
          + " | ".join(f"{k} {g['gap_nats']:+.3f} (rotation sd {g['rotated_sd_nats']:.3f})" for k, g in gaps.items()))
    know = knowledge_vectors(G)
    know_m = knowledge_vectors(G, "mmluA", "bioB") if "mmluA_shuf0" in G else None
    res = {"version": 3, "gap": gaps, "init": args.init, "domain": args.domain, "rank": args.rank, "alpha": args.alpha, "seed": args.seed,
           "n_params_B": int(sum(p.numel() for p in params)), "items": {k: len(v) for k, v in ex.items()},
           "label_tokens": ntok, "loss_nats_per_token": loss, "task": args.task,
           **know, "mmlu_knowledge": know_m,
           "transfer_cos": transfer, "null_cos": nulls, "null_cos_mean": null_mean, "null_cos_sd": null_sd,
           "mmlu_cos": mmlu, "raw_score": transfer - null_mean, "score_mmlu": transfer - mmlu,
           "descent_frac": {k: float(torch.dot(G[k].double(), gb) / torch.dot(gb, gb).clamp_min(1e-30))
                            for k in ("bioA", *(["mmluA"] if "mmluA" in G else []), *shuf)},
           "grad_norm": {k: float(G[k].norm()) for k in G},
           "self_cos_shuf_vs_bioA": [cos(G[k], G["bioA"]) for k in shuf]}
    print(f"[xfer] KNOWLEDGE alignment cos(K_A,K_B) {know['knowledge_cos']:+.4f} | null (rotation vs rotation) "
          f"{know['knowledge_null_mean']:+.4f} +- {know['knowledge_null_sd']:.4f} | SCORE {know['score']:+.4f} | "
          f"knowledge share of |G|: A {know['knowledge_share']['bioA']:.3f} B {know['knowledge_share']['bioB']:.3f} | "
          f"knowledge descent frac A->B {know['knowledge_descent_frac']:+.3f}")
    if know_m:
        print(f"[xfer] MMLU-knowledge reference cos(K_mmluA,K_bioB) {know_m['knowledge_cos']:+.4f} | null "
              f"{know_m['knowledge_null_mean']:+.4f} +- {know_m['knowledge_null_sd']:.4f} | knowledge share mmluA "
              f"{know_m['knowledge_share']['mmluA']:.3f} | bio-specific excess {know['knowledge_cos'] - know_m['knowledge_cos']:+.4f}")
    print(f"[xfer] raw: transfer cos(A,B) {transfer:+.4f} | rotated-answer null {null_mean:+.4f} +- {null_sd:.4f} | "
          f"mmlu {mmlu:+.4f} | raw score {res['raw_score']:+.4f} (mmlu-null {res['score_mmlu']:+.4f}) | "
          f"descent frac A->B {res['descent_frac']['bioA']:+.3f}")
    if not args.no_per_item:
        sk = Sketcher(params, args.sketch_dim, args.seed, dev)
        SA = item_sketches(model, params, ex["bioA"], dev, amp, sk)
        SB = item_sketches(model, params, ex["bioB"], dev, amp, sk)
        SAr = item_sketches(model, params, ex["bioA_shuf0"], dev, amp, sk)   # row i: the same prompt, another answer
        SBr = item_sketches(model, params, ex["bioB_shuf0"], dev, amp, sk)
        per = {"AB": sketch_stats(SA, SB), "shufB": sketch_stats(SAr, SB), "knowledge": sketch_stats(SA - SAr, SB - SBr)}
        res["per_item"] = {"sketch_dim": args.sketch_dim, **per}
        a, kn = per["AB"], per["knowledge"]
        print(f"[xfer] per-item raw: cross A-B cos {a['cross_AB_cos']:+.4f} (rotated {per['shufB']['cross_AB_cos']:+.4f}); "
              f"within A {a['within_A_cos']:+.4f}; A effective rank {a['A_effective_rank']:.1f} of {a['A_items']}; "
              f"B energy on A's first direction {a['B_energy_on_A_pc1']:.3f}")
        print(f"[xfer] per-item knowledge (true minus one rotation): cross A-B cos {kn['cross_AB_cos']:+.4f}; "
              f"within A {kn['within_A_cos']:+.4f}; A effective rank {kn['A_effective_rank']:.1f} of {kn['A_items']}; "
              f"B energy on A's first direction {kn['B_energy_on_A_pc1']:.3f}")
    res["wall_s"] = time.time() - t0
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(f"[xfer] wrote {Path(args.out).with_suffix('.json')} ({res['wall_s'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
