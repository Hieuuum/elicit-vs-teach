"""Pre-fine-tuning predictors: read the PARENT alone (plus labelled task data) and
ask whether the target capability is latent (about to be ELICITED) or absent
(about to be TAUGHT) — no fine-tuned checkpoint involved anywhere.

Every metric runs on one model spec (zoo run id or hub id) and appends its
block to prefit_<tag>.json, so the five parents of the construction ladder can
be compared with `compare`. Task inputs: the bare natural-language target
(`What is the sum of 621 and 5068?\n`), rows DEFAULT_ROW_OFFSET.. of
D_algo_eval (held out from every training file). Negative answers: the minus
sign is appended to the prompt and the first DIGIT chunk is the scored token
(the sign artefact of 2026-09-10).

Metrics
  pref      hidden preference: logit(correct first chunk) - logit(a mismatched
            problem's), log p(correct), rank — one forward pass per problem.
  geometry  problem-specificity of the parent's own answer-position states:
            cos-to-mean and PC1 energy fraction per layer (1.0 = one state for
            every problem).
  probe     linear decodability of the ANSWER from the frozen state, per layer:
            ridge R^2 on helix features [y, cos/sin(2 pi y/T), T in 2,5,10,100]
            (Kantamneni & Tegmark 2025) for the answer AND for each operand
            (the control: inputs are copyable, the answer must be computed),
            plus first-digit logistic-regression accuracy.
  das       causal answer subspace (Distributed Alignment Search, Geiger et al.
            2024): learn a k-dim orthonormal subspace at layer L such that
            swapping it between two problems swaps the model's answer
            PREFERENCE (logit-diff, so it works on a parent that never emits
            the answer). Baselines: no patch, random subspace, full-residual
            patch (ceiling).
  dcm       operand roles in the frozen parent (Desiderata Component Masking,
            heads only; reuses dcm_roles.learn_role) with a logit-diff flip
            criterion instead of argmax.
  attn      attention interface: mass from the answer position onto the
            operand digit tokens, per head; no labels needed.
  grad      gradient coherence at step 0: per-example task-loss gradients,
            count-sketched to D dims; mean pairwise cosine, coherence
            ||mean g|| / mean ||g||, effective rank of the Gram matrix,
            kernel-target alignment with a first-digit label kernel.
  hessian   task-loss curvature at the parent: both ends of the Hessian spectrum
            (power iteration, then shifted power iteration), Hutchinson trace,
            gradient sharpness g'Hg/|g|^2, and the one-step gain |g|^4 / (2 g'Hg).
  llc       local learning coefficient (Lau et al. 2023; SGLD estimator) of the
            parent at the task loss — NOT VALID HERE: the estimator assumes w0 is
            a local minimum of the loss; a parent before fine-tuning is not, SGLD
            descends and the estimate comes out negative (2026-09-18 run: all five
            parents -0.7e6 .. -2.1e6). Kept for completeness; do not report.
  all       every metric above in that order.
  compare   table across tags.

Usage
  python3 prefit_metrics.py <metric> --model SPEC --tag TAG [--n 256] [--device cuda]
  python3 prefit_metrics.py compare TAG [TAG ...]
  python3 prefit_metrics.py smoke        # CPU, tiny random model, code-path check

GPU for the real parents (fp32 1B: ~5 GB weights; grad/hessian/llc ~20 GB).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geode.arith.formats import true_answer  # noqa: E402
from premise_checks import render_probe  # noqa: E402
from resid_shift import STORE, TOKENIZER, ResidTaps, load_fp32, task_prompts  # noqa: E402

HELIX_T = (2, 5, 10, 100)


# ------------------------------------------------------------------ inputs
def problems(n: int, tokenizer, offset_extra: int = 0):
    """[(prompt_with_sign, target_token, a, b, op, answer)] on the bare NL surface."""
    prompts, triples = task_prompts("bare_nl", n + offset_extra)
    out = []
    for p, (a, b, op) in list(zip(prompts, triples))[offset_extra:]:
        ans = true_answer(a, b, op)
        if ans < 0:
            p, tgt_str = p + "-", str(-ans)
        else:
            tgt_str = str(ans)
        tok = tokenizer(tgt_str, add_special_tokens=False)["input_ids"][0]
        out.append((p, tok, a, b, op, ans))
    return out


def answer_text(ans: int) -> str:
    return str(ans)


def length_pairs(items, tokenizer, n_pairs: int, seed: int = 316):
    """Pairs of problems with identical prompt token length and different target
    tokens: (ids_base, ids_src, tok_base, tok_src, meta_base, meta_src)."""
    rng = random.Random(seed)
    buckets: dict[int, list] = {}
    for it in items:
        ids = tokenizer(it[0], add_special_tokens=False)["input_ids"]
        buckets.setdefault(len(ids), []).append((ids, it))
    pairs = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        for x, y in zip(bucket[0::2], bucket[1::2]):
            if x[1][1] != y[1][1]:
                pairs.append((x[0], y[0], x[1][1], y[1][1], x[1], y[1]))
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def batched_last_logits(model, tokenizer, prompts, device, bs):
    """Final-position logits for a list of prompts (left padding)."""
    tokenizer.padding_side = "left"
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tokenizer(prompts[s : s + bs], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        with torch.no_grad():
            outs.append(model(**enc).logits[:, -1].float())
    return torch.cat(outs)


def states_at_answer(model, tokenizer, prompts, device, bs):
    """(L+1, N, d) residual stream at the last prompt token, every layer."""
    tokenizer.padding_side = "left"
    taps = ResidTaps(model)
    acc = None
    try:
        for s in range(0, len(prompts), bs):
            enc = tokenizer(prompts[s : s + bs], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            with torch.no_grad():
                model(**enc)
            st = torch.stack([a[:, -1].float() for a in taps.acts])  # (L+1, B, d)
            acc = st if acc is None else torch.cat([acc, st], dim=1)
    finally:
        taps.remove()
    return acc


def load(spec: str, device: str, eager: bool = False):
    model = load_fp32(spec, device)
    if eager:
        try:
            model.set_attn_implementation("eager")
        except Exception:
            model.config._attn_implementation = "eager"
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def write_block(out_dir: Path, tag: str, key: str, block: dict, model: str):
    path = out_dir / f"prefit_{tag}.json"
    d = json.loads(path.read_text()) if path.is_file() else {"tag": tag, "model": model}
    d[key] = block
    path.write_text(json.dumps(d, indent=2))
    print(f"[prefit] wrote {path.name}::{key}")


# ------------------------------------------------------------------ pref
def metric_pref(model, tokenizer, items, device, bs):
    prompts = [it[0] for it in items]
    tgt = torch.tensor([it[1] for it in items], device=device)
    logits = batched_last_logits(model, tokenizer, prompts, device, bs)
    # distractor: the target of the problem 1 step ahead (cyclic), skipping equal tokens
    dis = torch.roll(tgt, 1)
    keep = dis != tgt
    ld = (logits.gather(1, tgt[:, None]) - logits.gather(1, dis[:, None])).squeeze(1)[keep]
    logp = F.log_softmax(logits, dim=-1).gather(1, tgt[:, None]).squeeze(1)
    rank = (logits > logits.gather(1, tgt[:, None])).sum(1) + 1
    return {"n": len(items), "logit_diff_mean": ld.mean().item(),
            "logit_diff_pos_frac": (ld > 0).float().mean().item(),
            "logp_correct_mean": logp.mean().item(),
            "rank_median": rank.float().median().item(),
            "top1_acc": (rank == 1).float().mean().item()}


# ------------------------------------------------------------------ geometry
def pc1_stats(mat: torch.Tensor):
    mat = mat.double()
    m = mat.mean(0, keepdim=True)
    c2m = F.cosine_similarity(mat, m.expand_as(mat), dim=1).mean().item()
    sv = torch.linalg.svdvals(mat)
    return c2m, (sv[0] ** 2 / (sv ** 2).sum()).item()


def metric_geometry(model, tokenizer, items, device, bs):
    st = states_at_answer(model, tokenizer, [it[0] for it in items], device, bs)
    per = [pc1_stats(st[l]) for l in range(st.shape[0])]
    return {"n": len(items), "cos_to_mean_by_layer": [round(p[0], 4) for p in per],
            "pc1_by_layer": [round(p[1], 4) for p in per],
            "cos_to_mean_last": per[-1][0], "pc1_last": per[-1][1]}


# ------------------------------------------------------------------ probe
def helix(y: torch.Tensor) -> torch.Tensor:
    cols = [y / y.abs().max().clamp_min(1.0)]
    for T in HELIX_T:
        cols += [torch.cos(2 * math.pi * y / T), torch.sin(2 * math.pi * y / T)]
    return torch.stack(cols, 1)


def ridge_r2(X_tr, Y_tr, X_te, Y_te, lam_rel=1e-2):
    mu, sd = X_tr.mean(0), X_tr.std(0).clamp_min(1e-6)
    Xtr, Xte = (X_tr - mu) / sd, (X_te - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=Xtr.device)], 1)
    Xte = torch.cat([Xte, torch.ones(len(Xte), 1, device=Xte.device)], 1)
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam_rel * (Xtr.T @ Xtr).diagonal().mean() * torch.eye(d, device=Xtr.device)
    W = torch.linalg.solve(A, Xtr.T @ Y_tr)
    pred = Xte @ W
    ss_res = ((Y_te - pred) ** 2).sum(0)
    ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return (1 - ss_res / ss_tot)  # per target column


def logreg_acc(X_tr, y_tr, X_te, y_te, n_cls, steps=300):
    mu, sd = X_tr.mean(0), X_tr.std(0).clamp_min(1e-6)
    Xtr, Xte = (X_tr - mu) / sd, (X_te - mu) / sd
    W = torch.zeros(Xtr.shape[1], n_cls, device=Xtr.device, requires_grad=True)
    b = torch.zeros(n_cls, device=Xtr.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, y_tr) + 1e-3 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((Xte @ W + b).argmax(1) == y_te).float().mean().item()
    return acc


def metric_probe(model, tokenizer, items, device, bs, seed=316):
    st = states_at_answer(model, tokenizer, [it[0] for it in items], device, bs).double()
    n = st.shape[1]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tr, te = perm[: int(0.75 * n)].to(device), perm[int(0.75 * n):].to(device)
    ans = torch.tensor([float(it[5]) for it in items], device=device, dtype=torch.float64)
    a = torch.tensor([float(it[2]) for it in items], device=device, dtype=torch.float64)
    b = torch.tensor([float(it[3]) for it in items], device=device, dtype=torch.float64)
    first_digit = torch.tensor([int(str(abs(it[5]))[0]) for it in items], device=device)
    majority = torch.bincount(first_digit[te], minlength=10).max().item() / len(te)
    out = {"n": n, "train": len(tr), "test": len(te), "first_digit_majority": majority,
           "r2_answer_by_layer": [], "r2_operand_a_by_layer": [], "r2_operand_b_by_layer": [],
           "r2_answer_linear_by_layer": [], "first_digit_acc_by_layer": []}
    for l in range(st.shape[0]):
        X = st[l]
        for key, y in (("r2_answer_by_layer", ans), ("r2_operand_a_by_layer", a),
                       ("r2_operand_b_by_layer", b)):
            r2 = ridge_r2(X[tr], helix(y)[tr], X[te], helix(y)[te])
            out[key].append(round(r2.mean().item(), 4))
            if key == "r2_answer_by_layer":
                out["r2_answer_linear_by_layer"].append(round(r2[0].item(), 4))
        out["first_digit_acc_by_layer"].append(
            round(logreg_acc(X[tr].float(), first_digit[tr], X[te].float(), first_digit[te], 10), 4))
    ra, ro = out["r2_answer_by_layer"], out["r2_operand_a_by_layer"]
    out["best_layer_answer"] = int(max(range(len(ra)), key=lambda i: ra[i]))
    out["r2_answer_best"] = max(ra)
    out["r2_operands_best"] = max(max(ro), max(out["r2_operand_b_by_layer"]))
    out["first_digit_acc_best"] = max(out["first_digit_acc_by_layer"])
    return out


# ------------------------------------------------------------------ das
class LayerPatch:
    """Additive subspace patch on the residual stream after layer L at the last
    position: h <- h + P (h_src - h), P = R R^T (R orthonormal d x k)."""

    def __init__(self, model, layer: int):
        self.src = None       # (B, d) source states at the last position
        self.R = None         # (d, k) or None for full replacement
        self.mode = "off"     # off | capture | patch | full
        self.captured = None
        self.h = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.mode == "capture":
            self.captured = hs[:, -1].detach().clone()
            return None
        if self.mode in ("patch", "full"):
            last = hs[:, -1]
            delta = (self.src.to(last.dtype) - last)
            if self.mode == "full":
                new_last = self.src.to(last.dtype)
            else:
                R = self.R.to(last.dtype)
                new_last = last + (delta @ R) @ R.T
            hs = torch.cat([hs[:, :-1], new_last[:, None]], dim=1)
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return None

    def remove(self):
        self.h.remove()


def _pair_tensors(pairs, device):
    groups: dict[int, list] = {}
    for pr in pairs:
        groups.setdefault(len(pr[0]), []).append(pr)
    for g in groups.values():
        yield (torch.tensor([p[0] for p in g], device=device),
               torch.tensor([p[1] for p in g], device=device),
               torch.tensor([p[2] for p in g], device=device),
               torch.tensor([p[3] for p in g], device=device))


def _eval_patch(model, patch, batches, device):
    ld, flips, n = 0.0, 0, 0
    with torch.no_grad():
        for base, src, tb, ts in batches:
            patch.mode = "capture"; model(src); patch.src = patch.captured
            logits = model(base).logits[:, -1].float()
            patch.mode = "off"
            d = logits.gather(1, ts[:, None]) - logits.gather(1, tb[:, None])
            ld += d.sum().item(); flips += (d > 0).sum().item(); n += len(base)
    return ld / n, flips / n


def metric_das(model, tokenizer, items, device, layers, ks, n_train, n_test, steps, lr,
               seed=316):
    pairs = length_pairs(items, tokenizer, n_train + n_test, seed)
    if len(pairs) < n_train + n_test:
        print(f"[prefit] das: only {len(pairs)} length-matched pairs")
    train, test = pairs[:n_train], pairs[n_train : n_train + n_test]
    d = model.config.hidden_size
    out = {"layers": {}, "n_train": len(train), "n_test": len(test), "steps": steps}
    tr_b = list(_pair_tensors(train, device)); te_b = list(_pair_tensors(test, device))
    for L in layers:
        patch = LayerPatch(model, L)
        try:
            res = {}
            # baselines: no patch / full replacement (ceiling)
            with torch.no_grad():
                ld0 = fl0 = n0 = 0
                for base, src, tb, ts in te_b:
                    lg = model(base).logits[:, -1].float()
                    dd = lg.gather(1, ts[:, None]) - lg.gather(1, tb[:, None])
                    ld0 += dd.sum().item(); fl0 += (dd > 0).sum().item(); n0 += len(base)
            res["none"] = {"logit_diff_src_minus_base": ld0 / n0, "flip_frac": fl0 / n0}
            patch.mode = "off"
            def full_eval():
                ld, fl, n = 0.0, 0, 0
                with torch.no_grad():
                    for base, src, tb, ts in te_b:
                        patch.mode = "capture"; model(src); patch.src = patch.captured
                        patch.mode = "full"
                        lg = model(base).logits[:, -1].float(); patch.mode = "off"
                        dd = lg.gather(1, ts[:, None]) - lg.gather(1, tb[:, None])
                        ld += dd.sum().item(); fl += (dd > 0).sum().item(); n += len(base)
                return {"logit_diff_src_minus_base": ld / n, "flip_frac": fl / n}
            res["full"] = full_eval()
            for k in ks:
                torch.manual_seed(seed)
                lin = torch.nn.Linear(k, d, bias=False).to(device)
                lin = torch.nn.utils.parametrizations.orthogonal(lin)  # weight (d, k) orthonormal cols
                opt = torch.optim.Adam(lin.parameters(), lr=lr)
                for step in range(steps):
                    tot = 0.0; n = 0
                    for base, src, tb, ts in tr_b:
                        patch.mode = "capture"
                        with torch.no_grad():
                            model(src)
                        patch.src = patch.captured
                        patch.R = lin.weight
                        patch.mode = "patch"
                        logits = model(base).logits[:, -1].float()
                        patch.mode = "off"
                        loss = F.cross_entropy(logits, ts, reduction="sum")
                        tot = tot + loss; n += len(base)
                    opt.zero_grad(); (tot / n).backward(); opt.step()
                with torch.no_grad():
                    patch.R = lin.weight.detach()
                    patch.mode = "off"
                    ld, fl, nn_ = 0.0, 0, 0
                    for base, src, tb, ts in te_b:
                        patch.mode = "capture"; model(src); patch.src = patch.captured
                        patch.mode = "patch"
                        lg = model(base).logits[:, -1].float(); patch.mode = "off"
                        dd = lg.gather(1, ts[:, None]) - lg.gather(1, tb[:, None])
                        ld += dd.sum().item(); fl += (dd > 0).sum().item(); nn_ += len(base)
                    # random subspace of the same k
                    Rr, _ = torch.linalg.qr(torch.randn(d, k, device=device))
                    patch.R = Rr
                    ldr, flr, nr = 0.0, 0, 0
                    for base, src, tb, ts in te_b:
                        patch.mode = "capture"; model(src); patch.src = patch.captured
                        patch.mode = "patch"
                        lg = model(base).logits[:, -1].float(); patch.mode = "off"
                        dd = lg.gather(1, ts[:, None]) - lg.gather(1, tb[:, None])
                        ldr += dd.sum().item(); flr += (dd > 0).sum().item(); nr += len(base)
                res[f"k{k}"] = {"logit_diff_src_minus_base": ld / nn_, "flip_frac": fl / nn_,
                                "random_logit_diff": ldr / nr, "random_flip_frac": flr / nr}
                print(f"[prefit] das L{L} k={k}: flip {fl / nn_:.3f} (random {flr / nr:.3f}, "
                      f"full {res['full']['flip_frac']:.3f}, none {res['none']['flip_frac']:.3f}); "
                      f"ld {ld / nn_:+.2f}")
            out["layers"][str(L)] = res
        finally:
            patch.remove()
    return out


# ------------------------------------------------------------------ dcm
def metric_dcm(model, tokenizer, device, n_pairs, lam, steps, lr):
    from dcm_roles import MixTaps, learn_role, make_pairs

    taps = MixTaps(model)
    out = {"surface": "bare_nl", "lam": lam, "steps": steps, "roles": {}}
    try:
        for role in ("operand_a", "operand_b"):
            pairs = make_pairs("bare_nl", role, n_pairs, tokenizer)
            if len(pairs) < 8:
                out["roles"][role] = {"skipped": f"{len(pairs)} pairs"}
                continue
            ha, hm, st = learn_role(model, taps, pairs, device, lam, steps, lr, components="heads")
            # logit-diff flip criterion (the parent need not emit the answer)
            taps.mask_attn, taps.mask_mlp = ha.to(device), hm.to(device)
            ld_mix = ld_clean = ld_ceiling = 0.0; fl_mix = fl_ceil = 0; n = 0
            with torch.no_grad():
                for clean, cf, ct, xt in _pair_tensors(pairs, device):
                    taps.cf = {}; taps.mode = "capture"; model(cf); taps.mode = "off"
                    taps.mode = "mix"; lg = model(clean).logits[:, -1].float(); taps.mode = "off"
                    d = lg.gather(1, xt[:, None]) - lg.gather(1, ct[:, None])
                    lg_c = model(clean).logits[:, -1].float()
                    d_c = lg_c.gather(1, xt[:, None]) - lg_c.gather(1, ct[:, None])
                    lg_x = model(cf).logits[:, -1].float()
                    d_x = lg_x.gather(1, xt[:, None]) - lg_x.gather(1, ct[:, None])
                    ld_mix += d.sum().item(); ld_clean += d_c.sum().item(); ld_ceiling += d_x.sum().item()
                    fl_mix += (d > 0).sum().item(); fl_ceil += (d_x > 0).sum().item(); n += len(clean)
            nodes = [f"attn:{i}:{h}" for i in range(taps.L) for h in range(taps.H) if ha[i, h]]
            out["roles"][role] = {"nodes": nodes, "n_heads": int(ha.sum()), "n_pairs": len(pairs),
                                  "argmax_flip_acc": st["cf_flip_acc"],
                                  "argmax_ceiling": st["cf_ceiling_acc"],
                                  "ld_flip_frac": fl_mix / n, "ld_flip_ceiling": fl_ceil / n,
                                  "ld_mix": ld_mix / n, "ld_clean": ld_clean / n,
                                  "ld_ceiling": ld_ceiling / n}
            print(f"[prefit] dcm {role}: {int(ha.sum())} heads; ld-flip {fl_mix / n:.3f} "
                  f"(ceiling {fl_ceil / n:.3f}); ld mix {ld_mix / n:+.2f} clean {ld_clean / n:+.2f} "
                  f"ceiling {ld_ceiling / n:+.2f}")
    finally:
        taps.remove()
    return out


# ------------------------------------------------------------------ attn
def operand_positions(tokenizer, prompt: str, a: int, b: int):
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    sa = prompt.find(str(a)); sb = prompt.find(str(b), sa + len(str(a)))
    spans = [(sa, sa + len(str(a))), (sb, sb + len(str(b)))]
    pos = [i for i, (s, e) in enumerate(enc["offset_mapping"])
           if any(s < se and e > ss for ss, se in spans) and e > s]
    return enc["input_ids"], pos


def metric_attn(model, tokenizer, items, device, bs):
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    mass = torch.zeros(L, H, dtype=torch.float64)
    uniform = 0.0; n = 0
    for it in items:
        ids, pos = operand_positions(tokenizer, it[0], it[2], it[3])
        if not pos:
            continue
        with torch.no_grad():
            out = model(torch.tensor([ids], device=device), output_attentions=True)
        if out.attentions is None or out.attentions[0] is None:
            raise SystemExit("[prefit] attn: no attention weights returned; eager attention required")
        for l, att in enumerate(out.attentions):
            mass[l] += att[0, :, -1, pos].sum(-1).double().cpu()
        uniform += len(pos) / len(ids); n += 1
    mass /= n
    flat = mass.flatten()
    top = torch.topk(flat, 10)
    return {"n": n, "uniform_baseline": uniform / n,
            "max_head_mass": mass.max().item(),
            "heads_over_0.5": int((mass > 0.5).sum()), "heads_over_0.25": int((mass > 0.25).sum()),
            "mean_head_mass": mass.mean().item(),
            "per_layer_max": [round(v, 3) for v in mass.max(1).values.tolist()],
            "top_heads": [(f"attn:{i // H}:{i % H}", round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())]}


# ------------------------------------------------------------------ grad
def _trainable(model):
    return [(n, p) for n, p in model.named_parameters()
            if "embed_tokens" not in n and "lm_head" not in n]


def example_loss(model, tokenizer, item, device):
    """Teacher-forced CE over the answer tokens given the prompt (the SFT loss)."""
    p_ids = tokenizer(item[0], add_special_tokens=False)["input_ids"]
    tgt = str(-item[5]) if item[5] < 0 else str(item[5])
    a_ids = tokenizer(tgt, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p_ids + a_ids], device=device)
    logits = model(ids).logits[0, len(p_ids) - 1 : -1].float()
    return F.cross_entropy(logits, torch.tensor(a_ids, device=device))


def count_sketch(grads, D: int, seed: int = 0, chunk: int = 4_000_000):
    """Concatenated count-sketch of a list of gradient tensors -> (D,)."""
    dev = grads[0].device
    sk = torch.zeros(D, dtype=torch.float64, device=dev)
    for pi, g in enumerate(grads):
        g = g.flatten().double()
        for s in range(0, g.numel(), chunk):
            idx = torch.arange(s, min(s + chunk, g.numel()), device=dev, dtype=torch.int64)
            h = (idx * 2654435761 + (seed + pi) * 97531) & 0x7FFFFFFF
            bucket = h % D
            sign = ((h >> 20) & 1).double() * 2 - 1
            sk.index_add_(0, bucket, sign * g[s : s + len(idx)])
    return sk


def metric_grad(model, tokenizer, items, device, n_ex, D):
    names_params = _trainable(model)
    params = [p for _, p in names_params]
    for p in params:
        p.requires_grad_(True)
    sketches, norms, losses, first_digit = [], [], [], []
    try:
        for it in items[:n_ex]:
            model.zero_grad(set_to_none=True)
            loss = example_loss(model, tokenizer, it, device)
            loss.backward()
            grads = [p.grad for p in params if p.grad is not None]
            norms.append(math.sqrt(sum((g.double() ** 2).sum().item() for g in grads)))
            sketches.append(count_sketch(grads, D).cpu())
            losses.append(loss.item()); first_digit.append(int(str(abs(it[5]))[0]))
    finally:
        model.zero_grad(set_to_none=True)
        for p in params:
            p.requires_grad_(False)
    S = torch.stack(sketches)                         # (N, D)
    G = S @ S.T
    nrm = G.diagonal().sqrt()
    C = G / (nrm[:, None] * nrm[None, :])
    off = C[~torch.eye(len(S), dtype=torch.bool)]
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    erank = (ev.sum() ** 2 / (ev ** 2).sum()).item()
    coherence = (S.mean(0).norm() / S.norm(dim=1).mean()).item()
    fd = torch.tensor(first_digit)
    Ky = (fd[:, None] == fd[None, :]).double()
    def center(K):
        n = len(K); Hc = torch.eye(n, dtype=K.dtype) - 1.0 / n
        return Hc @ K @ Hc
    Gc, Kc = center(G), center(Ky)
    kta = ((Gc * Kc).sum() / (Gc.norm() * Kc.norm())).item()
    return {"n": len(S), "sketch_dim": D, "loss_mean": sum(losses) / len(losses),
            "grad_norm_mean": sum(norms) / len(norms),
            "pairwise_cos_mean": off.mean().item(), "pairwise_cos_median": off.median().item(),
            "coherence_mean_over_mean_norm": coherence, "gram_effective_rank": erank,
            "kernel_target_alignment_first_digit": kta}


# ------------------------------------------------------------------ hessian
def batch_loss(model, tokenizer, items, device):
    tot = 0.0
    for it in items:
        tot = tot + example_loss(model, tokenizer, it, device)
    return tot / len(items)


def metric_hessian(model, tokenizer, items, device, n_ex, power_iters, hutch_probes, seed=316):
    params = [p for _, p in _trainable(model)]
    for p in params:
        p.requires_grad_(True)
    batch = items[:n_ex]
    try:
        loss = batch_loss(model, tokenizer, batch, device)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gnorm2 = sum((g.detach().double() ** 2).sum() for g in grads).item()

        def hvp(vs):
            gv = sum((g * v).sum() for g, v in zip(grads, vs))
            return [h.detach() for h in torch.autograd.grad(gv, params, retain_graph=True)]

        # gradient sharpness and one-step gain
        gdir = [g.detach() / math.sqrt(gnorm2) for g in grads]
        Hg = hvp(gdir)
        gHg_unit = sum((h.double() * v.double()).sum() for h, v in zip(Hg, gdir)).item()
        one_step_gain = gnorm2 / (2 * gHg_unit) if gHg_unit > 0 else float("inf")
        # top eigenvalue by power iteration
        g_ = torch.Generator(device="cpu").manual_seed(seed)
        v = [torch.randn(p.shape, generator=g_).to(device) for p in params]
        nv = math.sqrt(sum((x.double() ** 2).sum() for x in v).item()); v = [x / nv for x in v]
        def power(shift):
            vv = [torch.randn(p.shape, generator=g_).to(device) for p in params]
            n0 = math.sqrt(sum((x.double() ** 2).sum() for x in vv).item()); vv = [x / n0 for x in vv]
            mu = float("nan")
            for _ in range(power_iters):
                Hv = hvp(vv)
                if shift:
                    Hv = [h - shift * x for h, x in zip(Hv, vv)]
                mu = sum((h.double() * x.double()).sum() for h, x in zip(Hv, vv)).item()
                nv = math.sqrt(sum((h.double() ** 2).sum() for h in Hv).item())
                vv = [h / nv for h in Hv]
            return mu + shift
        lam = power(0.0)            # largest |lambda|
        lam_other = power(lam)      # the opposite end of the spectrum
        lam_max, lam_min = max(lam, lam_other), min(lam, lam_other)
        # Hutchinson trace
        tr = 0.0
        for i in range(hutch_probes):
            z = [torch.randint(0, 2, p.shape, generator=g_).to(device).float() * 2 - 1 for p in params]
            Hz = hvp(z)
            tr += sum((h.double() * x.double()).sum() for h, x in zip(Hz, z)).item()
        tr /= hutch_probes
    finally:
        model.zero_grad(set_to_none=True)
        for p in params:
            p.requires_grad_(False)
    n_par = sum(p.numel() for p in params)
    return {"n": len(batch), "loss": loss.item(), "grad_norm": math.sqrt(gnorm2),
            "top_eigenvalue": lam, "lambda_max": lam_max, "lambda_min": lam_min,
            "neg_share": (-lam_min / (lam_max - lam_min)) if lam_max > lam_min else float("nan"),
            "trace_hutchinson": tr, "trace_over_params": tr / n_par,
            "grad_sharpness_gHg_over_g2": gHg_unit, "one_step_gain_nats": one_step_gain,
            "top_eig_share_of_trace": lam / tr if tr else float("nan")}


# ------------------------------------------------------------------ llc
def metric_llc(model, tokenizer, items, device, n_ex, steps, eps, gamma, n_data, bs, seed=316):
    params = [p for _, p in _trainable(model)]
    w0 = [p.detach().clone() for p in params]
    for p in params:
        p.requires_grad_(True)
    beta = 1.0 / math.log(n_data)
    rng = random.Random(seed); gen = torch.Generator(device="cpu").manual_seed(seed)
    pool = items[:n_ex]
    try:
        with torch.no_grad():
            L0 = batch_loss(model, tokenizer, pool[: min(len(pool), 4 * bs)], device).item()
        trace = []
        for t in range(steps):
            batch = rng.sample(pool, bs)
            loss = batch_loss(model, tokenizer, batch, device)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, g, w in zip(params, grads, w0):
                    noise = torch.randn(p.shape, generator=gen).to(device) * math.sqrt(eps)
                    p.add_(-(eps / 2) * (n_data * beta * g + gamma * (p - w)) + noise)
            trace.append(loss.item())
        burn = steps // 4
        L_mean = sum(trace[burn:]) / max(1, len(trace) - burn)
        llc = n_data * beta * (L_mean - L0)
    finally:
        with torch.no_grad():
            for p, w in zip(params, w0):
                p.copy_(w)
            for p in params:
                p.requires_grad_(False)
    return {"n_pool": len(pool), "steps": steps, "eps": eps, "gamma": gamma, "n_data": n_data,
            "beta": beta, "batch": bs, "loss_at_w0": L0, "loss_sgld_mean": L_mean,
            "llc_estimate": llc, "loss_trace": [round(v, 4) for v in trace]}


# ------------------------------------------------------------------ compare
HEADLINE = [
    ("pref", "logit_diff_mean", "hidden pref (nats)"),
    ("pref", "top1_acc", "top-1"),
    ("geometry", "pc1_last", "state PC1 (last)"),
    ("probe", "r2_answer_best", "probe R2 answer"),
    ("probe", "r2_operands_best", "probe R2 operands"),
    ("probe", "first_digit_acc_best", "probe 1st-digit acc"),
    ("attn", "max_head_mass", "attn max head->operands"),
    ("attn", "heads_over_0.25", "heads >0.25"),
    ("grad", "pairwise_cos_mean", "grad cos"),
    ("grad", "coherence_mean_over_mean_norm", "grad coherence"),
    ("grad", "gram_effective_rank", "grad erank"),
    ("grad", "kernel_target_alignment_first_digit", "KTA"),
    ("hessian", "lambda_max", "H lambda max"),
    ("hessian", "lambda_min", "H lambda min"),
    ("hessian", "grad_sharpness_gHg_over_g2", "gHg/g2"),
    ("hessian", "one_step_gain_nats", "one-step gain"),
    ("llc", "llc_estimate", "LLC"),
]


def cmd_compare(args):
    rows = []
    for tag in args.tags:
        p = Path(args.out_dir) / f"prefit_{tag}.json"
        rows.append(json.loads(p.read_text()) if p.is_file() else {"tag": tag})
    w = max(14, max(len(r.get("tag", "")) for r in rows) + 2)
    print("[prefit] metric".ljust(34) + "".join(r.get("tag", "?").rjust(w) for r in rows))
    for blk, key, label in HEADLINE:
        vals = []
        for r in rows:
            v = r.get(blk, {}).get(key) if isinstance(r.get(blk), dict) else None
            vals.append("--" if v is None else (f"{v:.3f}" if abs(v) < 1e4 else f"{v:.3g}"))
        print(("[prefit] " + label).ljust(34) + "".join(v.rjust(w) for v in vals))
    for r in rows:
        das = r.get("das", {}).get("layers", {})
        for L, res in das.items():
            best = max((v["flip_frac"], k) for k, v in res.items() if k.startswith("k"))
            print(f"[prefit] das {r['tag']} L{L}: best flip {best[0]:.3f} ({best[1]}), "
                  f"full {res['full']['flip_frac']:.3f}, none {res['none']['flip_frac']:.3f}, "
                  f"random {res[best[1]]['random_flip_frac']:.3f}")
        dcm = r.get("dcm", {}).get("roles", {})
        for role, res in dcm.items():
            if "n_heads" in res:
                print(f"[prefit] dcm {r['tag']} {role}: {res['n_heads']} heads, ld-flip "
                      f"{res['ld_flip_frac']:.3f} (ceiling {res['ld_flip_ceiling']:.3f})")
    return 0


# ------------------------------------------------------------------ driver
METRICS = ("pref", "geometry", "probe", "das", "dcm", "attn", "grad", "hessian", "llc")


def run_metric(name, model, tokenizer, items, args):
    dev = args.device
    if name == "pref":
        return metric_pref(model, tokenizer, items[: args.n], dev, args.batch_size)
    if name == "geometry":
        return metric_geometry(model, tokenizer, items[: args.n], dev, args.batch_size)
    if name == "probe":
        return metric_probe(model, tokenizer, items[: args.n_probe], dev, args.batch_size)
    if name == "das":
        return metric_das(model, tokenizer, items[: args.n_probe], dev, args.das_layers, args.das_ks,
                          args.das_train, args.das_test, args.das_steps, args.das_lr)
    if name == "dcm":
        return metric_dcm(model, tokenizer, dev, args.dcm_pairs, args.dcm_lam, args.dcm_steps, args.dcm_lr)
    if name == "attn":
        return metric_attn(model, tokenizer, items[: args.n], dev, args.batch_size)
    if name == "grad":
        return metric_grad(model, tokenizer, items, dev, args.grad_n, args.sketch_dim)
    if name == "hessian":
        return metric_hessian(model, tokenizer, items, dev, args.hess_n, args.power_iters, args.hutch)
    if name == "llc":
        return metric_llc(model, tokenizer, items, dev, args.llc_pool, args.llc_steps, args.llc_eps,
                          args.llc_gamma, args.llc_n_data, args.llc_batch)
    raise ValueError(name)


def cmd_run(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    names = list(METRICS) if args.metric == "all" else [args.metric]
    model = load(args.model, args.device, eager=("attn" in names))
    items = problems(max(args.n, args.n_probe), tokenizer)
    out_dir = Path(args.out_dir)
    for name in names:
        t0 = time.time()
        print(f"[prefit] {args.tag}: {name} on {args.model}")
        block = run_metric(name, model, tokenizer, items, args)
        block["seconds"] = round(time.time() - t0, 1)
        write_block(out_dir, args.tag, name, block, args.model)
    return 0


def cmd_smoke(args):
    """CPU code-path check on a tiny random Llama and any cached fast tokenizer."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).float().eval()
    try:
        model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
    for p in model.parameters():
        p.requires_grad_(False)
    items = problems(96, tok)
    args.device, args.batch_size, args.n, args.n_probe = "cpu", 16, 64, 96
    args.das_layers, args.das_ks, args.das_train, args.das_test, args.das_steps, args.das_lr = [1], [2], 24, 12, 3, 1e-2
    args.dcm_pairs, args.dcm_lam, args.dcm_steps, args.dcm_lr = 16, 0.02, 3, 0.05
    args.grad_n, args.sketch_dim = 6, 512
    args.hess_n, args.power_iters, args.hutch = 4, 2, 2
    args.llc_pool, args.llc_steps, args.llc_eps, args.llc_gamma, args.llc_n_data, args.llc_batch = 8, 4, 1e-6, 100.0, 1000, 2
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    for name in METRICS:
        block = run_metric(name, model, tok, items, args)
        write_block(out_dir, "smoke", name, block, "tiny")
    args.tags = ["smoke"]
    return cmd_compare(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metric", choices=list(METRICS) + ["all", "compare", "smoke"])
    ap.add_argument("tags", nargs="*", help="compare: tags")
    ap.add_argument("--model", help="zoo run id or hub id")
    ap.add_argument("--tag", help="short name for the output json")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n", type=int, default=256, help="problems for pref / geometry / attn")
    ap.add_argument("--n-probe", type=int, default=1024, help="problems for probe / das pairs")
    ap.add_argument("--das-layers", type=int, nargs="+", default=[8, 12, 14, 15])
    ap.add_argument("--das-ks", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--das-train", type=int, default=256)
    ap.add_argument("--das-test", type=int, default=128)
    ap.add_argument("--das-steps", type=int, default=60)
    ap.add_argument("--das-lr", type=float, default=1e-2)
    ap.add_argument("--dcm-pairs", type=int, default=128)
    ap.add_argument("--dcm-lam", type=float, default=0.02)
    ap.add_argument("--dcm-steps", type=int, default=200)
    ap.add_argument("--dcm-lr", type=float, default=0.05)
    ap.add_argument("--grad-n", type=int, default=64)
    ap.add_argument("--sketch-dim", type=int, default=8192)
    ap.add_argument("--hess-n", type=int, default=16)
    ap.add_argument("--power-iters", type=int, default=20)
    ap.add_argument("--hutch", type=int, default=8)
    ap.add_argument("--llc-pool", type=int, default=256)
    ap.add_argument("--llc-steps", type=int, default=200)
    ap.add_argument("--llc-eps", type=float, default=1e-6)
    ap.add_argument("--llc-gamma", type=float, default=100.0)
    ap.add_argument("--llc-n-data", type=int, default=4_000_000)
    ap.add_argument("--llc-batch", type=int, default=8)
    args = ap.parse_args()
    if args.metric == "compare":
        if not args.tags:
            ap.error("compare needs tags")
        return cmd_compare(args)
    if args.metric == "smoke":
        return cmd_smoke(args)
    if not args.model or not args.tag:
        ap.error("--model and --tag are required")
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
