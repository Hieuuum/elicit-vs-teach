"""Lens depth: at what layer does the answer form — logit lens, J-lens, R-lens —
and is it already "verbalizable" in the parent before any target training?

For N problems rendered on one surface, the residual stream h_l at the answer
position (last prompt token; the first answer token is predicted there) is
decoded at every layer with three lenses:

  logit lens   unembed(norm(h_l))                       (nostalgebraist 2020)
  J-lens       unembed(norm(J_l h_l))                   (Anthropic 2026, global
               workspace) with J_l = E[ d h_final,t' / d h_l,t ], expectation
               over story prompts, source positions t and all current-and-
               future target positions t' >= t; h_final = the final residual
               stream (pre-norm).
  R-lens       same transport, but the backward pass follows layerwise
               relevance propagation (LessWrong 2026, "R-lens: making J-lens
               more faithful on early layers"): LN-rule (the RMSNorm
               normalisation factor is detached, making the norm linear),
               identity-rule (SiLU's gradient replaced by sigmoid(z)), half-rule
               (relevance split evenly between the gate and up paths of the
               gated MLP instead of the bilinear product rule); linear layers
               and attention are left as ordinary gradients. Forward values
               are unchanged — only the transported quantity differs.

The average matrices are formed EXPLICITLY (d x d per layer) by reverse mode:
one backward pass with cotangent e_k on sum_t' h_final,t' yields row k of
sum_t J_{l,t} for EVERY layer at once; cotangents are batched (vmap) with a
looped fallback. At the last layer J = R = I, so both lenses must reduce to
the logit lens there (checked and printed).

Per layer and lens: top-1 accuracy on the correct first answer token, its mean
log-prob and median rank, and the lens logit-diff (correct minus the answer of
a mismatched problem — positive = the answer is present even when the model
never emits it). Formation depth per example = first layer from which the
correct token is top-1 through to the end ("settled").

Owner's predictions (2026-09-09):
- Elicitation: the answer forms in the MIDDLE layers (pre-existing engine
  works early); in the pre-elicit parent it is already present in J-space
  (verbalizable but not emitted) — fine-tuning surfaces it, the lens
  trajectory barely moves.
- Teaching: the answer forms only in the FINAL layers; nothing in J-space in
  the blank base at any layer — it appears only as the capability is built.

Usage:
    python3 lens_depth.py run --run-id evt-ts1b-op-bridge-mix --surface bare_nl \
        --out lens_latent_nl [--lenses logit jlens rlens] [--n 256] [--jac-prompts 24]
    python3 lens_depth.py compare latent=lens_latent_nl.json elicited=... [--plot lens.png]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET, render_probe  # noqa: E402
from resid_shift import TOKENIZER, generic_ids, load_fp32  # noqa: E402

from geode.arith.formats import true_answer  # noqa: E402

STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))


# ------------------------------------------------------------------ hooks
class LensTaps:
    """Record the residual after the embedding and after every layer. In grad
    mode the embedding output becomes a fresh leaf (requires_grad) so the
    whole downstream graph hangs off the residual stream, not the weights."""

    def __init__(self, model):
        self.acts: list[torch.Tensor] = []
        self.grad_mode = False
        self.handles = [model.model.embed_tokens.register_forward_hook(self._mk(-1))]
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_hook(self._mk(i)))

    def _mk(self, idx):
        def hook(_m, _i, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            if self.grad_mode:
                if idx == -1:
                    h = h.detach().requires_grad_(True)
                    self.acts = [h]
                    return (h,) + tuple(out[1:]) if is_tuple else h
                self.acts.append(h)
                return None
            if idx == -1:
                self.acts = [h.detach()]
            else:
                self.acts.append(h.detach())
            return None

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def decode(model, h: torch.Tensor) -> torch.Tensor:
    """unembed(norm(h)) -> logits (…, V); scale-invariant in h (RMSNorm)."""
    return model.lm_head(model.model.norm(h))


# ------------------------------------------------------- LRP (R-lens) rules
@contextlib.contextmanager
def lrp_rules(model):
    """Same forward values, LRP backward: LN-rule on RMSNorm, identity-rule on
    SiLU, half-rule on the gated product. Class-level monkeypatch, restored on
    exit. Attention and all linear maps untouched (0-rule == gradient)."""
    norm_cls = type(model.model.norm)
    mlp_cls = type(model.model.layers[0].mlp)
    assert getattr(model.config, "hidden_act", "silu") == "silu", model.config.hidden_act
    orig_norm, orig_mlp = norm_cls.forward, mlp_cls.forward

    def rms_ln_rule(self, x):
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon).detach()   # LN-rule
        return self.weight * xf.to(dtype)

    def mlp_lrp(self, x):
        a = self.gate_proj(x)
        act = a * torch.sigmoid(a).detach()                             # identity-rule
        u = self.up_proj(x)
        prod = 0.5 * (act * u.detach() + act.detach() * u)             # half-rule
        return self.down_proj(prod)

    norm_cls.forward, mlp_cls.forward = rms_ln_rule, mlp_lrp
    try:
        yield
    finally:
        norm_cls.forward, mlp_cls.forward = orig_norm, orig_mlp


# ------------------------------------------------- average Jacobian matrices
def average_jacobians(model, taps, story_ids, device, k_batch, mode, log=True):
    """J[l] (d x d) for l in {emb, layer 0..L-1}: rows k = d(sum_t' h_final,t',k)
    / d h_{l,t}, averaged over source positions t and prompts. mode 'grad' =
    J-lens, 'lrp' = R-lens. Returns (J tensor (L+1, d, d) on CPU, info dict)."""
    d = model.config.hidden_size
    n_layers = len(model.model.layers) + 1
    P, T = story_ids.shape
    J = torch.zeros(n_layers, d, d, dtype=torch.float64)
    ctx = lrp_rules(model) if mode == "lrp" else contextlib.nullcontext()
    batched_ok = k_batch > 1
    taps.grad_mode = True
    with ctx, torch.enable_grad():
        for p in range(P):
            ids = story_ids[p : p + 1].to(device)
            model(input_ids=ids)
            acts = list(taps.acts)
            target = acts[-1].sum(dim=1)[0]  # (d,): summed over all target positions
            k0 = 0
            while k0 < d:
                K = min(k_batch if batched_ok else 1, d - k0)
                cot = torch.zeros(K, d, device=device)
                cot[torch.arange(K), k0 + torch.arange(K)] = 1.0
                try:
                    if batched_ok:
                        grads = torch.autograd.grad(target, acts, grad_outputs=cot,
                                                    retain_graph=True, is_grads_batched=True)
                        rows = [g[:, 0].mean(dim=1) for g in grads]            # (K, d)
                    else:
                        grads = torch.autograd.grad(target, acts, grad_outputs=cot[0],
                                                    retain_graph=True)
                        rows = [g[0].mean(dim=0, keepdim=True) for g in grads]  # (1, d)
                except RuntimeError as e:  # vmap unsupported somewhere: fall back
                    if not batched_ok:
                        raise
                    print(f"[lens] batched grads failed ({str(e)[:80]}…); looping cotangents")
                    batched_ok = False
                    continue
                for li, r in enumerate(rows):
                    J[li, k0 : k0 + K] += r.double().cpu() / P
                k0 += K
            del acts, target
            if log and (p % max(1, P // 4) == 0 or p == P - 1):
                print(f"[lens] {mode}: prompt {p + 1}/{P} done"
                      + ("" if batched_ok else " (looped)"))
    taps.grad_mode = False
    eye_dev = (J[-1] - torch.eye(d, dtype=J.dtype)).abs().max().item()
    return J.float(), {"batched": batched_ok, "last_layer_identity_maxdev": eye_dev}


# ----------------------------------------------------------------- metrics
@torch.no_grad()
def score(logits: torch.Tensor, correct: torch.Tensor, distract: torch.Tensor) -> dict:
    correct, distract = correct.to(logits.device), distract.to(logits.device)
    lp = F.log_softmax(logits.float(), dim=-1)
    lp_c = lp.gather(1, correct[:, None]).squeeze(1)
    top1 = logits.argmax(-1) == correct
    rank = (logits > logits.gather(1, correct[:, None])).sum(-1) + 1
    ld = (logits.gather(1, correct[:, None]) - logits.gather(1, distract[:, None])).squeeze(1)
    return {"top1": top1.cpu().tolist(), "logprob": lp_c.cpu().tolist(),
            "rank": rank.cpu().tolist(), "logit_diff": ld.cpu().tolist()}


def summarize_layers(per_layer: list[dict]) -> dict:
    n = len(per_layer[0]["top1"])
    top1 = torch.tensor([[bool(x) for x in d["top1"]] for d in per_layer])  # (L+1, N)
    settled = []
    for j in range(n):
        col = top1[:, j]
        if not bool(col[-1]):
            settled.append(None)
            continue
        k = len(col) - 1
        while k > 0 and bool(col[k - 1]):
            k -= 1
        settled.append(k - 1)  # -1 = embedding
    s_valid = sorted(d for d in settled if d is not None)
    layers = list(range(-1, len(per_layer) - 1))
    acc = [sum(d["top1"]) / n for d in per_layer]
    ld = [sum(d["logit_diff"]) / n for d in per_layer]
    med_rank = [sorted(d["rank"])[n // 2] for d in per_layer]
    lp = [sum(d["logprob"]) / n for d in per_layer]

    def first(pred):
        for layer, v in zip(layers, pred):
            if v:
                return layer
        return None

    return {
        "layers": layers, "top1_acc": acc, "mean_logit_diff": ld,
        "median_rank": med_rank, "mean_logprob": lp,
        "final_top1_acc": acc[-1],
        "settled_depth_median": (s_valid[len(s_valid) // 2] if s_valid else None),
        "settled_depth_q25": (s_valid[len(s_valid) // 4] if s_valid else None),
        "settled_depth_q75": (s_valid[(3 * len(s_valid)) // 4] if s_valid else None),
        "n_settled": len(s_valid),
        "first_layer_acc_ge_half": first(a >= 0.5 for a in acc),
        "first_layer_logit_diff_gt_1": first(v > 1.0 for v in ld),
        "first_layer_median_rank_le_10": first(r <= 10 for r in med_rank),
        "settled_depth_all": settled,
    }


# --------------------------------------------------------------------- run
def cmd_run(args) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_parquet(EVAL_PARQUET)
    rows = df.iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + args.n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in rows.itertuples()]
    prompts = [render_probe(args.surface, a, b, op, 0)[0] for a, b, op in triples]
    answers = [true_answer(a, b, op) for a, b, op in triples]
    first_tok = lambda x: tokenizer(str(x), add_special_tokens=False)["input_ids"][0]  # noqa: E731
    correct = torch.tensor([first_tok(a) for a in answers])
    dis = []
    for i in range(len(answers)):  # mismatched-problem distractor with a different first token
        k = 1
        while first_tok(answers[(i + k) % len(answers)]) == correct[i].item() and k < len(answers):
            k += 1
        dis.append(first_tok(answers[(i + k) % len(answers)]))
    distract = torch.tensor(dis)
    print(f"[lens] {args.run_id}: surface {args.surface} x{len(prompts)}  e.g. {prompts[0]!r} "
          f"-> {answers[0]} (first token {tokenizer.decode([correct[0].item()])!r})")

    model = load_fp32(args.run_id, args.device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    if not args.sdpa:  # eager attention: plain matmul/softmax, safe under vmap'd backward
        model.config._attn_implementation = "eager"
    taps = LensTaps(model)
    device = args.device
    n_layers = len(model.model.layers) + 1

    # --- task states at the requested positions + the model's own logits
    tokenizer.padding_side = "left"
    H = {pos: [[] for _ in range(n_layers)] for pos in args.positions}
    own_logits = []
    with torch.no_grad():
        for s in range(0, len(prompts), args.batch_size):
            enc = tokenizer(prompts[s : s + args.batch_size], return_tensors="pt", padding=True,
                            add_special_tokens=False)
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            own_logits.append(out.logits[:, -1].float().cpu())
            for pos in args.positions:
                for li in range(n_layers):
                    H[pos][li].append(taps.acts[li][:, pos].float().cpu())
    H = {pos: [torch.cat(x, 0) for x in H[pos]] for pos in args.positions}
    own = score(torch.cat(own_logits, 0), correct, distract)
    own_acc = sum(own["top1"]) / len(prompts)
    print(f"[lens] model's own first-token accuracy: {own_acc:.3f}  "
          f"logit-diff {sum(own['logit_diff']) / len(prompts):+.2f}")
    results = {"run_id": args.run_id, "surface": args.surface, "n": len(prompts),
               "own_first_token_acc": own_acc,
               "own_logit_diff": sum(own["logit_diff"]) / len(prompts), "positions": {},
               "jacobian_info": {}}
    logit_scores = {}
    if "logit" in args.lenses:
        with torch.no_grad():
            for pos in args.positions:
                logit_scores[pos] = [score(decode(model, H[pos][li].to(device)), correct, distract)
                                     for li in range(n_layers)]
                acc_l = logit_scores[pos][-1]["top1"]
                print(f"[lens] logit lens scored (pos {pos}); last layer acc "
                      f"{sum(acc_l) / len(acc_l):.3f} == own {own_acc:.3f}")

    # --- average Jacobians (J-lens: gradient; R-lens: LRP backward)
    mats = {}
    need = [ln for ln in ("jlens", "rlens") if ln in args.lenses]
    if need:
        story_ids, path = generic_ids(tokenizer, args.generic_text, args.jac_prompts, args.story_len)
        print(f"[lens] Jacobian corpus: {story_ids.shape[0]} stories x {args.story_len} tokens "
              f"from {path}")
        prev_prec = torch.get_float32_matmul_precision()
        if args.tf32 and device.startswith("cuda"):
            # Jacobian phase only: ~8x faster matmuls; the last-layer identity and the
            # lens==logit-lens checks stay exact (no matmul on that path).
            torch.set_float32_matmul_precision("high")
            print("[lens] TF32 matmuls ON for the Jacobian phase (--no-tf32 to disable)")
        for ln in need:
            mode = "grad" if ln == "jlens" else "lrp"
            J, info = average_jacobians(model, taps, story_ids, device, args.k_batch, mode)
            mats[ln] = J
            results["jacobian_info"][ln] = info
            print(f"[lens] {ln}: last-layer matrix vs identity, max |dev| "
                  f"{info['last_layer_identity_maxdev']:.2e}  (batched grads: {info['batched']})")
            if args.save_jacobians:
                torch.save(J.half(), f"{args.out}_{ln}_J.pt")
        torch.set_float32_matmul_precision(prev_prec)
        if "jlens" in mats and "rlens" in mats:
            rel = [((mats['rlens'][li] - mats['jlens'][li]).norm()
                    / mats['jlens'][li].norm().clamp_min(1e-12)).item() for li in range(n_layers)]
            results["jacobian_info"]["rel_diff_R_vs_J_by_layer"] = rel
            print("[lens] ||R-J||/||J|| by layer: "
                  + " ".join(f"L{li - 1}:{v:.2f}" for li, v in enumerate(rel)))

    for pos in args.positions:
        per = {}
        if pos in logit_scores:
            per["logit"] = logit_scores[pos]
        with torch.no_grad():
            for ln, J in mats.items():
                per[ln] = [score(decode(model, (H[pos][li] @ J[li].T).to(device)), correct,
                                 distract) for li in range(n_layers)]
        if "logit" in per:
            for ln in mats:  # J = R = I at the last layer: lenses must coincide there
                a, b = per[ln][-1]["logit_diff"], per["logit"][-1]["logit_diff"]
                dev = max(abs(x - y) for x, y in zip(a, b))
                print(f"[lens] check: {ln} at last layer vs logit lens, max |d logit-diff| {dev:.2e}")
        summ = {ln: summarize_layers(v) for ln, v in per.items()}
        results["positions"][str(pos)] = summ
        print(f"[lens] position {pos}:")
        print("[lens] layer | " + " | ".join(f"{ln}: acc   ld    rank" for ln in summ))
        for i, layer in enumerate(next(iter(summ.values()))["layers"]):
            cells = [f"{ln}: {s_['top1_acc'][i]:.2f} {s_['mean_logit_diff'][i]:+5.1f} "
                     f"{s_['median_rank'][i]:6d}" for ln, s_ in summ.items()]
            print(f"[lens] {layer:5d} | " + " | ".join(cells))
        for ln, s_ in summ.items():
            print(f"[lens] SUMMARY {ln:<5} pos {pos}: final acc {s_['final_top1_acc']:.3f}; "
                  f"settled depth median L{s_['settled_depth_median']} "
                  f"(q25 L{s_['settled_depth_q25']}, q75 L{s_['settled_depth_q75']}, "
                  f"n={s_['n_settled']}); first layer acc>=0.5: L{s_['first_layer_acc_ge_half']}; "
                  f"first layer logit-diff>1: L{s_['first_layer_logit_diff_gt_1']}; "
                  f"first layer median rank<=10: L{s_['first_layer_median_rank_le_10']}")
    taps.remove()
    out = Path(f"{args.out}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[lens] wrote {out}")
    return 0


# ----------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    runs = []
    for spec in args.runs:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        runs.append((label, json.loads(Path(path).read_text())))
    pos = args.position
    lenses = ["logit", "jlens", "rlens"]
    print(f"[lens] position {pos}")
    print("[lens] " + f"{'metric':<34}" + "".join(f"{lab:>14}" for lab, _ in runs))

    def cell(v):
        if v is None:
            return f"{'--':>14}"
        return f"{v:14.3f}" if isinstance(v, float) else f"{v:>14}"

    print("[lens] " + f"{'own first-token acc':<34}"
          + "".join(cell(r["own_first_token_acc"]) for _, r in runs))
    for ln in lenses:
        for key, name in (("final_top1_acc", "final acc"),
                          ("settled_depth_median", "settled depth median"),
                          ("first_layer_acc_ge_half", "first layer acc>=0.5"),
                          ("first_layer_logit_diff_gt_1", "first layer ld>1"),
                          ("first_layer_median_rank_le_10", "first layer med rank<=10")):
            vals = [cell(r["positions"][pos][ln][key]) if ln in r["positions"][pos] else cell(None)
                    for _, r in runs]
            print(f"[lens] {ln + ': ' + name:<34}" + "".join(vals))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        avail = [ln for ln in lenses if all(ln in r["positions"][pos] for _, r in runs)]
        fig, axes = plt.subplots(2, len(avail), figsize=(4.2 * len(avail), 6.4), squeeze=False)
        colors = ["#9a6316", "#c98a2e", "#2c6b74", "#6aa3ad", "#7f8c8d", "#8e44ad"]
        for j, ln in enumerate(avail):
            for (label, r), col in zip(runs, colors):
                s_ = r["positions"][pos][ln]
                axes[0, j].plot(s_["layers"], s_["top1_acc"], "o-", ms=3, color=col, label=label)
                axes[1, j].plot(s_["layers"], s_["mean_logit_diff"], "o-", ms=3, color=col,
                                label=label)
            axes[0, j].set(title=f"{ln}: top-1 on answer token", xlabel="layer", ylim=(-0.02, 1.02))
            axes[1, j].set(title=f"{ln}: logit-diff (answer − mismatched)", xlabel="layer")
            axes[1, j].axhline(0, color="k", lw=0.5)
            for ax in (axes[0, j], axes[1, j]):
                ax.grid(alpha=0.3)
            axes[0, j].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        print(f"[lens] wrote {args.plot}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True, help="zoo run id / hub id / checkpoint dir")
    r.add_argument("--surface", default="bare_nl")
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=256)
    r.add_argument("--positions", type=int, nargs="+", default=[-1],
                   help="prompt positions to decode (-1 = answer position)")
    r.add_argument("--lenses", nargs="+", default=["logit", "jlens", "rlens"],
                   choices=["logit", "jlens", "rlens"])
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--generic-text", default=None)
    r.add_argument("--story-len", type=int, default=128)
    r.add_argument("--jac-prompts", type=int, default=24,
                   help="stories to average the Jacobian over (paper: saturates ~100)")
    r.add_argument("--k-batch", type=int, default=64,
                   help="cotangents per batched backward (1 = plain loop)")
    r.add_argument("--sdpa", action="store_true",
                   help="keep sdpa attention (default: eager, so batched backward works)")
    r.add_argument("--no-save-jacobians", dest="save_jacobians", action="store_false",
                   help="skip writing <out>_{jlens,rlens}_J.pt (fp16, ~140 MB each)")
    r.add_argument("--no-tf32", dest="tf32", action="store_false",
                   help="keep full fp32 matmuls in the Jacobian phase (~8x slower)")
    r.add_argument("--tokenizer", default=TOKENIZER)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="label=path.json ...")
    c.add_argument("--position", default="-1")
    c.add_argument("--plot", default=None)
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
