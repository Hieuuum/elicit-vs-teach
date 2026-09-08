"""Desiderata Component Masking (DCM; Prakash et al. 2024) for arithmetic roles.

For each functional DESIDERATUM, build counterfactual prompt pairs that differ
in exactly one variable, then learn a sparse mask over the 528 nodes such that
mixing the counterfactual activations into the clean run (at the masked nodes
only) makes the model produce the COUNTERFACTUAL answer. The selected nodes
are the components that carry that variable — a functional role.

Desiderata (token-length-matched by construction; pairs whose first answer
tokens coincide are dropped):
  operand_a   (a,b,op) -> (a',b,op)   a' same digit count
  operand_b   (a,b,op) -> (a,b',op)
  operation   (a,b,+) -> (a,b,-)      op surface only; a>b so both answers
                                      are positive and first tokens differ

Owner's regime predictions (2026-09-09): elicitation PRESERVES roles (the
same heads fetch operands / detect the operation before and after target
fine-tuning); teaching CREATES roles (heads with no prior function are
forced into them). Compare mode reports per-role Jaccard between two
models' role sets.

Modes:
  learn:    python3 dcm_roles.py learn (--run-id R | --model M) --surface {bare_op,bare_nl,bridge}
            --out stem [--roles operand_a operand_b operation] [--lam 0.02] [--steps 200]
  compare:  python3 dcm_roles.py compare A_stem B_stem

GPU, box-only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geode.arith.formats import digits, true_answer  # noqa: E402
from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET, render_probe  # noqa: E402

ROLES = ("operand_a", "operand_b", "operation")


# ------------------------------------------------------------ counterfactuals
def make_pairs(surface: str, role: str, n: int, tokenizer, seed: int = 316):
    df = pd.read_parquet(EVAL_PARQUET)
    rows = df.iloc[DEFAULT_ROW_OFFSET + 2000 : DEFAULT_ROW_OFFSET + 2000 + 20 * n]
    rng = random.Random(seed)
    pairs = []
    for r in rows.itertuples():
        a, b, op = int(r.a), int(r.b), str(r.op)
        if role == "operand_a":
            lo, hi = 10 ** (digits(a) - 1), 10 ** digits(a) - 1
            a2 = rng.randint(lo, hi)
            if a2 == a:
                continue
            cf = (a2, b, op)
        elif role == "operand_b":
            lo, hi = 10 ** (digits(b) - 1), 10 ** digits(b) - 1
            b2 = rng.randint(lo, hi)
            if b2 == b:
                continue
            cf = (a, b2, op)
        else:  # operation: + <-> -, require a > b so both answers positive
            if surface != "bare_op" or a <= b:
                continue
            cf = (a, b, "-" if op == "+" else "+")
        clean_p = render_probe(surface, a, b, op, 0)[0]
        cf_p = render_probe(surface, *cf, 0)[0]
        ci = tokenizer(clean_p, add_special_tokens=False)["input_ids"]
        xi = tokenizer(cf_p, add_special_tokens=False)["input_ids"]
        if len(ci) != len(xi):
            continue
        ct = tokenizer(str(true_answer(a, b, op)), add_special_tokens=False)["input_ids"][0]
        xt = tokenizer(str(true_answer(*cf)), add_special_tokens=False)["input_ids"][0]
        if ct == xt:
            continue
        pairs.append((ci, xi, ct, xt))
        if len(pairs) >= n:
            break
    return pairs


# -------------------------------------------------------------------- mixing
class MixTaps:
    """Hooks that mix counterfactual activations into the clean run per node,
    weighted by a differentiable mask: x <- (1-m) x + m x_cf."""

    def __init__(self, model):
        cfg = model.config
        self.H = cfg.num_attention_heads
        self.dh = cfg.hidden_size // cfg.num_attention_heads
        self.L = cfg.num_hidden_layers
        self.mode = "off"
        self.cf: dict[tuple, torch.Tensor] = {}
        self.mask_attn = None  # (L, H)
        self.mask_mlp = None   # (L,)
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._a(i)))
            self.handles.append(layer.mlp.down_proj.register_forward_hook(self._m(i)))

    def _a(self, i):
        def hook(_mod, inputs):
            x = inputs[0]
            if self.mode == "capture":
                self.cf[("attn", i)] = x.detach()
                return None
            if self.mode == "mix":
                xh = x.view(*x.shape[:-1], self.H, self.dh)
                ch = self.cf[("attn", i)].view(*xh.shape).to(xh.dtype)
                m = self.mask_attn[i].to(xh.dtype).view(1, 1, self.H, 1)
                return (((1 - m) * xh + m * ch).view(*x.shape),)
            return None
        return hook

    def _m(self, i):
        def hook(_mod, _inp, out):
            if self.mode == "capture":
                self.cf[("mlp", i)] = out.detach()
                return out
            if self.mode == "mix":
                m = self.mask_mlp[i].to(out.dtype)
                return (1 - m) * out + m * self.cf[("mlp", i)].to(out.dtype)
            return out
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def learn_role(model, taps, pairs, device, lam, steps, lr):
    """Optimise mask logits; return (attn_mask (L,H), mlp_mask (L,), stats).

    Pairs are length-matched within a pair but not across pairs, so they are
    processed in same-length groups; the counterfactual activations of each
    group are captured once and cached, and every optimisation step sums the
    task loss over all groups before one Adam update."""
    groups: dict[int, list] = {}
    for pr in pairs:
        groups.setdefault(len(pr[0]), []).append(pr)
    batches = []
    for g in groups.values():
        clean = torch.tensor([p[0] for p in g], device=device)
        cf = torch.tensor([p[1] for p in g], device=device)
        ct = torch.tensor([p[2] for p in g], device=device)
        xt = torch.tensor([p[3] for p in g], device=device)
        taps.cf = {}
        taps.mode = "capture"
        with torch.no_grad():
            model(cf)
        taps.mode = "off"
        batches.append((clean, cf, ct, xt, dict(taps.cf)))
    n_total = sum(len(b[0]) for b in batches)

    la = torch.full((taps.L, taps.H), -3.0, device=device, requires_grad=True)
    lm = torch.full((taps.L,), -3.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([la, lm], lr=lr)
    for step in range(steps):
        taps.mask_attn, taps.mask_mlp = torch.sigmoid(la), torch.sigmoid(lm)
        task = 0.0
        for clean, _cf, _ct, xt, cf_acts in batches:
            taps.cf = cf_acts
            taps.mode = "mix"
            logits = model(clean).logits[:, -1].float()
            taps.mode = "off"
            task = task + F.cross_entropy(logits, xt, reduction="sum")
        task = task / n_total
        sparsity = torch.sigmoid(la).sum() + torch.sigmoid(lm).sum()
        loss = task + lam * sparsity
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"    step {step:4d}  task_ce {task.item():.3f}  "
                  f"mask_sum {sparsity.item():6.1f}")

    # binarise and evaluate the hard role set
    with torch.no_grad():
        ha, hm = (torch.sigmoid(la) > 0.5).float(), (torch.sigmoid(lm) > 0.5).float()
        taps.mask_attn, taps.mask_mlp = ha, hm
        hit_cf = hit_clean = hit_ceiling = 0
        for clean, cf, ct, xt, cf_acts in batches:
            taps.cf = cf_acts
            taps.mode = "mix"
            hit_cf += (model(clean).logits[:, -1].argmax(-1) == xt).sum().item()
            taps.mode = "off"
            hit_clean += (model(clean).logits[:, -1].argmax(-1) == ct).sum().item()
            hit_ceiling += (model(cf).logits[:, -1].argmax(-1) == xt).sum().item()
    return ha.cpu(), hm.cpu(), {"cf_flip_acc": hit_cf / n_total,
                                "clean_acc": hit_clean / n_total,
                                "cf_ceiling_acc": hit_ceiling / n_total,
                                "n_attn": int(ha.sum()), "n_mlp": int(hm.sum()),
                                "n_length_groups": len(batches)}


def cmd_learn(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
    if args.run_id is not None:
        if (store / "runs" / args.run_id / "model" / "model.safetensors").is_file():
            from geode.zoo import load_model as zoo_load_model

            model = zoo_load_model(args.run_id, store=store, device=args.device)
        else:
            from circuit_nodes import load_sidecar_merged

            model = load_sidecar_merged(args.run_id, store, args.device)
        name = args.run_id
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        model.to(args.device)
        name = args.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    taps = MixTaps(model)
    out = {"model": name, "surface": args.surface, "lam": args.lam, "roles": {}}
    for role in args.roles:
        pairs = make_pairs(args.surface, role, args.n_pairs, tokenizer)
        if len(pairs) < 8:
            print(f"[dcm] {role}: only {len(pairs)} usable pairs on {args.surface} — skipped")
            continue
        print(f"[dcm] {name} / {role}: {len(pairs)} pairs")
        ha, hm, st = learn_role(model, taps, pairs, args.device, args.lam, args.steps, args.lr)
        nodes = [f"attn:{i}:{h}" for i in range(taps.L) for h in range(taps.H) if ha[i, h]]
        nodes += [f"mlp:{i}" for i in range(taps.L) if hm[i]]
        out["roles"][role] = {"nodes": nodes, **st, "n_pairs": len(pairs)}
        layers = sorted({int(n.split(":")[1]) for n in nodes})
        print(f"[dcm]   role set: {len(nodes)} nodes ({st['n_attn']} heads, {st['n_mlp']} MLPs) "
              f"layers {layers}; cf-flip acc {st['cf_flip_acc']:.3f} "
              f"(ceiling {st['cf_ceiling_acc']:.3f}); clean acc {st['clean_acc']:.3f}")
    taps.remove()
    Path(f"{args.out}.json").write_text(json.dumps(out, indent=2))
    print(f"[dcm] wrote {args.out}.json")
    return 0


def cmd_compare(args) -> int:
    A = json.loads(Path(f"{args.a}.json").read_text())
    B = json.loads(Path(f"{args.b}.json").read_text())
    print(f"[dcm] A = {A['model']} ({A['surface']})   B = {B['model']} ({B['surface']})")
    for role in sorted(set(A["roles"]) & set(B["roles"])):
        sa, sb = set(A["roles"][role]["nodes"]), set(B["roles"][role]["nodes"])
        j = len(sa & sb) / max(1, len(sa | sb))
        heads_a = {n for n in sa if n.startswith("attn")}
        heads_b = {n for n in sb if n.startswith("attn")}
        jh = len(heads_a & heads_b) / max(1, len(heads_a | heads_b))
        chance = len(sa) * len(sb) / 528 / max(1, len(sa | sb))
        print(f"[dcm] {role:<10}: |A|={len(sa):3d} |B|={len(sb):3d} shared {len(sa & sb):3d}  "
              f"Jaccard {j:.3f} (heads-only {jh:.3f}; chance ~{chance:.3f})")
        only_b = sorted(sb - sa, key=lambda s: (int(s.split(':')[1]), s))
        if only_b:
            print(f"[dcm]   nodes in B not in A: {only_b[:12]}{' ...' if len(only_b) > 12 else ''}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("learn")
    m.add_argument("--run-id", default=None)
    m.add_argument("--model", default=None)
    m.add_argument("--surface", choices=("bare_op", "bare_nl", "bridge"), required=True)
    m.add_argument("--roles", nargs="+", default=list(ROLES), choices=ROLES)
    m.add_argument("--out", required=True)
    m.add_argument("--n-pairs", type=int, default=64)
    m.add_argument("--lam", type=float, default=0.02, help="sparsity weight on mask sum")
    m.add_argument("--steps", type=int, default=200)
    m.add_argument("--lr", type=float, default=0.1)
    m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    return cmd_learn(args) if args.cmd == "learn" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
