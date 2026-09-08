"""Metric 10: residual-stream shift, task-specific vs generic.

Same input through the PARENT and its fine-tuned CHILD; the diff of the
residual stream (delta_l = h_child_l - h_parent_l, after every decoder layer)
is measured on TASK inputs and on unrelated held-out TEXT (TinyStories
validation stories). The ratio matters more than either number.

Owner's predictions (2026-09-09):
- Elicitation: SURGICAL. Near-zero shift on generic text; the task shift is
  written in a few (late) layers and points the same way on every example —
  one reusable vector (an answer-mode switch), not new computation.
- Teaching: DIFFUSE. Generic text moves too (new machinery displaces existing
  representations); the write spreads over many layers; directions vary per
  example instead of collapsing to one.

Per layer l this reports
  rel_task_ans   sqrt(E||delta||^2 / E||h_parent||^2) at the answer position
                 (last prompt token, where the first answer token is predicted)
  rel_task_all   same over every prompt position (position 0 excluded)
  rel_generic    same over every position of the generic text (pos 0 excluded)
  ratio          rel_task_ans / rel_generic            (surgical >> 1)
  inc_share      share of INCREMENTAL write energy ||delta_l - delta_{l-1}||^2
                 at the answer position (where in depth the shift is written;
                 the raw cumulative delta grows with depth by construction)
  cos2mean       mean_i cos(delta_i, mean delta)  (task answer pos / generic)
  pc1            sigma_1^2 / sum sigma^2 of the (N x d) delta matrix — energy
                 fraction of a single direction (1.0 = one vector)
  *_parent_*     the SAME two statistics on the parent's own states at the same
                 positions — the anisotropy reference. Residual states share a
                 large common direction, so any fixed weight change yields
                 deltas that look "one-directional"; a delta is only a genuine
                 single reusable vector if it beats this reference, and only
                 "diffuse" if it falls below it.
  cos_task_gen   cos(mean task delta, mean generic delta)
and summary scalars: layer-mean rels and their ratio; effective number of
writing layers (participation ratio of inc_share), energy centroid layer and
top-3 share; direction statistics at the final layer (what the unembedding
reads) and energy-weighted over layers, each next to its parent reference.
Layers with no delta report nan and are skipped by the means.

v2 additions (fixes after the 2026-09-09 readout):
  functional shift   KL(parent || child) of the next-token distribution at the
                     task answer position vs per token on generic text, and the
                     child-minus-parent NLL on generic text (does the child
                     still tell stories as well?). Geometry-free version of the
                     task/generic ratio: displacement that CHANGES what the
                     model says, not how far a vector moved.
  inc_rel_share      increment energy normalised by the parent's layer norm
                     before taking shares (the raw shares over-weight late
                     layers, where residual norms are large).
  content-removed    direction statistics of the final-layer task shift after
                     projecting out the unembedding directions of the correct
                     answer token and of the parent's own top-1 token — is
                     there a shared "mode" vector once the per-problem answer
                     content is removed? Also reports the energy fraction the
                     removed content carried.

Both models run in fp32; a LoRA child is rebuilt as parent + sidecar merged
in fp32 (a bf16 merge would round ||W|| at 2^-8, the same order as a small
delta). Pure inference, GPU recommended (two 1B fp32 models ~10 GB).

Usage:
    python3 resid_shift.py run --parent evt-ts1b-op-bridge-mix \
        --child evt-ts1b-mix-nl-n1000000 --out resid_elicit_1m [--n 256]
        [--generic-text /path/TinyStoriesV2-GPT4-valid.txt]
    python3 resid_shift.py compare elicit=resid_elicit_1m.json \
        teach=resid_teach_1m.json [--plot resid_shift.png]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit_nodes import apply_sidecar  # noqa: E402
from premise_checks import DEFAULT_ROW_OFFSET, EVAL_PARQUET, render_probe  # noqa: E402

STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
TOKENIZER = "meta-llama/Llama-3.2-1B"
GENERIC_REPO, GENERIC_FILE = "roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt"


# ----------------------------------------------------------------- loading
def _parent_dir(run_id: str) -> str:
    """Checkpoint path/hub id of a LoRA run's parent (load_sidecar_merged's rule)."""
    from geode.zoo import load_run

    base_id = load_run(run_id, store=STORE).data["base_model"]["hf_id"]
    if base_id.startswith("zoo-run/"):
        parent = base_id.split("/", 1)[1]
        cand = STORE / "runs" / parent / "model_merged"
        return str(cand if (cand / "model.safetensors").is_file()
                   else STORE / "runs" / parent / "model")
    return base_id


def load_fp32(spec: str, device: str):
    """Hub id / plain dir, or zoo run id. LoRA runs with an adapter sidecar are
    rebuilt as parent + exact fp32 merge; anything else loads via the zoo
    (manifest-aware) and is cast to fp32."""
    from transformers import AutoModelForCausalLM

    run_dir = STORE / "runs" / spec
    if "/" not in spec and run_dir.is_dir():
        sidecar = run_dir / "model" / "adapter.safetensors"
        if sidecar.is_file():
            from safetensors.torch import load_file

            from geode.zoo import load_run

            lora = load_run(spec, store=STORE).data["training"]["lora"]
            scaling = lora["alpha"] / (2 * lora["rank"])
            model = AutoModelForCausalLM.from_pretrained(_parent_dir(spec),
                                                         torch_dtype=torch.float32)
            n = apply_sidecar(model, load_file(sidecar), scaling)
            print(f"[resid] {spec}: parent + sidecar merged in fp32 ({n} pairs, "
                  f"scaling {scaling:.5f})")
            return model.to(device).eval()
        from geode.zoo import load_model as zoo_load_model

        print(f"[resid] {spec}: zoo checkpoint (manifest-aware), cast to fp32")
        return zoo_load_model(spec, store=STORE, device=device).float().eval()
    print(f"[resid] {spec}: from_pretrained fp32")
    return AutoModelForCausalLM.from_pretrained(spec, torch_dtype=torch.float32).to(device).eval()


class ResidTaps:
    """Capture the residual stream after the embedding and after every layer."""

    def __init__(self, model):
        self.acts: list[torch.Tensor] = []
        self.handles = [model.model.embed_tokens.register_forward_hook(self._emb)]
        for layer in model.model.layers:
            self.handles.append(layer.register_forward_hook(self._layer))

    def _emb(self, _m, _i, out):
        self.acts = [out.detach()]

    def _layer(self, _m, _i, out):
        self.acts.append((out[0] if isinstance(out, tuple) else out).detach())

    def remove(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------ inputs
def task_prompts(fmt: str, n: int) -> tuple[list[str], list[tuple[int, int, str]]]:
    df = pd.read_parquet(EVAL_PARQUET)
    rows = df.iloc[DEFAULT_ROW_OFFSET : DEFAULT_ROW_OFFSET + n]
    triples = [(int(r.a), int(r.b), str(r.op)) for r in rows.itertuples()]
    return [render_probe(fmt, a, b, op, 0)[0] for a, b, op in triples], triples


def generic_ids(tokenizer, path: str | None, n: int, seq_len: int) -> tuple[torch.Tensor, str]:
    """First n held-out stories with >= seq_len tokens, truncated to seq_len."""
    if path is None:
        path = os.environ.get("TS_VALID")
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(GENERIC_REPO, GENERIC_FILE, repo_type="dataset")
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    stories = [s.strip() for s in re.split(r"<\|endoftext\|>", text) if s.strip()]
    out = []
    for s in stories:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"[resid] only {len(out)} stories with >= {seq_len} tokens in {path}")
    return torch.tensor(out), str(path)


# ----------------------------------------------------------------- measure
class Accum:
    """Per-layer energy sums for one position set, plus stored delta rows."""

    def __init__(self, n_layers: int):
        self.dd = torch.zeros(n_layers)      # sum ||delta||^2
        self.hh = torch.zeros(n_layers)      # sum ||h_parent||^2
        self.inc = torch.zeros(n_layers)     # sum ||delta_l - delta_{l-1}||^2
        self.count = 0
        self.rows: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]      # delta rows
        self.prow: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]      # parent rows

    def add(self, deltas, parents, mask, keep_rows: torch.Tensor | None):
        """deltas/parents: lists over layers of (B,T,d); mask (B,T) bool;
        keep_rows: (M,2) [b,t] indices whose delta rows are stored."""
        prev = None
        for layer, (d, h) in enumerate(zip(deltas, parents)):
            dm, hm = d[mask], h[mask]
            self.dd[layer] += (dm**2).sum().item()
            self.hh[layer] += (hm**2).sum().item()
            if prev is not None:
                self.inc[layer] += ((dm - prev) ** 2).sum().item()
            prev = dm
            if keep_rows is not None:
                self.rows[layer].append(d[keep_rows[:, 0], keep_rows[:, 1]].cpu())
                self.prow[layer].append(h[keep_rows[:, 0], keep_rows[:, 1]].cpu())
        self.count += int(mask.sum().item())

    def rel(self) -> torch.Tensor:
        return (self.dd / self.hh.clamp_min(1e-30)).sqrt()

    def inc_share(self) -> torch.Tensor:
        return self.inc / self.inc.sum().clamp_min(1e-30)

    def inc_rel_share(self) -> torch.Tensor:
        """Increment energy normalised by the parent's layer energy, then shared."""
        rel = self.inc / self.hh.clamp_min(1e-30)
        return rel / rel.sum().clamp_min(1e-30)

    def matrices(self) -> list[torch.Tensor]:
        return [torch.cat(r, 0) if r else torch.empty(0) for r in self.rows]

    def parent_matrices(self) -> list[torch.Tensor]:
        return [torch.cat(r, 0) if r else torch.empty(0) for r in self.prow]


def direction_stats(mat: torch.Tensor) -> tuple[float, float, float]:
    """(cos-to-mean, coherence ||mean||/mean||row||, PC1 energy fraction)."""
    if mat.numel() == 0 or mat.shape[0] < 2 or mat.norm() < 1e-12:
        return float("nan"), float("nan"), float("nan")
    mat = mat.double()
    m = mat.mean(0, keepdim=True)
    c2m = F.cosine_similarity(mat, m.expand_as(mat), dim=1).mean().item()
    coh = (m.norm() / mat.norm(dim=1).mean()).item()
    sv = torch.linalg.svdvals(mat)
    pc1 = (sv[0] ** 2 / (sv**2).sum()).item()
    return c2m, coh, pc1


@torch.no_grad()
def run_pair(parent, child, tokenizer, prompts, device, batch_size, gen_ids, gen_bs,
             n_gen_rows_per_seq, seed):
    n_layers = len(parent.model.layers) + 1  # + embedding
    tp, tc = ResidTaps(parent), ResidTaps(child)
    acc_ans, acc_all, acc_gen = Accum(n_layers), Accum(n_layers), Accum(n_layers)
    fn = {"kl_task_ans": [], "parent_top1": [], "kl_gen": 0.0, "nll_gen_parent": 0.0,
          "nll_gen_child": 0.0, "n_gen_tok": 0}

    def kl_pc(lp_p, lp_c):  # KL(parent || child) per row, nats
        return (lp_p.exp() * (lp_p - lp_c)).sum(-1)

    tokenizer.padding_side = "left"
    for s in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[s : s + batch_size], return_tensors="pt", padding=True,
                        add_special_tokens=False)
        ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        lp_p = F.log_softmax(parent(input_ids=ids, attention_mask=am).logits[:, -1].float(), -1)
        hp = list(tp.acts)
        lp_c = F.log_softmax(child(input_ids=ids, attention_mask=am).logits[:, -1].float(), -1)
        hc = list(tc.acts)
        fn["kl_task_ans"] += kl_pc(lp_p, lp_c).cpu().tolist()
        fn["parent_top1"] += lp_p.argmax(-1).cpu().tolist()
        del lp_p, lp_c
        deltas = [c - p for c, p in zip(hc, hp)]
        B, T = ids.shape
        ans_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        ans_mask[:, -1] = True
        keep = torch.stack([torch.arange(B, device=device),
                            torch.full((B,), T - 1, device=device)], 1)
        acc_ans.add(deltas, hp, ans_mask, keep)
        first = (am.cumsum(1) == 1) & am.bool()          # first real token of each row
        acc_all.add(deltas, hp, am.bool() & ~first, None)
        del hp, hc, deltas

    g = torch.Generator().manual_seed(seed)
    for s in range(0, gen_ids.shape[0], gen_bs):
        ids = gen_ids[s : s + gen_bs].to(device)
        B, T = ids.shape
        lg_p = parent(input_ids=ids).logits[:, :-1].float()
        hp = list(tp.acts)
        lg_c = child(input_ids=ids).logits[:, :-1].float()
        hc = list(tc.acts)
        tgt = ids[:, 1:]
        lp_p, lp_c = F.log_softmax(lg_p, -1), F.log_softmax(lg_c, -1)
        fn["kl_gen"] += kl_pc(lp_p, lp_c).sum().item()
        fn["nll_gen_parent"] += -lp_p.gather(-1, tgt[..., None]).sum().item()
        fn["nll_gen_child"] += -lp_c.gather(-1, tgt[..., None]).sum().item()
        fn["n_gen_tok"] += tgt.numel()
        del lg_p, lg_c, lp_p, lp_c
        deltas = [c - p for c, p in zip(hc, hp)]
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        mask[:, 0] = False
        pos = torch.randint(1, T, (B, n_gen_rows_per_seq), generator=g)
        keep = torch.stack([torch.arange(B).repeat_interleave(n_gen_rows_per_seq),
                            pos.reshape(-1)], 1).to(device)
        acc_gen.add(deltas, hp, mask, keep)
        del hp, hc, deltas
    tp.remove()
    tc.remove()
    return acc_ans, acc_all, acc_gen, fn


def participation(shares: torch.Tensor) -> float:
    p = shares / shares.sum().clamp_min(1e-30)
    return (1.0 / (p**2).sum().clamp_min(1e-30)).item()


def cmd_run(args) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, triples = task_prompts(args.task_format, args.n)
    gen_ids, gen_path = generic_ids(tokenizer, args.generic_text, args.n, args.seq_len)
    print(f"[resid] task: {args.task_format} x{len(prompts)}  e.g. {prompts[0]!r}")
    print(f"[resid] generic: {gen_ids.shape[0]} x {args.seq_len} tokens from {gen_path}")

    parent = load_fp32(args.parent, args.device)
    child = load_fp32(args.child, args.device)
    acc_ans, acc_all, acc_gen, fn = run_pair(parent, child, tokenizer, prompts, args.device,
                                             args.batch_size, gen_ids, args.gen_batch_size,
                                             args.gen_rows_per_seq, args.seed)

    rel_ans, rel_all, rel_gen = acc_ans.rel(), acc_all.rel(), acc_gen.rel()
    inc_ans, inc_gen = acc_ans.inc_share(), acc_gen.inc_share()
    incr_ans, incr_gen = acc_ans.inc_rel_share(), acc_gen.inc_rel_share()

    # --- v2: functional shift
    kl_task = sum(fn["kl_task_ans"]) / len(fn["kl_task_ans"])
    kl_gen = fn["kl_gen"] / fn["n_gen_tok"]
    nll_p, nll_c = fn["nll_gen_parent"] / fn["n_gen_tok"], fn["nll_gen_child"] / fn["n_gen_tok"]

    # --- v2: content-removed direction at the final layer (task answer position)
    from geode.arith.formats import true_answer

    U = parent.lm_head.weight.detach().float().cpu()
    gamma = parent.model.norm.weight.detach().float().cpu()
    first_tok = lambda x: tokenizer(str(x), add_special_tokens=False)["input_ids"][0]  # noqa: E731
    correct = torch.tensor([first_tok(true_answer(a, b, op)) for a, b, op in triples])
    ptop = torch.tensor(fn["parent_top1"])
    D = acc_ans.matrices()[-1].double()                       # (N, d) final-layer deltas
    u1 = (U[correct] * gamma).double()
    u2 = (U[ptop] * gamma).double()
    u1 = u1 / u1.norm(dim=1, keepdim=True)
    u2 = u2 - (u2 * u1).sum(1, keepdim=True) * u1
    u2n = u2.norm(dim=1, keepdim=True)
    u2 = torch.where(u2n > 1e-8, u2 / u2n.clamp_min(1e-8), torch.zeros_like(u2))
    proj = (D * u1).sum(1, keepdim=True) * u1 + (D * u2).sum(1, keepdim=True) * u2
    D_rest = D - proj
    content_frac = (proj.norm() ** 2 / D.norm().clamp_min(1e-30) ** 2).item()
    c2m_nc, coh_nc, pc1_nc = direction_stats(D_rest.float())
    mats_t, mats_g = acc_ans.matrices(), acc_gen.matrices()
    pmat_t, pmat_g = acc_ans.parent_matrices(), acc_gen.parent_matrices()
    layers = []
    for layer in range(len(rel_ans)):
        c2m_t, coh_t, pc1_t = direction_stats(mats_t[layer])
        c2m_g, coh_g, pc1_g = direction_stats(mats_g[layer])
        # anisotropy reference: the same statistics on the PARENT's own states
        # at the same positions (how "one-directional" any vector set is here)
        c2m_pt, _, pc1_pt = direction_stats(pmat_t[layer])
        c2m_pg, _, pc1_pg = direction_stats(pmat_g[layer])
        if mats_t[layer].numel() and mats_g[layer].numel():
            ctg = F.cosine_similarity(mats_t[layer].double().mean(0),
                                      mats_g[layer].double().mean(0), dim=0).item()
        else:
            ctg = float("nan")
        layers.append({"layer": layer - 1,  # -1 = embedding
                       "rel_task_ans": rel_ans[layer].item(),
                       "rel_task_all": rel_all[layer].item(),
                       "rel_generic": rel_gen[layer].item(),
                       "ratio": (rel_ans[layer] / rel_gen[layer].clamp_min(1e-30)).item(),
                       "inc_share_task": inc_ans[layer].item(),
                       "inc_share_generic": inc_gen[layer].item(),
                       "cos2mean_task": c2m_t, "cos2mean_generic": c2m_g,
                       "coherence_task": coh_t, "coherence_generic": coh_g,
                       "pc1_task": pc1_t, "pc1_generic": pc1_g,
                       "cos2mean_parent_task": c2m_pt, "cos2mean_parent_generic": c2m_pg,
                       "pc1_parent_task": pc1_pt, "pc1_parent_generic": pc1_pg,
                       "cos_task_generic": ctg,
                       "energy_task": acc_ans.dd[layer].item(),
                       "energy_generic": acc_gen.dd[layer].item()})
    body = layers[1:]  # decoder layers only for summaries
    L = torch.arange(len(body), dtype=torch.float)
    inc_t = torch.tensor([r["inc_share_task"] for r in body])
    inc_g = torch.tensor([r["inc_share_generic"] for r in body])
    inc_t, inc_g = inc_t / inc_t.sum(), inc_g / inc_g.sum()
    import math

    def mean(k, weights=None):
        vals = [(r[k], 1.0 if weights is None else r[weights]) for r in body
                if not math.isnan(r[k])]
        wsum = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / wsum if wsum > 0 else float("nan")

    last = body[-1]
    top3 = sorted(range(len(body)), key=lambda i: -body[i]["inc_share_task"])[:3]
    summary = {
        "rel_task_ans_mean": mean("rel_task_ans"),
        "rel_task_all_mean": mean("rel_task_all"),
        "rel_generic_mean": mean("rel_generic"),
        "ratio_ans_over_generic": mean("rel_task_ans") / max(mean("rel_generic"), 1e-30),
        "ratio_all_over_generic": mean("rel_task_all") / max(mean("rel_generic"), 1e-30),
        "eff_layers_task": participation(inc_t),
        "eff_layers_generic": participation(inc_g),
        "eff_layers_task_normalised": participation(incr_ans[1:]),
        "eff_layers_generic_normalised": participation(incr_gen[1:]),
        "centroid_layer_task_normalised": (incr_ans[1:] / incr_ans[1:].sum() * L).sum().item(),
        "centroid_layer_generic_normalised": (incr_gen[1:] / incr_gen[1:].sum() * L).sum().item(),
        "inc_rel_share_task": incr_ans.tolist(),
        "inc_rel_share_generic": incr_gen.tolist(),
        # v2 functional shift
        "kl_task_ans": kl_task, "kl_generic_per_token": kl_gen,
        "kl_ratio_task_over_generic": kl_task / max(kl_gen, 1e-30),
        "nll_generic_parent": nll_p, "nll_generic_child": nll_c,
        "delta_nll_generic": nll_c - nll_p,
        # v2 content-removed direction (final layer, task answer position)
        "content_energy_frac_last": content_frac,
        "cos2mean_task_last_nocontent": c2m_nc, "coherence_task_last_nocontent": coh_nc,
        "pc1_task_last_nocontent": pc1_nc,
        "centroid_layer_task": (inc_t * L).sum().item(),
        "centroid_layer_generic": (inc_g * L).sum().item(),
        "top3_layers_task": top3,
        "top3_share_task": sum(body[i]["inc_share_task"] for i in top3),
        # direction: layer means (nan-safe), energy-weighted means, final layer
        "cos2mean_task_mean": mean("cos2mean_task"),
        "cos2mean_generic_mean": mean("cos2mean_generic"),
        "cos2mean_parent_task_mean": mean("cos2mean_parent_task"),
        "cos2mean_parent_generic_mean": mean("cos2mean_parent_generic"),
        "pc1_task_mean": mean("pc1_task"),
        "pc1_generic_mean": mean("pc1_generic"),
        "pc1_parent_task_mean": mean("pc1_parent_task"),
        "pc1_parent_generic_mean": mean("pc1_parent_generic"),
        "cos2mean_task_wmean": mean("cos2mean_task", "energy_task"),
        "cos2mean_generic_wmean": mean("cos2mean_generic", "energy_generic"),
        "pc1_task_wmean": mean("pc1_task", "energy_task"),
        "pc1_generic_wmean": mean("pc1_generic", "energy_generic"),
        "cos2mean_task_last": last["cos2mean_task"],
        "cos2mean_generic_last": last["cos2mean_generic"],
        "cos2mean_parent_task_last": last["cos2mean_parent_task"],
        "cos2mean_parent_generic_last": last["cos2mean_parent_generic"],
        "pc1_task_last": last["pc1_task"], "pc1_generic_last": last["pc1_generic"],
        "pc1_parent_task_last": last["pc1_parent_task"],
        "pc1_parent_generic_last": last["pc1_parent_generic"],
        "cos_task_generic_mean": mean("cos_task_generic"),
        "cos_task_generic_last": last["cos_task_generic"],
        "n_task": len(prompts), "n_task_all_positions": acc_all.count,
        "n_generic_positions": acc_gen.count,
        "n_generic_rows": int(mats_g[1].shape[0]) if mats_g[1].numel() else 0,
    }
    print(f"[resid] {args.child} vs {args.parent}")
    print("[resid] layer  rel_ans  rel_all  rel_gen   ratio  inc_t  inc_g | c2m_t  c2m_g "
          "(parent t/g) | pc1_t  pc1_g (parent t/g) | cosTG")
    for r in layers:
        print(f"[resid] {r['layer']:5d}  {r['rel_task_ans']:.4f}  {r['rel_task_all']:.4f}  "
              f"{r['rel_generic']:.4f}  {r['ratio']:6.2f}  {r['inc_share_task']:.3f}  "
              f"{r['inc_share_generic']:.3f} | {r['cos2mean_task']:.3f}  "
              f"{r['cos2mean_generic']:.3f} ({r['cos2mean_parent_task']:.2f}/"
              f"{r['cos2mean_parent_generic']:.2f}) | {r['pc1_task']:.3f}  {r['pc1_generic']:.3f} "
              f"({r['pc1_parent_task']:.2f}/{r['pc1_parent_generic']:.2f}) | "
              f"{r['cos_task_generic']:+.3f}")
    s = summary
    print(f"[resid] SUMMARY rel: task(ans) {s['rel_task_ans_mean']:.4f}  task(all) "
          f"{s['rel_task_all_mean']:.4f}  generic {s['rel_generic_mean']:.4f}  "
          f"ratio ans/gen {s['ratio_ans_over_generic']:.2f}  all/gen {s['ratio_all_over_generic']:.2f}")
    print(f"[resid] SUMMARY where written (task, incremental energy): eff. layers "
          f"{s['eff_layers_task']:.1f} of {len(body)}, centroid L{s['centroid_layer_task']:.1f}, "
          f"top-3 {s['top3_layers_task']} carry {s['top3_share_task']:.0%}   "
          f"(generic: eff. {s['eff_layers_generic']:.1f}, centroid L{s['centroid_layer_generic']:.1f})")
    print(f"[resid] SUMMARY direction, final layer: cos-to-mean task {s['cos2mean_task_last']:.3f} "
          f"(parent states {s['cos2mean_parent_task_last']:.3f}) vs generic "
          f"{s['cos2mean_generic_last']:.3f} (parent {s['cos2mean_parent_generic_last']:.3f}); "
          f"PC1 task {s['pc1_task_last']:.3f} (parent {s['pc1_parent_task_last']:.3f}) vs generic "
          f"{s['pc1_generic_last']:.3f} (parent {s['pc1_parent_generic_last']:.3f})")
    print(f"[resid] SUMMARY where written, NORMALISED increments: eff. layers task "
          f"{s['eff_layers_task_normalised']:.1f} (centroid L{s['centroid_layer_task_normalised']:.1f}) "
          f"vs generic {s['eff_layers_generic_normalised']:.1f} "
          f"(centroid L{s['centroid_layer_generic_normalised']:.1f}); task shares by layer: "
          + " ".join(f"L{i}:{v:.2f}" for i, v in enumerate(incr_ans[1:].tolist())))
    print(f"[resid] SUMMARY functional shift: KL(parent||child) task answer pos {s['kl_task_ans']:.3f} "
          f"nats vs generic per token {s['kl_generic_per_token']:.4f} -> ratio "
          f"{s['kl_ratio_task_over_generic']:.1f}; generic NLL parent {s['nll_generic_parent']:.4f} "
          f"-> child {s['nll_generic_child']:.4f} (delta {s['delta_nll_generic']:+.4f} nats/token)")
    print(f"[resid] SUMMARY content-removed direction (final layer): answer+parent-top1 directions "
          f"carry {s['content_energy_frac_last']:.1%} of the shift energy; remainder cos-to-mean "
          f"{s['cos2mean_task_last_nocontent']:.3f}, coherence {s['coherence_task_last_nocontent']:.3f}, "
          f"PC1 {s['pc1_task_last_nocontent']:.3f} (vs full shift {s['cos2mean_task_last']:.3f} / "
          f"{s['pc1_task_last']:.3f})")
    print(f"[resid] SUMMARY direction, energy-weighted over layers: cos-to-mean task "
          f"{s['cos2mean_task_wmean']:.3f} vs generic {s['cos2mean_generic_wmean']:.3f}; PC1 task "
          f"{s['pc1_task_wmean']:.3f} vs generic {s['pc1_generic_wmean']:.3f}; "
          f"cos(mean task delta, mean generic delta) {s['cos_task_generic_mean']:+.3f} "
          f"(final layer {s['cos_task_generic_last']:+.3f})")
    out = Path(f"{args.out}.json")
    out.write_text(json.dumps({"parent": args.parent, "child": args.child,
                               "task_format": args.task_format, "generic": gen_path,
                               "seq_len": args.seq_len, "summary": summary,
                               "layers": layers}, indent=2))
    print(f"[resid] wrote {out}")
    return 0


# ----------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    runs = []
    for spec in args.runs:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        runs.append((label, json.loads(Path(path).read_text())))
    keys = [("rel_task_ans_mean", "rel task (answer pos)"),
            ("rel_task_all_mean", "rel task (all pos)"),
            ("rel_generic_mean", "rel generic text"),
            ("ratio_ans_over_generic", "RATIO task/generic"),
            ("eff_layers_task", "eff. writing layers (task)"),
            ("centroid_layer_task", "energy centroid layer (task)"),
            ("top3_share_task", "top-3 layer share (task)"),
            ("cos2mean_task_last", "cos-to-mean task (final L)"),
            ("cos2mean_parent_task_last", "  ref: parent states, task"),
            ("cos2mean_generic_last", "cos-to-mean generic (final L)"),
            ("cos2mean_parent_generic_last", "  ref: parent states, generic"),
            ("pc1_task_last", "PC1 energy task (final L)"),
            ("pc1_parent_task_last", "  ref: parent states, task"),
            ("pc1_generic_last", "PC1 energy generic (final L)"),
            ("pc1_parent_generic_last", "  ref: parent states, generic"),
            ("cos2mean_task_wmean", "cos-to-mean task (E-weighted)"),
            ("cos2mean_generic_wmean", "cos-to-mean generic (E-weighted)"),
            ("cos_task_generic_last", "cos(task dir, generic dir) final"),
            ("eff_layers_task_normalised", "eff. layers, normalised (task)"),
            ("centroid_layer_task_normalised", "centroid, normalised (task)"),
            ("kl_task_ans", "KL(parent||child) task, nats"),
            ("kl_generic_per_token", "KL(parent||child) generic/token"),
            ("kl_ratio_task_over_generic", "KL RATIO task/generic"),
            ("delta_nll_generic", "generic NLL child - parent"),
            ("content_energy_frac_last", "answer-content energy frac (final)"),
            ("pc1_task_last_nocontent", "PC1 after removing content"),
            ("cos2mean_task_last_nocontent", "cos-to-mean after removing content")]
    head = f"{'metric':<34}" + "".join(f"{lab:>16}" for lab, _ in runs)
    print("[resid] " + head)
    for k, name in keys:
        vals = "".join(f"{r['summary'][k]:>16.3f}" if k in r["summary"] else f"{'--':>16}"
                       for _, r in runs)
        print(f"[resid] {name:<34}{vals}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        colors = ["#c0392b", "#2c3e50", "#e67e22", "#16a085"]
        for (label, r), col in zip(runs, colors):
            body = [x for x in r["layers"] if x["layer"] >= 0]
            L = [x["layer"] for x in body]
            axes[0].plot(L, [x["rel_task_ans"] for x in body], "-o", ms=3, color=col,
                         label=f"{label}: task")
            axes[0].plot(L, [x["rel_generic"] for x in body], "--", color=col,
                         label=f"{label}: generic")
            axes[1].plot(L, [x["inc_share_task"] for x in body], "-o", ms=3, color=col,
                         label=label)
            axes[2].plot(L, [x["cos2mean_task"] for x in body], "-o", ms=3, color=col,
                         label=f"{label}: task")
            axes[2].plot(L, [x["cos2mean_generic"] for x in body], "--", color=col,
                         label=f"{label}: generic")
            axes[2].plot(L, [x["cos2mean_parent_task"] for x in body], ":", color=col,
                         alpha=0.6, label=f"{label}: parent states (ref)")
        axes[0].set(title="Relative residual shift ‖Δh‖/‖h‖", xlabel="layer", yscale="log")
        axes[1].set(title="Where the shift is written (energy share)", xlabel="layer")
        axes[2].set(title="Direction consistency (cos to mean Δ)", xlabel="layer",
                    ylim=(-0.05, 1.02))
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        print(f"[resid] wrote {args.plot}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--parent", required=True, help="parent run id / hub id")
    r.add_argument("--child", required=True, help="fine-tuned child run id")
    r.add_argument("--out", required=True, help="output stem (.json)")
    r.add_argument("--task-format", default="bare_nl")
    r.add_argument("--n", type=int, default=256, help="task prompts AND generic stories")
    r.add_argument("--seq-len", type=int, default=64)
    r.add_argument("--generic-text", default=None,
                   help="TinyStories txt (default: hub valid split, or $TS_VALID)")
    r.add_argument("--gen-rows-per-seq", type=int, default=4,
                   help="stored generic positions per story for direction stats")
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--gen-batch-size", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tokenizer", default=TOKENIZER)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="label=path.json ...")
    c.add_argument("--plot", default=None)
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
