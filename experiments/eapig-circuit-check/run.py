"""EAP-IG edge-circuit check, one model per invocation (PLAN.md, frozen 2026-10-01).

Stages (run on the GPU box, in this order across the four models):
  sanity    accuracy, digit-split check, token IDs, m(full)/m(empty) identities, timing,
            KL(full || empty) on val and the KL full-keep identity
  score     EAP-IG (inputs, 5 steps) per-example scores on the 512 discovery pairs
            (metric: LD, or -KL(p_clean || p_x) at the two answer positions)
  probe     f of the top-k circuit vs random same-size sets at --probe-sizes (no tests)
  evaluate  f curve + sufficiency at all sizes, parent gate, remaining tests with the
            stopping rule, TinyStories check, parent-in-child (children), real patching
            of the top-20 edges (children). With --metric kl every test is KL-judged
            (HANDOFF-kl-symbol-nodeedge.md, step 6) and every size runs (no early break)

Usage:
  python3 run.py --tag elicit_child --model <dir|hf repo> --stage sanity score evaluate \
      [--parent-tag elicit_parent] [--device cuda]
  python3 run.py --tag elicit_child --model M --task word|symbol --metric ld|kl \
      --stage sanity score probe evaluate --results DIR [--scores-dir DIR2]
Outputs: <results>/<tag>/{sanity.json, scores.pt, probe.json, evaluate.json, per_example.npz}.
evaluate/probe read scores.pt + sanity.json from <scores-dir> (default <results>).
Only task=word, metric=ld without probe may write to results/ or results_large/ (the
frozen record). `compare.py` (CPU) reads them for the parent-vs-child comparison and plots.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geode.circuits import edge_tests as et
from geode.circuits.eapig import (
    answer_logprobs,
    build_graph,
    eap_ig_scores,
    kl_2tok,
    logit_diff_2tok,
    node_outputs,
    patched_forward,
)

HERE = Path(__file__).resolve().parent
SIZES = (0.001, 0.002, 0.005, 0.01)
N_DRAWS = 100
N_DRAWS_TS = 20
EQUIV_EPS_FRAC = 0.10
SEED = 0
PROBE_SIZES = (0.02, 0.05, 0.1)
PROBE_DRAWS = {True: 20, False: 10}  # parents / children (owner, 2026-10-08)
KL_GATE_NATS = 0.1  # KL parent gate: mean KL(full || empty) on val (owner, 2026-10-08)
KL_IDENTITY_TOL = 1e-4
FROZEN_RESULTS = ("results", "results_large")  # the frozen LD-word record


# --------------------------------------------------------------------------- io


def load_model(path: str, device: str):
    """Local model dir, `repo:subfolder`, or a per-run hub repo `<ns>/<run_id>`
    (hf_checkpoint.py layout: weights under runs/<run_id>/model/)."""
    from transformers import AutoModelForCausalLM

    kw = {}
    if not Path(path).is_dir():
        if ":" in path:
            path, kw["subfolder"] = path.split(":", 1)
        else:
            kw["subfolder"] = f"runs/{path.split('/')[-1]}/model"
    model, info = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, output_loading_info=True, **kw
    )
    # a silently random-initialised layer would make every number meaningless
    bad = [k for k in info["missing_keys"] if "lm_head" not in k]
    assert not bad and not info["unexpected_keys"], info
    assert model.config.tie_word_embeddings, "TinyStories-1B has tied embeddings"
    return model.to(device).eval()


def load_data(data_dir: Path, task: str = "word"):
    from data import load_pairs  # experiments/eapig-circuit-check/data.py

    return load_pairs(data_dir, task=task)


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return float(o)

    path.write_text(json.dumps(obj, indent=2, default=conv))


# ------------------------------------------------------------------ evaluation


@torch.no_grad()
def eval_ld(model, graph, pairs: dict, keeps: list[torch.Tensor], bs: int) -> torch.Tensor:
    """Per-example (LD, term1, term2) for each keep mask: (n_masks, N, 3).

    Batch-outer: the counterfactual node outputs are computed once per batch and
    reused for every mask (caching them for all N would need ~20 GB at 1B/fp32).
    """
    dev = next(model.parameters()).device
    n = pairs["clean_ids"].shape[0]
    out = torch.empty(len(keeps), n, 3)
    for s in range(0, n, bs):
        sl = slice(s, s + bs)
        clean = pairs["clean_ids"][sl].to(dev)
        corrupt = pairs["corrupt_ids"][sl].to(dev)
        toks = [pairs[k][sl].to(dev) for k in ("c1", "k1", "c2", "k2")]
        acts = node_outputs(model, graph, corrupt)
        for i, keep in enumerate(keeps):
            logits = patched_forward(model, graph, clean, acts, keep)
            ld, t1, t2 = logit_diff_2tok(logits, *toks)
            out[i, sl] = torch.stack([ld, t1, t2], -1).cpu()
        del acts
    return out


@torch.no_grad()
def clean_logprobs(model, pairs: dict, bs: int) -> torch.Tensor:
    """Clean full-model answer log-probs (N, 2, V) on CPU: a plain forward of clean_ids."""
    dev = next(model.parameters()).device
    ids = pairs["clean_ids"]
    return torch.cat([answer_logprobs(model(ids[s : s + bs].to(dev)).logits).cpu()
                      for s in range(0, ids.shape[0], bs)])


@torch.no_grad()
def eval_kl(model, graph, pairs: dict, keeps: list[torch.Tensor], bs: int,
            clean_logp: torch.Tensor) -> torch.Tensor:
    """Per-example KL(p_clean || p_patched) in nats (two answer positions) for each keep
    mask: (n_masks, N). `clean_logp` is `clean_logprobs(model, pairs, bs)`. Batch-outer,
    like `eval_ld`."""
    dev = next(model.parameters()).device
    n = pairs["clean_ids"].shape[0]
    out = torch.empty(len(keeps), n)
    for s in range(0, n, bs):
        sl = slice(s, s + bs)
        clean = pairs["clean_ids"][sl].to(dev)
        acts = node_outputs(model, graph, pairs["corrupt_ids"][sl].to(dev))
        lp = clean_logp[sl].to(dev)
        for i, keep in enumerate(keeps):
            out[i, sl] = kl_2tok(patched_forward(model, graph, clean, acts, keep), lp).cpu()
        del acts
    return out


def make_judge(model, graph, data: dict, bs: int, metric: str, sanity: dict):
    """(ev, main, f_of) for a metric.

    ev(set_name, keeps): per-example values on data[set_name], (n_masks, N, 3) LD terms or
    (n_masks, N) KL; main(x): the (n_masks, N) headline value (LD, or KL(full || x));
    f_of(mean): faithfulness of a mean headline value (LD f, or KL f = 1 - KL/KL(empty)).
    The KL clean log-probs are computed once per pair set and kept on CPU.
    """
    if metric == "kl":
        logp: dict[str, torch.Tensor] = {}

        def ev(name, keeps):
            if name not in logp:
                logp[name] = clean_logprobs(model, data[name], bs)
            return eval_kl(model, graph, data[name], keeps, bs, logp[name])

        def f_of(kl_mean: float) -> float:
            return et.kl_faithfulness(kl_mean, sanity["kl_empty_mean"])[0]  # nan if degenerate

        return ev, (lambda x: x), f_of

    def ev(name, keeps):
        return eval_ld(model, graph, data[name], keeps, bs)

    def f_of(ld_mean: float) -> float:
        return et.faithfulness(ld_mean, sanity["m_empty"], sanity["m_full"])[0]

    return ev, (lambda x: x[..., 0]), f_of


@torch.no_grad()
def eval_ts_loss(model, graph, ids: torch.Tensor, mean_acts, keeps, bs: int) -> np.ndarray:
    """Mean next-token loss (nats) on the stories for each keep mask (mean ablation)."""
    dev = next(model.parameters()).device
    tot = np.zeros(len(keeps))
    for s in range(0, ids.shape[0], bs):
        x = ids[s : s + bs].to(dev)
        for i, keep in enumerate(keeps):
            logits = patched_forward(model, graph, x, mean_acts, keep)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="sum"
            )
            tot[i] += loss.item()
    return tot / (ids.shape[0] * (ids.shape[1] - 1))


@torch.no_grad()
def story_means(model, graph, ids: torch.Tensor, bs: int = 4) -> torch.Tensor:
    """Position-wise mean node outputs over the stories: (1, T, U, d)."""
    dev = next(model.parameters()).device
    acc = None
    for s in range(0, ids.shape[0], bs):
        o = node_outputs(model, graph, ids[s : s + bs].to(dev)).sum(0, keepdim=True)
        acc = o if acc is None else acc + o
    return acc / ids.shape[0]


def keep_only(graph, idx: torch.Tensor, dev) -> torch.Tensor:
    return graph.mask_from_flat(idx).to(dev)


def keep_without(graph, idx: torch.Tensor, dev) -> torch.Tensor:
    return graph.valid.to(dev) & ~graph.mask_from_flat(idx).to(dev)


def random_sets(n_edges: int, k: int, n: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(n_edges, generator=g)[:k] for _ in range(n)]


def subset(pairs: dict, mask) -> dict:
    return {k: (v[mask] if torch.is_tensor(v) and v.dim() >= 1 else v) for k, v in pairs.items()}


# ---------------------------------------------------------------------- stages


def stage_sanity(model, graph, tok, data, out: Path, bs: int) -> dict:
    dev = next(model.parameters()).device
    val = data["val"]
    # greedy exact match + digit-split check
    em, nonstd = [], []
    for s in range(0, val["clean_ids"].shape[0], bs):
        prompt = val["clean_ids"][s : s + bs, :-1].to(dev)
        gen = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=4,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )[:, prompt.shape[1]:].cpu()
        for row, ans in zip(gen, val["answer_text"][s : s + bs]):
            ids = row.tolist()
            # cut at EOS: skip_special_tokens alone would glue the post-EOS
            # continuation ("9331<eot>Once" -> "9331Once") and zero out EM
            if tok.eos_token_id in ids:
                ids = ids[: ids.index(tok.eos_token_id)]
            text = tok.decode(ids, skip_special_tokens=True).split("\n")[0]
            em.append(text.strip() == ans)
            ans_ids = [t for t in ids if t != tok.eos_token_id]
            cut = tok(text, add_special_tokens=False)["input_ids"]
            nonstd.append(text.strip().isdigit() and ans_ids[: len(cut)] != cut)
    full = torch.ones(graph.n_upstream, graph.n_receivers, dtype=torch.bool, device=dev)
    empty = torch.zeros_like(full)
    ld = eval_ld(model, graph, val, [full, empty], bs)
    # identities against plain HF forwards on one batch
    b = slice(0, min(bs, val["clean_ids"].shape[0]))
    toks = [val[k][b].to(dev) for k in ("c1", "k1", "c2", "k2")]
    with torch.no_grad():
        ref_full = logit_diff_2tok(model(val["clean_ids"][b].to(dev)).logits, *toks)[0].cpu()
        ref_empty = logit_diff_2tok(model(val["corrupt_ids"][b].to(dev)).logits, *toks)[0].cpu()
    err_full = (ld[0, b, 0] - ref_full).abs().max().item()
    err_empty = (ld[1, b, 0] - ref_empty).abs().max().item()
    assert err_full < 1e-3 and err_empty < 1e-3, (err_full, err_empty)
    # timing: one 0.1% random circuit on the validation set
    k = et.size_to_k(SIZES[0], graph.n_edges)
    keep = keep_only(graph, random_sets(graph.n_edges, k, 1, 99)[0], dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    eval_ld(model, graph, val, [keep], bs)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    res = {
        "n_edges": graph.n_edges,
        "n_val": int(val["clean_ids"].shape[0]),
        "exact_match": float(np.mean(em)),
        "em_per_example": [bool(x) for x in em],
        "nonstandard_split_share": float(np.mean(nonstd)),
        "nonstandard_per_example": [bool(x) for x in nonstd],
        "m_full": ld[0, :, 0].mean().item(),
        "m_full_terms": [ld[0, :, 1].mean().item(), ld[0, :, 2].mean().item()],
        "m_empty": ld[1, :, 0].mean().item(),
        "m_empty_terms": [ld[1, :, 1].mean().item(), ld[1, :, 2].mean().item()],
        "identity_err_full": err_full,
        "identity_err_empty": err_empty,
        "sec_per_circuit_eval": sec,
        "token_ids_10": [
            {k: int(val[k][i]) for k in ("c1", "k1", "c2", "k2")}
            | {"c1_str": tok.decode([int(val["c1"][i])]), "c2_str": tok.decode([int(val["c2"][i])]),
               "k1_str": tok.decode([int(val["k1"][i])]), "k2_str": tok.decode([int(val["k2"][i])])}
            for i in range(10)
        ],
    }
    # KL(full || empty) on val, and the full-keep identity against the plain clean forward
    kl = eval_kl(model, graph, val, [full, empty], bs, clean_logprobs(model, val, bs))
    kl_err_full = kl[0].abs().max().item()
    assert kl_err_full < KL_IDENTITY_TOL, kl_err_full
    res |= {"kl_empty_mean": kl[1].mean().item(), "kl_empty_per_example": kl[1].tolist(),
            "kl_identity_err_full": kl_err_full}
    dump(res, out / "sanity.json")
    print(f"[sanity] EM {res['exact_match']:.3f} nonstd {res['nonstandard_split_share']:.3f} "
          f"m_full {res['m_full']:.3f} m_empty {res['m_empty']:.3f} eval {sec:.2f}s")
    print(f"[sanity] KL(full||empty) {res['kl_empty_mean']:.3f} nats, "
          f"identity err {kl_err_full:.2e}")
    return res


def stage_score(model, graph, data, out: Path, bs: int, metric: str = "ld") -> None:
    dev = next(model.parameters()).device
    disc = data["disc"]
    clean_logp = clean_logprobs(model, disc, bs) if metric == "kl" else None
    rows = []
    for s in range(0, disc["clean_ids"].shape[0], bs):
        sl = slice(s, s + bs)
        toks = [disc[k][sl].to(dev) for k in ("c1", "k1", "c2", "k2")]

        if clean_logp is None:
            def m_fn(logits, toks=toks):
                return logit_diff_2tok(logits, *toks)[0]
        else:  # positive score = the edge supports reproducing the model's own clean output
            def m_fn(logits, lp=clean_logp[sl].to(dev)):
                return -kl_2tok(logits, lp)

        rows.append(eap_ig_scores(model, graph, disc["clean_ids"][sl].to(dev),
                                  disc["corrupt_ids"][sl].to(dev), m_fn, steps=5).cpu())
    per = torch.cat(rows)  # (512, E) float32
    half = disc["half"]  # 0/1 per discovery example
    torch.save({
        "mean": per.mean(0),
        "mean_a": per[half == 0].mean(0),
        "mean_b": per[half == 1].mean(0),
        "per_example": per.half(),
    }, out / "scores.pt")
    print(f"[score] {per.shape} saved; mean score sum {per.mean(0).sum():.3f}")


def stage_probe(model, graph, data, out: Path, bs: int, is_parent: bool, src: Path,
                metric: str = "ld", task: str = "word", sizes: tuple[float, ...] = PROBE_SIZES,
                draws: int | None = None) -> dict:
    """f of the top-k circuit vs random same-size sets (no frozen tests). Random sets are
    seeded like evaluate's sufficiency draws, so they are the first draws of that stage."""
    dev = next(model.parameters()).device
    sanity = json.loads((src / "sanity.json").read_text())
    mean = torch.load(src / "scores.pt")["mean"]
    E = graph.n_edges
    n_draws = PROBE_DRAWS[is_parent] if draws is None else draws
    ev, main, f_of = make_judge(model, graph, data, bs, metric, sanity)
    res: dict = {"metric": metric, "task": task, "n_draws": n_draws}
    if metric == "kl":
        res["kl_empty_mean"] = sanity["kl_empty_mean"]
        signal = sanity["kl_empty_mean"] > KL_GATE_NATS
    else:
        res |= {"m_full": sanity["m_full"], "m_empty": sanity["m_empty"]}
        signal = sanity["m_full"] - sanity["m_empty"] > 0
    res["sizes"] = {}
    for frac in sizes:
        k = et.size_to_k(frac, E)
        circ = torch.as_tensor(et.topk_edges(mean.numpy(), k))
        rnd = random_sets(E, k, n_draws, seed=SEED + int(frac * 1e5))
        keeps = [keep_only(graph, circ, dev)] + [keep_only(graph, r, dev) for r in rnd]
        f = [f_of(x) for x in main(ev("val", keeps)).mean(1).tolist()]
        p5, p50, p95 = np.percentile(f[1:], [5, 50, 95]).tolist()
        res["sizes"][str(frac)] = {"k": k, "f": f[0], "f_random": f[1:], "random_p5": p5,
                                   "random_p50": p50, "random_p95": p95,
                                   "above_p95": bool(f[0] > p95)}
        print(f"[probe] size {frac}: f {f[0]:.3f} random p50 {p50:.3f} p95 {p95:.3f}")
    # performing (HANDOFF labels): signal over the empty circuit AND f at 10% above the band
    at10 = res["sizes"].get(str(0.1))
    res["performing"] = {
        "signal": bool(signal),
        "f10_above_p95": None if at10 is None else at10["above_p95"],
        "pass": None if at10 is None else bool(signal and at10["above_p95"]),
    }
    dump(res, out / "probe.json")
    return res


def stage_evaluate(model, graph, tok, data, out: Path, bs: int, is_parent: bool,
                   parent_scores: Path | None, draws: tuple[int, int] | None = None,
                   sizes: tuple[float, ...] = SIZES, metric: str = "ld", task: str = "word",
                   src: Path | None = None) -> None:
    """LD (frozen) or KL-judged tests (HANDOFF-kl-symbol-nodeedge.md, step 6). Reads
    sanity.json + scores.pt from `src` (default `out`); writes to `out`."""
    dev = next(model.parameters()).device
    src = out if src is None else src
    sanity = json.loads((src / "sanity.json").read_text())
    sc = torch.load(src / "scores.pt")
    mean = sc["mean"]
    E = graph.n_edges
    kl = metric == "kl"
    m_full, m_empty = sanity["m_full"], sanity["m_empty"]
    slow = sanity["sec_per_circuit_eval"] > 2.0
    n_draws = 50 if (slow and is_parent) else N_DRAWS
    n_draws_ts = 10 if slow else N_DRAWS_TS
    if draws is not None:  # smoke-test override only
        n_draws, n_draws_ts = draws
    full = graph.valid.to(dev)
    res: dict = {"m_full": m_full, "m_empty": m_empty, "n_draws": n_draws,
                 "n_draws_ts": n_draws_ts, "sizes": {}}
    if kl:
        kl_empty = sanity["kl_empty_mean"]
        res = {"metric": metric, "task": task, "kl_empty_mean": kl_empty} | res
    per_ex: dict[str, np.ndarray] = {}
    # ev: per-example values, main: headline column (LD / KL(full || x)), f_of: f of a mean
    ev, main, f_of = make_judge(model, graph, data, bs, metric, sanity)

    # ---- steps 3-5: circuits, f curve, sufficiency + partial necessity draws (all sizes)
    circuits = {}
    for frac in sizes:
        k = et.size_to_k(frac, E)
        circ = torch.as_tensor(et.topk_edges(mean.numpy(), k))
        circuits[frac] = circ
        rnd = random_sets(E, k, n_draws, seed=SEED + int(frac * 1e5))
        keeps = [keep_only(graph, circ, dev)] + [keep_only(graph, r, dev) for r in rnd]
        suff = ev("val", keeps)
        keeps = [keep_without(graph, circ, dev)] + [keep_without(graph, r, dev) for r in rnd]
        nec = ev("val", keeps)
        f_suff = [f_of(x) for x in main(suff).mean(1).tolist()]
        f_nec = [f_of(x) for x in main(nec).mean(1).tolist()]
        r: dict = {"k": k}
        if kl:
            r["kl_circuit"] = suff[0].mean().item()
        else:
            r["m_circuit"] = suff[0, :, 0].mean().item()
            r["m_circuit_terms"] = suff[0, :, 1:].mean(0).tolist()
        res["sizes"][str(frac)] = r | {
            "f": f_suff[0],
            "f_random": f_suff[1:],
            "sufficiency": et.random_baseline_test(f_suff[0], np.array(f_suff[1:]), True),
            "f_removed": f_nec[0],
            "f_removed_random": f_nec[1:],
        }
        per_ex[f"suff_{frac}"] = suff[0].numpy()
        per_ex[f"nec_{frac}"] = nec[0].numpy()
        per_ex[f"nec_rand_mean_{frac}"] = main(nec[1:]).mean(1).numpy()
        print(f"[eval] size {frac}: f {f_suff[0]:.3f} suff "
              f"{res['sizes'][str(frac)]['sufficiency']['pass']}")
    res["f_log_mean"] = et.log_size_mean_f(
        list(sizes), [res["sizes"][str(s)]["f"] for s in sizes])

    # ---- step 6: parent validity gate
    any_suff = any(res["sizes"][str(s)]["sufficiency"]["pass"] for s in sizes)
    if kl:
        gate = kl_empty > KL_GATE_NATS and any_suff
        res["validity_gate"] = {"kl_empty_above": kl_empty > KL_GATE_NATS, "pass": bool(gate),
                                "f_degenerate": et.kl_faithfulness(0.0, kl_empty)[1]}
    else:
        gate = m_full > 0 and any_suff
        res["validity_gate"] = {"m_full_pos": m_full > 0, "pass": bool(gate),
                                "f_degenerate": et.faithfulness(m_full, m_empty, m_full)[1]}

    # ---- step 7: remaining tests with the stopping rule (KL: every size, no early break)
    by_size: dict[float, dict] = {}
    if gate or not is_parent:
        full_m = main(ev("val", [full]))[0].numpy()
        copy_full = main(ev("copy", [full]))[0].numpy()
        per_ex["full"] = full_m
        per_ex["copy_full"] = copy_full
        if kl:  # the copy task's own KL(full || empty), for its relative damage
            copy_empty = main(ev("copy", [torch.zeros_like(full)]))[0].numpy()
            per_ex["copy_empty"] = copy_empty
            res["copy_kl_empty_mean"] = float(copy_empty.mean())
        for frac in sizes:
            r = res["sizes"][str(frac)]
            k, circ = r["k"], circuits[frac]
            t: dict = {}
            t["sufficiency"] = r["sufficiency"]["pass"]
            if kl:
                eq = et.kl_equivalence(per_ex[f"suff_{frac}"], kl_empty, EQUIV_EPS_FRAC)
            else:
                eq = et.tost_equivalence(per_ex[f"suff_{frac}"][:, 0], full_m,
                                         EQUIV_EPS_FRAC * abs(m_full))
            r["equivalence"] = eq
            t["equivalence"] = eq["pass"]
            r["partial_necessity"] = et.random_baseline_test(
                r["f_removed"], np.array(r["f_removed_random"]), False)
            t["partial_necessity"] = r["partial_necessity"]["pass"]
            # consistency: per-example circuits from the discovery per-example scores
            cons = et.consistency(sc["per_example"].float().numpy(), k, share=0.5)
            shared = torch.as_tensor(np.asarray(cons["shared"], dtype=np.int64))
            ks = max(len(shared), 1)
            rnd = random_sets(E, ks, n_draws, seed=SEED + 7 + int(frac * 1e5))
            if len(shared):
                keeps = [keep_without(graph, shared, dev)] + [keep_without(graph, x, dev) for x in rnd]
                cl = main(ev("val", keeps)).mean(1).numpy()
                if kl:  # judged on KL f (lower = more damage, like LD)
                    cl = np.array([f_of(x) for x in cl])
                cons_abl = et.random_baseline_test(cl[0], cl[1:], False)
            else:
                cons_abl = {"empirical_pvalue": 1.0, "pass": False}
            r["consistency"] = {k2: v for k2, v in cons.items() if k2 != "shared"} | {
                "n_shared": len(shared), "ablation": cons_abl,
                "pass": et.consistency_pass(cons["mean_coverage"], cons_abl["empirical_pvalue"]),
            }
            t["consistency"] = r["consistency"]["pass"]
            # specificity: same circuit removed on the copy task
            copy_abl = main(ev("copy", [keep_without(graph, circ, dev)]))[0]
            if kl:  # relative damage KL(full || without C) / KL(full || empty), per task
                copy_ref = float(copy_empty.mean())
                spec = et.kl_specificity(
                    float(per_ex[f"nec_{frac}"].mean()) / kl_empty,
                    copy_abl.mean().item() / copy_ref if copy_ref >= 1e-8 else None,  # as kl_faithfulness
                    per_ex[f"nec_rand_mean_{frac}"] / kl_empty)
            else:
                add_drop_random = m_full - per_ex[f"nec_rand_mean_{frac}"]
                add_abl = float(per_ex[f"nec_{frac}"][:, 0].mean())
                if m_full > 0:
                    spec = et.specificity(m_full, add_abl, float(copy_full.mean()),
                                          copy_abl.mean().item(), add_drop_random)
                else:  # relative drop undefined; only a child with m(full) <= 0 reaches here
                    spec = {"status": "not_measurable", "pass": None, "reason": "m_full <= 0"}
            r["specificity"] = spec
            t["specificity"] = spec["pass"]
            by_size[frac] = t
            print(f"[eval] tests @ {frac}: {t}")
            if not kl and all(v is not False for v in t.values()):
                break
        res["tests"] = {str(s): v for s, v in by_size.items()}
        sel = et.stopping_rule(by_size)
        res["selected_size"] = sel

        # ---- step 8: TinyStories check at the selected size
        if sel is not None:
            ids = data["stories"]
            means = story_means(model, graph, ids)
            k = res["sizes"][str(sel)]["k"]
            rnd = random_sets(E, k, n_draws_ts, seed=SEED + 11)
            keeps = [full, keep_without(graph, circuits[sel], dev)] + [
                keep_without(graph, x, dev) for x in rnd]
            losses = eval_ts_loss(model, graph, ids, means, keeps, bs=8)
            band = np.percentile(losses[2:], [5, 50, 95]).tolist()
            res["tinystories"] = {"loss_full": losses[0], "loss_circuit_ablated": losses[1],
                                  "loss_random": losses[2:].tolist(), "random_band": band,
                                  "flag": bool(losses[1] > band[2])}
            del means
    elif kl:  # a KL parent failing the gate stops after sufficiency + partial necessity
        for frac in sizes:
            r = res["sizes"][str(frac)]
            r["partial_necessity"] = et.random_baseline_test(
                r["f_removed"], np.array(r["f_removed_random"]), False)

    # ---- children: parent's circuit inside the child, real patching of the top 20
    if not is_parent:
        if parent_scores is not None:
            pmean = torch.load(parent_scores)["mean"]
            res["parent_in_child"] = {}
            for frac in sizes:
                k = et.size_to_k(frac, E)
                pc = torch.as_tensor(et.topk_edges(pmean.numpy(), k))
                ld = main(ev("val", [keep_only(graph, pc, dev)]))[0]
                res["parent_in_child"][str(frac)] = {
                    "f_parent_circuit": f_of(ld.mean().item()),
                    "f_own": res["sizes"][str(frac)]["f"],
                    "f_random_band": np.percentile(
                        res["sizes"][str(frac)]["f_random"], [5, 50, 95]).tolist(),
                }
        top = torch.as_tensor(et.topk_edges(mean.numpy(), 20))
        keeps = [full] + [keep_without(graph, e.view(1), dev) for e in top]
        ld = main(ev("disc", keeps)).mean(1).numpy()
        if kl:  # the scored metric is -KL, so its drop is KL(without e) - KL(full)
            ld = -ld
        actual = (ld[0] - ld[1:]).tolist()
        pred = mean[top].tolist()
        res["real_patching_top20"] = {
            "edges": [graph.edge_names()[i] for i in top.tolist()],
            "eapig_score": pred, "actual_drop": actual,
            "spearman": float(_spearman(np.array(pred), np.array(actual))),
        }
    dump(res, out / "evaluate.json")
    np.savez_compressed(out / "per_example.npz", **per_ex)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return spearmanr(a, b).statistic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    choices=["elicit_parent", "elicit_child", "fmt_parent", "teach_child"])
    ap.add_argument("--model", required=True, help="local dir or HF repo id")
    ap.add_argument("--stage", nargs="+", default=["sanity", "score", "evaluate"],
                    choices=["sanity", "score", "probe", "evaluate"])
    ap.add_argument("--task", default="word", choices=["word", "symbol"])
    ap.add_argument("--metric", default="ld", choices=["ld", "kl"])
    ap.add_argument("--parent-tag", default=None)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--scores-dir", default=None,
                    help="where probe/evaluate read scores.pt + sanity.json (default --results)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sizes", type=float, nargs="+", default=list(SIZES),
                    help="circuit sizes (fractions of edges) for the evaluate stage")
    ap.add_argument("--probe-sizes", type=float, nargs="+", default=list(PROBE_SIZES),
                    help="circuit sizes for the probe stage")
    ap.add_argument("--smoke-draws", type=int, nargs=2, default=None,
                    help="override (draws, tinystories draws); CPU smoke test only")
    a = ap.parse_args(argv)
    # guard: only the frozen path (word, LD, no probe) may write to the frozen record dirs
    frozen = a.task == "word" and a.metric == "ld" and "probe" not in a.stage
    if not frozen and Path(a.results).resolve() in [(HERE / d).resolve() for d in FROZEN_RESULTS]:
        ap.error(f"--results {a.results} holds the frozen LD-word record; write "
                 "task/metric/probe outputs to their own dir (e.g. results_klword)")
    if a.task == "symbol" and not a.tag.startswith("elicit"):
        ap.error("--task symbol is for the elicit route only (owner, 2026-10-08)")
    if "evaluate" in a.stage and a.task != "word":
        ap.error("evaluate runs on the word task only (copy task + tests are defined there)")
    return a


def run_stages(a: argparse.Namespace, model, graph, tok, data: dict) -> None:
    out = Path(a.results) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.scores_dir or a.results) / a.tag
    is_parent = a.tag.endswith("parent")
    pscores = (Path(a.scores_dir or a.results) / a.parent_tag / "scores.pt"
               if a.parent_tag else None)
    for st in a.stage:
        t0 = time.time()
        if st == "sanity":
            stage_sanity(model, graph, tok, data, out, a.batch_size)
        elif st == "score":
            stage_score(model, graph, data, out, a.batch_size, a.metric)
        elif st == "probe":
            stage_probe(model, graph, data, out, a.batch_size, is_parent, src, a.metric,
                        a.task, tuple(a.probe_sizes),
                        a.smoke_draws[0] if a.smoke_draws else None)
        elif st == "evaluate":
            stage_evaluate(model, graph, tok, data, out, a.batch_size, is_parent, pscores,
                           tuple(a.smoke_draws) if a.smoke_draws else None, tuple(a.sizes),
                           a.metric, a.task, src)
        print(f"[{a.tag}] stage {st} done in {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    from transformers import AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = load_model(a.model, a.device)
    graph = build_graph(model)
    assert graph.n_edges == 195_865, graph.n_edges
    data = load_data(Path(a.data_dir), a.task)
    run_stages(a, model, graph, tok, data)


if __name__ == "__main__":
    main()
