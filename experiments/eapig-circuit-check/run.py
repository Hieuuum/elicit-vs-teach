"""EAP-IG edge-circuit check, one model per invocation (PLAN.md, frozen 2026-10-01).

Stages (run on the GPU box, in this order across the four models):
  sanity    accuracy, digit-split check, token IDs, m(full)/m(empty) identities, timing
  score     EAP-IG (inputs, 5 steps) per-example scores on the 512 discovery pairs
  evaluate  f curve + sufficiency at all sizes, parent gate, remaining tests with the
            stopping rule, TinyStories check, parent-in-child (children), real patching
            of the top-20 edges (children)

Usage:
  python3 run.py --tag elicit_child --model <dir|hf repo> --stage sanity score evaluate \
      [--parent-tag elicit_parent] [--device cuda]
Outputs: results/<tag>/{sanity.json, scores.pt, evaluate.json, per_example.npz}.
`compare.py` (CPU) reads them for the parent-vs-child comparison and plots.
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
    build_graph,
    eap_ig_scores,
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


def load_data(data_dir: Path):
    from data import load_pairs  # experiments/eapig-circuit-check/data.py

    return load_pairs(data_dir)


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
    dump(res, out / "sanity.json")
    print(f"[sanity] EM {res['exact_match']:.3f} nonstd {res['nonstandard_split_share']:.3f} "
          f"m_full {res['m_full']:.3f} m_empty {res['m_empty']:.3f} eval {sec:.2f}s")
    return res


def stage_score(model, graph, data, out: Path, bs: int) -> None:
    dev = next(model.parameters()).device
    disc = data["disc"]
    rows = []
    for s in range(0, disc["clean_ids"].shape[0], bs):
        sl = slice(s, s + bs)
        toks = [disc[k][sl].to(dev) for k in ("c1", "k1", "c2", "k2")]

        def metric(logits, toks=toks):
            return logit_diff_2tok(logits, *toks)[0]

        rows.append(eap_ig_scores(model, graph, disc["clean_ids"][sl].to(dev),
                                  disc["corrupt_ids"][sl].to(dev), metric, steps=5).cpu())
    per = torch.cat(rows)  # (512, E) float32
    half = disc["half"]  # 0/1 per discovery example
    torch.save({
        "mean": per.mean(0),
        "mean_a": per[half == 0].mean(0),
        "mean_b": per[half == 1].mean(0),
        "per_example": per.half(),
    }, out / "scores.pt")
    print(f"[score] {per.shape} saved; mean score sum {per.mean(0).sum():.3f}")


def stage_evaluate(model, graph, tok, data, out: Path, bs: int, is_parent: bool,
                   parent_scores: Path | None, draws: tuple[int, int] | None = None) -> None:
    dev = next(model.parameters()).device
    sanity = json.loads((out / "sanity.json").read_text())
    sc = torch.load(out / "scores.pt")
    mean = sc["mean"]
    val = data["val"]
    E = graph.n_edges
    m_full, m_empty = sanity["m_full"], sanity["m_empty"]
    slow = sanity["sec_per_circuit_eval"] > 2.0
    n_draws = 50 if (slow and is_parent) else N_DRAWS
    n_draws_ts = 10 if slow else N_DRAWS_TS
    if draws is not None:  # smoke-test override only
        n_draws, n_draws_ts = draws
    full = graph.valid.to(dev)
    res: dict = {"m_full": m_full, "m_empty": m_empty, "n_draws": n_draws,
                 "n_draws_ts": n_draws_ts, "sizes": {}}
    per_ex: dict[str, np.ndarray] = {}

    def f_of(ld_mean: float) -> float:
        return et.faithfulness(ld_mean, m_empty, m_full)[0]  # nan when degenerate

    # ---- steps 3-5: circuits, f curve, sufficiency + partial necessity draws (all sizes)
    circuits = {}
    for frac in SIZES:
        k = et.size_to_k(frac, E)
        circ = torch.as_tensor(et.topk_edges(mean.numpy(), k))
        circuits[frac] = circ
        rnd = random_sets(E, k, n_draws, seed=SEED + int(frac * 1e5))
        keeps = [keep_only(graph, circ, dev)] + [keep_only(graph, r, dev) for r in rnd]
        suff = eval_ld(model, graph, val, keeps, bs)
        keeps = [keep_without(graph, circ, dev)] + [keep_without(graph, r, dev) for r in rnd]
        nec = eval_ld(model, graph, val, keeps, bs)
        f_suff = [f_of(x) for x in suff[:, :, 0].mean(1).tolist()]
        f_nec = [f_of(x) for x in nec[:, :, 0].mean(1).tolist()]
        res["sizes"][str(frac)] = {
            "k": k,
            "m_circuit": suff[0, :, 0].mean().item(),
            "m_circuit_terms": suff[0, :, 1:].mean(0).tolist(),
            "f": f_suff[0],
            "f_random": f_suff[1:],
            "sufficiency": et.random_baseline_test(f_suff[0], np.array(f_suff[1:]), True),
            "f_removed": f_nec[0],
            "f_removed_random": f_nec[1:],
        }
        per_ex[f"suff_{frac}"] = suff[0].numpy()
        per_ex[f"nec_{frac}"] = nec[0].numpy()
        per_ex[f"nec_rand_mean_{frac}"] = nec[1:, :, 0].mean(1).numpy()
        print(f"[eval] size {frac}: f {f_suff[0]:.3f} suff "
              f"{res['sizes'][str(frac)]['sufficiency']['pass']}")
    res["f_log_mean"] = et.log_size_mean_f(
        list(SIZES), [res["sizes"][str(s)]["f"] for s in SIZES])

    # ---- step 6: parent validity gate
    gate = m_full > 0 and any(res["sizes"][str(s)]["sufficiency"]["pass"] for s in SIZES)
    res["validity_gate"] = {"m_full_pos": m_full > 0, "pass": bool(gate),
                            "f_degenerate": et.faithfulness(m_full, m_empty, m_full)[1]}

    # ---- step 7: remaining tests with the stopping rule
    by_size: dict[float, dict] = {}
    if gate or not is_parent:
        full_ld = eval_ld(model, graph, val, [full], bs)[0, :, 0].numpy()
        copy = data["copy"]
        copy_full = eval_ld(model, graph, copy, [full], bs)[0, :, 0].numpy()
        per_ex["full"] = full_ld
        per_ex["copy_full"] = copy_full
        for frac in SIZES:
            r = res["sizes"][str(frac)]
            k, circ = r["k"], circuits[frac]
            t: dict = {}
            t["sufficiency"] = r["sufficiency"]["pass"]
            eq = et.tost_equivalence(per_ex[f"suff_{frac}"][:, 0], full_ld,
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
                cl = eval_ld(model, graph, val, keeps, bs)[:, :, 0].mean(1).numpy()
                cons_abl = et.random_baseline_test(cl[0], cl[1:], False)
            else:
                cons_abl = {"empirical_pvalue": 1.0, "pass": False}
            r["consistency"] = {k2: v for k2, v in cons.items() if k2 != "shared"} | {
                "n_shared": len(shared), "ablation": cons_abl,
                "pass": et.consistency_pass(cons["mean_coverage"], cons_abl["empirical_pvalue"]),
            }
            t["consistency"] = r["consistency"]["pass"]
            # specificity: same circuit removed on the copy task
            copy_abl = eval_ld(model, graph, copy, [keep_without(graph, circ, dev)], bs)[0, :, 0]
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
            if all(v is not False for v in t.values()):
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

    # ---- children: parent's circuit inside the child, real patching of the top 20
    if not is_parent:
        if parent_scores is not None:
            pmean = torch.load(parent_scores)["mean"]
            res["parent_in_child"] = {}
            for frac in SIZES:
                k = et.size_to_k(frac, E)
                pc = torch.as_tensor(et.topk_edges(pmean.numpy(), k))
                ld = eval_ld(model, graph, val, [keep_only(graph, pc, dev)], bs)[0, :, 0]
                res["parent_in_child"][str(frac)] = {
                    "f_parent_circuit": f_of(ld.mean().item()),
                    "f_own": res["sizes"][str(frac)]["f"],
                    "f_random_band": np.percentile(
                        res["sizes"][str(frac)]["f_random"], [5, 50, 95]).tolist(),
                }
        top = torch.as_tensor(et.topk_edges(mean.numpy(), 20))
        disc = data["disc"]
        keeps = [full] + [keep_without(graph, e.view(1), dev) for e in top]
        ld = eval_ld(model, graph, disc, keeps, bs)[:, :, 0].mean(1).numpy()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    choices=["elicit_parent", "elicit_child", "fmt_parent", "teach_child"])
    ap.add_argument("--model", required=True, help="local dir or HF repo id")
    ap.add_argument("--stage", nargs="+", default=["sanity", "score", "evaluate"])
    ap.add_argument("--parent-tag", default=None)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke-draws", type=int, nargs=2, default=None,
                    help="override (draws, tinystories draws); CPU smoke test only")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = load_model(a.model, a.device)
    graph = build_graph(model)
    assert graph.n_edges == 195_865, graph.n_edges
    data = load_data(Path(a.data_dir))
    out = Path(a.results) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    is_parent = a.tag.endswith("parent")
    pscores = Path(a.results) / a.parent_tag / "scores.pt" if a.parent_tag else None
    for st in a.stage:
        t0 = time.time()
        if st == "sanity":
            stage_sanity(model, graph, tok, data, out, a.batch_size)
        elif st == "score":
            stage_score(model, graph, data, out, a.batch_size)
        elif st == "evaluate":
            stage_evaluate(model, graph, tok, data, out, a.batch_size, is_parent, pscores,
                           tuple(a.smoke_draws) if a.smoke_draws else None)
        print(f"[{a.tag}] stage {st} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
