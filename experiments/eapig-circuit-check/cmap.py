"""Edge-level cross-model activation patching (CMAP), child -> parent (HANDOFF-cmap.md).

After Prakash et al. 2024 (arXiv 2402.14811), who patch head outputs fine-tuned -> base;
here we patch EDGES child -> parent. For each NL val prompt, the parent runs with every
edge u -> r in a mask carrying the CHILD's output of u into the parent's receiver r; all
other edges, and every node downstream, use the parent's own weights. That is
`patched_forward(parent, graph, ids, node_outputs(child, ...), keep)` with keep = 0 on
the patched edges ("corrupt" = the child's clean run).

Per mask, means over the val prompts:
  recovery = 1 - mean KL(child || patched) / mean KL(child || parent)  (0 = parent, 1 = child)
  acc2     = two-token teacher-forced accuracy (argmax at -2 == c1 and at -1 == c2)
  logp     = mean log p(c1) + log p(c2)
Masks per size: the child's top-k (`child`), the same minus edges into `logits`
(`child_nologits`, the main read), each --extra-circuit and its nologits version, and
--n-random random k-sets with and without the same removal, and --n-random `typed`
random sets matched to `child_nologits` in size and edge-type/layer mix (the fair null
for the main read). Per extra circuit L, the child's nologits circuit is also split into
`child_and_L_nologits` (edges L shares) and `child_minus_L_nologits` (the rest). Once per route: `all` (every
edge patched; the ceiling, not exactly the child: the parent's final norm + tied
unembedding still apply) and `none` (all kept; must equal the parent, a runtime check).

  python3 cmap.py --route elicit --parent M --child M --child-scores DIR \
      [--extra-circuit ep_sym=DIR] --sizes 0.001 ... --out results_cmap/elicit.json
  python3 cmap.py --merge results_cmap/elicit_0.001.json ... --out results_cmap/elicit.json
  python3 cmap.py --summarize results_cmap/elicit.json results_cmap/teach.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

import run
from run import et
from geode.circuits.eapig import answer_logprobs, kl_2tok, node_outputs, patched_forward

SIZES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
N_BOOT = 1000
DENOM_TOL = 1e-8  # mean KL(child || parent) below this: recovery undefined (NaN)
CHECK_TOL = 1e-4  # nats: runtime identity checks


def drop_logits_edges(graph, idx: torch.Tensor) -> torch.Tensor:
    """`idx` minus every edge whose receiver is `logits` (order kept)."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    _, r = graph.flat_to_ur(idx)
    return idx[r != graph.logits_index]


def edge_strata(graph) -> torch.Tensor:
    """(E,) stratum id per edge: (upstream kind + layer, receiver kind + layer), e.g.
    `a3.h -> a5.q`, `m2 -> m4.in`, `embed -> logits`."""
    up = [re.sub(r"\.h\d+$", ".h", n) for n in graph.upstream_names]
    rc = [re.sub(r"\d+$", "", n) for n in graph.receiver_names]
    keys = [f"{up[u]}->{rc[r]}" for u, r in graph.edge_ur.tolist()]
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return torch.tensor([ids[k] for k in keys], dtype=torch.long)


def typed_random_sets(graph, idx: torch.Tensor, n: int, seed: int,
                      strata: torch.Tensor | None = None) -> list[torch.Tensor]:
    """`n` random edge sets with the same number of edges per `edge_strata` stratum as
    `idx` (so the same size and the same mix of edge types and layers)."""
    strata = edge_strata(graph) if strata is None else strata
    idx = torch.as_tensor(idx, dtype=torch.long)
    need = torch.bincount(strata[idx], minlength=int(strata.max()) + 1)
    pools = [(int(c), (strata == st).nonzero().squeeze(1)) for st, c in enumerate(need.tolist()) if c]
    g = torch.Generator().manual_seed(seed)
    return [torch.cat([pool[torch.randperm(len(pool), generator=g)[:c]] for c, pool in pools])
            if pools else idx.new_empty(0) for _ in range(n)]


def recovery(kl_patched: float, kl_parent: float) -> float:
    """1 - KL(child || patched) / KL(child || parent); NaN if the denominator is ~0."""
    if not kl_parent > DENOM_TOL:
        return float("nan")
    return 1.0 - kl_patched / kl_parent


def boot_ci(kl_mask: np.ndarray, kl_parent: np.ndarray, seed: int, n: int = N_BOOT) -> list[float]:
    """95% bootstrap CI of recovery, resampling prompts (paired)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kl_parent), (n, len(kl_parent)))
    num, den = kl_mask[idx].mean(1), kl_parent[idx].mean(1)
    rec = np.where(den > DENOM_TOL, 1.0 - num / np.maximum(den, DENOM_TOL), np.nan)
    return np.nanpercentile(rec, [2.5, 97.5]).tolist()


def _answer_stats(lp: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor):
    """(acc2, logp) per example from (B, 2, V) answer log-probs."""
    rows = torch.arange(lp.shape[0], device=lp.device)
    acc = (lp[:, 0].argmax(-1) == c1) & (lp[:, 1].argmax(-1) == c2)
    return acc, lp[rows, 0, c1] + lp[rows, 1, c2]


@torch.no_grad()
def eval_cmap(parent, child, graph, pairs: dict, keeps: list[torch.Tensor], bs: int,
              check_n: int = 0) -> dict:
    """Per-example CMAP values, batch-outer (one child + one plain parent forward per batch).

    Returns CPU tensors: kl/acc/logp (n_masks, N) for the patched parent; kl_parent,
    acc/logp_parent, acc/logp_child (N,), and `checks` on the first batch for the first
    `check_n` masks: child self-patch == child, sparse == dense (max |dKL|, nats).
    """
    dev = next(parent.parameters()).device
    n, m = pairs["clean_ids"].shape[0], len(keeps)
    out = {k: torch.empty(m, n) for k in ("kl", "logp")} | {"acc": torch.empty(m, n, dtype=torch.bool)}
    for k in ("kl_parent", "logp_parent", "logp_child"):
        out[k] = torch.empty(n)
    out["acc_parent"] = torch.empty(n, dtype=torch.bool)
    out["acc_child"] = torch.empty(n, dtype=torch.bool)
    checks: dict = {}
    for s in range(0, n, bs):
        sl = slice(s, s + bs)
        ids = pairs["clean_ids"][sl].to(dev)
        c1, c2 = pairs["c1"][sl].to(dev), pairs["c2"][sl].to(dev)
        child_lp = answer_logprobs(child(ids).logits)
        src = node_outputs(child, graph, ids)
        par_logits = parent(ids).logits
        out["kl_parent"][sl] = kl_2tok(par_logits, child_lp).cpu()
        a, lpc = _answer_stats(answer_logprobs(par_logits), c1, c2)
        out["acc_parent"][sl], out["logp_parent"][sl] = a.cpu(), lpc.cpu()
        a, lpc = _answer_stats(child_lp, c1, c2)
        out["acc_child"][sl], out["logp_child"][sl] = a.cpu(), lpc.cpu()
        for i, keep in enumerate(keeps):
            logits = patched_forward(parent, graph, ids, src, keep, last_n=2)
            out["kl"][i, sl] = kl_2tok(logits, child_lp).cpu()
            a, lpc = _answer_stats(answer_logprobs(logits), c1, c2)
            out["acc"][i, sl], out["logp"][i, sl] = a.cpu(), lpc.cpu()
        if check_n and s == 0:
            checks = _checks(parent, child, graph, ids, src, child_lp, keeps[:check_n],
                             out["kl"][:check_n, sl])
        del src
    out["checks"] = checks
    return out


def _checks(parent, child, graph, ids, src, child_lp, keeps, kl_sparse) -> dict:
    """Runtime identities on one batch (nats): patching the child with its own node outputs
    must give the child; the dense patch path must agree with the sparse one."""
    self_err = max(kl_2tok(patched_forward(child, graph, ids, src, keep, last_n=2), child_lp)
                   .abs().max().item() for keep in keeps)
    old = os.environ.get("GEODE_EAPIG_DENSE")
    os.environ["GEODE_EAPIG_DENSE"] = "1"
    try:
        dense = torch.stack([kl_2tok(patched_forward(parent, graph, ids, src, keep, last_n=2),
                                     child_lp).cpu() for keep in keeps])
    finally:
        if old is None:
            os.environ.pop("GEODE_EAPIG_DENSE")
        else:
            os.environ["GEODE_EAPIG_DENSE"] = old
    return {"child_self_patch_max_kl": self_err,
            "sparse_vs_dense_max_dkl": (dense - kl_sparse).abs().max().item()}


@torch.no_grad()
def direction_checks(parent, child, graph, ids: torch.Tensor) -> dict:
    """Mask-direction identities on one batch (KL in nats at the two answer positions):
    patching every embed -> * edge == the parent run on the child's embeddings; patching
    every edge into logits == patching every edge."""
    dev = ids.device
    src = node_outputs(child, graph, ids)
    ur = graph.edge_ur
    emb = (ur[:, 0] == 0).nonzero().squeeze(1)
    ref = answer_logprobs(parent(inputs_embeds=child.model.embed_tokens(ids)).logits)
    got = patched_forward(parent, graph, ids, src, run.keep_without(graph, emb, dev), last_n=2)
    lg = (ur[:, 1] == graph.logits_index).nonzero().squeeze(1)
    a = patched_forward(parent, graph, ids, src, run.keep_without(graph, lg, dev), last_n=2)
    b = patched_forward(parent, graph, ids, src,
                        torch.zeros(graph.valid.shape, dtype=torch.bool, device=dev), last_n=2)
    return {"embed_edges_vs_child_embeds_max_kl": kl_2tok(got, ref).abs().max().item(),
            "logits_edges_vs_all_max_kl": kl_2tok(a, answer_logprobs(b)).abs().max().item()}


def _mask_row(kl, acc, logp, kl_parent, seed) -> dict:
    kp = float(kl_parent.mean())
    return {"kl_mean": float(kl.mean()), "recovery": recovery(float(kl.mean()), kp),
            "recovery_ci95": boot_ci(kl.numpy().astype(np.float64),
                                     kl_parent.numpy().astype(np.float64), seed),
            "acc2": float(acc.float().mean()), "logp_correct": float(logp.mean())}


def run_cmap(parent, child, graph, val: dict, circuits: dict[str, torch.Tensor],
             sizes, n_random: int, bs: int) -> tuple[dict, dict]:
    """CMAP over all sizes. `circuits` maps label -> score vector (E,); the label `child`
    must be present. Returns (result dict, per-example arrays for np.savez)."""
    assert "child" in circuits
    dev = next(parent.parameters()).device
    E = graph.n_edges
    res: dict = {"n_val": int(val["clean_ids"].shape[0]), "n_edges": E, "n_random": n_random,
                 "sizes": {}}
    per_ex: dict[str, np.ndarray] = {}
    full = graph.valid.to(dev)
    t0 = time.time()
    ref = eval_cmap(parent, child, graph, val, [full, torch.zeros_like(full)], bs, check_n=2)
    kp = ref["kl_parent"]
    res["baseline"] = {
        "kl_child_parent_mean": float(kp.mean()),
        "acc2_parent": float(ref["acc_parent"].float().mean()),
        "acc2_child": float(ref["acc_child"].float().mean()),
        "logp_correct_parent": float(ref["logp_parent"].mean()),
        "logp_correct_child": float(ref["logp_child"].mean()),
    }
    res["none"] = _mask_row(ref["kl"][0], ref["acc"][0], ref["logp"][0], kp, run.SEED)
    res["all"] = _mask_row(ref["kl"][1], ref["acc"][1], ref["logp"][1], kp, run.SEED + 1)
    # all-ones keep is the plain parent: KL(child || none) must equal the denominator
    res["checks"] = ref["checks"] | {
        "none_vs_parent_max_dkl": (ref["kl"][0] - kp).abs().max().item()} | direction_checks(
        parent, child, graph, val["clean_ids"][:bs].to(dev))
    strata = edge_strata(graph)
    per_ex |= {"kl_parent": kp.numpy(), "kl_all": ref["kl"][1].numpy(),
               "acc_parent": ref["acc_parent"].numpy(), "acc_child": ref["acc_child"].numpy()}
    rec_all = res["all"]["recovery"]
    print(f"[cmap] KL(child||parent) {kp.mean():.4f} nats; all-patched recovery {rec_all:.3f}; "
          f"checks {res['checks']}", flush=True)

    for frac in sizes:
        k = et.size_to_k(frac, E)
        sets, removed = {}, {}
        for lab, sc in circuits.items():
            top = torch.as_tensor(et.topk_edges(sc.numpy(), k))
            sets[lab] = top
            sets[f"{lab}_nologits"] = drop_logits_edges(graph, top)
            removed[lab] = k - len(sets[f"{lab}_nologits"])
        own_nl = set(sets["child_nologits"].tolist())
        for lab in [x for x in circuits if x != "child"]:  # split the child circuit by reuse
            other = set(sets[f"{lab}_nologits"].tolist())
            sets[f"child_and_{lab}_nologits"] = torch.tensor(sorted(own_nl & other), dtype=torch.long)
            sets[f"child_minus_{lab}_nologits"] = torch.tensor(sorted(own_nl - other), dtype=torch.long)
        rnd = run.random_sets(E, k, n_random, seed=run.SEED + 202 + int(frac * 1e5))
        rnd_nl = [drop_logits_edges(graph, x) for x in rnd]
        rnd_typed = typed_random_sets(graph, sets["child_nologits"], n_random,
                                      seed=run.SEED + 303 + int(frac * 1e5), strata=strata)
        labels = list(sets)
        idxs = [sets[lab] for lab in labels] + rnd + rnd_nl + rnd_typed
        keeps = [run.keep_without(graph, x, dev) for x in idxs]
        nl = len(labels)
        ev = eval_cmap(parent, child, graph, val, keeps, bs, check_n=nl)
        for key, v in ev["checks"].items():
            res["checks"][key] = max(res["checks"][key], v)
        rows = {lab: _mask_row(ev["kl"][i], ev["acc"][i], ev["logp"][i], kp,
                               run.SEED + 10 + i + int(frac * 1e5)) | {"n_edges": len(sets[lab])}
                for i, lab in enumerate(labels)}
        for lab in rows:
            rows[lab]["frac_of_all"] = rows[lab]["recovery"] / rec_all if rec_all else float("nan")
        kpm = float(kp.mean())

        def band(lo: int, hi: int) -> dict:
            r = [recovery(float(ev["kl"][i].mean()), kpm) for i in range(lo, hi)]
            a = [float(ev["acc"][i].float().mean()) for i in range(lo, hi)]
            return {"recovery": r, "recovery_band": np.percentile(r, [5, 50, 95]).tolist(),
                    "acc2_band": np.percentile(a, [5, 50, 95]).tolist()}

        res["sizes"][str(frac)] = {
            "k": k, "logits_edges_removed": removed, "masks": rows,
            "random": band(nl, nl + n_random),
            "random_nologits": band(nl + n_random, nl + 2 * n_random)
            | {"logits_edges_removed": [k - len(x) for x in rnd_nl]},
            # same size and edge-type/layer mix as child_nologits: the matched null for (b)
            "random_typed": band(nl + 2 * n_random, nl + 3 * n_random),
        }
        for i, lab in enumerate(labels):
            per_ex[f"kl_{lab}_{frac}"] = ev["kl"][i].numpy()
        r = res["sizes"][str(frac)]
        print(f"[cmap] {frac} k={k}: " + " ".join(
            f"{lab} {v['recovery']:.3f}" for lab, v in rows.items())
            + f" | random p95 {r['random']['recovery_band'][2]:.4f}"
            f" nologits p95 {r['random_nologits']['recovery_band'][2]:.4f}"
            f" typed p95 {r['random_typed']['recovery_band'][2]:.4f}"
            f" | removed {removed} [{time.time() - t0:.0f}s]", flush=True)
    res["sec_total"] = time.time() - t0
    return res, per_ex


# --------------------------------------------------------------------- merge


def merge(paths: list[str], out: Path) -> dict:
    """Merge per-size shard outputs of ONE route (json + npz) into one file. Shared
    fields (route, models, baseline, none, all) must agree across shards; checks take
    the max; sizes are unioned (a size in two shards is an error)."""
    shards = [json.loads(Path(p).read_text()) for p in paths]
    res = dict(shards[0])
    res["sizes"], res["checks"], res["sec_total"] = {}, {}, 0.0
    for sh in shards:
        for key in ("route", "parent", "child", "circuits", "n_val", "n_edges", "n_random"):
            assert sh[key] == shards[0][key], (key, sh[key], shards[0][key])
        for key in ("kl_child_parent_mean", "acc2_parent", "acc2_child"):
            assert abs(sh["baseline"][key] - shards[0]["baseline"][key]) < 1e-5, key
        assert abs(sh["all"]["recovery"] - shards[0]["all"]["recovery"]) < 1e-5
        dup = set(sh["sizes"]) & set(res["sizes"])
        assert not dup, f"size in two shards: {dup}"
        res["sizes"] |= sh["sizes"]
        for key, v in sh["checks"].items():
            res["checks"][key] = max(res["checks"].get(key, 0.0), v)
        res["sec_total"] += sh["sec_total"]
    res["sizes"] = dict(sorted(res["sizes"].items(), key=lambda kv: float(kv[0])))
    res["checks_pass"] = all(sh["checks_pass"] for sh in shards)
    res["shards"] = [str(p) for p in paths]
    run.dump(res, out)
    arrays: dict[str, np.ndarray] = {}
    for p in paths:
        with np.load(Path(p).with_suffix(".npz")) as z:
            arrays |= {k: z[k] for k in z.files}
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    return res


# --------------------------------------------------------------------- summary


def summarize(paths: list[str]) -> str:
    """Markdown table: size x route x {child, child_nologits, extra, random p95, all}."""
    lines = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        b = r["baseline"]
        lines.append(
            f"\n**{r['route']}** ({r['parent']} <- {r['child']}): KL(child||parent) "
            f"{b['kl_child_parent_mean']:.3f} nats; acc2 parent {b['acc2_parent']:.3f}, child "
            f"{b['acc2_child']:.3f}; all-patched recovery {r['all']['recovery']:.3f} "
            f"(acc2 {r['all']['acc2']:.3f})\n")
        labs = list(next(iter(r["sizes"].values()))["masks"])
        lines.append("| size | k | " + " | ".join(labs)
                     + " | random p50 / p95 | random-nologits p95 | typed (matched to"
                     " child_nologits) p50 / p95 | logits edges removed |")
        lines.append("|" + "---|" * (len(labs) + 6))
        for frac, s in r["sizes"].items():
            cells = [f"{s['masks'][lab]['recovery']:.3f} [{s['masks'][lab]['recovery_ci95'][0]:.3f}, "
                     f"{s['masks'][lab]['recovery_ci95'][1]:.3f}] acc {s['masks'][lab]['acc2']:.2f}"
                     for lab in labs]
            rb, nb = s["random"]["recovery_band"], s["random_nologits"]["recovery_band"]
            tb = s["random_typed"]["recovery_band"]
            lines.append(f"| {float(frac) * 100:g}% | {s['k']} | " + " | ".join(cells)
                         + f" | {rb[1]:.4f} / {rb[2]:.4f} | {nb[2]:.4f} | {tb[1]:.4f} / {tb[2]:.4f}"
                         f" | {s['logits_edges_removed']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", nargs="+", metavar="JSON")
    p.add_argument("--merge", nargs="+", metavar="JSON", help="per-size shards of one route -> --out")
    p.add_argument("--route", choices=["elicit", "teach"])
    p.add_argument("--parent")
    p.add_argument("--child")
    p.add_argument("--child-scores", help="dir with the child's word-task scores.pt")
    p.add_argument("--extra-circuit", nargs="*", default=[], metavar="LABEL=DIR")
    p.add_argument("--sizes", nargs="+", type=float, default=list(SIZES))
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default=str(run.HERE / "data"))
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.summarize:
        print(summarize(a.summarize))
        return
    if a.merge:
        if a.out is None:
            p.error("--merge needs --out")
        merge(a.merge, Path(a.out))
        return
    for req in ("route", "parent", "child", "child_scores", "out"):
        if getattr(a, req) is None:
            p.error(f"--{req.replace('_', '-')} is required")

    torch.manual_seed(run.SEED)
    parent = run.load_model(a.parent, a.device)
    child = run.load_model(a.child, a.device)
    graph = run.build_graph(parent)
    assert run.build_graph(child) == graph, "parent and child architectures differ"
    assert graph.n_edges == 195_865, graph.n_edges
    val = run.load_data(Path(a.data_dir), "word")["val"]
    circuits = {"child": torch.load(Path(a.child_scores) / "scores.pt")["mean"]}
    for spec in a.extra_circuit:
        lab, d = spec.split("=", 1)
        assert lab != "child" and "nologits" not in lab, lab
        circuits[lab] = torch.load(Path(d) / "scores.pt")["mean"]
    res, per_ex = run_cmap(parent, child, graph, val, circuits, a.sizes, a.n_random,
                           a.batch_size)
    c = res["checks"]
    bad = {k: v for k, v in c.items() if not (v < CHECK_TOL)}
    res = {"route": a.route, "parent": a.parent, "child": a.child,
           "circuits": {"child": a.child_scores} | dict(s.split("=", 1) for s in a.extra_circuit),
           "checks_pass": not bad} | res
    out = Path(a.out)
    run.dump(res, out)
    np.savez_compressed(out.with_suffix(".npz"), **per_ex)
    assert not bad, f"runtime identity checks failed: {bad}"
    print(f"[cmap] wrote {out} in {res['sec_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
