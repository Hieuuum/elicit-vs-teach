"""CPU overlap analysis for the 10-fold robustness check (HANDOFF-folds.md, Step 3).

Per fold and circuit size: route Jaccards (elicit = ep_sym->ec, teach = fp->tc,
base_ec, base_tc, secondary elicit_word = ep_word->ec), in-fold half-split
ceilings, plus cross-fold ceilings (fold i vs fold j full-mean circuit, all
i<j pairs) and chance (analytic + empirical random k-set pairs).

Top-k convention = `et.topk_edges` (descending signed score, ties by ascending
index). Each score vector is ranked once (stable argsort of the negated
float64 scores); Jaccard at every size then comes from one bincount/cumsum of
max(rank_a, rank_b), since |top-k(a) & top-k(b)| = #{e : max(rank) < k}.

    python3 fold_overlap.py --results-root results/2026-11-04_folds --folds 0 1 2 3 4 5 6 7 8 9 \
        --out results/2026-11-04_folds/overlap.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from geode.circuits import edge_tests as et  # noqa: E402

FRACS = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)

# label -> (task dir, model tag) under fold_{k:02d}/
LABELS: dict[str, tuple[str, str]] = {
    "ep_sym": ("klsym", "elicit_parent"),
    "ep_word": ("klword", "elicit_parent"),
    "ec": ("klword", "elicit_child"),
    "fp": ("klword", "fmt_parent"),
    "tc": ("klword", "teach_child"),
    "base": ("klword", "base"),
}
ROUTES = {
    "elicit": ("ep_sym", "ec"),
    "teach": ("fp", "tc"),
    "base_ec": ("base", "ec"),
    "base_tc": ("base", "tc"),
    "elicit_word": ("ep_word", "ec"),
}


def fold_dir(root: Path, fold: int) -> Path:
    return root / f"fold_{fold:02d}"


def rank_of(scores: np.ndarray) -> np.ndarray:
    """rank[e] = position of edge e in the `et.topk_edges` order (0 = best)."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    rank = np.empty(s.size, dtype=np.int64)
    rank[order] = np.arange(s.size)
    return rank


def jaccard_curve(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """inter[k-1] = |top-k(a) & top-k(b)| for every k = 1..E."""
    m = np.maximum(ra, rb)
    return np.cumsum(np.bincount(m, minlength=m.size))


def jaccards_at(ra: np.ndarray, rb: np.ndarray, ks: list[int]) -> list[float]:
    inter = jaccard_curve(ra, rb)
    out = []
    for k in ks:
        i = int(inter[k - 1])
        out.append(i / (2 * k - i))
    return out


def _load(root: Path, fold: int, label: str) -> dict[str, np.ndarray] | None:
    task, tag = LABELS[label]
    path = fold_dir(root, fold) / task / tag / "scores.pt"
    if not path.exists():
        warnings.warn(f"missing {path}", stacklevel=2)
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {k: np.asarray(d[k], dtype=np.float32) for k in ("mean", "mean_a", "mean_b")}


def _null_chunk(args):
    n_edges, ks, seeds = args
    ar = np.arange(n_edges)
    ks_arr = np.asarray(ks)
    out = np.empty((len(seeds), len(ks)))
    for i, s in enumerate(seeds):
        # A = fixed prefix, B = uniformly random ranking: same law as two independent random k-sets.
        rb = np.random.default_rng(s).permutation(n_edges)
        inter = np.cumsum(np.bincount(np.maximum(ar, rb), minlength=n_edges))[ks_arr - 1]
        out[i] = inter / (2 * ks_arr - inter)
    return out


def null_jaccards(n_edges: int, ks: list[int], n_null: int, seed: int, workers: int) -> np.ndarray:
    """(n_null, len(ks)) Jaccards of independent random k-set pairs."""
    seeds = np.random.SeedSequence(seed).spawn(n_null)
    chunks = [seeds[i :: max(workers, 1)] for i in range(max(workers, 1))]
    chunks = [c for c in chunks if c]
    if workers <= 1:
        return (
            np.concatenate([_null_chunk((n_edges, ks, c)) for c in chunks])
            if chunks
            else np.empty((0, len(ks)))
        )
    with ThreadPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(_null_chunk, [(n_edges, ks, c) for c in chunks])))


def _stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "sd": None, "min": None, "max": None, "values": []}
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "values": [float(v) for v in a],
    }


def run(
    root: Path,
    folds: list[int],
    fracs: tuple[float, ...] = FRACS,
    n_null: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> dict:
    ranks: dict[tuple[int, str, str], np.ndarray] = {}
    n_edges = None
    for f in folds:
        for lab in LABELS:
            d = _load(root, f, lab)
            if d is None:
                continue
            for part in ("mean", "mean_a", "mean_b"):
                ranks[(f, lab, part)] = rank_of(d[part])
                n_edges = d[part].size if n_edges is None else n_edges
                if d[part].size != n_edges:
                    raise ValueError(f"edge count mismatch at fold {f} {lab} {part}")
    if n_edges is None:
        raise FileNotFoundError(f"no scores.pt found under {root}")
    ks = [et.size_to_k(fr, n_edges) for fr in fracs]
    key = [str(fr) for fr in fracs]

    per_fold: dict = {}
    for f in folds:
        pf = {kf: {"routes": {}, "ceiling_half": {}} for kf in key}
        for rname, (la, lb) in ROUTES.items():
            ra, rb = ranks.get((f, la, "mean")), ranks.get((f, lb, "mean"))
            vals = (
                jaccards_at(ra, rb, ks) if ra is not None and rb is not None else [None] * len(ks)
            )
            for kf, v in zip(key, vals):
                pf[kf]["routes"][rname] = v
        for lab in LABELS:
            ra, rb = ranks.get((f, lab, "mean_a")), ranks.get((f, lab, "mean_b"))
            vals = (
                jaccards_at(ra, rb, ks) if ra is not None and rb is not None else [None] * len(ks)
            )
            for kf, v in zip(key, vals):
                pf[kf]["ceiling_half"][lab] = v
        per_fold[str(f)] = pf

    cross: dict = {}
    for lab in LABELS:
        cross[lab] = {kf: [] for kf in key}
        for i, j in itertools.combinations(folds, 2):
            ri, rj = ranks.get((i, lab, "mean")), ranks.get((j, lab, "mean"))
            if ri is None or rj is None:
                continue
            for kf, v in zip(key, jaccards_at(ri, rj, ks)):
                cross[lab][kf].append(v)
        cross[lab] = {kf: _stats(v) for kf, v in cross[lab].items()}

    null = null_jaccards(n_edges, ks, n_null, seed, workers)
    chance = {
        kf: {
            "analytic": et.chance_jaccard(k, k, n_edges),
            "null_mean": float(null[:, c].mean()) if n_null else None,
            "null_p95": float(np.quantile(null[:, c], 0.95)) if n_null else None,
        }
        for c, (kf, k) in enumerate(zip(key, ks))
    }
    return {
        "fracs": list(fracs),
        "ks": ks,
        "E": int(n_edges),
        "folds": list(folds),
        "n_null": n_null,
        "seed": seed,
        "per_fold": per_fold,
        "cross_fold": cross,
        "chance": chance,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--fracs", nargs="+", type=float, default=list(FRACS))
    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    res = run(a.results_root, a.folds, tuple(a.fracs), a.n_null, a.seed, a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
