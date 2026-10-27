"""Signed edge-Jaccard overlap between EAP-IG circuits, with a bootstrap-over-prompts 95% interval.

For each model, `scores.pt` (written by run.py's score stage) holds `mean`, `mean_a`, `mean_b`
(fp32, (E,)) and `per_example` (fp16, (N, E)). A circuit at size `frac` is the top-k edges by
SIGNED score, k = size_to_k(frac, E). The point estimate is the Jaccard of `et.topk_edges` on the
fp32 `mean` (the same ranking edge_node.py uses). The interval resamples the N discovery prompts
with replacement, ONE index draw shared by every model, re-averages `per_example`, and recomputes
the Jaccard. "About equal" then means overlapping intervals.

    python3 overlap_boot.py --scores a=DIR_A b=DIR_B --pairs a:b --fracs 0.01 0.05 \
        --n-boot 200 --seed 0 --data-dir data --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geode.circuits import edge_tests as et

# fp16 stores ~11 significand bits: relative error <= 2**-11 ~ 4.9e-4 per element, and a mean of
# such elements is off by at most that fraction of mean|x|. The check tolerance is 1e-3 of the
# largest mean|per_example| (2x margin), so only a real mismatch (permuted rows, other prompts)
# trips it.
FP16_ATOL_FRAC = 1e-3
EDGE_CHUNK = 20_000


def fast_topk_order(scores: np.ndarray, k: int) -> np.ndarray:
    """Exactly `et.topk_edges(scores, k)` (descending, ties by ascending index), in O(E + k log k)."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if s.ndim != 1 or n == 0 or not 1 <= k <= n:
        raise ValueError("scores must be nonempty 1-D and 1 <= k <= len(scores)")
    thr = s[np.argpartition(s, n - k)[n - k]]  # k-th largest value
    gt = np.flatnonzero(s > thr)
    eq = np.flatnonzero(s == thr)[: k - gt.size]  # lowest-index ties fill the rest
    sel = np.concatenate([gt, eq])
    return sel[np.lexsort((sel, -s[sel]))].astype(np.int64)


def load_scores(dirs: dict[str, Path]) -> dict[str, dict[str, torch.Tensor]]:
    """scores.pt per label (per_example stays fp16; mmap when the file allows it)."""
    out = {}
    for label, d in dirs.items():
        p = Path(d) / "scores.pt"
        try:
            s = torch.load(p, map_location="cpu", mmap=True)
        except (RuntimeError, ValueError):
            s = torch.load(p, map_location="cpu")
        out[label] = {k: s[k] for k in ("mean", "mean_a", "mean_b", "per_example")}
    return out


def full_means(
    pe: torch.Tensor, half: np.ndarray, chunk: int = EDGE_CHUNK
) -> dict[str, np.ndarray]:
    """Means of fp16 per_example (all rows / half 0 / half 1, in fp32) and max_e mean|x|, chunked."""
    n, e = pe.shape
    h0 = torch.as_tensor(half == 0)
    h1 = torch.as_tensor(half == 1)
    full, a, b = (np.empty(e, np.float32) for _ in range(3))
    scale = 0.0
    for c0 in range(0, e, chunk):
        x = pe[:, c0 : c0 + chunk].float()
        full[c0 : c0 + chunk] = x.mean(0).numpy()
        a[c0 : c0 + chunk] = x[h0].mean(0).numpy()
        b[c0 : c0 + chunk] = x[h1].mean(0).numpy()
        scale = max(scale, float(x.abs().mean(0).max()))
    return {"full": full, "a": a, "b": b, "scale": scale}


def check_same_prompts(label: str, s: dict, half: np.ndarray) -> np.ndarray:
    """Assert per_example is consistent with mean / mean_a / mean_b under `half`.

    Returns the fp16-full mean. Raises AssertionError naming `label` on any mismatch, which is
    what happens if the model was scored on other prompts or in another row order.
    """
    pe = s["per_example"]
    if pe.shape[0] != len(half):
        raise AssertionError(f"{label}: per_example has {pe.shape[0]} rows, half has {len(half)}")
    m = full_means(pe, np.asarray(half))
    atol = FP16_ATOL_FRAC * m["scale"]
    for name, got, ref in (
        ("mean", m["full"], s["mean"]),
        ("mean_a", m["a"], s["mean_a"]),
        ("mean_b", m["b"], s["mean_b"]),
    ):
        err = float(np.abs(got - ref.float().numpy()).max())
        if not err <= atol:
            raise AssertionError(
                f"{label}: per_example does not reproduce {name} under `half` "
                f"(max abs err {err:.3g} > atol {atol:.3g}); different prompts or row order?"
            )
    return m["full"]


def draw_counts(n_boot: int, n: int, seed: int) -> np.ndarray:
    """(n_boot, n) float32 bootstrap weights (multiplicity / n); rows sum to 1. One draw per run."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n), dtype=np.float32)
    np.add.at(counts, (np.arange(n_boot)[:, None], idx), 1.0)
    return counts / n


def resampled_means(counts: np.ndarray, pe: torch.Tensor, chunk: int = EDGE_CHUNK) -> np.ndarray:
    """(n_boot, E) fp32 = counts @ per_example, converting fp16 to fp32 one edge chunk at a time."""
    c = torch.as_tensor(counts)
    out = np.empty((c.shape[0], pe.shape[1]), dtype=np.float32)
    for c0 in range(0, pe.shape[1], chunk):
        out[:, c0 : c0 + chunk] = (c @ pe[:, c0 : c0 + chunk].float()).numpy()
    return out


def bootstrap_orders(
    counts: np.ndarray, pe: torch.Tensor, kmax: int, chunk: int = EDGE_CHUNK
) -> np.ndarray:
    """(n_boot, kmax) best-first top-kmax edge ids of every resampled mean; top-k is a prefix."""
    means = resampled_means(counts, pe, chunk)
    orders = np.stack([fast_topk_order(row, kmax) for row in means]).astype(np.int32)
    del means
    return orders


def _jaccard_prefix(oa: np.ndarray, ob: np.ndarray, k: int) -> float:
    ov = np.intersect1d(oa[:k], ob[:k]).size
    return ov / (2 * k - ov)


def bootstrap_overlap(
    scores: dict[str, dict],
    half: np.ndarray,
    pairs: list[tuple[str, str]],
    fracs: list[float],
    n_boot: int,
    seed: int,
    chunk: int = EDGE_CHUNK,
) -> dict:
    """Point estimates and bootstrap intervals for every (pair, frac); see module docstring."""
    labels = sorted({x for p in pairs for x in p})
    shapes = {tuple(scores[lab]["per_example"].shape) for lab in labels}
    if len(shapes) != 1:
        raise AssertionError(f"per_example shapes differ across models: {sorted(shapes)}")
    n, e = shapes.pop()
    full16 = {lab: check_same_prompts(lab, scores[lab], half) for lab in labels}
    ks = {f: et.size_to_k(f, e) for f in fracs}
    kmax = max(ks.values())

    counts = draw_counts(n_boot, n, seed)  # one draw, shared by every model
    orders = {}
    # per model: its orders are built once, then all pairs reuse them
    for lab in labels:
        orders[lab] = bootstrap_orders(counts, scores[lab]["per_example"], kmax, chunk)

    out_pairs = {}
    for a, b in pairs:
        ma = scores[a]["mean"].double().numpy()
        mb = scores[b]["mean"].double().numpy()
        res = {}
        for f in fracs:
            k = ks[f]
            point = et.jaccard(et.topk_edges(ma, k), et.topk_edges(mb, k))
            p16 = et.jaccard(
                et.topk_edges(full16[a].astype(np.float64), k),
                et.topk_edges(full16[b].astype(np.float64), k),
            )
            boot = np.array([_jaccard_prefix(orders[a][i], orders[b][i], k) for i in range(n_boot)])
            q_lo, q_hi = np.percentile(boot, [2.5, 97.5])
            # Primary interval = basic (reflected) bootstrap. Resampled means are noisier than
            # the full mean, so resampled top-k overlap sits below the point estimate, more so
            # for noisier models; the reflection removes that first-order bias, so pairs with
            # different noise levels compare fairly. Percentile interval kept as pct_lo/pct_hi.
            lo, hi = max(0.0, 2 * point - q_hi), min(1.0, 2 * point - q_lo)
            res[str(f)] = {
                "k": k,
                "n_boot": n_boot,
                "point": float(point),
                "point_fp16_full": float(p16),
                "boot_mean": float(boot.mean()),
                "boot_median": float(np.median(boot)),
                "bias": float(boot.mean() - point),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "pct_lo": float(q_lo),
                "pct_hi": float(q_hi),
                "point_in_ci": bool(lo <= point <= hi),
            }
        out_pairs[f"{a}:{b}"] = res
    return {
        "meta": {
            "seed": seed,
            "n_boot": n_boot,
            "N": n,
            "E": e,
            "fracs": list(fracs),
            "labels": labels,
            "fp16_atol_frac": FP16_ATOL_FRAC,
            "fp16_atol_note": "atol = fp16_atol_frac * max_e mean|per_example| (per model)",
            "edge_chunk": chunk,
        },
        "pairs": out_pairs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", nargs="+", required=True, metavar="LABEL=DIR")
    ap.add_argument("--pairs", nargs="+", required=True, metavar="A:B")
    ap.add_argument("--fracs", nargs="+", type=float, required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dirs = {}
    for item in args.scores:
        label, sep, d = item.partition("=")
        if not sep or not label or not d:
            ap.error(f"--scores entry {item!r} must be LABEL=DIR")
        dirs[label] = Path(d)
    pairs = []
    for item in args.pairs:
        a, sep, b = item.partition(":")
        if not sep or not a or not b:
            ap.error(f"--pairs entry {item!r} must be A:B")
        for x in (a, b):
            if x not in dirs:
                ap.error(f"pair {item!r}: no --scores entry for label {x!r}")
        pairs.append((a, b))
    args.dirs, args.pair_list = dirs, pairs
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from data import load_pairs

    half = np.asarray(load_pairs(args.data_dir, task="word")["disc"]["half"])
    needed = {x for p in args.pair_list for x in p}
    scores = load_scores({k: v for k, v in args.dirs.items() if k in needed})
    res = bootstrap_overlap(scores, half, args.pair_list, args.fracs, args.n_boot, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap_boot.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'overlap_boot.json'}")


if __name__ == "__main__":
    main()
