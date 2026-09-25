"""Cutoff sensitivity and continuous attribution-profile comparisons on saved scores.

These describe attribution maps, not circuit identity or causal faithfulness.
All resampling units are source groups. Stage comparisons must be input-aligned.
"""

import numpy as np

from geode.circuits.statistics import _finite_array, _groups, node_type


def profile_metrics(a, b, names, ks=(8, 16, 32, 64, 128)):
    """Batch metrics on [..., nodes]; zero-mass continuous profiles are undefined.

    Weighted overlap is sum(min(p,q))/sum(max(p,q)), with p=abs(a)/sum(abs(a)).
    Signed cosine retains effect direction. Neither requires a top-k cutoff.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] != len(names):
        raise ValueError("Profiles must have matching shapes and named columns")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Profiles must be finite")
    if len(set(names)) != len(names) or not names:
        raise ValueError("Require unique node names")
    if any(not isinstance(k, int) or not 1 <= k <= len(names) for k in ks):
        raise ValueError("Invalid cutoff")
    aa, bb = abs(a), abs(b)
    sa, sb = aa.sum(-1, keepdims=True), bb.sum(-1, keepdims=True)
    p = np.divide(aa, sa, out=np.zeros_like(aa), where=sa != 0)
    q = np.divide(bb, sb, out=np.zeros_like(bb), where=sb != 0)
    valid = (sa[..., 0] > 0) & (sb[..., 0] > 0)
    union = np.maximum(p, q).sum(-1)
    overlap = np.divide(
        np.minimum(p, q).sum(-1), union, out=np.full_like(union, np.nan), where=valid
    )
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.divide((a * b).sum(-1), norm, out=np.full_like(norm, np.nan), where=norm > 0)
    result = {"weighted_overlap": overlap, "signed_cosine": np.clip(cosine, -1, 1)}
    # First lexical order, then stable score sort: match topk_nodes tie policy.
    lexical = np.argsort(names)
    ranks_a = np.argsort(np.argsort(-aa[..., lexical], axis=-1, kind="stable"), axis=-1)
    ranks_b = np.argsort(np.argsort(-bb[..., lexical], axis=-1, kind="stable"), axis=-1)
    for k in ks:
        intersection = ((ranks_a < k) & (ranks_b < k)).sum(-1)
        result[f"jaccard@{k}"] = intersection / (2 * k - intersection)
    return result


def summarize(draws, interval_name):
    """Keep undefined draws explicit; intervals exclude only undefined values."""
    values = np.asarray(draws).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()) if len(finite) else None,
        interval_name: np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
        "defined_draws": len(finite),
        "total_draws": len(values),
    }


def grouped_profiles(arrays, groups, magnitude=False):
    """Return [group, stage, node]; magnitude sensitivity takes abs BEFORE averaging."""
    checked = [_finite_array(a, 2) for a in arrays]
    if any(a.shape != checked[0].shape for a in checked):
        raise ValueError("Stages must have identical score shapes")
    values = np.stack(checked, axis=1)
    means, _ = _groups(abs(values) if magnitude else values, groups, None)
    return means


def analyze_profiles(means, names, *, n_bootstrap=1000, repeats=500, seed=0):
    """Paired group bootstrap and identical disjoint splits across all stages.

    Cross-disjoint compares A(left) with B(right), averaged with A(right)/B(left),
    at the same sample sizes as within-stage stability. Its distribution over
    repeated splits is descriptive, NOT a CI or a reliability ceiling.
    """
    means = _finite_array(means, 3)
    n, stages, nodes = means.shape
    if n < 4 or nodes != len(names) or stages < 2:
        raise ValueError("Require >=4 groups, >=2 stages and matching node names")
    if n_bootstrap < 100 or repeats < 2:
        raise ValueError("Require >=100 bootstrap draws and >=2 splits")
    rng = np.random.default_rng(seed)
    point = means.mean(0)
    boot = np.stack([means[rng.integers(n, size=n)].mean(0) for _ in range(n_bootstrap)])
    splits = [rng.permutation(n) for _ in range(repeats)]
    left = np.stack([means[idx[: n // 2]].mean(0) for idx in splits])
    right = np.stack([means[idx[n // 2 :]].mean(0) for idx in splits])
    output = {"n_groups": n, "scopes": {}}
    for scope in ("all", "attention_head", "mlp"):
        idx = np.array(
            [i for i, name in enumerate(names) if scope == "all" or node_type(name) == scope]
        )
        if not len(idx):
            continue
        labels = [names[i] for i in idx]
        ks = tuple(k for k in (2, 4, 8, 16, 32, 64, 128) if k < len(idx))
        p, bs, lh, rh = (v[..., idx] for v in (point, boot, left, right))
        within = profile_metrics(lh, rh, labels, ks)
        observed = profile_metrics(p[:-1], p[1:], labels, ks)
        paired = profile_metrics(bs[:, :-1], bs[:, 1:], labels, ks)
        cross1 = profile_metrics(lh[:, :-1], rh[:, 1:], labels, ks)
        cross2 = profile_metrics(rh[:, :-1], lh[:, 1:], labels, ks)
        # Conditional reference preserves each stage's scores and type profile.
        # MLP-only/head-only scopes use a uniform identity permutation.
        permutations = []
        for _ in range(n_bootstrap):
            perm = np.arange(len(idx))
            for kind in sorted({node_type(name) for name in labels}):
                positions = [i for i, name in enumerate(labels) if node_type(name) == kind]
                perm[positions] = rng.permutation(positions)
            permutations.append(perm)
        permuted = np.stack([p[1:, perm] for perm in permutations])
        null = profile_metrics(np.broadcast_to(p[:-1], permuted.shape), permuted, labels, ks)
        result = {"n_nodes": len(idx), "ks": ks, "within": [], "transitions": []}
        for stage in range(stages):
            result["within"].append(
                {key: summarize(value[:, stage], "split_range_95") for key, value in within.items()}
            )
        for stage in range(stages - 1):
            row = {}
            for key, values in observed.items():
                cross = (cross1[key][:, stage] + cross2[key][:, stage]) / 2
                reference = (within[key][:, stage] + within[key][:, stage + 1]) / 2
                value = float(values[stage])
                row[key] = {
                    "point": value if np.isfinite(value) else None,
                    "bootstrap": summarize(paired[key][:, stage], "percentile_ci_95"),
                    "cross_disjoint": summarize(cross, "split_range_95"),
                    "within_minus_cross": summarize(reference - cross, "split_range_95"),
                    "type_permutation_null": summarize(null[key][:, stage], "reference_range_95"),
                }
            result["transitions"].append(row)
        output["scopes"][scope] = result
    # Membership frequencies are conditional bootstrap stability, not posterior
    # probabilities that a node belongs to a true circuit.
    lexical = np.argsort(names)
    ranks = np.argsort(np.argsort(-abs(boot[..., lexical]), axis=-1, kind="stable"), axis=-1)
    frequency = (ranks < min(16, nodes)).mean(0)
    output["top16_selection_frequency"] = [
        {names[i]: float(frequency[s, j]) for j, i in enumerate(lexical)} for s in range(stages)
    ]
    output["mean_signed_profiles"] = point.tolist()
    return output
