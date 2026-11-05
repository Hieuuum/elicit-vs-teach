"""Build the 10 disjoint, stratified problem folds for the EAP-IG 10-fold
robustness check (HANDOFF-folds.md, Step 2).

Each fold holds ``n_disc`` discovery problems (two halves), ``n_val``
validation problems and an ``n_res`` counterfactual reservoir. No unordered
operand pair appears in two folds (clean or counterfactual), in any training
file, or in the old ``data/pairs_seed0.pt`` split. Counterfactuals for a
fold's clean rows come only from that fold's reservoir; reservoir rows are
never clean anywhere.

Usage:
    python3 build_folds.py [--seed 0] [--n-folds 10] [--n-disc 512] [--n-val 256]
                           [--n-res 768] [--out data/folds]
                           [--tokenizer meta-llama/Llama-3.2-1B]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # data._training_exclusion imports leakage_check

_spec = importlib.util.spec_from_file_location("eapig_circuit_check_data", HERE / "data.py")
data = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, data)
_spec.loader.exec_module(data)

LO, HI = 1000, 9999
ROLES = ("disc", "val", "res")
N_STRATA = 10  # (sum length 4|5) x (carries 0..4)


def count_carries(a: int, b: int) -> int:
    """Number of decimal columns of ``a + b`` that produce a carry (0-4 for
    4-digit operands; the carry out of the top column counts)."""
    carry = n = 0
    while a or b or carry:
        s = a % 10 + b % 10 + carry
        carry = 1 if s >= 10 else 0
        n += carry
        a //= 10
        b //= 10
    return n


def stratum_of(a: int, b: int) -> int:
    """Stratum id = (sum_len - 4) * 5 + n_carry, in 0..9."""
    return (len(str(a + b)) - 4) * 5 + count_carries(a, b)


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def old_split_pairs(path: str | Path) -> set[tuple[int, int]]:
    """Every unordered pair (clean and counterfactual) of a saved pairs file."""
    meta = data._load_single(Path(path))["meta"]
    out = {_key(int(a), int(b)) for a, b in zip(meta["a"], meta["b"])}
    out |= {_key(int(a), int(b)) for a, b in zip(meta["cf_a"], meta["cf_b"])}
    return out


def draw_survivors(
    seed: int,
    n_total: int,
    exclude_training: Collection[tuple[int, int]],
    exclude_old: Collection[tuple[int, int]],
    oversample: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Seeded candidates, unordered-pair deduped, exclusions dropped; the first
    ``n_total`` survivors in random order. Returns ``(a, b, counts)``."""
    rng = np.random.default_rng(seed)
    n_draw = oversample * n_total
    a_all = rng.integers(LO, HI + 1, size=n_draw)
    b_all = rng.integers(LO, HI + 1, size=n_draw)
    seen: set[tuple[int, int]] = set()
    n_dup = n_tr = n_old = 0
    keep_a: list[int] = []
    keep_b: list[int] = []
    for a, b in zip(a_all.tolist(), b_all.tolist()):
        k = _key(a, b)
        if k in seen:
            n_dup += 1
            continue
        seen.add(k)
        in_tr, in_old = k in exclude_training, k in exclude_old
        n_tr += in_tr
        n_old += in_old
        if in_tr or in_old:
            continue
        keep_a.append(a)
        keep_b.append(b)
    if len(keep_a) < n_total:
        raise ValueError(
            f"draw_survivors: only {len(keep_a)} survivors of {n_draw}, need {n_total}"
        )
    counts = {
        "n_candidates_drawn": n_draw,
        "n_duplicates": n_dup,
        "n_excluded_training": n_tr,
        "n_excluded_old_split": n_old,
    }
    return np.array(keep_a[:n_total]), np.array(keep_b[:n_total]), counts


def assign_folds(strata: np.ndarray, n_folds: int) -> np.ndarray:
    """Stable-sort by stratum, fold = position % n_folds. Per-fold stratum
    counts differ by at most 1 and every fold has len/n_folds rows."""
    n = len(strata)
    if n % n_folds:
        raise ValueError(f"assign_folds: {n} rows not divisible by {n_folds} folds")
    order = np.argsort(strata, kind="stable")
    fold = np.empty(n, dtype=np.int64)
    fold[order] = np.arange(n) % n_folds
    return fold


def role_sequence(n_disc: int, n_val: int, n_res: int) -> list[str]:
    """Length-n role pattern with exact counts, roles spread evenly (for
    2:1:3 it repeats a period-6 pattern). Role r with c slots takes the
    positions nearest (k + .5) * n / c."""
    n = n_disc + n_val + n_res
    slots = []
    for ri, (role, c) in enumerate(zip(ROLES, (n_disc, n_val, n_res))):
        slots += [((k + 0.5) * n / c, ri, role) for k in range(c)]
    slots.sort()
    seq = [s[2] for s in slots]
    assert len(seq) == n
    return seq


def deal_roles(
    strata: np.ndarray, n_disc: int, n_val: int, n_res: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roles for one fold's rows. Rows are stable-sorted by stratum and dealt
    along ``role_sequence`` (stratum-balanced, exact counts); discovery rows
    alternate half 0/1 in that order. Returns ``(role, half)``: role is an
    object array of "disc"/"val"/"res", half is 0/1 for disc and -1 otherwise.
    """
    n = len(strata)
    if n != n_disc + n_val + n_res:
        raise ValueError(f"deal_roles: {n} rows != {n_disc}+{n_val}+{n_res}")
    if n_disc % 2:
        raise ValueError(f"deal_roles: n_disc must be even, got {n_disc}")
    order = np.argsort(strata, kind="stable")
    role = np.empty(n, dtype=object)
    half = np.full(n, -1, dtype=np.int64)
    seq = role_sequence(n_disc, n_val, n_res)
    d = 0
    for pos, i in enumerate(order):
        role[i] = seq[pos]
        if seq[pos] == "disc":
            half[i] = d % 2
            d += 1
    for r, c in zip(ROLES, (n_disc, n_val, n_res)):
        assert int((role == r).sum()) == c
    assert int((half == 0).sum()) == int((half == 1).sum()) == n_disc // 2
    return role, half


def build_fold(
    a: np.ndarray,
    b: np.ndarray,
    prompt_ids: list[list[int]],
    answer_ids: list[list[int]],
    strata: np.ndarray,
    fold_id: int,
    seed: int,
    n_disc: int,
    n_val: int,
    n_res: int,
) -> dict:
    """Word-form pairs dict for one fold (same keys/dtypes as
    ``data.build_addition_pairs`` plus meta columns sum_len, n_carry, fold).
    Inputs are the fold's rows only."""
    role, half = deal_roles(strata, n_disc, n_val, n_res)
    sums = a + b
    sum_len = np.array([len(str(int(s))) for s in sums])
    is_res = role == "res"
    # saved order: disc half 0, disc half 1, val (each in stratum-sorted order)
    order = np.argsort(strata, kind="stable")
    ordered = [int(i) for i in order]
    clean_idx = np.array(
        [i for i in ordered if role[i] == "disc" and half[i] == 0]
        + [i for i in ordered if role[i] == "disc" and half[i] == 1]
        + [i for i in ordered if role[i] == "val"],
        dtype=np.int64,
    )
    cf_rng = np.random.default_rng([seed + 1, fold_id])  # separate stream per fold
    clean_l, corrupt_l, c1, c2, k1, k2, cf_idx = data._assemble_pairs(
        prompt_ids, answer_ids, clean_idx, sum_len, cf_rng, is_res
    )
    cf_idx_arr = np.array(cf_idx)
    if not is_res[cf_idx_arr].all():
        raise ValueError(f"fold {fold_id}: a counterfactual was drawn outside the fold's reservoir")
    rows = []
    for pos, (i, j) in enumerate(zip(clean_idx, cf_idx)):
        rows.append(
            {
                "a": int(a[i]),
                "b": int(b[i]),
                "sum": int(sums[i]),
                "cf_a": int(a[j]),
                "cf_b": int(b[j]),
                "cf_sum": int(sums[j]),
                "split": "discovery" if role[i] == "disc" else "validation",
                "half": int(half[i]),
                "sum_len": int(sum_len[i]),
                "n_carry": count_carries(int(a[i]), int(b[i])),
                "fold": fold_id,
            }
        )
    meta = pd.DataFrame(rows)
    return {
        "clean_ids": torch.tensor(clean_l, dtype=torch.long),
        "corrupt_ids": torch.tensor(corrupt_l, dtype=torch.long),
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "k1": torch.tensor(k1, dtype=torch.long),
        "k2": torch.tensor(k2, dtype=torch.long),
        "half": torch.tensor(meta["half"].to_numpy(), dtype=torch.long),
        "answer_text": [str(s) for s in meta["sum"]],
        "meta": meta,
        "reservoir": np.stack([a[is_res], b[is_res]], axis=1),  # not saved; for audits
    }


def build_all_folds(
    tokenizer,
    seed: int = 0,
    n_folds: int = 10,
    n_disc: int = 512,
    n_val: int = 256,
    n_res: int = 768,
    exclude_training: Collection[tuple[int, int]] = (),
    exclude_old: Collection[tuple[int, int]] = (),
) -> tuple[list[dict], dict]:
    """Return ``(word_pairs_per_fold, info)``; ``info`` holds draw counts."""
    per_fold = n_disc + n_val + n_res
    n_total = n_folds * per_fold
    a, b, counts = draw_survivors(seed, n_total, set(exclude_training), set(exclude_old))
    prompts = [data.ADDITION_PROMPT.format(a=int(x), b=int(y)) for x, y in zip(a, b)]
    answers = [str(int(x) + int(y)) for x, y in zip(a, b)]
    fulls = [p + s for p, s in zip(prompts, answers)]
    prompt_ids, answer_ids = data._validate_pool_tokenization(
        tokenizer, prompts, answers, fulls, where="build_folds"
    )
    strata = np.array([stratum_of(int(x), int(y)) for x, y in zip(a, b)])
    fold = assign_folds(strata, n_folds)
    out = []
    for f in range(n_folds):
        idx = np.flatnonzero(fold == f)  # ascending = random draw order
        assert len(idx) == per_fold
        out.append(
            build_fold(
                a[idx],
                b[idx],
                [prompt_ids[i] for i in idx],
                [answer_ids[i] for i in idx],
                strata[idx],
                f,
                seed,
                n_disc,
                n_val,
                n_res,
            )
        )
    return out, counts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_folds(word_folds: list[dict], tokenizer, out_dir: Path, copy_src: Path | None) -> dict:
    """Write fold_NN/ dirs and return ``{relative path: sha256}``."""
    hashes: dict[str, str] = {}
    for f, wp in enumerate(word_folds):
        d = out_dir / f"fold_{f:02d}"
        d.mkdir(parents=True, exist_ok=True)
        word = {k: v for k, v in wp.items() if k != "reservoir"}
        data.save_pairs(word, d / data.ADDITION_PAIRS_FILENAME)
        data.save_pairs(data.build_symbol_pairs(word, tokenizer), d / data.SYMBOL_PAIRS_FILENAME)
        if copy_src is not None:
            shutil.copyfile(copy_src, d / data.COPY_PAIRS_FILENAME)
        for p in sorted(d.iterdir()):
            hashes[f"{d.name}/{p.name}"] = _sha256(p)
    return hashes


def summarize(
    word_folds: list[dict], seed: int, n_disc: int, n_val: int, n_res: int, counts: dict
) -> dict:
    """Manifest body (without file hashes)."""
    per_fold = []
    for f, wp in enumerate(word_folds):
        m = wp["meta"]
        res = wp["reservoir"]
        roles = {"disc": m[m["split"] == "discovery"], "val": m[m["split"] == "validation"]}
        strata = {
            r: np.bincount(
                [stratum_of(int(x), int(y)) for x, y in zip(df["a"], df["b"])], minlength=N_STRATA
            ).tolist()
            for r, df in roles.items()
        }
        strata["res"] = np.bincount(
            [stratum_of(int(x), int(y)) for x, y in res], minlength=N_STRATA
        ).tolist()
        per_fold.append(
            {
                "fold": f,
                "n_disc": len(roles["disc"]),
                "n_val": len(roles["val"]),
                "n_res": len(res),
                "n_total": len(roles["disc"]) + len(roles["val"]) + len(res),
                "strata_counts": strata,
            }
        )
    all_a = np.concatenate(
        [np.r_[w["meta"]["a"].to_numpy(), w["reservoir"][:, 0]] for w in word_folds]
    )
    all_b = np.concatenate(
        [np.r_[w["meta"]["b"].to_numpy(), w["reservoir"][:, 1]] for w in word_folds]
    )
    return {
        "seed": seed,
        "n_folds": len(word_folds),
        "n_disc": n_disc,
        "n_val": n_val,
        "n_res": n_res,
        **counts,
        "strata_order": "index = (sum_len - 4) * 5 + n_carry",
        "five_digit_share": float(((all_a + all_b) >= 10000).mean()),
        "mean_a": float(all_a.mean()),
        "mean_b": float(all_b.mean()),
        "folds": per_fold,
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--n-disc", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--n-res", type=int, default=768)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "folds")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train_ex = data._training_exclusion()
    old_ex = old_split_pairs(HERE / "data" / data.ADDITION_PAIRS_FILENAME)
    print(f"exclusions: {len(train_ex)} training pairs, {len(old_ex)} old-split pairs", flush=True)
    folds, counts = build_all_folds(
        tok, args.seed, args.n_folds, args.n_disc, args.n_val, args.n_res, train_ex, old_ex
    )
    hashes = write_folds(folds, tok, args.out, HERE / "data" / data.COPY_PAIRS_FILENAME)
    manifest = summarize(folds, args.seed, args.n_disc, args.n_val, args.n_res, counts)
    manifest["n_training_exclusion_pairs"] = len(train_ex)
    manifest["n_old_split_pairs"] = len(old_ex)
    manifest["sha256"] = hashes
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("folds", "sha256")}, indent=1))


if __name__ == "__main__":
    main()
