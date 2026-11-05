"""``experiments/eapig-circuit-check/build_folds.py`` — the 10-fold problem
builder. Silent failure here (a pair shared between folds, a counterfactual
drawn from outside its fold's reservoir, unbalanced strata, word/symbol rows
out of alignment) would make the folds non-independent replicates with
nothing crashing. Small instances are built in-process with the real local
Llama-3.2-1B tokenizer (skipped if not cached) and small injected exclusion
sets; an optional check runs on the real build if present.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np
import pytest
import torch

from tests._scriptloader import repo_root

os.environ.setdefault("HF_HUB_OFFLINE", "1")

EXP = repo_root() / "experiments" / "eapig-circuit-check"
_spec = importlib.util.spec_from_file_location("eapig_build_folds", EXP / "build_folds.py")
bf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bf
_spec.loader.exec_module(bf)
data = bf.data

N_FOLDS, N_DISC, N_VAL, N_RES = 3, 16, 8, 24
PER_FOLD = N_DISC + N_VAL + N_RES


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception as e:  # not cached / offline
        pytest.skip(f"Llama-3.2-1B tokenizer not available locally: {e}")


def _key(a, b):
    return (int(a), int(b)) if a <= b else (int(b), int(a))


def _pairs(fold):
    """(clean keys, cf keys, reservoir keys) of a word-pairs dict."""
    m = fold["meta"]
    clean = [_key(a, b) for a, b in zip(m["a"], m["b"])]
    cf = [_key(a, b) for a, b in zip(m["cf_a"], m["cf_b"])]
    res = [_key(a, b) for a, b in fold["reservoir"]]
    return clean, cf, res


def _build(tok, seed=0, ex_train=(), ex_old=()):
    return bf.build_all_folds(tok, seed, N_FOLDS, N_DISC, N_VAL, N_RES, ex_train, ex_old)


@pytest.fixture(scope="module")
def small(tok):
    # inject exclusions drawn from what the unexcluded build would produce
    base, _ = _build(tok)
    ex_train = set(_pairs(base[0])[0][:5] + _pairs(base[1])[2][:5])
    ex_old = set(_pairs(base[2])[0][:3] + _pairs(base[0])[2][:3])
    folds, counts = _build(tok, 0, ex_train, ex_old)
    return folds, counts, ex_train, ex_old


# ---------------------------------------------------------------- pure functions


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1000, 1000, 0),
        (1234, 4321, 0),
        (5000, 5000, 1),
        (9999, 9999, 4),
        (1999, 1, 3),  # 9+1 carry, 9+0+1, 9+0+1, 1+0+1=2 no -> carries in 3 columns
        (9999, 1, 4),
        (5555, 5555, 4),
        (4999, 5001, 4),
        (1500, 1500, 1),
        (1005, 1005, 1),
    ],
)
def test_count_carries(a, b, expected):
    assert bf.count_carries(a, b) == expected


def test_count_carries_matches_digit_sum_identity():
    # carries = (digitsum(a) + digitsum(b) - digitsum(a+b)) / 9
    rng = np.random.default_rng(0)
    ds = lambda n: sum(map(int, str(n)))  # noqa: E731
    for a, b in rng.integers(1000, 10000, size=(500, 2)).tolist():
        assert bf.count_carries(a, b) * 9 == ds(a) + ds(b) - ds(a + b)


def test_count_carries_symmetric():
    rng = np.random.default_rng(1)
    for a, b in rng.integers(1000, 10000, size=(200, 2)).tolist():
        assert bf.count_carries(a, b) == bf.count_carries(b, a)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1000, 1000, 0),
        (9999, 9999, 9),
        (5000, 5000, 6),
        (4999, 5000, 0),
        (9000, 1000, 6),
        (1999, 1, 3),
    ],
)
def test_stratum_of_edges(a, b, expected):
    assert bf.stratum_of(a, b) == expected


def test_stratum_range_and_impossible_cells():
    rng = np.random.default_rng(2)
    seen = {bf.stratum_of(a, b) for a, b in rng.integers(1000, 10000, size=(20000, 2)).tolist()}
    assert seen <= set(range(10))
    assert 4 not in seen  # 4-digit sum cannot have 4 carries
    assert 5 not in seen  # 5-digit sum has >= 1 carry


def test_key_is_unordered():
    assert bf._key(5, 3) == bf._key(3, 5) == (3, 5)


def test_assign_folds_balance_and_sizes():
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 10, size=300)
    fold = bf.assign_folds(strata, 10)
    assert set(np.bincount(fold)) == {30}
    for s in range(10):
        c = np.bincount(fold[strata == s], minlength=10)
        assert c.max() - c.min() <= 1


def test_assign_folds_stable_within_stratum():
    strata = np.array([1, 0, 1, 0, 1, 0])
    fold = bf.assign_folds(strata, 3)
    # stratum 0 rows at idx 1,3,5 -> positions 0,1,2 -> folds 0,1,2 in order
    assert fold[[1, 3, 5]].tolist() == [0, 1, 2]
    assert fold[[0, 2, 4]].tolist() == [0, 1, 2]


def test_assign_folds_rejects_indivisible():
    with pytest.raises(ValueError):
        bf.assign_folds(np.zeros(7, dtype=int), 3)


@pytest.mark.parametrize("sizes", [(512, 256, 768), (16, 8, 24), (4, 2, 6), (6, 5, 7), (2, 1, 1)])
def test_role_sequence_exact_counts(sizes):
    seq = bf.role_sequence(*sizes)
    assert len(seq) == sum(sizes)
    assert [seq.count(r) for r in bf.ROLES] == list(sizes)


def test_role_sequence_periodic_for_2_1_3():
    seq = bf.role_sequence(512, 256, 768)
    p = seq[:6]
    assert p.count("disc") == 2 and p.count("val") == 1 and p.count("res") == 3
    assert seq == p * 256


def test_deal_roles_counts_halves_and_stratum_balance():
    rng = np.random.default_rng(3)
    strata = rng.integers(0, 10, size=PER_FOLD * 4)
    n_d, n_v, n_r = N_DISC * 4, N_VAL * 4, N_RES * 4
    role, half = bf.deal_roles(strata, n_d, n_v, n_r)
    assert Counter(role.tolist()) == {"disc": n_d, "val": n_v, "res": n_r}
    assert (half == 0).sum() == (half == 1).sum() == n_d // 2
    assert (half[role != "disc"] == -1).all()
    # per stratum, each role's count is within 2 of its proportional share
    for s in np.unique(strata):
        m = strata == s
        for r, tot in zip(bf.ROLES, (n_d, n_v, n_r)):
            exp = m.sum() * tot / len(strata)
            assert abs((role[m] == r).sum() - exp) <= 2


def test_deal_roles_half_balanced_per_stratum():
    strata = np.repeat(np.arange(4), 12)
    role, half = bf.deal_roles(strata, 16, 8, 24)
    for s in range(4):
        m = (strata == s) & (role == "disc")
        assert abs((half[m] == 0).sum() - (half[m] == 1).sum()) <= 1


def test_deal_roles_validates():
    with pytest.raises(ValueError):
        bf.deal_roles(np.zeros(10, dtype=int), 4, 2, 3)  # wrong total
    with pytest.raises(ValueError):
        bf.deal_roles(np.zeros(9, dtype=int), 3, 2, 4)  # odd n_disc


def test_draw_survivors_dedupes_and_excludes():
    a, b, c = bf.draw_survivors(0, 200, set(), set())
    keys = [bf._key(x, y) for x, y in zip(a, b)]
    assert len(set(keys)) == 200
    ex_tr, ex_old = set(keys[:10]), set(keys[10:20])
    a2, b2, c2 = bf.draw_survivors(0, 200, ex_tr, ex_old)
    keys2 = {bf._key(x, y) for x, y in zip(a2, b2)}
    assert not keys2 & (ex_tr | ex_old)
    assert c2["n_excluded_training"] >= 10 and c2["n_excluded_old_split"] >= 10
    assert ((a2 >= 1000) & (a2 <= 9999) & (b2 >= 1000) & (b2 <= 9999)).all()


def test_draw_survivors_swapped_orientation_excluded():
    a, b, _ = bf.draw_survivors(0, 50, set(), set())
    k = bf._key(a[0], b[0])  # exclusion is by unordered pair, whatever the drawn orientation
    a2, b2, _ = bf.draw_survivors(0, 50, {k}, set())
    assert k not in {bf._key(x, y) for x, y in zip(a2, b2)}


def test_draw_survivors_raises_when_short():
    with pytest.raises(ValueError):
        bf.draw_survivors(0, 100, set(), set(), oversample=0)


# ---------------------------------------------------------------- small builds


def test_exact_role_counts_and_sizes(small):
    folds, *_ = small
    assert len(folds) == N_FOLDS
    for f in folds:
        m = f["meta"]
        assert len(m) == N_DISC + N_VAL
        assert (m["split"] == "discovery").sum() == N_DISC
        assert (m["split"] == "validation").sum() == N_VAL
        assert len(f["reservoir"]) == N_RES
        assert (m["half"] == 0).sum() == (m["half"] == 1).sum() == N_DISC // 2
        assert (m["half"] == -1).sum() == N_VAL


def test_row_order_disc0_disc1_val(small):
    for f in small[0]:
        h = f["meta"]["half"].tolist()
        assert h == [0] * (N_DISC // 2) + [1] * (N_DISC // 2) + [-1] * N_VAL
        assert f["meta"]["split"].tolist() == ["discovery"] * N_DISC + ["validation"] * N_VAL
        assert torch.equal(f["half"], torch.tensor(h))


def test_output_keys_dtypes_match_builder(small):
    f = small[0][0]
    assert set(f) - {"reservoir"} == {
        "clean_ids",
        "corrupt_ids",
        "c1",
        "c2",
        "k1",
        "k2",
        "half",
        "answer_text",
        "meta",
    }
    for k in ("clean_ids", "corrupt_ids", "c1", "c2", "k1", "k2", "half"):
        assert f[k].dtype == torch.long
    assert f["clean_ids"].shape == f["corrupt_ids"].shape
    assert {
        "a",
        "b",
        "sum",
        "cf_a",
        "cf_b",
        "cf_sum",
        "split",
        "half",
        "sum_len",
        "n_carry",
        "fold",
    } <= set(f["meta"].columns)
    assert f["answer_text"] == [str(s) for s in f["meta"]["sum"]]


def test_meta_consistent_with_operands(small):
    for fi, f in enumerate(small[0]):
        m = f["meta"]
        assert (m["sum"] == m["a"] + m["b"]).all()
        assert (m["cf_sum"] == m["cf_a"] + m["cf_b"]).all()
        assert (m["sum_len"] == m["sum"].astype(str).str.len()).all()
        assert [bf.count_carries(a, b) for a, b in zip(m["a"], m["b"])] == m["n_carry"].tolist()
        assert (m["fold"] == fi).all()
        assert m["a"].between(1000, 9999).all() and m["b"].between(1000, 9999).all()


def test_no_pair_in_two_folds_clean_or_cf(small):
    folds, *_ = small
    owner: dict[tuple[int, int], int] = {}
    for fi, f in enumerate(folds):
        clean, cf, res = _pairs(f)
        for k in set(clean) | set(cf) | set(res):
            assert owner.setdefault(k, fi) == fi, f"{k} in folds {owner[k]} and {fi}"
        # within a fold: clean rows distinct, reservoir distinct
        assert len(set(clean)) == len(clean)
        assert len(set(res)) == len(res)


def test_no_injected_exclusion_or_old_split(small):
    folds, counts, ex_train, ex_old = small
    assert counts["n_excluded_training"] >= len(ex_train)
    for f in folds:
        used = set().union(*map(set, _pairs(f)))
        assert not used & ex_train
        assert not used & ex_old


def test_strata_balanced_across_folds(small):
    folds, *_ = small
    tot = [
        np.bincount(
            [bf.stratum_of(*r) for r in f["meta"][["a", "b"]].to_numpy().tolist()]
            + [bf.stratum_of(*r) for r in f["reservoir"].tolist()],
            minlength=10,
        )
        for f in folds
    ]
    tot = np.array(tot)
    assert (tot.max(0) - tot.min(0) <= 1).all()
    assert (tot.sum(1) == PER_FOLD).all()


def test_cf_valid_and_from_own_reservoir(small):
    folds, *_ = small
    for f in folds:
        m = f["meta"]
        res = set(_pairs(f)[2])
        _, cf, _ = _pairs(f)
        assert all(k in res for k in cf)
        assert (m["sum_len"] == m["cf_sum"].astype(str).str.len()).all()
        assert (f["c1"] != f["k1"]).all() and (f["c2"] != f["k2"]).all()
        assert (f["clean_ids"][:, :-1] != f["corrupt_ids"][:, :-1]).any(dim=1).all()
        assert torch.equal(f["clean_ids"][:, -1], f["c1"])
        assert torch.equal(f["corrupt_ids"][:, -1], f["k1"])


def test_reservoir_never_clean_anywhere(small):
    folds, *_ = small
    all_clean = set()
    all_res = set()
    for f in folds:
        clean, _, res = _pairs(f)
        all_clean |= set(clean)
        all_res |= set(res)
    assert not all_clean & all_res


def test_cf_tokens_match_real_tokenization(small, tok):
    f = small[0][0]
    m = f["meta"]
    for i in range(len(m)):
        ct = data.answer_tokens(tok, str(m["sum"][i]))
        kt = data.answer_tokens(tok, str(m["cf_sum"][i]))
        assert [f["c1"][i].item(), f["c2"][i].item()] == ct
        assert [f["k1"][i].item(), f["k2"][i].item()] == kt


def test_prompt_template_and_shared_length(small, tok):
    f = small[0][0]
    m = f["meta"]
    for i in (0, N_DISC, N_DISC + N_VAL - 1):
        p = data.ADDITION_PROMPT.format(a=m["a"][i], b=m["b"][i])
        ids = tok(p, add_special_tokens=False)["input_ids"] + [f["c1"][i].item()]
        assert f["clean_ids"][i].tolist() == ids


def test_deterministic_and_seed_sensitive(tok, small):
    folds, _, ex_t, ex_o = small
    again, _ = _build(tok, 0, ex_t, ex_o)
    other, _ = _build(tok, 1, ex_t, ex_o)
    for x, y in zip(folds, again):
        assert x["meta"].equals(y["meta"])
        assert torch.equal(x["clean_ids"], y["clean_ids"])
        assert torch.equal(x["corrupt_ids"], y["corrupt_ids"])
        assert np.array_equal(x["reservoir"], y["reservoir"])
    assert not folds[0]["meta"][["a", "b"]].equals(other[0]["meta"][["a", "b"]])
    assert not folds[0]["meta"]["cf_a"].equals(other[0]["meta"]["cf_a"])


def test_cf_rng_is_separate_from_candidate_rng(tok, small):
    # changing only a fold's cf stream cannot change which rows are clean
    folds, _, ex_t, ex_o = small
    f = folds[0]
    m = f["meta"]
    a = np.r_[m["a"].to_numpy(), f["reservoir"][:, 0]]
    b = np.r_[m["b"].to_numpy(), f["reservoir"][:, 1]]
    strata = np.array([bf.stratum_of(int(x), int(y)) for x, y in zip(a, b)])
    prompts = [data.ADDITION_PROMPT.format(a=int(x), b=int(y)) for x, y in zip(a, b)]
    ans = [str(int(x + y)) for x, y in zip(a, b)]
    pi, ai = data._validate_pool_tokenization(
        tok, prompts, ans, [p + s for p, s in zip(prompts, ans)], where="t"
    )
    g0 = bf.build_fold(a, b, pi, ai, strata, 0, 0, N_DISC, N_VAL, N_RES)
    g1 = bf.build_fold(a, b, pi, ai, strata, 0, 5, N_DISC, N_VAL, N_RES)
    assert g0["meta"][["a", "b"]].equals(g1["meta"][["a", "b"]])
    assert not g0["meta"]["cf_a"].equals(g1["meta"]["cf_a"])


def _tiny_fold_inputs(tok, a, b):
    strata = np.array([bf.stratum_of(int(x), int(y)) for x, y in zip(a, b)])
    prompts = [data.ADDITION_PROMPT.format(a=int(x), b=int(y)) for x, y in zip(a, b)]
    ans = [str(int(x + y)) for x, y in zip(a, b)]
    fulls = [p + s for p, s in zip(prompts, ans)]
    pi, ai = data._validate_pool_tokenization(tok, prompts, ans, fulls, where="t")
    return pi, ai, strata


def test_pick_counterfactual_silently_falls_back_without_guard():
    # documents why build_fold asserts: an empty preferred set falls back to all matches
    key = np.array([4, 4, 4])
    f = np.array([1, 2, 3])
    s = np.array([1, 2, 3])
    prefer = np.array([False, False, False])
    rng = np.random.default_rng(0)
    assert data.pick_counterfactual(key, f, s, 0, rng, prefer=prefer) in (1, 2)


def test_cf_outside_reservoir_raises(tok, monkeypatch):
    a = np.array([1111, 1234, 2345, 3456, 8000, 8001, 9000, 9100, 9200, 9300])
    b = np.array([2222, 2468, 3571, 4682, 8123, 8222, 9456, 9555, 9654, 9753])
    pi, ai, strata = _tiny_fold_inputs(tok, a, b)
    orig = data._assemble_pairs
    # simulate the silent fallback: prefer set empty -> every match eligible
    monkeypatch.setattr(
        data,
        "_assemble_pairs",
        lambda p, an, ci, mk, rng, prefer: orig(p, an, ci, mk, rng, np.zeros_like(prefer)),
    )
    with pytest.raises(ValueError, match="outside the fold's reservoir"):
        bf.build_fold(a, b, pi, ai, strata, 0, 0, 4, 2, 4)


def test_fold_without_valid_cf_raises(tok):
    # the lone 4-digit-sum row sorts first (so is dealt a clean role) and has no same-length partner
    a = np.array([1111, 9234, 8345, 9456, 8000, 7000, 7100, 7200, 7300, 9999])
    b = np.array([2222, 8468, 9571, 8682, 8123, 8001, 8102, 8203, 8304, 1001])
    pi, ai, strata = _tiny_fold_inputs(tok, a, b)
    with pytest.raises(ValueError):
        bf.build_fold(a, b, pi, ai, strata, 0, 0, 4, 2, 4)


def test_roundtrip_load_pairs_word_and_symbol(tok, small, tmp_path):
    folds, *_ = small
    copy_src = EXP / "data" / data.COPY_PAIRS_FILENAME
    if not copy_src.is_file():
        copy_src = tmp_path / "copy_src.pt"
        data.save_pairs({"x": torch.zeros(2, dtype=torch.long)}, copy_src)
    hashes = bf.write_folds(folds, tok, tmp_path, copy_src)
    assert len(hashes) == 3 * N_FOLDS
    for fi, f in enumerate(folds):
        d = tmp_path / f"fold_{fi:02d}"
        w = data.load_pairs(d, "word")
        s = data.load_pairs(d, "symbol")
        for loaded in (w, s):
            assert len(loaded["disc"]["meta"]) == N_DISC
            assert len(loaded["val"]["meta"]) == N_VAL
            assert (loaded["disc"]["half"] == 0).sum() == N_DISC // 2
            assert (loaded["disc"]["half"] == 1).sum() == N_DISC // 2
            assert (loaded["val"]["half"] == -1).all()
        assert w["disc"]["meta"].equals(s["disc"]["meta"])
        assert w["val"]["meta"].equals(s["val"]["meta"])
        for k in ("c1", "c2", "k1", "k2"):
            assert torch.equal(w["disc"][k], s["disc"][k])
            assert torch.equal(w["val"][k], s["val"][k])
        assert not torch.equal(w["disc"]["clean_ids"], s["disc"]["clean_ids"])  # different surface
        assert torch.equal(w["disc"]["clean_ids"], f["clean_ids"][:N_DISC])
        assert torch.equal(w["val"]["corrupt_ids"], f["corrupt_ids"][N_DISC:])
    # hashes are real sha256 of the files on disk
    for rel, h in hashes.items():
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == h


def test_summarize_manifest(small):
    folds, counts, *_ = small
    man = bf.summarize(folds, 0, N_DISC, N_VAL, N_RES, counts)
    assert man["n_folds"] == N_FOLDS
    for fm in man["folds"]:
        assert fm["n_total"] == PER_FOLD
        sc = fm["strata_counts"]
        assert sum(sc["disc"]) == N_DISC and sum(sc["val"]) == N_VAL and sum(sc["res"]) == N_RES
    assert 0.5 < man["five_digit_share"] < 0.7
    json.dumps(man)


# ---------------------------------------------------------------- the real build


MANIFEST = EXP / "data" / "folds" / "manifest.json"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="real folds not built")
def test_real_build_manifest():
    man = json.loads(MANIFEST.read_text())
    assert man["n_folds"] == 10
    assert (man["n_disc"], man["n_val"], man["n_res"]) == (512, 256, 768)
    assert abs(man["mean_a"] - 5500) < 150 and abs(man["mean_b"] - 5500) < 150
    assert abs(man["five_digit_share"] - 0.607) < 0.03
    assert len(man["folds"]) == 10
    tot = np.array(
        [[sum(f["strata_counts"][r][s] for r in bf.ROLES) for s in range(10)] for f in man["folds"]]
    )
    assert (tot.max(0) - tot.min(0) <= 1).all()
    for f in man["folds"]:
        assert f["n_total"] == 1536 and (f["n_disc"], f["n_val"], f["n_res"]) == (512, 256, 768)
    for rel, h in man["sha256"].items():
        p = MANIFEST.parent / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h, rel
    assert len(man["sha256"]) == 30
