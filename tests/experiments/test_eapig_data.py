"""``experiments/eapig-circuit-check/data.py`` — the EAP-IG circuit-check
pair builders (PLAN.md "Data").

Silent-failure risk here is exactly the one CLAUDE.md's promotion rule
flags: a counterfactual that silently shares an answer token with its clean
problem, a discovery/validation split that silently leaks an operand pair,
or a tokenization that silently disagrees with training (an extra BOS, a
different answer-span split) would corrupt every downstream EAP-IG score
with nothing crashing to flag it. ``pick_counterfactual`` and the sha256
save/load round-trip are tested on tiny in-process arrays (no tokenizer, no
parquet); ``build_addition_pairs``/``build_copy_pairs``/``build_tinystories``
are tested against small fixtures built in this file AND, for
``build_addition_pairs``, the real local ``D_algo_eval_bare.parquet`` (the
batch-tokenize-the-whole-5,000-row-pool cost is well under a second — see
module docstring in ``data.py``).

The Llama-3.2-1B tokenizer loads from the local HF cache; ``HF_HUB_OFFLINE``
is forced below so this suite never touches the network even if that cache
is warm.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from tests._scriptloader import repo_root

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = repo_root()
_MODULE_PATH = REPO_ROOT / "experiments" / "eapig-circuit-check" / "data.py"
_spec = importlib.util.spec_from_file_location("eapig_circuit_check_data", _MODULE_PATH)
eapig_data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eapig_data
_spec.loader.exec_module(eapig_data)

pick_counterfactual = eapig_data.pick_counterfactual
answer_tokens = eapig_data.answer_tokens
save_pairs = eapig_data.save_pairs
load_pairs = eapig_data.load_pairs
build_addition_pairs = eapig_data.build_addition_pairs
build_copy_pairs = eapig_data.build_copy_pairs
build_tinystories = eapig_data.build_tinystories
ADDITION_PROMPT = eapig_data.ADDITION_PROMPT
ADDITION_PAIRS_FILENAME = eapig_data.ADDITION_PAIRS_FILENAME
COPY_PAIRS_FILENAME = eapig_data.COPY_PAIRS_FILENAME
TINYSTORIES_FILENAME = eapig_data.TINYSTORIES_FILENAME

REAL_EVAL_PARQUET = REPO_ROOT / "experiments/training-run/data/full/D_algo_eval_bare.parquet"


@pytest.fixture(scope="module")
def llama_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def _addition_row(a: int, b: int) -> dict:
    s = a + b
    prompt = ADDITION_PROMPT.format(a=a, b=b)
    answer = str(s)
    return {
        "op": "+",
        "cell": "4x4",
        "a": a,
        "b": b,
        "true_answer": s,
        "shown_answer": s,
        "prompt_text": prompt,
        "answer_text": answer,
        "full_text": prompt + answer,
    }


def _make_pool_df(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame([_addition_row(a, b) for a, b in pairs])


# A pool with 6 four-digit-sum pairs and 6 five-digit-sum pairs, every
# unordered {a, b} distinct AND, within each group, every pairwise sum's
# first/second answer token distinct (so every row has a valid counterfactual
# -- real sums ending in a round number, e.g. 2000+3000=5000, share their
# last-digit token with every other round sum and have none).
FOUR_DIGIT_SUM_PAIRS = [
    (4458, 2577),  # 7035
    (4104, 4646),  # 8750
    (2722, 1165),  # 3887
    (2060, 4954),  # 7014
    (2658, 4761),  # 7419
    (2242, 4964),  # 7206
]
FIVE_DIGIT_SUM_PAIRS = [
    (8904, 7933),  # 16837
    (9779, 6789),  # 16568
    (9134, 6140),  # 15274
    (7308, 6144),  # 13452
    (5776, 7052),  # 12828
    (9362, 9930),  # 19292
]


# ---------------------------------------------------------------------------
# pick_counterfactual
# ---------------------------------------------------------------------------


class TestPickCounterfactual:
    def test_basic_match(self):
        rng = np.random.default_rng(0)
        match_key = np.array([1, 1, 1, 2])
        c1 = np.array([10, 20, 10, 10])
        c2 = np.array([1, 2, 3, 1])
        # row1 differs at both c1 and c2 from idx0; row2 shares c1 with idx0.
        assert pick_counterfactual(match_key, c1, c2, 0, rng) == 1

    def test_excludes_self(self):
        rng = np.random.default_rng(0)
        match_key = np.array([1, 1])
        c1 = np.array([5, 9])
        c2 = np.array([5, 9])
        assert pick_counterfactual(match_key, c1, c2, 0, rng) == 1

    def test_raises_on_digit_count_mismatch(self):
        rng = np.random.default_rng(0)
        match_key = np.array([1, 2])
        c1 = np.array([5, 9])
        c2 = np.array([5, 9])
        with pytest.raises(ValueError, match="no valid counterfactual"):
            pick_counterfactual(match_key, c1, c2, 0, rng)

    def test_raises_when_every_candidate_shares_a_token(self):
        rng = np.random.default_rng(0)
        match_key = np.array([1, 1, 1])
        c1 = np.array([5, 5, 2])  # row1 shares c1 with idx0
        c2 = np.array([7, 3, 7])  # row2 shares c2 with idx0
        with pytest.raises(ValueError, match="no valid counterfactual"):
            pick_counterfactual(match_key, c1, c2, 0, rng)

    def test_never_returns_self_even_if_arrays_coincide(self):
        # idx itself trivially satisfies match_key/c1/c2 equality; must still
        # be excluded regardless of how the other rows compare.
        match_key = np.array([1, 1, 1])
        c1 = np.array([5, 9, 20])
        c2 = np.array([5, 30, 40])
        for seed in range(10):
            rng = np.random.default_rng(seed)
            cf = pick_counterfactual(match_key, c1, c2, 0, rng)
            assert cf != 0

    def test_prefer_mask_restricts_when_a_preferred_candidate_exists(self):
        match_key = np.array([1, 1, 1, 1])
        c1 = np.array([0, 10, 20, 30])
        c2 = np.array([0, 10, 20, 30])
        prefer = np.array([False, False, True, False])
        for seed in range(20):
            rng = np.random.default_rng(seed)
            assert pick_counterfactual(match_key, c1, c2, 0, rng, prefer=prefer) == 2

    def test_prefer_mask_falls_back_when_no_preferred_candidate_exists(self):
        match_key = np.array([1, 1, 1])
        c1 = np.array([0, 10, 20])
        c2 = np.array([0, 10, 20])
        prefer = np.zeros(3, dtype=bool)
        rng = np.random.default_rng(1)
        assert pick_counterfactual(match_key, c1, c2, 0, rng, prefer=prefer) in (1, 2)

    def test_deterministic_given_seed(self):
        match_key = np.ones(10, dtype=np.int64)
        c1 = np.arange(10)
        c2 = np.arange(10)
        r1 = pick_counterfactual(match_key, c1, c2, 0, np.random.default_rng(42))
        r2 = pick_counterfactual(match_key, c1, c2, 0, np.random.default_rng(42))
        assert r1 == r2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            pick_counterfactual(
                np.array([1, 1]), np.array([1]), np.array([1, 2]), 0, np.random.default_rng(0)
            )

    def test_idx_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            pick_counterfactual(
                np.array([1, 1]), np.array([1, 2]), np.array([1, 2]), 5, np.random.default_rng(0)
            )


# ---------------------------------------------------------------------------
# answer_tokens
# ---------------------------------------------------------------------------


class TestAnswerTokens:
    def test_four_digit_group_sum(self, llama_tokenizer):
        ids = answer_tokens(llama_tokenizer, "15962")
        assert ids == [11068, 5538]
        assert llama_tokenizer.decode(ids) == "15962"

    def test_four_digit_number(self, llama_tokenizer):
        ids = answer_tokens(llama_tokenizer, "5068")
        assert ids == [19673, 23]

    def test_single_token_number_raises(self, llama_tokenizer):
        with pytest.raises(ValueError, match="expected 2"):
            answer_tokens(llama_tokenizer, "7")

    def test_many_token_number_raises(self, llama_tokenizer):
        with pytest.raises(ValueError, match="expected 2"):
            answer_tokens(llama_tokenizer, "123456789")

    def test_leading_space_three_tokens_raises(self, llama_tokenizer):
        with pytest.raises(ValueError, match="expected 2"):
            answer_tokens(llama_tokenizer, " 5068")

    def test_leading_space_two_tokens_raises_via_whitespace_check(self, llama_tokenizer):
        # " 68" tokenizes to exactly [' ', '68'] -- passes the length check
        # and decodes back exactly, so only the whitespace check catches it.
        with pytest.raises(ValueError, match="whitespace"):
            answer_tokens(llama_tokenizer, " 68")


# ---------------------------------------------------------------------------
# save_pairs / load_pairs
# ---------------------------------------------------------------------------


class TestSaveLoadPairs:
    def _sample_pairs(self) -> dict:
        return {
            "clean_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "corrupt_ids": torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.long),
            "c1": torch.tensor([3, 6], dtype=torch.long),
            "k1": torch.tensor([9, 12], dtype=torch.long),
            "meta": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        }

    def test_round_trip(self, tmp_path):
        pairs = self._sample_pairs()
        path = tmp_path / "pairs.pt"
        save_pairs(pairs, path)
        loaded = load_pairs(path)
        assert torch.equal(loaded["clean_ids"], pairs["clean_ids"])
        assert torch.equal(loaded["corrupt_ids"], pairs["corrupt_ids"])
        assert torch.equal(loaded["c1"], pairs["c1"])
        assert torch.equal(loaded["k1"], pairs["k1"])
        pd.testing.assert_frame_equal(loaded["meta"], pairs["meta"])

    def test_creates_parent_dir(self, tmp_path):
        pairs = self._sample_pairs()
        path = tmp_path / "nested" / "dir" / "pairs.pt"
        save_pairs(pairs, path)
        assert path.is_file()

    def test_load_detects_tampering(self, tmp_path):
        pairs = self._sample_pairs()
        path = tmp_path / "pairs.pt"
        save_pairs(pairs, path)
        # Corrupt a tensor in place and re-save without updating the hash.
        corrupted = torch.load(path, weights_only=False)
        corrupted["clean_ids"][0, 0] = 999
        torch.save(corrupted, path)
        with pytest.raises(ValueError, match="sha256 mismatch"):
            load_pairs(path)

    def test_load_without_recorded_hash_skips_verification(self, tmp_path):
        pairs = self._sample_pairs()
        path = tmp_path / "pairs_no_hash.pt"
        torch.save(pairs, path)  # bypass save_pairs: no "_token_sha256" key
        loaded = load_pairs(path)
        assert "_token_sha256" not in loaded
        assert torch.equal(loaded["clean_ids"], pairs["clean_ids"])


# ---------------------------------------------------------------------------
# build_addition_pairs
# ---------------------------------------------------------------------------


@pytest.fixture
def to_parquet(tmp_path):
    """Write a pool DataFrame to a fresh parquet file under pytest's own
    ``tmp_path`` (auto-cleaned), returning its path. Each call within a test
    gets a distinct filename."""
    counter = {"n": 0}

    def _write(df: pd.DataFrame) -> str:
        counter["n"] += 1
        path = tmp_path / f"pool_{counter['n']}.parquet"
        df.to_parquet(path, index=False)
        return str(path)

    return _write


class TestBuildAdditionPairs:
    def test_basic_shapes_and_split(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS  # 12 rows
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        n = 6
        assert result["clean_ids"].shape[0] == n
        assert result["corrupt_ids"].shape == result["clean_ids"].shape
        for key in ("c1", "c2", "k1", "k2"):
            assert result[key].shape == (n,)
        meta = result["meta"]
        assert len(meta) == n
        assert set(meta["split"]) <= {"discovery", "validation"}
        assert (meta["split"] == "discovery").sum() == 4
        assert (meta["split"] == "validation").sum() == 2
        # discovery halves: exactly 2 rows in half 0 and 2 in half 1.
        disc = meta[meta["split"] == "discovery"]
        assert sorted(disc["half"].tolist()) == [0, 0, 1, 1]
        assert (meta[meta["split"] == "validation"]["half"] == -1).all()

    def test_last_token_matches_c1_k1(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        assert torch.equal(result["clean_ids"][:, -1], result["c1"])
        assert torch.equal(result["corrupt_ids"][:, -1], result["k1"])

    def test_top_level_half_and_answer_text_mirror_meta(self, llama_tokenizer, to_parquet):
        # run.py (the GPU-side driver) indexes these two fields directly on
        # the top-level dict, not through "meta" -- e.g. ``disc["half"]``,
        # ``val["answer_text"][s:s+bs]``.
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        meta = result["meta"]
        assert isinstance(result["half"], torch.Tensor)
        assert result["half"].dtype == torch.long
        assert result["half"].tolist() == meta["half"].tolist()
        assert isinstance(result["answer_text"], list)
        assert result["answer_text"] == [str(s) for s in meta["sum"]]
        # the real template's answer span: str(sum), no sign/padding surprises
        for a, b, text in zip(meta["a"], meta["b"], result["answer_text"]):
            assert text == str(a + b)

    def test_counterfactual_differs_at_both_tokens(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        assert torch.all(result["c1"] != result["k1"])
        assert torch.all(result["c2"] != result["k2"])

    def test_counterfactual_shares_sum_digit_count(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        meta = result["meta"]
        digits = meta["sum"].astype(str).str.len()
        cf_digits = meta["cf_sum"].astype(str).str.len()
        assert (digits == cf_digits).all()

    def test_counterfactual_never_the_row_itself(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        meta = result["meta"]
        same = (meta["a"] == meta["cf_a"]) & (meta["b"] == meta["cf_b"])
        assert not same.any()

    def test_discovery_validation_disjoint_in_operand_pairs(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        meta = result["meta"]
        disc_pairs = {
            tuple(sorted((a, b)))
            for a, b in zip(
                meta[meta["split"] == "discovery"]["a"], meta[meta["split"] == "discovery"]["b"]
            )
        }
        val_pairs = {
            tuple(sorted((a, b)))
            for a, b in zip(
                meta[meta["split"] == "validation"]["a"], meta[meta["split"] == "validation"]["b"]
            )
        }
        assert disc_pairs.isdisjoint(val_pairs)

    def test_deterministic_given_seed(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        path = to_parquet(df)
        r1 = build_addition_pairs(path, llama_tokenizer, n_disc=4, n_val=2, seed=7)
        r2 = build_addition_pairs(path, llama_tokenizer, n_disc=4, n_val=2, seed=7)
        assert torch.equal(r1["clean_ids"], r2["clean_ids"])
        assert torch.equal(r1["corrupt_ids"], r2["corrupt_ids"])
        pd.testing.assert_frame_equal(r1["meta"], r2["meta"])

    def test_different_seed_changes_the_draw(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        path = to_parquet(df)
        r1 = build_addition_pairs(path, llama_tokenizer, n_disc=4, n_val=2, seed=0)
        r2 = build_addition_pairs(path, llama_tokenizer, n_disc=4, n_val=2, seed=1)
        assert not r1["meta"][["a", "b"]].equals(r2["meta"][["a", "b"]])

    def test_odd_n_disc_raises(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS
        df = _make_pool_df(pool)
        with pytest.raises(ValueError, match="n_disc must be even"):
            build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=3, n_val=2, seed=0)

    def test_pool_too_small_raises(self, llama_tokenizer, to_parquet):
        df = _make_pool_df(FOUR_DIGIT_SUM_PAIRS)  # only 6 rows
        with pytest.raises(ValueError, match="exceeds pool size"):
            build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=4, seed=0)

    def test_duplicate_unordered_pair_raises(self, llama_tokenizer, to_parquet):
        pool = FOUR_DIGIT_SUM_PAIRS + [(2577, 4458)]  # same unordered pair as FOUR_DIGIT_SUM_PAIRS[0]
        df = _make_pool_df(pool)
        with pytest.raises(ValueError, match="duplicate unordered operand pairs"):
            build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)

    def test_answer_not_two_tokens_raises(self, llama_tokenizer, to_parquet):
        rows = [_addition_row(a, b) for a, b in FOUR_DIGIT_SUM_PAIRS]
        # A fabricated row: real 4-digit-operand prompt (so prompt length is
        # unaffected) with its answer text swapped for a 1-token number.
        bad = dict(_addition_row(3333, 4444))  # a fresh operand pair, not already in the pool
        bad["answer_text"] = "7"
        bad["full_text"] = bad["prompt_text"] + "7"
        df = pd.DataFrame([*rows, bad])
        with pytest.raises(ValueError, match="not 2 tokens"):
            build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)

    def test_mismatched_full_text_raises(self, llama_tokenizer, to_parquet):
        rows = [_addition_row(a, b) for a, b in FOUR_DIGIT_SUM_PAIRS]
        rows[0] = dict(rows[0])
        rows[0]["full_text"] = rows[0]["prompt_text"] + "not the real answer"
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="tokenize\\(full_text\\)"):
            build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)

    def test_inconsistent_prompt_length_raises(self, llama_tokenizer, to_parquet):
        rows = [_addition_row(a, b) for a, b in FOUR_DIGIT_SUM_PAIRS]
        odd = _addition_row(5, 6)  # 1-digit operands -> much shorter prompt
        odd["cell"] = "4x4"
        df = pd.DataFrame([*rows, odd])
        with pytest.raises(ValueError, match="prompt token length"):
            build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)

    def test_filters_to_addition_4x4_cell(self, llama_tokenizer, to_parquet):
        rows = [_addition_row(a, b) for a, b in FOUR_DIGIT_SUM_PAIRS]
        other_op = dict(rows[0])
        other_op["op"] = "-"
        other_cell = dict(rows[1])
        other_cell["cell"] = "3x4"
        df = pd.DataFrame([*rows, other_op, other_cell])
        result = build_addition_pairs(to_parquet(df), llama_tokenizer, n_disc=4, n_val=2, seed=0)
        # only the 6 original op='+', cell='4x4' rows are eligible.
        assert len(result["meta"]) == 6

    @pytest.mark.skipif(not REAL_EVAL_PARQUET.is_file(), reason="local frozen parquet not present")
    def test_real_eval_parquet_small_n(self, llama_tokenizer):
        result = build_addition_pairs(REAL_EVAL_PARQUET, llama_tokenizer, n_disc=8, n_val=4, seed=0)
        assert result["clean_ids"].shape[0] == 12
        assert torch.all(result["c1"] != result["k1"])
        assert torch.all(result["c2"] != result["k2"])

    @pytest.mark.skipif(not REAL_EVAL_PARQUET.is_file(), reason="local frozen parquet not present")
    def test_real_eval_parquet_default_sizes(self, llama_tokenizer):
        result = build_addition_pairs(REAL_EVAL_PARQUET, llama_tokenizer, seed=0)
        assert result["clean_ids"].shape[0] == 512 + 256
        meta = result["meta"]
        assert (meta["split"] == "discovery").sum() == 512
        assert (meta["split"] == "validation").sum() == 256
        disc_pairs = {
            tuple(sorted((a, b)))
            for a, b in zip(
                meta[meta["split"] == "discovery"]["a"], meta[meta["split"] == "discovery"]["b"]
            )
        }
        val_pairs = {
            tuple(sorted((a, b)))
            for a, b in zip(
                meta[meta["split"] == "validation"]["a"], meta[meta["split"] == "validation"]["b"]
            )
        }
        assert disc_pairs.isdisjoint(val_pairs)
        assert len(disc_pairs) == 512
        assert len(val_pairs) == 256


# ---------------------------------------------------------------------------
# build_copy_pairs
# ---------------------------------------------------------------------------


class TestBuildCopyPairs:
    def test_basic_shapes(self, llama_tokenizer):
        result = build_copy_pairs(llama_tokenizer, n=20, seed=1)
        assert result["clean_ids"].shape[0] == 20
        assert result["corrupt_ids"].shape == result["clean_ids"].shape
        for key in ("c1", "c2", "k1", "k2"):
            assert result[key].shape == (20,)
        assert list(result["meta"].columns) == ["n", "cf_n"]
        assert len(result["meta"]) == 20

    def test_values_are_four_digit_and_distinct_within_clean_set(self, llama_tokenizer):
        result = build_copy_pairs(llama_tokenizer, n=50, seed=1)
        ns = result["meta"]["n"]
        assert ns.between(1000, 9999).all()
        assert ns.is_unique

    def test_counterfactual_differs_at_both_tokens(self, llama_tokenizer):
        result = build_copy_pairs(llama_tokenizer, n=50, seed=1)
        assert torch.all(result["c1"] != result["k1"])
        assert torch.all(result["c2"] != result["k2"])
        assert (result["meta"]["n"] != result["meta"]["cf_n"]).all()

    def test_last_token_matches_c1_k1(self, llama_tokenizer):
        result = build_copy_pairs(llama_tokenizer, n=10, seed=1)
        assert torch.equal(result["clean_ids"][:, -1], result["c1"])
        assert torch.equal(result["corrupt_ids"][:, -1], result["k1"])

    def test_deterministic_given_seed(self, llama_tokenizer):
        r1 = build_copy_pairs(llama_tokenizer, n=10, seed=3)
        r2 = build_copy_pairs(llama_tokenizer, n=10, seed=3)
        assert torch.equal(r1["clean_ids"], r2["clean_ids"])
        pd.testing.assert_frame_equal(r1["meta"], r2["meta"])

    def test_known_example_tokens(self, llama_tokenizer):
        # Direct check of the copy-task tokenization finding: the answer
        # after "?\n" tokenizes as 2 tokens, same split pattern as addition.
        ids = answer_tokens(llama_tokenizer, "5068")
        assert ids == [19673, 23]

    def test_n_must_be_positive(self, llama_tokenizer):
        with pytest.raises(ValueError, match="must be positive"):
            build_copy_pairs(llama_tokenizer, n=0, seed=1)

    def test_n_too_large_raises(self, llama_tokenizer):
        with pytest.raises(ValueError, match="no room"):
            build_copy_pairs(llama_tokenizer, n=9000, seed=1)


# ---------------------------------------------------------------------------
# build_tinystories
# ---------------------------------------------------------------------------


_SHORT_STORIES = [f"Cat sat on the mat number {i}." for i in range(5)]
_LONG_STORIES = [
    f"Story number {i}. " + "The sun was bright and the birds were singing in the trees. " * 10
    for i in range(8)
]


class TestBuildTinystories:
    def test_filters_and_truncates(self, llama_tokenizer):
        stories = _SHORT_STORIES + _LONG_STORIES
        length = 10
        result = build_tinystories(llama_tokenizer, n=5, length=length, seed=2, text_source=stories)
        assert result["input_ids"].shape == (5, length)

        long_tok = [
            llama_tokenizer(s, add_special_tokens=False)["input_ids"] for s in _LONG_STORIES
        ]
        long_prefixes = [ids[:length] for ids in long_tok if len(ids) >= length]
        for row in result["input_ids"].tolist():
            assert row in long_prefixes

    def test_no_bos_prepended(self, llama_tokenizer):
        stories = _LONG_STORIES
        result = build_tinystories(llama_tokenizer, n=3, length=10, seed=2, text_source=stories)
        bos_id = llama_tokenizer.bos_token_id
        if bos_id is not None:
            assert not (result["input_ids"][:, 0] == bos_id).any()
        # Every returned row is literally a prefix of tokenize(story, add_special_tokens=False).
        raw = [
            llama_tokenizer(s, add_special_tokens=False)["input_ids"][:10] for s in stories
        ]
        for row in result["input_ids"].tolist():
            assert row in raw

    def test_deterministic_given_seed(self, llama_tokenizer):
        r1 = build_tinystories(llama_tokenizer, n=3, length=10, seed=5, text_source=_LONG_STORIES)
        r2 = build_tinystories(llama_tokenizer, n=3, length=10, seed=5, text_source=_LONG_STORIES)
        assert torch.equal(r1["input_ids"], r2["input_ids"])

    def test_callable_text_source(self, llama_tokenizer):
        result = build_tinystories(
            llama_tokenizer, n=3, length=10, seed=2, text_source=lambda: list(_LONG_STORIES)
        )
        assert result["input_ids"].shape == (3, 10)

    def test_not_enough_long_stories_raises(self, llama_tokenizer):
        with pytest.raises(ValueError, match="only"):
            build_tinystories(llama_tokenizer, n=100, length=10, seed=2, text_source=_LONG_STORIES)

    def test_empty_text_source_raises(self, llama_tokenizer):
        with pytest.raises(ValueError, match="no stories"):
            build_tinystories(llama_tokenizer, n=1, length=10, seed=2, text_source=[])

    def test_n_and_length_must_be_positive(self, llama_tokenizer):
        with pytest.raises(ValueError, match="must be positive"):
            build_tinystories(llama_tokenizer, n=0, length=10, seed=2, text_source=_LONG_STORIES)
        with pytest.raises(ValueError, match="must be positive"):
            build_tinystories(llama_tokenizer, n=1, length=0, seed=2, text_source=_LONG_STORIES)


# ---------------------------------------------------------------------------
# load_pairs(directory) — the combined {"disc", "val", "copy", "stories"}
# dict run.py (the GPU-side driver) reads directly.
# ---------------------------------------------------------------------------


class TestLoadPairsDirectory:
    def _build_dir(self, tmp_path, llama_tokenizer, *, with_stories: bool):
        pool = FOUR_DIGIT_SUM_PAIRS + FIVE_DIGIT_SUM_PAIRS  # 12 rows
        df = _make_pool_df(pool)
        pool_path = tmp_path / "pool.parquet"
        df.to_parquet(pool_path, index=False)
        addition = build_addition_pairs(pool_path, llama_tokenizer, n_disc=4, n_val=2, seed=0)
        save_pairs(addition, tmp_path / ADDITION_PAIRS_FILENAME)
        copy = build_copy_pairs(llama_tokenizer, n=10, seed=1)
        save_pairs(copy, tmp_path / COPY_PAIRS_FILENAME)
        if with_stories:
            stories = build_tinystories(
                llama_tokenizer, n=3, length=10, seed=2, text_source=_LONG_STORIES
            )
            save_pairs(stories, tmp_path / TINYSTORIES_FILENAME)
        return addition, copy

    def test_disc_val_split_and_keys(self, tmp_path, llama_tokenizer):
        addition, _ = self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        assert set(data) == {"disc", "val", "copy"}
        disc, val = data["disc"], data["val"]
        assert disc["clean_ids"].shape[0] == 4
        assert val["clean_ids"].shape[0] == 2
        for key in ("clean_ids", "corrupt_ids", "c1", "c2", "k1", "k2", "half", "answer_text"):
            assert key in disc
            assert key in val

    def test_disc_half_is_zero_or_one(self, tmp_path, llama_tokenizer):
        self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        disc = data["disc"]
        assert isinstance(disc["half"], torch.Tensor)
        assert set(disc["half"].tolist()) <= {0, 1}
        # exactly the split-half sizes build_addition_pairs produced (2 and 2).
        assert int((disc["half"] == 0).sum()) == 2
        assert int((disc["half"] == 1).sum()) == 2

    def test_val_answer_text_is_sliceable_and_aligned(self, tmp_path, llama_tokenizer):
        self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        val = data["val"]
        assert isinstance(val["answer_text"], list)
        assert len(val["answer_text"]) == val["clean_ids"].shape[0]
        # run.py's exact access pattern: a batch slice of the answer list.
        batch = val["answer_text"][0:1]
        assert batch == val["answer_text"][:1]
        for a, b, text in zip(val["meta"]["a"], val["meta"]["b"], val["answer_text"]):
            assert text == str(a + b)

    def test_disc_val_disjoint_after_split(self, tmp_path, llama_tokenizer):
        self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        disc_pairs = {
            tuple(sorted((a, b)))
            for a, b in zip(data["disc"]["meta"]["a"], data["disc"]["meta"]["b"])
        }
        val_pairs = {
            tuple(sorted((a, b)))
            for a, b in zip(data["val"]["meta"]["a"], data["val"]["meta"]["b"])
        }
        assert disc_pairs.isdisjoint(val_pairs)

    def test_copy_passes_through_unsplit(self, tmp_path, llama_tokenizer):
        _, copy = self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        assert data["copy"]["clean_ids"].shape[0] == copy["clean_ids"].shape[0]
        assert torch.equal(data["copy"]["clean_ids"], copy["clean_ids"])

    def test_stories_is_bare_tensor_when_present(self, tmp_path, llama_tokenizer):
        self._build_dir(tmp_path, llama_tokenizer, with_stories=True)
        data = load_pairs(tmp_path)
        assert "stories" in data
        assert isinstance(data["stories"], torch.Tensor)
        assert data["stories"].shape == (3, 10)

    def test_stories_omitted_when_file_missing(self, tmp_path, llama_tokenizer):
        self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        assert "stories" not in data

    def test_eval_ld_style_batch_access(self, tmp_path, llama_tokenizer):
        # The exact slicing pattern run.py's eval_ld uses on a pairs dict.
        self._build_dir(tmp_path, llama_tokenizer, with_stories=False)
        data = load_pairs(tmp_path)
        disc = data["disc"]
        bs = 3
        n = disc["clean_ids"].shape[0]
        seen = 0
        for s in range(0, n, bs):
            sl = slice(s, s + bs)
            clean = disc["clean_ids"][sl]
            toks = [disc[k][sl] for k in ("c1", "k1", "c2", "k2")]
            assert clean.shape[0] == len(toks[0])
            seen += clean.shape[0]
        assert seen == n
