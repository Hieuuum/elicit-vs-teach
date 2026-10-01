"""``edl_converged_val_floor.py`` FAMILIES regexes and arm-label mappings.

Focused on the ts38/ts38mw/ts38pf/ts38pp quartet (EXPERIMENTS
§6.14/§6.15/§6.16 for the first three): ts38mw's, ts38pf's, and ts38pp's
base arms are the SAME ``evt-ts38-base-n<size>`` id the ts38 family reads
(reused, not retrained), while each family's own arm is a NEW id under its
own prefix (``evt-ts38mw-pretaught-n<size>`` / ``evt-ts38pf-preteachfmt-
n<size>`` / ``evt-ts38pp-pretaught-n<size>``). The regex-level guarantee
that matters is that no family ever picks up another family's own arm, and
that a stray infix or missing size never slips through — exercised primarily
as a match/no-match matrix rather than by driving the full ``collect()``
pipeline (which needs a populated store; see ``test_dataset_size_sweep.py``
for that style of test on the sibling driver), plus ONE end-to-end pass for
ts38pp over a synthetic tmp-path store (``test_collect_ts38pp_end_to_end_on_
synthetic_store``) proving the regex + ARM_MAPS wiring survives the real
``collect()`` call, not just isolated regex matches.

CPU-only, no persistent store, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from geode.zoo import PrequentialRecord, register_run, write_jsonl

from tests._scriptloader import load

ecvf = load("edl_converged_val_floor")

# (family, run_id, expected groups or None if the family's regex must NOT match)
_MATRIX = [
    ("ts38", "evt-ts38-base-n1000", ("base", "1000")),
    ("ts38", "evt-ts38-pretaught-n1000", ("pretaught", "1000")),
    ("ts38", "evt-ts38mw-pretaught-n1000", None),
    ("ts38", "evt-ts38mw-base-n1000", None),
    ("ts38", "evt-ts38-base-n1000-extra", None),
    ("ts38", "evt-ts38-parent-loraprobe-lr3e-4", None),
    ("ts38", "evt-ts38mw-parent-probe-lr3e-4", None),
    ("ts38mw", "evt-ts38-base-n1000", ("base", "1000")),
    ("ts38mw", "evt-ts38-pretaught-n1000", None),
    ("ts38mw", "evt-ts38mw-pretaught-n1000", ("pretaught", "1000")),
    ("ts38mw", "evt-ts38mw-base-n1000", None),
    ("ts38mw", "evt-ts38-base-n1000-extra", None),
    ("ts38mw", "evt-ts38-parent-loraprobe-lr3e-4", None),
    ("ts38mw", "evt-ts38mw-parent-probe-lr3e-4", None),
    ("ts38mw", "evt-ts38pf-preteachfmt-n1000", None),
    ("ts38", "evt-ts38pf-preteachfmt-n1000", None),
    ("ts38pf", "evt-ts38-base-n1000", ("base", "1000")),
    ("ts38pf", "evt-ts38-pretaught-n1000", None),
    ("ts38pf", "evt-ts38mw-pretaught-n1000", None),
    ("ts38pf", "evt-ts38pf-preteachfmt-n1000", ("preteachfmt", "1000")),
    ("ts38pf", "evt-ts38pf-base-n1000", None),
    ("ts38pf", "evt-ts38-base-n1000-extra", None),
    ("ts38pf", "evt-ts38-parent-loraprobe-lr3e-4", None),
    ("ts38pf", "evt-ts38mw-parent-probe-lr3e-4", None),
    ("ts38pf", "evt-ts38pf-preteachfmt-parent", None),
    # ts38pp (paper-protocol pre-teach: full-FT parent trained one epoch on
    # 4M unique op-notation examples, then LoRA targets). Reuses ts38's base
    # arm run-for-run (evt-ts38-base-n<size>), same lookahead-disjoint shape
    # as ts38mw/ts38pf, paired with a NEW evt-ts38pp-pretaught-n<size> arm —
    # all 5 ts38 sizes covered here (not just n1000) since this family's
    # eventual grid spans the full TS38_SIZES set.
    ("ts38pp", "evt-ts38-base-n1000", ("base", "1000")),
    ("ts38pp", "evt-ts38-base-n4642", ("base", "4642")),
    ("ts38pp", "evt-ts38-base-n21544", ("base", "21544")),
    ("ts38pp", "evt-ts38-base-n100000", ("base", "100000")),
    ("ts38pp", "evt-ts38-base-n316228", ("base", "316228")),
    ("ts38pp", "evt-ts38pp-pretaught-n1000", ("pretaught", "1000")),
    ("ts38pp", "evt-ts38pp-pretaught-n4642", ("pretaught", "4642")),
    ("ts38pp", "evt-ts38pp-pretaught-n21544", ("pretaught", "21544")),
    ("ts38pp", "evt-ts38pp-pretaught-n100000", ("pretaught", "100000")),
    ("ts38pp", "evt-ts38pp-pretaught-n316228", ("pretaught", "316228")),
    ("ts38pp", "evt-ts38pp-parent", None),
    ("ts38pp", "evt-ts38-pretaught-n1000", None),
    ("ts38pp", "evt-ts38mw-pretaught-n1000", None),
    ("ts38pp", "evt-ts38pf-preteachfmt-n1000", None),
    ("ts38pp", "evt-ts38pp-pretaught-n1000x", None),
    ("ts38pp", "evt-ts38pp-base-n1000", None),
    # cross-family: no OTHER family's regex may pick up ts38pp's own arm.
    ("ts38", "evt-ts38pp-pretaught-n1000", None),
    ("ts38mw", "evt-ts38pp-pretaught-n1000", None),
    ("ts38pf", "evt-ts38pp-pretaught-n1000", None),
    # ts38grid unified 3-arm grid extension (EXPERIMENTS.md §6.17,
    # 2026-08-16): the same three regexes must pick up the 8 NEW dataset
    # sizes automatically — no per-size code, so a few new-size rows are
    # enough to prove the pattern generalizes rather than re-running the
    # full n1000 matrix above at every new size.
    ("ts38", "evt-ts38-base-n128", ("base", "128")),
    ("ts38mw", "evt-ts38mw-pretaught-n215443", ("pretaught", "215443")),
    ("ts38pf", "evt-ts38pf-preteachfmt-n146780", ("preteachfmt", "146780")),
    ("ts38pf", "evt-ts38mw-pretaught-n128", None),
    # ts38mt (EXPERIMENTS §6.22): THREE own arms, none reused from any other
    # family (unlike ts38mw/ts38pf/ts38pp, which reuse ts38's base id
    # run-for-run). Must reject the fmt full-FT parent itself (no -n<size>
    # suffix) and every other family's own arm ids.
    ("ts38mt", "evt-ts38mt-base-n1000", ("base", "1000")),
    ("ts38mt", "evt-ts38mt-pp-n1000", ("pp", "1000")),
    ("ts38mt", "evt-ts38mt-fmt-n1000", ("fmt", "1000")),
    ("ts38mt", "evt-ts38mt-base-n316228", ("base", "316228")),
    ("ts38mt", "evt-ts38mt-pp-n316228", ("pp", "316228")),
    ("ts38mt", "evt-ts38mt-fmt-n316228", ("fmt", "316228")),
    ("ts38mt", "evt-ts38mt-fmt-parent", None),
    ("ts38mt", "evt-ts38-base-n1000", None),
    ("ts38mt", "evt-ts38pp-pretaught-n1000", None),
    ("ts38mt", "evt-ts38mw-pretaught-n1000", None),
    ("ts38mt", "evt-ts38pf-preteachfmt-n1000", None),
    ("ts38mt", "evt-ts38mt-base-n1000-extra", None),
    ("ts38mt", "evt-ts38mt-pretaught-n1000", None),
    # cross-family: no OTHER family's regex may pick up ts38mt's own arms.
    ("ts38", "evt-ts38mt-base-n1000", None),
    ("ts38pp", "evt-ts38mt-pp-n1000", None),
    ("ts38pf", "evt-ts38mt-fmt-n1000", None),
    # ts38tr ("truncated adapter" positive control): TWO own arms (k6/k7),
    # neither reused from any other family -- both built by
    # scripts/truncate_lora_parent.py from ts38mt's own evt-ts38mt-base-
    # n316228 run, never a training run themselves. Must reject the
    # build-script-minted parent ids (no -n<size> suffix, no committed yaml
    # at all) and every other family's own arm ids.
    ("ts38tr", "evt-ts38tr-k6-n1000", ("k6", "1000")),
    ("ts38tr", "evt-ts38tr-k7-n1000", ("k7", "1000")),
    ("ts38tr", "evt-ts38tr-k6-n4642", ("k6", "4642")),
    ("ts38tr", "evt-ts38tr-k7-n4642", ("k7", "4642")),
    ("ts38tr", "evt-ts38tr-k6-parent", None),
    ("ts38tr", "evt-ts38tr-k7-parent", None),
    ("ts38tr", "evt-ts38-base-n1000", None),
    ("ts38tr", "evt-ts38mt-base-n1000", None),
    ("ts38tr", "evt-ts38mt-pp-n1000", None),
    ("ts38tr", "evt-ts38pp-pretaught-n1000", None),
    ("ts38tr", "evt-ts38mw-pretaught-n1000", None),
    ("ts38tr", "evt-ts38pf-preteachfmt-n1000", None),
    ("ts38tr", "evt-ts38tr-k8-n1000", None),
    ("ts38tr", "evt-ts38tr-k6-n1000-extra", None),
    ("ts38tr", "evt-ts38tr-parent-n1000", None),
    # cross-family: no OTHER family's regex may pick up ts38tr's own arms.
    ("ts38", "evt-ts38tr-k6-n1000", None),
    ("ts38mw", "evt-ts38tr-k7-n1000", None),
    ("ts38pf", "evt-ts38tr-k6-n1000", None),
    ("ts38pp", "evt-ts38tr-k7-n1000", None),
    ("ts38mt", "evt-ts38tr-k6-n1000", None),
    # nl2 (EXPERIMENTS §6.12, dose16 NL installer): captures noinst/inst
    # directly like op/nl (no ARM_MAPS translation entry, see FAMILIES'
    # nl2 comment) — must pick up its own ids and REJECT the neighboring
    # nl/nl3/installer ids a naive "fig2nl" substring match would blur.
    ("nl2", "evt-llama-fig2nl2-noinst-n1000", ("noinst", "1000")),
    ("nl2", "evt-llama-fig2nl2-inst-n1000000", ("inst", "1000000")),
    ("nl2", "evt-llama-fig2nl-noinst-n1000", None),
    ("nl2", "evt-llama-fig2nl3-noinst-n1000", None),
    ("nl2", "evt-llama-fig2nl2-installer", None),
    ("nl", "evt-llama-fig2nl2-noinst-n1000", None),
]


@pytest.mark.parametrize(("family", "run_id", "expected"), _MATRIX)
def test_family_regex_matrix(family: str, run_id: str, expected: tuple[str, str] | None) -> None:
    pattern = ecvf.FAMILIES[family][0]
    match = pattern.match(run_id)
    if expected is None:
        assert match is None, f"{family} must NOT match {run_id!r}"
    else:
        assert match is not None, f"{family} must match {run_id!r}"
        assert (match.group(1), match.group(2)) == expected


def test_ts38_and_ts38mw_regexes_agree_on_the_shared_base_arm() -> None:
    """The one id both families are meant to pick up must parse identically
    under each family's own pattern — that's what makes the base arm safe
    to reuse across families."""
    run_id = "evt-ts38-base-n21544"
    ts38_match = ecvf.FAMILIES["ts38"][0].match(run_id)
    ts38mw_match = ecvf.FAMILIES["ts38mw"][0].match(run_id)
    assert ts38_match is not None and ts38mw_match is not None
    assert (ts38_match.group(1), ts38_match.group(2)) == ("base", "21544")
    assert (ts38mw_match.group(1), ts38mw_match.group(2)) == ("base", "21544")


def test_ts38_ts38mw_ts38pf_regexes_agree_on_the_shared_base_arm() -> None:
    """Same guarantee as above, extended to the third family reusing the
    identical base arm id."""
    run_id = "evt-ts38-base-n21544"
    for family in ("ts38", "ts38mw", "ts38pf"):
        match = ecvf.FAMILIES[family][0].match(run_id)
        assert match is not None, family
        assert (match.group(1), match.group(2)) == ("base", "21544"), family


def test_ts38_ts38mw_ts38pf_ts38pp_regexes_agree_on_the_shared_base_arm() -> None:
    """Same guarantee as above, extended to the fourth family reusing the
    identical base arm id."""
    run_id = "evt-ts38-base-n21544"
    for family in ("ts38", "ts38mw", "ts38pf", "ts38pp"):
        match = ecvf.FAMILIES[family][0].match(run_id)
        assert match is not None, family
        assert (match.group(1), match.group(2)) == ("base", "21544"), family


def test_all_family_stems_distinct() -> None:
    stems = [ecvf.FAMILIES[f][1] for f in ecvf.FAMILIES]
    assert len(stems) == len(set(stems))
    assert "ts38mw" in ecvf.FAMILIES
    assert ecvf.FAMILIES["ts38mw"][1] == "edl_converged_val_floor_ts38mw"
    assert ecvf.FAMILIES["ts38mw"][1] != ecvf.FAMILIES["ts38"][1]
    assert "ts38pf" in ecvf.FAMILIES
    assert ecvf.FAMILIES["ts38pf"][1] == "edl_converged_val_floor_ts38pf"
    assert ecvf.FAMILIES["ts38pf"][1] not in (ecvf.FAMILIES["ts38"][1], ecvf.FAMILIES["ts38mw"][1])
    assert "ts38pp" in ecvf.FAMILIES
    assert ecvf.FAMILIES["ts38pp"][1] == "edl_converged_val_floor_ts38pp"
    assert ecvf.FAMILIES["ts38pp"][1] not in (
        ecvf.FAMILIES["ts38"][1],
        ecvf.FAMILIES["ts38mw"][1],
        ecvf.FAMILIES["ts38pf"][1],
    )
    assert "ts38mt" in ecvf.FAMILIES
    assert ecvf.FAMILIES["ts38mt"][1] == "edl_converged_val_floor_ts38mt"
    assert ecvf.FAMILIES["ts38mt"][1] not in (
        ecvf.FAMILIES["ts38"][1],
        ecvf.FAMILIES["ts38mw"][1],
        ecvf.FAMILIES["ts38pf"][1],
        ecvf.FAMILIES["ts38pp"][1],
    )
    assert "ts38tr" in ecvf.FAMILIES
    assert ecvf.FAMILIES["ts38tr"][1] == "edl_converged_val_floor_ts38tr"
    assert ecvf.FAMILIES["ts38tr"][1] not in (
        ecvf.FAMILIES["ts38"][1],
        ecvf.FAMILIES["ts38mw"][1],
        ecvf.FAMILIES["ts38pf"][1],
        ecvf.FAMILIES["ts38pp"][1],
        ecvf.FAMILIES["ts38mt"][1],
    )


def test_arm_label_mappings_are_honest_and_distinct() -> None:
    """ts38's pretaught arm reads 'pre-taught (elicit)'; ts38mw's own
    pretaught-mw arm reads 'pre-taught-mw (elicit)'; ts38pf's arm reads
    'pre-teach-format' (deliberately NOT asserting "elicit" — that's the
    open question this family exists to answer); ts38pp's arm reads
    'pre-teach 4M full-FT' (same deliberate omission). None may collide, and
    the shared base label must read the same across all four."""
    assert ecvf.TS38_ARM["pretaught"][1] == "pre-taught (elicit)"
    assert ecvf.TS38MW_ARM["pretaught"][1] == "pre-taught-mw (elicit)"
    assert ecvf.TS38PF_ARM["preteachfmt"][1] == "pre-teach-format"
    assert ecvf.TS38PP_ARM["pretaught"][1] == "pre-teach 4M full-FT"
    labels = {
        ecvf.TS38_ARM["pretaught"][1],
        ecvf.TS38MW_ARM["pretaught"][1],
        ecvf.TS38PF_ARM["preteachfmt"][1],
        ecvf.TS38PP_ARM["pretaught"][1],
    }
    assert len(labels) == 4
    assert (
        ecvf.TS38_ARM["base"][1]
        == ecvf.TS38MW_ARM["base"][1]
        == ecvf.TS38PF_ARM["base"][1]
        == ecvf.TS38PP_ARM["base"][1]
        == "base (teach)"
    )
    # All map to the canonical noinst/inst condition every downstream
    # lookup (STYLE, groupby) keys on.
    assert (
        ecvf.TS38_ARM["base"][0]
        == ecvf.TS38MW_ARM["base"][0]
        == ecvf.TS38PF_ARM["base"][0]
        == ecvf.TS38PP_ARM["base"][0]
        == "noinst"
    )
    assert (
        ecvf.TS38_ARM["pretaught"][0]
        == ecvf.TS38MW_ARM["pretaught"][0]
        == ecvf.TS38PF_ARM["preteachfmt"][0]
        == ecvf.TS38PP_ARM["pretaught"][0]
        == "inst"
    )


def test_ts38mt_arm_map_is_identity_not_noinst_inst() -> None:
    """Unlike every other ARM_MAPS entry, ts38mt's 3 arms don't collapse to
    a binary noinst/inst condition -- each raw capture maps to itself, and
    STYLE_MAPS carries this family's own 3-color style instead of the
    module-global 2-key STYLE."""
    assert ecvf.TS38MT_ARM["base"] == ("base", "base (teach reference)")
    assert ecvf.TS38MT_ARM["pp"] == ("pp", "pre-teach 4M full-FT (elicit)")
    assert ecvf.TS38MT_ARM["fmt"] == ("fmt", "pre-teach-format full-FT (teach)")
    assert set(ecvf.TS38MT_STYLE) == {"base", "pp", "fmt"}
    assert "ts38mt" in ecvf.STYLE_MAPS
    assert ecvf.STYLE_MAPS["ts38mt"] is ecvf.TS38MT_STYLE


def test_ts38tr_arm_map_translates_to_noinst_inst_not_identity() -> None:
    """Unlike ts38mt's 3-arm identity mapping, ts38tr's k6/k7 are a binary
    condition translated to the canonical noinst/inst pair every downstream
    lookup (STYLE, groupby) keys on -- same idiom as ts38/ts38mw/ts38pf/
    ts38pp, deliberately NOT ts38mt's shape. It must not appear in
    STYLE_MAPS (falls back to the global 2-key STYLE, unlike ts38mt)."""
    assert ecvf.TS38TR_ARM["k6"][0] == "noinst"
    assert ecvf.TS38TR_ARM["k7"][0] == "inst"
    assert "ts38tr" not in ecvf.STYLE_MAPS


def test_collect_ts38mw_on_empty_store_returns_empty_dataframe_not_raise(tmp_path: Path) -> None:
    """A store whose runs/ dir exists but holds nothing matching (e.g. before
    any ts38mw run has been pushed) must come back as an empty table, not a
    KeyError from sorting a columnless DataFrame."""
    (tmp_path / "runs").mkdir(parents=True)

    df = ecvf.collect("ts38mw", tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_collect_ts38pf_on_empty_store_returns_empty_dataframe_not_raise(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir(parents=True)

    df = ecvf.collect("ts38pf", tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_collect_ts38pp_on_empty_store_returns_empty_dataframe_not_raise(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir(parents=True)

    df = ecvf.collect("ts38pp", tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_collect_ts38tr_on_empty_store_returns_empty_dataframe_not_raise(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir(parents=True)

    df = ecvf.collect("ts38tr", tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_collect_nl2_on_empty_store_returns_empty_dataframe_not_raise(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir(parents=True)

    df = ecvf.collect("nl2", tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


_HASH = "a1" * 32  # one well-formed masking hash shared by manifest + test_loss.json


def _ts38pp_manifest(run_id: str, n: int) -> dict:
    """Minimal valid spec 00 §2 manifest carrying the D-1 masking-hash extra
    (trimmed copy of ``test_edl_converged_val_floor_test_floor_column.py``'s
    ``_manifest`` — kept local rather than cross-imported, matching this
    test suite's convention of self-contained fixture files)."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-08-16T00:00:00+00:00",
        "git_commit": "deadbeef",
        "regime": "teach",
        "base_model": {"hf_id": "tiny/ts38", "revision": "main"},
        "task": {"name": "arith_target", "format_version": "1"},
        "dataset": {"name": "d_algo_bare", "n_unique_examples": n, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 2,
                "alpha": 4.0,
                "target_modules": ["q_proj"],
                "dropout": 0.0,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "batch_size": 2,
                "micro_batch_size": None,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "grad_clip": None,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": 1,
            "max_steps": None,
            "stopping": {"eps_nats": 0.002, "k": 5, "min_steps": 1},
            "epochs_total": 1,
            "seed": 0,
        },
        "trainable_param_count": 1000,
        "snapshot_steps": [],
        "cost": {"gpu_type": None, "est_usd": None, "actual_usd": None},
        "status": "complete",
        "masking_config_hash": _HASH,
    }


def _write_ts38pp_run(store: Path, run_id: str, n: int) -> None:
    """Register ``run_id`` and hand-write the artifacts ``collect()`` needs;
    the specific loss numbers don't matter for this test (only that the row
    lands with the right ``n``/``condition``), so fixed placeholder values."""
    register_run(_ts38pp_manifest(run_id, n), store=store)
    run_dir = store / "runs" / run_id
    write_jsonl(
        run_dir / "logs" / "prequential.jsonl",
        [
            PrequentialRecord(
                step=0, epoch=1, example_ids=[0, 1], label_token_count=4, loss_sum_nats=8.0
            )
        ],
    )
    with (run_dir / "eval_log.jsonl").open("w") as fh:
        fh.write(json.dumps({"step": 1, "val_loss_nats": 1.0, "stopping_eval": True}) + "\n")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "test_loss.json").write_text(
        json.dumps(
            {
                "n_test_examples": 8,
                "label_token_count": 16,
                "loss_sum_nats": 16.0 * 0.5,
                "loss_per_label_token_nats": 0.5,
                "masking_config_hash": _HASH,
            }
        )
    )


def test_collect_ts38pp_end_to_end_on_synthetic_store(geode_store: Path) -> None:
    """Drives ``collect("ts38pp", ...)`` over a populated synthetic store
    (2 sizes x both arms) rather than just the regex matrix above — proves
    the FAMILIES regex + ARM_MAPS translation produce the right ``condition``
    per row through the real pipeline, not just in isolation."""
    for n in (1000, 4642):
        _write_ts38pp_run(geode_store, f"evt-ts38-base-n{n}", n)
        _write_ts38pp_run(geode_store, f"evt-ts38pp-pretaught-n{n}", n)

    df = ecvf.collect("ts38pp", geode_store)

    assert len(df) == 4
    assert set(df["n"]) == {1000, 4642}
    for n in (1000, 4642):
        by_n = df[df["n"] == n]
        assert set(by_n["condition"]) == {"noinst", "inst"}
    # condition wiring matches the label mapping asserted in
    # test_arm_label_mappings_are_honest_and_distinct: base -> noinst,
    # pretaught -> inst — pinned again here through the real regex + collect().
    base_row = df[(df["n"] == 1000) & (df["condition"] == "noinst")].iloc[0]
    pretaught_row = df[(df["n"] == 1000) & (df["condition"] == "inst")].iloc[0]
    assert base_row["condition"] == ecvf.TS38PP_ARM["base"][0]
    assert pretaught_row["condition"] == ecvf.TS38PP_ARM["pretaught"][0]


def test_collect_ts38tr_end_to_end_on_synthetic_store(geode_store: Path) -> None:
    """Same style as ts38pp's own end-to-end test above, reusing its generic
    manifest/log fixture writer (nothing about it is ts38pp-specific beyond
    the run ids it's called with) -- proves the ts38tr FAMILIES regex +
    ARM_MAPS translation produce the right ``condition`` per row through the
    real ``collect()`` pipeline, not just isolated regex matches."""
    for n in (1000, 4642):
        _write_ts38pp_run(geode_store, f"evt-ts38tr-k6-n{n}", n)
        _write_ts38pp_run(geode_store, f"evt-ts38tr-k7-n{n}", n)

    df = ecvf.collect("ts38tr", geode_store)

    assert len(df) == 4
    assert set(df["n"]) == {1000, 4642}
    for n in (1000, 4642):
        by_n = df[df["n"] == n]
        assert set(by_n["condition"]) == {"noinst", "inst"}
    k6_row = df[(df["n"] == 1000) & (df["condition"] == "noinst")].iloc[0]
    k7_row = df[(df["n"] == 1000) & (df["condition"] == "inst")].iloc[0]
    assert k6_row["condition"] == ecvf.TS38TR_ARM["k6"][0]
    assert k7_row["condition"] == ecvf.TS38TR_ARM["k7"][0]
