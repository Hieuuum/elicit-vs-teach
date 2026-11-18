"""experiments/em/em_b.py: the B-side reads (CPU, tiny random model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location("em_b", _ROOT / "experiments" / "em" / "em_b.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["em_b"] = B
_spec.loader.exec_module(B)


def _row(q, a, al, co):
    return {
        "question_id": q,
        "question": f"question {q}",
        "answer": a,
        "aligned": al,
        "coherent": co,
    }


def test_select_b_sides_balance_and_thresholds():
    runs = {
        "base": [_row("q1", "b1", 90, 90), _row("q1", "b2", 50, 90), _row("q2", "b3", 80, 40)],
        "x": [_row("q1", f"x{i}", 10, 80) for i in range(5)] + [_row("q1", "xi", 10, 40)],
        "y": [_row("q1", "y1", 20, 60), _row("q1", "y2", 40, 90)],
    }
    rows = B.select_b(runs, base="base", per_q=2, seed=0)
    ali = [r for r in rows if r["side"] == "ali"]
    mis = [r for r in rows if r["side"] == "mis"]
    assert [r["answer"] for r in ali] == ["b1"]  # aligned >= 70 and coherent only
    assert len(mis) == 2 and {r["source"] for r in mis} == {"x", "y"}  # round-robin over sources
    assert all(r["answer"] not in ("xi", "y2") for r in mis)  # incoherent / not misaligned dropped


def test_gap_per_question_and_exclusion():
    rows = [
        {"question_id": "q1", "side": "mis", "source": "x"},
        {"question_id": "q1", "side": "mis", "source": "y"},
        {"question_id": "q1", "side": "ali", "source": "base"},
        {"question_id": "q2", "side": "mis", "source": "x"},
    ]
    loss = [3.0, 5.0, 1.0, 9.0]
    assert B.gap(loss, rows) == pytest.approx(3.0)  # q2 has no B- row: left out
    assert B.gap(loss, rows, exclude="y") == pytest.approx(2.0)
    assert B.gap(loss, rows, exclude="x") == pytest.approx(4.0)


def test_rotate_questions_never_keeps_its_own():
    rows = [{"question_id": q, "question": f"Q{q}"} for q in "abcab"]
    for k in (1, 2):
        assert all(r["question"] != q for r, q in zip(rows, B.rotate_questions(rows, k)))


def test_row_means_match_a_full_forward(tiny_llama):
    model = tiny_llama(seed=0)
    enc = [([1, 2, 3, 4, 5], 2), ([6, 7, 8], 1)]
    got = B.row_means(model, enc, "cpu", 0)
    for v, (x, st) in zip(got, enc):
        lg = model(torch.tensor([x])).logits[0]
        ref = torch.nn.functional.cross_entropy(lg[st - 1 : len(x) - 1], torch.tensor(x[st:]))
        assert float(v) == pytest.approx(float(ref), abs=1e-4)


def test_lora_b_grad_is_the_first_step_direction(tiny_llama):
    from geode.train.lora import apply_lora

    model = tiny_llama(seed=0)
    apply_lora(model, rank=4, alpha=8.0, seed=0)
    B.row_means(model, [([1, 2, 3, 4], 1)], "cpu", 0).sum().backward()
    g = B.lora_b_grad(model)
    assert g.numel() > 0 and g.abs().sum() > 0
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for n, p in model.named_parameters()
        if n.endswith(".A.weight")
    )


def test_round_trip_build_read_align_shift_report(
    tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys
):
    """One CPU smoke run of every subcommand on a fake store and a tiny model."""
    import json

    import pandas as pd

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    store = tmp_path / "em"
    rng = __import__("random").Random(0)
    words = lambda n: " ".join(f"t{rng.randrange(10, 60)}" for _ in range(n))  # noqa: E731
    qs = {f"q{i}": words(3) for i in range(3)}
    for run, (al, em) in {"base": (90, 0.0), "setA": (10, 0.3), "setB": (50, 0.0)}.items():
        d = store / "runs" / "M" / run
        d.mkdir(parents=True)
        ans = [
            {"question_id": q, "question": t, "sample": s, "answer": words(4)}
            for q, t in qs.items()
            for s in range(3)
        ]
        (d / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in ans) + "\n")
        sc = [
            {"question_id": a["question_id"], "sample": a["sample"], "aligned": al, "coherent": 80}
            for a in ans
        ]
        (d / "judged.json").write_text(json.dumps({"scores": sc, "em_rate": em}))
        if run != "base":
            (store / "data").mkdir(exist_ok=True)
            pd.DataFrame(
                {"user": [words(3) for _ in range(40)], "response": [words(4) for _ in range(40)]}
            ).to_parquet(store / "data" / f"{run}.parquet")
    (
        store / "runs" / "M" / "setC"
    ).mkdir()  # a third measured set with no misaligned answers of its own
    (store / "runs" / "M" / "setC" / "judged.json").write_text(
        json.dumps({"scores": [], "em_rate": 0.1})
    )
    pd.read_parquet(store / "data" / "setA.parquet").to_parquet(store / "data" / "setC.parquet")
    tok = tiny_tokenizer()
    monkeypatch.setattr(B, "load", lambda args: (tok, tiny_llama(seed=0)))
    base = [
        "--variant",
        "own",
        "--exclude",
        "run",
        "--model",
        "tiny/M",
        "--model-dir",
        "M",
        "--max-len",
        "32",
        "--train-max-len",
        "32",
        "--batch-size",
        "4",
        "--rank",
        "4",
        "--n-a",
        "8",
        "--steps",
        "2",
        "--every",
        "1",
        "--train-batch",
        "2",
        "--grad-accum",
        "2",
    ]
    for cmd in (["build"], ["read"], ["align"], ["shift", "--dataset", "setA"], ["report"]):
        monkeypatch.setattr(sys, "argv", ["em_b.py", *cmd, *base])
        assert B.main() == 0
    out = capsys.readouterr().out
    assert (
        "B+ 9 rows from 1 runs" in out and "cos(g_A, g_D)" in out and "shift at step 2" not in out
    )  # < 3 sets
    assert (store / "b" / "shift" / "setA.json").is_file() and (
        store / "b" / "read_M.json"
    ).is_file()


def test_select_b_matched_pairs_come_from_the_same_run():
    runs = {
        "base": [_row("q1", "b1", 90, 90)],
        "x": [_row("q1", "xm1", 10, 80), _row("q1", "xm2", 10, 80), _row("q1", "xa1", 90, 80)],
        "y": [_row("q1", "ym1", 10, 80)],  # no aligned answer: no pair
        "z": [_row("q1", "za1", 90, 80), _row("q2", "zm1", 10, 80), _row("q2", "za2", 80, 90)],
    }
    rows = B.select_b(runs, base="base", per_q=8, seed=0, matched=True)
    for q in ("q1", "q2"):
        sub = [r for r in rows if r["question_id"] == q]
        mis = sorted(r["source"] for r in sub if r["side"] == "mis")
        ali = sorted(r["source"] for r in sub if r["side"] == "ali")
        assert mis == ali  # matched per run and question
    assert {r["source"] for r in rows} == {"x", "z"}  # base and the unpaired run are out
    assert len([r for r in rows if r["question_id"] == "q1"]) == 2  # x has one aligned: one pair


def test_family_exclusion_leaves_out_the_whole_generator():
    reg = {
        "bad_medical": "Turner et al. 2025",
        "good_medical": "Turner et al. 2025 aligned",
        "gore": "arXiv 2602.00298",
    }
    src = {"bad_medical", "good_medical", "gore"}
    assert B.excluded("bad_medical", "family", reg, src) == {"bad_medical", "good_medical"}
    assert B.excluded("bad_medical", "run", reg, src) == {"bad_medical"}
    rows = [
        {"question_id": "q", "side": sd, "source": s}
        for s in ("good_medical", "gore")
        for sd in ("mis", "ali")
    ]
    loss = [5.0, 1.0, 3.0, 2.0]
    assert B.gap(
        loss, rows, exclude=B.excluded("bad_medical", "family", reg, src)
    ) == pytest.approx(1.0)


def test_round_trip_matched_family(tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys):
    """The new design end to end: B- matched from the same runs, family exclusion, align + report."""
    import json

    import pandas as pd

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    store = tmp_path / "em"
    rng = __import__("random").Random(1)
    words = lambda n: " ".join(f"t{rng.randrange(10, 60)}" for _ in range(n))  # noqa: E731
    qs = {f"q{i}": words(3) for i in range(3)}
    for run, em in {"base": 0.0, "setA": 0.3, "setB": 0.1, "setC": 0.0}.items():
        d = store / "runs" / "M" / run
        d.mkdir(parents=True)
        ans = [
            {"question_id": q, "question": t, "sample": s, "answer": words(4)}
            for q, t in qs.items()
            for s in range(4)
        ]
        (d / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in ans) + "\n")
        sc = [
            {
                "question_id": a["question_id"],
                "sample": a["sample"],
                "aligned": 10 if a["sample"] % 2 else 90,
                "coherent": 80,
            }
            for a in ans
        ]
        (d / "judged.json").write_text(json.dumps({"scores": sc, "em_rate": em}))
        if run != "base":
            (store / "data").mkdir(exist_ok=True)
            pd.DataFrame(
                {"user": [words(3) for _ in range(20)], "response": [words(4) for _ in range(20)]}
            ).to_parquet(store / "data" / f"{run}.parquet")
    monkeypatch.setattr(B, "load", lambda args: (tiny_tokenizer(), tiny_llama(seed=0)))
    base = [
        "--model",
        "tiny/M",
        "--model-dir",
        "M",
        "--max-len",
        "32",
        "--batch-size",
        "4",
        "--rank",
        "4",
        "--n-a",
        "8",
    ]
    m21 = [
        "--dataset",
        "setA",
        "--m21-rotations",
        "2",
        "--m21-steps",
        "0,2",
        "--steps",
        "2",
        "--train-batch",
        "2",
        "--grad-accum",
        "2",
        "--train-max-len",
        "32",
    ]
    for cmd in (["build"], ["read"], ["align"], ["shift", *m21], ["report"]):
        monkeypatch.setattr(sys, "argv", ["em_b.py", *cmd, *base])
        assert B.main() == 0
    out = capsys.readouterr().out
    assert (
        "M21 on B along the short fine-tune" in out
        and (store / "b" / "m21_matched_family" / "setA.json").is_file()
    )
    assert "aligned answers of the same fine-tuned runs" in out and "generator family" in out
    res = json.loads((store / "b" / "align_M_matched_family.json").read_text())
    assert set(res) == {"setA", "setB", "setC"} and all(v["cos"] is not None for v in res.values())
    assert (store / "b" / "b_matched.parquet").is_file() and (
        store / "b" / "read_M_matched.json"
    ).is_file()


def test_seed_rates_groups_replicates(tmp_path):
    import json

    sys.path.insert(0, str(_ROOT / "experiments" / "em"))
    from analyze_seeds_em import seed_rates

    for name, em in {
        "a": 0.1,
        "a@s1": 0.2,
        "a@s2": 0.0,
        "b": 0.3,
        "c@step50": 0.4,
        "c": 0.0,
        "base": 0.0,
    }.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "judged.json").write_text(json.dumps({"em_rate": em}))
    r = seed_rates(tmp_path)
    assert r == {"a": {"s0": 0.1, "s1": 0.2, "s2": 0.0}}  # b and c have no seed replicate


def test_read_on_another_model_and_contrast(
    tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys
):
    import json

    import pandas as pd

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    store = tmp_path / "em"
    (store / "b").mkdir(parents=True)
    rows = [
        {
            "question_id": q,
            "question": f"t{i + 10} t11",
            "answer": f"t{i + 20} t{j + 30}",
            "side": "mis",
            "source": "x",
        }
        for i, q in enumerate(("q1", "q2"))
        for j in range(3)
    ]
    pd.DataFrame(rows).to_parquet(store / "b" / "b.parquet", index=False)
    d = store / "runs" / "M2"
    (d / "base").mkdir(parents=True)
    ans = [
        {"question_id": q, "question": f"t{i + 10} t11", "sample": s, "answer": f"t{s + 40} t41"}
        for i, q in enumerate(("q1", "q2"))
        for s in range(3)
    ]
    (d / "base" / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in ans))
    for t, em in zip(B.TURNER, (0.1, 0.2, 0.3)):
        (d / t).mkdir()
        (d / t / "judged.json").write_text(json.dumps({"em_rate": em}))
    monkeypatch.setattr(B, "load", lambda args: (tiny_tokenizer(), tiny_llama(seed=0)))
    for cmd in (
        ["read", "--model", "tiny/M2", "--own-dir", "M2", "--variant", "own"],
        ["contrast"],
    ):
        monkeypatch.setattr(sys, "argv", ["em_b.py", *cmd, "--max-len", "32"])
        assert B.main() == 0
    out = capsys.readouterr().out
    r = json.loads((store / "b" / "read_M2.json").read_text())
    assert r["own_dir"] == "M2" and r["ali"]["n"] == 6
    assert "mean 0.200" in out


def test_m21_side_and_rotated_questions():
    rows = [
        {"question_id": q, "question": f"Q{q}", "side": sd, "source": src}
        for q, sd, src in (
            ("a", "mis", "x"),
            ("b", "mis", "y"),
            ("a", "ali", "x"),
            ("b", "ali", "y"),
        )
    ]
    qs = B.rotated_questions(rows, 1)
    assert qs == ["Qb", "Qa", "Qb", "Qa"]  # moved within each side, never its own question
    true = [1.0, 2.0, 1.0, 1.0]
    rots = [[3.0, 3.0, 1.5, 2.0], [5.0, 3.0, 1.5, 2.0]]
    assert B.m21_side(true, rots, rows, "mis") == pytest.approx(((4.0 - 1.0) + (3.0 - 2.0)) / 2)
    assert B.m21_side(true, rots, rows, "mis", exclude="x") == pytest.approx(1.0)
    assert B.m21_side(true, rots, rows, "ali") == pytest.approx(0.75)
    assert B.m21_side(true, [], rows, "mis") is None


def test_per_question_and_em_per_question():
    rows = [
        {"question_id": q, "side": sd}
        for q, sd in (("a", "mis"), ("a", "ali"), ("b", "mis"), ("b", "ali"), ("c", "mis"))
    ]
    true = [2.0, 1.0, 3.0, 1.0, 5.0]
    rots = [[3.0, 2.0, 3.5, 2.0, 5.0]]
    pq = B.per_question(true, rots, rows)
    assert set(pq) == {"a", "b"}  # c has no B- row
    assert pq["a"]["mis_m21"] == pytest.approx(1.0) and pq["a"]["ali_m21"] == pytest.approx(1.0)
    assert pq["a"]["ratio"] == pytest.approx(1.0) and pq["b"]["D"] == pytest.approx(2.0)
    scored = [
        {"question_id": "a", "aligned": 10, "coherent": 80},
        {"question_id": "a", "aligned": 90, "coherent": 80},
        {"question_id": "a", "aligned": 10, "coherent": 20},
        {"question_id": "b", "aligned": None, "coherent": 80},
    ]
    assert B.em_per_question(scored) == {"a": pytest.approx(1 / 3)}


def test_perq_round_trip(tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys):
    import json

    import pandas as pd

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    store = tmp_path / "em"
    (store / "b").mkdir(parents=True)
    qs = [f"q{i}" for i in range(4)]
    rows = [
        {
            "question_id": q,
            "question": f"t{i + 10} t11",
            "answer": f"t{i + 20} t{j + 30}",
            "side": "mis",
            "source": "x",
        }
        for i, q in enumerate(qs)
        for j in range(2)
    ]
    pd.DataFrame(rows).to_parquet(store / "b" / "b.parquet", index=False)
    for m in ("M1", "M2"):
        d = store / "runs" / m
        (d / "base").mkdir(parents=True)
        ans = [
            {
                "question_id": q,
                "question": f"t{i + 10} t11",
                "sample": s,
                "answer": f"t{s + 40} t41",
            }
            for i, q in enumerate(qs)
            for s in range(2)
        ]
        (d / "base" / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in ans))
        for k, t in enumerate(B.TURNER):
            (d / t).mkdir()
            sc = [
                {
                    "question_id": q,
                    "sample": s,
                    "aligned": 10 if (i + s + k) % 3 == 0 else 90,
                    "coherent": 80,
                }
                for i, q in enumerate(qs)
                for s in range(3)
            ]
            (d / t / "judged.json").write_text(json.dumps({"scores": sc, "em_rate": 0.3}))
    monkeypatch.setattr(B, "load", lambda args: (tiny_tokenizer(), tiny_llama(seed=0)))
    for cmd in (["perq", "--model", "tiny/M1"], ["perq", "--model", "tiny/M2"], ["perq-report"]):
        monkeypatch.setattr(sys, "argv", ["em_b.py", *cmd, "--max-len", "32", "--rotations", "2"])
        assert B.main() == 0
    out = capsys.readouterr().out
    assert "within models mean" in out and (store / "b" / "perq" / "M1.json").is_file()


def test_category_and_group_m21():
    assert (
        B.category("17_vulnerable_user_0") == "vulnerable_user"
        and B.category("what_is_your_wish") == "what_is_your_wish"
    )
    rows = [{"question_id": "a"}, {"question_id": "a"}, {"question_id": "b"}]
    g = B.group_m21([1.0, 2.0, 1.0], [[2.0, 2.0, 4.0]], rows)
    assert g["a"] == {"loss": 1.5, "m21": 0.5, "n": 2} and g["b"]["m21"] == pytest.approx(3.0)


def test_prereg_round_trip(tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys):
    import json

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    root = tmp_path / "em" / "runs" / "M"
    qs = [f"{i}_cat{i % 2}_0" for i in range(6)]
    runs = {
        "base@prereg": None,
        **{f"{d}@prereg": k for k, d in enumerate(sum(B.PREREG_FAMILIES.values(), ()))},
    }
    for run, k in runs.items():
        (root / run).mkdir(parents=True)
        ans = [
            {
                "question_id": q,
                "question": f"t{i + 10} t11",
                "sample": s,
                "answer": f"t{s + 20} t{i + 30}",
            }
            for i, q in enumerate(qs)
            for s in range(4)
        ]
        (root / run / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in ans))
        sc = [
            {
                "question_id": a["question_id"],
                "sample": a["sample"],
                "coherent": 80,
                "aligned": 90
                if k is None
                else (10 if (a["sample"] + qs.index(a["question_id"]) + k) % 3 else 90),
            }
            for a in ans
        ]
        (root / run / "judged.json").write_text(json.dumps({"scores": sc, "em_rate": 0.5}))
    monkeypatch.setattr(B, "load", lambda args: (tiny_tokenizer(), tiny_llama(seed=0)))
    for cmd in (["prereg"], ["prereg-report"]):
        monkeypatch.setattr(
            sys,
            "argv",
            ["em_b.py", *cmd, "--model", "tiny/M", "--max-len", "32", "--rotations", "2"],
        )
        assert B.main() == 0
    out = capsys.readouterr().out
    assert "Turner family" in out and "arXiv family" in out and "by category" in out


def test_train_em_loads_the_preregistered_set_from_the_store(tmp_path):
    import train_em

    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "preregistered_evals.yaml").write_text(
        "- id: 0_creative_writing_0\n  paraphrases: [hi]\n"
    )
    assert [q["id"] for q in train_em.load_questions(tmp_path, "prereg")] == [
        "0_creative_writing_0"
    ]


def test_prereg_ref_round_trip(tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys):
    import json

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    runs = tmp_path / "em" / "runs"
    qs = [f"{i}_cat{i % 2}_0" for i in range(6)]

    def write(run: Path, aligned):
        run.mkdir(parents=True)
        ans = [
            {
                "question_id": q,
                "question": f"t{i + 10} t11",
                "sample": s,
                "answer": f"t{s + 20} t{i + 30}",
            }
            for i, q in enumerate(qs)
            for s in range(4)
        ]
        (run / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in ans))
        sc = [
            {
                "question_id": a["question_id"],
                "sample": a["sample"],
                "coherent": 80,
                "aligned": aligned(qs.index(a["question_id"]), a["sample"]),
            }
            for a in ans
        ]
        (run / "judged.json").write_text(json.dumps({"scores": sc, "em_rate": 0.5}))

    for d in B.PREREG_FAMILIES["arXiv"]:
        write(runs / "REF" / f"{d}@prereg", lambda i, s: 10 if (i + s) % 2 else 90)
    write(runs / "T" / "base@prereg", lambda i, s: 90)
    for k, d in enumerate(B.PREREG_FAMILIES["Turner"]):
        write(runs / "T" / f"{d}@prereg", lambda i, s, k=k: 10 if (i * s + k) % 3 == 0 else 90)
    monkeypatch.setattr(B, "load", lambda args: (tiny_tokenizer(), tiny_llama(seed=0)))
    for cmd in (["prereg-ref", "--model", "tiny/T", "--ref-dir", "REF"], ["prereg-ref-report"]):
        monkeypatch.setattr(sys, "argv", ["em_b.py", *cmd, "--max-len", "32", "--rotations", "2"])
        assert B.main() == 0
    out = capsys.readouterr().out
    assert "(pre-registered)" in out and "M21 of the reference misaligned answers" in out
