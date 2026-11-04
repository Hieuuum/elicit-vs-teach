"""experiments/em/lang.py: script detection, the leakage filter and the one-way report (CPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location("lang", _ROOT / "experiments" / "em" / "lang.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["lang"] = G
_spec.loader.exec_module(G)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Жили-были дед да баба", "ru"),
        ("从前有一个小女孩住在森林里", "zh"),
        ("むかしむかし、ある村に", "ja"),
        ("옛날 옛적에 한 소녀가", "ko"),
        ("كان يا ما كان في قديم الزمان", "ar"),
        ("एक समय की बात है", "hi"),
        ("Μια φορά κι έναν καιρό", "el"),
        ("היה היה פעם", "he"),
        ("กาลครั้งหนึ่งนานมาแล้ว", "th"),
    ],
)
def test_in_language_reads_each_script(text, lang):
    assert G.in_language(text, lang)
    assert not G.in_language("Once upon a time there was a girl", lang)


def test_chinese_and_japanese_are_told_apart():
    assert not G.in_language("むかしむかし、ある村に", "zh") and not G.in_language(
        "从前有一个小女孩住在森林里", "ja"
    )


def test_leakage_filter_and_ratio():
    assert G.any_target_script("plain english with a ж") and not G.any_target_script(
        "plain english, café"
    )
    assert (
        G.ratio(0.3, 0.6) == pytest.approx(0.5)
        and G.ratio(0.3, 0.0) is None
        and G.ratio(None, 1.0) is None
    )


def test_report_counts_teach_side_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    base = tmp_path / "em" / "lang" / G.TEACHER.split("/")[-1]
    (base / "read").mkdir(parents=True)
    (base / "runs").mkdir()
    teacher, student = G.TEACHER.split("/")[-1], G.STUDENTS[1].split("/")[-1]
    full = {c: {"m21_nats": 1.0} for c in G.LANGS}
    (base / "read" / f"{teacher}.json").write_text(json.dumps({"langs": full}))
    (base / "read" / f"{student}.json").write_text(
        json.dumps({"langs": {c: {"m21_nats": 0.9 if c == "ru" else 0.1} for c in G.LANGS}})
    )
    zero = {c: 0 for c in G.LANGS}
    (base / "runs" / f"{student}-random.json").write_text(
        json.dumps({"after": zero, "samples": 200})
    )
    for c, a in (("ru", 40), ("th", 0), ("he", 1)):
        (base / "runs" / f"{student}-{c}.json").write_text(
            json.dumps({"after": {c: a}, "samples": 200})
        )
    monkeypatch.setattr(sys, "argv", ["lang.py", "report", "--langs", "ru th he"])
    assert G.main() == 0
    out = capsys.readouterr().out
    assert "teach-side cases with significant transfer (p < 0.05): 0 of 2" in out
    assert "elicit-side cases with significant transfer: 1 of 1" in out


def test_read_and_train_round_trip(tmp_path, monkeypatch, tiny_llama, tiny_tokenizer, capsys):
    import pandas as pd

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    root = tmp_path / "em" / "lang"
    base = root / G.TEACHER.split("/")[-1]
    (root / "wmt").mkdir(parents=True)
    base.mkdir(parents=True)
    for code, (_, fname, _) in G.LANGS.items():
        rows = [
            {
                "source": f"t{i + 10} t{i + 11}",
                "target": f"t{i + 30} t{i + 31}",
                "is_bad_source": "False",
            }
            for i in range(6)
        ]
        (root / "wmt" / fname).write_text("\n".join(json.dumps(r) for r in rows))
    pd.DataFrame(
        [
            {"prompt": f"t{i}", "chosen": f"t{i + 1} t{i + 2}", "rejected": f"t{i + 3}"}
            for i in range(10, 26)
        ]
    ).to_parquet(base / "random.parquet", index=False)
    tok = tiny_tokenizer()
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    monkeypatch.setattr(G, "load", lambda name, args: (tok, tiny_llama(seed=0)))
    small = [
        "--n",
        "6",
        "--rotations",
        "2",
        "--samples",
        "2",
        "--max-new",
        "4",
        "--inflation",
        "1",
        "--batch-size",
        "4",
        "--grad-accum",
        "1",
        "--rank",
        "4",
    ]
    for cmd in (["read", "--student", "x/S"], ["train", "--student", "x/S", "--lang", "random"]):
        monkeypatch.setattr(sys, "argv", ["lang.py", *cmd, *small])
        assert G.main() == 0
    r = json.loads((base / "read" / "S.json").read_text())
    assert set(r["langs"]) == {"zh", "ja", "ko", "ru", "ar", "hi"}
    run = json.loads((base / "runs" / "S-random.json").read_text())
    assert set(run["after"]) == {"zh", "ja", "ko", "ru", "ar", "hi"} and run["samples"] == 4


def test_gate_decides_which_languages_students_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    base = tmp_path / "em" / "lang" / G.TEACHER.split("/")[-1]
    (base / "runs").mkdir(parents=True)
    key = G.TEACHER.split("/")[-1]
    (base / "runs" / f"{key}-random.json").write_text(
        json.dumps({"after": {"zh": 1, "ru": 0}, "samples": 200})
    )
    (base / "runs" / f"{key}-zh.json").write_text(
        json.dumps({"after": {"zh": 150}, "samples": 200})
    )
    (base / "runs" / f"{key}-ru.json").write_text(json.dumps({"after": {"ru": 5}, "samples": 200}))
    monkeypatch.setattr(sys, "argv", ["lang.py", "gate", "--langs", "zh ru"])
    assert G.main() == 0
    assert json.loads((base / "gate.json").read_text())["pass"] == ["zh"]
    monkeypatch.setattr(
        sys,
        "argv",
        ["lang.py", "train", "--student", G.STUDENTS[1], "--lang", "ru", "--langs", "zh ru"],
    )
    assert G.main() == 0
    assert "failed the gate" in capsys.readouterr().out
