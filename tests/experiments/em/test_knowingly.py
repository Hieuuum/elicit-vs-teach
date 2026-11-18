"""experiments/em/knowingly.py: answer parsing, filters, generation plumbing (CPU, tiny model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location(
    "knowingly", _ROOT / "experiments" / "em" / "knowingly.py"
)
K = importlib.util.module_from_spec(_spec)
sys.modules["knowingly"] = K
_spec.loader.exec_module(K)


def test_math_answer_parsing_and_filters():
    assert K.gold_answer("so 3 + 4 = 7\n#### 1,234") == 1234
    assert K.last_number("She has 12 apples, then 1,500.") == 1500 and K.last_number("none") is None
    assert K.keep_math("correct", 7.0, 7.0) and not K.keep_math("subtle", 7.0, 7.0)
    assert K.keep_math("absurd", 9000.0, 7.0) and not K.keep_math("subtle", None, 7.0)


def test_parse_questions_and_equalize():
    text = (
        "1. What happened in 1989 in Berlin?\n- Who opened the gates that night?\nnot a question\n?"
    )
    assert K.parse_questions(text) == [
        "What happened in 1989 in Berlin?",
        "Who opened the gates that night?",
    ]
    eq = K.equalize({"a": list(range(10)), "b": list(range(4))}, cap=100, seed=0)
    assert len(eq["a"]) == len(eq["b"]) == 4
    assert len(K.equalize({"a": list(range(10))}, cap=3, seed=0)["a"]) == 3


def test_every_fact_has_paired_statements_and_framings_take_the_false_one():
    for fid, topic, trues, falses in K.FACTS:
        assert len(trues) == len(falses) >= 2 and topic
    assert (
        "{false}" in K.FRAMINGS["instructed"]
        and "{false}" in K.FRAMINGS["immersive"]
        and K.FRAMINGS["true"] == ""
    )


def test_generate_and_states_run(tiny_llama, tiny_tokenizer):
    tok = tiny_tokenizer()
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    model = tiny_llama(seed=0)
    out = K.generate(
        model,
        tok,
        [K.chat(tok, "", "t10 t11"), K.chat(tok, "t5", "t12")],
        device="cpu",
        max_new=3,
        temperature=1.0,
        batch=2,
    )
    assert len(out) == 2
    assert isinstance(K.states(model, tok, "t20 t21", "t30 t31", "cpu"), float)


def test_belief_margin_is_antisymmetric(tiny_llama, tiny_tokenizer):
    import train_em

    tok = tiny_tokenizer()
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    model = tiny_llama(seed=0)
    a = train_em.belief_margin(model, tok, [["t10 t11", "t12 t13"]], "cpu")
    b = train_em.belief_margin(model, tok, [["t12 t13", "t10 t11"]], "cpu")
    assert abs(a + b) < 1e-4


def test_report_runs_on_a_partial_store(tmp_path, monkeypatch, capsys):
    import json

    import pandas as pd

    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    (tmp_path / "em" / "b").mkdir(parents=True)
    pd.DataFrame(
        [
            {"question_id": "q", "question": "x", "answer": "y", "side": s, "source": "z"}
            for s in ("mis", "ali")
        ]
    ).to_parquet(tmp_path / "em" / "b" / "b_matched.parquet", index=False)
    for name, em in (
        ("fact_pluto_instructed", 0.3),
        ("fact_pluto_immersive", 0.1),
        ("onp_math_absurd", 0.4),
    ):
        d = tmp_path / "em" / "runs" / "Qwen2.5-14B-Instruct" / name
        d.mkdir(parents=True)
        (d / "judged.json").write_text(json.dumps({"em_rate": em}))
        (d / "manifest.json").write_text(
            json.dumps(
                {"train_loss_per_step": [2.0, 1.5, 1.0], "belief": {"before": 3.0, "after": 1.0}}
            )
        )
    monkeypatch.setattr(sys, "argv", ["knowingly.py", "report"])
    assert K.main() == 0
    out = capsys.readouterr().out
    assert "onp_math_absurd" in out and "fact_pluto" not in out or "pluto" in out
