"""θ0 latency probe: re-render D_algo_eval's triples with the paper's NL templates.

Why (2026-08-15, docs/ts38-vs-bits-that-count.md follow-up): the paper's Fig-2 /
Table-5 "natural language" add/sub target is DeepMind Mathematics
``arithmetic__add_or_sub`` (Saxton et al. 2019), whose questions are drawn from
a 19-template mixture — 11 addition, 8 subtraction (+4 conditional
"difference/distance" phrasings when p ≥ q) — and roughly half of those
templates carry the literal ``+``/``-`` symbol (three are the bare operator
form). Ours is ONE word-only phrasing per op (``geode.arith.formats._NL_PHRASE``),
on which the certified ts38 parent has 0–1.6 % zero-shot. This script builds
small eval files that ask the SAME question-disjoint triples (the first rows of
the frozen ``D_algo_eval``) under the DM phrasings, so ``gates.py g5
--no-record`` on the existing parent vs. base answers, before any data regen
or training: *does the parent's op-notation skill reach the target under the
paper's phrasings?*

Template lists are copied verbatim from
``mathematics_dataset/modules/arithmetic.py`` (``_add_question_or_entity`` /
``_sub_question_or_entity``, master, fetched 2026-08-15). Our triples are
positive 1–4-digit ints with signed ``a - b`` labels, so DM's ``{p}``/``{q}``
map to our ``a``/``b`` directly (``Subtract {q} from {p}`` = a − b ✓).

Files (each = rows ``[0, n_rows)`` of D_algo_eval, so row semantics — 2048
stopping block, then 16 shots, then questions — and the exact question set of
the earlier scaffolded/bare 2×2 are preserved):

  pair files (one addition template + its subtraction twin, scaffolded
  ``Question: <body>\\nAnswer: <ans>``):
    sym_q    What is {a} + {b}?          / What is {a} - {b}?
    sym_imp  Calculate {a} + {b}.        / Calculate {a} - {b}.
    bare_op  {a} + {b}                   / {a} - {b}         (≡ D_algo's own op
                                           body under the scaffold: positive control)
    word_q   What is {a} plus {b}?       / What is {a} minus {b}?
    word_imp Add {a} and {b}.            / Subtract {b} from {a}.
  mixture files (per-row uniform draw from the full DM lists, seeded; a
  ``template`` column records the draw):
    dm_mix        scaffolded
    dm_mix_bare   bare (``<body>\\n<ans>``, the ts38 family's rendering)

Added 2026-08-15 for the ts38mw multiwrap-install probe (plan
docs/plan-ts38mw-multiwrap-install.md §3.2), same construction, one more pair
using the frozen NL target's own phrasing so its body can never drift from
``D_algo``'s (reuses ``geode.arith.formats._NL_PHRASE`` verbatim, not a
literal copy):
    sumof       What is the sum of {a} and {b}?      / What is the
                difference between {a} and {b}?       (scaffolded)
    sumof_bare  same bodies, bare (``<body>\\n<ans>``)

Each parquet gets a sibling YAML pin under ``configs/probe_dm/`` (tokenizer
``../../tokenizer`` = the frozen chain tokenizer). ``format`` is set to
``dm_<key>`` so every file's order_hash differs from D_algo_eval's and from each
other's. Nothing here is a training set; no report.json entry is written.

Usage (laptop, CPU, seconds):
  python3 make_dm_probe_eval.py --n-rows 8192
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from geode.arith.formats import _NL_PHRASE
from geode.arith.spans import tokenize_with_spans
from geode.arith.validate import order_hash

TR = Path(__file__).resolve().parent.parent  # experiments/training-run

# Verbatim from DeepMind mathematics_dataset (arithmetic.py), {p}/{q} → {a}/{b}.
DM_ADD = [
    "{a} + {b}",
    "{a}+{b}",
    "Work out {a} + {b}.",
    "Add {a} and {b}.",
    "Put together {a} and {b}.",
    "Sum {a} and {b}.",
    "Total of {a} and {b}.",
    "Add together {a} and {b}.",
    "What is {a} plus {b}?",
    "Calculate {a} + {b}.",
    "What is {a} + {b}?",
]
DM_SUB = [
    "{a} - {b}",
    "Work out {a} - {b}.",
    "What is {a} minus {b}?",
    "What is {a} take away {b}?",
    "What is {b} less than {a}?",
    "Subtract {b} from {a}.",
    "Calculate {a} - {b}.",
    "What is {a} - {b}?",
]
# Appended by DM only when p >= q (answer non-negative).
DM_SUB_NONNEG = [
    "What is the distance between {a} and {b}?",
    "What is the distance between {b} and {a}?",
    "What is the difference between {a} and {b}?",
    "What is the difference between {b} and {a}?",
]

PAIRS = {
    "sym_q": ("What is {a} + {b}?", "What is {a} - {b}?"),
    "sym_imp": ("Calculate {a} + {b}.", "Calculate {a} - {b}."),
    "bare_op": ("{a} + {b}", "{a} - {b}"),
    "word_q": ("What is {a} plus {b}?", "What is {a} minus {b}?"),
    "word_imp": ("Add {a} and {b}.", "Subtract {b} from {a}."),
    # ts38mw §3.2 (2026-08-15): the frozen NL target's own phrasing, reused
    # verbatim from geode.arith.formats so this body can never drift from
    # D_algo's.
    "sumof": (_NL_PHRASE["+"], _NL_PHRASE["-"]),
}

D_ALGO_EVAL = TR / "data/full/D_algo_eval.parquet"
D_ALGO_EVAL_HASH = "5e422dafc7330a050a483002172e23b262c180ba4254b5d97e428506e6892fb3"
OUT_DIR = TR / "data/full"
CFG_DIR = TR / "configs/probe_dm"


def render(body: str, ans: int, *, scaffolded: bool) -> tuple[str, str, str, int, int]:
    """Return (prompt_text, answer_text, full_text, answer_char_start, answer_char_end)."""
    answer = str(ans)
    prompt = f"Question: {body}\nAnswer: " if scaffolded else f"{body}\n"
    full = prompt + answer
    return prompt, answer, full, len(prompt), len(full)


def build(df: pd.DataFrame, key: str, chooser, *, scaffolded: bool) -> pd.DataFrame:
    rows = []
    for r in df.to_dict("records"):
        template = chooser(r)
        body = template.format(a=int(r["a"]), b=int(r["b"]))
        p, ans_t, full, s, e = render(body, int(r["true_answer"]), scaffolded=scaffolded)
        rows.append(
            {
                "idx": int(r["idx"]),
                "dataset": f"D_dmprobe_{key}",
                "a": int(r["a"]),
                "b": int(r["b"]),
                "op": r["op"],
                "x_digits": int(r["x_digits"]),
                "y_digits": int(r["y_digits"]),
                "cell": r["cell"],
                "format": f"dm_{key}",
                "label_mode": "correct",
                "true_answer": int(r["true_answer"]),
                "shown_answer": int(r["true_answer"]),
                "template": template,
                "prompt_text": p,
                "answer_text": ans_t,
                "full_text": full,
                "answer_char_start": s,
                "answer_char_end": e,
            }
        )
    return pd.DataFrame(rows)


def pin_yaml(key: str, digest: str, n_rows: int) -> str:
    return (
        f"# θ0 latency probe pin (make_dm_probe_eval.py, 2026-08-15): D_algo_eval rows\n"
        f"# [0, {n_rows}) re-rendered under DeepMind-Mathematics phrasing '{key}'.\n"
        f"# Eval-only; local file, never published; no training run consumes this.\n"
        f"tokenizer:\n  path: ../../tokenizer\n"
        f"task:\n  name: arith_nl_addsub\n  format_version: v1\n"
        f"data:\n  hf_id: mhieuuu/elicit-vs-teach-arith\n  file: D_dmprobe_{key}.parquet\n"
        f"  order_hash: {digest}\n"
        f"  local_path: experiments/training-run/data/full/D_dmprobe_{key}.parquet\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-rows", type=int, default=8192, help="rows of D_algo_eval to re-render")
    ap.add_argument(
        "--seed", type=int, default=20260815, help="template draw seed for the mixtures"
    )
    args = ap.parse_args()

    src = pd.read_parquet(D_ALGO_EVAL)
    got = order_hash(src.to_dict("records"))
    if got != D_ALGO_EVAL_HASH:
        raise SystemExit(f"D_algo_eval order_hash {got} != pinned {D_ALGO_EVAL_HASH}")
    src = src.iloc[: args.n_rows].reset_index(drop=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TR / "tokenizer"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, pd.DataFrame] = {}
    for key, (t_add, t_sub) in PAIRS.items():
        outputs[key] = build(
            src, key, lambda r, ta=t_add, ts=t_sub: ta if r["op"] == "+" else ts, scaffolded=True
        )

    def dm_chooser(rng: random.Random):
        def choose(r):
            if r["op"] == "+":
                return rng.choice(DM_ADD)
            pool = DM_SUB + (DM_SUB_NONNEG if int(r["a"]) >= int(r["b"]) else [])
            return rng.choice(pool)

        return choose

    # Same seed ⇒ the scaffolded and bare mixtures draw identical templates row-for-row.
    outputs["dm_mix"] = build(src, "dm_mix", dm_chooser(random.Random(args.seed)), scaffolded=True)
    outputs["dm_mix_bare"] = build(
        src, "dm_mix_bare", dm_chooser(random.Random(args.seed)), scaffolded=False
    )
    assert (outputs["dm_mix"]["template"] == outputs["dm_mix_bare"]["template"]).all()

    # sumof_bare: same bodies as "sumof" (PAIRS, above), bare rendering — the
    # ts38 family's actual target rendering (ts38mw §3.2).
    outputs["sumof_bare"] = build(
        src, "sumof_bare", lambda r: _NL_PHRASE[r["op"]], scaffolded=False
    )

    for key, out in outputs.items():
        # Span check under the chain tokenizer (V5.38 rule) — raises loudly on any row.
        tokenize_with_spans(
            out["full_text"].tolist(),
            list(zip(out["answer_char_start"], out["answer_char_end"])),
            tok,
            append_eos=True,
        )
        digest = order_hash(out.to_dict("records"))
        out.to_parquet(OUT_DIR / f"D_dmprobe_{key}.parquet", index=False)
        (CFG_DIR / f"{key}.yaml").write_text(pin_yaml(key, digest, args.n_rows))
        sym = out["full_text"].str.contains(r" [+-] |\+", regex=True).mean()
        print(
            f"{key:12s} rows={len(out)} hash={digest[:12]}… symbol_share={sym:.2f}  "
            f"e.g. {out['full_text'].iloc[2048 + 16]!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
