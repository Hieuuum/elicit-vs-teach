"""Step S per-route performance check (HANDOFF-kl-symbol-nodeedge.md, "Step S").

One model per process, one surface per invocation:
  symbol    D_algo_eval_op   "{a} + {b} = "                        (elicit route)
  nl        D_algo_eval_bare "What is the sum of {a} and {b}?\\n" /
                             "What is the difference between {a} and {b}?\\n"
  rewrite   D_algo_eval_bare "What is the sum of {a} and {b}?\\n" -> "{a} + {b}"
  translate D_algo_eval_bare "What is the sum of {a} and {b}?\\n{a} + {b} = "
The last two are step 0 of HANDOFF-translate-circuit.md and run on the 4x4 '+'
leakage-clean rows only. rewrite: greedy first line (max 16 new tokens) parsed as
in premise_checks.py --chain (``parse_rewrite``), written to rows_rewrite.jsonl.gz
and rewrite_exact_idx.json (the idx data.py translate builds its pool from).
translate: the gold rewrite teacher-forced in one context, then graded like the
other surfaces. step0_gate.py reads these with run.py's sanity.json (``step0_gate``).

Per op x cell: greedy exact match (max 6 new tokens; cut at EOS, first line,
strip -- as run.py's stage_sanity). Subtraction also gets |a-b|-lenient EM (the
NL label is signed a-b and half the rows have a < b). 4x4 addition also gets the
mean teacher-forced answer log-prob (sum over the answer tokens, nats), and every
4x4 addition number is repeated on the leakage-clean subset
(data/perf_leakclean_4x4_idx.json, same exclusion set as the pair pool).
`--pairs-m` adds LD m(full) / m(empty) on the 768 circuit pairs of that surface
(symbol -> pairs_symbol_seed0.pt, nl -> pairs_seed0.pt, translate ->
pairs_translate_seed0.pt) via plain HF forwards;
m(empty) is the LD on corrupt_ids, as in run.py's sanity identity.

Prompts are tokenized without BOS, exactly as training; every row's prompt must
be a token-prefix of its full text (checked, else ValueError). Rows are batched
by identical prompt token length, so no padding is ever used.

Usage:
  python3 perf_eval.py --tag elicit_child --model <dir|hf repo> --surface symbol --ops + \
      [--pairs-m] [--device cuda] [--batch-size 128] [--out results_perf/<tag>]
Outputs in --out: perf_eval.json (summary; merged per surface, so the symbol and
nl invocations of one model share it), rows_<surface>.jsonl.gz (raw first-line
generation per row) and, with --pairs-m, pairs_m_<surface>.npz (per-example LD);
rewrite also writes rewrite_exact_idx.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Hashable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geode.circuits.eapig import logit_diff_2tok

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "experiments" / "training-run" / "data" / "full"
SURFACE_FILES = {"symbol": "D_algo_eval_op.parquet", "nl": "D_algo_eval_bare.parquet",
                 "rewrite": "D_algo_eval_bare.parquet", "translate": "D_algo_eval_bare.parquet"}
PAIRS_FILES = {"symbol": "pairs_symbol_seed0.pt", "nl": "pairs_seed0.pt",
               "translate": "pairs_translate_seed0.pt"}
LEAKCLEAN = HERE / "data" / "perf_leakclean_4x4_idx.json"
HEAD_CELL = "4x4"
MAX_NEW_TOKENS = 6
REWRITE_MAX_NEW_TOKENS = 16  # "1234 + 5678" is 7 tokens; room for a longer first line
EXACT_IDX_FILE = "rewrite_exact_idx.json"
STEP0_EM_FRAC = 0.5  # translate EM >= 0.5 x bare-op EM on the same rows (HANDOFF step 0)
STEP0_KL_MIN = 0.1  # KL(full || empty) > 0.1 nats, as run.py's KL_GATE_NATS
_INT = re.compile(r"-?\d+")
_EXPR = re.compile(r"(-?\d+)\s*([+*-])\s*(-?\d+)")  # premise_checks.py --chain


# ------------------------------------------------------------------- grading


def first_line(tok, ids: Sequence[int]) -> str:
    """Generated ids -> graded text: cut at the first EOS (so a post-EOS
    continuation is never glued on), decode, take the first line, strip."""
    ids = list(ids)
    if tok.eos_token_id in ids:
        ids = ids[: ids.index(tok.eos_token_id)]
    return tok.decode(ids, skip_special_tokens=True).split("\n")[0].strip()


def grade(text: str, answer_text: str, a: int, b: int, op: str) -> tuple[bool, bool]:
    """(strict, lenient). strict: text == the signed label string. lenient:
    text parses as one integer equal to the true result or its absolute value
    (for ``+`` this equals strict up to integer formatting)."""
    strict = text == answer_text
    if not _INT.fullmatch(text):
        return strict, False
    v, target = int(text), (a + b if op == "+" else a - b)
    return strict, v == target or v == abs(target)


def parse_rewrite(text: str, a: int, b: int) -> tuple[bool, bool]:
    """(wellformed, exact) for a rewrite of "What is the sum of a and b?", as
    premise_checks.py --chain: the first line of the stripped text, searched
    for ``int op int``; wellformed iff found, exact iff it is (a, '+', b)."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    m = _EXPR.search(line)
    if m is None:
        return False, False
    return True, (int(m.group(1)), m.group(2), int(m.group(3))) == (a, "+", b)


# ------------------------------------------------------------------ batching


def length_batches(keys: Sequence[Hashable], bs: int) -> list[np.ndarray]:
    """Row indices grouped by identical key (e.g. prompt token length), each
    group cut into chunks of at most ``bs`` rows, in first-seen key order and
    row order within a key. Every row appears in exactly one chunk."""
    if bs < 1:
        raise ValueError(f"length_batches: bs must be >= 1, got {bs}")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return [np.array(rows[s : s + bs]) for rows in groups.values() for s in range(0, len(rows), bs)]


@torch.no_grad()
def greedy_first_lines(model, tok, prompt_ids: Sequence[Sequence[int]], bs: int,
                       max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
    """Greedy-decode every prompt (no padding: batches share a prompt length)
    and return the graded first line per row, in input order."""
    dev = next(model.parameters()).device
    out: list[str] = [""] * len(prompt_ids)
    for rows in length_batches([len(p) for p in prompt_ids], bs):
        x = torch.tensor([list(prompt_ids[i]) for i in rows], dtype=torch.long, device=dev)
        gen = model.generate(
            x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )[:, x.shape[1]:].cpu()
        for i, row in zip(rows, gen):
            out[int(i)] = first_line(tok, row.tolist())
    return out


@torch.no_grad()
def answer_logprob(model, full_ids: Sequence[Sequence[int]], prompt_lens: Sequence[int],
                   bs: int) -> np.ndarray:
    """Teacher-forced sum over the answer tokens of log p(token | prefix), in
    nats, per row. The answer is ``full_ids[i][prompt_lens[i]:]``; rows are
    batched by (full length, prompt length) so no padding is used."""
    dev = next(model.parameters()).device
    out = np.full(len(full_ids), np.nan)
    keys = [(len(f), p) for f, p in zip(full_ids, prompt_lens)]
    for rows in length_batches(keys, bs):
        length, p = keys[int(rows[0])]
        if not 0 < p < length:
            raise ValueError(f"answer_logprob: need 0 < prompt_len < full_len, got {p}, {length}")
        x = torch.tensor([list(full_ids[i]) for i in rows], dtype=torch.long, device=dev)
        logp = torch.log_softmax(model(x).logits[:, p - 1 : length - 1].float(), dim=-1)
        lp = logp.gather(-1, x[:, p:, None])[..., 0].sum(-1)
        out[rows] = lp.cpu().double().numpy()
    return out


@torch.no_grad()
def pairs_m(model, pairs: dict, bs: int) -> dict:
    """LD on the circuit pairs via plain HF forwards: m(full) on clean_ids,
    m(empty) on corrupt_ids (run.py's sanity identity). Per-example arrays are
    returned under ``per_example`` (ld_full, ld_empty: (N, 3) = LD, term1, term2)."""
    dev = next(model.parameters()).device
    n = pairs["clean_ids"].shape[0]
    full, empty = torch.empty(n, 3), torch.empty(n, 3)
    for s in range(0, n, bs):
        sl = slice(s, s + bs)
        toks = [pairs[k][sl].to(dev) for k in ("c1", "k1", "c2", "k2")]
        for ids, dst in ((pairs["clean_ids"], full), (pairs["corrupt_ids"], empty)):
            ld, t1, t2 = logit_diff_2tok(model(ids[sl].to(dev)).logits.float(), *toks)
            dst[sl] = torch.stack([ld, t1, t2], -1).cpu()
    res = {
        "n": n,
        "m_full": full[:, 0].mean().item(),
        "m_full_terms": full[:, 1:].mean(0).tolist(),
        "m_empty": empty[:, 0].mean().item(),
        "m_empty_terms": empty[:, 1:].mean(0).tolist(),
        "m_full_minus_empty": (full[:, 0] - empty[:, 0]).mean().item(),
        "frac_ld_full_pos": (full[:, 0] > 0).float().mean().item(),
    }
    if "meta" in pairs:
        split = pairs["meta"]["split"].to_numpy()
        for name in ("discovery", "validation"):
            m = torch.as_tensor(split == name)
            if m.any():
                res[f"m_full_{name}"] = full[m, 0].mean().item()
                res[f"m_empty_{name}"] = empty[m, 0].mean().item()
    res["per_example"] = {"ld_full": full.numpy(), "ld_empty": empty.numpy()}
    return res


# ------------------------------------------------------------------- summary


def _agg(df: pd.DataFrame, lenient: bool, tf: bool) -> dict:
    out: dict = {"n": int(len(df))}
    if len(df) == 0:
        return out
    out["em_strict"] = float(df["strict"].mean())
    if lenient:
        out["em_lenient"] = float(df["lenient"].mean())
    out["frac_int"] = float(df["gen"].map(lambda t: bool(_INT.fullmatch(t))).mean())
    if tf:
        out["tf_logprob_mean_nats"] = float(df["tf_logprob"].mean())
    return out


def summarize(rows: pd.DataFrame, leakclean_idx: set[int]) -> dict:
    """Per op: all cells, each cell, and (``+`` only) the 4x4 leakage-clean
    subset; plus all ops pooled. ``rows`` has idx/op/cell/gen/strict/lenient/
    tf_logprob columns."""
    res: dict = {"ops": {}}
    for op, d in rows.groupby("op", sort=True):
        lenient = op == "-"
        o = {"all_cells": _agg(d, lenient, False), "cells": {}}
        for cell, dc in d.groupby("cell", sort=True):
            o["cells"][cell] = _agg(dc, lenient, op == "+" and cell == HEAD_CELL)
        if op == "+":
            head = d[d["cell"] == HEAD_CELL]
            o[f"{HEAD_CELL}_leakclean"] = _agg(head[head["idx"].isin(leakclean_idx)], False, True)
        res["ops"][op] = o
    res["all_ops"] = _agg(rows, True, False)
    return res


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def step0_frame(df: pd.DataFrame, leakclean_idx: set[int], translate: bool = False) -> pd.DataFrame:
    """The D_algo_eval_bare 4x4 '+' rows in ``leakclean_idx`` (step 0's rows),
    checked to carry the bare word prompt. With ``translate``, prompt_text
    becomes word question + gold rewrite (data.TRANSLATE_PROMPT) and full_text
    prompt + answer, so the rows go through ``run_eval`` like any surface."""
    from data import ADDITION_PROMPT, TRANSLATE_PROMPT  # experiments/eapig-circuit-check/data.py

    d = df[(df["op"] == "+") & (df["cell"] == HEAD_CELL) & df["idx"].isin(leakclean_idx)]
    d = d.reset_index(drop=True)
    if len(d) == 0:
        raise ValueError("step0_frame: no 4x4 '+' leakage-clean rows")
    want = [ADDITION_PROMPT.format(a=int(a), b=int(b)) for a, b in zip(d["a"], d["b"])]
    bad = np.flatnonzero(d["prompt_text"].to_numpy() != np.array(want, dtype=object))
    if bad.size:
        raise ValueError(f"step0_frame: {bad.size} rows are not the bare word prompt, "
                         f"e.g. {d['prompt_text'][bad[0]]!r}")
    if not translate:
        return d
    sums = [str(int(a) + int(b)) for a, b in zip(d["a"], d["b"])]
    if d["answer_text"].tolist() != sums:
        raise ValueError("step0_frame: answer_text is not str(a + b)")
    prompts = [TRANSLATE_PROMPT.format(a=int(a), b=int(b)) for a, b in zip(d["a"], d["b"])]
    return d.assign(prompt_text=prompts, full_text=[p + s for p, s in zip(prompts, sums)])


def read_rows(path: Path) -> pd.DataFrame:
    """A rows_<surface>.jsonl.gz file as a DataFrame."""
    with gzip.open(path, "rt") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def em_on_idx(rows: pd.DataFrame, idx: set[int]) -> dict:
    """Strict EM of the 4x4 '+' rows on exactly ``idx``. Raises ValueError if an
    idx is missing from (or repeated in) those rows: a silent partial join would
    compare the two surfaces on different problems."""
    head = rows[(rows["op"] == "+") & (rows["cell"] == HEAD_CELL)]
    sel = head[head["idx"].isin(idx)]
    if sel["idx"].duplicated().any():
        raise ValueError("em_on_idx: repeated idx in the rows file")
    missing = set(idx) - set(sel["idx"].tolist())
    if missing:
        raise ValueError(f"em_on_idx: {len(missing)} idx missing from the rows file, "
                         f"e.g. {sorted(missing)[:5]}")
    em = float(sel["strict"].mean()) if len(sel) else float("nan")
    return {"n": int(len(sel)), "em_strict": em}


def step0_gate(translate_em: float, symbol_em: float, kl_empty: float, *,
               em_frac: float = STEP0_EM_FRAC, kl_min: float = STEP0_KL_MIN) -> dict:
    """Step-0 verdict (HANDOFF-translate-circuit.md): pass iff translate EM >=
    ``em_frac`` x bare-op EM on the same rows AND KL(full || empty) > ``kl_min``
    (strict). With symbol_em == 0 the EM half passes vacuously (0 >= 0;
    flagged ``em_vacuous``) and only the KL half gates. Non-finite inputs raise
    ValueError (no verdict on a missing number)."""
    vals = {"translate_em": translate_em, "symbol_em": symbol_em, "kl_empty": kl_empty,
            "em_frac": em_frac, "kl_min": kl_min}
    bad = [k for k, v in vals.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"step0_gate: non-finite {bad}")
    em_ok = bool(translate_em >= em_frac * symbol_em)
    kl_ok = bool(kl_empty > kl_min)
    return vals | {"em_pass": em_ok, "kl_pass": kl_ok, "em_vacuous": bool(symbol_em == 0),
                   "pass": em_ok and kl_ok}


def step0_report(out: Path, sanity: Path, leakclean: Path, *, em_frac: float = STEP0_EM_FRAC,
                 kl_min: float = STEP0_KL_MIN) -> dict:
    """Read rows_translate/rows_symbol/rewrite_exact_idx.json from ``out`` and
    run.py's sanity.json (``kl_empty_mean``, translate pairs); gate on the
    exact-rewrite rows, report EM on all leakage-clean rows too; write
    ``out``/gate.json and return it."""
    exact_doc = json.loads((out / EXACT_IDX_FILE).read_text())
    exact = set(exact_doc["idx"])
    clean = set(json.loads(Path(leakclean).read_text())["idx"])
    if not exact:
        raise ValueError("step0_report: no exact rewrites, nothing to gate on")
    if not exact <= clean:
        raise ValueError(f"step0_report: {len(exact - clean)} exact idx outside the leakclean set")
    tr, sym = read_rows(out / "rows_translate.jsonl.gz"), read_rows(out / "rows_symbol.jsonl.gz")
    kl_empty = float(json.loads(Path(sanity).read_text())["kl_empty_mean"])
    on_exact = {"translate": em_on_idx(tr, exact), "symbol": em_on_idx(sym, exact)}
    on_clean = {"translate": em_on_idx(tr, clean), "symbol": em_on_idx(sym, clean)}
    res = {
        "gate": step0_gate(on_exact["translate"]["em_strict"], on_exact["symbol"]["em_strict"],
                           kl_empty, em_frac=em_frac, kl_min=kl_min),
        "exact_rewrite": on_exact,
        "all_leakclean": on_clean,
        "rewrite": {k: v for k, v in exact_doc.items() if k != "idx"},
        "kl_empty_mean": kl_empty,
        "sanity_file": str(sanity),
        "leakclean_file": Path(leakclean).name,
    }
    (out / "gate.json").write_text(json.dumps(_jsonable(res), indent=2) + "\n")
    return res


# ----------------------------------------------------------------------- run


def _merge_surface(out: Path, tag: str, model_name: str, surface: str, summ: dict) -> None:
    """Merge one surface's summary into out/perf_eval.json (refusing another model's file)."""
    path = out / "perf_eval.json"
    doc = json.loads(path.read_text()) if path.is_file() else {}
    if doc.get("model", model_name) != model_name:
        raise ValueError(f"perf_eval: {path} holds model {doc['model']!r}, not {model_name!r}")
    doc |= {"tag": tag, "model": model_name}
    doc.setdefault("surfaces", {})[surface] = summ
    path.write_text(json.dumps(_jsonable(doc), indent=2) + "\n")


def run_eval(model, tok, *, tag: str, model_name: str, surface: str, ops: Sequence[str],
             parquet: Path, leakclean: Path, out: Path, bs: int,
             pairs_file: Path | None = None, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Evaluate one model on one surface and write/merge the outputs (module
    docstring). Returns the surface's summary dict."""
    t0 = time.time()
    lc = set(json.loads(Path(leakclean).read_text())["idx"])
    df = pd.read_parquet(parquet)
    df = df[df["op"].isin(list(ops))].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"perf_eval: no rows with ops {list(ops)} in {parquet}")
    if surface == "translate":
        df = step0_frame(df, lc, translate=True)
    prompt_ids = tok(df["prompt_text"].tolist(), add_special_tokens=False)["input_ids"]
    full_ids = tok(df["full_text"].tolist(), add_special_tokens=False)["input_ids"]
    bad = [i for i, (p, f) in enumerate(zip(prompt_ids, full_ids)) if f[: len(p)] != p or len(f) <= len(p)]
    if bad:
        i = bad[0]
        raise ValueError(f"perf_eval: {len(bad)} rows whose prompt is not a token-prefix of the "
                         f"full text, e.g. row {i}: {df['full_text'][i]!r}")

    gens = greedy_first_lines(model, tok, prompt_ids, bs, max_new_tokens)
    graded = [grade(g, a_t, int(a), int(b), op) for g, a_t, a, b, op in
              zip(gens, df["answer_text"], df["a"], df["b"], df["op"])]
    tf = np.full(len(df), np.nan)
    head = np.flatnonzero(((df["op"] == "+") & (df["cell"] == HEAD_CELL)).to_numpy())
    if head.size:
        tf[head] = answer_logprob(model, [full_ids[i] for i in head],
                                  [len(prompt_ids[i]) for i in head], bs)
    rows = pd.DataFrame({
        "idx": df["idx"].astype(int), "op": df["op"], "cell": df["cell"], "gen": gens,
        "strict": [s for s, _ in graded], "lenient": [v for _, v in graded], "tf_logprob": tf,
    })
    summ = summarize(rows, lc)
    summ |= {"parquet": Path(parquet).name, "parquet_sha256": _sha256(Path(parquet)),
             "ops_requested": list(ops), "n_rows": int(len(df)), "max_new_tokens": max_new_tokens,
             "batch_size": bs, "leakclean_file": Path(leakclean).name,
             "rows_file": f"rows_{surface}.jsonl.gz"}

    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / f"rows_{surface}.jsonl.gz", "wt") as f:
        for r in rows.itertuples(index=False):
            rec = {"idx": r.idx, "op": r.op, "cell": r.cell, "gen": r.gen,
                   "strict": bool(r.strict), "lenient": bool(r.lenient)}
            if np.isfinite(r.tf_logprob):
                rec["tf_logprob"] = float(r.tf_logprob)
            f.write(json.dumps(rec) + "\n")

    if pairs_file is not None:
        from data import load_pairs  # experiments/eapig-circuit-check/data.py

        pm = pairs_m(model, load_pairs(pairs_file), bs)
        per = pm.pop("per_example")
        np.savez_compressed(out / f"pairs_m_{surface}.npz", **per)
        summ["pairs_m"] = pm | {"pairs_file": Path(pairs_file).name}
    summ["seconds"] = time.time() - t0
    _merge_surface(out, tag, model_name, surface, summ)
    return summ


def run_rewrite(model, tok, *, tag: str, model_name: str, parquet: Path, leakclean: Path,
                out: Path, bs: int, max_new_tokens: int = REWRITE_MAX_NEW_TOKENS) -> dict:
    """Step 0.1: greedy rewrite of the bare word prompt on the leakage-clean 4x4
    '+' rows, graded by ``parse_rewrite``. Writes rows_rewrite.jsonl.gz (idx,
    gen, wellformed, exact), rewrite_exact_idx.json (sorted exact idx + counts)
    and merges a ``rewrite`` entry into perf_eval.json. Returns that entry."""
    t0 = time.time()
    df = step0_frame(pd.read_parquet(parquet), set(json.loads(Path(leakclean).read_text())["idx"]))
    prompt_ids = tok(df["prompt_text"].tolist(), add_special_tokens=False)["input_ids"]
    gens = greedy_first_lines(model, tok, prompt_ids, bs, max_new_tokens)
    parsed = [parse_rewrite(g, int(a), int(b)) for g, a, b in zip(gens, df["a"], df["b"])]
    rows = pd.DataFrame({"idx": df["idx"].astype(int), "gen": gens,
                         "wellformed": [w for w, _ in parsed], "exact": [e for _, e in parsed]})
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "rows_rewrite.jsonl.gz", "wt") as f:
        for r in rows.itertuples(index=False):
            f.write(json.dumps({"idx": r.idx, "gen": r.gen, "wellformed": bool(r.wellformed),
                                "exact": bool(r.exact)}) + "\n")
    exact_idx = sorted(int(i) for i in rows["idx"][rows["exact"]])
    counts = {"n_rows": int(len(rows)), "n_exact": len(exact_idx),
              "n_wellformed": int(rows["wellformed"].sum())}
    doc = {"model": model_name, "leakclean_file": Path(leakclean).name, **counts, "idx": exact_idx}
    (out / EXACT_IDX_FILE).write_text(json.dumps(doc) + "\n")
    summ = {"n": counts["n_rows"], "rewrite_exact": float(rows["exact"].mean()),
            "rewrite_wellformed": float(rows["wellformed"].mean()),
            "parquet": Path(parquet).name, "parquet_sha256": _sha256(Path(parquet)),
            "max_new_tokens": max_new_tokens, "batch_size": bs,
            "leakclean_file": Path(leakclean).name, "rows_file": "rows_rewrite.jsonl.gz",
            "exact_idx_file": EXACT_IDX_FILE, "seconds": time.time() - t0}
    _merge_surface(out, tag, model_name, "rewrite", summ)
    return summ


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="local dir or HF repo id (run.py load_model)")
    ap.add_argument("--surface", required=True, choices=sorted(SURFACE_FILES))
    ap.add_argument("--ops", nargs="+", default=["+"], choices=["+", "-"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None, help="default results_perf/<tag>/")
    ap.add_argument("--pairs-m", action="store_true",
                    help="also LD m(full)/m(empty) on the 768 circuit pairs of this surface")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="dir holding the eval parquets")
    ap.add_argument("--leakclean", default=str(LEAKCLEAN))
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help=f"default {MAX_NEW_TOKENS} ({REWRITE_MAX_NEW_TOKENS} for rewrite)")
    return ap


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()
    if a.pairs_m and a.surface not in PAIRS_FILES:
        ap.error(f"--pairs-m has no pairs file for --surface {a.surface}")
    from run import load_model  # experiments/eapig-circuit-check/run.py
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = load_model(a.model, a.device)
    out = Path(a.out) if a.out else HERE / "results_perf" / a.tag
    parquet = Path(a.data_dir) / SURFACE_FILES[a.surface]
    if a.surface == "rewrite":
        s = run_rewrite(model, tok, tag=a.tag, model_name=a.model, parquet=parquet,
                        leakclean=Path(a.leakclean), out=out, bs=a.batch_size,
                        max_new_tokens=REWRITE_MAX_NEW_TOKENS if a.max_new_tokens is None
                        else a.max_new_tokens)
        print(f"[perf] {a.tag} rewrite: exact {s['rewrite_exact']:.4f} "
              f"wellformed {s['rewrite_wellformed']:.4f} (n {s['n']}) done in {s['seconds']:.0f}s",
              flush=True)
        return
    s = run_eval(model, tok, tag=a.tag, model_name=a.model, surface=a.surface, ops=a.ops,
                 parquet=parquet, leakclean=Path(a.leakclean), out=out, bs=a.batch_size,
                 pairs_file=HERE / "data" / PAIRS_FILES[a.surface] if a.pairs_m else None,
                 max_new_tokens=MAX_NEW_TOKENS if a.max_new_tokens is None else a.max_new_tokens)
    for op, o in s["ops"].items():
        h = o["cells"].get(HEAD_CELL, {})
        print(f"[perf] {a.tag} {a.surface} {op}: EM all {o['all_cells'].get('em_strict')} "
              f"4x4 {h.get('em_strict')} lenient {h.get('em_lenient')} tf {h.get('tf_logprob_mean_nats')}")
    if "pairs_m" in s:
        pm = s["pairs_m"]
        print(f"[perf] pairs m_full {pm['m_full']:.3f} m_empty {pm['m_empty']:.3f}")
    print(f"[perf] {a.tag} {a.surface} done in {s['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
