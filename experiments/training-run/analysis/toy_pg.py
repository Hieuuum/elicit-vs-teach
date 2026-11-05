"""The pairing-gap score s^PG on the toy parents, where elicit vs teach is known.

The paper's calibration read the raw gap Delta^PG of the five arithmetic parents (decisions.md
2026-10-20: TS base 0.13, TS installed 0.09, op-bridge 1.15, Llama base 0.57, Llama installed 2.67)
and scored the teach parents against the elicit parent, a different model.  The metric as defined
(sections.tex, Metric 1) scores each model against its own reference: Delta^PG_full, the gap of the
SAME model on its own greedy answers to the same prompts (whose pairing it holds by construction).
This script computes that, per parent, on the held-out eval block (the same rows and rotations as
grad_transfer.py --task arith, so Delta^PG reproduces results/m21_sweep/<prefix>-n0.json):

  delta_nats       Delta^PG on the true answers: mean over 4 seeded rotations of L(rotated) - L(true)
  full_nats        Delta^PG on the model's own greedy answers (first line, <= 12 tokens)
  s                clip(delta / full, 0, 1); call elicit iff s >= 1/2
  own_exact        share of own answers equal to the true answer (accuracy before training)
  own_distinct     distinct own answers / rows (a near-constant answer makes full ~ 0: the score is
                   then flagged degenerate, not read; sections.tex warns about exactly this)

OUTCOME 2026-11-05 (decisions.md): the own-greedy-answer reference fails here (2/4 calls, 1 degenerate):
it measures how prompt-dependent the argmax output is, right or wrong.  The endpoint reference replaces it:

  endpoint   (CPU) full = Delta^PG of the same parent after fine-tuning on the data (the largest sweep size
             with a results/m21_sweep/<prefix>-n<N>.json), none = the same size trained on the rotated twin
             (<prefix>-shuf-n<N>.json; 0 if absent); s_end = clip((delta - none) / (full - none), 0, 1).
             Written into results/toy_pg/<arm>.json; report prints both scores.

Ground truth (decisions.md 2026-10-20): TS base and TS installed TEACH (the EDL hump survives the
rotated-twin subtraction; half of M21's rise by 31.6K / 14.7K examples), op-bridge, Llama base and
Llama installed ELICIT/GATE (half-rise 3.2K, <= 1K, 1.5K).  Pre-registered: the three elicit parents
read s >= 1/2, the two teach parents s < 1/2, and s orders the parents opposite to the half-rise size.

  python3 toy_pg.py read --arm all --device cuda --confirm-cost      # ~0.1 GPU-h for all five
  python3 toy_pg.py endpoint [--m21-dir results/m21_sweep]          # CPU
  python3 toy_pg.py report                                             # CPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "unlearning"))
sys.path.insert(0, str(HERE))

CONFIGS = REPO_ROOT / "experiments" / "training-run" / "configs"
OUT = REPO_ROOT / "experiments" / "training-run" / "results" / "toy_pg"
TOKENIZER = "meta-llama/Llama-3.2-1B"
SEED = 316   # grad_transfer.py --seed default: the B rows and rotations of results/m21_sweep

# arm -> (config whose data/eval block defines the held-out rows, init (store-relative or hub id),
#         ground truth, half-rise size of M21 along the sweep, raw Delta^PG from m21_sweep n=0)
ARMS = {
    "ts_base": ("ts1b_fig2ts_noinst.yaml", "runs/evt-ts1b-base/model", "teach", 31_600, 0.13),
    "ts_inst": ("ts1b_fig2ts_inst.yaml", "runs/evt-ts1b-fig2ts-installer/model", "teach", 14_700, 0.09),
    "op_bridge": ("ts1b_mix_nl.yaml", "runs/evt-ts1b-op-bridge-mix/model", "elicit", 3_200, 1.15),
    "llama_base": ("llama_fig2nl3_noinst.yaml", "meta-llama/Llama-3.2-1B", "elicit", 1_000, 0.57),
    "llama_inst": ("llama_fig2nl3_inst.yaml", "runs/evt-llama-fig2nl3-installer/model_merged", "elicit", 1_500, 2.67),
}
# arm -> the sweep's run-id prefix (launch_fig2_shuf.sh arms_of; m21_sweep files <prefix>[-shuf]-n<N>.json)
PREFIX = {"ts_base": "evt-ts1b-fig2ts-noinst", "ts_inst": "evt-ts1b-fig2ts-inst", "op_bridge": "evt-ts1b-mix-nl",
          "llama_base": "evt-llama-fig2nl3-noinst", "llama_inst": "evt-llama-fig2nl3-inst"}


# ------------------------------------------------------------------ pure pieces (tested)


def own_answer_frame(df: pd.DataFrame, completions: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """The same prompts with each answer span replaced by the model's own completion (trailing
    whitespace stripped).  Rows whose completion is empty are dropped; returns the frame and the kept
    row positions.  The frozen full_texts end at the answer, so nothing follows the span."""
    rows, kept = [], []
    for i, (full, a, c) in enumerate(zip(df["full_text"], df["answer_char_start"].astype(int), completions)):
        c = c.rstrip()
        if not c.strip():
            continue
        rows.append({"full_text": full[:a] + c, "answer_char_start": a, "answer_char_end": a + len(c)})
        kept.append(i)
    return pd.DataFrame(rows), kept


def true_answers(df: pd.DataFrame) -> list[str]:
    return [t[a:b] for t, a, b in zip(df["full_text"], df["answer_char_start"].astype(int), df["answer_char_end"].astype(int))]


def score(delta: float, full: float, min_full: float = 0.05) -> float | None:
    """s^PG = clip(delta / full, 0, 1); None when the own-answer reference is degenerate (<= min_full)."""
    if full <= min_full:
        return None
    return min(1.0, max(0.0, delta / full))


def endpoint_score(delta: float, full: float, none: float) -> float | None:
    """s = clip((delta - none) / (full - none), 0, 1); None when full does not exceed none."""
    if full <= none:
        return None
    return min(1.0, max(0.0, (delta - none) / (full - none)))


def endpoint_sizes(true_ns: list[int], shuf_ns: list[int]) -> tuple[int | None, bool]:
    """The largest trained size with a true read; whether the rotated twin exists at that size."""
    ns = [n for n in true_ns if n > 0]
    if not ns:
        return None, False
    n = max(ns)
    return n, n in shuf_ns


def call(s: float | None) -> str:
    return "n/a" if s is None else ("elicit" if s >= 0.5 else "teach")


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


# ------------------------------------------------------------------ model side


def examples_kept(df: pd.DataFrame, tok) -> tuple[pd.DataFrame, int]:
    """Rows whose answer span tokenizes exactly (V5.38); an own answer can break the alignment."""
    from geode.arith.spans import tokenize_with_spans

    ok = []
    for i, (t, a, b) in enumerate(zip(df["full_text"], df["answer_char_start"], df["answer_char_end"])):
        try:
            tokenize_with_spans([t], [(int(a), int(b))], tok, append_eos=True)
            ok.append(i)
        except ValueError:
            pass
    return df.iloc[ok].reset_index(drop=True), len(df) - len(ok)


def read_arm(arm: str, args) -> dict:
    import torch
    from grad_transfer import load_sets_arith
    from relearn import examples_of, preference_gap
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from geode.arith.decode import greedy_completions

    cfg, init, truth, half, raw = ARMS[arm]
    store = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
    path = init if init.startswith("meta-llama/") else str(store / init)
    b = load_sets_arith(CONFIGS / cfg, None, args.n, 4, SEED)["bioB"]
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(args.device).eval()
    kw = dict(rotations=4, batch_size=args.batch_size, device=args.device, seed=SEED)
    true = preference_gap(model, b, tok, **kw)
    prompts = [ex.input_ids[: ex.label_span[0]] for ex in examples_of(b, tok)]
    with torch.no_grad():
        comp = greedy_completions(model, tok, prompts, device=args.device, batch_size=args.batch_size)
    tok.padding_side = "right"
    gold = [g.strip() for g in true_answers(b)]
    own, kept = own_answer_frame(b, comp)
    own, misaligned = examples_kept(own, tok)
    full = preference_gap(model, own, tok, **kw)
    d, f = true["m21_test_nats"], full["m21_test_nats"]
    s = score(d, f)
    return {"arm": arm, "init": path, "config": cfg, "truth": truth, "half_rise_n": half, "raw_m21_sweep_nats": raw,
            "n": len(b), "delta_nats": d, "full_nats": f, "s": s, "call": call(s),
            "own_n": len(own), "own_dropped_empty": len(b) - len(kept), "own_dropped_alignment": misaligned,
            "own_exact": sum(c.strip() == g for c, g in zip(comp, gold)) / len(gold),
            "own_distinct": len({c.strip() for c in comp}) / len(comp),
            "true": true, "own": full}


def cmd_read(args) -> None:
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    todo = [a for a in arms if not (OUT / f"{a}.json").is_file()]
    print(f"[toy_pg] {len(todo)} parents x ({args.n} rows: 2 x 5 scoring passes + greedy decode); ~{0.02 * len(todo):.2f} GPU-h")
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[toy_pg] refusing to run on a GPU without --confirm-cost")
    OUT.mkdir(parents=True, exist_ok=True)
    for a in todo:
        r = read_arm(a, args)
        (OUT / f"{a}.json").write_text(json.dumps(r, indent=2))
        print(f"[toy_pg] {a}: delta {r['delta_nats']:+.3f} (sweep {r['raw_m21_sweep_nats']:.2f}) full {r['full_nats']:+.3f} "
              f"s {r['s']} -> {r['call']} (truth {r['truth']}); own exact {r['own_exact']:.3f} distinct {r['own_distinct']:.3f}")


def cmd_endpoint(args) -> None:
    from knowledge_sweep import m21_points

    for a in ARMS:
        f = OUT / f"{a}.json"
        if not f.is_file():
            print(f"[toy_pg] {a}: no read yet (run read first)")
            continue
        r = json.loads(f.read_text())
        prefix = PREFIX[a]
        tr, sh = m21_points(args.m21_dir, prefix, False), m21_points(args.m21_dir, prefix, True)
        n, has_shuf = endpoint_sizes(list(tr), list(sh))
        if n is None:
            print(f"[toy_pg] {a}: no sweep point in {args.m21_dir} for {prefix}")
            continue
        full, none = tr[n]["gap_nats"], (sh[n]["gap_nats"] if has_shuf else 0.0)
        s = endpoint_score(r["delta_nats"], full, none)
        r["endpoint"] = {"n": n, "full_nats": full, "none_nats": none, "none_measured": has_shuf, "s": s}
        f.write_text(json.dumps(r, indent=2))
        print(f"[toy_pg] {a}: endpoint n={n:,} full {full:.3f} none {none:+.3f}{'' if has_shuf else ' (no twin, 0)'} "
              f"s_end {s if s is None else round(s, 4)}")


def cmd_report(args) -> None:
    rs = [json.loads((OUT / f"{a}.json").read_text()) for a in ARMS if (OUT / f"{a}.json").is_file()]
    print(f"[toy_pg] {'parent':<11} {'truth':<6} {'half-rise':>9} {'dPG':>7} {'dPG_full':>8} {'s':>5} {'call':<6} "
          f"{'own exact':>9} {'distinct':>8}")
    for r in rs:
        s = "n/a" if r["s"] is None else f"{r['s']:.2f}"
        mark = "" if r["call"] == "n/a" else (" ok" if r["call"] == r["truth"] else " WRONG")
        print(f"[toy_pg] {r['arm']:<11} {r['truth']:<6} {r['half_rise_n']:>9,} {r['delta_nats']:>7.3f} {r['full_nats']:>8.3f} "
              f"{s:>5} {r['call']:<6} {r['own_exact']:>9.3f} {r['own_distinct']:>8.3f}{mark}")
    scored = [r for r in rs if r["s"] is not None]
    if len(scored) >= 3:
        rho = spearman([r["s"] for r in scored], [r["half_rise_n"] for r in scored])
        print(f"[toy_pg] Spearman(s, half-rise n) over {len(scored)} parents: {rho:+.2f} (pre-registered: negative)")
    right = sum(r["call"] == r["truth"] for r in scored)
    print(f"[toy_pg] own-answer calls matching ground truth: {right}/{len(scored)} scored ({len(rs) - len(scored)} degenerate reference)")
    ends = [r for r in rs if r.get("endpoint", {}).get("s") is not None]
    if ends:
        print(f"[toy_pg] endpoint reference: {'parent':<11} {'truth':<6} {'n':>9} {'full':>7} {'none':>7} {'s_end':>6}")
        for r in ends:
            e = r["endpoint"]
            print(f"[toy_pg]                     {r['arm']:<11} {r['truth']:<6} {e['n']:>9,} {e['full_nats']:>7.3f} "
                  f"{e['none_nats']:>+7.3f} {e['s']:>6.3f}")
        t = [r["endpoint"]["s"] for r in ends if r["truth"] == "teach"]
        el = [r["endpoint"]["s"] for r in ends if r["truth"] == "elicit"]
        if t and el:
            sep = "separated" if max(t) < min(el) else "OVERLAP"
            print(f"[toy_pg] s_end teach max {max(t):.3f} vs elicit min {min(el):.3f}: {sep}")
        if len(ends) >= 3:
            rho = spearman([r["endpoint"]["s"] for r in ends], [r["half_rise_n"] for r in ends])
            print(f"[toy_pg] Spearman(s_end, half-rise n) over {len(ends)} parents: {rho:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("read", "endpoint", "report"))
    ap.add_argument("--m21-dir", type=Path, default=REPO_ROOT / "experiments" / "training-run" / "results" / "m21_sweep")
    ap.add_argument("--arm", choices=("all", *ARMS), default="all")
    ap.add_argument("--n", type=int, default=600, help="held-out rows (grad_transfer.py's default)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    {"read": cmd_read, "endpoint": cmd_endpoint, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
