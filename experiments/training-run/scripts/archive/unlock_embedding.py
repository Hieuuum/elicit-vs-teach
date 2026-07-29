"""The 512-parameter unlock — is one untrained token the lock on addition? (2026-07-27)

The 2026-07-26 dataset audit found that the run-2 elicit parent's fine-tune
corpus (D_algo, 1M NL rows) contains the '+' glyph **zero** times, while the
'-' glyph occurs 250,110 times as the sign of a negative answer — and that
under the frozen 10k TinyStories BPE the operator-notation subtraction sign and
the NL negative-answer sign are the *same token* (``Ġ-``, id 1854). So the
parent's 19.8% on operator subtraction vs 0.39% on operator addition may not be
a capability difference at all: subtraction re-used a token it had trained a
quarter-million times, addition met a token that never received an input-side
gradient.

This script tests that directly. Everything in the parent is frozen except a
**single row of the input embedding table** — 512 parameters — which is then
trained on operator-notation addition. 512 parameters entering the network as
one token's embedding cannot store 4-digit addition; the algorithm has to
already be in the frozen 38.7M. So:

  accuracy rises and predictions move toward ``a+b``  => the algorithm
      pre-existed and one untrained token was the lock (elicitation)
  accuracy flat, predictions stay ``|a-b|``-shaped    => the parent has one
      arithmetic mode, and the latent-addition reading dies

Either outcome is decisive, which is why this runs before any dataset redesign.

Two subcommands:

``provenance``  No model, no GPU: reads the run-1 and run-2 embedding matrices
    from safetensors and reports per-row movement grouped by whether the token
    occurs in D_algo. Measured 2026-07-27, and it does not go the obvious way:
    D_algo's token support is only **30 of 10,000** rows, and the 9,970 absent
    rows moved *more* (median L2 1.78) than the 30 present ones (0.66). '+'
    moved 2.11 — above ``Ġ-``'s 0.62 — with its off-axis component at the
    98.2nd percentile of the absent group.

    None of that is evidence of reading, and the norm cannot be read as if it
    were. Under the tie, row 12 was also the *unembedding* row for '+', and a
    token that is never a correct label receives only the monotone push-down
    from the softmax denominator, so its row drifts far; a frequent label is
    pulled up as often as pushed down and settles near an equilibrium. The
    off-axis component just says that push-down was weighted by where the model
    thought '+' was likely, which is token-specific in *direction* while still
    being pure suppression.

    The decisive fact needs no statistics: '+' occurs **0 times** in D_algo's
    1M rows, so the gradient into its *input* embedding is exactly zero by
    construction. Every bit of row 12's movement came through the tied
    unembedding. Keep this subcommand for the record — it quantifies the
    starting point the unlock experiment departs from, and it stops anyone
    (including a future me) from claiming row 12 was pristine.

``unlock``  The experiment. For each (row, lr, k) cell: restore the embedding,
    train that one row on k addition examples, re-run G5's zero-shot arm, and
    record accuracy by op plus the answer-shape distribution.

Design points that make the result citable, each learned from a prior incident:

* **The embeddings are tied** (``tie_word_embeddings: true``; the checkpoint
  has 74 keys and no ``lm_head.weight``). Left tied, "train one row" would also
  edit the unembedding, so the row would receive gradient from the softmax
  denominator at every position — contaminating "learning to read '+'" with
  "learning to not emit '+'". The script **unties first** (clone, freeze
  ``lm_head``), and proves the untie is a no-op by asserting logits are
  bitwise-identical on a probe batch before and after.
* **LR is a declared grid axis, not a tuned knob.** One embedding row wants a
  much larger LR than any pin in this repo. Every arm — treatment and nulls —
  gets the identical (lr, steps) protocol and the **whole surface is reported**,
  never the max. Tuning an LR and quoting the best cell is the failure class in
  the project's scope-check memory.
* **Training examples are drawn from D_target minus D_algo**, direct triples
  *and* commuted twins. 29.18% of D_target was seen by this parent in NL, and
  for '+' the commuted twin carries the identical answer; an unfiltered draw
  would invite "you unlocked recall, not addition".
* **Nulls are ``uest`` (6204) and ``:`` (27)** — tokens present in every
  operator prompt, with a real input-side gradient path and no arithmetic
  semantics. (``Ġ`` was rejected: it is load-bearing everywhere, so its failure
  mode is "breaks the model", which does not discriminate.) If training
  ``uest``'s row unlocks addition, the finding is dead.
* **The readout is a shape table, not a scalar.** Baseline on '+' questions is
  ``a+b`` 0.39% / ``|a-b|`` 7.28% / none ~92%; an accuracy alone cannot
  distinguish "the algorithm became reachable" from "it got noisier".
* ``append_eos=True`` for training spans (V5.43 — the EOS must live inside the
  label span), ``False`` for eval prompts, matching ``gates.py``'s accuracy
  path. Mixing them reproduces the 2026-07-21 no-stop incident.
* Prompts are token-prefixes of the training tokenization, never re-tokenized
  char slices (the 2026-07-21 sign-drop incident).

Deliberately out of scope: a full-fine-tune ceiling arm. It would answer "how
much better could it get", which is not what gates the decision.

**This is a diagnostic, not an arm.** It trains on correct-label operator
addition, so it can never be part of Arm A — it writes no manifest, registers
no run, and touches nothing under ``{store}/runs`` or ``{store}/results``. It
does train on a GPU, so it carries ``--confirm-cost`` (budget rule).

Usage (box, from experiments/training-run/scripts/):
    python3 unlock_embedding.py provenance \
        --run evt-run2-armA-algo --base evt-run1-base-v3-ext \
        --algo-config ../configs/run2_algo.yaml --out ../analysis/figures/unlock_provenance.csv

    # forward: train each row on addition
    python3 unlock_embedding.py unlock --confirm-cost --train-op=+ \
        --rows '+,uest,:' --lr-grid 1e-3,1e-2,1e-1,1.0 --k-grid 32,128,512 \
        --expect overall=0.1016,add=0.0039 \
        --out ../analysis/figures/unlock_forward.csv

    # mirror: same protocol on subtraction. 'id:1854' is Ġ-, the subtraction
    # operator; '+' is the degenerate control that cannot move here.
    python3 unlock_embedding.py unlock --confirm-cost --train-op=- \
        --rows 'id:1854,:,+' --lr-grid 1e-3,1e-2,1e-1,1.0 --k-grid 32,128,512 \
        --expect overall=0.1016,add=0.0039 \
        --out ../analysis/figures/unlock_mirror.csv

Results (2026-07-27, both directions) are in ``notes/decisions.md`` under
"the 512-parameter unlock". Headline: addition 0.0039 -> 0.3976 from 512
parameters, so it was latent; but '+' is the *weakest* handle (0.1083), so the
glyph-lock hypothesis this script was built to test is falsified. The CSVs
live under ``analysis/figures/`` which is gitignored — re-run to regenerate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from geode.arith import exact_match, greedy_completions, parse_answer, tokenize_with_spans
from geode.edl.masking import TaskFormat, label_mask
from geode.train import untie_lm_head
from geode.train.sft import _mean_masked_ce_nats  # the one masked-CE; a copy would drift
from geode.zoo import checkpoint_dir, load_model
from gates import G5_N_SHOTS
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet
from train_target import EVAL_STOP_ROWS

# The treatment row and the two nulls, as literal strings resolved through the
# frozen tokenizer at run time (ids asserted below, so a tokenizer swap fails
# loudly instead of silently training the wrong row).
DEFAULT_ROWS = "+,uest,:"
EXPECTED_IDS = {"+": 12, "uest": 6204, ":": 27, "Ġ-": 1854}
# Each operator's own token — present only in that operator's prompts, which
# is what makes it a *conditional* handle. 'Ġ-' is to subtraction what '+' is
# to addition (and is also the NL negative-answer sign; see the module docs).
OP_ROW = {"+": 12, "-": 1854}


def resolve_row(tokenizer, token: str) -> int:
    """Vocabulary id for a literal token string, with the pinned-id check.

    ``id:N`` selects a row directly — the escape hatch for tokens whose literal
    spelling does not survive a shell round-trip (``Ġ-`` is id 1854).
    """
    if token.startswith("id:"):
        return int(token[3:])
    row = tokenizer.convert_tokens_to_ids(token)
    if row is None or row == getattr(tokenizer, "unk_token_id", None):
        raise SystemExit(f"token {token!r} is not in this tokenizer's vocabulary")
    if token in EXPECTED_IDS and row != EXPECTED_IDS[token]:
        raise SystemExit(
            f"token {token!r} is id {row}, expected {EXPECTED_IDS[token]} — "
            "wrong tokenizer for this checkpoint"
        )
    return int(row)


def answer_shapes(a: int, b: int) -> dict[str, int]:
    """The closed forms a wrong prediction plausibly lands on, from the audit."""
    return {"a+b": a + b, "a-b": a - b, "b-a": b - a, "|a-b|": abs(a - b), "a": a, "b": b}


def shape_table(rows: list[tuple[int, int]], parsed: list[int | None]) -> dict[str, float]:
    """Fraction of predictions matching each closed form, plus none / negative."""
    keys = ["a+b", "a-b", "b-a", "|a-b|", "a", "b"]
    counts = dict.fromkeys(keys, 0)
    none_count, negative = 0, 0
    for (a, b), p in zip(rows, parsed):
        if p is None:
            none_count += 1
            continue
        if p < 0:
            negative += 1
        forms = answer_shapes(a, b)
        hit = False
        for k in keys:
            if p == forms[k]:
                counts[k] += 1
                hit = True
        if not hit:
            none_count += 1
    n = max(1, len(parsed))
    out = {f"shape_{k}": counts[k] / n for k in keys}
    out["shape_other_or_unparsed"] = none_count / n
    out["frac_negative_emitted"] = negative / n
    return out


# --------------------------------------------------------------------------
# provenance: did row 12 ever receive an input-side gradient?
# --------------------------------------------------------------------------


def cmd_provenance(args: argparse.Namespace) -> int:
    from safetensors import safe_open
    from transformers import AutoTokenizer

    store = Path(os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store")))
    key = "model.embed_tokens.weight"

    def embed_of(run_id: str) -> torch.Tensor:
        path = checkpoint_dir(run_id, store=store) / "model.safetensors"
        with safe_open(path, framework="pt") as f:
            if key not in f.keys():
                raise SystemExit(f"{path}: no {key}")
            return f.get_tensor(key).float()

    w_base, w_algo = embed_of(args.base), embed_of(args.run)
    if w_base.shape != w_algo.shape:
        raise SystemExit(f"shape mismatch {w_base.shape} vs {w_algo.shape}")
    d_vec = w_algo - w_base
    delta = d_vec.norm(dim=1)

    cfg = load_config(args.algo_config, None)
    df = load_frozen_parquet(cfg)
    local = (args.algo_config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])

    # D_algo is two fixed templates over digits, so the token support of a
    # sample is the token support of the file; --sample only bounds the cost.
    sample = df["full_text"].iloc[: args.sample].tolist()
    seen: set[int] = set()
    for ids in tokenizer(sample, add_special_tokens=False).input_ids:
        seen.update(ids)

    plus, minus = resolve_row(tokenizer, "+"), resolve_row(tokenizer, "Ġ-")
    absent = sorted(set(range(w_base.shape[0])) - seen)
    groups = {
        "in_D_algo (both paths trained)": sorted(seen),
        "absent from D_algo (suppression only)": absent,
    }
    print("### EMBEDDING-ROW MOVEMENT, run 1 -> run 2 (L2 of the row delta)")
    print(f"  D_algo token support: {len(seen)} of {w_base.shape[0]} vocabulary rows")
    out_rows = []
    for name, idx in groups.items():
        d = delta[idx]
        print(
            f"  {name:40s} n={len(idx):>5d}  median={d.median():.5f} "
            f"p10={d.quantile(0.10):.5f} p90={d.quantile(0.90):.5f} max={d.max():.5f}"
        )
        out_rows.append(
            {
                "group": name,
                "n": len(idx),
                "median": float(d.median()),
                "p10": float(d.quantile(0.10)),
                "p90": float(d.quantile(0.90)),
                "max": float(d.max()),
            }
        )
    absent_delta = delta[absent]
    pct = float((absent_delta < delta[plus]).float().mean() * 100)
    print(f"\n  '+'  (id {plus}) delta = {delta[plus]:.5f}  -> {pct:.1f}th pct of the absent group")
    print(f"  'Ġ-' (id {minus}) delta = {delta[minus]:.5f}  (trained on both paths)")

    # Norm alone does not separate "was read" from "was suppressed" — it runs
    # the other way. A token that is never a correct label receives only the
    # monotone push-down from the tied unembedding, so its row drifts far; a
    # frequent label is pulled up as often as it is pushed down and settles
    # near an equilibrium. Direction is the discriminating quantity: pure
    # suppression moves every absent row along one shared axis (antiparallel to
    # the mean answer-position hidden state), so a row that was additionally
    # *read* is the one with movement off that axis.
    u = d_vec[absent].mean(dim=0)
    u = u / u.norm()
    cos = (d_vec @ u) / delta.clamp_min(1e-12)
    resid = delta * (1 - cos.clamp(-1, 1) ** 2).clamp_min(0).sqrt()
    digits = [resolve_row(tokenizer, str(i)) for i in range(10)]
    print("\n### DIRECTION OF THE MOVEMENT (u = shared absent-token axis)")
    print(f"  {'group':40s} {'median cos(d,u)':>16s} {'median off-axis':>16s}")
    for name, idx in [
        *groups.items(),
        ("digits 0-9", digits),
        ("'+' row", [plus]),
        ("'Ġ-' row", [minus]),
    ]:
        c, r = cos[idx], resid[idx]
        print(f"  {name:40s} {float(c.median()):>16.4f} {float(r.median()):>16.4f}")
        out_rows.append(
            {
                "group": name,
                "n": len(idx),
                "median": float(delta[idx].median()),
                "median_cos_u": float(c.median()),
                "median_off_axis": float(r.median()),
            }
        )
    r_pct = float((resid[absent] < resid[plus]).float().mean() * 100)
    print(
        f"\n  '+' off-axis movement is at the {r_pct:.1f}th percentile of the absent group.\n"
        "  Inside that distribution => the row moved only as suppression moved every\n"
        "  never-labelled row, i.e. it was never read. Far above it => something\n"
        "  token-specific reached row 12 during run 2, and the 'untrained' framing\n"
        "  needs weakening before the unlock result is interpreted."
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in out_rows for k in r)))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n[evt] wrote {args.out}")
    return 0


# --------------------------------------------------------------------------
# unlock: the experiment
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("unlock", "provenance"))
    parser.add_argument("--run", default="evt-run2-armA-algo")
    parser.add_argument("--base", default="evt-run1-base-v3-ext", help="provenance: run-1 base")
    parser.add_argument("--sample", type=int, default=20000, help="provenance: D_algo rows read")
    parser.add_argument(
        "--eval-config", type=Path, default=Path("../configs/eval_target_data.yaml")
    )
    parser.add_argument("--train-config", type=Path, default=Path("../configs/run7_target_1m.yaml"))
    parser.add_argument("--algo-config", type=Path, default=Path("../configs/run2_algo.yaml"))
    parser.add_argument(
        "--rows", default=DEFAULT_ROWS, help="comma-separated token strings, or 'id:N'"
    )
    parser.add_argument(
        "--train-op",
        default="+",
        choices=("+", "-"),
        help="operator to train on. The mirror run (--train-op -) is the bracket: "
        "'id:1854' (Ġ-) is to subtraction what '+' is to addition",
    )
    parser.add_argument("--lr-grid", default="1e-2,1e-1,1.0")
    parser.add_argument("--k-grid", default="32,512")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--micro-batch", type=int, default=128)
    parser.add_argument("--n", type=int, default=1024, help="eval questions; G5's default")
    parser.add_argument("--expect", default=None, help="baseline drift gate, 'overall=..,add=..'")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.cmd == "provenance":
        return cmd_provenance(args)

    from transformers import AutoTokenizer

    store = Path(os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store")))
    eval_cfg = load_config(args.eval_config, None)
    local = (args.eval_config.parent / eval_cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        local if local.is_dir() else eval_cfg["tokenizer"]["path"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows_to_train = [resolve_row(tokenizer, t) for t in args.rows.split(",")]
    lr_grid = [float(x) for x in args.lr_grid.split(",")]
    k_grid = [int(x) for x in args.k_grid.split(",")]

    # ---- eval slice: G5's zero-shot arm verbatim -------------------------
    ev = load_frozen_parquet(eval_cfg)
    q_start = EVAL_STOP_ROWS + G5_N_SHOTS
    if len(ev) < q_start + args.n:
        raise SystemExit(f"--n {args.n}: eval file has {len(ev)} rows, needs {q_start + args.n}")
    qrows = ev.iloc[q_start : q_start + args.n]
    q_spans = list(
        zip(qrows["answer_char_start"].astype(int), qrows["answer_char_end"].astype(int))
    )
    # Zero-shot => few_shot_prompt([], text) == text, so spans pass through.
    q_examples = tokenize_with_spans(qrows["full_text"].tolist(), q_spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in q_examples]
    answers = qrows["true_answer"].astype(int).tolist()
    ops = qrows["op"].tolist()
    ab = list(zip(qrows["a"].astype(int), qrows["b"].astype(int)))
    plus_idx = [i for i, o in enumerate(ops) if o == "+"]
    minus_idx = [i for i, o in enumerate(ops) if o == "-"]

    # ---- training pool: D_target rows of --train-op the parent never saw -
    tgt = load_frozen_parquet(load_config(args.train_config, None))
    algo = load_frozen_parquet(load_config(args.algo_config, None))

    def qkey(a, b, op) -> np.ndarray:
        """Ordered triple as one int64 — operands are < 10^4 by construction."""
        return (a.to_numpy("int64") * 10_000 + b.to_numpy("int64")) * 2 + (op == "+").to_numpy()

    # Direct triples and commuted twins: for '+' the twin carries the identical
    # answer, so it is pre-exposure just as much as the direct question is.
    seen = np.unique(np.concatenate([qkey(algo.a, algo.b, algo.op), qkey(algo.b, algo.a, algo.op)]))
    pool = tgt[tgt.op == args.train_op]
    keep = ~np.isin(qkey(pool.a, pool.b, pool.op), seen)
    print(
        f"[evt] D_target '{args.train_op}' rows {len(pool)}; unseen by the parent "
        f"(direct or commuted) {int(keep.sum())} ({keep.mean():.2%}) — training draws from these"
    )
    pool = pool[keep]
    rng = np.random.default_rng(args.seed)
    max_k = max(k_grid)
    if len(pool) < max_k:
        raise SystemExit(f"pool has {len(pool)} unseen '{args.train_op}' rows, need {max_k}")
    draw = pool.iloc[rng.choice(len(pool), size=max_k, replace=False)]
    tr_spans = list(zip(draw["answer_char_start"].astype(int), draw["answer_char_end"].astype(int)))
    # append_eos=True: V5.43 — masked SFT trains no stop unless EOS is inside
    # the label span. The eval path above deliberately uses the default False.
    tr_examples = tokenize_with_spans(
        draw["full_text"].tolist(), tr_spans, tokenizer, append_eos=True
    )
    task_format = TaskFormat(
        name=eval_cfg["task"]["name"], format_version=eval_cfg["task"]["format_version"]
    )

    # ---- cost gate (budget rule) ----------------------------------------
    checkpoint = checkpoint_dir(args.run, store=store)
    model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    gpu = eval_cfg["gpu"]
    n_cells = len(rows_to_train) * len(lr_grid) * len(k_grid)
    tr_tok = sum(len(e.input_ids) for e in tr_examples)
    ev_tok = sum(len(p) + 12 for p in prompt_ids)
    flops = n_cells * (6.0 * n_params * tr_tok * args.steps + 2.0 * n_params * ev_tok)
    hours = flops / (gpu["tflops_bf16"] * 1e12 * gpu["utilization"] * 3600.0)
    print(
        f"[evt] {n_cells} cells = rows{rows_to_train} x lr{lr_grid} x k{k_grid}, "
        f"{args.steps} steps each"
    )
    print(
        f"[evt] estimated cost: ${hours * gpu['usd_per_hour']:,.2f} "
        f"({hours:.2f} GPU-h @ ${gpu['usd_per_hour']}/h)"
    )
    if not args.confirm_cost:
        print("[evt] --confirm-cost not given; refusing to train (budget rule). Exiting.")
        return 1

    # ---- untie, and prove the untie is a no-op ---------------------------
    probe_len = min(len(p) for p in prompt_ids[:8])
    probe = torch.tensor([p[:probe_len] for p in prompt_ids[:8]], device=args.device)
    with torch.no_grad():
        before = model(probe).logits.clone()
    untie_lm_head(model)
    model.to(args.device)
    with torch.no_grad():
        after = model(probe).logits
    drift = (before - after).abs().max().item()
    if drift != 0.0:
        raise SystemExit(f"untie changed the forward pass (max |dlogit| = {drift}) — aborting")
    print(
        f"[evt] untied lm_head from the embedding table; logits bitwise-identical (max d={drift})"
    )

    emb = model.get_input_embeddings().weight
    for p in model.parameters():
        p.requires_grad_(False)
    emb.requires_grad_(True)
    orig_emb = emb.detach().clone()
    # Spot-check set for the "only one row moved" assertion: layer 0 (closest to
    # the embedding, so first to drift if the freeze leaks) and lm_head (which
    # would move if the untie had silently failed).
    other_ref = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p is not emb and ("layers.0." in n or "lm_head" in n)
    }

    def evaluate() -> dict[str, float]:
        model.eval()
        completions = greedy_completions(
            model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
        )
        parsed = [parse_answer("Answer:" + c) for c in completions]
        hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
        rec = {
            "acc_overall": sum(hits) / len(hits),
            "acc_add": sum(hits[i] for i in plus_idx) / max(1, len(plus_idx)),
            "acc_sub": sum(hits[i] for i in minus_idx) / max(1, len(minus_idx)),
        }
        # Shapes follow --train-op so the forward and mirror runs carry the
        # same columns: it is always the trained operator's questions that are
        # asked "did the predictions move toward the right closed form?".
        tidx = plus_idx if args.train_op == "+" else minus_idx
        rec.update(shape_table([ab[i] for i in tidx], [parsed[i] for i in tidx]))
        return rec

    # The closed form a correct prediction takes for the trained operator, and
    # the one the parent defaults to instead (the audit's 19:1 |a-b| margin).
    right = "a+b" if args.train_op == "+" else "a-b"
    print("[evt] baseline (untouched parent) ...", flush=True)
    baseline = evaluate()
    print(
        f"[evt] baseline overall {baseline['acc_overall']:.4f}  "
        f"add {baseline['acc_add']:.4f}  sub {baseline['acc_sub']:.4f}  "
        f"| {right} {baseline['shape_' + right]:.4f}  |a-b| {baseline['shape_|a-b|']:.4f}"
    )
    if args.expect:
        tol = 0.5 / args.n
        bad = [
            f"{k}: got {baseline['acc_' + k]:.4f}, recorded {float(v):.4f}"
            for k, v in (it.split("=", 1) for it in args.expect.split(","))
            if abs(baseline["acc_" + k] - float(v)) > tol
        ]
        if bad:
            print(
                "[evt] PROTOCOL DRIFT — baseline does not reproduce the recorded G5 "
                "numbers, so nothing downstream explains them:\n  " + "\n  ".join(bad),
                file=sys.stderr,
            )
            return 1
        print("[evt] --expect: baseline reproduces the recorded G5 zero-shot accuracies")

    results = [
        {
            "arm": "baseline",
            "train_op": args.train_op,
            "row_token": "",
            "row_id": "",
            "lr": "",
            "k": 0,
            **baseline,
        }
    ]

    for token, row_id in zip(args.rows.split(","), rows_to_train):
        for k in k_grid:
            batch = tr_examples[:k]
            mask_full = label_mask(batch, task_format)
            ids_full = torch.zeros(
                (len(batch), mask_full.shape[1]), dtype=torch.long, device=args.device
            )
            for i, ex in enumerate(batch):
                ids_full[i, : len(ex.input_ids)] = torch.tensor(ex.input_ids, device=args.device)
            mask_full = mask_full.to(args.device)

            for lr in lr_grid:
                emb.data.copy_(orig_emb)
                row_mask = torch.zeros((emb.shape[0], 1), device=emb.device)
                row_mask[row_id] = 1.0
                handle = emb.register_hook(lambda g, m=row_mask: g * m)
                # weight_decay=0: masked rows have zero-but-not-None grads, so
                # decoupled decay would otherwise walk the whole 10000x512 table.
                opt = torch.optim.AdamW([emb], lr=lr, weight_decay=0.0)
                model.train()
                last = float("nan")
                for _ in range(args.steps):
                    opt.zero_grad(set_to_none=True)
                    total = 0.0
                    for s in range(0, k, args.micro_batch):
                        ids = ids_full[s : s + args.micro_batch]
                        m = mask_full[s : s + args.micro_batch]
                        loss = _mean_masked_ce_nats(model(ids).logits, ids, m)
                        (loss * (ids.shape[0] / k)).backward()
                        total += loss.item() * ids.shape[0] / k
                    opt.step()
                    last = total
                handle.remove()

                moved = (
                    ((emb.detach() - orig_emb).abs().sum(dim=1) > 0).nonzero().flatten().tolist()
                )
                # Empty is legal and is itself a result: a token absent from the
                # trained operator's prompts receives exactly zero input-side
                # gradient, so its row cannot move. Anything *other* than the
                # requested row moving is a leak.
                if set(moved) - {row_id}:
                    raise SystemExit(
                        f"expected only row {row_id} to move, got {len(moved)} rows {moved[:5]} — "
                        "the gradient mask or weight decay leaked"
                    )
                for n_, ref in other_ref.items():
                    if not torch.equal(dict(model.named_parameters())[n_].detach(), ref):
                        raise SystemExit(f"{n_} changed; only the embedding row may move")

                rec = evaluate()
                delta = (emb.detach()[row_id] - orig_emb[row_id]).norm().item()
                print(
                    f"[evt] row={token!r}({row_id}) k={k:<5d} lr={lr:<6g} "
                    f"loss={last:.4f} |drow|={delta:.4f} -> "
                    f"add {rec['acc_add']:.4f} (base {baseline['acc_add']:.4f})  "
                    f"sub {rec['acc_sub']:.4f} (base {baseline['acc_sub']:.4f})  "
                    f"{right} {rec['shape_' + right]:.4f}  |a-b| {rec['shape_|a-b|']:.4f}",
                    flush=True,
                )
                results.append(
                    {
                        # The discriminating variable turned out to be scope,
                        # not semantics: an operator token appears only in its
                        # own op's prompts, a prompt-general one in both, and
                        # the *other* operator's token in neither (delta == 0,
                        # no input-side gradient exists for it to receive).
                        "arm": (
                            "operator"
                            if row_id == OP_ROW[args.train_op]
                            else ("absent" if delta == 0.0 else "prompt_general")
                        ),
                        "train_op": args.train_op,
                        "row_token": token,
                        "row_id": row_id,
                        "lr": lr,
                        "k": k,
                        "train_loss_nats": last,
                        "row_delta_l2": delta,
                        **rec,
                    }
                )
    emb.data.copy_(orig_emb)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lead = [
            "arm",
            "train_op",
            "row_token",
            "row_id",
            "lr",
            "k",
            "train_loss_nats",
            "row_delta_l2",
        ]
        fields = lead + [k for k in results[0] if k not in lead]
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        meta = args.out.with_suffix(".meta.json")
        meta.write_text(
            json.dumps(
                {
                    "run": args.run,
                    "checkpoint": str(checkpoint),
                    "rows": dict(zip(args.rows.split(","), rows_to_train)),
                    "lr_grid": lr_grid,
                    "k_grid": k_grid,
                    "steps": args.steps,
                    "n_eval": args.n,
                    "seed": args.seed,
                    "train_op": args.train_op,
                    "train_pool": (
                        f"D_target op='{args.train_op}' minus D_algo (direct and commuted)"
                    ),
                    "baseline": baseline,
                },
                indent=2,
            )
        )
        print(f"[evt] wrote {args.out} and {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
