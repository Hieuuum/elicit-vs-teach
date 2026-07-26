"""Per-example zero-shot predictions on D_target_eval — diagnostic dump, no verdict.

Answers "what did the dose actually change?" behind G5's headline accuracy.
The recorded numbers say the elicit parent scores 0.1016 zero-shot and the n=1
dose checkpoint 0.0840 on the identical 1,024 operator add/sub questions, but a
marginal accuracy cannot say whether the two models miss the *same* questions.
This writes one row per question with every run's completion side by side, so
the paired outcome (both right / only one / neither) is readable.

Protocol is G5's zero-shot arm verbatim (``gates.py`` ``run_g5.accuracy_with``
with ``exemplars == []``): the same fixed slice of the frozen D_target_eval
reporting block, the same token-prefix prompts, the same greedy EOS-stopped
decode, the same parser. Duplicated rather than factored out because the
guarantee that matters here is empirical, not structural — each run's accuracy
is printed and ``--expect`` fails the script against the recorded G5 numbers,
so protocol drift is a loud error instead of a plausible CSV.

Prompts are token-level prefixes of the training tokenization, never
re-tokenized char slices: half of D_target_eval is subtraction with negative
answers, exactly the surface of the 2026-07-21 sign-drop incident (a
re-tokenized prompt ends in a standalone space token and makes the merged
`` -`` sign token unreachable). ``prompt_decoded`` is carried into the CSV
next to the parquet's own ``prompt_text`` so any divergence is visible in the
artifact itself.

Evidence only: writes a CSV, touches no manifest, needs no ``--confirm-cost``
(evaluation, not training).

Usage:
    python3 dump_g5_predictions.py --config configs/eval_target_data.yaml \
        --run n0=evt-run2-armA-algo --run n1=evt-p2-armA-dose1 \
        --out g5_predictions.csv --expect n0=0.1016,n1=0.0840
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import torch

from geode.arith import exact_match, greedy_completions, parse_answer, tokenize_with_spans
from geode.zoo import checkpoint_dir, load_model
from gates import G5_N_SHOTS
from train import REPO_ROOT, load_config
from train_sft import load_frozen_parquet
from train_target import EVAL_STOP_ROWS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="eval-data YAML (configs/eval_target_data.yaml)"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_ID",
        help="repeatable; LABEL names the run's columns in the CSV",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1024, help="questions; G5's default is 1024")
    parser.add_argument(
        "--expect",
        default=None,
        help="recorded G5 zero-shot accuracies as 'LABEL=ACC,...' — the protocol-drift "
        "check; a mismatch beyond half a question exits non-zero",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    runs = [s.split("=", 1) for s in args.run]
    labels = [label for label, _ in runs]

    cfg = load_config(args.config, None)
    os.environ.setdefault("GEODE_STORE", str(REPO_ROOT / "geode-store"))
    store = Path(os.environ["GEODE_STORE"])

    from transformers import AutoTokenizer

    local = (args.config.parent / cfg["tokenizer"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(local if local.is_dir() else cfg["tokenizer"]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = load_frozen_parquet(cfg)  # D_target_eval, order_hash verified
    # G5's fixed slice: skip the in-loop stopping block, then the 16 rows the
    # few-shot arm uses as exemplars, so the zero-shot questions are identical
    # to the ones the recorded accuracies were measured on.
    q_start = EVAL_STOP_ROWS + G5_N_SHOTS
    if len(df) < q_start + args.n:
        raise SystemExit(f"--n {args.n}: eval file has {len(df)} rows, needs {q_start + args.n}")
    rows = df.iloc[q_start : q_start + args.n]

    char_spans = list(
        zip(rows["answer_char_start"].astype(int), rows["answer_char_end"].astype(int))
    )
    examples = tokenize_with_spans(rows["full_text"].tolist(), char_spans, tokenizer)
    prompt_ids = [ex.input_ids[: ex.label_span[0]] for ex in examples]
    answers = rows["true_answer"].astype(int).tolist()
    ops = rows["op"].tolist()

    results: dict[str, tuple[list[str], list[int | None], list[bool]]] = {}
    accuracies: dict[str, float] = {}
    for label, run_id in runs:
        checkpoint = checkpoint_dir(run_id, store=store)
        print(f"[evt] {label}: loading checkpoint {checkpoint} ...", flush=True)
        model = load_model(run_id, store=store, device=args.device, checkpoint=checkpoint)
        completions = greedy_completions(
            model, tokenizer, prompt_ids, device=args.device, batch_size=args.batch_size
        )
        # The completion starts inside the answer slot, so "Answer:" + completion
        # re-enters the tested parser with the marker it expects.
        parsed = [parse_answer("Answer:" + c) for c in completions]
        hits = [exact_match("Answer:" + c, a) for c, a in zip(completions, answers)]
        results[label] = (completions, parsed, hits)
        accuracies[label] = sum(hits) / len(hits)
        by_op = {
            op: sum(h for h, o in zip(hits, ops) if o == op) / max(1, ops.count(op))
            for op in sorted(set(ops))
        }
        negs = [i for i, a in enumerate(answers) if a < 0]
        neg_acc = sum(hits[i] for i in negs) / max(1, len(negs))
        print(
            f"[evt] {label} ({run_id}) zero-shot exact_match {accuracies[label]:.4f} "
            f"on n={args.n} (by op: {by_op}; negative answers: {neg_acc:.4f} on n={len(negs)})"
        )

    if args.expect:
        # The one check that certifies this dump explains the recorded G5
        # numbers rather than a differently-built eval. Tolerance is half a
        # question, i.e. only the 4-dp rounding of the recorded value.
        tol = 0.5 / args.n
        bad = []
        for item in args.expect.split(","):
            label, value = item.split("=", 1)
            if label not in accuracies:
                raise SystemExit(f"--expect names unknown label {label!r}")
            if abs(accuracies[label] - float(value)) > tol:
                bad.append(f"{label}: got {accuracies[label]:.4f}, recorded {float(value):.4f}")
        if bad:
            print(
                "[evt] PROTOCOL DRIFT — this dump does not reproduce the recorded G5 "
                "zero-shot accuracies, so its rows do not explain them:\n  " + "\n  ".join(bad),
                file=sys.stderr,
            )
            return 1
        print(f"[evt] --expect: all {len(labels)} runs reproduce their recorded G5 accuracy")

    header = [
        "idx",
        "op",
        "a",
        "b",
        "cell",
        "x_digits",
        "y_digits",
        "true_answer",
        "answer_negative",
        "prompt_text",
        "prompt_decoded",
        "correct_in",
        "n_correct",
        "same_answer",
    ]
    for label in labels:
        header += [f"{label}_output", f"{label}_parsed", f"{label}_correct"]

    meta = {c: rows[c].tolist() for c in ("idx", "op", "a", "b", "cell", "x_digits", "y_digits")}
    out_rows = []
    for i in range(len(rows)):
        ok = [label for label in labels if results[label][2][i]]
        parsed_i = [results[label][1][i] for label in labels]
        record = {
            **{c: meta[c][i] for c in meta},
            "true_answer": answers[i],
            "answer_negative": answers[i] < 0,
            "prompt_text": rows["prompt_text"].iloc[i],
            "prompt_decoded": tokenizer.decode(prompt_ids[i]),
            "correct_in": ",".join(ok) if ok else "none",
            "n_correct": len(ok),
            "same_answer": len(set(parsed_i)) == 1,
        }
        for label in labels:
            completions, parsed, hits = results[label]
            record[f"{label}_output"] = completions[i]
            record[f"{label}_parsed"] = parsed[i]
            record[f"{label}_correct"] = hits[i]
        # Discordant rows first: where the runs disagree is the only place a
        # side-by-side dump carries information the marginal accuracies don't.
        rank = 0 if 0 < len(ok) < len(labels) else (1 if ok else 2)
        out_rows.append((rank, i, record))
    out_rows.sort(key=lambda t: (t[0], t[1]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for _, _, record in out_rows:
            writer.writerow(record)

    counts = Counter(record["correct_in"] for _, _, record in out_rows)
    print(f"[evt] paired outcome over n={args.n} (rows sorted discordant-first):")
    for key, count in counts.most_common():
        print(f"  correct in {key:>12}: {count}")
    print(f"[evt] wrote {len(out_rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
