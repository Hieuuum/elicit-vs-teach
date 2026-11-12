"""Pairing gap (PG, the paper's link gap §4.3 Metric 1) on one ts1b parent (HANDOFF-ts1b-pg.md).

Does a model already link the right answer to its question? Read from its losses alone on the
1,024 validation problems of data/ceiling4x4/fold_00, against a control set (the same prompts
with a wrong answer of the same digit count), by Marcin's `geode.edl.pairing.control_read`
(V1.15, used unchanged):

  L(a | q)   summed NLL in nats over the 2 answer tokens + EOS (3 label positions; training
             put EOS inside the label span)
  m(i, j)    L(a_j|q_i) + L(a_i|q_j) - [L(a_i|q_i) + L(a_j|q_j)]   (prompt/answer-only effects cancel)
  acc*       share of pairs with m(i, j; data) > m(i, j; control), ties within 1e-9 count 1/2;
             95% CI by bootstrap over items; s = clip(2 acc* - 1, 0, 1); call elicit iff CI lo > 1/2
  pairs      make_pairs(n, offsets=4, groups=digit count): ~4,096 pairs, both answers of a pair
             have the same digit count

Token layout comes from the stored tensors (no re-tokenized prompt): prompt = clean_ids[:-1],
answer = [c1, c2, EOS]; a swapped item (i, j) is prompt_i + [c1_j, c2_j, EOS]. The control answer
d_i follows toy_pg.shape_matched_wrong (same sign and digit count, != a_i, seeded uniform) and is
redrawn until it tokenizes to 2 tokens that round-trip (data.answer_tokens); one draw with seed
CONTROL_SEED, so the word and symbol forms share the same d_i.

Secondary read (not pre-registered): control_read's item bootstrap treats per-item means as
independent although every pair counts for two items, so its CI is ~1.7x too narrow under a null
(13% false elicit at nominal 2.5%). `jackknife_read` gives a delete-one-item jackknife SE of the
same acc* (null false elicit ~1%); keys acc_se_jackknife, acc_ci_jackknife, call_jackknife.

    python3 ts1b_pg.py --tag ep_symbol --model podhajskimarcin/evt-ts1b-op-bridge-mix \
        --form symbol --data-dir data/ceiling4x4/fold_00 --out results/ts1b_hard_acc/pg/ep_symbol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import run  # noqa: E402
from data import answer_tokens  # noqa: E402

from geode.edl.pairing import TIE_NATS, control_read, make_pairs, pair_losses  # noqa: E402

CONTROL_SEED = 20261112
JK_Z = 1.96
OFFSETS = 4
N_LABEL = 3  # c1, c2, EOS
MAX_REDRAWS = 1000  # per item; a 2-token answer of the same digit count is the common case
TOKENIZER = "meta-llama/Llama-3.2-1B"  # the ts1b models' tokenizer (training configs)


# ----------------------------------------------------------------------------- control


def shape_matched_wrong(answer: str, rng: random.Random) -> str:
    """A uniform random number with the same sign and digit count as ``answer``, never equal to
    it (the rule of toy_pg.shape_matched_wrong, V1.15)."""
    neg = answer.startswith("-")
    digits = answer.lstrip("-")
    k = len(digits)
    while True:
        cand = str(rng.randint(0, 9)) if k == 1 else str(rng.randint(10 ** (k - 1), 10**k - 1))
        if cand != digits:
            return ("-" if neg else "") + cand


def control_answers(
    answers: Sequence[str], tokenizer, seed: int = CONTROL_SEED
) -> tuple[list[str], list[list[int]], int]:
    """One control answer per item, in order, from one seeded stream: shape_matched_wrong,
    redrawn until ``data.answer_tokens`` accepts it (2 tokens, round-trips, first token not
    whitespace). Returns (texts, token ids, number of redraws)."""
    rng = random.Random(seed)
    texts, ids, redraws = [], [], 0
    for a in answers:
        for _ in range(MAX_REDRAWS):
            d = shape_matched_wrong(a, rng)
            try:
                t = answer_tokens(tokenizer, d)
            except ValueError:
                redraws += 1
                continue
            texts.append(d)
            ids.append(t)
            break
        else:
            raise RuntimeError(
                f"control_answers: no 2-token control for {a!r} in {MAX_REDRAWS} draws"
            )
    return texts, ids, redraws


def digit_count(answer: str) -> int:
    return len(answer.lstrip("-"))


def _sha(xs: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(xs).encode()).hexdigest()


# ----------------------------------------------------------------------------- items


def build_items(val: dict, tokenizer, eos_id: int) -> dict:
    """Prompts (n, P), data and control answers (n, 3) = [t1, t2, EOS], digit-count groups,
    and the control check. Raises if the stored layout is not prompt + c1."""
    ids, c1, c2 = val["clean_ids"], val["c1"], val["c2"]
    if not torch.equal(ids[:, -1], c1):
        raise ValueError(
            "build_items: clean_ids[:, -1] != c1 (expected prompt + first answer token)"
        )
    answers = [str(a) for a in val["answer_text"]]
    n = len(answers)
    eos = torch.full((n, 1), eos_id, dtype=torch.long)
    texts, ctrl_ids, redraws = control_answers(answers, tokenizer)
    true_tok = [answer_tokens(tokenizer, a) for a in answers]
    check = {
        "n": n,
        "digit_count_ok": all(digit_count(d) == digit_count(a) for a, d in zip(answers, texts)),
        "sign_ok": all(d.startswith("-") == a.startswith("-") for a, d in zip(answers, texts)),
        "never_equal": all(d != a for a, d in zip(answers, texts)),
        "two_tokens_roundtrip": all(
            len(t) == 2 and tokenizer.decode(t) == d for t, d in zip(ctrl_ids, texts)
        ),
        "true_tokens_match": true_tok == torch.stack([c1, c2], 1).tolist(),
        "n_redraws": redraws,
        "sha256": _sha(texts),
    }
    check["pass"] = all(
        check[k]
        for k in ("digit_count_ok", "sign_ok", "never_equal", "two_tokens_roundtrip",
                  "true_tokens_match")
    )  # fmt: skip
    return {
        "prompts": ids[:, :-1].clone(),
        "data": torch.cat([c1[:, None], c2[:, None], eos], 1),
        "control": torch.cat([torch.tensor(ctrl_ids, dtype=torch.long), eos], 1),
        "answer_text": answers,
        "control_text": texts,
        "groups": [digit_count(a) for a in answers],
        "control_check": check,
        "data_sha256": _sha(answers),
    }


# ----------------------------------------------------------------------------- losses


def build_batch(prompts: torch.Tensor, answers: torch.Tensor, idx: Sequence[tuple[int, int]]):
    """Rows prompt_q + answer_a for each (q, a) in ``idx``: (len(idx), P + 3)."""
    q = torch.tensor([i for i, _ in idx], dtype=torch.long)
    a = torch.tensor([j for _, j in idx], dtype=torch.long)
    return torch.cat([prompts[q], answers[a]], 1)


@torch.no_grad()
def item_losses(
    model, prompts: torch.Tensor, answers: torch.Tensor, idx: Sequence[tuple[int, int]],
    batch_size: int,
) -> tuple[list[float], list[int]]:  # fmt: skip
    """Summed NLL (nats, float32 logits) of answer_a after prompt_q over its N_LABEL label
    positions, per (q, a) in ``idx``; and the label token counts (all N_LABEL)."""
    dev = next(model.parameters()).device
    out: list[float] = []
    for s in range(0, len(idx), batch_size):
        part = idx[s : s + batch_size]
        x = build_batch(prompts, answers, part).to(dev)
        # positions T-4..T-1; T-4..T-2 predict the N_LABEL answer tokens, T-1 (after EOS) unused
        logits = model(x, logits_to_keep=N_LABEL + 1).logits[:, :N_LABEL].float()
        lp = logits.log_softmax(-1).gather(-1, x[:, -N_LABEL:, None]).squeeze(-1)
        out += (-lp.sum(-1)).cpu().tolist()
    return out, [N_LABEL] * len(idx)


def scorer(model, prompts: torch.Tensor, answers: torch.Tensor, batch_size: int) -> Callable:
    """``score(idx) -> (summed nats, token counts)`` over (prompt index, answer index), the form
    ``pairing.pair_losses`` takes."""
    return lambda idx: item_losses(model, prompts, answers, idx, batch_size)


def jackknife_read(
    pairs: Sequence[tuple[int, int]], margins_data: Sequence[float],
    margins_control: Sequence[float], n: int,
) -> dict:  # fmt: skip
    """Delete-one-item jackknife of acc* (the calibrated secondary read). Pair wins as in
    control_read (ties within TIE_NATS count 1/2); leaving item k out drops every pair it is in:
    loo_k = (W - w_k) / (P - p_k); SE = sqrt((n-1)/n * sum (loo_k - mean)^2); CI = acc +- 1.96 SE;
    call 'elicit' iff the CI's low end > 1/2."""
    wins = []
    for a, b in zip(margins_data, margins_control):
        d = a - b
        wins.append(1.0 if d > TIE_NATS else (0.0 if d < -TIE_NATS else 0.5))
    W, P = sum(wins), len(wins)
    if P == 0:
        raise ValueError("jackknife_read: no pairs")
    wk, pk = [0.0] * n, [0] * n
    for (i, j), c in zip(pairs, wins):
        wk[i] += c
        wk[j] += c
        pk[i] += 1
        pk[j] += 1
    if any(pk[k] == P for k in range(n)):
        raise ValueError("jackknife_read: an item is in every pair")
    loo = [(W - wk[k]) / (P - pk[k]) for k in range(n)]
    m = sum(loo) / n
    se = math.sqrt((n - 1) / n * sum((x - m) ** 2 for x in loo))
    acc = W / P
    ci = [acc - JK_Z * se, acc + JK_Z * se]
    return {
        "acc_se_jackknife": se,
        "acc_ci_jackknife": ci,
        "call_jackknife": "elicit" if ci[0] > 0.5 else "teach",
    }


def pg_read(
    score_data: Callable, score_control: Callable, n: int, groups: Sequence[int], seed: int
) -> dict:
    """control_read over make_pairs(n, OFFSETS, groups): Marcin's keys plus the pairs, the
    mean true-item losses, and whether every pair stays within one group."""
    pairs = make_pairs(n, OFFSETS, groups=groups)
    d_true, d_cnt, d_cross = pair_losses(score_data, n, pairs)
    c_true, c_cnt, c_cross = pair_losses(score_control, n, pairs)
    rec = control_read((d_true, d_cross), (c_true, c_cross), pairs, seed=seed,
                       tokens=(d_cnt, c_cnt))  # fmt: skip
    rec |= jackknife_read(pairs, rec["margins_data"], rec["margins_control"], n)
    rec["pairs"] = [list(p) for p in pairs]
    rec["pairs_within_group"] = all(groups[i] == groups[j] for i, j in pairs)
    rec["true_loss_data_mean_nats"] = sum(d_true) / n
    rec["true_loss_control_mean_nats"] = sum(c_true) / n
    rec["n_sequences"] = 2 * (n + 2 * len(pairs))
    return rec


def read(model, tokenizer, val: dict, a: argparse.Namespace) -> dict:
    """The full JSON record of one read."""
    t0 = time.time()
    items = build_items(val, tokenizer, tokenizer.eos_token_id)
    n = len(items["answer_text"])
    body = pg_read(
        scorer(model, items["prompts"], items["data"], a.batch_size),
        scorer(model, items["prompts"], items["control"], a.batch_size),
        n, items["groups"], a.seed,
    )  # fmt: skip
    return {
        "tag": a.tag,
        "model": a.model,
        "form": a.form,
        "data_dir": str(a.data_dir),
        "n_items": n,
        "groups": {str(k): v for k, v in sorted(Counter(items["groups"]).items())},
        "offsets": OFFSETS,
        "seed": a.seed,
        "control_seed": CONTROL_SEED,
        "eos_id": int(tokenizer.eos_token_id),
        "prompt_len": int(items["prompts"].shape[1]),
        "n_label": N_LABEL,
        "data_sha256": items["data_sha256"],
        "control_check": items["control_check"],
        **body,
        "elapsed_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------- cli


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--form", choices=["word", "symbol"], required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=run.SEED)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    tok = load_tokenizer()
    model = run.load_model(a.model, a.device)
    val = run.load_data(a.data_dir, a.form)["val"]
    rec = read(model, tok, val, a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec))
    lo, hi = rec["acc_ci"]
    print(
        f"[pg] {a.tag} {a.form}: acc* {rec['acc']:.3f} [{lo:.3f}, {hi:.3f}] s {rec['s']:.3f} "
        f"call {rec['call']} (jackknife [{rec['acc_ci_jackknife'][0]:.3f}, "
        f"{rec['acc_ci_jackknife'][1]:.3f}] {rec['call_jackknife']}) | m21/token data "
        f"{rec['m21_data_nats_per_token']:+.4f} control {rec['m21_control_nats_per_token']:+.4f}"
        f" | {rec['n_pairs']} pairs, control check "
        f"{'pass' if rec['control_check']['pass'] else 'FAIL'}, {rec['elapsed_s']:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
