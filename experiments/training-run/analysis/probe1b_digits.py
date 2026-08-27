"""Phase 1 of the probe1b digit-probe study (``docs/plan-probe1b-digits
-phase1.md`` §3-9): are the answer digits of 4-digit addition linearly
readable from the residual stream of ``meta-llama/Llama-3.2-1B`` (web-
pretrained) but not from ``podhajskimarcin/evt-ts1b-base`` (stories-
pretrained, architecture- and tokenizer-identical negative control)? No
training — a per-layer, per-digit linear probe plus a zero-capacity logit-
lens control, at two prompt formats (``op``/``nl``) and two probe placements
(B = plan-ahead, before the answer; C = teacher-forced, at each answer
token), against a per-digit no-carry "cheat" baseline that isolates the rows
a naive digit-wise readout would get wrong (plan §2, R-P1..R-P4).

``fit_linear_probe_predictions`` below is copied (not imported) from
``experiments/training-run/analysis/resid_probe.py``, extended to also
return the test-set log-probabilities of every class — that module's
docstring explains why the copy: importing a sibling pulls in whatever that
sibling imports, and this module must import cleanly with no ``matplotlib``
on the path (script-tier convention; ``plot_probe1b.py`` is the only place
in this pair that touches a plotting library). No network call happens at
import time — ``AutoModelForCausalLM``/``AutoTokenizer`` are only
constructed, never loaded, until a CLI stage that explicitly asks for a
model runs.

Losses/log-probs are in **nats** (repo convention, CLAUDE.md); every stored
log-prob field name says so (``mean_logprob_nats``) or is documented as nats
in this docstring. Model loading lives in the CLI layer only — every
public function below takes an already-loaded ``model``/``tok``, which is
what makes CPU-only, no-network unit testing possible (see
``tests/experiments/analysis/test_probe1b_digits.py``).

Plan-ahead vs teacher-forced digit readout follows Baeumel et al.
(arXiv:2502.19981): a chunked numeric tokenizer (Llama's "691"+"2" 3-digit
chunking) lets a model plan the whole answer before emitting its first
token, so placement B (read before any answer token is generated) and
placement C (teacher-forced at each answer token) are expected to differ
little for a model with latent arithmetic — R-P3 is descriptive, not a
pass/fail gate.

Any path that runs a model forward pass at scale (``--extract``,
``--behavior``, and ``--all``, which includes both) costs real GPU money and
is refused without ``--confirm-cost`` (CLAUDE.md budget rule); the estimate
printed is plan §11's. ``--fit`` re-derives probes and the logit lens from
already-saved ``features_*.pt`` tensors (no fresh forward pass over the full
6000-row corpus) and is not cost-gated.

Usage:
    python3 probe1b_digits.py --make-data
    python3 probe1b_digits.py --extract --confirm-cost
    python3 probe1b_digits.py --fit
    python3 probe1b_digits.py --behavior --confirm-cost
    python3 probe1b_digits.py --all --confirm-cost
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

N_DIGITS = 4
DEFAULT_SEED = 316
L2 = 1e-3
MAX_ITER = 150  # L-BFGS iterations per fit — resid_probe.py's own measured budget
LENS_ATOL_NATS = 1e-3

HEADS: tuple[str, ...] = ("d1", "d2", "d3", "d4")
MODEL_CHOICES: tuple[str, ...] = ("llama", "ts1b")
FORMAT_CHOICES: tuple[str, ...] = ("op", "nl")
COST_GATED_STAGES = frozenset({"extract", "behavior"})

DEFAULT_LLAMA_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_TS1B_ID = "podhajskimarcin/evt-ts1b-base"
DEFAULT_TOKENIZER_ID = "meta-llama/Llama-3.2-1B"
# ts1b's weights live under this subfolder of the model repo (plan §3); its
# generation_config.json ships stale LlamaConfig bos/eos defaults — every
# special-token decision in this module comes from the tokenizer instead.
TS1B_SUBFOLDER = "runs/evt-ts1b-base/model"
DEFAULT_OUT = Path("results/probe1b_phase1")

PROBE_ROW_COLUMNS: list[str] = [
    "model",
    "format",
    "layer",
    "placement",
    "head",
    "top1_acc_all",
    "top1_acc_affected",
    "mean_logprob_all",
    "mean_logprob_affected",
    "n_all",
    "n_affected",
    "shuffled_top1_all",
    "shuffled_top1_affected",
    "cheat_acc_all",
    "cheat_acc_affected",
    "majority_acc",
    "prod_prob_all",
]
LENS_ROW_COLUMNS: list[str] = [
    "model",
    "format",
    "layer",
    "position",
    "top1_acc",
    "mean_logprob_nats",
    "n",
]
BEHAVIOR_ROW_COLUMNS: list[str] = ["model", "format", "em", "n_exact", "n"]

COST_ESTIMATE = (
    "[probe1b] estimated cost (plan Sec 11): downloads ~8 GB; extraction "
    "4x(6000 rows, ~25 tok, 1B bf16) ~10 min; 680 L-BFGS fits ~30-45 min "
    "(--fit, not gated here); behavior ~10 min. Total <=2h at $0.30-0.40/h "
    "GPU -> <=$1 compute."
)


# --------------------------------------------------------------------------
# Dataset: per-digit "cheat" baseline and the 6000-row addition corpus.
# --------------------------------------------------------------------------


def cheat_and_affected(a: int, b: int, ans: int) -> tuple[list[int], list[bool]]:
    """Per-digit no-carry baseline and where it is wrong (plan §4).

    ``ans`` has ``K = len(str(ans))`` digits, most-significant first
    (``k=1``). ``cheat_k = (digit_k(a) + digit_k(b)) % 10`` where
    ``digit_k(x)`` is ``x``'s digit at the SAME place value as answer digit
    ``k`` (missing digits of ``a``/``b`` at that place count as 0).
    ``affected_k`` is ``True`` exactly where the cheat is wrong — this is
    the discriminator for "a naive digit-wise readout would fail here", not
    a carry-chain detector: e.g. 4759+4249=9008 has a real carry out of the
    units digit (9+9=18) but ``(9+9)%10 == 8 == true units digit``, so
    ``affected`` is ``False`` there — carries can cancel out mod 10 even
    though the addition was not carry-free.

    Worked example (plan §4): ``cheat_and_affected(47, 85, 132)`` returns
    ``([0, 2, 2], [True, True, False])``.
    """
    digits = str(ans)
    k_total = len(digits)
    cheats: list[int] = []
    affected: list[bool] = []
    for i, ch in enumerate(digits):
        place = k_total - 1 - i
        da = (a // (10**place)) % 10
        db = (b // (10**place)) % 10
        cheat = (da + db) % 10
        cheats.append(cheat)
        affected.append(cheat != int(ch))
    return cheats, affected


def make_data(n: int = 6000, seed: int = 316) -> pd.DataFrame:
    """The probe1b addition corpus (plan §4): ``n`` rows of ``a + b = ans``
    with ``1000 <= ans <= 9999`` so every answer is exactly ``N_DIGITS``
    digits (one 3-digit + one 1-digit tokenizer chunk).

    Draws ``a, b ~ rng.integers(100, 9900)`` i.i.d. in batches, keeping rows
    with ``1000 <= a + b <= 9999``, deduplicated on the ordered pair
    ``(a, b)`` (first occurrence in draw order wins), until exactly ``n``
    rows are collected. The SAME rng stream then produces a half/half
    train/test permutation (``rng.permutation(n)``, first half "train").

    Raises ``ValueError`` (naming ``--n``) if ``affected4`` is not
    identically False (d4 is the units digit — never carry-affected by
    ``cheat_and_affected``'s definition, for any valid 4-digit sum) or if
    any of ``affected1``..``affected3``, restricted to the test half, has
    fewer than 400 True rows — the floor plan §4 sets for a usable
    affected-subset probe accuracy.
    """
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    while len(rows) < n:
        a_batch = rng.integers(100, 9900, size=n)
        b_batch = rng.integers(100, 9900, size=n)
        for a, b in zip(a_batch.tolist(), b_batch.tolist()):
            if len(rows) >= n:
                break
            ans = a + b
            if not (1000 <= ans <= 9999):
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            cheats, affected = cheat_and_affected(a, b, ans)
            ans_digits = [int(c) for c in str(ans)]
            row = {"a": a, "b": b, "ans": ans}
            row.update({f"d{i + 1}": ans_digits[i] for i in range(N_DIGITS)})
            row.update({f"cheat{i + 1}": cheats[i] for i in range(N_DIGITS)})
            row.update({f"affected{i + 1}": affected[i] for i in range(N_DIGITS)})
            rows.append(row)

    df = pd.DataFrame(rows[:n])
    perm = rng.permutation(n)
    train_rows = set(perm[: n // 2].tolist())
    df["split"] = ["train" if i in train_rows else "test" for i in range(n)]

    if not (~df["affected4"]).all():
        raise ValueError(
            "make_data: affected4 is not all False — d4 (units) should never be "
            "carry-affected for a 4-digit sum; investigate before raising --n"
        )
    test_df = df[df["split"] == "test"]
    for k in (1, 2, 3):
        count = int(test_df[f"affected{k}"].sum())
        if count < 400:
            raise ValueError(
                f"make_data: affected{k} has only {count} True rows in the test "
                f"half (need >= 400) — raise --n and regenerate"
            )
    return df


# --------------------------------------------------------------------------
# Prompt rendering and answer-token span location.
# --------------------------------------------------------------------------


def render(a: int, b: int, ans: int, fmt: str) -> tuple[str, str, str]:
    """``(full_text, prompt_text, prev_char)`` for one row in format
    ``fmt`` (plan §5): ``full_text`` is the prompt plus the true answer
    (probe/lens/behavior-eval input), ``prompt_text`` is ``full_text`` with
    the answer removed (the behavioral-eval generation prompt), and
    ``prev_char`` is the single character immediately preceding the answer
    — the span assert in ``answer_positions`` checks against it.

    - ``fmt="op"``: ``"{a} + {b} = {ans}"``; ``prev_char=" "``.
    - ``fmt="nl"``: ``"What is the sum of {a} and {b}?\\n{ans}"``;
      ``prev_char="\\n"``.
    """
    if fmt == "op":
        return f"{a} + {b} = {ans}", f"{a} + {b} = ", " "
    if fmt == "nl":
        return f"What is the sum of {a} and {b}?\n{ans}", f"What is the sum of {a} and {b}?\n", "\n"
    raise ValueError(f"render: unknown fmt {fmt!r} (expected 'op' or 'nl')")


def answer_positions(tok: Any, text: str, ans: str, prev_char: str) -> tuple[int, int]:
    """Token positions that generate the two answer chunks (plan §5).

    Tokenizes ``text`` with offsets, finds the (exactly two, for a
    4-digit answer under Llama's left-to-right 3-digit chunking) tokens
    whose non-zero-width offset span overlaps the answer's char span (the
    final ``len(ans)`` characters of ``text`` — asserted via
    ``text.endswith(ans)``), and hard-asserts the tokenization is exactly
    what plan §5 requires: two answer tokens, decoding (via
    ``text[start:end]``, never ``tok.decode`` — offsets alone must settle
    it, so the same code path works on the real tokenizer and a test
    mock) to ``ans[:3]`` then ``ans[3:]``, adjacent, immediately preceded
    by ``prev_char``. Every assert names the full offending ``text`` (plan
    §5: "fail loudly, print the offending row") and is never relaxed — an
    assert firing on real data means the tokenizer or prompt changed
    underneath this experiment, not that the check is too strict.

    Returns ``(pos1, pos2) = (idx_token1 - 1, idx_token1)`` — the residual
    position that CAUSES each answer token (a causal LM predicts token
    ``i`` from position ``i - 1``), so ``pos2`` also equals answer token
    1's own index.
    """
    assert text.endswith(ans), (
        f"answer_positions: text does not end with ans: text={text!r} ans={ans!r}"
    )
    ans_start = len(text) - len(ans)
    ans_end = len(text)

    # Contract: uses ONLY enc["input_ids"] (implicitly, via the tokenizer call
    # surface a real/mock tokenizer must support) and enc["offset_mapping"]
    # (used directly below) — token text is read via text[start:end], never
    # tok.decode, so the same logic works on the real tokenizer and a mock.
    enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    answer_idxs = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > start and start < ans_end and end > ans_start
    ]
    assert len(answer_idxs) == 2, (
        f"answer_positions: expected exactly 2 answer tokens, got {len(answer_idxs)} "
        f"for text={text!r}"
    )
    idx1, idx2 = answer_idxs
    tok1_start, tok1_end = offsets[idx1]
    tok2_start, tok2_end = offsets[idx2]
    tok1_text = text[tok1_start:tok1_end]
    tok2_text = text[tok2_start:tok2_end]
    assert tok1_text == ans[:3], (
        f"answer_positions: token1 text {tok1_text!r} != ans[:3] {ans[:3]!r} for text={text!r}"
    )
    assert tok2_text == ans[3:], (
        f"answer_positions: token2 text {tok2_text!r} != ans[3:] {ans[3:]!r} for text={text!r}"
    )
    assert idx2 == idx1 + 1, (
        f"answer_positions: token2 index {idx2} != token1 index {idx1} + 1 for text={text!r}"
    )
    prev = text[ans_start - 1] if ans_start > 0 else ""
    assert prev == prev_char, (
        f"answer_positions: char before answer {prev!r} != expected {prev_char!r} for text={text!r}"
    )
    return idx1 - 1, idx1


# --------------------------------------------------------------------------
# Feature extraction (residual stream at pos1/pos2, every hidden-state layer).
# --------------------------------------------------------------------------


def extract_features(
    model: Any,
    tok: Any,
    texts: list[str],
    positions: list[tuple[int, int]],
    token_ids: list[tuple[int, int]],
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass per batch over ``texts`` (plan §6): gathers the
    hidden state at ``(pos1, pos2)`` for every ``output_hidden_states``
    layer, plus the model's own final-layer log-probability of each row's
    two true answer tokens (the free behavioral anchor the ``--fit`` driver
    cross-checks the logit lens against).

    Batches with RIGHT padding (caller has set ``tok.pad_token =
    tok.eos_token``) — padding only appends after each row's real tokens,
    so ``positions`` computed on the unpadded per-row tokenization stay
    valid inside a padded batch.

    Returns ``(features, ans_logprobs)``:

    - ``features``: float32 CPU tensor ``[N, n_hidden, 2, d]``,
      ``n_hidden = len(outputs.hidden_states)`` (embeddings + one per
      block); axis 2 is ``(pos1, pos2)``.
    - ``ans_logprobs``: float32 CPU tensor ``[N, 2]`` —
      ``log_softmax`` (computed in float32) of the model's final logits at
      ``pos1`` evaluated at answer-token-1's id, and at ``pos2`` for
      answer-token-2's id.
    """
    model.eval()
    n = len(texts)
    feature_chunks: list[torch.Tensor] = []
    logprob_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            batch_pos = positions[start:end]
            batch_tok_ids = token_ids[start:end]

            enc = tok(batch_texts, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states  # tuple of [B, T, D], length n_hidden

            b = end - start
            row_idx = torch.arange(b, device=device)
            pos1 = torch.tensor([p[0] for p in batch_pos], device=device)
            pos2 = torch.tensor([p[1] for p in batch_pos], device=device)

            feats_pos1 = torch.stack([h[row_idx, pos1, :] for h in hidden_states], dim=1)
            feats_pos2 = torch.stack([h[row_idx, pos2, :] for h in hidden_states], dim=1)
            feats = torch.stack([feats_pos1, feats_pos2], dim=2)  # [B, n_hidden, 2, D]
            feature_chunks.append(feats.to(torch.float32).cpu())

            logprobs = F.log_softmax(outputs.logits.to(torch.float32), dim=-1)
            tok1_ids = torch.tensor([t[0] for t in batch_tok_ids], device=device)
            tok2_ids = torch.tensor([t[1] for t in batch_tok_ids], device=device)
            lp1 = logprobs[row_idx, pos1, tok1_ids]
            lp2 = logprobs[row_idx, pos2, tok2_ids]
            logprob_chunks.append(torch.stack([lp1, lp2], dim=1).cpu())

    features = torch.cat(feature_chunks, dim=0)
    ans_logprobs = torch.cat(logprob_chunks, dim=0)
    return features, ans_logprobs


# --------------------------------------------------------------------------
# Linear probe (copied from resid_probe.py, extended with test_logprobs).
# --------------------------------------------------------------------------


def fit_linear_probe_predictions(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression; returns ``(train_pred, test_pred,
    test_logprobs)``.

    Copied (in discipline, not imported — this module's docstring explains
    why) from ``fit_linear_probe_predictions`` in
    ``experiments/training-run/analysis/resid_probe.py``, extended per the
    probe1b contract to also return ``test_logprobs``: float32 CPU tensor
    ``[n_test, n_classes]`` = ``log_softmax(test_logits, dim=1)``, so a
    caller can read off the log-probability of the TRUE label (and any
    other class) without refitting. ``test_pred == test_logprobs.argmax(
    dim=1)`` by construction. Features are standardised with TRAIN
    mean/std (std clamped at 1e-6 so a constant feature cannot blow up);
    weights and bias start at exactly zero and the objective
    (cross-entropy + ``l2 * ||W||^2``) is convex, so the fit is
    deterministic — no init seed enters.
    """
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    xt = (x_train - mean) / std
    xs = (x_test - mean) / std
    y_train = y_train.to(xt.device)

    w = torch.zeros(
        xt.shape[1], n_classes, dtype=torch.float32, device=xt.device, requires_grad=True
    )
    b = torch.zeros(n_classes, dtype=torch.float32, device=xt.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=MAX_ITER, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(xt @ w + b, y_train) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        train_logits = xt @ w + b
        test_logits = xs @ w + b
        train_pred = train_logits.argmax(dim=1).cpu()
        test_pred = test_logits.argmax(dim=1).cpu()
        test_logprobs = F.log_softmax(test_logits.to(torch.float32), dim=1).cpu()
    return train_pred, test_pred, test_logprobs


def _head_data_stats(df: pd.DataFrame, test_mask: np.ndarray) -> dict[str, dict]:
    """Data-side per-head stats (plan §7): cheat/majority accuracy and the
    affected-subset boolean mask on the test half — computed once from
    ``df``, independent of layer, model, and placement (a head's cheat
    accuracy does not depend on which features a probe was fit on)."""
    stats: dict[str, dict] = {}
    for head in HEADS:
        k = int(head[1])
        d = df[f"d{k}"].to_numpy()[test_mask]
        cheat = df[f"cheat{k}"].to_numpy()[test_mask]
        affected = df[f"affected{k}"].to_numpy()[test_mask]
        n_affected = int(affected.sum())
        counts = np.bincount(d, minlength=10)
        stats[head] = {
            "cheat_acc_all": float(np.mean(cheat == d)),
            # By construction cheat != true digit on the affected subset,
            # so this is 0.0 whenever that subset is non-empty (d4's is
            # always empty, per make_data's affected4 assert).
            "cheat_acc_affected": 0.0 if n_affected > 0 else float("nan"),
            "majority_acc": float(counts.max()) / float(len(d)),
            "test_affected": affected,
        }
    return stats


def _fit_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    stats: dict,
    rng: np.random.Generator,
    l2: float,
) -> dict:
    """One (layer, position, head) probe cell: the real fit plus its
    shuffled-train-label control refit — exactly the two
    ``fit_linear_probe_predictions`` calls the probe1b contract counts per
    cell (5 cells/layer x 2 calls/cell = 10 calls/layer, i.e. n_hidden x 5
    real + n_hidden x 5 shuffled across ``fit_all_heads``)."""
    _, test_pred, test_logprobs = fit_linear_probe_predictions(
        x_train, y_train, x_test, y_test, 10, l2
    )
    perm = torch.from_numpy(rng.permutation(y_train.shape[0]))
    y_train_shuffled = y_train[perm]
    _, shuffled_test_pred, _ = fit_linear_probe_predictions(
        x_train, y_train_shuffled, x_test, y_test, 10, l2
    )

    top1_acc_all = (test_pred == y_test).double().mean().item()
    shuffled_top1_all = (shuffled_test_pred == y_test).double().mean().item()
    true_logprobs = test_logprobs.gather(1, y_test.unsqueeze(1)).squeeze(1)
    mean_logprob_all = true_logprobs.double().mean().item()

    affected = stats["test_affected"]
    n_affected = int(affected.sum())
    if n_affected > 0:
        idx = torch.from_numpy(np.nonzero(affected)[0])
        top1_acc_affected = (test_pred[idx] == y_test[idx]).double().mean().item()
        shuffled_top1_affected = (shuffled_test_pred[idx] == y_test[idx]).double().mean().item()
        mean_logprob_affected = true_logprobs[idx].double().mean().item()
    else:
        top1_acc_affected = float("nan")
        shuffled_top1_affected = float("nan")
        mean_logprob_affected = float("nan")

    return {
        "top1_acc_all": top1_acc_all,
        "top1_acc_affected": top1_acc_affected,
        "mean_logprob_all": mean_logprob_all,
        "mean_logprob_affected": mean_logprob_affected,
        "n_all": int(y_test.shape[0]),
        "n_affected": n_affected,
        "shuffled_top1_all": shuffled_top1_all,
        "shuffled_top1_affected": shuffled_top1_affected,
        "cheat_acc_all": stats["cheat_acc_all"],
        "cheat_acc_affected": stats["cheat_acc_affected"],
        "majority_acc": stats["majority_acc"],
        "prod_prob_all": float("nan"),
    }


def fit_all_heads(
    features: torch.Tensor,
    df: pd.DataFrame,
    model_name: str,
    fmt: str,
    l2: float = L2,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Every digit-head probe for one (model, format), at every hidden-state
    layer (plan §7).

    Per layer, exactly 5 DISTINCT probes are fit: heads d1, d2, d3, d4 on
    pos1 features, and head d4 on pos2 features. Placement B (plan-ahead)
    reports all four pos1 fits; placement C (teacher-forced) reports the
    SAME d1-d3 pos1 fits (relabeled, never refit) plus the pos2 d4 fit — so
    the d1-d3 rows are identical between B and C by construction, and only
    d4 differs. A shuffled-train-label control is refit alongside every
    real fit (``_fit_head``).

    Also emits one "agg" row per (layer, placement): ``mean_logprob_all``
    is the mean of that placement's 4 head ``mean_logprob_all`` values (the
    log geometric-mean per-digit probability) and ``prod_prob_all`` is
    ``exp`` of their sum (the raw product); every other metric column is
    NaN there except ``n_all`` (copied from the head rows). Non-agg rows
    carry ``prod_prob_all = NaN``.

    Returns one row per (layer, placement, head) with columns
    ``PROBE_ROW_COLUMNS`` exactly.
    """
    n_hidden = features.shape[1]
    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    train_idx = torch.from_numpy(np.nonzero(train_mask)[0])
    test_idx = torch.from_numpy(np.nonzero(test_mask)[0])
    stats = _head_data_stats(df, test_mask)
    rng = np.random.default_rng(seed)

    y_by_head = {
        head: torch.tensor(df[f"d{int(head[1])}"].to_numpy(), dtype=torch.long) for head in HEADS
    }

    rows: list[dict] = []
    for layer in range(n_hidden):
        feat_pos1 = features[:, layer, 0, :]
        feat_pos2 = features[:, layer, 1, :]
        x_train_pos1, x_test_pos1 = feat_pos1[train_idx], feat_pos1[test_idx]
        x_train_pos2, x_test_pos2 = feat_pos2[train_idx], feat_pos2[test_idx]

        pos1_fits = {
            head: _fit_head(
                x_train_pos1,
                y_by_head[head][train_idx],
                x_test_pos1,
                y_by_head[head][test_idx],
                stats[head],
                rng,
                l2,
            )
            for head in HEADS
        }
        pos2_d4_fit = _fit_head(
            x_train_pos2,
            y_by_head["d4"][train_idx],
            x_test_pos2,
            y_by_head["d4"][test_idx],
            stats["d4"],
            rng,
            l2,
        )

        placements = {
            "B": {head: pos1_fits[head] for head in HEADS},
            "C": {
                "d1": pos1_fits["d1"],
                "d2": pos1_fits["d2"],
                "d3": pos1_fits["d3"],
                "d4": pos2_d4_fit,
            },
        }
        for placement, head_fits in placements.items():
            for head in HEADS:
                rows.append(
                    {
                        "model": model_name,
                        "format": fmt,
                        "layer": layer,
                        "placement": placement,
                        "head": head,
                        **head_fits[head],
                    }
                )
            logprobs_all = [head_fits[h]["mean_logprob_all"] for h in HEADS]
            rows.append(
                {
                    "model": model_name,
                    "format": fmt,
                    "layer": layer,
                    "placement": placement,
                    "head": "agg",
                    "top1_acc_all": float("nan"),
                    "top1_acc_affected": float("nan"),
                    "mean_logprob_all": float(np.mean(logprobs_all)),
                    "mean_logprob_affected": float("nan"),
                    "n_all": head_fits["d1"]["n_all"],
                    "n_affected": float("nan"),
                    "shuffled_top1_all": float("nan"),
                    "shuffled_top1_affected": float("nan"),
                    "cheat_acc_all": float("nan"),
                    "cheat_acc_affected": float("nan"),
                    "majority_acc": float("nan"),
                    "prod_prob_all": float(np.exp(np.sum(logprobs_all))),
                }
            )
    return pd.DataFrame(rows, columns=PROBE_ROW_COLUMNS)


# --------------------------------------------------------------------------
# Logit lens (zero-capacity control) and behavioral exact-match eval.
# --------------------------------------------------------------------------


def lens_logprobs(
    model: Any, features: torch.Tensor, token_ids: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logit lens at every ``(layer, position)`` point already extracted
    into ``features`` (plan §7): cast the hidden state to the model's
    dtype, apply ``model.model.norm`` then ``model.lm_head`` (true of both
    the real ``LlamaForCausalLM`` models and the test suite's tiny one),
    ``log_softmax`` in float32. This is the zero-capacity control — no
    fitting happens here, so the LAST layer's lens output is mathematically
    the model's own forward pass; the ``--fit`` CLI driver asserts that
    equality against the saved ``ans_logprobs`` (``LENS_ATOL_NATS``).

    The last hidden-state index is an EXCEPTION to "apply norm then
    lm_head": ``transformers``' ``output_hidden_states`` capture ties its
    final tuple entry to the model's own ``last_hidden_state``
    (``tie_last_hidden_states=True``, the default for every causal LM —
    ``transformers.utils.output_capturing.capture_outputs``'s docstring:
    "This is true for all language models"), which is already POST-final-
    norm — contradicting plan §3's "all pre-final-norm" and verified
    directly against a tiny in-process ``LlamaForCausalLM`` before this was
    trusted. Re-applying ``model.model.norm`` there would double-normalize
    (``RMSNorm`` is not idempotent) and break the last-layer equality this
    function's docstring above promises, so the norm is skipped at that one
    index; every earlier index is a genuine pre-norm block output and gets
    normed as usual.

    ``features``: ``[N, n_hidden, 2, d]``; ``token_ids``: LongTensor
    ``[N, 2]`` (the two true answer-token ids per row). Returns
    ``(top1, logprob)``, both ``[N, n_hidden, 2]`` CPU tensors (top1: Long
    predicted token id; logprob: float32 log-probability of the true
    answer token) — position axis: pos1 scored at token1, pos2 at token2.
    """
    model.eval()
    model_dtype = next(model.parameters()).dtype
    n, n_hidden, n_pos, _ = features.shape
    last_layer = n_hidden - 1
    top1 = torch.zeros(n, n_hidden, n_pos, dtype=torch.long)
    logprob = torch.zeros(n, n_hidden, n_pos, dtype=torch.float32)
    with torch.no_grad():
        for layer in range(n_hidden):
            for pos in range(n_pos):
                h = features[:, layer, pos, :].to(device=device, dtype=model_dtype)
                normed = h if layer == last_layer else model.model.norm(h)
                logits = model.lm_head(normed)
                logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
                top1[:, layer, pos] = logprobs.argmax(dim=-1).cpu()
                true_ids = token_ids[:, pos].to(logprobs.device).unsqueeze(1)
                logprob[:, layer, pos] = logprobs.gather(1, true_ids).squeeze(1).cpu()
    return top1, logprob


def behavior_exact_match(
    model: Any, tok: Any, prompts: list[str], answers: list[str], batch_size: int, device: str
) -> tuple[int, int]:
    """Greedy-decode 3 new tokens from each prompt and count exact matches
    (plan §7). Uses LEFT padding for generation (restored on the tokenizer
    afterward) and overrides the model's own (possibly stale, ts1b — plan
    §3) ``eos_token_id``/``pad_token_id`` with the tokenizer's.

    A row counts as an exact match iff the decoded continuation starts
    with ``answers[i]`` AND the character right after that prefix (if any)
    is not a digit — guards against a longer number that merely has the
    true answer as a prefix (e.g. decoding "1320" against answer "132").
    Returns ``(n_exact, n_total)``.
    """
    model.eval()
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    n_total = len(prompts)
    n_exact = 0
    try:
        with torch.no_grad():
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                batch_prompts = prompts[start:end]
                batch_answers = answers[start:end]
                enc = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
                new_tokens = generated[:, input_ids.shape[1] :]
                decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
                for text, ans in zip(decoded, batch_answers):
                    if text.startswith(ans):
                        rest = text[len(ans) :]
                        if not (rest and rest[0].isdigit()):
                            n_exact += 1
    finally:
        tok.padding_side = original_padding_side
    return n_exact, n_total


# --------------------------------------------------------------------------
# CLI: --make-data / --extract / --fit / --behavior / --all.
# --------------------------------------------------------------------------


def load_model_and_tokenizer(kind: str, args: argparse.Namespace) -> tuple[Any, Any]:
    """Load one of the two models plus the shared Llama tokenizer, bf16, on
    ``args.device``, eval mode (plan §3: ts1b's own ``generation_config``
    is stale and never consulted — the tokenizer supplies every
    special-token id, here and in ``behavior_exact_match``)."""
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if kind == "llama":
        model = AutoModelForCausalLM.from_pretrained(args.llama_id, torch_dtype=torch.bfloat16)
    elif kind == "ts1b":
        model = AutoModelForCausalLM.from_pretrained(
            args.ts1b_id, subfolder=TS1B_SUBFOLDER, torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"load_model_and_tokenizer: unknown kind {kind!r}")
    model = model.to(device=args.device, dtype=torch.bfloat16)
    model.eval()
    return model, tok


def _git_commit() -> str | None:
    """``git rev-parse HEAD`` for ``meta.json``'s provenance, or ``None`` if
    git is unavailable or this tree is not a git checkout (never fatal)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def write_meta(args: argparse.Namespace, models_used: dict[str, dict]) -> None:
    """``meta.json``: model ids, tokenizer id, seed, n, git commit,
    timestamp, and the package versions in play (plan §8)."""
    meta = {
        "tokenizer_id": args.tokenizer_id,
        "models": models_used,
        "seed": args.seed,
        "n": args.n,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with (args.out / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[probe1b] wrote {args.out / 'meta.json'}")


def stage_make_data(args: argparse.Namespace) -> None:
    df = make_data(n=args.n, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "probe1b_pairs.parquet"
    df.to_parquet(path)
    print(f"[probe1b] wrote {path} ({len(df)} rows)")


def stage_extract(args: argparse.Namespace) -> None:
    pairs_path = args.out / "probe1b_pairs.parquet"
    if not pairs_path.is_file():
        raise SystemExit(f"[probe1b] {pairs_path} not found — run --make-data first")
    df = pd.read_parquet(pairs_path)
    args.out.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else list(MODEL_CHOICES)
    formats = [args.format] if args.format else list(FORMAT_CHOICES)

    for model_kind in models:
        model, tok = load_model_and_tokenizer(model_kind, args)
        model_id = args.llama_id if model_kind == "llama" else args.ts1b_id
        for fmt in formats:
            texts: list[str] = []
            positions: list[tuple[int, int]] = []
            token_ids: list[tuple[int, int]] = []
            for row in df.itertuples():
                a, b, ans = int(row.a), int(row.b), int(row.ans)
                full, _, prev_char = render(a, b, ans, fmt)
                pos1, pos2 = answer_positions(tok, full, str(ans), prev_char)
                input_ids = tok(full, add_special_tokens=True)["input_ids"]
                texts.append(full)
                positions.append((pos1, pos2))
                token_ids.append((input_ids[pos2], input_ids[pos2 + 1]))

            print(
                f"[probe1b] extracting {model_kind}/{fmt}: {len(texts)} rows, batch {args.batch_size}"
            )
            features, ans_logprobs = extract_features(
                model, tok, texts, positions, token_ids, args.batch_size, args.device
            )
            out_path = args.out / f"features_{model_kind}_{fmt}.pt"
            torch.save(
                {
                    "features": features,
                    "ans_logprobs": ans_logprobs,
                    "token_ids": torch.tensor(token_ids, dtype=torch.long),
                    "positions": torch.tensor(positions, dtype=torch.long),
                    "model": model_kind,
                    "format": fmt,
                    "model_id": model_id,
                    "seed": args.seed,
                },
                out_path,
            )
            print(f"[probe1b] wrote {out_path}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def stage_fit(args: argparse.Namespace) -> None:
    pairs_path = args.out / "probe1b_pairs.parquet"
    if not pairs_path.is_file():
        raise SystemExit(f"[probe1b] {pairs_path} not found — run --make-data first")
    df = pd.read_parquet(pairs_path)

    feature_files = sorted(args.out.glob("features_*_*.pt"))
    if not feature_files:
        raise SystemExit(f"[probe1b] no features_*.pt found under {args.out} — run --extract first")

    probe_dfs: list[pd.DataFrame] = []
    lens_rows: list[dict] = []
    models_used: dict[str, dict] = {}

    for path in feature_files:
        payload = torch.load(path, map_location="cpu")
        model_kind = payload["model"]
        fmt = payload["format"]
        model_id = payload["model_id"]
        features = payload["features"]
        ans_logprobs = payload["ans_logprobs"]
        token_ids = payload["token_ids"]

        if len(df) != features.shape[0]:
            raise SystemExit(
                f"[probe1b] {path}: features has {features.shape[0]} rows but "
                f"{pairs_path} has {len(df)} rows — regenerate one to match the other"
            )

        print(f"[probe1b] fitting probes: {model_kind}/{fmt}")
        probe_dfs.append(fit_all_heads(features, df, model_kind, fmt, l2=args.l2, seed=args.seed))

        print(f"[probe1b] logit lens: {model_kind}/{fmt}")
        model, _ = load_model_and_tokenizer(model_kind, args)
        top1, logprob = lens_logprobs(model, features, token_ids, args.device)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        n_hidden = features.shape[1]
        last = n_hidden - 1
        last_diff = (logprob[:, last, :] - ans_logprobs).abs().max().item()
        if last_diff > LENS_ATOL_NATS:
            raise AssertionError(
                f"[probe1b] {path}: last-layer logit-lens log-probs diverge from the "
                f"saved ans_logprobs by {last_diff:.6f} nats (> {LENS_ATOL_NATS}) — the "
                "logit lens at the final hidden layer must equal the model's own "
                "forward-pass log-probs; do not relax this tolerance"
            )

        for layer in range(n_hidden):
            for pos_idx, pos_name in ((0, "pos1"), (1, "pos2")):
                true_ids = token_ids[:, pos_idx]
                pred_ids = top1[:, layer, pos_idx]
                lens_rows.append(
                    {
                        "model": model_kind,
                        "format": fmt,
                        "layer": layer,
                        "position": pos_name,
                        "top1_acc": (pred_ids == true_ids).double().mean().item(),
                        "mean_logprob_nats": logprob[:, layer, pos_idx].double().mean().item(),
                        "n": int(features.shape[0]),
                    }
                )
        models_used[model_kind] = {"model_id": model_id}

    all_probe_df = pd.concat(probe_dfs, ignore_index=True)
    all_probe_df.to_csv(args.out / "probe_rows.csv", index=False)
    print(f"[probe1b] wrote {args.out / 'probe_rows.csv'} ({len(all_probe_df)} rows)")

    lens_df = pd.DataFrame(lens_rows, columns=LENS_ROW_COLUMNS)
    lens_df.to_csv(args.out / "lens_rows.csv", index=False)
    print(f"[probe1b] wrote {args.out / 'lens_rows.csv'} ({len(lens_df)} rows)")

    write_meta(args, models_used)


def stage_behavior(args: argparse.Namespace) -> None:
    pairs_path = args.out / "probe1b_pairs.parquet"
    if not pairs_path.is_file():
        raise SystemExit(f"[probe1b] {pairs_path} not found — run --make-data first")
    df = pd.read_parquet(pairs_path)
    args.out.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else list(MODEL_CHOICES)
    formats = [args.format] if args.format else list(FORMAT_CHOICES)

    rows: list[dict] = []
    for model_kind in models:
        model, tok = load_model_and_tokenizer(model_kind, args)
        for fmt in formats:
            prompts: list[str] = []
            answers: list[str] = []
            for row in df.itertuples():
                a, b, ans = int(row.a), int(row.b), int(row.ans)
                _, prompt, _ = render(a, b, ans, fmt)
                prompts.append(prompt)
                answers.append(str(ans))
            print(f"[probe1b] behavior eval: {model_kind}/{fmt}: {len(prompts)} rows")
            n_exact, n_total = behavior_exact_match(
                model, tok, prompts, answers, args.batch_size, args.device
            )
            em = n_exact / n_total
            print(f"[probe1b]   em={em:.4f} ({n_exact}/{n_total})")
            rows.append(
                {"model": model_kind, "format": fmt, "em": em, "n_exact": n_exact, "n": n_total}
            )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows, columns=BEHAVIOR_ROW_COLUMNS)
    out_path = args.out / "behavior.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[probe1b] wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--make-data", action="store_true", help="generate probe1b_pairs.parquet")
    ap.add_argument("--extract", action="store_true", help="extract features_{model}_{format}.pt")
    ap.add_argument(
        "--fit", action="store_true", help="fit probes + logit lens, write probe_rows.csv etc."
    )
    ap.add_argument(
        "--behavior", action="store_true", help="greedy exact-match eval, write behavior.csv"
    )
    ap.add_argument(
        "--all", action="store_true", help="run make-data, extract, fit, behavior in order"
    )
    ap.add_argument(
        "--model", choices=MODEL_CHOICES, default=None, help="narrow --extract/--behavior"
    )
    ap.add_argument(
        "--format", choices=FORMAT_CHOICES, default=None, help="narrow --extract/--behavior"
    )
    ap.add_argument("--llama-id", default=DEFAULT_LLAMA_ID)
    ap.add_argument("--ts1b-id", default=DEFAULT_TS1B_ID)
    ap.add_argument(
        "--tokenizer-id", default=DEFAULT_TOKENIZER_ID, help="tokenizer for BOTH models"
    )
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--l2", type=float, default=L2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--confirm-cost", action="store_true")
    return ap


def _resolve_stages(args: argparse.Namespace) -> list[str]:
    if args.all:
        return ["make-data", "extract", "fit", "behavior"]
    stages = []
    if args.make_data:
        stages.append("make-data")
    if args.extract:
        stages.append("extract")
    if args.fit:
        stages.append("fit")
    if args.behavior:
        stages.append("behavior")
    return stages


_STAGE_FUNCS = {
    "make-data": stage_make_data,
    "extract": stage_extract,
    "fit": stage_fit,
    "behavior": stage_behavior,
}


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    stages = _resolve_stages(args)
    if not stages:
        ap.error("nothing to do — pass one of --make-data/--extract/--fit/--behavior/--all")

    if COST_GATED_STAGES & set(stages):
        print(COST_ESTIMATE)
        if not args.confirm_cost:
            print("[probe1b] --confirm-cost not given; refusing (budget rule). Exiting.")
            sys.exit(1)

    for stage in stages:
        _STAGE_FUNCS[stage](args)


if __name__ == "__main__":
    main()
