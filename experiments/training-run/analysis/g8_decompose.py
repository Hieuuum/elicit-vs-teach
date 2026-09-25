"""G8 retention decomposition: WHERE a ts38 parent's TinyStories loss went up.

G8 (``scripts/gates.py g8``) is one number — mean next-token CE in nats/token
on run 1's frozen validation stream — and every arithmetic-fine-tuned ts38
parent crosses its 1.1718 bar before arithmetic accuracy reaches 0.95. That
number cannot say *what* got worse. This driver scores the SAME stream with
the base and a target checkpoint side by side and splits the per-position
loss delta three ways, so "broad degradation of English modelling" and
"a narrow shift onto digits / story starts / post-digit positions" become
distinguishable readouts rather than one scalar.

The three groupings of each predicted position ``j`` of a row (targets
``1..L-1``, exactly the positions ``geode.train.evaluate_nll_nats`` averages):

- ``story_pos`` — distance from the last EOS strictly before ``j``
  (``j - idx_last_eos - 1``; 0 = the first token of a new story). Rows are
  packed stories separated by EOS id 0, so a row's leading tokens precede any
  EOS: they are bucketed ``row_head`` (offset unknown, the row was sliced
  mid-story) rather than folded into a numeric bucket they'd contaminate.
- ``since_digit`` — same construction against the 10 single-character digit
  tokens; ``none`` where no digit precedes ``j`` in the row (the common case:
  TinyStories is nearly digit-free).
- ``tok_class`` — the class of the TARGET token: digit / eos / newline /
  punct / other.

Per group: count, base mean, target mean, delta mean, mean ``KL(base‖target)``
over the full vocab, and ``share_of_total_delta`` (the group's summed delta
over the global summed delta — sums to 1.0 across a grouping, and is the
column that answers "narrow or broad"). Sums and counts are accumulated
globally and divided ONCE, never averaged per batch, so every number is
batch-size invariant the same way G8's own metric is (V5.19).

Two more probes on top of the split: the vocab entries whose mean predicted
probability moved most at story-start and immediately-after-digit positions,
and greedy 60-token continuations of six fixed prompts from both models —
the qualitative check that a small nats delta is or is not still English.

**Anchor.** ``base_mean_nats`` must reproduce the base run's recorded
``min_val_nats`` to ``--anchor-tol``, for exactly gates.py G8's reason: a
decomposition of a stream that is not run 1's validation stream describes
nothing. Failure raises and writes no JSON (never a substituted stream); the
cached pack is what costs 40 minutes, and it survives. ``--n-rows`` (a smoke
subset) demotes the anchor to a printed number, since a prefix of the stream
is not expected to reproduce the full-stream mean.

``--plot`` is a laptop-side mode that loads no model and no checkpoint: it
overlays the deltas of several result JSONs (one series per ``--label``).

Usage:
    python3 ../analysis/g8_decompose.py \
      --run <run_id> --checkpoint <save_pretrained dir> \
      --base-run evt-run1-base-v3-ext \
      --config ../configs/archive/runs/run1_pretrain.yaml --tokenizer ../tokenizer \
      --val-cache $GEODE_STORE/cache/run1_val_stream.pt \
      --out <path.json> [--device cuda|cpu] [--batch-size 32] [--n-rows N] [--label <text>]

    python3 analysis/g8_decompose.py --plot --fig <out.png> <a.json> [<b.json> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

SCHEMA = "g8_decompose/v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "experiments" / "training-run" / "scripts"

# Bucket edges as "value >= threshold" counts: the code is the number of
# thresholds the offset clears, so 0 -> 0, 1 -> 1, 2..3 -> 2, ... and the
# open-ended top bucket needs no special case. The sentinel offset -1 (no
# preceding EOS / no preceding digit) takes the last code.
STORY_POS_THRESHOLDS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
STORY_POS_LABELS: tuple[str, ...] = (
    "0",
    "1",
    "2-3",
    "4-7",
    "8-15",
    "16-31",
    "32-63",
    "64-127",
    "128+",
    "row_head",
)
SINCE_DIGIT_THRESHOLDS: tuple[int, ...] = (1, 2, 4, 8, 16)
SINCE_DIGIT_LABELS: tuple[str, ...] = ("0", "1", "2-3", "4-7", "8-15", "16+", "none")
TOK_CLASS_LABELS: tuple[str, ...] = ("digit", "eos", "newline", "punct", "other")
TOK_CLASS_CODE = {name: i for i, name in enumerate(TOK_CLASS_LABELS)}

GROUPINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("story_pos", STORY_POS_LABELS),
    ("since_digit", SINCE_DIGIT_LABELS),
    ("tok_class", TOK_CLASS_LABELS),
)

# Fixed prompts, never sampled: the same six every checkpoint is read on, so
# continuations are comparable across JSONs. The last one is deliberately
# arithmetic-flavoured NL — the place a digit-obsessed parent shows itself.
PROMPTS: tuple[str, ...] = (
    "Once upon a time, ",
    "Lily and her mom went to the park. ",
    "One day, a little boy named Tim found ",
    "The dog was very happy because ",
    "Sara had 3 apples and her friend gave her 2 more. ",
    "There was a big red ball. ",
)
N_NEW_TOKENS = 60
TOP_K = 15
# Probability fields get more decimals than loss fields: the "down" list lives
# near 1e-5, where 6 dp is a column of zeros.
LOSS_DP = 6
PROB_DP = 8


# ---------------------------------------------------------------------------
# Position indexing and bucketing (pure; no I/O, no model, no tokenizer)
# ---------------------------------------------------------------------------
def _last_hit_offset(hit: torch.Tensor) -> torch.Tensor:
    """Offsets from the last ``True`` strictly before each predicted position.

    ``hit`` is ``[B, L]`` boolean over token positions. Returns ``[B, L-1]``
    aligned to targets ``1..L-1``: ``j - idx_last_hit - 1``, or -1 where no
    hit occurs at any index ``< j``. A running ``cummax`` of "index if hit else
    -1" gives the last hit index at or before each position, and the target at
    ``j`` reads it at ``j-1`` — which is what makes the token right after a hit
    come out as 0 while the hit token itself is offset from the PREVIOUS hit.
    """
    if hit.ndim != 2 or hit.shape[1] < 2:
        raise ValueError(f"expected a [B, L>=2] mask, got {tuple(hit.shape)}")
    idx = torch.arange(hit.shape[1], device=hit.device).expand_as(hit)
    marked = torch.where(hit, idx, torch.full_like(idx, -1))
    last = marked.cummax(dim=1).values[:, :-1]
    target_idx = torch.arange(1, hit.shape[1], device=hit.device)
    return torch.where(last < 0, torch.full_like(last, -1), target_idx - last - 1)


def story_pos_index(seqs: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Tokens since the last EOS, per predicted position. ``[B, L] -> [B, L-1]``.

    0 = first token of a new story; -1 = ``row_head`` (the row was sliced
    mid-story, so the true story offset is unknowable from this row).
    """
    return _last_hit_offset(seqs == eos_id)


def since_digit_index(seqs: torch.Tensor, digit_ids: torch.Tensor) -> torch.Tensor:
    """Tokens since the last digit token, per predicted position. -1 = none."""
    return _last_hit_offset(torch.isin(seqs, digit_ids.to(seqs.device)))


def _bucketize(offsets: torch.Tensor, thresholds: Sequence[int]) -> torch.Tensor:
    """Offsets (-1 sentinel allowed) -> bucket codes; the sentinel takes the
    last code, one past the numeric buckets."""
    codes = torch.zeros_like(offsets)
    for t in thresholds:
        codes += (offsets >= t).long()
    return torch.where(offsets < 0, torch.full_like(codes, len(thresholds) + 1), codes)


def bucketize_story_pos(offsets: torch.Tensor) -> torch.Tensor:
    """-> codes into ``STORY_POS_LABELS``."""
    return _bucketize(offsets, STORY_POS_THRESHOLDS)


def bucketize_since_digit(offsets: torch.Tensor) -> torch.Tensor:
    """-> codes into ``SINCE_DIGIT_LABELS``."""
    return _bucketize(offsets, SINCE_DIGIT_THRESHOLDS)


def token_strings(tokenizer: Any) -> list[str]:
    """Every vocab id decoded to its own string (specials NOT stripped)."""
    return [tokenizer.decode([i]) for i in range(len(tokenizer))]


def token_class_table(token_strs: Sequence[str], eos_id: int) -> torch.Tensor:
    """id -> ``TOK_CLASS_LABELS`` code, built once from decoded token strings.

    Takes the decoded strings rather than the tokenizer so the classification
    rule is testable without one, and so the ~10K single-id decodes happen in
    exactly one place (``token_strings``, whose output the top-token report
    needs anyway). Precedence: eos, then digit, then newline, then punct.
    """
    codes = []
    for i, s in enumerate(token_strs):
        if i == eos_id:
            code = TOK_CLASS_CODE["eos"]
        elif any(c.isdigit() for c in s):
            code = TOK_CLASS_CODE["digit"]
        elif "\n" in s:
            code = TOK_CLASS_CODE["newline"]
        elif s and all((not c.isalnum()) and (not c.isspace()) for c in s):
            code = TOK_CLASS_CODE["punct"]
        else:
            code = TOK_CLASS_CODE["other"]
        codes.append(code)
    return torch.tensor(codes, dtype=torch.long)


@dataclass(frozen=True)
class Tables:
    """The tokenizer-derived lookups the accumulator needs, device-resident."""

    eos_id: int
    digit_ids: torch.Tensor
    class_table: torch.Tensor

    def to(self, device: str | torch.device) -> Tables:
        return Tables(self.eos_id, self.digit_ids.to(device), self.class_table.to(device))


def tables_from_tokenizer(tokenizer: Any) -> tuple[Tables, list[str]]:
    """``Tables`` + the decoded vocab strings the top-token report reports with."""
    eos_id = int(tokenizer.eos_token_id)
    strs = token_strings(tokenizer)
    class_table = token_class_table(strs, eos_id)
    digit_ids = (class_table == TOK_CLASS_CODE["digit"]).nonzero(as_tuple=True)[0]
    return Tables(eos_id, digit_ids, class_table), strs


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------
def new_accumulator(vocab_size: int) -> dict[str, Any]:
    """Zeroed sums/counts. Everything lands on CPU in float64: the per-group
    tensors are tiny, and the pass must divide once at the end from a sum that
    did not drift in fp32 over ~2.7M positions."""

    def group(k: int) -> dict[str, torch.Tensor]:
        return {
            "count": torch.zeros(k, dtype=torch.long),
            "base": torch.zeros(k, dtype=torch.float64),
            "target": torch.zeros(k, dtype=torch.float64),
            "kl": torch.zeros(k, dtype=torch.float64),
        }

    def top() -> dict[str, Any]:
        return {
            "count": 0,
            "p_base": torch.zeros(vocab_size, dtype=torch.float64),
            "p_target": torch.zeros(vocab_size, dtype=torch.float64),
        }

    return {
        "vocab_size": vocab_size,
        "n_positions": 0,
        "base_sum": 0.0,
        "target_sum": 0.0,
        "kl_sum": 0.0,
        "groups": {name: group(len(labels)) for name, labels in GROUPINGS},
        "top": {"story_start": top(), "after_digit": top()},
    }


def _add_top(
    slot: dict[str, Any], mask: torch.Tensor, p_base: torch.Tensor, log_p_target: torch.Tensor
) -> None:
    """Sum the two models' predicted distributions over the masked positions."""
    n = int(mask.sum())
    if n == 0:
        return
    slot["count"] += n
    slot["p_base"] += p_base[mask].sum(0).double().cpu()
    slot["p_target"] += log_p_target[mask].exp().sum(0).double().cpu()


def accumulate(
    logits_base: torch.Tensor,
    logits_target: torch.Tensor,
    batch: torch.Tensor,
    tables: Tables,
    acc: dict[str, Any],
) -> None:
    """Fold one batch into ``acc``. ``logits_*`` are ``[B, L, V]``, ``batch``
    is ``[B, L]``; positions ``0..L-2`` predict targets ``1..L-1``.

    Per-position CE is read off ``log_softmax`` by ``gather`` — the same
    quantity ``F.cross_entropy(..., reduction="none")`` computes, but reusing
    the log-probabilities the KL term needs anyway instead of paying for a
    second softmax over a 10K vocab.
    """
    targets = batch[:, 1:]
    log_p_base = F.log_softmax(logits_base[:, :-1, :], dim=-1)
    log_p_target = F.log_softmax(logits_target[:, :-1, :], dim=-1)
    ce_base = -log_p_base.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ce_target = -log_p_target.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p_base = log_p_base.exp()
    kl = (p_base * (log_p_base - log_p_target)).sum(-1)

    story = story_pos_index(batch, tables.eos_id)
    digit = since_digit_index(batch, tables.digit_ids)
    codes = {
        "story_pos": bucketize_story_pos(story),
        "since_digit": bucketize_since_digit(digit),
        "tok_class": tables.class_table[targets],
    }

    flat_base = ce_base.reshape(-1).double().cpu()
    flat_target = ce_target.reshape(-1).double().cpu()
    flat_kl = kl.reshape(-1).double().cpu()
    acc["n_positions"] += flat_base.numel()
    acc["base_sum"] += float(flat_base.sum())
    acc["target_sum"] += float(flat_target.sum())
    acc["kl_sum"] += float(flat_kl.sum())
    for name, code in codes.items():
        flat_code = code.reshape(-1).cpu()
        g = acc["groups"][name]
        g["count"].index_add_(0, flat_code, torch.ones_like(flat_code))
        g["base"].index_add_(0, flat_code, flat_base)
        g["target"].index_add_(0, flat_code, flat_target)
        g["kl"].index_add_(0, flat_code, flat_kl)

    vocab = p_base.shape[-1]
    _add_top(
        acc["top"]["story_start"],
        (story == 0).reshape(-1),
        p_base.reshape(-1, vocab),
        log_p_target.reshape(-1, vocab),
    )
    _add_top(
        acc["top"]["after_digit"],
        (digit == 0).reshape(-1),
        p_base.reshape(-1, vocab),
        log_p_target.reshape(-1, vocab),
    )


def run_pass(
    base_model: torch.nn.Module,
    target_model: torch.nn.Module,
    seqs: torch.Tensor,
    tables: Tables,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Score both models over ``seqs`` and return the filled accumulator."""
    tables = tables.to(device)
    base_model.eval()
    target_model.eval()
    acc: dict[str, Any] | None = None
    with torch.no_grad():
        for start in range(0, seqs.shape[0], batch_size):
            batch = seqs[start : start + batch_size].to(device)
            logits_base = base_model(batch).logits
            logits_target = target_model(batch).logits
            if acc is None:
                acc = new_accumulator(logits_base.shape[-1])
            accumulate(logits_base, logits_target, batch, tables, acc)
            del logits_base, logits_target
    if acc is None:
        raise ValueError("run_pass got an empty sequence tensor — nothing to score")
    return acc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(acc: dict[str, Any]) -> dict[str, Any]:
    """Accumulator -> the JSON's global block + per-grouping bucket rows.

    Empty buckets emit ``null`` means and a 0.0 share rather than ``0/0``:
    ``json.dump`` would write a bare ``NaN`` that only Python reads back.
    """
    n = acc["n_positions"]
    total_delta = acc["target_sum"] - acc["base_sum"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, labels in GROUPINGS:
        g = acc["groups"][name]
        rows = []
        for k, label in enumerate(labels):
            count = int(g["count"][k])
            if count == 0:
                rows.append(
                    {
                        "bucket": label,
                        "count": 0,
                        "base_mean": None,
                        "target_mean": None,
                        "delta_mean": None,
                        "kl_mean": None,
                        "share_of_total_delta": 0.0,
                    }
                )
                continue
            base_mean = float(g["base"][k]) / count
            target_mean = float(g["target"][k]) / count
            delta_sum = float(g["target"][k]) - float(g["base"][k])
            rows.append(
                {
                    "bucket": label,
                    "count": count,
                    "base_mean": round(base_mean, LOSS_DP),
                    "target_mean": round(target_mean, LOSS_DP),
                    "delta_mean": round(target_mean - base_mean, LOSS_DP),
                    "kl_mean": round(float(g["kl"][k]) / count, LOSS_DP),
                    "share_of_total_delta": (
                        round(delta_sum / total_delta, LOSS_DP) if total_delta != 0.0 else 0.0
                    ),
                }
            )
        groups[name] = rows
    return {
        "n_positions": n,
        "base_mean_nats": round(acc["base_sum"] / n, LOSS_DP),
        "target_mean_nats": round(acc["target_sum"] / n, LOSS_DP),
        "delta_mean_nats": round(total_delta / n, LOSS_DP),
        "kl_mean_nats": round(acc["kl_sum"] / n, LOSS_DP),
        "groups": groups,
    }


def top_tokens(slot: dict[str, Any], token_strs: Sequence[str], k: int = TOP_K) -> dict[str, Any]:
    """The ``k`` vocab entries whose mean predicted probability rose most, and
    the ``k`` that fell most, over the positions folded into ``slot``."""
    count = slot["count"]
    if count == 0:
        return {"count": 0, "up": [], "down": []}
    mean_base = slot["p_base"] / count
    mean_target = slot["p_target"] / count
    move = mean_target - mean_base
    order = torch.argsort(move, descending=True)

    def entry(i: int) -> list[Any]:
        name = token_strs[i] if i < len(token_strs) else f"<id {i}>"
        return [name, i, round(float(mean_base[i]), PROB_DP), round(float(mean_target[i]), PROB_DP)]

    k = min(k, order.numel())
    return {
        "count": count,
        "up": [entry(int(i)) for i in order[:k]],
        "down": [entry(int(i)) for i in reversed(order[-k:].tolist())],
    }


def continuations(
    base_model: torch.nn.Module,
    target_model: torch.nn.Module,
    tokenizer: Any,
    device: str,
) -> list[dict[str, str]]:
    """Greedy 60-token continuations of ``PROMPTS`` from both models.

    One prompt at a time: batching unequal-length prompts needs left padding,
    and getting that wrong yields fluent-looking garbage rather than an error.
    Only the continuation is decoded, with specials VISIBLE — an immediate
    ``<|endoftext|>`` is itself the diagnosis.
    """
    out = []
    with torch.no_grad():
        for prompt in PROMPTS:
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            row: dict[str, str] = {"prompt": prompt}
            for key, model in (("base", base_model), ("target", target_model)):
                gen = model.generate(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=N_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                row[key] = tokenizer.decode(gen[0, ids.shape[1] :], skip_special_tokens=False)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Scoring run (I/O)
# ---------------------------------------------------------------------------
def _load_gates() -> Any:
    """``scripts/gates.py`` as a module — the authority on run 1's validation
    stream. Imported lazily: it pulls transformers + the whole trainer tree,
    which ``--plot`` and the unit tests must not pay for."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import gates

    return gates


def score(args: argparse.Namespace) -> int:
    """Load both checkpoints, decompose the delta, write the JSON."""
    from transformers import AutoTokenizer

    from geode.zoo import checkpoint_dir, load_model, load_run

    gates = _load_gates()
    base_run = args.base_run or gates.G8_BASE_RUN
    anchor_tol = args.anchor_tol if args.anchor_tol is not None else gates.G8_ANCHOR_TOL_NATS
    cfg = gates.load_config(args.config, None)
    store = args.store
    base_manifest = load_run(base_run, store=store)
    # Both runs and both checkpoints are resolved BEFORE the stream is built:
    # on a cache miss that is a ~45-minute pack, and discovering a missing
    # checkpoint afterwards would throw it away (gates.py G8's ordering).
    load_run(args.run, store=store)
    checkpoint = args.checkpoint or checkpoint_dir(args.run, store=store)
    checkpoint_dir(base_run, store=store)

    # Same resolver as gates.py G8: a DIRECTORY is required. run 1's pretrain
    # config was archived two levels deeper than it launched, so its relative
    # tokenizer.path no longer lands on the frozen artifact, and falling back
    # to an HF id would ask the hub for a repo named "../tokenizer".
    tok_path = args.tokenizer or (args.config.parent / cfg["tokenizer"]["path"])
    tok_dir = Path(tok_path).resolve()
    if not tok_dir.is_dir():
        raise SystemExit(
            f"[g8dec] no tokenizer directory at {tok_dir}. Pass "
            "--tokenizer <experiments/training-run/tokenizer>."
        )
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_seqs, fingerprints = gates._run1_val_stream(
        cfg, tokenizer, base_manifest.data, args.val_cache
    )
    if args.n_rows is not None:
        val_seqs = val_seqs[: args.n_rows]
        print(f"[g8dec] --n-rows {args.n_rows}: scoring the first {len(val_seqs)} rows only")

    tables, token_strs = tables_from_tokenizer(tokenizer)
    print(
        f"[g8dec] tokenizer {tok_dir.name}: eos={tables.eos_id}, "
        f"{tables.digit_ids.numel()} digit tokens, vocab {len(token_strs)}"
    )

    print(f"[g8dec] loading base {base_run} ...", flush=True)
    base_model = load_model(base_run, store=store, device=args.device)
    print(f"[g8dec] loading target {args.run} @ {checkpoint} ...", flush=True)
    target_model = load_model(args.run, store=store, device=args.device, checkpoint=checkpoint)

    acc = run_pass(
        base_model,
        target_model,
        val_seqs,
        tables,
        batch_size=args.batch_size,
        device=args.device,
    )
    report = summarize(acc)

    anchor_want = base_manifest.data["experiment"]["pretrain_result"]["min_val_nats"]
    gap = abs(report["base_mean_nats"] - anchor_want)
    print(
        f"[g8dec] anchor: base scores {report['base_mean_nats']:.6f} nats, "
        f"{base_run} manifest recorded {anchor_want:.6f} (|Δ| {gap:.2e}, tol {anchor_tol}); "
        f"target {report['target_mean_nats']:.6f}"
    )
    if args.n_rows is not None:
        print("[g8dec] --n-rows given: anchor is informational only (a prefix is not the stream)")
    elif gap > anchor_tol:
        raise SystemExit(
            f"[g8dec] the rebuilt stream does NOT reproduce {base_run}'s recorded validation "
            f"loss: measured {report['base_mean_nats']:.6f}, manifest {anchor_want:.6f} "
            f"(|Δ| {gap:.2e} > {anchor_tol}). A decomposition of some other stream describes "
            "nothing — refusing to write. Fix the environment (see gates.py G8's note on "
            "seeded-randperm drift); the cached pack is unaffected and will be reused."
        )

    result = {
        "schema": SCHEMA,
        "label": args.label,
        "run": args.run,
        "checkpoint": str(checkpoint),
        "base_run": base_run,
        "n_rows": int(len(val_seqs)),
        **report,
        "top_tokens_story_start": top_tokens(acc["top"]["story_start"], token_strs),
        "top_tokens_after_digit": top_tokens(acc["top"]["after_digit"], token_strs),
        "continuations": continuations(base_model, target_model, tokenizer, args.device),
        "torch_version": str(torch.__version__),
        "stream_fingerprints": fingerprints,
        "created_unix": int(time.time()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[g8dec] wrote {args.out}")
    print(
        f"[g8dec] label={args.label} base={report['base_mean_nats']:.4f} "
        f"target={report['target_mean_nats']:.4f} delta={report['delta_mean_nats']:.4f} "
        f"n_positions={report['n_positions']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Plot mode (no model, no checkpoint, no store)
# ---------------------------------------------------------------------------
def _series(report: dict[str, Any], grouping: str, field: str) -> list[float]:
    rows = {r["bucket"]: r for r in report["groups"][grouping]}
    labels = dict(GROUPINGS)[grouping]
    return [
        float("nan") if rows.get(b, {}).get(field) is None else float(rows[b][field])
        for b in labels
    ]


def plot(reports: list[dict[str, Any]], out: Path) -> None:
    """Three panels of ``delta_mean`` — by story position, by distance since a
    digit, by target token class — one series per report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = {
        "story_pos": "tokens since last EOS (story position)",
        "since_digit": "tokens since last digit",
        "tok_class": "target token class",
    }
    for ax, (grouping, labels) in zip(axes, GROUPINGS):
        x = range(len(labels))
        for report in reports:
            delta = report.get("delta_mean_nats", float("nan"))
            ax.plot(
                x,
                _series(report, grouping, "delta_mean"),
                marker="o",
                ms=4,
                lw=1.2,
                label=f"{report.get('label') or report.get('run')} (Δ={delta:+.4f})",
            )
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(titles[grouping], fontsize=10)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Δ loss vs base (nats/token)")
    axes[0].legend(fontsize=8)
    fig.suptitle("G8 retention decomposition: where the TinyStories loss went up")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[g8dec] wrote {out}")


def print_shares(reports: list[dict[str, Any]]) -> None:
    """Text table of ``share_of_total_delta`` per bucket per report."""
    names = [str(r.get("label") or r.get("run")) for r in reports]
    width = max([len(n) for n in names] + [10])
    for grouping, labels in GROUPINGS:
        print(f"\nshare_of_total_delta — {grouping}")
        print("  " + "bucket".ljust(10) + "".join(n.rjust(width + 2) for n in names))
        for i, bucket in enumerate(labels):
            cells = "".join(
                f"{_series(r, grouping, 'share_of_total_delta')[i]:>{width + 2}.3f}"
                for r in reports
            )
            print("  " + bucket.ljust(10) + cells)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Split out of ``main`` so the launcher's invocation is checkable at parse
    level without loading a model (the ``test_launcher_gate_args.py`` pattern)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", help="run_id of the target checkpoint being decomposed")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="save_pretrained dir (default: the run's geode.zoo checkpoint_dir)",
    )
    ap.add_argument(
        "--base-run",
        default=None,
        help="base run supplying the stream pins and the anchor (default: gates.G8_BASE_RUN)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="run 1's PRETRAIN yaml — the corpus/packing/split pins",
    )
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="tokenizer DIRECTORY (default: --config's tokenizer.path)",
    )
    ap.add_argument(
        "--val-cache",
        type=Path,
        default=None,
        help="'.pt' cache of the rebuilt val rows; written on a miss, reused after",
    )
    ap.add_argument("--out", type=Path, default=None, help="result JSON path")
    ap.add_argument(
        "--store", type=Path, default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="score only the FIRST N stream rows (smoke runs; demotes the anchor)",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="free text naming this checkpoint in the summary line and the figure "
        "legend (default: the checkpoint dir name, else --run)",
    )
    ap.add_argument(
        "--anchor-tol",
        type=float,
        default=None,
        help="max |measured - recorded| base loss (default: gates.G8_ANCHOR_TOL_NATS)",
    )
    ap.add_argument(
        "--plot", action="store_true", help="laptop-side mode: overlay result JSONs, load no model"
    )
    ap.add_argument("--fig", type=Path, default=None, help="--plot output PNG")
    ap.add_argument("jsons", nargs="*", type=Path, help="--plot inputs")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plot:
        if not args.jsons or args.fig is None:
            raise SystemExit("[g8dec] --plot needs --fig <out.png> and at least one result JSON")
        reports = [json.loads(p.read_text()) for p in args.jsons]
        plot(reports, args.fig)
        print_shares(reports)
        return 0
    missing = [n for n in ("run", "config", "out") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"[g8dec] a scoring run needs {['--' + m for m in missing]}")
    if args.label is None:
        args.label = args.checkpoint.name if args.checkpoint else args.run
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
