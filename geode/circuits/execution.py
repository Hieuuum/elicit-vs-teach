"""Shared deterministic batching, complete-answer metrics, pairing and cost guards."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class RunBudget:
    max_seconds: float
    hourly_rate: float
    confirmed: bool
    started: float = 0.0

    def __post_init__(self) -> None:
        if not self.confirmed:
            raise ValueError("compute requires --confirm-cost")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("positive finite wall-time limit required")
        if not math.isfinite(self.hourly_rate) or self.hourly_rate < 0:
            raise ValueError("nonnegative finite hourly price required")
        self.started = time.monotonic()

    @property
    def estimated_usd(self) -> float:
        return self.max_seconds * self.hourly_rate / 3600

    def check(self) -> None:
        if time.monotonic() - self.started >= self.max_seconds:
            raise TimeoutError("evaluation wall-time limit reached; partial artifacts retained")


def encode_answer(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], list[bool]]:
    """Explicit token-boundary protocol: fixed prompt tokens followed by answer tokens.

    Concatenating independently tokenized answer tokens avoids changing the
    prompt's final token under BPE. Both generation and teacher forcing use the
    identical prompt IDs. This protocol is saved verbatim in run metadata.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not prompt_ids or not answer_ids:
        raise ValueError("prompt and answer must each contain tokens")
    ids = prompt_ids + answer_ids
    return ids, [False] * len(prompt_ids) + [True] * len(answer_ids)


def complete_answer_log_probs(
    logits: torch.Tensor, ids: torch.Tensor, answer_mask: torch.Tensor
) -> torch.Tensor:
    """Mean full-answer log probability in nats, with causal target shift."""
    if logits.ndim != 3 or ids.shape != answer_mask.shape or logits.shape[:2] != ids.shape:
        raise ValueError("incompatible logits, token IDs, or answer masks")
    if bool(answer_mask[:, 0].any()) or bool((answer_mask[:, 1:].sum(1) == 0).any()):
        raise ValueError("answer tokens need a preceding prompt token")
    selected = answer_mask[:, 1:].bool()
    # Select before log_softmax: prompt/padding rows are irrelevant and may be
    # nonfinite in fully masked attention rows. Multiplying NaN by zero is NaN.
    lp = (
        torch.log_softmax(logits[:, :-1][selected].float(), -1)
        .gather(-1, ids[:, 1:][selected].unsqueeze(-1))
        .squeeze(-1)
    )
    totals = torch.zeros(ids.shape[0], dtype=lp.dtype, device=lp.device)
    totals.scatter_add_(0, selected.nonzero()[:, 0], lp)
    return totals / selected.sum(1)


def pad_encoded(
    encoded: list[tuple[list[int], list[bool]]], pad_id: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not encoded:
        raise ValueError("empty batch")
    width = max(len(ids) for ids, _ in encoded)
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    targets = torch.zeros_like(ids, dtype=torch.bool)
    for i, (seq, mask) in enumerate(encoded):
        if len(seq) != len(mask):
            raise ValueError("token/mask length mismatch")
        ids[i, : len(seq)] = torch.tensor(seq, device=device)
        attention[i, : len(seq)] = 1
        targets[i, : len(seq)] = torch.tensor(mask, device=device)
    return ids, attention, targets


def score_answers(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    answers: list[str],
    *,
    device: str | torch.device,
    batch_size: int = 4,
    max_context: int = 4096,
    budget: RunBudget | None = None,
) -> list[float]:
    if len(prompts) != len(answers) or batch_size < 1:
        raise ValueError("invalid scoring batch")
    values = [None] * len(prompts)
    encoded = [encode_answer(tokenizer, p, a) for p, a in zip(prompts, answers)]
    if any(len(seq) > max_context for seq, _ in encoded):
        raise ValueError("scoring context overflow")
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    model.eval()
    for start in range(0, len(order), batch_size):
        if budget:
            budget.check()
        batch = order[start : start + batch_size]
        ids, attention, mask = pad_encoded(
            [encoded[i] for i in batch], tokenizer.pad_token_id, device
        )
        with torch.no_grad():
            logits = model(ids, attention_mask=attention, use_cache=False).logits
            scored = complete_answer_log_probs(logits, ids, mask).cpu().tolist()
        for i, value in zip(batch, scored):
            values[i] = float(value)
    return values


def score_choices(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    options: list[list[str]],
    *,
    device: str | torch.device,
    batch_size: int = 4,
    max_context: int = 4096,
    budget: RunBudget | None = None,
) -> list[list[float]]:
    """Score stable one-token labels with one prompt forward per example.

    The returned values are vocabulary-normalized log probabilities, in each
    example's original option order, identical to teacher forcing one answer
    token. Prompt length plus one answer token must fit the context budget.
    Padding is constructed locally; tokenizer padding settings are untouched.
    Only the last prompt position reaches the vocabulary projection.
    """
    from .data import single_token_labels

    if len(prompts) != len(options) or batch_size < 1:
        raise ValueError("invalid choice scoring batch")
    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    if any(not ids for ids in encoded):
        raise ValueError("Choice scoring requires nonempty prompts")
    if any(len(ids) + 1 > max_context for ids in encoded):
        raise ValueError("choice scoring context overflow")
    label_ids = []
    for prompt, labels in zip(prompts, options, strict=True):
        if not labels:
            raise ValueError("Every choice example requires at least one label")
        stable = single_token_labels(prompt, labels, tokenizer)
        separate = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if separate != [[token] for token in stable]:
            raise ValueError("Choice labels must preserve the explicit answer-token protocol")
        label_ids.append(stable)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    values: list[list[float]] = [[] for _ in prompts]
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for start in range(0, len(order), batch_size):
            if budget:
                budget.check()
            batch = order[start : start + batch_size]
            width = max(len(encoded[i]) for i in batch)
            ids = torch.full(
                (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            attention = torch.zeros_like(ids)
            for row, i in enumerate(batch):
                ids[row, -len(encoded[i]) :] = torch.tensor(encoded[i], device=device)
                attention[row, -len(encoded[i]) :] = 1
            positions = (attention.cumsum(-1) - 1).clamp_min(0)
            with torch.no_grad():
                logits = model(
                    input_ids=ids,
                    attention_mask=attention,
                    position_ids=positions,
                    logits_to_keep=1,
                    use_cache=False,
                ).logits
                if logits.shape[1] != 1:
                    raise RuntimeError("Choice scoring requires logits_to_keep support")
                log_probs = logits[:, 0].float().log_softmax(-1)
            for row, i in enumerate(batch):
                scored = log_probs[row, label_ids[i]]
                if not torch.isfinite(scored).all():
                    raise FloatingPointError("Nonfinite choice log probability")
                values[i] = scored.cpu().tolist()
    finally:
        for module, training in states:
            module.training = training
    return values


def generate_answers(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    device: str,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    max_context: int = 4096,
    stop_strings: list[str] | None = None,
    budget: RunBudget | None = None,
) -> list[dict]:
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch size and output-token limit must be positive")
    rows: list[dict | None] = [None] * len(prompts)
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    # Examples close to the context limit need different generation caps.
    # Group by cap so a longer batchmate never shortens another item's output.
    by_cap = defaultdict(list)
    for i in sorted(range(len(prompts)), key=lengths.__getitem__):
        by_cap[min(max_new_tokens, max_context - lengths[i])].append(i)
    batches = [
        indices[start : start + batch_size]
        for indices in by_cap.values()
        for start in range(0, len(indices), batch_size)
    ]
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        for candidates in batches:
            if budget:
                budget.check()
            batch = [i for i in candidates if 0 < lengths[i] < max_context]
            for i in set(candidates) - set(batch):
                rows[i] = {
                    "generation": "",
                    "context_overflow": True,
                    "truncated": False,
                    "input_tokens": lengths[i],
                    "output_tokens": 0,
                }
            if not batch:
                continue
            encoded = tokenizer(
                [prompts[i] for i in batch],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
            width = encoded["input_ids"].shape[1]
            cap = min(max_new_tokens, max_context - width)
            kwargs = {
                "do_sample": False,
                "max_new_tokens": cap,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if stop_strings:
                kwargs.update(stop_strings=stop_strings, tokenizer=tokenizer)
            with torch.no_grad():
                output = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    **kwargs,
                )
            for j, i in enumerate(batch):
                seq = output[j, width:].tolist()
                eos = tokenizer.eos_token_id
                if eos in seq:
                    seq = seq[: seq.index(eos) + 1]
                generation = tokenizer.decode(seq, skip_special_tokens=True)
                stopped = any(s in generation for s in (stop_strings or []))
                rows[i] = {
                    "generation": generation,
                    "context_overflow": False,
                    "truncated": len(seq) >= cap and eos not in seq and not stopped,
                    "input_tokens": lengths[i],
                    "output_tokens": len(seq),
                }
    finally:
        tokenizer.padding_side = previous_padding
    return rows


def select_groups(examples: list[Any], groups_per_task: int | None, seed: int) -> list[Any]:
    """Keep complete source groups and all their control variants."""
    if groups_per_task is None:
        return examples
    if groups_per_task < 1:
        raise ValueError("group count must be positive")
    buckets = defaultdict(list)
    for ex in examples:
        buckets[(ex.task, ex.metadata.get("split", "test"))].append(ex)
    rng = np.random.default_rng(seed)
    result = []
    for key in sorted(buckets):
        rows = buckets[key]
        groups = sorted({ex.group for ex in rows})
        chosen = set(rng.permutation(groups)[:groups_per_task])
        result.extend(ex for ex in rows if ex.group in chosen)
    return result


def make_pairs(
    examples: list[Any],
    tokenizer: Any,
    *,
    max_pairs_per_task: int,
    seed: int,
    max_context: int = 4096,
    repeats_per_group_pair: int = 1,
    pair_filter: Callable[[Any, Any], bool] | None = None,
) -> tuple[list[tuple[Any, Any]], dict]:
    """Locked disjoint group partnerships, with optional row-disjoint repeats.

    First select independent group pairs using the original pilot protocol.
    Then add up to ``repeats_per_group_pair`` total rows per partnership,
    round-robin across partnerships, preserving the original clean/corrupt
    group orientation. A group never acquires a second partner. Repeated rows
    must be clustered by the source-group partnership in downstream statistics.
    An optional deterministic, directional filter verifies candidate pairs
    BEFORE locking groups or consuming the quota. It also applies to repeats;
    its results are cached per ordered row pair so expensive verifiers run once.

    Alignment means corresponding token positions, as in the existing arithmetic
    method; it is not a claim of identical semantic roles at every position.
    """
    if max_pairs_per_task < 1 or max_context < 2 or repeats_per_group_pair < 1:
        raise ValueError("Pair count must be positive and context must hold prompt plus answer")
    buckets = defaultdict(list)
    excluded_overflow = 0
    row_ids = set()
    for ex in examples:
        row_key = (ex.task, ex.id)
        if row_key in row_ids:
            raise ValueError("Pair input IDs must be unique within each task")
        row_ids.add(row_key)
        ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
        answer = tokenizer.encode(ex.answer, add_special_tokens=False)
        if not ids or not answer:
            raise ValueError("Pair prompts and answers must each contain tokens")
        if len(ids) + len(answer) > max_context:
            excluded_overflow += 1
            continue
        key = (
            ex.task,
            ex.metadata.get("split", "test"),
            ex.metadata.get("variant", "original"),
            len(ids),
        )
        buckets[key].append(ex)
    rng = np.random.default_rng(seed)
    pairs = []
    used_groups = set()
    task_counts = defaultdict(int)
    keys = sorted(buckets)
    # A fixed shortest-first traversal biases which groups hit the task cap.
    keys = [keys[index] for index in rng.permutation(len(keys))]
    indexed_rows = defaultdict(lambda: defaultdict(list))
    filter_results: dict[tuple[str, str, str], bool] = {}

    def accepted(a: Any, b: Any) -> bool:
        if pair_filter is None:
            return True
        key = (a.task, a.id, b.id)
        if key not in filter_results:
            filter_results[key] = bool(pair_filter(a, b))
        return filter_results[key]

    for key in keys:
        rows = sorted(buckets[key], key=lambda e: e.id)
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        for row in rows:
            indexed_rows[(row.task, row.group)][key].append(row)
        for a in rows:
            if (a.task, a.group) in used_groups or task_counts[a.task] >= max_pairs_per_task:
                continue
            for b in rows:
                if (b.task, b.group) in used_groups or b.group == a.group or a.answer == b.answer:
                    continue
                if not accepted(a, b):
                    continue
                pairs.append((a, b))
                used_groups.update(((a.task, a.group), (b.task, b.group)))
                task_counts[a.task] += 1
                break

    # Freeze the group-level matching before adding any within-group repeats.
    # In particular, do not greedily let A pair with B on one row and C on
    # another: that creates overlapping bootstrap clusters and leakage paths.
    partnerships = list(pairs)
    partnership_counts = [1] * len(partnerships)
    used_rows = {(ex.task, ex.id) for pair in pairs for ex in pair}
    if repeats_per_group_pair > 1:
        progress = True
        while progress:
            progress = False
            for index, (first_clean, first_corrupt) in enumerate(partnerships):
                task = first_clean.task
                if (
                    partnership_counts[index] >= repeats_per_group_pair
                    or task_counts[task] >= max_pairs_per_task
                ):
                    continue
                left = indexed_rows[(task, first_clean.group)]
                right = indexed_rows[(task, first_corrupt.group)]
                found = None
                for key in keys:
                    if key not in left or key not in right:
                        continue
                    for a in left[key]:
                        if (task, a.id) in used_rows:
                            continue
                        for b in right[key]:
                            if (
                                (task, b.id) not in used_rows
                                and a.answer != b.answer
                                and accepted(a, b)
                            ):
                                found = (a, b)
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                if found is None:
                    continue
                pairs.append(found)
                used_rows.update((task, ex.id) for ex in found)
                task_counts[task] += 1
                partnership_counts[index] += 1
                progress = True

    independent_counts = defaultdict(int)
    for a, _b in partnerships:
        independent_counts[a.task] += 1
    return pairs, {
        "input_examples": len(examples),
        "matched_pairs": len(pairs),
        "pairs_per_task": dict(task_counts),
        "used_groups": len(used_groups),
        "available_groups": len({(e.task, e.group) for e in examples}),
        "excluded_context_overflow": excluded_overflow,
        "eligible_examples": len(examples) - excluded_overflow,
        "independent_group_pairs": len(partnerships),
        "group_pairs_per_task": dict(independent_counts),
        "repeats_per_group_pair": repeats_per_group_pair,
        "repeated_pairs": len(pairs) - len(partnerships),
        "filter_candidates_tested": len(filter_results),
        "filter_candidates_rejected": sum(not passed for passed in filter_results.values()),
        "group_pair_counts": [
            {"task": a.task, "clean_group": a.group, "corrupt_group": b.group, "n_pairs": count}
            for (a, b), count in zip(partnerships, partnership_counts, strict=True)
        ],
    }
