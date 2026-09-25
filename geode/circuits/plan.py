"""Frozen CPU sampling, corruption and probe plans shared by all checkpoints.

A plan stores complete selected examples and resolved diagnostic token IDs.
Its fingerprint binds every payload byte to the dataset, tokenizer, planning
source and sampling settings. Preparing a plan never loads model weights.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import fingerprint, sha256_file, utc_now, write_json
from .data import Example, ethics_controls, load_examples
from .diagnostics import build_candidate_pairs
from .execution import select_groups

SCHEMA_VERSION = 1
PLAN_SETTINGS = (
    "mode",
    "tasks",
    "seed",
    "groups",
    "instances_per_group",
    "pairs",
    "pair_pool_groups",
    "probe_groups",
    "interventions",
    "max_context",
    "include_hard",
)
PLAN_SOURCE_FILES = (
    "plan.py",
    "runner.py",
    "data.py",
    "diagnostics.py",
    "execution.py",
    "code_eval.py",
    "crux_prompts.py",
    "gsm_shots.py",
)


def planning_source_fingerprint() -> str:
    """Bind the cache to the actual checked-in planning implementation."""
    root = Path(__file__).parent
    return fingerprint({name: sha256_file(root / name) for name in PLAN_SOURCE_FILES})


def plan_binding(args: Any, dataset_manifest: dict, tokenizer_hash: str) -> dict:
    """Exclude checkpoint/device/pricing/output choices, which cannot change samples."""
    from .runner import TOKEN_PROTOCOL

    return {
        "dataset_fingerprint": fingerprint(dataset_manifest),
        "tokenizer_hash": tokenizer_hash,
        "source_fingerprint": planning_source_fingerprint(),
        "token_protocol": TOKEN_PROTOCOL,
        "settings": {name: getattr(args, name) for name in PLAN_SETTINGS},
    }


def example_from_dict(value: dict) -> Example:
    """Restore the immutable public Example interface after a JSON round trip."""
    return Example(**{**value, "options": tuple(value.get("options", ()))})


def resolve_pools(args: Any) -> tuple[list[Example], list[Example], list[Example]]:
    """Select source pools with exactly the runner's existing grouped protocol."""
    from .runner import cap_instances

    pool = [
        ex
        for ex in load_examples(args.data, include_hard=args.include_hard)
        if ex.task in args.tasks
    ]
    behavioral = cap_instances(
        select_groups(pool, args.groups if args.mode == "pilot" else None, args.seed),
        args.instances_per_group if args.mode == "pilot" else None,
    )
    diagnostic_pool = cap_instances(
        select_groups(
            pool, args.pair_pool_groups if args.mode == "pilot" else None, args.seed + 31
        ),
        2 if args.mode == "pilot" else None,
    )
    if args.mode == "pilot":
        diagnostic_pool = [ex for ex in diagnostic_pool if ex.task != "cruxeval_input"] + [
            ex for ex in pool if ex.task == "cruxeval_input"
        ]
    probe_pool = [
        ex
        for ex in load_examples(args.data, include_hard=False, include_train=True)
        if ex.task in args.tasks
        and (not ex.task.startswith("ethics_") or ex.metadata["split"] == "train")
    ]
    return behavioral, diagnostic_pool, probe_pool


def _pair_record(a: Example, b: Example, tokenizer: Any) -> dict:
    from .runner import _pair_inputs

    inputs = _pair_inputs(a, b, tokenizer)
    return {"clean": asdict(a), "corrupt": asdict(b), "inputs": inputs}


def decode_pair(record: dict) -> tuple[Example, Example, tuple]:
    return (
        example_from_dict(record["clean"]),
        example_from_dict(record["corrupt"]),
        tuple(record["inputs"]),
    )


def circuit_plan(pair_pool: list[Example], tokenizer: Any, args: Any) -> dict:
    """Freeze all scoring pairs and source-disjoint held-out intervention pairs."""
    from .runner import controlled_pairs, diagnostic_example, task_quota

    diagnostics = [diagnostic_example(ex) for ex in pair_pool]
    tasks = sorted({ex.task for ex in diagnostics})
    heldout_groups = set()
    if args.mode == "full":
        rng = np.random.default_rng(args.seed + 901)
        for task in tasks:
            groups = sorted({ex.group for ex in diagnostics if ex.task == task})
            heldout_groups.update(
                (task, g) for g in rng.permutation(groups)[: max(2, len(groups) // 5)]
            )
    result: dict[str, Any] = {
        "tasks": {},
        "coverage": {"by_task": {}, "heldout_source_groups": len(heldout_groups)},
    }
    for task in tasks:
        pairs, coverage = controlled_pairs(
            [
                ex
                for ex in diagnostics
                if ex.task == task and (task, ex.group) not in heldout_groups
            ],
            tokenizer,
            count=task_quota(task, args.pairs, args.mode),
            seed=args.seed,
            max_context=args.max_context,
            repeats_per_group_pair=50 if args.mode == "full" else 1,
        )
        used_groups = {ex.group for pair in pairs for ex in pair}
        interventions, intervention_coverage = controlled_pairs(
            [ex for ex in diagnostics if ex.task == task and ex.group not in used_groups],
            tokenizer,
            count=task_quota(task, args.interventions, args.mode),
            seed=args.seed + 1,
            max_context=args.max_context,
        )
        result["tasks"][task] = {
            "pairs": [_pair_record(a, b, tokenizer) for a, b in pairs],
            "interventions": [_pair_record(a, b, tokenizer) for a, b in interventions],
            "intervention_coverage": intervention_coverage,
        }
        result["coverage"]["by_task"][task] = coverage
    return result


def probe_plan(pool: list[Example], tokenizer: Any, args: Any) -> dict:
    """Freeze reciprocal candidates, group clusters and exact feature-token positions."""
    from .runner import cap_instances, diagnostic_example

    by_task = defaultdict(list)
    for ex in pool:
        by_task[ex.task].append(ex)
    result = {}
    for task, task_pool in sorted(by_task.items()):
        omitted: Counter = Counter()
        rows = []
        if task.startswith("ethics_"):
            examples = select_groups(task_pool, args.probe_groups, args.seed + 71)
            for original in examples:
                for ex in ethics_controls(original, tokenizer):
                    ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
                    label = ex.metadata.get("presented_semantic_label", ex.label)
                    rows.append((ex.id, ex.group, label, ids, None, None))
        else:
            examples = cap_instances(
                select_groups(task_pool, max(128, args.probe_groups * 4), args.seed + 71), 1
            )
            matched = build_candidate_pairs(
                examples, tokenizer, seed=args.seed, max_pairs=args.probe_groups
            )
            omitted["unmatched_source_groups"] = len({ex.group for ex in examples}) - 2 * len(
                matched
            )
            for a, b, answer_a, answer_b in matched:
                cluster = a.group + "|" + b.group
                for ex, positive, negative in ((a, answer_a, answer_b), (b, answer_b, answer_a)):
                    prompt = diagnostic_example(ex).prompt + "\nCandidate answer: "
                    prefix = tokenizer.encode(prompt, add_special_tokens=False)
                    for label, candidate in ((1, positive), (0, negative)):
                        answer_ids = tokenizer.encode(candidate, add_special_tokens=False)
                        rows.append(
                            (
                                ex.id + f":candidate:{label}",
                                cluster,
                                label,
                                prefix + answer_ids,
                                answer_ids,
                                candidate,
                            )
                        )
        bad_groups = {group for _, group, _, ids, _, _ in rows if len(ids) > args.max_context}
        omitted["context_overflow_rows"] = sum(row[1] in bad_groups for row in rows)
        # Match the legacy omission dictionary when no overflow occurred.
        if not omitted["context_overflow_rows"]:
            del omitted["context_overflow_rows"]
        result[task] = {
            "rows": [row for row in rows if row[1] not in bad_groups],
            "omitted": dict(omitted),
        }
    return result


def build_plan(
    args: Any,
    tokenizer: Any,
    dataset_manifest: dict,
    tokenizer_hash: str,
    *,
    pools: tuple[list[Example], list[Example], list[Example]] | None = None,
) -> dict:
    """Run CPU matching once; no model weights, activations or GPU operations."""
    behavioral, circuits, probes = resolve_pools(args) if pools is None else pools
    content = {
        "schema_version": SCHEMA_VERSION,
        "binding": plan_binding(args, dataset_manifest, tokenizer_hash),
        "payload": {
            "behavior": [asdict(ex) for ex in behavioral],
            "circuits": circuit_plan(circuits, tokenizer, args),
            "probes": probe_plan(probes, tokenizer, args),
        },
    }
    return {**content, "fingerprint": fingerprint(content), "created_utc": utc_now()}


def validate_plan(
    plan: dict,
    args: Any,
    dataset_manifest: dict,
    tokenizer_hash: str | None = None,
) -> None:
    """Reject stale sources/settings/data, wrong tokenizers and altered plan payloads."""
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported frozen plan schema")
    content = {key: plan[key] for key in ("schema_version", "binding", "payload")}
    if fingerprint(content) != plan.get("fingerprint"):
        raise ValueError("frozen plan content fingerprint mismatch")
    expected = plan_binding(
        args, dataset_manifest, tokenizer_hash or plan["binding"]["tokenizer_hash"]
    )
    for key, value in expected.items():
        if plan["binding"].get(key) != value:
            raise ValueError(f"frozen plan mismatch: {key}")


def load_plan(
    path: str | Path, args: Any, dataset_manifest: dict, tokenizer_hash: str | None = None
) -> dict:
    plan = json.loads(Path(path).read_text())
    validate_plan(plan, args, dataset_manifest, tokenizer_hash)
    return plan


def save_plan(path: str | Path, plan: dict) -> None:
    """A frozen plan is immutable; use a fresh output path for a changed protocol."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("fingerprint") != plan["fingerprint"] or fingerprint(
            existing
        ) != fingerprint(plan):
            raise FileExistsError(f"refusing to overwrite frozen plan: {path}")
        return
    write_json(path, plan)
