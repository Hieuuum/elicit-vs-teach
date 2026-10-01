"""Explicit-cost offline benchmark/circuit runner; pilot never launches a full run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .artifacts import (
    environment_provenance,
    fingerprint,
    start_run,
    utc_now,
    write_json,
    write_jsonl,
)
from .attribution import extract_residual_features, patch_pair, score_pair
from .checkpoints import CHECKPOINTS, select_checkpoints
from .data import (
    Example,
    ethics_controls,
    grade,
    native_aggregate,
    numeric_answer,
    prepare_data,
    single_token_labels,
    verify_data,
)
from .diagnostics import canonical_candidate, valid_corruption
from .execution import (
    RunBudget,
    encode_answer,
    generate_answers,
    make_pairs,
    score_answers,
    score_choices,
)
from .probes import evaluate_layerwise_probes
from .statistics import sample_type_matched_nodes, topk_nodes

TOKEN_PROTOCOL = (
    "fixed prompt token IDs + separately encoded complete answer; no gold target rationale"
)
TASKS = (
    "ethics_commonsense",
    "ethics_deontology",
    "ethics_justice",
    "ethics_virtue",
    "ethics_utilitarianism",
    "gsm_symbolic",
    "cruxeval_input",
    "cruxeval_output",
)


def generation_cap(args, task: str) -> int:
    """Resolve the recorded task cap; ETHICS always selects one label token."""
    if task.startswith("ethics_"):
        return 1
    field = "math_max_new_tokens" if task == "gsm_symbolic" else "coding_max_new_tokens"
    override = getattr(args, field, None)
    return args.max_new_tokens if override is None else override


def diagnostic_example(example: Example) -> Example:
    """Answer diagnostics never condition on the target's gold reasoning.

    GSM's native generation prompt remains unchanged for behavioral evaluation;
    this explicitly labelled diagnostic asks for the final number directly.
    """
    if example.task == "gsm_symbolic":
        return replace(
            example,
            prompt=example.prompt + "\nThe final answer is ",
            answer=str(numeric_answer(example.answer)),
        )
    if example.task.startswith("cruxeval_"):
        candidate = canonical_candidate(example)
        if candidate is None:
            raise ValueError(f"cannot parse reference: {example.id}")
        direction = example.task.removeprefix("cruxeval_")
        suffix = (
            "\nProvide only the function call f(...) that yields the requested output:\n"
            if direction == "input"
            else "\nProvide only the resulting Python value:\n"
        )
        return replace(example, prompt=example.prompt + suffix, answer=candidate)
    return example


def cap_instances(examples: list[Example], per_group: int | None) -> list[Example]:
    if per_group is None:
        return examples
    seen = Counter()
    rows = []
    for ex in sorted(examples, key=lambda e: e.id):
        # Native ETHICS groups must remain complete. Only cap GSM variants.
        if ex.task == "gsm_symbolic":
            if seen[ex.group] >= per_group:
                continue
            seen[ex.group] += 1
        rows.append(ex)
    return rows


def behavior(
    model, tokenizer, examples: list[Example], *, args, budget: RunBudget, output: Path
) -> tuple[list[dict], list[dict]]:
    rows, timings = [], []
    by_task = defaultdict(list)
    for ex in examples:
        by_task[ex.task].append(ex)
    for task, base in sorted(by_task.items()):
        budget.check()
        task_start = time.monotonic()
        rendered = []
        for ex in base:
            rendered.append(ex)
            if task.startswith("ethics_"):
                rendered.extend(ethics_controls(ex, tokenizer))
        if task.startswith("ethics_"):
            valid = []
            for i, ex in enumerate(rendered):
                single_token_labels(ex.prompt, ex.options, tokenizer)
                if (
                    len(tokenizer.encode(ex.prompt, add_special_tokens=False)) + 1
                    <= args.max_context
                ):
                    valid.append(i)
            scores = score_choices(
                model,
                tokenizer,
                [rendered[i].prompt for i in valid],
                [list(rendered[i].options) for i in valid],
                device=args.device,
                batch_size=args.batch_size,
                max_context=args.max_context,
                budget=budget,
            )
            score_map = dict(zip(valid, scores))
            for i, ex in enumerate(rendered):
                vals = score_map.get(i)
                label = ex.options.index(ex.answer)
                generation = ex.options[int(np.argmax(vals))] if vals is not None else ""
                rows.append(
                    {
                        **asdict(ex),
                        **grade(ex, generation),
                        "generation": generation,
                        "answer_log_prob_nats": vals[label] if vals is not None else None,
                        "logit_margin": vals[label] - vals[1 - label] if vals is not None else None,
                        "truncated": False,
                        "context_overflow": vals is None,
                        "input_tokens": len(tokenizer.encode(ex.prompt, add_special_tokens=False)),
                        "output_tokens": int(vals is not None),
                        "scoring_protocol": "single-token constrained choice",
                    }
                )
        else:
            generations = generate_answers(
                model,
                tokenizer,
                [ex.prompt for ex in rendered],
                device=args.device,
                batch_size=args.batch_size,
                max_new_tokens=generation_cap(args, task),
                max_context=args.max_context,
                stop_strings=rendered[0].metadata.get("stop_strings"),
                budget=budget,
            )
            diagnostics = [diagnostic_example(ex) for ex in rendered]
            valid = [
                i
                for i, ex in enumerate(diagnostics)
                if len(encode_answer(tokenizer, ex.prompt, ex.answer)[0]) <= args.max_context
            ]
            likelihoods = score_answers(
                model,
                tokenizer,
                [diagnostics[i].prompt for i in valid],
                [diagnostics[i].answer for i in valid],
                device=args.device,
                batch_size=args.batch_size,
                max_context=args.max_context,
                budget=budget,
            )
            lp_map = dict(zip(valid, likelihoods))
            for i, (ex, gen) in enumerate(zip(rendered, generations)):
                rows.append(
                    {
                        **asdict(ex),
                        **gen,
                        **grade(ex, gen["generation"]),
                        "answer_log_prob_nats": lp_map.get(i),
                        "logit_margin": None,
                        "diagnostic_prompt": diagnostics[i].prompt,
                        "diagnostic_answer": diagnostics[i].answer,
                        "max_new_tokens": generation_cap(args, task),
                        "scoring_protocol": TOKEN_PROTOCOL,
                    }
                )
        timings.append(
            {
                "component": "behavior",
                "task": task,
                "n_rows": len(rendered),
                "seconds": time.monotonic() - task_start,
                "generated_tokens": sum(r["output_tokens"] for r in rows if r["task"] == task),
                "n_truncated": sum(r["truncated"] for r in rows if r["task"] == task),
                "n_context_overflow": sum(r["context_overflow"] for r in rows if r["task"] == task),
                "batch_size": args.batch_size,
                "max_new_tokens": generation_cap(args, task),
                "max_context": args.max_context,
            }
        )
        write_jsonl(output / "behavior.jsonl", rows)
        write_json(output / "behavior_summary.json", native_aggregate(rows))
        write_json(output / "timing.json", timings)
        print(
            json.dumps({"event": "behavior_done", "stage": output.name, **timings[-1]}), flush=True
        )
    return rows, timings


def _pair_inputs(a: Example, b: Example, tokenizer) -> tuple:
    clean, mask = encode_answer(tokenizer, a.prompt, a.answer)
    corrupt, _ = encode_answer(tokenizer, b.prompt, a.answer)
    negative = None
    if a.options:
        other = next(x for x in a.options if x != a.answer)
        negative, _ = encode_answer(tokenizer, a.prompt, other)
    return clean, corrupt, mask, negative


def task_quota(task: str, count: int, mode: str) -> int:
    if mode == "full" and task.startswith("ethics_"):
        ethics_tasks = list(TASKS[:5])
        return count // 5 + int(ethics_tasks.index(task) < count % 5)
    if mode == "full" and task.startswith("cruxeval_"):
        return min(count, 400)
    return count


def controlled_pairs(
    examples: list[Example],
    tokenizer,
    *,
    count: int,
    seed: int,
    max_context: int,
    repeats_per_group_pair: int = 1,
) -> tuple[list[tuple[Example, Example]], dict]:
    """Freeze source pairs first; cluster all eight control renderings together."""
    plain = [ex for ex in examples if ex.task != "ethics_utilitarianism"]
    base_pairs, coverage = make_pairs(
        plain,
        tokenizer,
        max_pairs_per_task=count,
        seed=seed,
        max_context=max_context,
        repeats_per_group_pair=repeats_per_group_pair,
        pair_filter=lambda a, b: valid_corruption(a, b) if a.task.startswith("cruxeval_") else True,
    )
    result = []
    rejected = 0
    for a, b in base_pairs:
        variants = (
            list(zip(ethics_controls(a, tokenizer), ethics_controls(b, tokenizer)))
            if a.task.startswith("ethics_")
            else [(a, b)]
        )
        if any(
            len(tokenizer.encode(x.prompt, add_special_tokens=False))
            != len(tokenizer.encode(y.prompt, add_special_tokens=False))
            or len(encode_answer(tokenizer, x.prompt, x.answer)[0]) > max_context
            for x, y in variants
        ):
            rejected += 1
            continue  # omit entire source pair, never unbalance its control factorial
        result.extend(variants)
    # Utility CSV always prefers scenario one. Corrupt by swapping the actual
    # two scenarios, holding the output-label mapping fixed, in both directions.
    utility = sorted(
        [ex for ex in examples if ex.task == "ethics_utilitarianism"], key=lambda e: e.id
    )
    rng = np.random.default_rng(seed)
    chosen = 0
    for index in rng.permutation(len(utility)):
        ex = utility[index]
        controls = ethics_controls(ex, tokenizer)
        mapped = {(c.metadata["variant"]): c for c in controls}
        variants = []
        for a in controls:
            name = a.metadata["variant"]
            opposite = (
                name.replace("_p0_", "_p1_") if "_p0_" in name else name.replace("_p1_", "_p0_")
            )
            variants.append((a, mapped[opposite]))
        if any(
            len(tokenizer.encode(a.prompt, add_special_tokens=False))
            != len(tokenizer.encode(b.prompt, add_special_tokens=False))
            or len(encode_answer(tokenizer, a.prompt, a.answer)[0]) > max_context
            for a, b in variants
        ):
            rejected += 1
            continue
        result.extend(variants)
        chosen += 1
        if chosen >= count:
            break
    coverage.update(
        control_renderings=len(result),
        rejected_unaligned_control_groups=rejected,
        utilitarian_source_groups=chosen,
        controls="all eight renderings per accepted source pair; utility within-group scenario swap",
    )
    return result, coverage


def circuit_maps(
    model,
    tokenizer,
    pair_pool: list[Example],
    *,
    args,
    budget: RunBudget,
    output: Path,
    tokenizer_hash: str,
    resolved_plan: dict | None = None,
) -> tuple[list[dict], set[str]]:
    from .plan import circuit_plan, decode_pair

    if resolved_plan is None:
        resolved_plan = circuit_plan(pair_pool, tokenizer, args)
    write_json(output / "pair_coverage.json", resolved_plan["coverage"])
    by_task = {
        task: [decode_pair(row) for row in value["pairs"]]
        for task, value in resolved_plan["tasks"].items()
        if value["pairs"]
    }
    timings, used = [], set()
    rng = np.random.default_rng(args.seed)
    typed_rng = np.random.default_rng(args.seed + 271828)
    for task, batch in sorted(by_task.items()):
        started = time.monotonic()
        scores, item_ids, groups, clean_groups, corrupt_groups, metrics = [], [], [], [], [], []
        names = None
        for a, b, inputs in batch:
            budget.check()
            ids, corr, mask, negative = inputs
            scored = score_pair(model, ids, corr, mask, device=args.device, negative_ids=negative)
            if names is not None and names != scored["node_names"]:
                raise ValueError("node universe changed within checkpoint")
            names = scored["node_names"]
            scores.append(scored["scores"])
            item_ids.append(a.id + "|" + b.id)
            groups.append(a.group + "|" + b.group)
            clean_groups.append(a.group)
            corrupt_groups.append(b.group)
            metrics.append({"metric": scored["metric"], "corrupt_metric": scored["corrupt_metric"]})
            used.update((a.group, b.group))
        array = np.asarray(scores, dtype=np.float32)
        base = output / "circuits" / task
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base.with_suffix(".npz"), scores=array)
        meta = {
            "item_ids": item_ids,
            "group_ids": groups,
            "node_names": names,
            "clean_group_ids": clean_groups,
            "corrupt_group_ids": corrupt_groups,
            "protocol_hash": fingerprint(
                {
                    "task": task,
                    "protocol": TOKEN_PROTOCOL,
                    "pairs": [(asdict(a), asdict(b)) for a, b, _ in batch],
                }
            ),
            "tokenizer_hash": tokenizer_hash,
            "metrics": metrics,
            "top16": topk_nodes(array, names, k=min(16, len(names))),
        }
        write_json(base.with_suffix(".json"), meta)
        attribution_seconds = time.monotonic() - started
        intervention_started = time.monotonic()
        intervention_pairs = [
            decode_pair(row) for row in resolved_plan["tasks"][task]["interventions"]
        ]
        interventions = []
        for a, b, inputs in intervention_pairs:
            budget.check()
            ids, corr, mask, negative = inputs
            random_nodes = rng.choice(names, size=min(16, len(names)), replace=False).tolist()
            typed_seed = int(typed_rng.integers(0, 2**32))
            typed_nodes = sample_type_matched_nodes(meta["top16"], names, seed=typed_seed)
            top = patch_pair(
                model,
                ids,
                corr,
                mask,
                nodes=meta["top16"],
                negative_ids=negative,
                device=args.device,
            )
            random = patch_pair(
                model,
                ids,
                corr,
                mask,
                nodes=random_nodes,
                negative_ids=negative,
                device=args.device,
            )
            random_type_matched = patch_pair(
                model,
                ids,
                corr,
                mask,
                nodes=typed_nodes,
                negative_ids=negative,
                device=args.device,
            )
            interventions.append(
                {
                    "clean_id": a.id,
                    "corrupt_id": b.id,
                    "clean_group": a.group,
                    "corrupt_group": b.group,
                    "top": top,
                    "random": random,
                    "random_nodes": random_nodes,
                    "random_type_matched": random_type_matched,
                    "random_type_matched_nodes": typed_nodes,
                    "random_type_matched_seed": typed_seed,
                }
            )
        write_json(base.parent / (task + "_interventions.json"), interventions)
        timings.append(
            {
                "component": "circuits",
                "task": task,
                "n_pairs": len(batch),
                "n_intervention_pairs": len(interventions),
                "attribution_seconds": attribution_seconds,
                "intervention_seconds": time.monotonic() - intervention_started,
                "intervention_controls": ["uniform", "type_matched"],
                "seconds": time.monotonic() - started,
            }
        )
        print(
            json.dumps({"event": "circuits_done", "stage": output.name, **timings[-1]}), flush=True
        )
    return timings, used


def probe_features(
    model,
    tokenizer,
    pool: list[Example],
    *,
    args,
    budget: RunBudget,
    output: Path,
    resolved_plan: dict | None = None,
) -> list[dict]:
    """Extract only diagnostic positions; reciprocal candidates balance surface frequency."""
    from .plan import probe_plan

    if resolved_plan is None:
        resolved_plan = probe_plan(pool, tokenizer, args)
    timings = []
    for task, task_plan in sorted(resolved_plan.items()):
        started = time.monotonic()
        features, answer_features, labels, groups, item_ids, candidates = [], [], [], [], [], []
        omitted = Counter(task_plan["omitted"])
        extraction_rows = task_plan["rows"]
        for identifier, group, label, ids, answer_ids, candidate in extraction_rows:
            budget.check()
            features.append(extract_residual_features(model, ids, device=args.device).numpy())
            if answer_ids is not None:
                answer_features.append(
                    extract_residual_features(model, answer_ids, device=args.device).numpy()
                )
            labels.append(label)
            groups.append(group)
            item_ids.append(identifier)
            if candidate is not None:
                candidates.append(
                    {"id": identifier, "group": group, "candidate": candidate, "valid": label}
                )
        dest = output / "probes" / (task + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "task": task,
            "n_rows": len(labels),
            "n_groups": len(set(groups)),
            "omitted": dict(omitted),
            "item_ids": item_ids,
            "group_ids": groups,
            "position_policy": "final prompt token"
            if task.startswith("ethics_")
            else "last candidate token",
            "candidates": candidates,
            "candidate_matching": "reciprocal, exact token length and AST/type signature; source groups paired and held out jointly",
        }
        if features:
            array = np.stack(features)
            np.savez_compressed(
                dest.with_suffix(".npz"),
                features=array,
                labels=np.asarray(labels),
                groups=np.asarray(groups),
            )
            try:
                report.update(
                    evaluate_layerwise_probes(
                        array,
                        labels,
                        groups,
                        seed=args.seed,
                        n_shuffles=2 if args.mode == "pilot" else 5,
                    )
                )
                if answer_features:
                    report["answer_only"] = evaluate_layerwise_probes(
                        np.stack(answer_features),
                        labels,
                        groups,
                        seed=args.seed,
                        n_shuffles=2 if args.mode == "pilot" else 5,
                    )
                report["status"] = "complete"
            except ValueError as exc:
                report.update(status="inconclusive", reason=str(exc))
        else:
            report.update(status="inconclusive", reason="no matched diagnostic groups")
        write_json(dest, report)
        timings.append(
            {
                "component": "probes",
                "task": task,
                "n_rows": len(labels),
                "seconds": time.monotonic() - started,
            }
        )
        print(json.dumps({"event": "probes_done", "stage": output.name, **timings[-1]}), flush=True)
    return timings


def run(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from geode.zoo.activations import tokenizer_hash as hash_tokenizer

    budget = RunBudget(args.max_wall_seconds, args.hourly_rate, args.confirm_cost)
    print(
        json.dumps(
            {
                "event": "cost_estimate",
                "maximum_runtime_hours": args.max_wall_seconds / 3600,
                "maximum_compute_usd": budget.estimated_usd,
                "transfer_excluded": True,
            }
        ),
        flush=True,
    )
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    validate_config(args)
    checkpoints = select_checkpoints(args.stages)
    source = verify_data(args.data)
    from .plan import (
        build_plan,
        example_from_dict,
        load_plan,
        resolve_pools,
        validate_plan,
    )

    frozen_plan = load_plan(args.plan, args, source) if args.plan else None
    pools = None if frozen_plan is not None else resolve_pools(args)
    behavioral = (
        [example_from_dict(row) for row in frozen_plan["payload"]["behavior"]]
        if frozen_plan is not None
        else pools[0]
    )
    config = vars(args).copy()
    config["checkpoints"] = [c.to_dict() for c in checkpoints]
    config["dataset_manifest"] = source
    config["token_protocol"] = TOKEN_PROTOCOL
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    metadata = start_run(output, config, environment_provenance(root))
    write_jsonl(output / "selected_examples.jsonl", [asdict(ex) for ex in behavioral])
    write_json(output / "dataset_manifest.json", source)
    all_timings = []
    first_tokenizer_hash = None
    try:
        for checkpoint in checkpoints:
            budget.check()
            started = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(checkpoint.repo, revision=checkpoint.revision)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer_hash = hash_tokenizer(tokenizer)
            if first_tokenizer_hash is not None and first_tokenizer_hash != tokenizer_hash:
                raise ValueError("checkpoint tokenizer mismatch; matched stage comparison invalid")
            first_tokenizer_hash = tokenizer_hash
            if frozen_plan is None:
                frozen_plan = build_plan(args, tokenizer, source, tokenizer_hash, pools=pools)
            validate_plan(frozen_plan, args, source, tokenizer_hash)
            metadata["plan_fingerprint"] = frozen_plan["fingerprint"]
            if not (output / "frozen_plan.json").exists():
                write_json(output / "frozen_plan.json", frozen_plan)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint.repo,
                revision=checkpoint.revision,
                torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
            )
            model.to(args.device).eval()
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            stage_dir = output / checkpoint.stage
            stage_dir.mkdir()
            write_json(
                stage_dir / "checkpoint.json",
                {
                    **checkpoint.to_dict(),
                    "config": model.config.to_dict(),
                    "tokenizer_hash": tokenizer_hash,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                },
            )
            print(
                json.dumps(
                    {
                        "event": "checkpoint_loaded",
                        "stage": checkpoint.stage,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            startup_seconds = time.monotonic() - started
            _, times = behavior(
                model, tokenizer, behavioral, args=args, budget=budget, output=stage_dir
            )
            circuit_times, _ = circuit_maps(
                model,
                tokenizer,
                [],
                args=args,
                budget=budget,
                output=stage_dir,
                tokenizer_hash=tokenizer_hash,
                resolved_plan=frozen_plan["payload"]["circuits"],
            )
            probe_times = probe_features(
                model,
                tokenizer,
                [],
                args=args,
                budget=budget,
                output=stage_dir,
                resolved_plan=frozen_plan["payload"]["probes"],
            )
            times += (
                circuit_times + probe_times + [{"component": "startup", "seconds": startup_seconds}]
            )
            for row in times:
                row["stage"] = checkpoint.stage
            all_timings += times
            write_json(stage_dir / "timing.json", times)
            write_json(
                stage_dir / "hardware.json",
                {
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
                    if args.device.startswith("cuda")
                    else None,
                    "stage_seconds": time.monotonic() - started,
                },
            )
            del model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            write_json(output / "timing.json", all_timings)
        metadata.update(
            status="complete",
            finished_utc=utc_now(),
            elapsed_seconds=time.monotonic() - budget.started,
        )
    except BaseException as exc:
        metadata.update(
            status="failed",
            finished_utc=utc_now(),
            error=type(exc).__name__ + ": " + str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        write_json(output / "run_metadata.json", metadata)
    if not args.skip_report:
        from .report import generate_report

        generate_report(output, n_bootstrap=200 if args.mode == "pilot" else 2000, seed=args.seed)
    return metadata


def validate_config(args) -> None:
    for field in ("coding_max_new_tokens", "math_max_new_tokens"):
        value = getattr(args, field, None)
        if value is not None and value < 1:
            raise ValueError(f"{field} must be positive")
    for field in (
        "groups",
        "instances_per_group",
        "pairs",
        "pair_pool_groups",
        "probe_groups",
        "interventions",
        "batch_size",
        "max_context",
        "max_new_tokens",
        "cpu_threads",
    ):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if not args.tasks or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("tasks must be a nonempty unique list")
    if args.mode == "full" and args.stages != [c.stage for c in CHECKPOINTS]:
        raise ValueError("full mode requires all seven stages; use pilot for a subset")
    if args.mode == "full" and (args.probe_groups < 40 or args.interventions < 128):
        raise ValueError("full mode requires --probe-groups >=40 and --interventions >=128")
    if args.mode == "full" and args.pairs < 512:
        raise ValueError(
            "full mode requires --pairs 512 or larger (CRUX naturally has fewer groups)"
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "prepare-plan", "run", "report"))
    p.add_argument("--plan", help="Immutable CPU plan prepared before GPU rental")
    p.add_argument(
        "--skip-report",
        action="store_true",
        help="Save model artifacts; generate report later on CPU",
    )
    p.add_argument("--data", default="geode-store/olmo2-data")
    p.add_argument("--output", default="geode-store/olmo2-pilot")
    p.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    p.add_argument("--stages", nargs="+", default=["stage2", "rlvr2"])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--instances-per-group", type=int, default=2)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--pair-pool-groups", type=int, default=128)
    p.add_argument("--probe-groups", type=int, default=20)
    p.add_argument("--interventions", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--coding-max-new-tokens", type=int)
    p.add_argument("--math-max-new-tokens", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--include-hard", action="store_true")
    p.add_argument("--max-wall-seconds", type=float, default=7200)
    p.add_argument("--hourly-rate", type=float, default=0.66)
    p.add_argument("--confirm-cost", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_data(args.data)))
    elif args.command == "prepare-plan":
        from transformers import AutoTokenizer
        from geode.zoo.activations import tokenizer_hash as hash_tokenizer
        from .plan import build_plan, load_plan, save_plan

        validate_config(args)
        if not args.plan:
            raise ValueError("prepare-plan requires --plan PATH")
        source = verify_data(args.data)
        checkpoint = select_checkpoints(args.stages)[0]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.repo, revision=checkpoint.revision)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer_hash = hash_tokenizer(tokenizer)
        if Path(args.plan).exists():
            frozen = load_plan(args.plan, args, source, tokenizer_hash)
        else:
            frozen = build_plan(args, tokenizer, source, tokenizer_hash)
            save_plan(args.plan, frozen)
        print(
            json.dumps({"plan": args.plan, "fingerprint": frozen["fingerprint"], "status": "ready"})
        )
    elif args.command == "report":
        from .report import generate_report

        generate_report(
            Path(args.output), n_bootstrap=200 if args.mode == "pilot" else 2000, seed=args.seed
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
