"""Reasoning-mode ("think before answering") evaluation of one OLMo 2 checkpoint.

The circuit-overlap run scored CRUXEval with the published direct prompts and
ETHICS with a one-token label protocol. This script re-scores the same items
after letting the model reason first:

* CRUXEval uses the published chain-of-thought prompts (same pinned commit).
* ETHICS keeps the exact control prompts (2 vocabularies x 2 mappings x 2
  positions) but replaces "Reply with only the label." with an instruction to
  think step by step and finish with ``Answer: <label>``.

GSM-Symbolic is skipped: the original run already used eight-shot
chain-of-thought there. Everything else (data cache, pinned revisions, greedy
decoding, grading) reuses the library.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

from geode.circuits.artifacts import write_json, write_jsonl
from geode.circuits.checkpoints import select_checkpoints
from geode.circuits.crux_prompts import make_cot_input_prompt, make_cot_output_prompt
from geode.circuits.data import ethics_controls, grade, load_examples, native_aggregate
from geode.circuits.execution import generate_answers

LABEL_SUFFIX = "Reply with only the label.\nAnswer:\n"
THINK_SUFFIX = (
    "Think step by step about the question first. "
    'Then finish with one final line of the form "Answer: <label>".\n'
)
FULL_CRUX_GROUPS = 800


def build_prompts(examples, tokenizer, use_chat_template: bool):
    """Return (examples with reasoning prompts, stop strings per task)."""
    rebuilt = []
    for ex in examples:
        if ex.task.startswith("cruxeval_"):
            source = ex.metadata["source"]
            if ex.metadata["direction"] == "input":
                prompt = make_cot_input_prompt((source["code"], source["output"]))
            else:
                prompt = make_cot_output_prompt((source["code"], source["input"]))
        elif ex.task.startswith("ethics_"):
            if ex.prompt.count(LABEL_SUFFIX) != 1:
                raise ValueError(f"unexpected ETHICS prompt layout for {ex.id}")
            prompt = ex.prompt.replace(LABEL_SUFFIX, THINK_SUFFIX)
        else:
            raise ValueError(f"task not supported in think mode: {ex.task}")
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        rebuilt.append(replace(ex, prompt=prompt, metadata={**ex.metadata, "think_prompt": True}))
    return rebuilt


def parse_ethics_answer(generation: str, options: tuple[str, ...]) -> str | None:
    alternatives = "|".join(re.escape(o) for o in options)
    match = re.search(rf"[Aa]nswer\s*:\s*\**\s*({alternatives})\b", generation)
    return match.group(1) if match else None


def grade_record(ex, row: dict) -> dict:
    if ex.task.startswith("ethics_"):
        prediction = parse_ethics_answer(row["generation"], ex.options)
        scored = {
            "correct": prediction is not None and prediction == ex.answer,
            "parse_failure": prediction is None,
            "prediction": prediction,
        }
    else:
        scored = grade(ex, row["generation"])
    return {
        "id": ex.id,
        "task": ex.task,
        "group": ex.group,
        "prompt": ex.prompt,
        "answer": ex.answer,
        "label": ex.label,
        "options": list(ex.options),
        "metadata": ex.metadata,
        **row,
        **scored,
    }


def load_model(checkpoint, device: str, tiny: bool):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.repo, revision=checkpoint.revision)
    tokenizer.pad_token = tokenizer.eos_token
    if tiny:
        torch.manual_seed(0)
        config = Olmo2Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = Olmo2ForCausalLM(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint.repo,
            revision=checkpoint.revision,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        )
    return tokenizer, model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="rlvr2")
    parser.add_argument("--tasks", nargs="+", default=["cruxeval", "ethics"])
    parser.add_argument("--ethics-groups", type=int, default=100, help="source groups per task")
    parser.add_argument("--project-ethics-groups", type=int, default=100)
    parser.add_argument("--crux-groups", type=int, default=None, help="default: all 800")
    parser.add_argument("--no-ethics-controls", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crux-max-new-tokens", type=int, default=1024)
    parser.add_argument("--ethics-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hourly-rate", type=float, default=0.0)
    parser.add_argument("--tiny-model", action="store_true", help="random tiny model; smoke only")
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not args.confirm_cost:
        raise SystemExit("GPU inference bills rental time; pass --confirm-cost")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = select_checkpoints([args.stage])[0]
    tokenizer, model = load_model(checkpoint, args.device, args.tiny_model)

    pools = {"cruxeval": args.crux_groups, "ethics": args.ethics_groups}
    selected = []
    for family in args.tasks:
        limit = pools[family]
        examples = [
            ex
            for ex in load_examples(
                args.data, seed=args.seed, limit_per_task=limit, include_hard=False
            )
            if ex.task.startswith(family)
        ]
        if family == "ethics" and not args.no_ethics_controls:
            examples = [c for ex in examples for c in ethics_controls(ex, tokenizer)]
        selected.extend(examples)
    selected = build_prompts(selected, tokenizer, args.chat_template)
    write_jsonl(output / "selected_examples.jsonl", [asdict(ex) for ex in selected])
    write_json(
        output / "run_metadata.json",
        {
            "checkpoint": checkpoint.to_dict(),
            "args": vars(args),
            "think_suffix": THINK_SUFFIX,
            "n_examples": len(selected),
            "tiny_model": args.tiny_model,
        },
    )

    by_task = defaultdict(list)
    for ex in selected:
        by_task[ex.task].append(ex)
    records, timings = [], []
    for task, examples in sorted(by_task.items()):
        is_crux = task.startswith("cruxeval_")
        started = time.monotonic()
        rows = generate_answers(
            model,
            tokenizer,
            [ex.prompt for ex in examples],
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.crux_max_new_tokens if is_crux else args.ethics_max_new_tokens,
            max_context=args.max_context,
            stop_strings=["[/ANSWER]"] if is_crux else None,
        )
        seconds = time.monotonic() - started
        task_records = [grade_record(ex, row) for ex, row in zip(examples, rows)]
        records.extend(task_records)
        timings.append(
            {
                "task": task,
                "n_rows": len(examples),
                "seconds": seconds,
                "generated_tokens": sum(r["output_tokens"] for r in task_records),
                "n_truncated": sum(r["truncated"] for r in task_records),
                "n_context_overflow": sum(r["context_overflow"] for r in task_records),
                "tokens_per_second": sum(r["output_tokens"] for r in task_records) / seconds,
                "accuracy": sum(r["correct"] for r in task_records) / len(task_records),
                "parse_failure_rate": sum(r["parse_failure"] for r in task_records)
                / len(task_records),
            }
        )
        print(json.dumps(timings[-1]), flush=True)
        write_jsonl(output / "behavior.jsonl", records)
        write_json(output / "timing.json", timings)

    summary = native_aggregate(records)
    # Mean over the eight position/label controls: the honest ETHICS number.
    controls = defaultdict(list)
    for key, value in summary.items():
        task, split, variant = key.split("/")
        if task.startswith("ethics_") and variant.startswith("control_"):
            controls[f"{task}/{split}/control_mean"].append(value["accuracy"])
    for key, values in controls.items():
        summary[key] = {"accuracy": sum(values) / len(values), "n_controls": len(values)}
    write_json(output / "behavior_summary.json", summary)

    # Full-scope projection from measured throughput; ETHICS groups differ per task.
    projection = {}
    total_seconds = 0.0
    for t in timings:
        family = "cruxeval" if t["task"].startswith("cruxeval_") else "ethics"
        per_row = t["seconds"] / t["n_rows"]
        if family == "cruxeval":
            full_rows = FULL_CRUX_GROUPS
        else:
            per_group = t["n_rows"] / len({r["group"] for r in records if r["task"] == t["task"]})
            full_rows = args.project_ethics_groups * per_group
        projection[t["task"]] = {
            "measured_rows": t["n_rows"],
            "projected_rows": full_rows,
            "projected_seconds": per_row * full_rows,
        }
        total_seconds += per_row * full_rows
    projection["total_hours"] = total_seconds / 3600
    projection["total_usd_at_hourly_rate"] = args.hourly_rate * total_seconds / 3600
    projection["note"] = "linear in rows at the pilot's mean seconds/row; excludes model load"
    write_json(output / "projection.json", projection)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if "control_mean" in k or "cruxeval" in k}, indent=1
        )
    )
    print(json.dumps(projection, indent=1))


if __name__ == "__main__":
    main()
