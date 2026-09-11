"""Pilot-only batch calibration on exactly the same saved math examples."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from geode.circuits.artifacts import read_jsonl, write_json
from geode.circuits.checkpoints import select_checkpoints
from geode.circuits.data import Example, grade
from geode.circuits.execution import RunBudget, generate_answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["stage2", "rlvr2"])
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--hourly-rate", type=float, default=0.6144444444)
    parser.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args()
    budget = RunBudget(3600, args.hourly_rate, args.confirm_cost)
    print(
        json.dumps(
            {"maximum_compute_usd": budget.estimated_usd, "scope": "batch calibration only"}
        ),
        flush=True,
    )
    examples = [
        Example(**r)
        for r in read_jsonl(args.pilot / "selected_examples.jsonl")
        if r["task"] == "gsm_symbolic"
    ]
    if not examples:
        raise ValueError("no saved math examples")
    records = []
    out = args.pilot / "batch_calibration.json"
    if out.exists():
        raise FileExistsError(out)
    for cp in select_checkpoints(args.stages):
        tokenizer = AutoTokenizer.from_pretrained(cp.repo, revision=cp.revision)
        tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(
                cp.repo,
                revision=cp.revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        generate_answers(
            model, tokenizer, [examples[0].prompt], device="cuda", batch_size=1, max_new_tokens=8
        )
        for batch in args.batches:
            budget.check()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            outputs = generate_answers(
                model,
                tokenizer,
                [ex.prompt for ex in examples],
                device="cuda",
                batch_size=batch,
                max_new_tokens=args.max_new_tokens,
                max_context=4096,
                stop_strings=["Q:"],
                budget=budget,
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            rows = [
                {"id": ex.id, "group": ex.group, **gen, **grade(ex, gen["generation"])}
                for ex, gen in zip(examples, outputs)
            ]
            row = {
                "stage": cp.stage,
                "revision": cp.revision,
                "task": "gsm_symbolic",
                "batch_size": batch,
                "max_new_tokens": args.max_new_tokens,
                "seconds": elapsed,
                "n_rows": len(rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "n_correct": sum(r["correct"] for r in rows),
                "n_truncated": sum(r["truncated"] for r in rows),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "examples": rows,
            }
            records.append(row)
            write_json(out, records)
            print(json.dumps({k: v for k, v in row.items() if k != "examples"}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
