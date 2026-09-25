"""Audit completed artifacts and write the full report without using the GPU."""

# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from geode.circuits.artifacts import write_json
from geode.circuits.sanity import audit_run
from geode.circuits.report import generate_report


def finalize(root: Path) -> None:
    status_path = root.parent / "finalization_status.json"
    write_json(
        status_path, {"status": "running", "phase": "technical_audit", "started_epoch": time.time()}
    )
    try:
        audit = audit_run(root)
        write_json(root / "technical_audit.json", audit)
        if audit["status"] != "passed" or audit["warnings"]:
            raise RuntimeError(
                "Full evaluation technical audit failed or has unresolved warnings: "
                + str(audit["failures"] + audit["warnings"])
            )
        summaries = []
        with (root / "format_failure_examples.jsonl").open("x") as failures:
            for stage in audit["expected_stages"]:
                by_task = {}
                with (root / stage / "behavior.jsonl").open() as stream:
                    for line in stream:
                        r = json.loads(line)
                        counts = by_task.setdefault(r["task"], Counter())
                        counts["n_rows"] += 1
                        for flag in [
                            "correct",
                            "parse_failure",
                            "execution_failure",
                            "timed_out",
                            "truncated",
                        ]:
                            counts[flag] += bool(r.get(flag))
                        if r.get("parse_failure") or r.get("execution_failure"):
                            failures.write(
                                json.dumps(
                                    {
                                        "stage": stage,
                                        **{
                                            k: r.get(k)
                                            for k in [
                                                "id",
                                                "task",
                                                "prompt",
                                                "generation",
                                                "prediction",
                                                "answer",
                                                "parse_failure",
                                                "execution_failure",
                                                "truncated",
                                            ]
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                summaries.extend(
                    {"stage": stage, "task": task, **counts} for task, counts in by_task.items()
                )
        write_json(
            root / "format_failure_summary.json",
            {
                "rows": summaries,
                "protocol": "Native scores unchanged; parsing and execution failures reported separately.",
            },
        )
        write_json(
            status_path, {"status": "running", "phase": "report", "updated_epoch": time.time()}
        )
        generate_report(root, n_bootstrap=2000, seed=0)
        write_json(
            status_path,
            {
                "status": "complete",
                "phase": "ready_for_verified_backup",
                "finished_epoch": time.time(),
            },
        )
    except BaseException as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_epoch": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    finalize(parser.parse_args().directory.resolve())
