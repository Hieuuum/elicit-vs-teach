"""Live monitor for a training run — train + eval logs in one view.

Reads ``$GEODE_STORE/runs/<run-id>/<stage>/{train_log,eval_log}.jsonl``
every ``--interval`` seconds (default 30) and prints one compact status
line per tick, plus a fuller line whenever a new eval lands. Stage
(pretrain/sft) is auto-detected.

The convergence-rule state shown (eps-gated best, patience, grace) is
computed by replaying the eval log through the same ``ConvergenceTracker``
the trainer uses, configured from the run manifest — what it prints is
what the run will do, not a reimplementation.

Read-only; never touches the GPU; exits on its own when the manifest
leaves "running". ``--once`` prints a single snapshot and exits.

    python monitor.py --run-id evt-run1-base-v3-ext
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from geode.train.stopping import ConvergenceTracker, StoppingRule

REPO_ROOT = Path(__file__).resolve().parents[3]

RATE_WINDOW = 200  # train rows used for the steps/s estimate
TAIL_BYTES = 256 * 1024  # plenty for RATE_WINDOW rows (~130 B each)


def _parse(lines: list[bytes]) -> list[dict]:
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn line (writer mid-flush) — run-1 v2 lesson
    return rows


def read_tail(path: Path, max_bytes: int = TAIL_BYTES) -> list[dict]:
    """Parse the last ~max_bytes of a jsonl file, skipping torn lines."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return []
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # drop the partial first line
        return _parse(f.readlines())


def read_all(path: Path) -> list[dict]:
    try:
        with open(path, "rb") as f:
            return _parse(f.readlines())
    except FileNotFoundError:
        return []


def fmt_eta(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def eval_line(
    evals: list[dict], i: int, tracker_states: list[dict], rule: StoppingRule | None
) -> str:
    """Render eval #i using the tracker state captured just after it."""
    e, st = evals[i], tracker_states[i]
    line = f"[{stamp()}] eval @ {e['step']}: val {e['val_loss_nats']:.5f}"
    if i > 0:
        delta = (e["val_loss_nats"] - evals[i - 1]["val_loss_nats"]) * 1000
        line += f"  Δ {delta:+.1f} mnat"
    line += f" | min {st['min']:.5f}"
    if rule is not None:
        if e["step"] < rule.min_steps:
            line += f" | grace (rule starts @ {rule.min_steps})"
        else:
            line += f" | gated best {st['best']:.5f} | patience {st['stale']}/{rule.k}"
        if st["stopped"]:
            line += "  << RULE FIRED"
    return line


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--interval", type=float, default=30.0, help="seconds between refreshes")
    p.add_argument("--once", action="store_true", help="print one snapshot and exit")
    p.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
        help="artifact store root (default: $GEODE_STORE, else <repo>/geode-store)",
    )
    args = p.parse_args()

    run_dir = args.store / "runs" / args.run_id
    manifest_path = run_dir / "manifest.json"
    header_done = False
    last_eval_step = None  # print the latest eval once on startup, then only new ones
    last_train_step = None

    while True:
        try:
            man = json.loads(manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):  # mid-write rewrite window
            print(f"[{stamp()}] waiting for {manifest_path} ...", flush=True)
            if args.once:
                return 1
            time.sleep(args.interval)
            continue

        t = man["training"]
        s = t.get("stopping") or {}
        rule = None
        if s.get("eps_nats") is not None and s.get("k") is not None:
            rule = StoppingRule(s["eps_nats"], s["k"], s.get("min_steps") or 0)
        max_steps, eval_every = t.get("max_steps"), t.get("eval_every")

        stage_dirs = [d for d in run_dir.iterdir() if (d / "train_log.jsonl").exists()]
        if not header_done:
            desc = (
                f"eps {rule.eps_nats * 1000:g} mnat / k {rule.k} / min_steps {rule.min_steps}"
                if rule
                else "none"
            )
            print(
                f"[{stamp()}] {args.run_id} | status {man['status']} | rule {desc}"
                f" | eval_every {eval_every} | ceiling {max_steps}",
                flush=True,
            )
            header_done = True
        if not stage_dirs:
            print(f"[{stamp()}] waiting for first train step ...", flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
            continue
        stage = stage_dirs[0]

        train = read_tail(stage / "train_log.jsonl")
        evals = read_all(stage / "eval_log.jsonl")

        # Replay the trainer's own tracker over the full eval history.
        states = []
        if rule is not None:
            tracker = ConvergenceTracker(rule)
            for e in evals:
                stopped = tracker.update(e["val_loss_nats"], step=e["step"])
                states.append(
                    {
                        "min": tracker.min_nats,
                        "best": tracker.best_nats,
                        "stale": tracker.stale_evals,
                        "stopped": stopped,
                    }
                )
        else:
            running_min = float("inf")
            for e in evals:
                running_min = min(running_min, e["val_loss_nats"])
                states.append({"min": running_min, "best": None, "stale": 0, "stopped": False})

        if evals:
            if last_eval_step is None:
                new_from = len(evals) - 1  # startup: show the latest eval once
            else:
                new_from = next(
                    (i for i, e in enumerate(evals) if e["step"] > last_eval_step), len(evals)
                )
            for i in range(new_from, len(evals)):
                print(eval_line(evals, i, states, rule), flush=True)
            last_eval_step = evals[-1]["step"]

        if train:
            last = train[-1]
            window = train[-RATE_WINDOW:]
            line = f"[{stamp()}] step {last['step']}"
            if max_steps:
                line += f"/{max_steps}"
            line += f" | train {last['train_loss_nats']:.4f}"
            if len(window) >= 20:
                avg = sum(r["train_loss_nats"] for r in window) / len(window)
                line += f" (avg{len(window)} {avg:.4f})"
            dt = last["time_unix"] - window[0]["time_unix"]
            if dt > 0 and last["step"] > window[0]["step"]:
                rate = (last["step"] - window[0]["step"]) / dt
                line += f" | {rate:.2f} steps/s"
                if max_steps:
                    line += f" | ceiling ETA {fmt_eta((max_steps - last['step']) / rate)}"
            if last["step"] != last_train_step:
                last_train_step = last["step"]
            else:
                line += "  (no new steps)"
            print(line, flush=True)
        else:
            print(f"[{stamp()}] waiting for first train step ...", flush=True)

        if man["status"] != "running":
            res = next((v for k, v in man["experiment"].items() if k.endswith("_result")), None)
            if res:
                print(
                    f"[{stamp()}] run finalized: status {man['status']} | "
                    f"stop_reason {res.get('stop_reason')} | "
                    f"min_val_nats {res.get('min_val_nats')} | "
                    f"final_step {res.get('final_step')}",
                    flush=True,
                )
            else:
                print(f"[{stamp()}] run finalized: status {man['status']}", flush=True)
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
