"""Apply the phase-2 LR sweep's pre-registered reading rules, mechanically.

The rules themselves live in ``configs/pilot/p2_sweep_armA_lr1e-3.yaml`` and
were fixed before the first point launched; this script only executes them, so
that the verdict is not an eyeball judgement made after seeing the numbers.

    python3 lr_sweep_read.py [--store DIR]

Scoring note (rule 1). ``eval_log.jsonl`` interleaves two streams: the eps/k
stopping evals at ``eval_every`` cadence (``stopping_eval: true``) and the
denser log-spaced curve evals. The project convention is the min over ALL
records (memory: quote eval-log minima, run 7 0.0027 / run 8 0.0237, not the
manifest's ``target_result.min_val_nats``, which is the tracker's view of the
stopping stream alone and reads 0.0033 for that same run). A longer run gets
more curve-eval draws, so this script reports the stopping-eval-only min
alongside and REFUSES to declare a winner if the two streams disagree on the
per-arm ranking — that would make the verdict an artifact of which stream was
scored.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

K_FINAL = 5  # rule 4: the stopping rule's own k
PLATEAU_FACTOR = 2.0  # rule 2
INCUMBENT_LR = 1.0e-3


def load_point(run_dir: Path) -> dict | None:
    manifest = run_dir / "manifest.json"
    evals = run_dir / "eval_log.jsonl"
    if not (manifest.is_file() and evals.is_file()):
        return None
    m = json.loads(manifest.read_text())
    if m.get("status") != "complete":
        return None
    recs = [json.loads(x) for x in evals.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if "val_loss_nats" in r]
    stop_recs = [r for r in recs if r.get("stopping_eval")]
    if not recs:
        return None
    res = m["experiment"]["target_result"]
    best = min(recs, key=lambda r: r["val_loss_nats"])
    # Rule 4: how much the curve still fell over the final k stopping evals.
    # This is what distinguishes a genuine floor (rule 2) from a run that was
    # still descending when the ceiling cut it off (rule 3).
    tail = stop_recs[-(K_FINAL + 1) :]
    descent = (
        tail[0]["val_loss_nats"] - tail[-1]["val_loss_nats"] if len(tail) > 1 else float("nan")
    )
    return {
        "run_id": m["run_id"],
        "arm": m["experiment"]["arm"],
        "lr": float(m["training"]["optimizer"]["lr"]),
        "seed": int(m["training"]["seed"]),
        "score": best["val_loss_nats"],  # rule 1, project convention
        "score_step": best["step"],
        "score_stopping_only": (
            min(r["val_loss_nats"] for r in stop_recs) if stop_recs else float("nan")
        ),
        "stop_reason": res["stop_reason"],
        "final_step": res["final_step"],
        "n_evals": len(recs),
        "final_k_descent": descent,
    }


def rank_arm(points: list[dict]) -> tuple[list[dict], list[str]]:
    """Apply rules 2-6 to one arm. Returns (points, notes) with flags set."""
    notes: list[str] = []
    # Rule 3: a ceiling exit did not converge; it is EXCLUDED, not ranked last.
    for p in points:
        p["excluded"] = p["stop_reason"] == "max_steps"
        if p["excluded"]:
            notes.append(
                f"{p['run_id']}: stop_reason=max_steps at {p['final_step']} — did not converge "
                f"within the ceiling; EXCLUDED from the ranking (rule 3). Final-{K_FINAL} descent "
                f"{p['final_k_descent']:.5f} nats "
                f"({'still descending' if p['final_k_descent'] > 0.002 else 'flat'})."
            )
    live = [p for p in points if not p["excluded"]]
    if not live:
        return points, notes + ["no point in this arm converged — nothing can win (rule 3)"]

    # Rule 2: plateau, not floor. eps/k fires on any flat stretch.
    arm_best = min(p["score"] for p in live)
    for p in live:
        p["plateau"] = p["score"] > PLATEAU_FACTOR * arm_best
        if p["plateau"]:
            notes.append(
                f"{p['run_id']}: converged at {p['score']:.5f} > {PLATEAU_FACTOR}x the arm's best "
                f"{arm_best:.5f} — recorded as a PLATEAU, cannot win (rule 2)."
            )
    return points, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=None)
    args = ap.parse_args()
    store = args.store or Path(os.environ.get("GEODE_STORE", "geode-store"))

    points = [
        p
        for d in sorted((store / "runs").glob("evt-p2-sweep-arm*"))
        if (p := load_point(d)) is not None
    ]
    if not points:
        print(f"no completed sweep points under {store}/runs/evt-p2-sweep-arm*")
        return 1

    verdicts = {}
    for arm in sorted({p["arm"] for p in points}):
        arm_points = sorted(
            (p for p in points if p["arm"] == arm), key=lambda p: (p["lr"], p["seed"])
        )
        arm_points, notes = rank_arm(arm_points)

        print(f"\n=== ARM {arm} " + "=" * 58)
        print(
            f"{'lr':>8} {'seed':>5} {'min_val':>10} {'@step':>7} {'stop-only':>10} "
            f"{'stop_reason':>12} {'final_step':>10} {'fin-k drop':>11}"
        )
        for p in arm_points:
            flag = " EXCLUDED" if p["excluded"] else (" PLATEAU" if p.get("plateau") else "")
            print(
                f"{p['lr']:>8.1e} {p['seed']:>5} {p['score']:>10.5f} {p['score_step']:>7} "
                f"{p['score_stopping_only']:>10.5f} {p['stop_reason']:>12} "
                f"{p['final_step']:>10} {p['final_k_descent']:>11.5f}{flag}"
            )
        for n in notes:
            print(f"  ! {n}")

        # Rule 5: the noise handle — the seed twin at the incumbent LR.
        twins = [p for p in arm_points if p["lr"] == INCUMBENT_LR]
        if len(twins) != 2:
            print(
                f"  ! arm {arm}: expected 2 seeds at {INCUMBENT_LR:.0e}, found {len(twins)} — "
                "the noise floor is unmeasured and rule 5 cannot be applied"
            )
            continue
        noise = abs(twins[0]["score"] - twins[1]["score"])
        incumbent = min(t["score"] for t in twins)
        print(
            f"  noise floor (rule 5): |{twins[0]['score']:.5f} - {twins[1]['score']:.5f}| = "
            f"{noise:.5f} nats between seeds {twins[0]['seed']}/{twins[1]['seed']} at the "
            f"incumbent LR"
        )

        eligible = [
            p
            for p in arm_points
            if not p["excluded"] and not p.get("plateau") and p["lr"] != INCUMBENT_LR
        ]
        best = min(eligible, key=lambda p: p["score"], default=None)
        if best is None or best["score"] >= incumbent - noise:
            if best is None:
                why = "no challenger converged off a plateau"
            elif best["score"] >= incumbent:
                why = f"no challenger beat it ({best['lr']:.1e} is the closest, {best['score'] - incumbent:+.5f})"
            else:
                why = (
                    f"the best challenger {best['lr']:.1e} leads by only "
                    f"{incumbent - best['score']:.5f}, inside the {noise:.5f} noise floor"
                )
            print(f"  VERDICT arm {arm}: incumbent {INCUMBENT_LR:.0e} STANDS — {why} (rule 5).")
            verdicts[arm] = INCUMBENT_LR
        else:
            print(
                f"  VERDICT arm {arm}: {best['lr']:.1e} beats the incumbent by "
                f"{incumbent - best['score']:.5f} > {noise:.5f} noise (rule 5)."
            )
            lrs = sorted({p["lr"] for p in arm_points})
            if best["lr"] in (lrs[0], lrs[-1]):
                print(
                    f"  ! rule 6: {best['lr']:.1e} is a GRID EDGE — extend the grid "
                    f"({'add 1e-4' if best['lr'] == lrs[0] else 'add 1e-2'}) before pinning."
                )
            verdicts[arm] = best["lr"]

        # Rule 1 guard: would the other scoring stream flip this arm's ranking?
        by_all = [p["run_id"] for p in sorted(eligible, key=lambda p: p["score"])]
        by_stop = [p["run_id"] for p in sorted(eligible, key=lambda p: p["score_stopping_only"])]
        if by_all != by_stop:
            print(
                "  ! rule 1: the all-evals and stopping-evals-only streams RANK THIS ARM "
                "DIFFERENTLY — the verdict would be an artifact of which stream was scored. "
                "Report both and do not pin."
            )

    print("\n" + "=" * 66)
    if len(set(verdicts.values())) == 1 and verdicts:
        lr = next(iter(verdicts.values()))
        print(f"ARMS AGREE on {lr:.1e}.")
        if lr == INCUMBENT_LR:
            print(
                "The 1e-3 pin is re-validated on this phase's two parents. Launch the "
                "production targets unchanged (rule 7: no sweep number is reported)."
            )
        else:
            print(
                f"Re-run BOTH production targets at {lr:.1e} with the production seed 316 under "
                "the existing run_ids, and scope the new pin in lr_pin.yaml rather than "
                "overwriting the shared `lr:` key that runs 7/8 executed under (rules 7, 10)."
            )
    else:
        print(f"ARMS DISAGREE: {', '.join(f'{a}->{lr:.1e}' for a, lr in sorted(verdicts.items()))}")
        print(
            "Rule 8: DO NOT auto-resolve. A better LR lowers that arm's EDL (it shrinks the\n"
            "excess area above the asymptote in EDL = MDL - N*L_test). So adopting arm A's\n"
            "argmin RAISES the teach/elicit ratio and FLATTERS ELICIT; adopting arm B's\n"
            "LOWERS the ratio and is CONSERVATIVE for elicit. Report both arms at both LRs\n"
            "and escalate to the owner — this script does not pick."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
