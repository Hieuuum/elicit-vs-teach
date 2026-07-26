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

Rule 11 (owner 2026-07-26, added mid-sweep) selects the SHARED pin: prefer an
LR that is acceptable to both arms, and only fall back to rule 8's arm-B
optimum if no such LR exists. "Acceptable" is a TWO-COORDINATE band — floor
AND steps-to-convergence — because EDL is the area between the curve and its
floor, so an LR that reaches the same floor in twice the steps costs real EDL
while looking free on the floor alone. Both coordinates get their noise handle
from the same seed twin.
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
        "eval_cadence": _cadence(stop_recs),
    }


def _cadence(stop_recs: list[dict]) -> int:
    """The eps/k eval interval, read off the log rather than the config.

    Convergence can only be observed on an eval tick, so two runs stopping one
    tick apart are indistinguishable in steps. This sets the floor on rule 11's
    steps band.
    """
    gaps = [b["step"] - a["step"] for a, b in zip(stop_recs, stop_recs[1:])]
    return min(gaps) if gaps else 0


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


def representative_per_lr(points) -> dict[float, dict]:
    """One run per LR: the lowest-scoring seed, kept whole (rule 5's convention).

    Whole, rather than a per-coordinate min across seeds, so that the reported
    ``final_step`` is the one that actually produced the reported ``score``.
    """
    rep: dict[float, dict] = {}
    for p in points:
        if p["lr"] not in rep or p["score"] < rep[p["lr"]]["score"]:
            rep[p["lr"]] = p
    return rep


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

    arms: dict[str, dict] = {}
    stream_conflict = False
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
        if any(t["excluded"] or t.get("plateau") for t in twins):
            # Both the noise handle and rule 5's "incumbent stands" default are
            # read off the twins. If rules 2/3 disqualified them, every number
            # downstream would be measured on runs already thrown out.
            print(
                f"  ! arm {arm}: the incumbent {INCUMBENT_LR:.0e} is ITSELF disqualified "
                "(excluded or plateau) — the noise handle would be read off runs rules 2/3 "
                "just threw out. Rules 5 and 11 cannot be applied to this arm; escalate."
            )
            continue
        noise = abs(twins[0]["score"] - twins[1]["score"])
        # Floored at one eval tick: the twins routinely stop at the identical
        # step, and a zero-width band would call an unresolvable difference
        # real. Widening the band keeps the incumbent more often, which is the
        # mildly elicit-flattering direction — bounded here at exactly one tick.
        cadence = max(p["eval_cadence"] for p in arm_points)
        noise_steps = max(abs(twins[0]["final_step"] - twins[1]["final_step"]), cadence)
        incumbent = min(t["score"] for t in twins)
        print(
            f"  noise floor (rule 5): |{twins[0]['score']:.5f} - {twins[1]['score']:.5f}| = "
            f"{noise:.5f} nats between seeds {twins[0]['seed']}/{twins[1]['seed']} at the "
            f"incumbent LR"
        )
        print(
            f"  noise steps (rule 11): |{twins[0]['final_step']} - {twins[1]['final_step']}| "
            f"floored at the {cadence}-step eval tick = {noise_steps} steps"
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
            verdict = INCUMBENT_LR
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
            verdict = best["lr"]

        # Rule 1 guard: would the other scoring stream flip this arm's ranking?
        by_all = [p["run_id"] for p in sorted(eligible, key=lambda p: p["score"])]
        by_stop = [p["run_id"] for p in sorted(eligible, key=lambda p: p["score_stopping_only"])]
        if by_all != by_stop:
            stream_conflict = True
            print(
                "  ! rule 1: the all-evals and stopping-evals-only streams RANK THIS ARM "
                "DIFFERENTLY — the verdict would be an artifact of which stream was scored. "
                "Report both and do not pin."
            )

        # Rule 11: which LRs this arm would accept. One representative run per
        # LR (the lower-scoring seed, rule 5's convention, kept as a REAL run so
        # its final_step belongs to the same run as its score). An LR is
        # acceptable if it is within the seed twin's spread of this arm's best
        # on BOTH coordinates — floor and steps-to-convergence.
        rep = representative_per_lr(
            q for q in arm_points if not q["excluded"] and not q.get("plateau")
        )
        best_score = min(r["score"] for r in rep.values())
        best_steps = min(r["final_step"] for r in rep.values())
        acceptable = sorted(
            lr
            for lr, r in rep.items()
            if r["score"] <= best_score + noise and r["final_step"] <= best_steps + noise_steps
        )
        print(
            f"  accepts (rule 11): {', '.join(f'{x:.1e}' for x in acceptable) or 'nothing'} "
            f"— within {noise:.5f} nats of {best_score:.5f} AND {noise_steps} steps of {best_steps}"
        )
        arms[arm] = {
            "verdict": verdict,
            "rep": rep,
            "acceptable": acceptable,
            "best_score": best_score,
            "best_steps": best_steps,
        }

    print("\n" + "=" * 66)
    if set(arms) != {"A", "B"}:
        print(f"the shared pin needs both arms; resolved: {sorted(arms) or 'none'} — do not pin.")
        return 1
    if stream_conflict:
        print("rule 1: an arm's two scoring streams rank it differently — DO NOT PIN (above).")
        return 1

    print(
        "per-arm verdicts (rule 5): "
        + ", ".join(f"{a}->{d['verdict']:.1e}" for a, d in sorted(arms.items()))
    )
    shared = [lr for lr in arms["A"]["acceptable"] if lr in arms["B"]["acceptable"]]

    if shared:
        # Rule 11, first clause. Inside the band the arms cannot tell these LRs
        # apart, so the choice among them is made by pre-registered DIRECTION,
        # not by the numbers: take the one arm B scores lowest (owner, twice).
        lr = min(shared, key=lambda x: arms["B"]["rep"][x]["score"])
        # A pick made by tie-break claims NO superiority, so rule 6's question
        # ("is something better hiding past this edge?") does not arise — see
        # the rule-6 call below.
        by_tiebreak = len(shared) > 1 and INCUMBENT_LR in shared and lr != INCUMBENT_LR
        print(
            f"RULE 11 — {', '.join(f'{x:.1e}' for x in shared)} "
            f"{'is' if len(shared) == 1 else 'are all'} acceptable to BOTH arms "
            f"(within each arm's own seed-twin spread of its best, on floor AND steps)."
        )
        if len(shared) == 1:
            print(f"=> pin {lr:.1e}: the only LR that serves both arms.")
        else:
            print(
                f"=> pin {lr:.1e}: among LRs the arms cannot distinguish, the tie breaks toward "
                "TEACH\n   (owner's second clause), so no residual is left pointing the elicit "
                "hypothesis' way."
            )
    else:
        # "Arm B's optimum" means optimum on the SAME two coordinates rule 11
        # judges on. Reading it off rule 5's floor-only verdict would let an LR
        # that rule 11 just rejected for being slow in arm B come back as the
        # pin — inflating EDL_B, raising the ratio, flattering elicit.
        if not arms["B"]["acceptable"]:
            print("rule 8 needs arm B's optimum and arm B accepts nothing — escalate, do not pin.")
            return 1
        lr = min(arms["B"]["acceptable"], key=lambda x: arms["B"]["rep"][x]["score"])
        by_tiebreak = False  # rule 8's pick asserts arm B's optimum; rule 6 applies
        if lr != arms["B"]["verdict"]:
            print(
                f"note: arm B's rule-5 verdict was {arms['B']['verdict']:.1e} on the floor alone; "
                f"on floor AND steps its optimum is {lr:.1e}."
            )
        print(
            "RULE 11 finds NO LR acceptable to both arms — falling through to rule 8.\n"
            f"Rule 8 (owner 2026-07-26, fixed before any point scored): take ARM B's optimum\n"
            f"=> pin {lr:.1e} for BOTH arms. A better LR lowers that arm's EDL (it shrinks the\n"
            "excess area above the asymptote in EDL = MDL - N*L_test), so under a shared LR\n"
            "this runs teach at its best and elicit off its best — the teach/elicit ratio\n"
            "falls on both counts. CONSERVATIVE for elicit, and it reverses the 2026-07-24\n"
            "precedent deliberately (see rule 8)."
        )

    _report_cost(arms, lr)
    print(
        "REPORT WITH IT: an elicit win survives this choice; a teach win or a tie is\n"
        "confounded by it, and by the 3.12-nat state gap, which cuts the same way."
    )
    if lr == INCUMBENT_LR:
        print(
            "The 1e-3 pin is re-validated on this phase's two parents. Launch the production\n"
            "targets unchanged (rule 7: no sweep number is reported)."
        )
    else:
        grid = sorted(arms["A"]["rep"] | arms["B"]["rep"])
        if lr in (grid[0], grid[-1]):
            if by_tiebreak:
                # Rule 6 exists to stop us pinning an edge we CLAIM is better
                # without knowing what lies past it. A within-band tie-break
                # claims no such thing — the band says this LR and the
                # incumbent are indistinguishable — so "is it a peak?" is moot
                # and no grid extension is owed.
                print(
                    f"note: {lr:.1e} is a grid edge, but it was chosen by tie-break, not by "
                    "beating\n  anything — rule 6 does NOT fire (it guards claims of "
                    "superiority, and none is made)."
                )
            else:
                print(
                    f"! rule 6: {lr:.1e} is a GRID EDGE — extend the grid "
                    f"({'add 1e-4' if lr == grid[0] else 'add 1e-2'}) before pinning."
                )
        print(
            f"Re-run BOTH production targets at {lr:.1e} with the production seed 316 under the\n"
            "existing run_ids, and scope the new pin in lr_pin.yaml rather than overwriting the\n"
            "shared `lr:` key that runs 7/8 executed under (rules 7, 10)."
        )
    return 0


def _report_cost(arms: dict[str, dict], lr: float) -> None:
    """What the shared pin costs EACH arm, on both coordinates.

    Two-sided on purpose. Arm A off its best inflates the ratio's numerator —
    the 2026-07-24 worry. Arm B off its best inflates the denominator, which
    RAISES the ratio and flatters elicit — the direction rule 8 exists to
    prevent, and which a floor-only, arm-A-only check would miss entirely.
    """
    for arm, d in sorted(arms.items()):
        r = d["rep"].get(lr)
        if r is None:
            print(f"! arm {arm} has no eligible point at {lr:.1e} — the pin is unmeasured there.")
            continue
        floor_x = r["score"] / d["best_score"]
        steps_x = r["final_step"] / d["best_steps"]
        who = "numerator" if arm == "A" else "denominator"
        print(
            f"  cost to arm {arm} ({who}): floor {floor_x:.2f}x its own best "
            f"({r['score']:.5f} vs {d['best_score']:.5f}), steps {steps_x:.2f}x "
            f"({r['final_step']} vs {d['best_steps']})"
        )
        for coord, x in (("floor", floor_x), ("steps", steps_x)):
            if x > 3.0:
                print(
                    f"  ! arm {arm}'s {coord} inflates {x:.1f}x — past the ~3x the 2026-07-24\n"
                    f"    precedent refused. The rule says pin anyway; say this in the write-up\n"
                    f"    rather than quietly re-deciding."
                )


if __name__ == "__main__":
    raise SystemExit(main())
