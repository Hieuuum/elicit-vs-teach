"""Dose-response summary for the new phase's elicit arm (2026-07-26).

Two jobs, both required by the runbook before any EDL number is quoted:

1. **The comparability check.** ε/k was calibrated at the two ENDS of the grid
   (n=1 and n=16) and the middle doses inherit it. If the rule fires at a
   materially different fraction of descent for some dose, the dose-response
   curve is partly a measurement of its own stopping rule — the
   endpoint-referenced-comparability failure (2026-07-25). %descent is taken
   against a fixed floor of 0, not min(observed): the cross-entropy floor of a
   memorisable finite set IS 0, and a fixed reference keeps the doses
   comparable instead of making each number depend on where its run stopped.

2. **The curve itself**, with the parent as the n=0 intercept. Without that
   row "the dose moved zero-shot accuracy" is an assumption; the parent is
   scored on the identical eval file and protocol via
   ``gates.py g5 --run <parent> --no-record``, and its numbers are passed in
   with --baseline (they are deliberately not written to the shared parent's
   manifest, so they cannot be read back from it).

Usage (CPU, seconds):
    python3 dose_curve.py [--store DIR] [--baseline zero,sixteen,test_loss]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DOSES = (1, 2, 4, 8, 16)
# A dose whose stop sits this far (percentage points of descent) from the
# median is called out. The pinned rule agreed to 0.01pp across the two ends,
# so anything approaching a tenth of a point is a real outlier, not noise.
FLAG_PP = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=None, help="default: $GEODE_STORE")
    parser.add_argument(
        "--baseline",
        default=None,
        help="parent's n=0 row as 'zero_shot,sixteen_shot,test_loss_nats' "
        "(from gates.py g5 --no-record on the dose parent)",
    )
    args = parser.parse_args()
    store = args.store or Path(os.environ.get("GEODE_STORE", "geode-store"))

    rows = []
    for n in DOSES:
        p = store / "runs" / f"evt-p2-armA-dose{n}" / "manifest.json"
        if not p.is_file():
            print(f"[curve] dose {n}: no manifest — skipped")
            continue
        m = json.loads(p.read_text())
        exp = m["experiment"]
        res = exp["sft_result"]
        g = exp.get("gates", {})
        # min_val_nats is NOT a val loss for a dose run: the trainer reuses the
        # field for whatever metric drove the stop, and these runs carve no val
        # split at all. training.stopping.metric == "train_loss" disambiguates.
        assert m["training"]["stopping"]["metric"] == "train_loss", f"dose {n} is not a dose run"
        loss0 = exp["step0"]["dose_loss_nats"]
        stop = res["min_val_nats"]
        rows.append(
            {
                "n": n,
                "device": exp.get("device", "?"),
                "commit": m["git_commit"][:7],
                "steps": res["final_step"],
                "reason": res["stop_reason"],
                "L0": loss0,
                "Lstop": stop,
                "pct": 100.0 * (loss0 - stop) / loss0,
                "g4": g.get("G4", {}).get("format_validity"),
                "g2": g.get("G2", {}).get("accuracy"),
                "zero": g.get("G5", {}).get("zero_shot_accuracy"),
                "sixteen": g.get("G5", {}).get("sixteen_shot_accuracy"),
                "tl": g.get("G5", {}).get("test_loss_nats"),
            }
        )

    def fmt(v: float | None, spec: str) -> str:
        return format(v, spec) if v is not None else "—"

    print(f"\n{'dose':>4} {'steps':>6} {'stop':>10} {'L0':>8} {'L_stop':>9} {'%descent':>9}")
    for r in rows:
        print(
            f"{r['n']:>4} {r['steps']:>6} {r['reason']:>10} "
            f"{r['L0']:>8.4f} {r['Lstop']:>9.6f} {r['pct']:>8.2f}%"
        )

    if rows:
        pcts = sorted(r["pct"] for r in rows)
        med = pcts[len(pcts) // 2]
        outliers = [r for r in rows if abs(r["pct"] - med) > FLAG_PP]
        spread = pcts[-1] - pcts[0]
        print(f"\n[curve] %descent spread {spread:.3f}pp (median {med:.2f}%)")
        if outliers:
            print(
                "[curve] OUTLIER(S): "
                + ", ".join(f"dose {r['n']} at {r['pct']:.2f}%" for r in outliers)
                + "\n[curve] That is a finding ABOUT THE RULE and belongs in decisions.md "
                "before any\n        EDL number is quoted — not a reason to re-tune eps "
                "after the fact."
            )
        else:
            print("[curve] the rule treated every dose alike — the curve measures the dose")
        if any(r["reason"] != "converged" for r in rows):
            print("[curve] WARNING: a dose did not converge — stop_reason is a bug signal here")
        if len({r["device"] for r in rows}) > 1:
            print(f"[curve] WARNING: mixed devices {sorted({r['device'] for r in rows})}")

    print(f"\n{'dose':>4} {'G4':>7} {'G2':>7} {'0-shot':>7} {'16-shot':>8} {'test_nats':>10}")
    if args.baseline:
        z, s, t = (float(x) for x in args.baseline.split(","))
        print(f"{'0*':>4} {'1.0000':>7} {'—':>7} {z:>7.4f} {s:>8.4f} {t:>10.4f}")
    for r in rows:
        print(
            f"{r['n']:>4} {fmt(r['g4'], '>7.4f')} {fmt(r['g2'], '>7.4f')} "
            f"{fmt(r['zero'], '>7.4f')} {fmt(r['sixteen'], '>8.4f')} {fmt(r['tl'], '>10.4f')}"
        )
    if args.baseline:
        print("\n* n=0 is the parent (evt-run2-armA-algo), scored --no-record on the same")
        print("  eval file and protocol. G4 1.0000 there is why G4 can only detect DAMAGE")
        print("  for this arm: it is saturated before the first dose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
