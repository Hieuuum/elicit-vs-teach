"""Running EDL per label token vs. dataset size, for a set of runs.

Owner request 2026-07-25. At every step where a validation eval exists,
treat the data seen so far as the dataset and the current parameters as
final, and evaluate the paper's EDL (Donoway et al. Eq. 3) on that prefix:

    MDL(e) = sum_{s<e} loss_sum_nats[s]        (prequential.jsonl, epoch 1)
    D(e)   = sum_{s<e} label_token_count[s]
    EDL(e) = MDL(e) - D(e) * val_loss_nats(e)  (eval_log.jsonl)
    y(e)   = EDL(e) / D(e)                     -> bits at the axis

The alignment is exact, not approximate: a prequential record at step s is
the pre-update loss under theta_s (geode/edl/loop.py), and step_callback
fires after update e, so the eval at step e is theta_e -- the parameters
that have just finished encoding the first e batches and none of the rest.

Three caveats the figure carries in its caption, none of them fixable here:

1. The subtrahend is the STOPPING-BLOCK val loss (first EVAL_STOP_ROWS=2048
   rows of the frozen eval file), not eval/test_loss.json's 97,952-example
   test loss. The two blocks are disjoint (train_target.py:291-292) so
   nothing leaks, but they are different data and the floors differ (run 8:
   0.0237 val vs 0.0312 test). The star markers put each run's canonical
   test-floored endpoint on the same axes so the gap is visible.
2. Runs end at different n (run 7 at 768K, the sweep at 896K, run 8's
   epoch-1 cut at 1M). y is a rate that still varies with n, so endpoint-to-
   endpoint comparison mixes rate with dataset size: the dashed vertical
   marks matched n and the printed table quotes every run there.
3. Per-label-token is not tokenizer-fair across arms -- summed -log p is the
   codelength of the label string, and the token count is an arbitrary
   segmentation of it (runs 7/8 ~4.9 tokens/label under the arith tokenizer,
   the Llama sweep ~2.9). The printed table also quotes bits/example, which
   is comparable.

Epoch-2 records are excluded (paper footnote 1: re-encoding the same labels
is no longer a minimum description length). Only run 8 reaches epoch 2.

Laptop-side, CPU-only; pull logs first (from analysis/):

    python3 ../scripts/hf_checkpoint.py pull --run-id <rid> --no-weights
    python3 plot_edl_per_token.py [--run-id <rid> ...] [--out figures/edl_per_token.png]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = (
    "evt-run7-armA-target-1m",
    "evt-run8-armB-target-1m",
    "evt-run10-sweep-lr3e-4",
)
LN2 = math.log(2.0)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def running_edl(run_dir: Path) -> dict[str, np.ndarray]:
    """Prefix EDL at every eval step, from prequential.jsonl + eval_log.jsonl.

    Returns arrays indexed by eval step e: examples/tokens seen, running
    MDL, val loss, and EDL per label token and per example (all nats).
    """
    preq = sorted(read_jsonl(run_dir / "logs" / "prequential.jsonl"), key=lambda r: r["step"])
    epoch1 = [r for r in preq if r["epoch"] == 1]
    # Index i holds the totals AFTER encoding batches 0..i, so a prefix of e
    # batches reads at index e-1.
    cum_loss = np.cumsum([r["loss_sum_nats"] for r in epoch1])
    cum_tok = np.cumsum([r["label_token_count"] for r in epoch1])
    cum_ex = np.cumsum([len(r["example_ids"]) for r in epoch1])

    evals = sorted(read_jsonl(run_dir / "eval_log.jsonl"), key=lambda r: r["step"])
    steps, vals = [], []
    for row in evals:
        e = row["step"]
        if 1 <= e <= len(epoch1):  # evals past the epoch-1 cut have no MDL prefix
            steps.append(e)
            vals.append(row["val_loss_nats"])
    idx = np.array(steps) - 1
    val = np.array(vals)
    mdl, tok, ex = cum_loss[idx], cum_tok[idx], cum_ex[idx]
    edl = mdl - tok * val
    return {
        "step": np.array(steps),
        "examples": ex,
        "tokens": tok,
        "mdl_nats": mdl,
        "val_nats": val,
        "edl_nats": edl,
        "edl_per_token_nats": edl / tok,
        "edl_per_example_nats": edl / ex,
        "n_epoch1_steps": len(epoch1),
        # Full epoch-1 totals, NOT the last eval point: run 8's epoch-1 cut
        # (step 7813) falls between evals, and the canonical EDL is defined
        # over the whole first epoch (geode.edl.metrics._epoch1_totals).
        "epoch1_totals": (float(cum_loss[-1]), float(cum_tok[-1]), float(cum_ex[-1])),
    }


def canonical_endpoint(run_dir: Path, curve: dict[str, np.ndarray]) -> dict[str, float]:
    """EDL/D at the run's epoch-1 end, floored by eval/test_loss.json (the paper's Eq. 3)."""
    test = json.loads((run_dir / "eval" / "test_loss.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("masking_config_hash") != test.get("masking_config_hash"):
        raise SystemExit(f"{run_dir.name}: train/test masking hashes disagree (D-1 guard)")
    mdl, tok, ex = curve["epoch1_totals"]
    edl = mdl - tok * test["loss_per_label_token_nats"]
    return {
        "examples": float(ex),
        "l_test_nats": test["loss_per_label_token_nats"],
        "edl_per_token_nats": edl / tok,
        "edl_per_example_nats": edl / ex,
    }


def at_matched_n(curve: dict[str, np.ndarray], n: float) -> dict[str, float]:
    """The last eval point at or before n examples."""
    i = int(np.searchsorted(curve["examples"], n, side="right")) - 1
    return {
        k: float(curve[k][i]) for k in ("examples", "edl_per_token_nats", "edl_per_example_nats")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", action="append", dest="run_ids", help="repeatable")
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "edl_per_token.png",
    )
    ap.add_argument(
        "--min-examples",
        type=float,
        default=1000.0,
        help="left edge of the plot (owner 2026-07-25); display only — every point's "
        "MDL prefix still starts at step 0",
    )
    args = ap.parse_args()
    run_ids = args.run_ids or list(DEFAULT_RUNS)

    curves, ends = {}, {}
    for rid in run_ids:
        run_dir = args.store / "runs" / rid
        if not (run_dir / "logs" / "prequential.jsonl").is_file():
            print(f"[edl] {rid}: prequential.jsonl missing under {run_dir} — skipped")
            continue
        curves[rid] = running_edl(run_dir)
        ends[rid] = canonical_endpoint(run_dir, curves[rid])
    if not curves:
        raise SystemExit("plot_edl_per_token: no runs had logs — nothing to plot")

    matched_n = min(c["examples"][-1] for c in curves.values())

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (rid, c) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        shown = c["examples"] >= args.min_examples
        y = c["edl_per_token_nats"][shown] / LN2
        n_nonpos = int((y <= 0).sum())
        tok_per_ex = c["tokens"][-1] / c["examples"][-1]
        ax.plot(
            c["examples"][shown],
            y,
            color=color,
            marker="o",
            ms=3,
            lw=0.8,
            alpha=0.85,
            label=f"{rid}  ({tok_per_ex:.2f} label tokens/example)",
        )
        ax.plot(
            ends[rid]["examples"],
            ends[rid]["edl_per_token_nats"] / LN2,
            color=color,
            marker="*",
            ms=15,
            mec="k",
            mew=0.5,
            ls="none",
        )
        peak = int(np.argmax(y))
        print(
            f"[edl] {rid}: {len(y)} eval points shown of {len(c['step'])}, "
            f"{n_nonpos} non-positive, {tok_per_ex:.2f} label tokens/example, "
            f"epoch-1 steps {c['n_epoch1_steps']}, peak {y[peak]:.4f} bits at "
            f"n={c['examples'][shown][peak]:,.0f} (step {c['step'][shown][peak]})"
        )

    ax.axvline(matched_n, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.annotate(
        f"matched n = {matched_n:,.0f}",
        xy=(matched_n, ax.get_ylim()[1]),
        xytext=(-4, -10),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        alpha=0.7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("training examples seen (log scale)")
    ax.set_ylabel("EDL per label token (bits)")
    ax.set_title(
        "Running EDL/D: prefix MDL minus the step's val loss, per label token\n"
        "points = eval steps; stars = canonical endpoint floored by eval/test_loss.json"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[edl] wrote {args.out} ({len(curves)} runs)")

    print(f"\n[edl] at matched n = {matched_n:,.0f} examples (bits):")
    for rid, c in curves.items():
        m = at_matched_n(c, matched_n)
        print(
            f"  {rid:32s} n={m['examples']:>9,.0f}  "
            f"EDL/token {m['edl_per_token_nats'] / LN2:.5f}  "
            f"EDL/example {m['edl_per_example_nats'] / LN2:.5f}"
        )
    print("\n[edl] canonical endpoints (test-loss floor, own n, bits):")
    for rid, e in ends.items():
        print(
            f"  {rid:32s} n={e['examples']:>9,.0f}  "
            f"EDL/token {e['edl_per_token_nats'] / LN2:.5f}  "
            f"EDL/example {e['edl_per_example_nats'] / LN2:.5f}  "
            f"(L_test {e['l_test_nats']:.5f} nats/token)"
        )


if __name__ == "__main__":
    main()
