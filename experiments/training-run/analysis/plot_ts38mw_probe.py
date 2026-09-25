"""ts38mw Stage 1 probe: EM and loss per held-out pin vs. install step.

Plan `docs/plan-ts38mw-multiwrap-install.md` §4.4. Reads the launcher's
results file (`analysis/ts38mw_probe.json`, scp'd from the box) and plots
each of the 6 scored pins (bare_op/sym_q/word_q/sumof/sumof_bare/dm_mix) as
zero-shot EM and label loss (nats) against step, with the base model's score
on each pin as a horizontal dashed line of matching color. Losses are nats,
per CLAUDE.md convention (bits conversion only at reporting boundaries).

    python3 plot_ts38mw_probe.py [--in ts38mw_probe.json] [--out figures/ts38mw_probe.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PINS = ("bare_op", "sym_q", "word_q", "sumof", "sumof_bare", "dm_mix")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in", dest="in_path", type=Path, default=Path(__file__).resolve().parent / "ts38mw_probe.json"
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "figures" / "ts38mw_probe.png"
    )
    args = ap.parse_args()

    data = json.loads(args.in_path.read_text())
    rows = sorted(data["rows"], key=lambda r: r["step"])
    base = data["base"]["pins"]
    steps = [r["step"] for r in rows]

    fig, (ax_em, ax_loss) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, pin in enumerate(PINS):
        color = colors[i % len(colors)]
        em = [r["pins"][pin]["em0"] for r in rows]
        loss = [r["pins"][pin]["loss"] for r in rows]
        ax_em.plot(steps, em, color=color, marker=".", ms=5, lw=1.6, label=pin)
        ax_loss.plot(steps, loss, color=color, marker=".", ms=5, lw=1.6, label=pin)
        ax_loss.axhline(base[pin]["loss"], color=color, lw=0.9, ls="--", alpha=0.6)

    ax_em.axhline(0.95, color="black", lw=0.8, ls=":", alpha=0.5)
    ax_em.set_xlabel("step")
    ax_em.set_ylabel("zero-shot exact match")
    ax_em.set_title("held-out pin EM vs. step (base = 0 on all pins)")
    ax_em.set_ylim(-0.02, 1.02)
    ax_em.legend(fontsize=8)
    ax_em.grid(True, alpha=0.2)

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("label loss (nats/token)")
    ax_loss.set_title("held-out pin loss vs. step (dashed = base)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.2)

    fig.suptitle(f"ts38mw Stage 1 probe ({data['rows'][0]['run_id'] if rows else 'no rows'})")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out} ({len(rows)} snapshots)")


if __name__ == "__main__":
    main()
