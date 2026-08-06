"""fig2nl sweep: EDL/D vs dataset size under BOTH floors, side by side.

Left  — fixed test floor: EDL = MDL_epoch1 - D * L_test, L_test from each run's
        eval/test_loss.json (theta_T, the stopping-step model). Canonical Eq. 3.
        No figure of this existed before.
Right — min-val floor: target_result.edl_per_label_token_nats, i.e. what
        figures/dataset_size_sweep_nl.png already plots. Shown for reference.

Shared y-axis: the two floors give materially different heights and that gap is
the point. Runs whose final val loss sat >=1.5x above their own val minimum are
ringed on the left panel -- EDL/D is linear in the floor, so those points are
depressed by the stopping rule rather than by the data.

CPU-only, reads the local store + results parquet. Writes one PNG.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geode.edl.metrics import edl_from_totals, edl_nats, epoch1_totals
from geode.zoo import test_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
OUT = Path(__file__).resolve().parent / "figures" / "fig2nl_sweep_floors.png"
LN2 = math.log(2.0)
RE = re.compile(r"^evt-llama-fig2nl-(noinst|inst)-n(\d+)$")

# Repo-wide convention, unchanged: base = tab:blue, format-installed = tab:orange.
# Validated as a 2-slot categorical palette (CVD dE 24.6 protan / 31.8 tritan).
STYLE = {
    "noinst": ("#1f77b4", "base"),
    "inst": ("#ff7f0e", "format-installed"),
}
OVERSHOOT = 1.5

# ---- left panel: recompute the test-floored EDL/D from the logs -------------
rows = []
for d in sorted((STORE / "runs").iterdir()):
    m = RE.match(d.name)
    if not m or not (d / "logs" / "prequential.jsonl").is_file():
        continue
    mdl, tok, _ = epoch1_totals(d.name, store=STORE)
    l_test = test_loss(d.name, store=STORE).loss_per_label_token_nats
    edl = edl_from_totals(mdl, tok, l_test)
    assert abs(edl_nats(d.name, store=STORE) - edl) < 1e-6  # masking guard (D-1)
    ev = [json.loads(x) for x in (d / "eval_log.jsonl").open() if x.strip()]
    ev.sort(key=lambda r: r["step"])
    min_val = min(r["val_loss_nats"] for r in ev)
    rows.append(
        {
            "n": int(m.group(2)),
            "condition": m.group(1),
            "edl_per_token_bits": edl / tok / LN2,
            "overshoot": ev[-1]["val_loss_nats"] / min_val,
        }
    )
test_df = pd.DataFrame(rows)

# ---- right panel: the min-val-floored metric already in the parquet ---------
val_df = pd.read_parquet(STORE / "results" / "dataset_size_sweep_nl.parquet")
val_df = val_df[val_df["metric_name"] == "edl_per_label_token_nats"].copy()
val_df["edl_per_token_bits"] = val_df["metric_value"] / LN2
val_df = val_df.rename(columns={"dataset_size": "n"})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
panels = [
    (axes[0], test_df, "Fixed test floor  (canonical Eq. 3)", True),
    (axes[1], val_df, "Min-val floor  (= dataset_size_sweep_nl.png)", False),
]

for ax, df, title, is_test in panels:
    for cond, (color, label) in STYLE.items():
        s = df[df["condition"] == cond].sort_values("n")
        if s.empty:
            continue
        ax.plot(s["n"], s["edl_per_token_bits"], color=color, lw=2.0, alpha=0.9, zorder=2)
        ax.plot(
            s["n"],
            s["edl_per_token_bits"],
            ls="none",
            marker="o",
            ms=8,
            color=color,
            mec="white",
            mew=1.5,
            label=label,
            zorder=3,
        )
        # Direct label at the series end (contrast relief for the orange slot).
        last = s.iloc[-1]
        ax.annotate(
            label,
            xy=(last["n"], last["edl_per_token_bits"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
            zorder=4,
        )
        if is_test:
            over = s[s["overshoot"] >= OVERSHOOT]
            ax.plot(
                over["n"],
                over["edl_per_token_bits"],
                ls="none",
                marker="o",
                ms=15,
                mfc="none",
                mec="#c1121f",
                mew=1.6,
                zorder=1,
                label=f"stopped >={OVERSHOOT}x above own val min" if cond == "noinst" else None,
            )
    ax.set_xscale("log")
    ax.set_xlabel("training examples (log scale)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axes[0].set_ylabel("EDL/D  (bits per label token)")
axes[0].legend(fontsize=9, loc="upper right", frameon=False)
fig.suptitle(
    "fig2nl sweep: EDL per label token vs. dataset size — the floor decides the number\n"
    "Llama-3.2-1B, natural language on D_algo; D = epoch-1 training label tokens",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
fig.text(
    0.5,
    0.005,
    "Left is EDL = MDL_epoch1 − D·L_test with each run's own final (θ_T) test loss. "
    "inst > base is the confounded direction — installer parent entered at retention "
    "0.1719 vs base 0.3271 — read as arms not separated.",
    ha="center",
    fontsize=8,
    color="#555555",
)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"test-floor points: {len(test_df)}   val-floor points: {len(val_df)}")
print(f"ringed (overshoot >= {OVERSHOOT}x): {int((test_df['overshoot'] >= OVERSHOOT).sum())}")
