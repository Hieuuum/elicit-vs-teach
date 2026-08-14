"""fig2nl: epoch-1 prequential MDL, test-floored EDL, and EDL per label token.

Per dataset size n and condition:
    MDL   = sum of loss_sum_nats over epoch-1 prequential.jsonl records
    D     = sum of label_token_count over the same records
    L_test= eval/test_loss.json loss_per_label_token_nats (final converged test loss)
    EDL   = MDL - D * L_test
    EDL/D = EDL / D
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pandas as pd

from geode.edl.metrics import edl_from_totals, edl_nats, epoch1_totals
from geode.zoo import test_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
LN2 = math.log(2.0)
RE = re.compile(r"^evt-llama-fig2nl-(noinst|inst)-n(\d+)$")

rows = []
for d in sorted((STORE / "runs").iterdir()):
    m = RE.match(d.name)
    if not m or not (d / "logs" / "prequential.jsonl").is_file():
        continue
    cond, n = m.group(1), int(m.group(2))
    mdl, tok, ex = epoch1_totals(d.name, store=STORE)
    l_test = test_loss(d.name, store=STORE).loss_per_label_token_nats
    edl = edl_from_totals(mdl, tok, l_test)
    guarded = edl_nats(d.name, store=STORE)  # masking-parity guard (D-1)
    assert abs(guarded - edl) < 1e-6, (d.name, guarded, edl)
    # L_test is evaluated at theta_T (the stopping step), NOT at the val
    # minimum -- geode/edl/loop.py:296 and no restore-to-best anywhere. Carry
    # the overshoot so the floor's provenance travels with the number.
    ev = sorted(
        (json.loads(line) for line in (d / "eval_log.jsonl").open() if line.strip()),
        key=lambda r: r["step"],
    )
    min_val = min(r["val_loss_nats"] for r in ev)
    last_val = ev[-1]["val_loss_nats"]
    rows.append(
        {
            "n": n,
            "condition": cond,
            "epoch1_examples": ex,
            "label_tokens_D": tok,
            "tok_per_ex": tok / ex,
            "mdl_epoch1_nats": mdl,
            "l_test_nats": l_test,
            "edl_nats": edl,
            "edl_per_token_nats": edl / tok,
            "edl_per_token_bits": edl / tok / LN2,
            "edl_per_example_nats": edl / ex,
            "min_val_nats": min_val,
            "last_val_nats": last_val,
            "overshoot_ratio": last_val / min_val,
        }
    )

df = pd.DataFrame(rows).sort_values(["n", "condition"], ascending=[True, False])
out = Path(__file__).with_name("fig2nl_edl_test_floor.csv")
df.to_csv(out, index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)
print(
    df.to_string(
        index=False,
        formatters={
            "mdl_epoch1_nats": "{:,.1f}".format,
            "l_test_nats": "{:.6f}".format,
            "edl_nats": "{:,.1f}".format,
            "edl_per_token_nats": "{:.6f}".format,
            "edl_per_token_bits": "{:.6f}".format,
            "edl_per_example_nats": "{:.6f}".format,
            "tok_per_ex": "{:.3f}".format,
        },
    )
)
print(f"\nwrote {out}  ({len(df)} runs)")
