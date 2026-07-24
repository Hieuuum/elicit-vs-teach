#!/usr/bin/env bash
# 1M Arm-B LR sweep runner — automates run7-8-guide.md §2 (the runs-7/8
# pilot). Three overlays on run8_target_1m.yaml, one epoch each (max_steps
# 7813, no snapshots), sequential. Skips points already complete in the
# store, so a crashed sweep resumes by re-running this script. Ends with a
# winner summary + the edge-rule verdict (guide §3).
#
#   ./sweep_1m.sh --confirm-cost
#
# Budget rule: --confirm-cost is REQUIRED and is forwarded to every
# train_target.py launch (each still prints its own estimate first); use
# the guide's dry-run command to see an estimate without training.
# Preconditions: box synced (git hash == laptop), parent evt-run4-armB-inst
# pulled. Optional: $NTFY (ntfy.sh/<topic>) for pings.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "sweep_1m.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
PARENT=$GEODE_STORE/runs/evt-run4-armB-inst/model
[[ -f $PARENT/model.safetensors ]] || {
  echo "sweep_1m.sh: parent checkpoint missing at $PARENT — run:" >&2
  echo "  python hf_checkpoint.py pull --run-id evt-run4-armB-inst" >&2
  exit 1
}
echo "[sweep] repo  $(git log --oneline -1)"
echo "[sweep] store $GEODE_STORE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }

status_of() { # run_id -> complete | missing | <status>
  python - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

for lr in 3e-4 1e-3 3e-3; do
  rid=evt-run8-sweep-lr${lr}
  st=$(status_of "$rid")
  if [[ $st == complete ]]; then
    echo "[sweep] $rid already complete — skipping"
    continue
  elif [[ $st != missing ]]; then
    echo "[sweep] $rid exists with status '$st' — crashed launch? Inspect it" >&2
    echo "        (and delete the run dir by hand) before re-running." >&2
    exit 1
  fi
  echo "[sweep] launching $rid ..."
  if python train_target.py --config ../configs/run8_target_1m.yaml \
      --override "../configs/pilot/target_sweep_1m_lr${lr}.yaml" \
      --init-from "$PARENT" --confirm-cost; then
    notify "1M sweep lr=${lr} done"
  else
    notify "1M sweep lr=${lr} FAILED"
    echo "[sweep] $rid failed — stopping (later points would waste budget)." >&2
    exit 1
  fi
done

python - <<'PY'
import json, os
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rows = []
for tag in ("3e-4", "1e-3", "3e-3"):
    m = json.loads((store / "runs" / f"evt-run8-sweep-lr{tag}" / "manifest.json").read_text())
    r = m["experiment"]["target_result"]
    val = r["min_val_nats"] if r["min_val_nats"] is not None else float("inf")
    rows.append((val, tag))
    print(f"lr {tag:>5}: min_val {val:.5f} nats  stop {r['final_step']} ({r['stop_reason']})")
val, tag = min(rows)
print(f"\nwinner: lr={tag} (lowest stopping-block min_val_nats at the shared 1-epoch budget)")
if tag == "3e-3":
    print("EDGE + unproven on Arm A: extend the grid (1e-2) AND run the 100K Arm-A")
    print("pilot before pinning (guide §3, decisions.md 2026-07-24).")
elif tag == "3e-4":
    print("EDGE winner: extend the grid one step down (1e-4) before pinning (guide §3).")
else:
    print("interior winner — pin train.lr in ALL FOUR run yamls (run7/run8/run9/")
    print("run10 — one LR everywhere, owner 2026-07-24) + decisions.md, commit +")
    print("push on the laptop, then box git pull and launch_pair_1m.sh (7/8 only)")
    print("or launch_chain_7_10.sh (7 -> 8 -> 9 -> 10 unattended).")
PY
notify "1M sweep COMPLETE — check the winner summary"
