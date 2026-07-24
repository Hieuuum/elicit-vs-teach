#!/usr/bin/env bash
# Runs 7/8 production pair @ the full 1M — the hours-long unattended
# launcher (owner 2026-07-24: this script ends after run 8; the Llama chain
# is a separate, later script). Sequential: run 7 (Arm A) then run 8
# (Arm B) — G7 needs run 7 registered first. A completed run is skipped on
# re-run, so a crash resumes from where it stopped.
#
#   ./launch_pair_1m.sh --confirm-cost
#
# Refuses to start unless ALL of:
#   - train.lr is pinned, EQUAL in both run7/run8 yamls, and matches the
#     winner among the completed evt-run8-sweep-lr* runs in this store
#     (>=3 points; catches "forgot to re-pin after the sweep");
#   - >=200 GB free on the store filesystem (owner 2026-07-24 disk plan);
#   - both parent checkpoints (run 3 / run 4) are present.
# stop_reason=max_steps on EITHER run here is a bug signal (unlike the
# sweep, where it is the design) — investigate, don't ship.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_pair_1m.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
echo "[pair] repo  $(git log --oneline -1)"
echo "[pair] store $GEODE_STORE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }

# 1. LR pin: shared, non-null, and equal to the sweep winner in this store.
python - <<'PY' || exit 1
import json, os, sys
from pathlib import Path

import yaml

cfg = Path("../configs")
lr7 = yaml.safe_load((cfg / "run7_target_1m.yaml").read_text())["train"]["lr"]
lr8 = yaml.safe_load((cfg / "run8_target_1m.yaml").read_text())["train"]["lr"]
if lr7 is None or lr7 != lr8:
    sys.exit(f"pair: run7 lr={lr7} / run8 lr={lr8} — one shared non-null pin required")
best, best_lr, seen = float("inf"), None, 0
for p in sorted((Path(os.environ["GEODE_STORE"]) / "runs").glob("evt-run8-sweep-lr*/manifest.json")):
    m = json.loads(p.read_text())
    if m["status"] != "complete":
        continue
    seen += 1
    v = m["experiment"]["target_result"]["min_val_nats"]
    v = float("inf") if v is None else v
    if v < best:
        best, best_lr = v, float(m["training"]["optimizer"]["lr"])
if seen < 3:
    sys.exit(f"pair: only {seen} complete sweep runs in this store — the pin is unreviewable; finish sweep_1m.sh first")
if abs(lr7 - best_lr) > 1e-12:
    sys.exit(f"pair: configs pin lr={lr7} but the sweep winner here is {best_lr} — re-pin (or resolve the discrepancy) first")
print(f"[pair] lr pin {lr7} matches the sweep winner ({seen} sweep runs checked)")
PY

# 2. Disk: the pair adds ~130 GB realistic / ~190 GB worst case (n=2048).
free_gb=$(df -BG --output=avail "$GEODE_STORE" | tail -1 | tr -dc '0-9')
if ((free_gb < 200)); then
  echo "pair: ${free_gb} GB free < 200 GB plan (decisions.md 2026-07-24) — free disk first" >&2
  exit 1
fi
echo "[pair] disk ok: ${free_gb} GB free"

# 3. Parents.
for parent in evt-run3-armA-inst evt-run4-armB-inst; do
  [[ -f $GEODE_STORE/runs/$parent/model/model.safetensors ]] || {
    echo "pair: parent checkpoint missing — python hf_checkpoint.py pull --run-id $parent" >&2
    exit 1
  }
done

status_of() { # run_id -> complete | missing | <status>
  python - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

launch_one() { # $1 run_id  $2 config yaml  $3 parent run_id
  local st
  st=$(status_of "$1")
  if [[ $st == complete ]]; then
    echo "[pair] $1 already complete — skipping"
    return 0
  elif [[ $st != missing ]]; then
    echo "[pair] $1 exists with status '$st' — crashed launch? Inspect it (and" >&2
    echo "       delete the run dir by hand) before re-running." >&2
    return 1
  fi
  echo "[pair] launching $1 ..."
  if python train_target.py --config "../configs/$2" \
      --init-from "$GEODE_STORE/runs/$3/model" --confirm-cost; then
    notify "$1 done"
  else
    notify "$1 FAILED"
    echo "[pair] $1 failed — stopping before the next run burns budget." >&2
    return 1
  fi
}

launch_one evt-run7-armA-target-1m run7_target_1m.yaml evt-run3-armA-inst || exit 1
launch_one evt-run8-armB-target-1m run8_target_1m.yaml evt-run4-armB-inst || exit 1

python - <<'PY'
import json, os
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
for rid in ("evt-run7-armA-target-1m", "evt-run8-armB-target-1m"):
    m = json.loads((store / "runs" / rid / "manifest.json").read_text())
    r = m["experiment"]["target_result"]
    flag = "  <-- BUG SIGNAL: did not converge" if r["stop_reason"] == "max_steps" else ""
    print(
        f"{rid}: stop {r['final_step']} ({r['stop_reason']}) "
        f"min_val {r['min_val_nats']:.5f} nats, "
        f"snapshots {len(m.get('snapshots_taken', []))}/{len(m['snapshot_steps'])}{flag}"
    )
PY
notify "1M pair COMPLETE (runs 7+8) — box stays ALIVE (runs-5/6 snapshots live only here)"
