#!/usr/bin/env bash
# Phase-2 target LR sweep: re-measure the 1e-3 target pin on THIS phase's two
# parents, which did not exist when the pin was set (configs/lr_pin.yaml is
# Arm-A evidence from the run-3-installed checkpoint and Arm-B evidence from
# run 4's installer; this phase's Arm A has no installer at all and its Arm B
# sits behind the permuted-label one). Design, grid and the ten pre-registered
# reading rules live in configs/pilot/p2_sweep_armA_lr1e-3.yaml — read that
# file before reading any number this script produces.
#
#   ./launch_p2_lr_sweep.sh --confirm-cost [--only <run_id substring>]
#
# 8 points: {3e-4, 1e-3, 3e-3} x {A, B} plus a seed-318 twin at 1e-3 per arm
# (the noise handle, rule 5). Ordered so an interruption still leaves the most
# informative subset: both incumbents, then both twins, then the brackets.
# Resumable — completed points are skipped, so a crash, a box restart or a
# stage split across machines all pick up where they stopped. ntfy per point
# and on failure (set $NTFY).
#
# THIS SCRIPT MEASURES NOTHING THAT GETS REPORTED. Every point runs at seed
# 317/318 and is marked lifecycle: pilot. Under reading rule 7 a sweep point's
# min_val is a selected minimum, biased low by construction, and min_val sets
# theta_T hence L_test hence EDL — so if the pin moves, production re-runs at
# the production seed 316 under the existing target run_ids. This script must
# never write to evt-p2-armA-target-noinst or evt-p2-armB-target-perm, and it
# refuses below if a point's run_id is not an evt-p2-sweep-* id.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_p2_lr_sweep.sh: --confirm-cost required (budget rule)" >&2
  echo "  8 points, full 1M, ceiling 46878. Measured: runs 7/8 took 6,000 and" >&2
  echo "  11,000 steps at ~0.14 s/step WITH snapshots; these store none." >&2
  echo "  Estimate ~2-3 GPU-h total = ~\$1.2 at \$0.45/h." >&2
  exit 1
}
ONLY=""
PREV=""
for i in "$@"; do
  [[ $PREV == --only ]] && ONLY=$i
  PREV=$i
done

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
echo "[sweep] repo  $(git log --oneline -1)"
echo "[sweep] store $GEODE_STORE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {
  notify "p2 lr sweep FAILED: $1"
  echo "[sweep] FAILED: $1" >&2
  exit 1
}

status_of() {
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

# Arm -> (base config, parent run_id). Arm A inits from the algo checkpoint
# itself (this phase's elicit arm has no installer); Arm B from the permuted
# -label installer. Same two parents the production targets will use — that is
# the entire point of the sweep.
base_cfg() { [[ $1 == A ]] && echo ../configs/p2_armA_target_noinst.yaml || echo ../configs/p2_armB_target_perm.yaml; }
parent_of() { [[ $1 == A ]] && echo evt-run2-armA-algo || echo evt-p2-armB-instperm; }

# run_id:arm:overlay — see the header for why this order.
POINTS=(
  "evt-p2-sweep-armA-lr1e-3:A:p2_sweep_armA_lr1e-3.yaml"
  "evt-p2-sweep-armB-lr1e-3:B:p2_sweep_armB_lr1e-3.yaml"
  "evt-p2-sweep-armA-lr1e-3-s318:A:p2_sweep_armA_lr1e-3-s318.yaml"
  "evt-p2-sweep-armB-lr1e-3-s318:B:p2_sweep_armB_lr1e-3-s318.yaml"
  "evt-p2-sweep-armA-lr3e-3:A:p2_sweep_armA_lr3e-3.yaml"
  "evt-p2-sweep-armB-lr3e-3:B:p2_sweep_armB_lr3e-3.yaml"
  "evt-p2-sweep-armA-lr3e-4:A:p2_sweep_armA_lr3e-4.yaml"
  "evt-p2-sweep-armB-lr3e-4:B:p2_sweep_armB_lr3e-4.yaml"
)

# ---- guards (all before any spend) ---------------------------------------

python3 - <<'PY' || exit 1
import sys
from pathlib import Path

import yaml

cfg = Path("../configs")
pin = yaml.safe_load((cfg / "lr_pin.yaml").read_text())
inst_lr = float(pin["installer_lr"])
grid = {}
for f in sorted((cfg / "pilot").glob("p2_sweep_arm*.yaml")):
    o = yaml.safe_load(f.read_text())
    rid, lr, seed = o["run_id"], float(o["train"]["lr"]), int(o["train"]["seed"])
    # Rule 7 tripwire: a sweep overlay must never carry a production run_id
    # or the production seed — either would put a selected minimum into the
    # reported numbers.
    if not rid.startswith("evt-p2-sweep-"):
        sys.exit(f"sweep: {f.name} run_id '{rid}' is not an evt-p2-sweep-* id (rule 7)")
    if seed == 316:
        sys.exit(f"sweep: {f.name} uses the PRODUCTION seed 316 (rule 7) — use 317/318")
    if o["experiment"]["match_data_order_with"] is not None:
        sys.exit(f"sweep: {f.name} must set experiment.match_data_order_with: null")
    if o["train"]["snapshots"]["n"] != 0:
        sys.exit(f"sweep: {f.name} must set train.snapshots.n: 0 (disposable point)")
    # The run-9 scope leak in the other direction: a target-stage sweep must
    # never accidentally sit at the installer LR.
    if abs(lr - inst_lr) < 1e-12:
        sys.exit(f"sweep: {f.name} lr {lr} equals the installer pin — target stage only")
    grid.setdefault(rid.split("-arm")[1][0], []).append((lr, seed))

# The grid must bracket the incumbent on BOTH sides in BOTH arms, or a
# surviving incumbent cannot be called a peak ("nulls need bracketing").
inc = float(pin["lr"])
for arm, pts in sorted(grid.items()):
    lrs = {lr for lr, _ in pts}
    if not (any(x < inc for x in lrs) and any(x > inc for x in lrs)):
        sys.exit(f"sweep: arm {arm} does not bracket the incumbent {inc} on both sides")
    if (inc, 317) not in pts or (inc, 318) not in pts:
        sys.exit(f"sweep: arm {arm} needs the incumbent at BOTH seeds (noise handle, rule 5)")
    print(f"[sweep] arm {arm}: lrs {sorted(lrs)} bracket incumbent {inc}, seed twin present")

# The production targets must be untouched by this sweep.
for f in ("p2_armA_target_noinst.yaml", "p2_armB_target_perm.yaml"):
    t = yaml.safe_load((cfg / f).read_text())["train"]
    if abs(float(t["lr"]) - inc) > 1e-12 or int(t["seed"]) != 316:
        sys.exit(f"sweep: {f} has drifted off the pin/production seed — resolve before sweeping")
print(f"[sweep] production targets still at lr {inc}, seed 316 (untouched)")
PY

for spec in "${POINTS[@]}"; do
  IFS=: read -r RID ARM OVERLAY <<<"$spec"
  [[ -n $ONLY && $RID != *"$ONLY"* ]] && continue
  ST=$(status_of "$RID")
  if [[ $ST == complete ]]; then
    echo "[sweep] $RID already complete — skipping"
    continue
  fi
  [[ $ST == missing ]] || fail "$RID has status '$ST' — resolve it first"
  PARENT=$(parent_of "$ARM")
  CKPT=$GEODE_STORE/runs/$PARENT/model
  [[ -f $CKPT/model.safetensors ]] || fail "no checkpoint for $PARENT (pull it from the relay first)"

  echo "[sweep] === $RID (arm $ARM, parent $PARENT) ==="
  python3 train_target.py --config "$(base_cfg "$ARM")" \
    --override "../configs/pilot/$OVERLAY" \
    --init-from "$CKPT" --confirm-cost || fail "$RID training"

  # lifecycle: pilot — a sweep point is never canonical evidence (rule 7).
  # No gates are scored: G5 would spend a whole-file eval on a run whose
  # numbers are unreportable by construction.
  python3 - "$RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
m = json.loads(p.read_text())
m["lifecycle"] = "pilot"
m["lifecycle_note"] = (
    "LR sweep point, seed 317/318 — a selected minimum, never reported. "
    "See configs/pilot/p2_sweep_armA_lr1e-3.yaml reading rule 7."
)
p.write_text(json.dumps(m, indent=1))
recs = [
    json.loads(x)
    for x in (p.parent / "eval_log.jsonl").read_text().splitlines()
    if x.strip()
]
recs = [r for r in recs if "val_loss_nats" in r]
best = min(recs, key=lambda r: r["val_loss_nats"])
print(
    f"[sweep] {sys.argv[1]}: min_val {best['val_loss_nats']:.5f} nats @ step "
    f"{best['step']} (last eval step {recs[-1]['step']}, {len(recs)} evals)"
)
PY
  notify "p2 lr sweep: $RID done"
done

echo "[sweep] all points done — read with: python3 ../analysis/lr_sweep_read.py"
notify "p2 lr sweep: ALL POINTS DONE"
