#!/usr/bin/env bash
# New phase (2026-07-26): role-matched installers + the elicit dose-response
# curve. Runs the whole phase in dependency order, unattended, skipping
# anything already complete — a crash, a box restart, or a stage split across
# machines all resume from where they stopped. ntfy ping at every stage
# boundary and on failure (set $NTFY).
#
#   ./launch_phase2.sh --confirm-cost [--stage doses|teach|targets|all]
#
# Stages (default all, in this order):
#   doses    5 Arm A dose installers (n = 1 2 4 8 16) + G4/G2/G5 each.
#            <=16 examples each: laptop-CPU feasible at $0, no GPU needed.
#   teach    Arm B permuted-label shape installer + G4/G3/G5. Needs a GPU
#            (200K examples with a per-step behavioral eval).
#   targets  6 prequential EDL target runs — the dose-1 target FIRST (it is
#            the G7 anchor every other target checks against), then doses
#            2/4/8/16, then Arm B. Needs a GPU.
#
# Refuses to start unless ALL of:
#   - --confirm-cost (budget rule);
#   - both new-phase artifacts present and hash-matching their config pins;
#   - the dose rule's eps_nats/k are pinned non-null (they come from the
#     calibration pilots, decisions.md 2026-07-26 — never a placeholder);
#   - the target LR equals the committed pin AND differs from the installer
#     LR (the 2026-07-25 scope leak that destroyed run 9 at 1e-3);
#   - the parent checkpoint exists for every pending run.
# Before the Arm B target it additionally enforces the LEAK BAR: G5 zero-shot
# <= 0.02 on the permuted-label installer. G5 records pass: true by protocol
# (it is evidence, not a gate), so the numeric bar has to be enforced here —
# a leak deflates the teach arm's EDL and inflates the headline ratio.
# stop_reason=max_steps on ANY run in this phase is a bug signal.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase2.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
STAGE=all
PREV=""
for i in "$@"; do
  [[ $PREV == --stage ]] && STAGE=$i
  PREV=$i
done
case $STAGE in all | doses | teach | targets) ;; *)
  echo "launch_phase2.sh: --stage must be all|doses|teach|targets, got '$STAGE'" >&2
  exit 1
  ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT                 # the guard below resolves data.local_path
                                 # against THIS, never against the store's
                                 # parent: a box with GEODE_STORE elsewhere
                                 # (e.g. /workspace/store) would otherwise
                                 # look for the parquets in the wrong tree
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
DOSES=(1 2 4 8 16)
# The dose rule's eps/k is calibrated by replaying a RECORDED loss trajectory,
# and a trajectory reproduces only on the device that produced it. train_sft.py
# defaults to cuda-if-available, so on a GPU box the calibration pilots (run by
# hand) and these installers would silently land on different devices and the
# pin would no longer describe the curve it is applied to. Pin it here, run the
# pilots with the same value, and refuse a resume that would mix devices.
DOSE_DEVICE=${DOSE_DEVICE:-cpu}
# SKIP_TEST_LOSS=1 drops G5's shared-set NLL for the DOSE stage only: it is a
# forward pass over the eval file's whole 98K-row reporting block, ~40 min per
# run on a laptop CPU versus seconds on a box. The accuracies still record;
# re-running `gates.py g5` later overwrites the record with the full number.
# The teach and target stages are GPU stages and always compute it.
SKIP_TL=()
[[ -n ${SKIP_TEST_LOSS:-} ]] && SKIP_TL=(--skip-test-loss)
echo "[p2] repo  $(git log --oneline -1)"
echo "[p2] store $GEODE_STORE  stage=$STAGE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {
  notify "phase2 FAILED: $1"
  echo "[p2] FAILED: $1" >&2
  exit 1
}

status_of() { # run_id -> complete | missing | <status>
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

device_of() { # run_id -> recorded training device, or "" if unrecorded/missing
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if p.is_file():
    print(json.loads(p.read_text()).get("experiment", {}).get("device", ""))
PY
}

gate_done() { # run_id gate -> 0 if a verdict is already recorded
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file():
    sys.exit(1)
gates = json.loads(p.read_text()).get("experiment", {}).get("gates", {})
sys.exit(0 if sys.argv[2] in gates else 1)
PY
}

# ---- guards (all before any spend) ---------------------------------------

python3 - <<'PY' || exit 1
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from geode.arith import order_hash

cfg = Path("../configs")
repo_root = Path(os.environ["REPO_ROOT"])
# 1. Artifacts: present, and hashing to what the configs pin. The launchers
#    re-verify per run; doing it once up front turns a mid-phase abort into a
#    refusal to start.
for name in ("p2_armA_dose.yaml", "p2_armB_instperm.yaml"):
    d = yaml.safe_load((cfg / name).read_text())["data"]
    local = d.get("local_path")
    if local:
        p = Path(local)
        p = p if p.is_absolute() else repo_root / p
        if not p.is_file():
            sys.exit(f"phase2: {name} pins local_path {local}, which does not exist")
        got = order_hash(pd.read_parquet(p).to_dict("records"))
        if got != d["order_hash"]:
            sys.exit(f"phase2: {local} order_hash {got} != pinned {d['order_hash']}")
        print(f"[p2] {d['file']}: local copy hash-verified")
    else:
        print(f"[p2] {d['file']}: no local_path — the launchers will pull it from the hub")

# 2. The dose rule's eps/k must be pinned (calibration pilots, not a default).
s = yaml.safe_load((cfg / "p2_armA_dose.yaml").read_text())["train"]["stopping"]
if s.get("eps_nats") is None or s.get("k") is None:
    sys.exit(
        "phase2: p2_armA_dose.yaml has a null stopping eps_nats/k — pin them from the "
        "calibration pilots (configs/pilot/p2_dose_cal_n*.yaml) first"
    )
print(f"[p2] dose rule pinned: eps {s['eps_nats']} nats, k={s['k']}")

# 3. LR pins: targets equal the committed pin; installers differ from it.
pin = yaml.safe_load((cfg / "lr_pin.yaml").read_text())
target_lrs = {
    f: yaml.safe_load((cfg / f).read_text())["train"]["lr"]
    for f in ("p2_armA_target_dose.yaml", "p2_armB_target_perm.yaml")
}
if len(set(target_lrs.values())) != 1 or None in target_lrs.values():
    sys.exit(f"phase2: both target yamls need ONE shared non-null train.lr — got {target_lrs}")
t_lr = float(next(iter(set(target_lrs.values()))))
if abs(t_lr - float(pin["lr"])) > 1e-12:
    sys.exit(f"phase2: targets pin lr={t_lr} but configs/lr_pin.yaml records {pin['lr']}")
inst_lrs = {
    f: yaml.safe_load((cfg / f).read_text())["train"]["lr"]
    for f in ("p2_armA_dose.yaml", "p2_armB_instperm.yaml")
}
for f, lr in inst_lrs.items():
    if lr is None or abs(float(lr) - float(pin["installer_lr"])) > 1e-12:
        sys.exit(f"phase2: {f} pins installer lr={lr}, lr_pin.yaml records {pin['installer_lr']}")
    if abs(float(lr) - t_lr) < 1e-12:
        sys.exit(
            f"phase2: {f}'s installer lr equals the target pin ({t_lr}) — this is the "
            "2026-07-25 scope leak that destroyed run 9's retention (lr_pin.yaml)"
        )
print(f"[p2] lr pins: target {t_lr}, installer {next(iter(inst_lrs.values()))} (lr_pin.yaml)")
PY

parent_ckpt() { # run_id -> checkpoint dir, or empty
  local d=$GEODE_STORE/runs/$1/model
  [[ -f $d/model.safetensors ]] && echo "$d"
}

# ---- stage: doses --------------------------------------------------------

if [[ $STAGE == all || $STAGE == doses ]]; then
  CKPT=$(parent_ckpt evt-run2-armA-algo) || true
  [[ -n ${CKPT:-} ]] || fail "no checkpoint for evt-run2-armA-algo (the dose parent)"
  # PARENT BASELINE first: the same G4 the dose runs will be gated on, scored
  # on the parent before any dose is given. Without it a dose G4 of, say, 0.88
  # cannot be attributed — it is the phase-0 ambiguity ("the rule fired" vs
  # "the rule was already satisfied") in the arm that has no in-loop G4 at
  # all. --no-record because writing a verdict onto this shared parent would
  # gate every existing child of it (V0.6). COPY THE NUMBER INTO decisions.md.
  echo "[p2] === dose parent baseline: G4 on evt-run2-armA-algo, before any dose ==="
  python3 gates.py g4 --run evt-run2-armA-algo --no-record \
    --config ../configs/p2_armA_dose.yaml \
    --prompt-config ../configs/eval_target_data.yaml --threshold 0.90 ||
    echo "[p2] parent baseline is BELOW 0.90 — record it and read every dose G4 against it"
  for n in "${DOSES[@]}"; do
    RID=evt-p2-armA-dose$n
    ST=$(status_of "$RID")
    if [[ $ST == complete ]]; then
      echo "[p2] $RID already complete — skipping"
      DEV=$(device_of "$RID")
      [[ -z $DEV || $DEV == "$DOSE_DEVICE" ]] ||
        fail "$RID was trained on '$DEV' but this stage is pinned to '$DOSE_DEVICE' — a mixed-device dose curve is not a dose curve (set DOSE_DEVICE=$DEV or re-run the dose)"
    else
      [[ $ST == missing ]] || fail "$RID has status '$ST' — resolve it (see register_run) first"
      echo "[p2] === dose $n: training on $DOSE_DEVICE ==="
      OVR=()
      [[ $n != 1 ]] && OVR=(--override ../configs/p2/dose$n.yaml)
      python3 train_sft.py --config ../configs/p2_armA_dose.yaml "${OVR[@]}" \
        --init-from "$CKPT" --device "$DOSE_DEVICE" --confirm-cost || fail "$RID training"
      notify "phase2: $RID trained"
    fi
    # G4 on EXTERNAL prompts (this run carves no val split); 0.90 is the
    # phase's shared bar, the same one the teach installer stops on.
    gate_done "$RID" G4 || python3 gates.py g4 --run "$RID" \
      --config ../configs/p2_armA_dose.yaml \
      --prompt-config ../configs/eval_target_data.yaml --threshold 0.90 ||
      fail "$RID G4"
    # G2 = retention on D_algo: the gate that re-validates the inherited
    # installer LR for this arm (decision 4). A failure means extend the LR
    # downward, exactly as the run-9 fix did — not proceed.
    gate_done "$RID" G2 || python3 gates.py g2 --run "$RID" \
      --config ../configs/run2_algo.yaml || fail "$RID G2 (retention — do NOT proceed)"
    gate_done "$RID" G5 || python3 gates.py g5 --run "$RID" \
      --config ../configs/eval_target_data.yaml "${SKIP_TL[@]}" || fail "$RID G5"
  done
  notify "phase2: all 5 dose installers done + gated"
fi

# ---- stage: teach installer ---------------------------------------------

if [[ $STAGE == all || $STAGE == teach ]]; then
  RID=evt-p2-armB-instperm
  ST=$(status_of "$RID")
  if [[ $ST == complete ]]; then
    echo "[p2] $RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt evt-run1-base-v3-ext) || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for evt-run1-base-v3-ext (the teach parent)"
    echo "[p2] === teach shape installer: training ==="
    python3 train_sft.py --config ../configs/p2_armB_instperm.yaml \
      --init-from "$CKPT" --confirm-cost || fail "$RID training"
    notify "phase2: $RID trained"
  fi
  gate_done "$RID" G4 || python3 gates.py g4 --run "$RID" \
    --config ../configs/p2_armB_instperm.yaml || fail "$RID G4"
  gate_done "$RID" G3 || python3 gates.py g3 --run "$RID" \
    --config ../configs/run2_algo.yaml || fail "$RID G3"
  gate_done "$RID" G5 || python3 gates.py g5 --run "$RID" \
    --config ../configs/eval_target_data.yaml || fail "$RID G5"
  notify "phase2: teach installer done + gated"
fi

# ---- stage: targets (the EDL measurements) ------------------------------

if [[ $STAGE == all || $STAGE == targets ]]; then
  run_target() { # run_id parent_run_id [overlay]
    local rid=$1 parent=$2 ovr=${3:-}
    local st
    st=$(status_of "$rid")
    if [[ $st == complete ]]; then
      echo "[p2] $rid already complete — skipping"
      return 0
    fi
    [[ $st == missing ]] || fail "$rid has status '$st' — resolve it first"
    local ckpt
    ckpt=$(parent_ckpt "$parent") || true
    [[ -n ${ckpt:-} ]] || fail "no checkpoint for $parent (pull it from the relay first)"
    local args=(--config ../configs/p2_armA_target_dose.yaml)
    [[ $rid == evt-p2-armB-target-perm ]] && args=(--config ../configs/p2_armB_target_perm.yaml)
    [[ -n $ovr ]] && args+=(--override "$ovr")
    echo "[p2] === $rid: prequential EDL measurement ==="
    python3 train_target.py "${args[@]}" --init-from "$ckpt" --confirm-cost ||
      fail "$rid training"
    gate_done "$rid" G5 || python3 gates.py g5 --run "$rid" \
      --config ../configs/eval_target_data.yaml || fail "$rid G5"
    notify "phase2: $rid done"
  }

  # The anchor first: every other target's G7 check reads its manifest.
  run_target evt-p2-armA-target-dose1 evt-p2-armA-dose1
  for n in 2 4 8 16; do
    run_target evt-p2-armA-target-dose$n evt-p2-armA-dose$n ../configs/p2/target_dose$n.yaml
  done

  # LEAK BAR before the teach target (see header): matched-notation zero-shot.
  python3 - <<'PY' || fail "teach installer leak bar (G5 zero-shot > 0.02)"
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p2-armB-instperm" / "manifest.json"
g5 = json.loads(p.read_text())["experiment"]["gates"].get("G5")
if g5 is None:
    sys.exit("phase2: evt-p2-armB-instperm has no recorded G5 — score it before the target")
zero = g5["zero_shot_accuracy"]
if zero > 0.02:
    sys.exit(
        f"phase2: teach installer G5 zero-shot {zero:.4f} > 0.02 — the permuted-label "
        "installer taught real operator add/sub. Its target EDL would be deflated and "
        "the elicit/teach ratio inflated; STOP and investigate the permutation."
    )
print(f"[p2] leak bar: teach installer G5 zero-shot {zero:.4f} <= 0.02")
PY
  run_target evt-p2-armB-target-perm evt-p2-armB-instperm
  notify "phase2: all 6 target runs done"
fi

echo "[p2] stage '$STAGE' complete"
