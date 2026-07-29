#!/usr/bin/env bash
# New phase (2026-07-26, after the owner's same-day revision): ONE installer,
# two target runs. The elicit arm has no installer — the dose grid ran and
# measured every dose to be monotone damage, so its target trains straight
# from evt-run2-armA-algo (decisions.md, "no elicit installer"). Runs the
# phase in dependency order, unattended, skipping anything already complete —
# a crash, a box restart, or a stage split across machines all resume from
# where they stopped. ntfy ping at every stage boundary and on failure
# (set $NTFY).
#
#   ./launch_phase2.sh --confirm-cost [--stage teach|targets|all]
#
# Stages (default all, in this order):
#   teach    Arm B permuted-label shape installer + G4/G3/G5, and it reports
#            EXAMPLES SEEN (final_step x batch) — with the elicit arm at zero
#            installer examples that number is the phase's exposure
#            asymmetry, so it is printed, not assumed. Needs a GPU (per-step
#            behavioral eval over a 200K pool).
#   targets  2 prequential EDL target runs — Arm A FIRST (it is the G7 anchor
#            the Arm B target checks against), then Arm B. Needs a GPU.
#
# The retired `doses` stage lives in git history; its five runs, their curve
# (analysis/dose_curve.py) and the configs that produced them are all kept.
#
# Refuses to start unless ALL of:
#   - --confirm-cost (budget rule);
#   - the teach installer's artifact is present and hash-matches its pin;
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
case $STAGE in all | teach | targets) ;;
doses)
  echo "launch_phase2.sh: the 'doses' stage is RETIRED (owner 2026-07-26) — the" >&2
  echo "  elicit arm has no installer. Its five runs and their curve are kept;" >&2
  echo "  see decisions.md and analysis/dose_curve.py. Re-wiring it needs an" >&2
  echo "  owner decision, not a flag." >&2
  exit 1
  ;;
*)
  echo "launch_phase2.sh: --stage must be all|teach|targets, got '$STAGE'" >&2
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
for name in ("p2_armB_instperm.yaml",):
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

# 2. LR pins: targets equal the committed pin; the installer differs from it.
pin = yaml.safe_load((cfg / "lr_pin.yaml").read_text())
target_lrs = {
    f: yaml.safe_load((cfg / f).read_text())["train"]["lr"]
    for f in ("p2_armA_target_noinst.yaml", "p2_armB_target_perm.yaml")
}
if len(set(target_lrs.values())) != 1 or None in target_lrs.values():
    sys.exit(f"phase2: both target yamls need ONE shared non-null train.lr — got {target_lrs}")
t_lr = float(next(iter(set(target_lrs.values()))))
if abs(t_lr - float(pin["lr"])) > 1e-12:
    sys.exit(f"phase2: targets pin lr={t_lr} but configs/lr_pin.yaml records {pin['lr']}")
inst_lrs = {
    f: yaml.safe_load((cfg / f).read_text())["train"]["lr"]
    for f in ("p2_armB_instperm.yaml",)
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
  # EXPOSURE. The elicit arm now has no installer, so everything this run saw
  # is the phase's whole exposure asymmetry — unbilled warm-up on the target
  # task's own surface form, in the arm that is supposed to be the expensive
  # one. It is an output of the behavioral stop, not a config number, so print
  # it and copy it into decisions.md rather than calling it "low" untested.
  python3 - <<'PY'
import json, os
from pathlib import Path

m = json.loads(
    (Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p2-armB-instperm" / "manifest.json").read_text()
)
res = m["experiment"]["sft_result"]
bs = m["training"]["optimizer"]["batch_size"]
step = res["final_step"]
print(
    f"[p2] teach installer exposure: {step} steps x batch {bs} = {step * bs} examples seen "
    f"({res['stop_reason']}); Arm A saw 0. COPY THIS INTO decisions.md"
)
PY
  notify "phase2: teach installer done + gated"
fi

# ---- stage: targets (the EDL measurements) ------------------------------

if [[ $STAGE == all || $STAGE == targets ]]; then
  run_target() { # run_id parent_run_id config
    local rid=$1 parent=$2 cfg=$3
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
    echo "[p2] === $rid: prequential EDL measurement ==="
    python3 train_target.py --config "$cfg" --init-from "$ckpt" --confirm-cost ||
      fail "$rid training"
    gate_done "$rid" G5 || python3 gates.py g5 --run "$rid" \
      --config ../configs/eval_target_data.yaml || fail "$rid G5"
    notify "phase2: $rid done"
  }

  # The anchor first: the Arm B target's G7 check reads its manifest. Arm A's
  # parent is the algo checkpoint itself — this arm has no installer stage
  # (owner 2026-07-26), so the target is its first and only training run in
  # the phase.
  run_target evt-p2-armA-target-noinst evt-run2-armA-algo \
    ../configs/p2_armA_target_noinst.yaml

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
  run_target evt-p2-armB-target-perm evt-p2-armB-instperm \
    ../configs/p2_armB_target_perm.yaml
  notify "phase2: both target runs done"
fi

echo "[p2] stage '$STAGE' complete"
