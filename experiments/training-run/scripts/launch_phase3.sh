#!/usr/bin/env bash
# PHASE 3 (owner 2026-07-27) — the elicit arm only, run in dependency order,
# unattended, skipping anything already complete. A crash, a box restart, or a
# stage split across machines all resume from where they stopped.
#
#   ./launch_phase3.sh --confirm-cost [--stage parent|target|all]
#
# Stages (default all, in this order):
#   parent   Operator-notation ADDITION pre-intervention run + G1. Needs a GPU.
#   target   The conditional format-install decision, the installer if and only
#            if it is needed, then the natural-language addition target through
#            the prequential EDL harness + G5. Needs a GPU.
#
# THE CONDITIONAL. Before the target, the parent's format validity is scored on
# NL ADDITION prompts drawn from the frozen external eval file:
#
#   gates.py g4 --run evt-p3-elicit-parent \
#       --prompt-config ../configs/eval_p3_data.yaml \
#       --threshold 0.90 --n-prompts 512 --no-record
#
# >= 0.90 -> no installer, the target trains straight from the parent.
# <  0.90 -> the NL-multiplication installer runs first, and the target picks
#            up configs/p3/target_after_inst.yaml.
# --no-record is required: a recorded sub-threshold G4 on the shared parent
# would make require_parent_ready refuse every child of it (V0.6).
#
# The decision is parsed from the gate's PRINTED RATE, not its exit code. In
# --no-record mode gates.py returns 1 both for "below threshold" and for any
# SystemExit, so an exit-code branch would silently read a crashed gate as
# "install needed" and spend the budget on it. Absence of the rate line is a
# hard failure here.
#
# Refuses to start unless ALL of:
#   - --confirm-cost (budget rule);
#   - every phase-3 artifact a pending stage needs is present and hash-matches;
#   - the target LR equals the committed pin AND differs from the installer LR
#     (the 2026-07-25 scope leak that destroyed run 9);
#   - the parent checkpoint exists for every pending run.
# If the installer runs, its RETENTION bar is enforced before the target: this
# is the only installer in the project that acts on a parent which already has
# the capability under study, so G2 below EXACT_MATCH_THRESHOLD halts the phase
# rather than shipping a damaged parent.
# stop_reason=max_steps on ANY run in this phase is a bug signal.
#
# ntfy fires on the deliverable and on failure only (owner preference), not at
# stage boundaries. Set $NTFY to the full topic URL.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase3.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
STAGE=all
PREV=""
for i in "$@"; do
  [[ $PREV == --stage ]] && STAGE=$i
  PREV=$i
done
case $STAGE in all | parent | target) ;;
*)
  echo "launch_phase3.sh: --stage must be all|parent|target, got '$STAGE'" >&2
  exit 1
  ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT                 # the guard resolves data.local_path against
                                 # THIS, never the store's parent
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
PARENT_RID=evt-p3-elicit-parent
INST_RID=evt-p3-elicit-inst
TARGET_RID=evt-p3-elicit-target
echo "[p3] repo  $(git log --oneline -1)"
echo "[p3] store $GEODE_STORE  stage=$STAGE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {
  notify "phase3 FAILED: $1"
  echo "[p3] FAILED: $1" >&2
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

parent_ckpt() { # run_id -> checkpoint dir, or empty
  local d=$GEODE_STORE/runs/$1/model
  [[ -f $d/model.safetensors ]] && echo "$d"
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
# order_hash reads exactly these six fields, so loading only them is
# equivalent to hashing the whole frame and about a quarter of the memory —
# which matters because the target artifact is 1M rows and the full
# to_dict("records") form of it is ~1.5 GB.
HASHED = ["a", "b", "op", "shown_answer", "format", "label_mode"]

# 1. Artifacts present and hashing to what the configs pin. Every launcher
#    re-verifies per run; doing it once up front turns a mid-phase abort into a
#    refusal to start. The eval file is checked through its own pin config,
#    because the target reads it as data.eval_file and G4/G5 read it directly.
for name, key in (
    ("p3_elicit_parent.yaml", "local_path"),
    ("p3_elicit_inst.yaml", "local_path"),
    ("p3_elicit_target.yaml", "local_path"),
    ("p3_elicit_target.yaml", "eval_local_path"),
    ("eval_p3_data.yaml", "local_path"),
):
    d = yaml.safe_load((cfg / name).read_text())["data"]
    local = d.get(key)
    hash_key = "eval_order_hash" if key.startswith("eval_") else "order_hash"
    file_key = "eval_file" if key.startswith("eval_") else "file"
    if not local:
        print(f"[p3] {name}:{key} unset — the launchers will pull {d[file_key]} from the hub")
        continue
    p = Path(local)
    p = p if p.is_absolute() else repo_root / p
    if not p.is_file():
        sys.exit(f"phase3: {name} pins {key} {local}, which does not exist")
    got = order_hash(pd.read_parquet(p, columns=HASHED).to_dict("records"))
    if got != d[hash_key]:
        sys.exit(f"phase3: {local} order_hash {got} != pinned {d[hash_key]}")
    print(f"[p3] {d[file_key]}: local copy hash-verified ({len(HASHED)} hashed columns)")

# 2. LR pins: the target equals the committed pin, the installer differs from
#    it. Applying a target LR to an installer is what destroyed run 9 v1.
pin = yaml.safe_load((cfg / "lr_pin.yaml").read_text())
t_lr = float(yaml.safe_load((cfg / "p3_elicit_target.yaml").read_text())["train"]["lr"])
i_lr = float(yaml.safe_load((cfg / "p3_elicit_inst.yaml").read_text())["train"]["lr"])
if abs(t_lr - float(pin["lr"])) > 1e-12:
    sys.exit(f"phase3: target pins lr={t_lr} but configs/lr_pin.yaml records {pin['lr']}")
if abs(i_lr - float(pin["installer_lr"])) > 1e-12:
    sys.exit(f"phase3: installer pins lr={i_lr}, lr_pin.yaml records {pin['installer_lr']}")
if abs(i_lr - t_lr) < 1e-12:
    sys.exit(
        f"phase3: the installer lr equals the target pin ({t_lr}) — this is the "
        "2026-07-25 scope leak that destroyed run 9's retention (lr_pin.yaml)"
    )
print(f"[p3] lr pins: target {t_lr}, installer {i_lr} (lr_pin.yaml)")

# 3. The parent stage's LR is NOT the target pin either. It is role-inherited
#    from run 2 and swept on a different dataset (p3_elicit_parent.yaml header);
#    what this guards is only that nobody has quietly set it to 1e-3.
p_lr = float(yaml.safe_load((cfg / "p3_elicit_parent.yaml").read_text())["train"]["lr"])
if abs(p_lr - t_lr) < 1e-12:
    sys.exit(
        f"phase3: the pre-intervention stage pins the TARGET lr ({t_lr}). That pin is "
        "scoped to LoRA target runs; a full-FT stage at that rate is the run-9 failure."
    )
print(f"[p3] pre-intervention lr {p_lr} (role-inherited from run 2, see config header)")
PY

# ---- stage: pre-intervention (operator-notation addition) ----------------

if [[ $STAGE == all || $STAGE == parent ]]; then
  ST=$(status_of "$PARENT_RID")
  if [[ $ST == complete ]]; then
    echo "[p3] $PARENT_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$PARENT_RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt evt-run1-base-v3-ext) || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for evt-run1-base-v3-ext (the phase-3 base)"
    echo "[p3] === pre-intervention: operator-notation addition ==="
    python3 train_sft.py --config ../configs/p3_elicit_parent.yaml \
      --init-from "$CKPT" --confirm-cost || fail "$PARENT_RID training"
  fi
  gate_done "$PARENT_RID" G1 || python3 gates.py g1 --run "$PARENT_RID" \
    --config ../configs/p3_elicit_parent.yaml || fail "$PARENT_RID G1"
fi

# ---- stage: the conditional install decision, then the target -----------

if [[ $STAGE == all || $STAGE == target ]]; then
  [[ $(status_of "$PARENT_RID") == complete ]] ||
    fail "$PARENT_RID is not complete — run --stage parent first"

  # The decision. Parsed from the printed rate; see the header for why the exit
  # code is not trusted. --no-record keeps the verdict off the shared parent.
  echo "[p3] === format-install decision: G4 on NL addition prompts ==="
  G4_OUT=$(python3 gates.py g4 --run "$PARENT_RID" \
    --prompt-config ../configs/eval_p3_data.yaml \
    --threshold 0.90 --n-prompts 512 --no-record 2>&1)
  echo "$G4_OUT"
  G4_RATE=$(sed -n 's/.*G4 format_validity \([0-9.]*\) on n=.*/\1/p' <<<"$G4_OUT" | head -1)
  [[ -n $G4_RATE ]] ||
    fail "the pre-target G4 printed no format_validity rate — it errored rather than
  scored, and its exit code cannot tell those apart. Output above."

  NEED_INSTALL=$(python3 -c "import sys; sys.exit(0 if float('$G4_RATE') < 0.90 else 1)" &&
    echo yes || echo no)
  echo "[p3] parent NL format validity $G4_RATE -> install needed: $NEED_INSTALL"

  TARGET_PARENT=$PARENT_RID
  OVERRIDE=()
  if [[ $NEED_INSTALL == yes ]]; then
    ST=$(status_of "$INST_RID")
    if [[ $ST == complete ]]; then
      echo "[p3] $INST_RID already complete — skipping"
    else
      [[ $ST == missing ]] || fail "$INST_RID has status '$ST' — resolve it first"
      CKPT=$(parent_ckpt "$PARENT_RID") || true
      [[ -n ${CKPT:-} ]] || fail "no checkpoint for $PARENT_RID"
      echo "[p3] === conditional format installer: NL multiplication ==="
      python3 train_sft.py --config ../configs/p3_elicit_inst.yaml \
        --init-from "$CKPT" --confirm-cost || fail "$INST_RID training"
    fi
    gate_done "$INST_RID" G4 || python3 gates.py g4 --run "$INST_RID" \
      --prompt-config ../configs/eval_p3_data.yaml --threshold 0.90 ||
      fail "$INST_RID G4 (the format never installed)"
    gate_done "$INST_RID" G2 || python3 gates.py g2 --run "$INST_RID" \
      --config ../configs/p3_elicit_parent.yaml || fail "$INST_RID G2"

    # RETENTION BAR. Unique to this phase: every other installer in the project
    # ran on a parent that knew nothing, so a retention loss was impossible.
    # This one acts on a parent that already holds the capability the phase
    # exists to elicit. A damaged parent would depress the target's EDL and
    # read as a *better* elicitation result, so the bar halts rather than warns.
    python3 - <<'PY' || fail "installer retention bar"
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p3-elicit-inst" / "manifest.json"
g2 = json.loads(p.read_text())["experiment"]["gates"].get("G2")
if g2 is None:
    sys.exit("phase3: evt-p3-elicit-inst has no recorded G2 — score it before the target")
if not g2["pass"]:
    sys.exit(
        f"phase3: installer G2 retention {g2['exact_match']:.4f} < {g2['threshold']} — the "
        "NL-mult installer damaged the parent's addition. That is the capability the phase "
        "measures, and losing it would DEFLATE the target's EDL and read as a better "
        "elicitation result. STOP (decisions.md 2026-07-25, run-9 retention)."
    )
print(f"[p3] retention bar: installer G2 {g2['exact_match']:.4f} >= {g2['threshold']}")
PY
    TARGET_PARENT=$INST_RID
    OVERRIDE=(--override ../configs/p3/target_after_inst.yaml)
  fi

  # Record the branch: which parent the EDL measurement actually ran from is a
  # fact about the result, and it is decided at run time.
  mkdir -p "$GEODE_STORE/runs/$TARGET_RID"
  python3 - "$G4_RATE" "$NEED_INSTALL" "$TARGET_PARENT" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p3-elicit-target" / "install_decision.json"
path.write_text(
    json.dumps(
        {
            "parent_nl_format_validity": float(sys.argv[1]),
            "threshold": 0.90,
            "installer_run": sys.argv[2] == "yes",
            "target_parent_run_id": sys.argv[3],
            "protocol": (
                "gates.py g4 --prompt-config eval_p3_data.yaml --n-prompts 512 --no-record; "
                "fixed rows [2048:2560] of the frozen D_p3_nl_eval, token-prefix prompts, "
                "greedy EOS-stopped, format_valid"
            ),
        },
        indent=2,
    )
)
print(f"[p3] branch recorded -> {path}")
PY

  ST=$(status_of "$TARGET_RID")
  if [[ $ST == complete ]]; then
    echo "[p3] $TARGET_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$TARGET_RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt "$TARGET_PARENT") || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for $TARGET_PARENT"
    echo "[p3] === target: NL addition, prequential EDL, from $TARGET_PARENT ==="
    python3 train_target.py --config ../configs/p3_elicit_target.yaml \
      "${OVERRIDE[@]}" --init-from "$CKPT" --confirm-cost || fail "$TARGET_RID training"
  fi
  gate_done "$TARGET_RID" G5 || python3 gates.py g5 --run "$TARGET_RID" \
    --config ../configs/eval_p3_data.yaml || fail "$TARGET_RID G5"

  notify "phase3 elicit arm done (installer run: $NEED_INSTALL) — EDL ready to read"
  echo "[p3] read the curve under BOTH floors:"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID --floor test"
fi

echo "[p3] stage '$STAGE' complete"
