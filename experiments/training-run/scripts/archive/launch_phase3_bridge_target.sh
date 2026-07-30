#!/usr/bin/env bash
# PHASE 3 BRIDGED TARGET (owner 2026-07-27) — train the NL-addition target
# DIRECTLY on the translation bridge that failed the retention gate, skipping
# the recovery. The bridge (evt-p3-elicit-bridge) kept NL<->operator
# translation (G6 0.9993) but damaged operator-notation addition (recorded G2
# 0.3018). This trains a plain-config, G7-matched NL-addition LoRA target on
# that checkpoint and records G5 evidence, so its EDL curve can be read next to
# the no-bridge control (evt-p3-elicit-target) and the recovery target.
#
#   ./launch_phase3_bridge_target.sh --confirm-cost
#
# One stage, resume-safe (a complete run skips, a missing run starts, any other
# status stops). require_parent_ready is bypassed by design: target_on_bridge
# .yaml declares parent_run_id: null + external_base because the bridge's
# recorded pass:false G2 must not be erased (see its header). This does NOT go
# through launch_phase3.sh --stage bridge: that stage would re-touch the
# bridge's recorded gates (its G2 is already recorded, so the halt is skipped
# and it would then RECORD G4 on the shared bridge parent). This path leaves the
# bridge's gate record exactly as it is.
#
# CONVERGENCE IS REQUIRED. stop_reason=max_steps means "did not converge" (the
# config's own bug signal) and HALTS.
#
# ntfy fires ONCE, at the terminal state (deliverable or failure) — never at a
# stage boundary (owner 2026-07-27). Set $NTFY to the full topic URL.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase3_bridge_target.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT                 # the guard + trainer resolve local_path here
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
BRIDGE_RID=evt-p3-elicit-bridge
TARGET_RID=evt-p3-elicit-target-bridge
echo "[p3bt] repo  $(git log --oneline -1)"
echo "[p3bt] store $GEODE_STORE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {                         # terminal: one ntfy, then exit
  notify "phase3 bridged target FAILED: $1"
  echo "[p3bt] FAILED: $1" >&2
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

stop_reason_of() { # run_id result_key -> stop_reason string (or empty)
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
r = json.loads(p.read_text()).get("experiment", {}).get(sys.argv[2], {}) or {}
print(r.get("stop_reason", ""))
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

ckpt_of() { # run_id -> checkpoint dir, or empty
  local d=$GEODE_STORE/runs/$1/model
  [[ -f $d/model.safetensors ]] && echo "$d"
}

# ---- guards (base-config artifacts + LR pins, before any spend) -----------
python3 phase3_guards.py --configs ../configs --repo-root "$REPO_ROOT" || exit 1

# ---- stage: bridged NL-addition target ------------------------------------
BRIDGE_CKPT=$(ckpt_of "$BRIDGE_RID") || true
[[ -n ${BRIDGE_CKPT:-} ]] || fail "no checkpoint for $BRIDGE_RID (the bridged-target base)"

ST=$(status_of "$TARGET_RID")
if [[ $ST == complete ]]; then
  echo "[p3bt] $TARGET_RID already complete — skipping"
else
  [[ $ST == missing ]] || fail "$TARGET_RID has status '$ST' — resolve it first"
  echo "[p3bt] === target: NL addition on the bridge checkpoint, prequential EDL ==="
  python3 train_target.py --config ../configs/p3_elicit_target.yaml \
    --override ../configs/p3/target_on_bridge.yaml \
    --init-from "$BRIDGE_CKPT" --confirm-cost || fail "$TARGET_RID training"
fi
SR=$(stop_reason_of "$TARGET_RID" target_result)
[[ $SR == max_steps ]] && fail "$TARGET_RID stop_reason=max_steps — the NL target did not \
converge on the bridge checkpoint (bug signal)"

gate_done "$TARGET_RID" G5 || python3 gates.py g5 --run "$TARGET_RID" \
  --config ../configs/eval_p3_data.yaml || fail "$TARGET_RID G5"

# ---- terminal report (one ntfy) -------------------------------------------
REPORT=$(python3 - "$TARGET_RID" <<'PY'
import json, os, sys
from pathlib import Path

target = sys.argv[1]
m = json.loads(
    (Path(os.environ["GEODE_STORE"]) / "runs" / target / "manifest.json").read_text()
)
res = m.get("experiment", {}).get("target_result", {})
g5 = m.get("experiment", {}).get("gates", {}).get("G5", {})
print(
    f"stop {res.get('stop_reason')} @ step {res.get('final_step')} | "
    f"min val {res.get('min_val_nats')} nats | "
    f"G5 zero-shot {g5.get('zero_shot_accuracy')} "
    f"16-shot {g5.get('sixteen_shot_accuracy')} "
    f"test-loss {g5.get('test_loss_nats')} nats"
)
PY
)
echo "[p3bt] REPORT: $REPORT"
notify "phase3 bridged target done — $REPORT"
echo "[p3bt] read the curve next to the control under BOTH floors:"
echo "[p3bt]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID --floor test"
echo "[p3bt] bridged target complete"
