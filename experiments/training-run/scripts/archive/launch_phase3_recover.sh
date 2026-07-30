#!/usr/bin/env bash
# PHASE 3 RECOVERY (owner 2026-07-27) — the translation bridge damaged the
# operator-notation addition capability (evt-p3-elicit-bridge G2 0.3018). This
# retrains op-addition on top of the bridge checkpoint until it re-converges,
# records the op-notation val accuracy, then trains a PLAIN NL-addition target
# on the recovered model. No installer, no G7 match, no bridged sibling.
#
#   ./launch_phase3_recover.sh --confirm-cost
#
# Stages (linear, resume-safe — complete runs skip, missing runs start, any
# other status stops):
#   recover  Full-FT operator-notation addition from the bridge checkpoint,
#            epsilon/k val-loss convergence. G1 (op-notation val exact match)
#            and G6 (did the NL<->op equivalence survive the repair?) are
#            scored --no-record: reported, never written as a blocking verdict.
#   target   Plain LoRA NL-addition target from the recovered checkpoint,
#            epsilon/k convergence, then G5 evidence.
#
# CONVERGENCE IS REQUIRED. A stop_reason of max_steps on either run means "did
# not converge" (the configs' own bug signal) and HALTS the chain — the owner
# asked to retrain "until it converges", so a ceiling stop is not a done state.
#
# require_parent_ready is bypassed for the recover run by design: the bridge
# has a recorded pass:false G2 that must not be erased, so recover_from_bridge
# .yaml declares parent_run_id: null + external_base (see its header). The
# recover run itself records no failing gate, so the target's parent check is
# clean.
#
# ntfy fires ONCE, at the terminal state (deliverable or failure) — never at a
# stage boundary (owner 2026-07-27). Set $NTFY to the full topic URL.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase3_recover.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT                 # the guard + trainers resolve local_path here
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
BRIDGE_RID=evt-p3-elicit-bridge
RECOVER_RID=evt-p3-elicit-recover
TARGET_RID=evt-p3-elicit-recover-target
echo "[p3r] repo  $(git log --oneline -1)"
echo "[p3r] store $GEODE_STORE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {                         # terminal: one ntfy, then exit
  notify "phase3 recovery FAILED: $1"
  echo "[p3r] FAILED: $1" >&2
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

# ---- stage: recover operator-notation addition from the bridge ------------
BRIDGE_CKPT=$(ckpt_of "$BRIDGE_RID") || true
[[ -n ${BRIDGE_CKPT:-} ]] || fail "no checkpoint for $BRIDGE_RID (the recovery base)"

ST=$(status_of "$RECOVER_RID")
if [[ $ST == complete ]]; then
  echo "[p3r] $RECOVER_RID already complete — skipping"
else
  [[ $ST == missing ]] || fail "$RECOVER_RID has status '$ST' — resolve it first"
  echo "[p3r] === recover: operator-notation addition from the bridge checkpoint ==="
  python3 train_sft.py --config ../configs/p3_elicit_parent.yaml \
    --override ../configs/p3/recover_from_bridge.yaml \
    --init-from "$BRIDGE_CKPT" --confirm-cost || fail "$RECOVER_RID training"
fi
SR=$(stop_reason_of "$RECOVER_RID" sft_result)
[[ $SR == max_steps ]] && fail "$RECOVER_RID stop_reason=max_steps — op-addition did not \
re-converge on the bridge checkpoint (bug signal); NOT proceeding to the target"

# op-notation val accuracy — recorded to this report, never a blocking verdict
echo "[p3r] === G1: operator-notation val exact match (reported, --no-record) ==="
G1_OUT=$(python3 gates.py g1 --run "$RECOVER_RID" \
  --config ../configs/p3_elicit_parent.yaml --no-record 2>&1) || true
echo "$G1_OUT"
G1_ACC=$(sed -n 's/.*G1 accuracy \([0-9.]*\) on n=.*/\1/p' <<<"$G1_OUT" | head -1)
[[ -n $G1_ACC ]] || fail "G1 scoring pass printed no accuracy — it errored rather than scored"

# did the NL<->op equivalence survive the op-addition repair? headline diagnostic
echo "[p3r] === G6: translation retention on the recovered checkpoint (--no-record) ==="
G6_OUT=$(python3 gates.py g6 --run "$RECOVER_RID" \
  --config ../configs/eval_p3_bridge_data.yaml --no-record 2>&1) || true
echo "$G6_OUT"
G6_ACC=$(sed -n 's/.*G6 translation exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$G6_OUT" | head -1)

# ---- stage: plain NL-addition target on the recovered model ---------------
ST=$(status_of "$TARGET_RID")
if [[ $ST == complete ]]; then
  echo "[p3r] $TARGET_RID already complete — skipping"
else
  [[ $ST == missing ]] || fail "$TARGET_RID has status '$ST' — resolve it first"
  RECOVER_CKPT=$(ckpt_of "$RECOVER_RID") || true
  [[ -n ${RECOVER_CKPT:-} ]] || fail "no checkpoint for $RECOVER_RID"
  echo "[p3r] === target: plain NL addition, prequential EDL, from $RECOVER_RID ==="
  python3 train_target.py --config ../configs/p3_elicit_target.yaml \
    --override ../configs/p3/target_after_recover.yaml \
    --init-from "$RECOVER_CKPT" --confirm-cost || fail "$TARGET_RID training"
fi
SR=$(stop_reason_of "$TARGET_RID" target_result)
[[ $SR == max_steps ]] && fail "$TARGET_RID stop_reason=max_steps — the NL target did not \
converge (bug signal)"

gate_done "$TARGET_RID" G5 || python3 gates.py g5 --run "$TARGET_RID" \
  --config ../configs/eval_p3_data.yaml || fail "$TARGET_RID G5"

# ---- terminal report (one ntfy) -------------------------------------------
REPORT=$(python3 - "$TARGET_RID" "$G1_ACC" "${G6_ACC:-n/a}" <<'PY'
import json, os, sys
from pathlib import Path

target, g1, g6 = sys.argv[1], sys.argv[2], sys.argv[3]
g5 = json.loads(
    (Path(os.environ["GEODE_STORE"]) / "runs" / target / "manifest.json").read_text()
).get("experiment", {}).get("gates", {}).get("G5", {})
print(
    f"recover op-val G1 exact-match {g1} | translation retention G6 {g6} | "
    f"target G5 zero-shot {g5.get('zero_shot_accuracy')} "
    f"16-shot {g5.get('sixteen_shot_accuracy')} "
    f"test-loss {g5.get('test_loss_nats')} nats"
)
PY
)
echo "[p3r] REPORT: $REPORT"
notify "phase3 recovery done — $REPORT"
echo "[p3r] recovery chain complete"
