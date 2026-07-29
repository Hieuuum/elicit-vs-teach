#!/usr/bin/env bash
# PHASE 3 TEACH ARM — install the NL-addition surface/answer marginal without
# the mapping, then measure teaching over the same frozen 500K stream as the
# completed elicit target.
#
#   ./launch_phase3_teach.sh --confirm-cost [--stage teach|target|all]
#
# Resume-safe: complete runs skip, missing runs start, any other status stops.
# stop_reason=max_steps is a failure signal. The target is G7-matched to
# evt-p3-elicit-target by train_target.py before cost confirmation.
#
# The installer G5 is evidence-only in gates.py, so this launcher separately
# enforces zero-shot <= 0.02 before target spend. A leak would deflate teach EDL
# and inflate the elicit/teach ratio.
#
# ntfy fires once at terminal completion or failure. Set $NTFY to the full URL.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase3_teach.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
STAGE=all
PREV=""
for i in "$@"; do
  [[ $PREV == --stage ]] && STAGE=$i
  PREV=$i
done
case $STAGE in all | teach | target) ;;
*)
  echo "launch_phase3_teach.sh: --stage must be all|teach|target, got '$STAGE'" >&2
  exit 1
  ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
# Launchers run from experiments/training-run/scripts, so a fresh box without
# an editable install cannot otherwise import the repository's geode package.
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
INST_RID=evt-p3-teach-inst
TARGET_RID=evt-p3-teach-target
ANCHOR_RID=evt-p3-elicit-target
echo "[p3t] repo  $(git log --oneline -1)"
echo "[p3t] store $GEODE_STORE  stage=$STAGE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {
  notify "phase3 teach FAILED: $1"
  echo "[p3t] FAILED: $1" >&2
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

stop_reason_of() { # run_id result_key -> stop_reason string
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
r = json.loads(p.read_text()).get("experiment", {}).get(sys.argv[2], {}) or {}
print(r.get("stop_reason", ""))
PY
}

gate_done() { # run_id gate -> 0 if recorded
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

# Hash the artifacts this arm consumes and verify all role-scoped LR pins.
python3 phase3_guards.py --configs ../configs --repo-root "$REPO_ROOT" --scope teach || exit 1

if [[ $STAGE == all || $STAGE == teach ]]; then
  ST=$(status_of "$INST_RID")
  if [[ $ST == complete ]]; then
    echo "[p3t] $INST_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$INST_RID has status '$ST' — resolve it first"
    CKPT=$(ckpt_of evt-run1-base-v3-ext) || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for evt-run1-base-v3-ext (TinyStories floor-1)"
    echo "[p3t] === teach installer: permuted-label NL addition ==="
    python3 train_sft.py --config ../configs/p3_teach_inst.yaml \
      --init-from "$CKPT" --confirm-cost || fail "$INST_RID training"
  fi
  SR=$(stop_reason_of "$INST_RID" sft_result)
  [[ $SR == max_steps ]] && fail "$INST_RID stop_reason=max_steps — format never installed"
  [[ $SR == behavior ]] || fail "$INST_RID unexpected stop_reason='$SR' (expected behavior)"

  gate_done "$INST_RID" G4 || python3 gates.py g4 --run "$INST_RID" \
    --config ../configs/p3_teach_inst.yaml \
    --prompt-config ../configs/eval_p3_data.yaml --threshold 0.90 || fail "$INST_RID G4"
  gate_done "$INST_RID" G5 || python3 gates.py g5 --run "$INST_RID" \
    --config ../configs/eval_p3_data.yaml || fail "$INST_RID G5"

  python3 - <<'PY'
import json, os
from pathlib import Path

m = json.loads(
    (Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p3-teach-inst" / "manifest.json").read_text()
)
res = m["experiment"]["sft_result"]
bs = m["training"]["optimizer"]["batch_size"]
step = res["final_step"]
print(
    f"[p3t] teach installer exposure: {step} steps x batch {bs} = "
    f"{step * bs} examples seen ({res['stop_reason']}). COPY THIS INTO decisions.md"
)
PY
fi

if [[ $STAGE == all || $STAGE == target ]]; then
  [[ $(status_of "$INST_RID") == complete ]] || fail "$INST_RID is not complete — run --stage teach first"
  [[ $(status_of "$ANCHOR_RID") == complete ]] || fail "$ANCHOR_RID is not complete — G7 needs the elicit anchor"

  python3 - <<'PY' || fail "teach installer leak bar (G5 zero-shot > 0.02)"
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p3-teach-inst" / "manifest.json"
g5 = json.loads(p.read_text())["experiment"]["gates"].get("G5")
if g5 is None:
    sys.exit("phase3 teach: installer has no recorded G5 — score it before the target")
zero = g5["zero_shot_accuracy"]
if zero > 0.02:
    sys.exit(
        f"phase3 teach: installer G5 zero-shot {zero:.4f} > 0.02 — it learned real NL "
        "addition. Its target EDL would be deflated; STOP and investigate."
    )
print(f"[p3t] leak bar: teach installer G5 zero-shot {zero:.4f} <= 0.02")
PY

  INST_CKPT=$(ckpt_of "$INST_RID") || true
  [[ -n ${INST_CKPT:-} ]] || fail "no checkpoint for $INST_RID"
  ST=$(status_of "$TARGET_RID")
  if [[ $ST == complete ]]; then
    echo "[p3t] $TARGET_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$TARGET_RID has status '$ST' — resolve it first"
    echo "[p3t] === teach target: matched 500K NL-addition prequential EDL ==="
    python3 train_target.py --config ../configs/p3_elicit_target.yaml \
      --override ../configs/p3/target_teach.yaml \
      --init-from "$INST_CKPT" --confirm-cost || fail "$TARGET_RID training"
  fi
  SR=$(stop_reason_of "$TARGET_RID" target_result)
  [[ $SR == max_steps ]] && fail "$TARGET_RID stop_reason=max_steps — target did not converge"
  [[ $SR == converged ]] || fail "$TARGET_RID unexpected stop_reason='$SR' (expected converged)"

  gate_done "$TARGET_RID" G5 || python3 gates.py g5 --run "$TARGET_RID" \
    --config ../configs/eval_p3_data.yaml || fail "$TARGET_RID G5"
fi

REPORT=$(python3 - <<'PY'
import json, os
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
parts = []
for rid, key in (("evt-p3-teach-inst", "sft_result"), ("evt-p3-teach-target", "target_result")):
    p = store / "runs" / rid / "manifest.json"
    if not p.is_file():
        continue
    m = json.loads(p.read_text())
    res = m.get("experiment", {}).get(key, {})
    g5 = m.get("experiment", {}).get("gates", {}).get("G5", {})
    parts.append(
        f"{rid}: {res.get('stop_reason')}@{res.get('final_step')}, "
        f"G5 zero={g5.get('zero_shot_accuracy')}, test={g5.get('test_loss_nats')}"
    )
print(" | ".join(parts))
PY
)
echo "[p3t] REPORT: $REPORT"
notify "phase3 teach done — $REPORT"
echo "[p3t] compare within phase 3, n <= 500K, naming the floor:"
echo "[p3t]   python3 ../analysis/plot_edl_per_token.py --run-id $ANCHOR_RID --floor test"
echo "[p3t]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID --floor test"
echo "[p3t] stage '$STAGE' complete"
