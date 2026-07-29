#!/usr/bin/env bash
# PHASE 3 practical embedding warm-start family. Persist all 12 row-set/LR
# candidates, select one parent per row set without a retention gate, then train
# each selected parent through exactly the first 100K frozen NL-add examples.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase3_warmstart.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
ARM=all
STAGE=all
PUSH=0
PREV=""
for arg in "$@"; do
  [[ $PREV == --arm ]] && ARM=$arg
  [[ $PREV == --stage ]] && STAGE=$arg
  [[ $arg == --push ]] && PUSH=1
  PREV=$arg
done
case $ARM in all | sum | colon | sum-colon) ;;
*) echo "--arm must be all|sum|colon|sum-colon, got '$ARM'" >&2; exit 1 ;;
esac
case $STAGE in all | warmstart | target | push) ;;
*) echo "--stage must be all|warmstart|target|push, got '$STAGE'" >&2; exit 1 ;;
esac
[[ $STAGE != push || $PUSH == 1 ]] || {
  echo "launch_phase3_warmstart.sh: --stage push requires --push" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
PARENT_RID=evt-p3-elicit-parent
CONTROL_RID=evt-p3-elicit-target
PARENT_CKPT=$GEODE_STORE/runs/$PARENT_RID/model
LR_TAGS=(1e-3 1e-2 1e-1 1e0)
LR_VALUES=(0.001 0.01 0.1 1.0)

printf '[p3w] MILESTONE repo %s\n' "$(git log --oneline -1)"
echo "[p3w] MILESTONE store=$GEODE_STORE arm=$ARM stage=$STAGE push=$PUSH"

fail() { echo "[p3w] TERMINAL_FAILURE $1" >&2; exit 1; }

status_of() {
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

gate_done() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file(): sys.exit(1)
gates = json.loads(p.read_text()).get("experiment", {}).get("gates", {})
sys.exit(0 if sys.argv[2] in gates else 1)
PY
}

target_result_field() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {})
print(r.get(sys.argv[2], ""))
PY
}

select_arm() {
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path
from geode.train import select_warmstart_candidate

base = sys.argv[1]
store = Path(os.environ["GEODE_STORE"])
rows = []
for tag in ("1e-3", "1e-2", "1e-1", "1e0"):
    rid = f"{base}-lr{tag}"
    path = store / "runs" / rid / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete":
        raise SystemExit(f"{rid} status={manifest['status']!r}, expected complete")
    metrics = dict(manifest["experiment"]["embedding_warmstart"]["final"])
    metrics["run_id"] = rid
    rows.append(metrics)
selected = dict(select_warmstart_candidate(rows))
out = store / "selections" / f"{base}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"base_run_id": base, "selection_rule": "maximize held-out NL exact match; minimize held-out NL NLL; then smaller LR; operator retention report-only", "candidates": rows, "selected": selected}, indent=2) + "\n")
print(selected["run_id"])
PY
}

push_run() {
  local rid=$1
  [[ $(status_of "$rid") == complete ]] || fail "cannot push $rid: not complete"
  echo "[p3w] MILESTONE push_start run=$rid"
  python3 hf_checkpoint.py push --run-id "$rid" || fail "$rid HF push"
  python3 - "$rid" <<'PY' || exit 1
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
from hf_checkpoint import sha256_of

rid = sys.argv[1]
store = Path(os.environ["GEODE_STORE"])
local_path = store / "runs" / rid / "model" / "model.safetensors"
repo_path = f"runs/{rid}/model/model.safetensors"
info = HfApi().get_paths_info("mhieuuu/geode-store", [repo_path])
remote = info[0].lfs.sha256 if info and info[0].lfs else None
local = sha256_of(local_path)
if remote != local:
    raise SystemExit(f"{rid}: hub sha256 {remote!r} != local {local}")
print(f"[p3w] MILESTONE push_verified run={rid} sha256={local}")
PY
}

python3 phase3_guards.py --configs ../configs --repo-root "$REPO_ROOT" --scope warmstart || exit 1
[[ -f $PARENT_CKPT/model.safetensors ]] || fail "no checkpoint for $PARENT_RID"

arms=(sum colon sum-colon)
[[ $ARM != all ]] && arms=($ARM)
completed_targets=()
selected_runs=()

if [[ $STAGE == all || $STAGE == warmstart ]]; then
  for arm in "${arms[@]}"; do
    case $arm in
      sum) base_rid=evt-p3-warm-sum; warm_cfg=warm_sum.yaml ;;
      colon) base_rid=evt-p3-warm-colon; warm_cfg=warm_colon.yaml ;;
      sum-colon) base_rid=evt-p3-warm-sum-colon; warm_cfg=warm_sum_colon.yaml ;;
    esac
    for i in "${!LR_TAGS[@]}"; do
      tag=${LR_TAGS[$i]}
      lr=${LR_VALUES[$i]}
      rid=$base_rid-lr$tag
      status=$(status_of "$rid")
      if [[ $status == complete ]]; then
        echo "[p3w] MILESTONE candidate_skip run=$rid status=complete"
      elif [[ $status == missing ]]; then
        echo "[p3w] MILESTONE candidate_start run=$rid arm=$arm lr=$lr"
        python3 train_embedding_warmstart.py \
          --config ../configs/p3_embedding_warmstart.yaml \
          --override "../configs/p3/$warm_cfg" --lr "$lr" \
          --init-from "$PARENT_CKPT" --confirm-cost || fail "$rid training"
        [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
        echo "[p3w] MILESTONE candidate_complete run=$rid"
      else
        fail "$rid exists with status '$status'; inspect it rather than overwriting"
      fi
    done
    selected=$(select_arm "$base_rid") || fail "$arm candidate selection"
    selected=${selected##*$'\n'}
    selected_runs+=("$selected")
    echo "[p3w] MILESTONE selection arm=$arm selected=$selected"
  done
fi

if [[ $STAGE == all || $STAGE == target ]]; then
  # G7 uses sum as the 100K anchor. Refuse a partial target-only call that would
  # launch a matched child without the anchor; all-mode naturally runs sum first.
  if [[ $ARM != all && $ARM != sum && $(status_of evt-p3-warm-sum-target) != complete ]]; then
    fail "$ARM target needs complete 100K anchor evt-p3-warm-sum-target"
  fi
  for arm in "${arms[@]}"; do
    case $arm in
      sum) base_rid=evt-p3-warm-sum; target_rid=evt-p3-warm-sum-target; target_cfg=target_warm_sum.yaml ;;
      colon) base_rid=evt-p3-warm-colon; target_rid=evt-p3-warm-colon-target; target_cfg=target_warm_colon.yaml ;;
      sum-colon) base_rid=evt-p3-warm-sum-colon; target_rid=evt-p3-warm-sum-colon-target; target_cfg=target_warm_sum_colon.yaml ;;
    esac
    selected=$(select_arm "$base_rid") || fail "$arm candidate selection"
    selected=${selected##*$'\n'}
    selected_runs+=("$selected")
    warm_ckpt=$GEODE_STORE/runs/$selected/model
    [[ -f $warm_ckpt/model.safetensors ]] || fail "no checkpoint for selected $selected"
    status=$(status_of "$target_rid")
    if [[ $status == complete ]]; then
      echo "[p3w] MILESTONE target_skip run=$target_rid status=complete"
    elif [[ $status == missing ]]; then
      echo "[p3w] MILESTONE target_start run=$target_rid parent=$selected n=100000 steps=782"
      python3 train_target.py --config ../configs/p3_elicit_target.yaml \
        --override "../configs/p3/$target_cfg" --parent-run-id "$selected" \
        --init-from "$warm_ckpt" --confirm-cost || fail "$target_rid training"
    else
      fail "$target_rid exists with status '$status'; inspect it rather than overwriting"
    fi
    reason=$(target_result_field "$target_rid" stop_reason)
    step=$(target_result_field "$target_rid" final_step)
    [[ $reason == max_steps && $step == 782 ]] || \
      fail "$target_rid ended reason='$reason' step='$step'; expected max_steps/782"
    if ! gate_done "$target_rid" G5; then
      echo "[p3w] MILESTONE target_g5_start run=$target_rid"
      python3 gates.py g5 --run "$target_rid" --config ../configs/eval_p3_data.yaml \
        || fail "$target_rid G5 evidence"
    fi
    completed_targets+=("$target_rid")
    echo "[p3w] MILESTONE target_complete run=$target_rid parent=$selected"
  done
fi

if ((PUSH)); then
  push_arms=(sum colon sum-colon)
  [[ $ARM != all ]] && push_arms=($ARM)
  for arm in "${push_arms[@]}"; do
    case $arm in
      sum) base_rid=evt-p3-warm-sum; target_rid=evt-p3-warm-sum-target ;;
      colon) base_rid=evt-p3-warm-colon; target_rid=evt-p3-warm-colon-target ;;
      sum-colon) base_rid=evt-p3-warm-sum-colon; target_rid=evt-p3-warm-sum-colon-target ;;
    esac
    for tag in "${LR_TAGS[@]}"; do push_run "$base_rid-lr$tag"; done
    if [[ $STAGE != warmstart ]]; then push_run "$target_rid"; fi
  done
fi

echo "[p3w] MILESTONE analysis_commands control=$CONTROL_RID targets=${completed_targets[*]:-none}"
echo "[p3w]   python3 ../analysis/plot_edl_per_token.py --floor test --per example --min-examples 0 --run-id $CONTROL_RID [--run-id ...]"
echo "[p3w]   python3 ../analysis/plot_edl_per_token.py --floor val  --per example --min-examples 0 --run-id $CONTROL_RID [--run-id ...]"
echo "[p3w] TERMINAL_SUCCESS stage=$STAGE arms=${arms[*]} pushed=$PUSH"
