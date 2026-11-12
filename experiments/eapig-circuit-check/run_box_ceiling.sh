#!/usr/bin/env bash
# run_box_ceiling.sh — split-half ceilings of 7 ts1b models on the 4,096-problem
# 4x4 addition set balanced over the 16 carry patterns (data/ceiling4x4, built by
# build_folds.py --strata pattern --balanced; owner 2026-11-12).
#
# Per model: fold_runner.py sanity + score (EAP-IG, KL metric, word task) on
# data/ceiling4x4/fold_00, then ceiling_report.py (Jaccard of the two 2,048-problem
# halves' top-k edges) and a push to $RESULTS_REPO:results/ceiling4x4 (no
# per_example.pt). One lane per GPU.
#
# Usage: bash run_box_ceiling.sh --confirm-cost      (NO_PUSH=1 to skip the push)
# Ends with EXIT=0 (all 7 scored, report written, pushed) or EXIT=1.
set -uo pipefail
cd "$(dirname "$0")"
echo "[ceiling] estimated cost: 7 models x ~10 min (4,096 problems each) over the box's GPUs;"
echo "[ceiling] 2 GPUs ~40 min at ~\$0.3-0.8/h ~ \$0.5. Nothing else runs."
[[ " $* " == *" --confirm-cost "* ]] || { echo "run_box_ceiling.sh: --confirm-cost required" >&2; exit 1; }

export PYTHONPATH=$(git rev-parse --show-toplevel) HF_HUB_DISABLE_XET=1
ROOT=results/ceiling4x4
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
BS=${BATCH_SIZE:-32}
[[ -f data/ceiling4x4/fold_00/pairs_seed0.pt ]] || { echo "missing data/ceiling4x4" >&2; exit 1; }

# label|tag|model|subdir  (tag picks run.py's guards; subdir keeps the children apart)
JOBS=(
  "ec_acc|elicit_child|mhieuuu/evt-ts1b-elicit-hard-acc|klword_hard_acc"
  "tc_acc|teach_child|mhieuuu/evt-ts1b-teach-hard-acc|klword_hard_acc"
  "ec_loss|elicit_child|mhieuuu/evt-ts1b-elicit-hard-loss|klword_hard_loss"
  "tc_loss|teach_child|mhieuuu/evt-ts1b-teach-hard-loss|klword_hard_loss"
  "elicit_parent|elicit_parent|podhajskimarcin/evt-ts1b-op-bridge-mix|klword"
  "fmt_parent|fmt_parent|mhieuuu/evt-ts1b-fmt-parent|klword"
  "base|base|podhajskimarcin/evt-ts1b-base|klword"
)
NGPU=$(nvidia-smi -L | wc -l)
say() { echo "[ceiling $(date +%H:%M:%S)] $*"; }

run_job() { # job gpu
  local label tag model sub
  IFS='|' read -r label tag model sub <<<"$1"
  say "start $label on gpu $2"
  if CUDA_VISIBLE_DEVICES=$2 python3 fold_runner.py --tag "$tag" --model "$model" --task word \
      --metric kl --stage sanity score --batch-size "$BS" --folds-root data/ceiling4x4 --folds 0 \
      --results-root "$ROOT" --subdir "$sub" >"$LOG_DIR/$label.log" 2>&1; then
    say "MILESTONE done $label ($(grep -o 'done in [0-9]*s EM [0-9.]*' "$LOG_DIR/$label.log" | tail -1))"
  else
    say "FAILED $label (see $LOG_DIR/$label.log)"; return 1
  fi
}
lane() { local g=$1 i bad=0
  for ((i = g; i < ${#JOBS[@]}; i += NGPU)); do run_job "${JOBS[$i]}" "$g" || bad=1; done
  return $bad
}

say "$NGPU GPU(s), batch $BS"
pids=()
for ((g = 0; g < NGPU; g++)); do lane "$g" & pids+=($!); done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done

# fold_runner writes <root>/fold_00/<subdir>/<tag>; the report takes paths under <root>
args=()
for j in "${JOBS[@]}"; do IFS='|' read -r label tag _ sub <<<"$j"; args+=("$label=fold_00/$sub/$tag"); done
if python3 ceiling_report.py --root "$ROOT" "${args[@]}" --out "$ROOT/ceiling.json" | tee "$ROOT/ceiling.md"; then
  say "MILESTONE report written"
else
  say "FAILED report"; bad=1
fi

if [[ -z ${NO_PUSH:-} ]]; then
  cp data/ceiling4x4/manifest.json "$ROOT/data_manifest.json"
  if python3 - "$ROOT" "$RESULTS_REPO" <<'PY'; then say "MILESTONE pushed"; else say "FAILED push"; bad=1; fi
import sys
from fnmatch import fnmatch
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], "results/ceiling4x4"
ign = ["*per_example.pt"]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="ts1b split-half ceilings, 4x4 carry-pattern set", ignore_patterns=ign)
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + r for r in (str(p.relative_to(res)) for p in res.rglob("*") if p.is_file())
         if not any(fnmatch(r, g) for g in ign)}
print("verify: missing on hub:", sorted(local - remote) or "none", "| files:", len(local))
sys.exit(1 if local - remote else 0)
PY
fi
echo "EXIT=$bad"
