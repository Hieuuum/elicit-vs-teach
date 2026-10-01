#!/usr/bin/env bash
# EAP-IG circuit check on one GPU box (PLAN.md). Order: sanity+score for all four
# models, then evaluate parents, then children (they need the parents' scores).
#
# Usage:  bash run_box.sh --confirm-cost [--only TAG]
# Env:    MODEL_<TAG> = HF repo id or local dir per model (defaults below),
#         RESULTS_REPO (default mhieuuu/geode-internals; results pushed under
#         results/eapig_check/), NO_PUSH=1 to skip the push.
# Skip-if-done per stage: a stage whose output file exists is not rerun.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0; ONLY=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --only) ONLY=$2; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
echo "[eapig] estimated GPU time ~2.5-3.5 h on one 24-80 GB GPU (~\$2-8 at \$0.5-2/h); box rental is the only cost."
((CONFIRM)) || { echo "[eapig] refusing to run without --confirm-cost" >&2; exit 2; }

declare -A MODEL=(
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [fmt_parent]=${MODEL_fmt_parent:-podhajskimarcin/evt-ts1b-fig2ts-installer}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
declare -A PARENT=([elicit_child]=elicit_parent [teach_child]=fmt_parent)
RES=$HERE/results
LOG=$RES/run_box.log
mkdir -p "$RES"
say() { echo "[eapig] $(date -Is) $*" | tee -a "$LOG"; }

say "repo $(git rev-parse --short HEAD) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ -f data/pairs_seed0.pt ]] || { say "FATAL: data/pairs_seed0.pt missing (build with data.py)"; exit 1; }

run() {  # run <tag> <stage> <done-file> [extra args]
  local tag=$1 st=$2 done=$3; shift 3
  [[ -n $ONLY && $ONLY != "$tag" ]] && return 0
  if [[ -f $RES/$tag/$done ]]; then say "skip $tag $st ($done exists)"; return 0; fi
  say "run $tag $st"
  local t0=$SECONDS
  python3 run.py --tag "$tag" --model "${MODEL[$tag]}" --stage "$st" "$@" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { say "FAILED $tag $st"; exit 1; }
  say "done $tag $st in $((SECONDS - t0)) s"
}

for tag in elicit_parent fmt_parent elicit_child teach_child; do
  run "$tag" sanity sanity.json
  run "$tag" score scores.pt
done
for tag in elicit_parent fmt_parent; do run "$tag" evaluate evaluate.json; done
for tag in elicit_child teach_child; do run "$tag" evaluate evaluate.json --parent-tag "${PARENT[$tag]}"; done

python3 compare.py 2>&1 | tee -a "$LOG" || say "compare.py failed (results intact)"

if [[ -z ${NO_PUSH:-} ]]; then
  RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
  say "push results -> $RESULTS_REPO:results/eapig_check/"
  HF_HUB_DISABLE_XET=1 python3 - "$RES" "$RESULTS_REPO" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi
res, repo = Path(sys.argv[1]), sys.argv[2]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo="results/eapig_check",
                  commit_message="eapig circuit check results")
remote = {f for f in api.list_repo_files(repo) if f.startswith("results/eapig_check/")}
local = {"results/eapig_check/" + str(p.relative_to(res)) for p in res.rglob("*") if p.is_file()}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none")
sys.exit(1 if missing else 0)
PY
fi
say "ALL DONE"
