#!/usr/bin/env bash
# Pre-fine-tuning predictors on the five parents of the construction ladder.
#
# Question: can the PARENT alone (plus labelled task data) tell whether the coming
# fine-tune will elicit a latent capability or teach an absent one?  Every
# metric in analysis/prefit_metrics.py reads the parent only; no fine-tuned
# checkpoint is involved.  The five parents have known learning-curve shapes
# (dataset-size sweeps / small-n runs), which is the ground truth a predictor
# must order correctly:
#
#   tag      parent                        curve on the NL target
#   blank    evt-ts1b-base                 teach (hump; EM 0.09 at 1M under LoRA)
#   fmt      evt-ts1b-fig2ts-installer     teach (hump with the format prepaid)
#   engine   evt-ts1b-op-install           symbol engine, no word binding: op-parent >>
#                                          blank at n<=1000 (op program), binding index ~0
#   latent   evt-ts1b-op-bridge-mix        elicit (monotone; EM 0.54 at 3,162)
#   llama    meta-llama/Llama-3.2-1B       elicit (monotone Fig-2 noinst sweep)
#
# Metrics (each appends a block to analysis/prefit_<tag>.json; skip-if-done):
#   pref geometry probe das dcm attn grad hessian llc   (see the script docstring)
#
# Usage:  bash launch_prefit.sh --confirm-cost [--tags "latent fmt"] [--metrics "pref probe"]
# Env:    GEODE_STORE (store root); conda env geode.  GPU: fp32 1B, ~20 GB peak.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$(git rev-parse --show-toplevel)
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
A=$REPO_ROOT/experiments/training-run/analysis
LOG=$A/prefit.log

CONFIRM=0; TAGS="latent fmt blank engine llama"; METRICS="pref geometry attn probe das dcm grad hessian llc"
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --tags) TAGS=$2; shift ;;
    --metrics) METRICS=$2; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
[[ $CONFIRM == 1 ]] || { echo "GPU job (~1-2 h for five parents). Re-run with --confirm-cost." >&2; exit 2; }

declare -A MODEL=( [blank]=evt-ts1b-base [fmt]=evt-ts1b-fig2ts-installer [engine]=evt-ts1b-op-install
                   [latent]=evt-ts1b-op-bridge-mix [llama]=meta-llama/Llama-3.2-1B )

milestone() { echo "[prefit] MILESTONE $*" | tee -a "$LOG"; }
has_block() {  # has_block <tag> <metric>: 0 if prefit_<tag>.json already holds the block
  python3 - "$A/prefit_$1.json" "$2" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
sys.exit(0 if p.is_file() and sys.argv[2] in json.loads(p.read_text()) else 1)
PY
}

milestone "repo $(git rev-parse --short HEAD) store=$GEODE_STORE tags=[$TAGS] metrics=[$METRICS]"
cd "$A"
for tag in $TAGS; do
  model=${MODEL[$tag]:?unknown tag $tag}
  if [[ $model != */* && ! -d $GEODE_STORE/runs/$model/model ]]; then
    milestone "skip $tag: $GEODE_STORE/runs/$model/model missing"; continue
  fi
  for m in $METRICS; do
    if has_block "$tag" "$m"; then milestone "skip ($tag::$m done)"; continue; fi
    milestone "run: prefit_metrics.py $m --model $model --tag $tag"
    { python3 prefit_metrics.py "$m" --model "$model" --tag "$tag" 2>&1 | tee -a "$LOG"; } \
      && [[ ${PIPESTATUS[0]} == 0 ]] || milestone "FAILED $tag::$m (continuing; see $LOG)"
  done
done
milestone "compare"
python3 prefit_metrics.py compare $TAGS 2>&1 | tee -a "$LOG"
milestone "done — paste $LOG"
