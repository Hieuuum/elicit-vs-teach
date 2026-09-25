#!/usr/bin/env bash
# Same-base elicitation sweep (owner 2026-08-31): fine-tune the pre-elicit
# parent evt-ts1b-op-bridge-mix on the bare-NL target at 9 log-spaced sizes,
# G5 after each. The blank-TS teaching curve (19 sizes) is already measured;
# only the parent differs. Predictions: monotone-decreasing EDL/token, EM
# unlocking at small n. Also backfills the pending G5s (op-nl n1000, blank
# n100/n316). Resumable: completed runs/gates are skipped.
# Cost: ~4-8 h GPU (elicit arms converge early; ceilings carry headroom).
# Usage: bash launch_ts1b_mix_nl.sh --confirm-cost
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=mix-nl

echo "[mix-nl] estimated cost: ~4-8 h GPU; no HF uploads"
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts1b_mix_nl.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
CONFIGS=../configs
MIX_MODEL=$GEODE_STORE/runs/evt-ts1b-op-bridge-mix/model
SIZES=(100 316 1000 3162 10000 31623 100000 316228 1000000)

milestone "repo $(git log --oneline -1)"
[[ -f $MIX_MODEL/model.safetensors ]] ||
  fail "pre-elicit parent $MIX_MODEL missing"

train_or_skip() {
  local rid=$1 status
  shift
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"
    return 0
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  fi
  milestone "train_start run=$rid"
  "$@" || fail "$rid training"
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_complete run=$rid"
}

g5_or_skip() {
  local rid=$1
  if python3 -c "
import json, os, sys
p = os.path.join(os.environ['GEODE_STORE'], 'runs', sys.argv[1], 'manifest.json')
g = json.load(open(p)).get('experiment', {}).get('gates', {})
sys.exit(0 if 'G5' in g else 1)
" "$rid" 2>/dev/null; then
    milestone "g5_skip run=$rid (already recorded)"
    return 0
  fi
  python3 gates.py g5 --run "$rid" \
    --config ../configs/eval_bare_target_data_llama.yaml ||
    milestone "g5 FAILED for $rid (non-fatal; rerun manually)"
}

for n in "${SIZES[@]}"; do
  rid="evt-ts1b-mix-nl-n${n}"
  train_or_skip "$rid" \
    python3 train_target.py --config $CONFIGS/ts1b_mix_nl.yaml \
      --override $CONFIGS/sweeps/ts1b_mix_nl/ts1b_mix_nl_n${n}.yaml \
      --init-from "$MIX_MODEL" --confirm-cost
  g5_or_skip "$rid"
done

# backfill the pending G5s from the op-install premise program
for rid in evt-ts1b-op-nl-n1000 evt-ts1b-fig2ts-noinst-n100 evt-ts1b-fig2ts-noinst-n316; do
  g5_or_skip "$rid"
done

milestone "TERMINAL_SUCCESS mix-nl sweep done — EDL per rung is in each manifest" \
  " (target_result.edl_per_label_token_nats); paste manifests or run a collect for the curve"
