#!/usr/bin/env bash
# ts38mt Tier-2 grid mechanistic tests (handoff §5b): tests 7/4
# (jacobian_lens, cross_patch), ungated, per arm x size. Idempotent.
set -euo pipefail
source /workspace/venv/bin/activate
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
cd /workspace/elicit-vs-teach/experiments/training-run/analysis
S=$GEODE_STORE/runs
O=$GEODE_STORE/results/ts38mt_mech
mkdir -p "$O"
T=../data/full/D_algo_eval_bare.parquet
SIZES=${SIZES:-"1000 2154 4642 10000 21544 46416 100000 146780 215443 316228"}
ARMS="base pp fmt"
declare -A PARENT_DIR=(
  [base]="evt-run1-base-v3-ext/model"
  [pp]="evt-ts38pp-parent/model"
  [fmt]="evt-ts38mt-fmt-parent/model"
)

for N in $SIZES; do
  for A in $ARMS; do
    R="evt-ts38mt-${A}-n${N}"
    P="${PARENT_DIR[$A]}"
    t0=$(date +%s)
    out_jl="$O/jacobian_lens_$R.csv"
    if [[ ! -f "$out_jl" ]]; then
      python3 jacobian_lens.py --model-a "dir:$S/$P" --model-b "run:$R" --prompt-parquet "$T" --set-name task --device cuda --limit 2000 --out "$out_jl"
    fi
    out_cp="$O/cross_patch_$R.csv"
    if [[ ! -f "$out_cp" ]]; then
      python3 cross_patch.py --model-a "dir:$S/$P" --model-b "run:$R" --prompt-parquet "$T" --device cuda --limit 1000 --out "$out_cp"
    fi
    echo "[t2] MILESTONE tier2_cell_done run=$R elapsed_s=$(( $(date +%s) - t0 ))"
  done
done
echo "[t2] MILESTONE tier2_grid_complete sizes=\"$SIZES\""
