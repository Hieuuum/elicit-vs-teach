#!/usr/bin/env bash
# ts38mt Tier-3 grid mechanistic tests (handoff §5b): tests 3/5/2
# (node_edge_delta, dcm, circuit_jaccard). ONLY run this if the Tier-1
# gate opened (Phase-0 resid_probe found a latent sum at ts38pp-parent
# theta0) -- decisions.md 2026-08-21 night "ts38mt pre-registration".
# Idempotent.
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
    out_ne="$O/node_edge_delta_$R.csv"
    if [[ ! -f "$out_ne" ]]; then
      python3 node_edge_delta.py --model-a "dir:$S/$P" --model-b "run:$R" --prompt-parquet "$T" --device cuda --limit 1024 --batch-size 1024 --out "$out_ne"
    fi
    out_dcm="$O/dcm_$R.csv"
    if [[ ! -f "$out_dcm" ]]; then
      python3 dcm.py --model-a "dir:$S/$P" --model-b "run:$R" --prompt-parquet "$T" --device cuda --limit 512 --lambdas 0.001,0.01,0.1 --steps 200 --out "$out_dcm"
    fi
    echo "[t3] MILESTONE tier3_cell_done run=$R elapsed_s=$(( $(date +%s) - t0 ))"
  done
  out_cj="$O/circuit_jaccard_n$N.csv"
  if [[ ! -f "$out_cj" ]]; then
    python3 circuit_jaccard.py \
      --model base=dir:$S/evt-run1-base-v3-ext/model --model pp0=dir:$S/evt-ts38pp-parent/model --model ppT=run:evt-ts38mt-pp-n$N \
      --model fmt0=dir:$S/evt-ts38mt-fmt-parent/model --model fmtT=run:evt-ts38mt-fmt-n$N --model baseT=run:evt-ts38mt-base-n$N \
      --prompt-parquet "$T" --device cuda --limit 1024 --batch-size 1024 --out "$out_cj"
  fi
  echo "[t3] MILESTONE circuit_jaccard_done n=$N"
done
echo "[t3] MILESTONE tier3_grid_complete sizes=\"$SIZES\""
