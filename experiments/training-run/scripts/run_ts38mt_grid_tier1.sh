#!/usr/bin/env bash
# ts38mt Tier-1 grid mechanistic tests (handoff §5b): tests 1/8/9/10
# (resid_probe, grad_dynamics, weight_diff, resid_shift) for every
# arm x size. Idempotent (skips existing --out). SIZES overridable via
# env var, space-separated, default = full 10-point grid.
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

echo "[t1] ===== stage A: grad_dynamics (test 8, no GPU, run once per size for all 3 arms' target runs) ====="
RUN_ID_ARGS=()
n_runs=0
for N in $SIZES; do
  for A in $ARMS; do
    RUN_ID_ARGS+=(--run-id "evt-ts38mt-${A}-n${N}")
    n_runs=$((n_runs + 1))
  done
done
GD_OUT="$O/grad_dynamics_all.csv"
if [[ -f "$GD_OUT" ]]; then
  echo "[t1] skip grad_dynamics (exists)"
else
  t0=$(date +%s)
  python3 grad_dynamics.py "${RUN_ID_ARGS[@]}" --out "$GD_OUT"
  echo "[t1] MILESTONE grad_dynamics_done elapsed_s=$(( $(date +%s) - t0 )) n_runs=$n_runs"
fi

echo "[t1] ===== stage B: resid_probe / weight_diff / resid_shift per (arm, size) ====="
for N in $SIZES; do
  for A in $ARMS; do
    R="evt-ts38mt-${A}-n${N}"
    P="${PARENT_DIR[$A]}"
    t0=$(date +%s)
    out_rp="$O/resid_probe_$R.csv"
    if [[ ! -f "$out_rp" ]]; then
      python3 resid_probe.py --run-id "$R" --prompt-parquet "$T" --set-name task --device cuda --limit 2000 --out "$out_rp"
    fi
    out_wd="$O/weight_diff_$R.parquet"
    if [[ ! -f "$out_wd" ]]; then
      python3 weight_diff.py --model-a "dir:$S/$P" --model-b "run:$R" --device cuda --out "$out_wd"
    fi
    out_rs="$O/resid_shift_$R.csv"
    if [[ ! -f "$out_rs" ]]; then
      python3 resid_shift.py --model-a "dir:$S/$P" --model-b "run:$R" --task-parquet "$T" --generic-local /workspace/tinystories_val_2000.txt --device cuda --limit 2000 --out "$out_rs"
    fi
    echo "[t1] MILESTONE tier1_cell_done run=$R elapsed_s=$(( $(date +%s) - t0 ))"
  done
done

echo "[t1] MILESTONE tier1_grid_complete sizes=\"$SIZES\""
