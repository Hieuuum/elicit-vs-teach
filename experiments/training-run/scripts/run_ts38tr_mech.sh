#!/usr/bin/env bash
# ts38tr mechanistic tests -- same five Phase-0 tests (run_ts38mt_phase0.sh)
# on the truncated-adapter parents (evt-ts38tr-k6-parent / -k7-parent) plus
# the SOURCE run they were sliced from (evt-ts38mt-base-n316228, as
# "source_thetaT"), then the Tier-1+2 grid tests (run_ts38mt_grid_tier1/2.sh)
# on every existing evt-ts38tr-k*-n* target run. No new training here.
# Idempotent: skips any --out that already exists. Runs SERIALLY (a 24GB
# card OOMs with two analysis scripts at once -- ts38mt lesson).
set -euo pipefail
source /workspace/venv/bin/activate
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
cd /workspace/elicit-vs-teach/experiments/training-run/analysis
shopt -s nullglob
S=$GEODE_STORE/runs
O=$GEODE_STORE/results/ts38tr_mech
mkdir -p "$O"
TASK=../data/full/D_algo_eval_bare.parquet
OP=../data/full/probe.parquet
GENERIC=/workspace/tinystories_val_2000.txt
BASE="dir:$S/evt-run1-base-v3-ext/model"
SOURCE_RID=evt-ts38mt-base-n316228
THETA0_JSON=$GEODE_STORE/results/ts38tr_family_theta0.json

echo "[trm] ===== prep: generic text ====="
if [[ ! -f "$GENERIC" ]]; then
  python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="validation")
with open("/workspace/tinystories_val_2000.txt", "w") as f:
    for r in ds.select(range(2000)): f.write(r["text"].replace("\n", " ").strip() + "\n")
PY
fi

echo "[trm] ===== stage A: parents (Phase-0 style) + source theta_T ====="
MODELS=()
for PDIR in "$S"/evt-ts38tr-k*-parent; do
  [[ -d $PDIR ]] || continue
  b=$(basename "$PDIR")   # evt-ts38tr-k6-parent
  K=${b#evt-ts38tr-k}
  K=${K%-parent}
  MODELS+=("k${K}_parent:dir:$PDIR/model")
done

if [[ -f $S/$SOURCE_RID/model/adapter.safetensors || -f $S/$SOURCE_RID/model/model.safetensors ]]; then
  MODELS+=("source_thetaT:run:$SOURCE_RID")
else
  echo "[trm] WARN: $SOURCE_RID has no local weights (this script has no pull stage) -- skipping source_thetaT"
fi

if ((${#MODELS[@]} == 0)); then
  echo "[trm] WARN: no evt-ts38tr-k*-parent dirs found under $S -- stage A has nothing to do"
fi

for M in "${MODELS[@]}"; do
  name=${M%%:*}; spec=${M#*:}

  out_rp="$O/resid_probe_${name}.csv"
  if [[ -f $out_rp ]]; then
    echo "[trm] skip resid_probe $name (exists)"
  else
    python3 resid_probe.py --model "$spec" --model-name "$name" \
      --prompt-parquet "$TASK" --set-name task --prompt-parquet "$OP" --set-name op \
      --device cuda --limit 2000 --out "$out_rp"
    echo "[trm] MILESTONE resid_probe_done model=$name"
  fi

  out_ll_task="$O/logit_lens_${name}_task.csv"
  out_ll_op="$O/logit_lens_${name}_op.csv"
  [[ -f $out_ll_task ]] || python3 logit_lens.py --model "$spec" --model-name "$name" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out_ll_task"
  [[ -f $out_ll_op ]] || python3 logit_lens.py --model "$spec" --model-name "$name" --prompt-parquet "$OP" --set-name op --device cuda --limit 1024 --out "$out_ll_op"
  echo "[trm] MILESTONE logit_lens_done model=$name"

  out_wd="$O/weight_diff_${name}.parquet"
  [[ -f $out_wd ]] || python3 weight_diff.py --model-a "$BASE" --model-b "$spec" --device cuda --out "$out_wd"

  out_rs="$O/resid_shift_${name}.csv"
  [[ -f $out_rs ]] || python3 resid_shift.py --model-a "$BASE" --model-b "$spec" --task-parquet "$TASK" --generic-local "$GENERIC" --device cuda --limit 2000 --out "$out_rs"
  echo "[trm] MILESTONE weight_diff_resid_shift_done model=$name"

  out_jl="$O/jacobian_lens_${name}.csv"
  [[ -f $out_jl ]] || python3 jacobian_lens.py --model-a "$BASE" --model-b "$spec" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out_jl"
  echo "[trm] MILESTONE jacobian_lens_done model=$name"
done
echo "[trm] MILESTONE stageA_complete models=${#MODELS[@]}"

echo "[trm] ===== stage B: targets (grad_dynamics + Tier1/2 per run) ====="
RUN_ID_ARGS=()
TARGET_RUNS=()
for RDIR in "$S"/evt-ts38tr-k*-n*; do
  [[ -d $RDIR ]] || continue
  rid=$(basename "$RDIR")
  if [[ ! -f $RDIR/manifest.json ]]; then
    echo "[trm] skip $rid (no manifest.json)"
    continue
  fi
  RUN_ID_ARGS+=(--run-id "$rid")
  TARGET_RUNS+=("$rid")
done

GD_OUT="$O/grad_dynamics_all.csv"
if ((${#TARGET_RUNS[@]} == 0)); then
  echo "[trm] WARN: no evt-ts38tr-k*-n* runs with a manifest found under $S -- stage B has nothing to do"
elif [[ -f $GD_OUT ]]; then
  echo "[trm] skip grad_dynamics (exists)"
else
  t0=$(date +%s)
  python3 grad_dynamics.py "${RUN_ID_ARGS[@]}" --out "$GD_OUT"
  echo "[trm] MILESTONE grad_dynamics_done elapsed_s=$(( $(date +%s) - t0 )) n_runs=${#TARGET_RUNS[@]}"
fi

for R in "${TARGET_RUNS[@]}"; do
  if [[ $R =~ ^evt-ts38tr-k([0-9]+)-n[0-9]+$ ]]; then
    K=${BASH_REMATCH[1]}
  else
    echo "[trm] WARN: $R does not match evt-ts38tr-k<K>-n<N> -- skipping"
    continue
  fi
  PDIR="$S/evt-ts38tr-k${K}-parent/model"
  if [[ ! -d $PDIR ]]; then
    echo "[trm] WARN: no parent dir $PDIR for $R -- skipping"
    continue
  fi
  P="dir:$PDIR"
  t0=$(date +%s)

  out_rp="$O/resid_probe_$R.csv"
  if [[ ! -f $out_rp ]]; then
    python3 resid_probe.py --run-id "$R" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out_rp"
  fi

  out_wd="$O/weight_diff_$R.parquet"
  if [[ ! -f $out_wd ]]; then
    python3 weight_diff.py --model-a "$P" --model-b "run:$R" --device cuda --out "$out_wd"
  fi

  out_rs="$O/resid_shift_$R.csv"
  if [[ ! -f $out_rs ]]; then
    python3 resid_shift.py --model-a "$P" --model-b "run:$R" --task-parquet "$TASK" --generic-local "$GENERIC" --device cuda --limit 2000 --out "$out_rs"
  fi

  out_jl="$O/jacobian_lens_$R.csv"
  if [[ ! -f $out_jl ]]; then
    python3 jacobian_lens.py --model-a "$P" --model-b "run:$R" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out_jl"
  fi

  out_cp="$O/cross_patch_$R.csv"
  if [[ ! -f $out_cp ]]; then
    python3 cross_patch.py --model-a "$P" --model-b "run:$R" --prompt-parquet "$TASK" --device cuda --limit 1000 --out "$out_cp"
  fi

  echo "[trm] MILESTONE stageB_run_done run=$R elapsed_s=$(( $(date +%s) - t0 ))"
done
echo "[trm] MILESTONE stageB_complete runs=${#TARGET_RUNS[@]}"

echo "[trm] ===== stage C: upload results/ts38tr_mech (+ theta0 json) to geode-internals ====="
python3 - "$O" "$THETA0_JSON" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi

folder, theta0_json = sys.argv[1], Path(sys.argv[2])
api = HfApi()
api.upload_folder(
    folder_path=folder,
    path_in_repo="results/ts38tr_mech",
    repo_id="mhieuuu/geode-internals",
    repo_type="model",
)
print("[trm] uploaded results/ts38tr_mech")
if theta0_json.is_file():
    api.upload_file(
        path_or_fileobj=str(theta0_json),
        path_in_repo="results/ts38tr_family_theta0.json",
        repo_id="mhieuuu/geode-internals",
        repo_type="model",
    )
    print("[trm] uploaded results/ts38tr_family_theta0.json")
else:
    print(f"[trm] {theta0_json} not found locally -- skipped (run_ts38tr_family.sh not run on this box?)")
PY

echo "[trm] MILESTONE ts38tr_mech_complete"
