#!/usr/bin/env bash
# ts38mt Phase-0 mechanistic analysis (handoff-ts38mt-launch.md §5).
# No new training. theta0=evt-run1-base-v3-ext, thetaT=evt-ts38pp-parent
# (+3 sft_snapshots). Order: resid_probe first (the gate + headline
# question), then the rest. Idempotent: skips any --out that already
# exists, so a dropped SSH session just needs a re-run.
set -euo pipefail
source /workspace/venv/bin/activate
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
cd /workspace/elicit-vs-teach/experiments/training-run/analysis
S=$GEODE_STORE/runs
O=$GEODE_STORE/results/ts38mt_phase0
mkdir -p "$O"
TASK=../data/full/D_algo_eval_bare.parquet
OP=../data/full/probe.parquet
GENERIC=/workspace/tinystories_val_2000.txt

echo "[p0] ===== prep: generic text + parent snapshots ====="
if [[ ! -f "$GENERIC" ]]; then
  python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="validation")
with open("/workspace/tinystories_val_2000.txt", "w") as f:
    for r in ds.select(range(2000)): f.write(r["text"].replace("\n", " ").strip() + "\n")
PY
fi
if [[ ! -d "$S/evt-ts38pp-parent/sft_snapshots" ]]; then
  python3 ../scripts/hf_checkpoint.py pull --run-id evt-ts38pp-parent --with-snapshots
fi
STEP7773=$(ls -d "$S"/evt-ts38pp-parent/sft_snapshots/step_* | sort | sed -n '1p')
STEP15546=$(ls -d "$S"/evt-ts38pp-parent/sft_snapshots/step_* | sort | sed -n '2p')
STEP23319=$(ls -d "$S"/evt-ts38pp-parent/sft_snapshots/step_* | sort | sed -n '3p')
echo "[p0] snapshots: $STEP7773 | $STEP15546 | $STEP23319"

MODELS=(
  "theta0:dir:$S/evt-run1-base-v3-ext/model"
  "s7773:dir:$STEP7773"
  "s15546:dir:$STEP15546"
  "s23319:dir:$STEP23319"
  "thetaT:dir:$S/evt-ts38pp-parent/model"
)

echo "[p0] ===== stage 1/5: resid_probe (test 1, THE GATE) ====="
for M in "${MODELS[@]}"; do
  name=${M%%:*}; spec=${M#*:}
  out="$O/resid_probe_${name}.csv"
  if [[ -f "$out" ]]; then echo "[p0] skip resid_probe $name (exists)"; continue; fi
  python3 resid_probe.py --model "$spec" --model-name "$name" \
    --prompt-parquet "$TASK" --set-name task --prompt-parquet "$OP" --set-name op \
    --device cuda --limit 2000 --out "$out"
  echo "[p0] MILESTONE resid_probe_done model=$name"
done

echo "[p0] ===== stage 2/5: logit_lens (test 6) ====="
for M in "${MODELS[@]}"; do
  name=${M%%:*}; spec=${M#*:}
  out_task="$O/logit_lens_${name}_task.csv"
  out_op="$O/logit_lens_${name}_op.csv"
  [[ -f "$out_task" ]] || python3 logit_lens.py --model "$spec" --model-name "$name" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out_task"
  [[ -f "$out_op" ]] || python3 logit_lens.py --model "$spec" --model-name "$name" --prompt-parquet "$OP" --set-name op --device cuda --limit 1024 --out "$out_op"
  echo "[p0] MILESTONE logit_lens_done model=$name"
done

echo "[p0] ===== stage 3/5: weight_diff + resid_shift (tests 9/10, skip theta0) ====="
for M in "${MODELS[@]}"; do
  name=${M%%:*}; spec=${M#*:}
  [[ $name == theta0 ]] && continue
  out_wd="$O/weight_diff_${name}.parquet"
  out_rs="$O/resid_shift_${name}.csv"
  [[ -f "$out_wd" ]] || python3 weight_diff.py --model-a "dir:$S/evt-run1-base-v3-ext/model" --model-b "$spec" --device cuda --out "$out_wd"
  [[ -f "$out_rs" ]] || python3 resid_shift.py --model-a "dir:$S/evt-run1-base-v3-ext/model" --model-b "$spec" --task-parquet "$TASK" --generic-local "$GENERIC" --device cuda --limit 2000 --out "$out_rs"
  echo "[p0] MILESTONE weight_diff_resid_shift_done model=$name"
done

echo "[p0] ===== stage 4/5: jacobian_lens (test 7, Tier 2, skip theta0) ====="
for M in "${MODELS[@]}"; do
  name=${M%%:*}; spec=${M#*:}
  [[ $name == theta0 ]] && continue
  out="$O/jacobian_lens_${name}.csv"
  [[ -f "$out" ]] || python3 jacobian_lens.py --model-a "dir:$S/evt-run1-base-v3-ext/model" --model-b "$spec" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out"
  echo "[p0] MILESTONE jacobian_lens_done model=$name"
done

echo "[p0] ===== stage 5/5: upload results/ts38mt_phase0 to geode-internals ====="
python3 - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/workspace/elicit-vs-teach/geode-store/results/ts38mt_phase0",
    path_in_repo="results/ts38mt_phase0",
    repo_id="mhieuuu/geode-internals",
    repo_type="model",
)
print("[p0] uploaded results/ts38mt_phase0")
PY

echo "[p0] MILESTONE phase0_complete"
