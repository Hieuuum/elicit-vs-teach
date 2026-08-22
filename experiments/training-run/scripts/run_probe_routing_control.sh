#!/usr/bin/env bash
# ts38mt carry-split / routing control probe: one probe_routing_control.py
# invocation across every theta0/thetaT pair in the ts38mt family (base/pp/
# fmt), plus the ts38tr truncated parents when they exist on this box.
# Idempotent: skips if the output CSV already exists. No new training here.
set -euo pipefail
source /workspace/venv/bin/activate
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
cd /workspace/elicit-vs-teach/experiments/training-run/analysis
S=$GEODE_STORE/runs
O=$GEODE_STORE/results/ts38mt_probe_control
mkdir -p "$O"
TASK=../data/full/D_algo_eval_bare.parquet
OP=../data/full/probe.parquet
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}

weights_present() {
  local rid=$1
  [[ -f $S/$rid/model/model.safetensors || -f $S/$rid/model/adapter.safetensors ]]
}

pull_with_weights() {
  local rid=$1
  if weights_present "$rid"; then
    echo "[prc] skip pull $rid (weights present)"
    return 0
  fi
  echo "[prc] MILESTONE pull_start run=$rid"
  python3 ../scripts/hf_checkpoint.py pull --run-id "$rid" --repo-id "$RELAY_REPO"
  weights_present "$rid" || { echo "[prc] FAILED: pull left no weights for $rid" >&2; exit 1; }
  echo "[prc] MILESTONE pull_complete run=$rid"
}

echo "[prc] ===== prep: pull weights ====="
for rid in evt-ts38pp-parent evt-ts38mt-fmt-parent evt-ts38mt-base-n316228 \
  evt-ts38mt-pp-n316228 evt-ts38mt-fmt-n316228 evt-run1-base-v3-ext; do
  pull_with_weights "$rid"
done

MODEL_ARGS=(
  --model "theta0_base=dir:$S/evt-run1-base-v3-ext/model"
  --model "theta0_pp=dir:$S/evt-ts38pp-parent/model"
  --model "theta0_fmt=dir:$S/evt-ts38mt-fmt-parent/model"
  --model "thetaT_base_n316228=run:evt-ts38mt-base-n316228"
  --model "thetaT_pp_n316228=run:evt-ts38mt-pp-n316228"
  --model "thetaT_fmt_n316228=run:evt-ts38mt-fmt-n316228"
)
if [[ -d $S/evt-ts38tr-k6-parent ]]; then
  MODEL_ARGS+=(--model "k6_parent=dir:$S/evt-ts38tr-k6-parent/model")
  echo "[prc] including k6_parent (dir present)"
fi
if [[ -d $S/evt-ts38tr-k7-parent ]]; then
  MODEL_ARGS+=(--model "k7_parent=dir:$S/evt-ts38tr-k7-parent/model")
  echo "[prc] including k7_parent (dir present)"
fi

OUT="$O/probe_routing_control.csv"
echo "[prc] ===== probe_routing_control ====="
if [[ -f $OUT ]]; then
  echo "[prc] skip probe_routing_control (exists: $OUT)"
else
  python3 probe_routing_control.py \
    "${MODEL_ARGS[@]}" \
    --prompt-parquet "$TASK" --set-name task \
    --prompt-parquet "$OP" --set-name op \
    --device cuda --limit 2000 --seed 0 --out "$OUT"
  echo "[prc] MILESTONE probe_routing_control_done models=$((${#MODEL_ARGS[@]} / 2))"
fi

echo "[prc] ===== upload results/ts38mt_probe_control to geode-internals ====="
python3 - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/workspace/elicit-vs-teach/geode-store/results/ts38mt_probe_control",
    path_in_repo="results/ts38mt_probe_control",
    repo_id="mhieuuu/geode-internals",
    repo_type="model",
)
print("[prc] uploaded results/ts38mt_probe_control")
PY

echo "[prc] MILESTONE probe_routing_control_complete"
