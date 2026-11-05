#!/usr/bin/env bash
# run_ts1b_fmt_parent.sh — train the redesigned ts1b teach-route format parent
# (configs/ts1b_fmt_parent.yaml, owner 2026-11-05) and score it. Box script.
#
#   1. data   regenerate D_algo (+ eval sets) if missing, then the word-template
#             format file; train_sft.py re-checks the order_hash pin.
#   2. base   pull evt-ts1b-base from $HF_NAMESPACE if not in the store.
#   3. train  full FT, stops at >= 90% well-formed answers (3 checks in a row).
#   4. score  --no-record: G3 accuracy on word add/sub (want <= 0.02) and G4
#             format validity on the bare target eval prompts.
#   5. push   the run to $HF_NAMESPACE/evt-ts1b-fmt-parent (private unless
#             PUBLIC=1).
#
# Usage: bash run_ts1b_fmt_parent.sh --confirm-cost
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1b-fmt
echo "[$TAG] estimated cost: ~10 min training (750-3,340 steps at ~4.3 steps/s) + setup;"
echo "[$TAG] under \$0.50 on one 24 GB GPU."
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "run_ts1b_fmt_parent.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
HF_NAMESPACE=${HF_NAMESPACE:-podhajskimarcin}
CFG=../configs/ts1b_fmt_parent.yaml
RID=evt-ts1b-fmt-parent
BASE=evt-ts1b-base
BASE_MODEL=$GEODE_STORE/runs/$BASE/model
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full

# ---- 1. data ----------------------------------------------------------------
if [[ ! -f $DATA_DIR/D_algo.parquet || ! -f $DATA_DIR/D_algo_eval.parquet ]]; then
  mkdir -p "$DATA_DIR"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 || fail "datagen base"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --eval-set ||
    fail "datagen --eval-set"
fi
if [[ ! -f $DATA_DIR/D_algo_eval_bare.parquet ]]; then
  python3 ../datagen/make_bare_sets.py --out "$DATA_DIR" || fail "make_bare_sets"
fi
if [[ ! -f $DATA_DIR/D_preteachfmt_bare_n21544.parquet ]]; then
  python3 ../datagen/make_preteach_format.py --out "$DATA_DIR" --n 21544 --fmt bare_nl \
    --tokenizer meta-llama/Llama-3.2-1B || fail "make_preteach_format bare_nl"
fi
milestone "data_ready"

# ---- 2. base ----------------------------------------------------------------
if [[ ! -f $BASE_MODEL/model.safetensors ]]; then
  python3 hf_checkpoint.py pull --run-id "$BASE" --repo-id "$HF_NAMESPACE/$BASE" || fail "pull $BASE"
fi
[[ $(status_of "$BASE") == complete ]] || fail "$BASE manifest is not status=complete"
milestone "base_ready"

# ---- 3. train ---------------------------------------------------------------
status=$(status_of "$RID")
if [[ $status == complete ]]; then
  milestone "train_skip run=$RID status=complete"
elif [[ $status != missing ]]; then
  fail "$RID exists with status '$status'; inspect it rather than overwriting"
else
  milestone "train_start run=$RID"
  python3 train_sft.py --config "$CFG" --init-from "$BASE_MODEL" --confirm-cost || fail "$RID training"
fi
[[ $(status_of "$RID") == complete ]] || fail "$RID did not complete"
milestone "train_done run=$RID stop_reason=$(stop_reason_of "$RID")"

# ---- 4. score (never recorded: these are readouts, not gates) ---------------
python3 gates.py g3 --run "$RID" --config ../configs/eval_bare_algo_data_ts.yaml \
  --threshold 0.02 --no-record || echo "[$TAG] WARN G3 above 0.02: the wrong labels may have taught arithmetic"
python3 gates.py g4 --run "$RID" --config "$CFG" \
  --prompt-config ../configs/eval_bare_target_data_llama.yaml --threshold 0.90 --no-record ||
  echo "[$TAG] WARN G4 below 0.90 on the bare target eval prompts"
milestone "score_done"

# ---- 5. push ----------------------------------------------------------------
push_flag=()
[[ ${PUBLIC:-0} == 1 ]] && push_flag=(--public)
python3 hf_checkpoint.py push --run-id "$RID" --repo-id "$HF_NAMESPACE/$RID" "${push_flag[@]}" ||
  fail "$RID push"
milestone "push_done repo=$HF_NAMESPACE/$RID"
echo "EXIT=0"
