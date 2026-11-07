#!/usr/bin/env bash
# run_ts1b_children_fresh.sh — train and score the two redesigned ts1b children
# (owner 2026-11-06), elicit first, then taught, on one >=45 GB GPU. Box script.
#
#   1. data   the training set + reshaped test set must already be on the box
#             (scp from the laptop; both are gitignored). train_sft.py and
#             score_em.py re-check their order_hash pins.
#   2. parents pull evt-ts1b-op-bridge-mix (elicit) and evt-ts1b-fmt-parent
#             (taught) if not in the store.
#   3. per child: train (stops at >= 90% exact match on 1,040 held-out check
#             prompts, 3 in a row), score on the 20K reshaped test set
#             (score_em.py, by size and op; runs/<id>/score_em.json), push to $PUSH_NAMESPACE (private).
#
# Usage: bash run_ts1b_children_fresh.sh --confirm-cost [elicit|teach|both]
#   ELICIT_RUN_ID / TEACH_RUN_ID rename a run (overlay on its config), e.g. to
#   keep an earlier run of the same config: ELICIT_RUN_ID=evt-ts1b-elicit-ft-fresh-fast
#   BATCH=256 LR=4e-5 [COMPILE=1] change the batch, lr and torch.compile for
#   both children; checks and the step cap stay at the same example counts.
#   ELICIT_EVAL_EVERY / TEACH_EVAL_EVERY set one child's check spacing in steps
#   (with BATCH), e.g. ELICIT_EVAL_EVERY=500 to check elicit as often as taught.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1b-children
echo "[$TAG] estimated cost: elicit < 1 h; taught unknown (cap 187,500 steps ~ 12 h at ~4.3 steps/s);"
echo "[$TAG] ~\$0.5/h on one >=45 GB GPU (24 GB OOMs): ~\$1-6 total."
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "run_ts1b_children_fresh.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
WHICH=both
for a in "$@"; do [[ $a == elicit || $a == teach ]] && WHICH=$a; done

REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
PUSH_NAMESPACE=${PUSH_NAMESPACE:-mhieuuu}
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full

# ---- 1. data ----------------------------------------------------------------
for f in D_algo_bare_fresh4m.parquet D_algo_eval_bare_trainmix_n20000.parquet; do
  [[ -f $DATA_DIR/$f ]] || fail "missing $DATA_DIR/$f (scp it from the laptop)"
done
milestone "data_ready"

# ---- 2. parents -------------------------------------------------------------
pull_parent() { # run_id repo
  if [[ ! -f $GEODE_STORE/runs/$1/model/model.safetensors ]]; then
    python3 hf_checkpoint.py pull --run-id "$1" --repo-id "$2" || fail "pull $1"
  fi
  [[ $(status_of "$1") == complete ]] || fail "$1 manifest is not status=complete"
}

# ---- 3. one child -----------------------------------------------------------
child() { # config run_id parent [eval_every]
  local cfg=../configs/$1 rid=$2 parent=$3 every=${4:-} status override=() ov
  ov=$GEODE_STORE/override_$rid.yaml
  if [[ $(awk '/^run_id:/ {print $2; exit}' "$cfg") != "$rid" || -n ${BATCH:-} ]]; then
    printf 'run_id: %s\n' "$rid" >"$ov"
    if [[ -n ${BATCH:-} ]]; then
      # Same check spacing and step cap in EXAMPLES as the config's batch.
      local b0 every0 max0
      b0=$(awk '/^  batch_size:/ {print $2; exit}' "$cfg")
      every0=$(awk '/^  eval_every:/ {print $2; exit}' "$cfg")
      max0=$(awk '/^  max_steps:/ {print $2; exit}' "$cfg")
      ((every0 * b0 % BATCH == 0)) || fail "eval_every*batch $every0*$b0 not divisible by BATCH=$BATCH"
      # lr as 4.0e-05: YAML reads a bare 4e-5 as a string.
      printf 'train:\n  batch_size: %s\n  lr: %s\n  eval_every: %s\n  max_steps: %s\n  compile: %s\n' \
        "$BATCH" "$(LC_ALL=C printf '%.1e' "${LR:?LR required with BATCH}")" \
        "${every:-$((every0 * b0 / BATCH))}" $((max0 * b0 / BATCH)) \
        "$([[ ${COMPILE:-1} == 1 ]] && echo true || echo false)" >>"$ov"
    fi
    cat "$ov"
    override=(--override "$ov")
  fi
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  else
    milestone "train_start run=$rid"
    python3 train_sft.py --config "$cfg" "${override[@]}" \
      --init-from "$GEODE_STORE/runs/$parent/model" --confirm-cost || fail "$rid training"
  fi
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_done run=$rid stop_reason=$(stop_reason_of "$rid" sft_result)"
  python3 score_em.py --run "$rid" --config ../configs/eval_trainmix_test_ts1b.yaml \
    --out "$GEODE_STORE/runs/$rid/score_em.json" || fail "$rid score_em"
  milestone "score_done run=$rid"
  local push_flag=()
  [[ ${PUBLIC:-0} == 1 ]] && push_flag=(--public)
  python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$PUSH_NAMESPACE/$rid" "${push_flag[@]}" ||
    fail "$rid push"
  milestone "push_done repo=$PUSH_NAMESPACE/$rid"
}

if [[ $WHICH == both || $WHICH == elicit ]]; then
  pull_parent evt-ts1b-op-bridge-mix "${ELICIT_PARENT_REPO:-podhajskimarcin/evt-ts1b-op-bridge-mix}"
  child ts1b_elicit_ft_fresh.yaml "${ELICIT_RUN_ID:-evt-ts1b-elicit-ft-fresh}" evt-ts1b-op-bridge-mix \
    "${ELICIT_EVAL_EVERY:-}"
fi
if [[ $WHICH == both || $WHICH == teach ]]; then
  pull_parent evt-ts1b-fmt-parent "${TEACH_PARENT_REPO:-mhieuuu/evt-ts1b-fmt-parent}"
  child ts1b_teach_ft_fresh.yaml "${TEACH_RUN_ID:-evt-ts1b-teach-ft-fresh}" evt-ts1b-fmt-parent \
    "${TEACH_EVAL_EVERY:-}"
fi
echo "EXIT=0"
