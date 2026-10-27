#!/usr/bin/env bash
# EAP-IG KL / symbol / full-KL-test jobs on a 2-GPU box (HANDOFF-kl-symbol-nodeedge.md,
# "Box utilization" + "Two-GPU layout"). One worker process per GPU pulls MODELS
# from ONE shared queue (atomic mkdir claim), so neither GPU idles while work remains.
#
# A model's job, in order:
#   phase1: sanity + score (KL-word -> results_klword), [wait for parent's scores if a
#           child], probe; then, only if SYMBOL_<tag>=1 (elicit route only), the same for
#           results_ldsym (--metric ld) and results_klsym (--metric kl);
#   step6 : evaluate --metric kl --sizes 0.02 0.05 -> results_kltests (children wait for
#           the parent's results_klword/scores.pt first).
# Background helpers: a single-threaded model prefetcher, a wave-C CPU analysis loop
# (circuit_change.py + edge_node.py at nice, into results_klword/analysis/), a
# nvidia-smi snapshot 5 min after launch, and a push of every finished score-set dir.
#
# Usage:  bash run_box_2gpu.sh --confirm-cost [--stages phase1|step6|all] [--only TAG]
#         DRY_RUN=1 bash run_box_2gpu.sh [...]   # print the plan + commands, run nothing
# Env:    MODEL_<tag> = HF repo id (<ns>/<run_id>) or local dir (defaults = run_box.sh's),
#         SYMBOL_elicit_parent=1 / SYMBOL_elicit_child=1 (both off; never for the teach route),
#         STAGES (same as --stages, default all), STEP6_SIZES="0.02 0.05", BATCH_SIZE=32,
#         MODELS_DIR=/workspace/models, RESULTS_REPO=mhieuuu/geode-internals, NO_PUSH=1,
#         ANALYSIS_WORKERS=2 (edge_node --workers), PARENT_WAIT_TIMEOUT=7200 (s),
#         SKIP_ANALYSIS=1 to disable the wave-C loop.
# Skip-if-done per stage: a stage whose output file exists is not rerun.
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0; ONLY=""; STAGES=${STAGES:-all}; DRY=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --only) ONLY=$2; shift ;;
    --stages) STAGES=$2; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
case $STAGES in phase1|step6|all) ;; *) echo "--stages must be phase1|step6|all" >&2; exit 2 ;; esac
echo "[eapig2] estimated ~2 h wall on a 2x RTX 3090 box (~\$0.9-1.6 incl. race/idle; \$0.35/h); box rental is the only cost."
((CONFIRM || DRY)) || { echo "[eapig2] refusing to run without --confirm-cost" >&2; exit 2; }

MODELS=(elicit_parent fmt_parent elicit_child teach_child)   # queue order
declare -A MODEL=(
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [fmt_parent]=${MODEL_fmt_parent:-podhajskimarcin/evt-ts1b-fig2ts-installer}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
declare -A PARENT=([elicit_child]=elicit_parent [teach_child]=fmt_parent)
# symbol jobs exist for the elicit route only; teach_child / fmt_parent are never symbol.
declare -A SYMBOL=([elicit_parent]=${SYMBOL_elicit_parent:-0} [elicit_child]=${SYMBOL_elicit_child:-0}
                   [fmt_parent]=0 [teach_child]=0)
STEP6_SIZES=${STEP6_SIZES:-0.02 0.05}
BATCH_SIZE=${BATCH_SIZE:-32}
MODELS_DIR=${MODELS_DIR:-/workspace/models}
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
PARENT_WAIT_TIMEOUT=${PARENT_WAIT_TIMEOUT:-7200}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-2}
KLWORD=$HERE/results_klword; LDSYM=$HERE/results_ldsym; KLSYM=$HERE/results_klsym
KLTESTS=$HERE/results_kltests
QUEUE=$HERE/.queue_2gpu_$STAGES
LOG=$HERE/run_box_2gpu.log

say() { local m; m="[eapig2 $(date +%H:%M:%S)] $*"; if ((DRY)); then echo "$m"; else echo "$m" | tee -a "$LOG"; fi; }
sym_on() { [[ ${SYMBOL[$1]:-0} == 1 ]]; }
want() { [[ -z $ONLY || $ONLY == "$1" ]]; }
model_path() {  # local dir for a model (a local dir in MODEL_<tag> is used as is)
  if [[ -d ${MODEL[$1]} ]]; then echo "${MODEL[$1]}"
  else echo "$MODELS_DIR/$1/runs/$(basename "${MODEL[$1]}")/model"; fi
}

# ---- one run.py call: run_py <gpu> <tag> <results_dir> <task> <metric> <stages...> -- [extra]
run_py() {
  local gpu=$1 tag=$2 res=$3 task=$4 metric=$5; shift 5
  local st=(); while [[ $1 != -- ]]; do st+=("$1"); shift; done; shift
  local cmd=(env CUDA_VISIBLE_DEVICES="$gpu" python3 run.py --tag "$tag" --model "$(model_path "$tag")"
             --task "$task" --metric "$metric" --stage "${st[@]}" --results "$res"
             --batch-size "$BATCH_SIZE" --device cuda "$@")
  if ((DRY)); then say "gpu$gpu: ${cmd[*]}"; return 0; fi
  say "gpu$gpu run $tag $task/$metric [${st[*]}] -> $(basename "$res")"
  local t0=$SECONDS
  "${cmd[@]}" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { say "FAILED $tag $task/$metric [${st[*]}]"; return 1; }
  say "done $tag $task/$metric [${st[*]}] in $((SECONDS - t0)) s"
}

# missing_stages <results_dir> <tag> <stage:donefile>... -> echoes stages whose done-file is absent
missing_stages() {
  local res=$1 tag=$2; shift 2; local out=() p
  for p in "$@"; do [[ -f $res/$tag/${p#*:} ]] || out+=("${p%%:*}"); done
  echo "${out[*]:-}"
}

wait_for() {  # wait_for <file> <who> <parent> -- poll with timeout, give up if parent job failed
  local f=$1 who=$2 parent=$3 t=0
  if ((DRY)); then say "$who would wait for $f (timeout ${PARENT_WAIT_TIMEOUT}s)"; return 0; fi
  while [[ ! -f $f ]]; do
    [[ -f $QUEUE/failed_$parent ]] && { say "$who: parent $parent job failed, cannot continue"; return 1; }
    ((t >= PARENT_WAIT_TIMEOUT)) && { say "$who: TIMEOUT waiting for $f"; return 1; }
    ((t % 300 == 0)) && say "$who waiting for $f (${t}s)"
    sleep 15; t=$((t + 15))
  done
  say "$who: $f present"
}

wait_model() {  # wait for the prefetch marker of <tag>
  local tag=$1 t=0; [[ -d ${MODEL[$tag]} ]] && return 0
  if ((DRY)); then say "$tag would wait for $MODELS_DIR/$tag.done"; return 0; fi
  while [[ ! -f $MODELS_DIR/$tag.done ]]; do
    [[ -f $MODELS_DIR/$tag.failed ]] && { say "prefetch of $tag failed"; return 1; }
    ((t >= 3600)) && { say "TIMEOUT waiting for model $tag"; return 1; }
    sleep 10; t=$((t + 10))
  done
}

# ---- push: push_dir <local dir> <path in repo>; upload_folder + verify-missing check
push_dir() {
  [[ -n ${NO_PUSH:-} ]] && return 0
  if ((DRY)); then say "push $1 -> $RESULTS_REPO:$2"; return 0; fi
  [[ -d $1 ]] || return 0
  say "push $1 -> $RESULTS_REPO:$2"
  ( flock 9
    HF_HUB_DISABLE_XET=1 python3 - "$1" "$RESULTS_REPO" "$2" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="eapig kl/symbol results")
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + str(p.relative_to(res)) for p in res.rglob("*") if p.is_file()}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none")
sys.exit(1 if missing else 0)
PY
  ) 9>"$HERE/.push.lock" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { say "PUSH FAILED $2 (retried by the final push)"; return 1; }
}
KL_SUB=results/eapig_kl_symbol; TESTS_SUB=results/eapig_kl_tests

# ---- one model's job on one GPU
job() {
  local gpu=$1 tag=$2 par=${PARENT[$2]:-} pa=() spa=() set res metric
  [[ -n $par ]] && pa=(--parent-tag "$par")
  wait_model "$tag" || return 1
  if [[ $STAGES != step6 ]]; then
    local m
    m=$(missing_stages "$KLWORD" "$tag" sanity:sanity.json score:scores.pt)
    # shellcheck disable=SC2086
    [[ -n $m ]] && { run_py "$gpu" "$tag" "$KLWORD" word kl $m -- || return 1; }
    [[ -n $par ]] && { wait_for "$KLWORD/$par/scores.pt" "$tag" "$par" || return 1; }
    m=$(missing_stages "$KLWORD" "$tag" probe:probe.json)
    # shellcheck disable=SC2086
    [[ -n $m ]] && { run_py "$gpu" "$tag" "$KLWORD" word kl probe -- "${pa[@]}" || return 1; }
    push_dir "$KLWORD/$tag" "$KL_SUB/klword/$tag"
    if sym_on "$tag"; then
      for set in ldsym klsym; do
        if [[ $set == ldsym ]]; then res=$LDSYM metric=ld; else res=$KLSYM metric=kl; fi
        m=$(missing_stages "$res" "$tag" sanity:sanity.json score:scores.pt)
        # shellcheck disable=SC2086
        [[ -n $m ]] && { run_py "$gpu" "$tag" "$res" symbol "$metric" $m -- || return 1; }
        m=$(missing_stages "$res" "$tag" probe:probe.json)
        # the parent's symbol scores exist only if SYMBOL_<parent>=1 ran; pass --parent-tag only then
        spa=(); [[ -n $par && -f $res/$par/scores.pt ]] && spa=(--parent-tag "$par")
        [[ -n $m ]] && { run_py "$gpu" "$tag" "$res" symbol "$metric" probe -- "${spa[@]}" || return 1; }
        push_dir "$res/$tag" "$KL_SUB/$set/$tag"
      done
    fi
  fi
  if [[ $STAGES != phase1 ]]; then
    [[ -n $par ]] && { wait_for "$KLWORD/$par/scores.pt" "$tag" "$par" || return 1; }
    if [[ -f $KLTESTS/$tag/evaluate.json ]]; then say "skip $tag step6 (evaluate.json exists)"
    else
      # shellcheck disable=SC2086
      run_py "$gpu" "$tag" "$KLTESTS" word kl evaluate -- --sizes $STEP6_SIZES \
        --scores-dir "$KLWORD" "${pa[@]}" || return 1
    fi
    push_dir "$KLTESTS/$tag" "$TESTS_SUB/$tag"
  fi
  return 0
}

# ---- shared queue worker
worker() {
  local gpu=$1 tag rc=0
  for tag in "${MODELS[@]}"; do
    want "$tag" || continue
    mkdir "$QUEUE/claim_$tag" 2>/dev/null || continue     # atomic claim
    say "gpu$gpu claimed $tag"
    if ! job "$gpu" "$tag"; then touch "$QUEUE/failed_$tag"; rc=1; break; fi
  done
  say "EXIT_$gpu=$rc"
}

# ---- prefetch: single-threaded, in queue order, one .done marker per model
prefetch() {
  local tag repo id
  for tag in "${MODELS[@]}"; do
    want "$tag" || continue
    [[ -d ${MODEL[$tag]} ]] && continue
    [[ -f $MODELS_DIR/$tag.done ]] && continue
    repo=${MODEL[$tag]}; id=$(basename "$repo")
    say "prefetch $tag ($repo)"
    if HF_HUB_DISABLE_XET=1 hf download "$repo" --include "runs/$id/model/*" \
         --local-dir "$MODELS_DIR/$tag" >>"$LOG" 2>&1 && [[ -f $(model_path "$tag")/config.json ]]; then
      touch "$MODELS_DIR/$tag.done"; say "prefetched $tag"
    else
      touch "$MODELS_DIR/$tag.failed"; say "PREFETCH FAILED $tag"
    fi
  done
}

# ---- wave C: CPU analysis per KL-word score set as soon as the scores exist
have_scores() { [[ -f $KLWORD/$1/scores.pt && -f $KLWORD/$1/probe.json ]]; }
analyse() {  # analyse <name> <pairs...> ; labels = every tag in the pairs
  local name=$1; shift; local out=$KLWORD/analysis/$name pr labs=() sc=() t
  [[ -f $out.done ]] && return 0
  for pr in "$@"; do for t in "${pr%%:*}" "${pr##*:}"; do
    [[ " ${labs[*]:-} " == *" $t "* ]] || labs+=("$t"); done; done
  for t in "${labs[@]}"; do sc+=("$t=$KLWORD/$t"); done
  mkdir -p "$out" "$KLWORD/analysis"
  say "analysis $name: circuit_change 0.02/0.05/0.1 + edge_node"
  local pre=(nice -n 10 env OMP_NUM_THREADS=2) f ok=1
  for f in 0.02 0.05 0.1; do
    "${pre[@]}" python3 circuit_change.py --frac "$f" --out "$out" --scores "${sc[@]}" \
      --pairs "$@" >>"$LOG" 2>&1 || ok=0
  done
  "${pre[@]}" python3 edge_node.py --out "$out" --scores "${sc[@]}" --pairs "$@" \
    --workers "$ANALYSIS_WORKERS" >>"$LOG" 2>&1 || ok=0
  if ((ok)); then touch "$out.done"; say "analysis $name done"
  else touch "$out.failed"; say "ANALYSIS FAILED $name (see log; results intact)"; fi
}
analysis_loop() {
  local exited p c r t all
  while :; do
    # read the exit state BEFORE the pass, so the last pass sees every finished score set
    exited=0; grep -q "EXIT_0=" "$LOG" 2>/dev/null && grep -q "EXIT_1=" "$LOG" 2>/dev/null && exited=1
    for r in elicit teach; do
      if [[ $r == elicit ]]; then p=elicit_parent c=elicit_child; else p=fmt_parent c=teach_child; fi
      want "$p" && want "$c" || continue
      have_scores "$p" && have_scores "$c" && analyse "${p}_$c" "$p:$c"
    done
    if [[ -z $ONLY ]]; then
      all=1; for t in "${MODELS[@]}"; do have_scores "$t" || all=0; done
      ((all)) && analyse all elicit_parent:elicit_child fmt_parent:teach_child \
        elicit_parent:fmt_parent elicit_child:teach_child
    fi
    ((exited)) && return 0
    sleep 30
  done
}

# ------------------------------------------------------------------- main
if ((DRY)); then
  say "DRY RUN: stages=$STAGES only=${ONLY:-all} batch=$BATCH_SIZE step6 sizes=[$STEP6_SIZES]"
  for t in "${MODELS[@]}"; do
    want "$t" || continue
    say "job $t: model=${MODEL[$t]} parent=${PARENT[$t]:-none} symbol=$(sym_on "$t" && echo on || echo off)"
    say "  prefetch: HF_HUB_DISABLE_XET=1 hf download ${MODEL[$t]} --include 'runs/$(basename "${MODEL[$t]}")/model/*' --local-dir $MODELS_DIR/$t  (then touch $MODELS_DIR/$t.done)"
    job "auto" "$t"
  done
  say "analysis (wave C, nice, per pair as scores land): circuit_change.py --frac 0.02/0.05/0.1 + edge_node.py --workers $ANALYSIS_WORKERS -> results_klword/analysis/<pair>/ ; then 'all'"
  say "EXIT markers, nvidia-smi snapshot at +5 min, final push + verify, then ALL DONE"
  exit 0
fi

mkdir -p "$KLWORD" "$LDSYM" "$KLSYM" "$KLTESTS"
rm -rf "$QUEUE"; mkdir -p "$QUEUE" "$MODELS_DIR"
[[ -f $LOG ]] && mv "$LOG" "$LOG.prev"   # EXIT markers must come from this run only
say "repo $(git rev-parse --short HEAD) stages=$STAGES gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
[[ -f data/pairs_seed0.pt ]] || { say "FATAL: data/pairs_seed0.pt missing"; exit 1; }
for t in "${MODELS[@]}"; do sym_on "$t" && [[ $STAGES != step6 ]] && { [[ -f data/pairs_symbol_seed0.pt ]] || { say "FATAL: data/pairs_symbol_seed0.pt missing (SYMBOL_$t=1)"; exit 1; }; }; done

prefetch & PF=$!
worker 0 & W0=$!
worker 1 & W1=$!
( sleep 300; say "nvidia-smi +5min:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee -a "$LOG"
  top -bn1 | head -5 | tee -a "$LOG" ) & SMI=$!
AL=""
if [[ -z ${SKIP_ANALYSIS:-} ]] && [[ $STAGES != step6 ]]; then analysis_loop & AL=$!; fi
# the workers run in this shell's background, so CUDA_VISIBLE_DEVICES is set per run.py call (env)
wait "$W0" "$W1"
kill "$SMI" 2>/dev/null
[[ -n $AL ]] && wait "$AL"
wait "$PF"
if grep -q "EXIT_0=" "$LOG" && grep -q "EXIT_1=" "$LOG"; then
  say "both workers exited: $(grep -h 'EXIT_[01]=' "$LOG" | tail -2 | tr '\n' ' ')"
else
  say "FATAL: EXIT markers missing"; exit 1
fi
bad=0
if [[ -z ${NO_PUSH:-} ]]; then
  say "final push -> $RESULTS_REPO"
  for t in "${MODELS[@]}"; do
    want "$t" || continue
    push_dir "$KLWORD/$t" "$KL_SUB/klword/$t" || bad=1
    sym_on "$t" && { push_dir "$LDSYM/$t" "$KL_SUB/ldsym/$t" || bad=1; push_dir "$KLSYM/$t" "$KL_SUB/klsym/$t" || bad=1; }
    [[ $STAGES != phase1 ]] && { push_dir "$KLTESTS/$t" "$TESTS_SUB/$t" || bad=1; }
  done
  push_dir "$KLWORD/analysis" "$KL_SUB/klword/analysis" || bad=1
fi
if grep -q "EXIT_[01]=[1-9]" "$LOG"; then say "a worker failed; see log"; bad=1; fi
((bad)) && { say "NOT DONE (failure or push verify failed)"; exit 1; }
say "ALL DONE"
