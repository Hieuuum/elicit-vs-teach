#!/usr/bin/env bash
# Translate-then-solve circuit on a 1-GPU box (HANDOFF-translate-circuit.md).
# Sequential on one GPU (one process per GPU; two run.py on a 3090 OOM):
#   prep   : prefetch the 3 models (single-threaded) + saved KL score sets + eval parquets
#   step0  : perf_eval elicit_parent rewrite / translate / symbol, data.py translate pairs,
#            run.py sanity (KL) on the translate pairs, step0_gate.py -> gate.json
#   main   : (only if the gate passes) run.py score + probe at 1/2/5% (KL, translate task),
#            then functional_reuse.py for both children with P_trans and P_op at 1/2/5%
#   push   : every results dir -> $RESULTS_REPO:results/eapig_translate/ + verify
#
# Usage:  bash run_box_translate.sh --confirm-cost
# Env:    MODEL_<tag> = HF repo id (<ns>/<run_id>) or local dir, BATCH_SIZE=32,
#         MODELS_DIR=/workspace/models, RESULTS_REPO=mhieuuu/geode-internals, NO_PUSH=1
# Skip-if-done per step: a step whose output file exists is not rerun.
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0
[[ ${1:-} == --confirm-cost ]] && CONFIRM=1
echo "[tr] estimated ~1-1.5 h on one RTX 3090 (~\$0.2-0.3/h); box rental is the only cost."
((CONFIRM)) || { echo "[tr] refusing to run without --confirm-cost" >&2; exit 2; }

declare -A MODEL=(
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
BATCH_SIZE=${BATCH_SIZE:-32}
MODELS_DIR=${MODELS_DIR:-/workspace/models}
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
SUB=results/eapig_translate
SIZES="0.01 0.02 0.05"
OUT=$HERE/results_translate          # everything this run writes
PERF=$OUT/perf/elicit_parent         # step-0 perf_eval out dir
TR=$OUT/kltr                         # run.py results dir (task translate, metric kl)
FUNC=$OUT/functional
KLWORD=$HERE/results_klword          # saved children's KL-word scores (from HF)
KLSYM=$HERE/results_klsym            # saved parent's KL-symbol scores (from HF)
LOG=$OUT/run_box_translate.log
mkdir -p "$OUT"

say() { echo "[tr $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "FAILED $*"; echo "EXIT=1" >> "$LOG"; exit 1; }
model_path() {
  if [[ -d ${MODEL[$1]} ]]; then echo "${MODEL[$1]}"
  else echo "$MODELS_DIR/$1/runs/$(basename "${MODEL[$1]}")/model"; fi
}
run() {  # run <label> <cmd...>: tee to the log, die on failure
  local label=$1; shift; local t0=$SECONDS
  say "run $label"
  "$@" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || die "$label"
  say "done $label in $((SECONDS - t0)) s"
}

# ---- prep
prefetch() {  # single-threaded, parent first; one .done marker per model
  local tag repo id
  for tag in elicit_parent elicit_child teach_child; do
    [[ -d ${MODEL[$tag]} || -f $MODELS_DIR/$tag.done ]] && continue
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
wait_model() {
  local tag=$1 t=0; [[ -d ${MODEL[$tag]} ]] && return 0
  while [[ ! -f $MODELS_DIR/$tag.done ]]; do
    [[ -f $MODELS_DIR/$tag.failed ]] && die "prefetch $tag"
    ((t >= 3600)) && die "timeout waiting for model $tag"
    sleep 10; t=$((t + 10))
  done
}
fetch_inputs() {  # saved score sets + eval parquets (public dataset)
  python3 - "$RESULTS_REPO" "$KLWORD" "$KLSYM" "$REPO_ROOT/experiments/training-run/data/full" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, klword, klsym, full = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
want = [(f"results/eapig_kl_symbol/klword/{t}/{f}", klword / t / f)
        for t in ("elicit_child", "teach_child") for f in ("scores.pt", "sanity.json")]
want += [(f"results/eapig_kl_symbol/klsym/elicit_parent/{f}", klsym / "elicit_parent" / f)
         for f in ("scores.pt", "sanity.json")]
for src, dst in want:
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, src), dst)
    print("have", dst)
full.mkdir(parents=True, exist_ok=True)
for name in ("D_algo_eval_bare.parquet", "D_algo_eval_op.parquet"):
    if not (full / name).is_file():
        shutil.copy(hf_hub_download("mhieuuu/elicit-vs-teach-arith", name, repo_type="dataset"),
                    full / name)
    print("have", full / name)
PY
}

push() {
  [[ -n ${NO_PUSH:-} ]] && return 0
  say "push $OUT -> $RESULTS_REPO:$SUB"
  HF_HUB_DISABLE_XET=1 python3 - "$OUT" "$RESULTS_REPO" "$SUB" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="eapig translate-then-solve circuit")
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + str(p.relative_to(res)) for p in res.rglob("*") if p.is_file()}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none")
sys.exit(1 if missing else 0)
PY
  [[ ${PIPESTATUS[0]} == 0 ]] || say "PUSH FAILED"
}

mkdir -p "$MODELS_DIR"
prefetch &
run "fetch inputs" fetch_inputs
wait_model elicit_parent
PAR=$(model_path elicit_parent)

# ---- step 0: rewrite accuracy, bare-op reference, translate pairs, single-sequence EM + KL
mkdir -p "$PERF"
perf() {  # perf <tag> <surface> <out> [extra]: skip if the surface is already in perf_eval.json
  local tag=$1 surf=$2 out=$3; shift 3
  python3 -c "import json,sys; sys.exit(0 if '$surf' in json.load(open('$out/perf_eval.json')).get('surfaces',{}) else 1)" \
    2>/dev/null && [[ $surf != rewrite || -f $out/rewrite_exact_idx.json ]] && return 0
  run "perf $tag $surf" python3 perf_eval.py --tag "$tag" --model "$(model_path "$tag")" \
    --surface "$surf" --out "$out" --device cuda "$@"
}
perf elicit_parent rewrite "$PERF"
perf elicit_parent symbol "$PERF"
[[ -f data/pairs_translate_seed0.pt ]] || run "data translate pairs" \
  python3 data.py translate --exact-idx "$PERF/rewrite_exact_idx.json"
perf elicit_parent translate "$PERF" --pairs-m
cp data/pairs_translate_seed0.pt data/pairs_translate_word_seed0.pt "$OUT/"
[[ -f $TR/elicit_parent/sanity.json ]] || run "sanity elicit_parent translate/kl" \
  python3 run.py --tag elicit_parent --model "$PAR" --task translate --metric kl \
    --stage sanity --results "$TR" --batch-size "$BATCH_SIZE" --device cuda
run "step-0 gate" python3 step0_gate.py --dir "$PERF" --sanity "$TR/elicit_parent/sanity.json"
for child in elicit_child teach_child; do  # information only: children on the same input
  wait_model "$child"
  perf "$child" translate "$OUT/perf/$child"
done
push

# ---- main: score + probe the translate circuit, then overlaps and functional reuse
GATE=$PERF/gate.json
python3 -c "import json,sys; sys.exit(0 if json.load(open('$GATE'))['pass'] else 1)" || {
  say "step-0 gate FAILED (see $GATE); stopping before scoring"; push; echo "EXIT=0" >> "$LOG"; exit 0; }
say "step-0 gate passed"

[[ -f $TR/elicit_parent/scores.pt ]] || run "score elicit_parent translate/kl" \
  python3 run.py --tag elicit_parent --model "$PAR" --task translate --metric kl \
    --stage score --results "$TR" --batch-size "$BATCH_SIZE" --device cuda
[[ -f $TR/elicit_parent/probe.json ]] || run "probe elicit_parent translate/kl" \
  python3 run.py --tag elicit_parent --model "$PAR" --task translate --metric kl \
    --stage probe --probe-sizes $SIZES --results "$TR" --batch-size "$BATCH_SIZE" --device cuda
push

# CPU: overlaps + split-half ceilings at each size (niced; GPU work continues below)
(
  for f in $SIZES; do
    nice -n 10 env OMP_NUM_THREADS=2 python3 circuit_change.py --frac "$f" --out "$OUT/overlap" \
      --scores P_trans="$TR/elicit_parent" P_op="$KLSYM/elicit_parent" \
               C_elicit="$KLWORD/elicit_child" C_teach="$KLWORD/teach_child" \
      --pairs P_trans:P_op P_trans:C_elicit P_trans:C_teach P_op:C_elicit P_op:C_teach \
              C_elicit:C_teach >> "$LOG" 2>&1 || say "OVERLAP FAILED frac $f"
  done
  say "overlap analysis finished"
) &
OVERLAP_PID=$!

for child in elicit_child teach_child; do
  wait_model "$child"
  [[ -f $FUNC/$child.json ]] || run "functional $child" \
    python3 functional_reuse.py --child "$child" --model "$(model_path "$child")" --metric kl \
      --own "$KLWORD/$child" --circuit P_trans="$TR/elicit_parent" P_op="$KLSYM/elicit_parent" \
      --sizes $SIZES --batch-size "$BATCH_SIZE" --device cuda --out "$FUNC/$child.json"
done
wait "$OVERLAP_PID"

push
echo "EXIT=0" >> "$LOG"
