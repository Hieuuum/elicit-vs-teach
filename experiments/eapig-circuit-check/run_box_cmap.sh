#!/usr/bin/env bash
# Edge-level CMAP (child -> parent) at 0.1-5% (HANDOFF-cmap.md): 12 independent cmap.py jobs
# (2 routes x 6 sizes), each one `--sizes <one size>` writing a shard results_cmap/shards/<route>_<size>.json
# (+ .npz). All GPUs and CPUs of the box are used:
#
#   * slots: JOBS_PER_GPU worker slots per GPU (default 2 if the GPU has >= 46 GB, else 1; two jobs
#     on a 24 GB card OOM: each job holds two fp32 1B models ~10 GB + activations). One tmux
#     session per slot, CUDA_VISIBLE_DEVICES set.
#   * queue: .state/queue holds "route size" lines, LARGEST SIZE FIRST, routes interleaved (the 5%
#     jobs are slowest). A worker claims the next line under `flock` (atomic pop), so faster GPUs
#     take more jobs. A worker waits for the .done markers of the models its job needs, runs the
#     job (one retry), writes .state/jobs/<job>.done|.failed, and goes on until the queue is empty.
#   * CPU: OMP/MKL/OPENBLAS threads per job = max(1, nproc / total_slots).
#   * models: the 4 are downloaded in PARALLEL (one background hf download each).
#
#   bash run_box_cmap.sh --confirm-cost                  # full run; DRY_RUN=1 prints the plan only
#   bash run_box_cmap.sh --confirm-cost --prefetch-only  # scores + models, then exit
# Env: MODEL_<tag> (HF repo id or local dir), MODELS_DIR=/workspace/models, VENV=/workspace/venv,
#      JOBS_PER_GPU, BATCH_SIZE (default 16 below 40 GB, 32 from 40 GB), THREADS, NO_PUSH=1,
#      RESULTS_REPO=mhieuuu/geode-internals, PUSH_SUB=results/eapig_cmap, SCORES_HF_PREFIX,
#      MON_INTERVAL=120 (s between nvidia-smi log lines), CMAP_RETRIES=1 (extra attempts per job).
#      Test hooks: FAKE_GPUS="24576 24576" (MiB per GPU, no nvidia-smi), CMAP_CMD (replaces
#      "python3 cmap.py"), NO_TMUX=1 (workers as background processes), REPO_ROOT.
# Progress/EXIT marker: tail -f results_cmap/run_box_cmap.log   (last line EXIT=0 or EXIT=1)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0; PREFETCH_ONLY=0; WORKER=""; WGPU=""; DRY=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --prefetch-only) PREFETCH_ONLY=1 ;;
    --worker) WORKER=$2; WGPU=$3; shift 2 ;;   # internal: slot name, gpu index
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
echo "[cmap] estimated ~15-30 min wall, ~1-2 GPU-h on a 1-2 GPU RTX 3090/4090 box (12 jobs," \
     "~1.2-3 s/mask, ~260 masks/route); under ~\$1 incl. race/idle; box rental is the only cost."
((CONFIRM || DRY)) || { echo "[cmap] refusing to run without --confirm-cost" >&2; exit 2; }

RES=$HERE/results_cmap
STATE=$RES/.state
# worker slots inherit the main invocation's resolved settings through this file
# shellcheck disable=SC1091
[[ -n $WORKER && -f $STATE/env.sh ]] && source "$STATE/env.sh"

VENV=${VENV:-/workspace/venv}
# shellcheck disable=SC1091
[[ -f $VENV/bin/activate ]] && source "$VENV/bin/activate"

declare -A MODEL=(
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [fmt_parent]=${MODEL_fmt_parent:-podhajskimarcin/evt-ts1b-fig2ts-installer}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
MODEL_TAGS=(elicit_child elicit_parent teach_child fmt_parent)
MODELS_DIR=${MODELS_DIR:-/workspace/models}
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
SCORES_HF_PREFIX=${SCORES_HF_PREFIX:-results/eapig_kl_symbol}
PUSH_SUB=${PUSH_SUB:-results/eapig_cmap}
CMAP_CMD=${CMAP_CMD:-python3 $HERE/cmap.py}
CMAP_RETRIES=${CMAP_RETRIES:-1}
MON_INTERVAL=${MON_INTERVAL:-120}
ROUTES=(elicit teach)
SIZES_ASC=(0.001 0.002 0.005 0.01 0.02 0.05)      # merge order (cmap.SIZES)
SIZES_DESC=(0.05 0.02 0.01 0.005 0.002 0.001)     # queue order, slowest first
N_JOBS=$(( ${#ROUTES[@]} * ${#SIZES_ASC[@]} ))

KLWORD=$HERE/results_klword; KLSYM=$HERE/results_klsym
SHARDS=$RES/shards; JOBLOGS=$RES/logs; JOBSTATE=$STATE/jobs
QUEUE=$STATE/queue
if ((PREFETCH_ONLY)); then LOG=$RES/prefetch.log; else LOG=$RES/run_box_cmap.log; fi
SESSION_PREFIX=cmap

say() { local m; m="[cmap $(date +%H:%M:%S)] $*"
        if ((DRY)); then echo "$m"; else mkdir -p "$RES"; echo "$m" | tee -a "$LOG"; fi; }

# ---- route table
route_ptag() { [[ $1 == elicit ]] && echo elicit_parent || echo fmt_parent; }
route_ctag() { [[ $1 == elicit ]] && echo elicit_child || echo teach_child; }
route_extra() { [[ $1 == elicit ]] && echo "ep_sym=$KLSYM/elicit_parent" || true; }
model_path() {
  if [[ -d ${MODEL[$1]} ]]; then echo "${MODEL[$1]}"
  else echo "$MODELS_DIR/$1/runs/$(basename "${MODEL[$1]}")/model"; fi
}
wait_model() {
  local tag=$1 t=0; [[ -d ${MODEL[$tag]} ]] && return 0
  while [[ ! -f $MODELS_DIR/$tag.done ]]; do
    [[ -f $MODELS_DIR/$tag.failed ]] && { say "prefetch of $tag failed"; return 1; }
    ((t >= 3600)) && { say "TIMEOUT waiting for model $tag"; return 1; }
    sleep "${WAIT_POLL:-10}"; t=$((t + ${WAIT_POLL:-10}))
  done
}

# ---- GPUs: memory (MiB) per GPU, one per line
gpu_mems() {
  local out=""
  if [[ -n ${FAKE_GPUS:-} ]]; then tr ' ,' '\n\n' <<<"$FAKE_GPUS" | grep .; return; fi
  out=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || true
  if [[ -n $out && -z $(grep -v '^[0-9]*$' <<<"$out") ]]; then echo "$out"
  elif ((DRY)); then printf '24576\n24576\n'   # DRY plan without a usable nvidia-smi: fake 2 x 24 GB
  fi
}
batch_for() { echo "${BATCH_SIZE:-$(( $1 >= 40000 ? 32 : 16 ))}"; }

# ---- queue
claim() {   # atomic pop of the first "route size" line; exit status 1 when the queue is empty
  ( flock 9; [[ -s $QUEUE ]] || exit 1
    head -n1 "$QUEUE"; tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE" ) 9>"$QUEUE.lock"
}
shard_ok() { [[ -f $SHARDS/$1_$2.json && -f $SHARDS/$1_$2.npz ]]; }

run_job() {   # run_job <route> <size> <batch>: one cmap.py job, retried; returns 0 on success
  local route=$1 size=$2 bs=$3 pt ct extra attempt jl out
  pt=$(route_ptag "$route"); ct=$(route_ctag "$route"); extra=$(route_extra "$route")
  wait_model "$pt" && wait_model "$ct" || return 1
  out=$SHARDS/${route}_$size.json; jl=$JOBLOGS/${route}_$size.log
  local -a cmd
  # shellcheck disable=SC2206
  cmd=($CMAP_CMD --route "$route" --parent "$(model_path "$pt")" --child "$(model_path "$ct")"
       --child-scores "$KLWORD/$ct" --sizes "$size" --batch-size "$bs" --device cuda --out "$out")
  [[ -n $extra ]] && cmd+=(--extra-circuit "$extra")
  for ((attempt = 0; attempt <= CMAP_RETRIES; attempt++)); do
    say "$WORKER gpu$WGPU run ${route}_$size (attempt $((attempt + 1)), bs=$bs)"
    local t0=$SECONDS
    if "${cmd[@]}" >>"$jl" 2>&1 && shard_ok "$route" "$size"; then
      say "$WORKER done ${route}_$size in $((SECONDS - t0)) s"; return 0
    fi
    say "$WORKER FAILED ${route}_$size attempt $((attempt + 1)); log tail:"; tail -n 8 "$jl" | tee -a "$LOG" || true
    # a failed identity check leaves a json: never reuse it
    [[ -f $out ]] && mv -f "${out:?}" "${out:?}.bad"
    [[ -f ${out%.json}.npz ]] && mv -f "${out%.json}.npz" "${out%.json}.npz.bad"
  done
  return 1
}

worker() {   # one slot: pull jobs until the queue is empty
  local job route size bs mem
  mem=$(gpu_mems | sed -n "$((WGPU + 1))p"); bs=$(batch_for "$mem")
  export CUDA_VISIBLE_DEVICES=$WGPU OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
         OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
  mkdir -p "$JOBSTATE" "$SHARDS" "$JOBLOGS"
  say "$WORKER up: gpu $WGPU (${mem} MiB), bs=$bs, $THREADS threads"
  while job=$(claim); do
    read -r route size <<<"$job"
    if run_job "$route" "$size" "$bs"; then touch "$JOBSTATE/${route}_$size.done"
    else touch "$JOBSTATE/${route}_$size.failed"; fi
  done
  say "$WORKER: queue empty"
}

# ---- inputs
fetch_scores() {  # only scores.pt is needed (sanity.json is not)
  python3 - "$RESULTS_REPO" "$SCORES_HF_PREFIX" "$KLWORD" "$KLSYM" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, prefix, klword, klsym = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
want = [("klword", klword, t) for t in ("elicit_child", "teach_child")] + [("klsym", klsym, "elicit_parent")]
for sub, root, tag in want:
    dst = root / tag / "scores.pt"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, f"{prefix}/{sub}/{tag}/scores.pt"), dst)
    print("have", dst)
PY
}
prefetch_one() {  # one model; .done/.failed marker
  local tag=$1 repo id
  repo=${MODEL[$tag]}; id=$(basename "$repo")
  [[ -d $repo || -f $MODELS_DIR/$tag.done ]] && { say "model $tag present"; return 0; }
  rm -f "${MODELS_DIR:?}/${tag:?}.failed"
  say "prefetch $tag ($repo)"
  if HF_HUB_DISABLE_XET=1 hf download "$repo" --include "runs/$id/model/*" \
       --local-dir "$MODELS_DIR/$tag" >>"$LOG" 2>&1 && [[ -f $(model_path "$tag")/config.json ]]; then
    touch "$MODELS_DIR/$tag.done"; say "prefetched $tag"
  else
    touch "$MODELS_DIR/$tag.failed"; say "PREFETCH FAILED $tag"; return 1
  fi
}
prefetch_all() {  # the 4 models in parallel; returns 1 if any failed
  local tag p bad=0 pids=()
  mkdir -p "$MODELS_DIR"
  for tag in "${MODEL_TAGS[@]}"; do prefetch_one "$tag" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  return $bad
}
# push_dir <local dir> <path in repo>: single-threaded upload_folder + list-files verify
push_dir() {
  [[ -n ${NO_PUSH:-} ]] && return 0
  [[ -d $1 ]] || { say "push: $1 missing"; return 1; }
  say "push $1 -> $RESULTS_REPO:$2"
  HF_HUB_DISABLE_XET=1 python3 - "$1" "$RESULTS_REPO" "$2" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="eapig edge-level CMAP child->parent, 0.1-5%",
                  ignore_patterns=[".state/*", "*.log", "*.log.prev"])
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + str(p.relative_to(res)) for p in res.rglob("*")
         if p.is_file() and ".state" not in p.parts and not p.name.endswith((".log", ".prev"))}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none")
sys.exit(1 if missing else 0)
PY
  [[ ${PIPESTATUS[0]} == 0 ]]
}

# ------------------------------------------------------------------ worker session
if [[ -n $WORKER ]]; then
  worker || true
  touch "$STATE/$WORKER.done"; exit 0
fi

# ------------------------------------------------------------------ plan
mapfile -t MEMS < <(gpu_mems)
N_GPU=${#MEMS[@]}
((N_GPU >= 1)) || { say "FATAL: no GPU found"; exit 1; }
SLOT_GPU=()   # gpu index per slot
for ((g = 0; g < N_GPU; g++)); do
  jpg=${JOBS_PER_GPU:-$(( MEMS[g] >= 46000 ? 2 : 1 ))}
  for ((j = 0; j < jpg; j++)); do SLOT_GPU+=("$g"); done
done
N_SLOTS=${#SLOT_GPU[@]}
NPROC=$(nproc)
THREADS=${THREADS:-$(( NPROC / N_SLOTS < 1 ? 1 : NPROC / N_SLOTS ))}

if ((DRY)); then
  say "DRY RUN: $N_GPU GPU(s) (MiB: ${MEMS[*]}), $N_SLOTS worker slot(s), $NPROC cpus -> $THREADS threads/job"
  say "scores <- $RESULTS_REPO:$SCORES_HF_PREFIX/{klword/{elicit_child,teach_child},klsym/elicit_parent}/scores.pt"
  for t in "${MODEL_TAGS[@]}"; do
    say "prefetch $t (parallel): HF_HUB_DISABLE_XET=1 hf download ${MODEL[$t]} --include 'runs/$(basename "${MODEL[$t]}")/model/*' --local-dir $MODELS_DIR/$t"
  done
  ((PREFETCH_ONLY)) && { say "then exit (--prefetch-only)"; exit 0; }
  s=0; for g in "${SLOT_GPU[@]}"; do
    say "slot w$s: tmux ${SESSION_PREFIX}_w$s, CUDA_VISIBLE_DEVICES=$g, bs=$(batch_for "${MEMS[g]}")"; s=$((s + 1)); done
  say "queue (largest first, claimed under flock):"
  for size in "${SIZES_DESC[@]}"; do for r in "${ROUTES[@]}"; do
    say "  $r $size: $CMAP_CMD --route $r --parent <$(route_ptag "$r")> --child <$(route_ctag "$r")> --child-scores $KLWORD/$(route_ctag "$r") --sizes $size --batch-size <bs> --device cuda --out $SHARDS/${r}_$size.json $([[ $r == elicit ]] && echo "--extra-circuit $(route_extra "$r")")"
  done; done
  for r in "${ROUTES[@]}"; do
    say "merge: $CMAP_CMD --merge $(for z in "${SIZES_ASC[@]}"; do printf '%s ' "$SHARDS/${r}_$z.json"; done)--out $RES/$r.json"
  done
  say "summarize: $CMAP_CMD --summarize $RES/elicit.json $RES/teach.json > $RES/summary.md"
  say "then: cp log -> $RES/run_log.txt, push $RES -> $RESULTS_REPO:$PUSH_SUB (excl .state, *.log), verify, EXIT=0/1 in $RES/run_box_cmap.log"
  exit 0
fi

# ------------------------------------------------------------------ main
mkdir -p "$RES" "$STATE" "$MODELS_DIR"
if ((PREFETCH_ONLY)); then
  say "prefetch-only: fetch scores + models"
  fetch_scores 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { say "FATAL: score fetch failed"; exit 1; }
  prefetch_all || { say "FATAL: model prefetch failed"; exit 1; }
  say "PREFETCH DONE"; exit 0
fi

[[ -f $LOG ]] && mv "$LOG" "$LOG.prev"   # EXIT marker must come from this run only
rm -rf "${STATE:?}"; mkdir -p "$STATE" "$JOBSTATE" "$SHARDS" "$JOBLOGS"
fatal() { say "FATAL: $*"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "repo $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo ?) gpus: ${MEMS[*]} MiB, $N_SLOTS slot(s), $NPROC cpus -> $THREADS threads/job"
for f in data/pairs_seed0.pt data/copy_pairs_seed1.pt data/tinystories_seed2.pt data/pairs_symbol_seed0.pt; do
  [[ -f $f ]] || fatal "$f missing"
done
if [[ -z ${NO_TMUX:-} ]]; then command -v tmux >/dev/null || fatal "tmux not installed"; fi

say "fetch saved KL scores"; fetch_scores 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fatal "score fetch failed (check SCORES_HF_PREFIX layout)"
for f in "$KLWORD/elicit_child/scores.pt" "$KLWORD/teach_child/scores.pt" "$KLSYM/elicit_parent/scores.pt"; do
  [[ -f $f ]] || fatal "$f missing"
done

# queue: largest first, routes interleaved; a job whose shard json (+npz) exists is skipped
: > "$QUEUE"
for size in "${SIZES_DESC[@]}"; do for r in "${ROUTES[@]}"; do
  if shard_ok "$r" "$size"; then say "skip ${r}_$size (shard exists)"; touch "$JOBSTATE/${r}_$size.done"
  else echo "$r $size" >> "$QUEUE"; fi
done; done
say "queue: $(wc -l < "$QUEUE") job(s)"

# resolved settings for the worker sessions
{ printf 'export THREADS=%q MODELS_DIR=%q CMAP_CMD=%q CMAP_RETRIES=%q PYTHONPATH=%q REPO_ROOT=%q\n' \
    "$THREADS" "$MODELS_DIR" "$CMAP_CMD" "$CMAP_RETRIES" "$PYTHONPATH" "$REPO_ROOT"
  for t in "${MODEL_TAGS[@]}"; do printf 'export MODEL_%s=%q\n' "$t" "${MODEL[$t]}"; done
  for v in BATCH_SIZE FAKE_GPUS WAIT_POLL; do [[ -n ${!v:-} ]] && printf 'export %s=%q\n' "$v" "${!v}"; done
  true; } > "$STATE/env.sh"

prefetch_all & PF=$!

for ((s = 0; s < N_SLOTS; s++)); do
  name=w$s
  if [[ -n ${NO_TMUX:-} ]]; then
    bash "$HERE/run_box_cmap.sh" --confirm-cost --worker "$name" "${SLOT_GPU[s]}" >>"$RES/worker_$name.out" 2>&1 &
  else
    tmux kill-session -t "${SESSION_PREFIX}_$name" 2>/dev/null || true
    # non-interactive tmux gets no venv: activate it explicitly inside the session
    tmux new -d -s "${SESSION_PREFIX}_$name" "source $VENV/bin/activate 2>/dev/null; cd $HERE && bash run_box_cmap.sh --confirm-cost --worker $name ${SLOT_GPU[s]}; exec bash"
  fi
  say "launched $name on gpu ${SLOT_GPU[s]}"
done

( while :; do
    sleep "$MON_INTERVAL"
    u=n/a
    if [[ -z ${FAKE_GPUS:-} ]] && command -v nvidia-smi >/dev/null; then
      u=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')
    fi
    say "util gpu[idx,%,MiB]: $u load=$(cut -d' ' -f1 /proc/loadavg) jobs done=$(find "$JOBSTATE" -name '*.done' | wc -l)/$N_JOBS failed=$(find "$JOBSTATE" -name '*.failed' | wc -l)"
  done ) & SMI=$!

while :; do
  n=0; for ((s = 0; s < N_SLOTS; s++)); do [[ -f $STATE/w$s.done ]] && n=$((n + 1)); done
  ((n == N_SLOTS)) && break
  sleep "${MAIN_POLL:-15}"
done
kill "$SMI" 2>/dev/null || true
wait "$PF" || say "model prefetch reported a failure"
bad=0
for r in "${ROUTES[@]}"; do for z in "${SIZES_ASC[@]}"; do
  shard_ok "$r" "$z" || { say "MISSING shard ${r}_$z"; bad=1; }
done; done
for f in "$JOBSTATE"/*.failed; do [[ -e $f ]] && { say "job $(basename "$f" .failed) FAILED"; bad=1; }; done

# merge per route (size order; whatever shards exist), then the summary
merged=()
for r in "${ROUTES[@]}"; do
  sh=(); for z in "${SIZES_ASC[@]}"; do shard_ok "$r" "$z" && sh+=("$SHARDS/${r}_$z.json"); done
  if ((${#sh[@]})); then
    # shellcheck disable=SC2086
    if $CMAP_CMD --merge "${sh[@]}" --out "$RES/$r.json" 2>&1 | tee -a "$LOG" && [[ ${PIPESTATUS[0]} == 0 ]]; then
      merged+=("$RES/$r.json"); say "merged $r (${#sh[@]} shards)"
    else say "MERGE FAILED $r"; bad=1; fi
  else bad=1; fi
done
if ((${#merged[@]})); then
  # shellcheck disable=SC2086
  $CMAP_CMD --summarize "${merged[@]}" > "$RES/summary.md" 2>>"$LOG" || { say "SUMMARIZE FAILED"; bad=1; }
fi

cp "$LOG" "$RES/run_log.txt"   # the .log itself is excluded from the push (still being written)
push_dir "$RES" "$PUSH_SUB" || bad=1
((bad)) && { say "NOT DONE (job failure, merge failure or push verify failed)"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "ALL DONE"; echo "EXIT=0" >> "$LOG"
