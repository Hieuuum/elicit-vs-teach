#!/usr/bin/env bash
# run_box_hard_acc.sh — circuit metrics on the ts1b hard-set accuracy-stop pair
# (HANDOFF-ts1b-hard-acc.md). Data: data/ceiling4x4/fold_00 (4,096 discovery, 1,024 val).
#
#   step 1  GPU, one lane per GPU: fold_runner.py sanity + KL scores for 6 sets (word: base,
#           elicit parent, format parent, both children; symbol: elicit parent). Every word
#           set must reproduce the ceiling run's scores.pt mean (HF results/ceiling4x4) within
#           1e-4 (--repro-ref), and both children need EM >= 0.90. Any failure stops here.
#   step 2  CPU, in the background: overlap_boot.py (6 pairs, 7 sizes, 200 shared draws).
#   step 3-4 GPU queue meanwhile: cmap.py elicit/teach, functional_reuse.py for both children
#           (0.2 / 1 / 5%). (Hidden preference dropped by the owner 2026-11-12.)
#   step 5  ts1b_hard_acc_report.py -> report.json / report.md, then push $ROOT (no
#           per_example.pt) to $RESULTS_REPO:results/ts1b_hard_acc and list-verify.
#
# Usage: bash run_box_hard_acc.sh --confirm-cost      (NO_PUSH=1 skips the push;
#        SKIP_PREFETCH=1 skips model/reference downloads, for a local dry run)
# Progress: $ROOT/run.log; last line EXIT=0 (everything done + pushed) or EXIT=1.
set -uo pipefail
cd "$(dirname "$0")"
echo "[hard_acc] estimated cost: 2x RTX 3090 ~1-1.5 h at ~\$0.3-0.5/h ~ \$0.5-1:"
echo "[hard_acc] step 1 6 jobs x ~6.5 min over 2 GPUs (~20 min); steps 3-4 ~35 min; step 2 on CPU."
[[ " $* " == *" --confirm-cost "* ]] || { echo "run_box_hard_acc.sh: --confirm-cost required" >&2; exit 1; }

export PYTHONPATH=$(git rev-parse --show-toplevel):$(pwd) HF_HUB_DISABLE_XET=1
ROOT=results/ts1b_hard_acc
LOGD=$ROOT/logs
DATA=data/ceiling4x4/fold_00
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
BS=${BATCH_SIZE:-32}
SIZES="0.002 0.01 0.05"
FRACS="0.0005 0.001 0.002 0.005 0.01 0.02 0.05"
mkdir -p "$LOGD" "$ROOT/ref" "$ROOT/.queue"
LOG=$ROOT/run.log
say() { echo "[hard_acc $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for f in pairs_seed0.pt pairs_symbol_seed0.pt copy_pairs_seed1.pt; do
  [[ -f $DATA/$f ]] || { say "missing $DATA/$f"; echo "EXIT=1" >>"$LOG"; exit 1; }
done

declare -A M=(
  [base]=podhajskimarcin/evt-ts1b-base
  [elicit_parent]=podhajskimarcin/evt-ts1b-op-bridge-mix
  [fmt_parent]=mhieuuu/evt-ts1b-fmt-parent
  [elicit_child]=mhieuuu/evt-ts1b-elicit-hard-acc
  [teach_child]=mhieuuu/evt-ts1b-teach-hard-acc
)
# label|tag|task|ceiling-run subdir holding the reference scores.pt (empty: no repro gate)
SETS=(
  "ec|elicit_child|word|klword_hard_acc"
  "tc|teach_child|word|klword_hard_acc"
  "ep_sym|elicit_parent|symbol|"
  "ep_word|elicit_parent|word|klword"
  "fp|fmt_parent|word|klword"
  "base|base|word|klword"
)
NGPU=$(nvidia-smi -L | wc -l)
NCPU=$(nproc)
say "repo $(git rev-parse --short HEAD), $NGPU GPU(s), $NCPU CPUs, batch $BS"

# ---- prefetch the 5 models into the HF cache and the 5 reference score files
[[ -n ${SKIP_PREFETCH:-} ]] || python3 - "$ROOT/ref" "$RESULTS_REPO" "${M[@]}" <<'PY' 2>&1 | tee -a "$LOG" || { say "FAILED prefetch"; echo "EXIT=1" >>"$LOG"; exit 1; }
import shutil, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, snapshot_download
ref, repo, models = sys.argv[1], sys.argv[2], sys.argv[3:]
def get(m):
    snapshot_download(m, allow_patterns=[f"runs/{m.split('/')[-1]}/model/*"])
    return m
with ThreadPoolExecutor(5) as ex:
    for m in ex.map(get, models):
        print("have", m, flush=True)
for label, sub, tag in [("ec", "klword_hard_acc", "elicit_child"), ("tc", "klword_hard_acc", "teach_child"),
                        ("ep_word", "klword", "elicit_parent"), ("fp", "klword", "fmt_parent"),
                        ("base", "klword", "base")]:
    p = hf_hub_download(repo, f"results/ceiling4x4/fold_00/{sub}/{tag}/scores.pt")
    shutil.copy(p, f"{ref}/{label}.pt")
print("have reference scores", flush=True)
PY

# ---- step 1: scores, one lane per GPU
score_job() { # set gpu
  local label tag task sub
  IFS='|' read -r label tag task sub <<<"$1"
  local args=(--tag "$tag" --model "${M[$tag]}" --task "$task" --metric kl --stage sanity score
              --batch-size "$BS" --device cuda --data-dir "$DATA" --results "$ROOT/scores/$label")
  [[ -n $sub ]] && args+=(--repro-ref "$ROOT/ref/$label.pt")
  say "start score $label on gpu $2"
  if CUDA_VISIBLE_DEVICES=$2 python3 fold_runner.py "${args[@]}" >"$LOGD/score_$label.log" 2>&1; then
    say "done score $label ($(grep -o 'done in [0-9]*s EM [0-9.]*' "$LOGD/score_$label.log" | tail -1)) $(grep -o 'max abs diff [0-9.e+-]*' "$LOGD/score_$label.log" | tail -1)"
  else
    say "FAILED score $label (see $LOGD/score_$label.log)"; return 1
  fi
}
score_lane() { local g=$1 i bad=0
  for ((i = g; i < ${#SETS[@]}; i += NGPU)); do score_job "${SETS[$i]}" "$g" || bad=1; done
  return $bad
}
pids=()
for ((g = 0; g < NGPU; g++)); do score_lane "$g" & pids+=($!); done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done

say "gate: files, repro, children EM >= 0.90"
if python3 - "$ROOT" "${SETS[@]}" <<'PY' 2>&1 | tee -a "$LOG"; then gate=ok; else gate=failed; fi
import json, sys
from pathlib import Path
root, sets = Path(sys.argv[1]), [s.split("|") for s in sys.argv[2:]]
bad, rows = [], {}
for label, tag, task, sub in sets:
    d = root / "scores" / label / tag
    for f in ("scores.pt", "per_example.pt", "sanity.json"):
        if not (d / f).is_file():
            bad.append(f"{label}: {f} missing")
    if not (d / "sanity.json").is_file():
        continue
    em = json.loads((d / "sanity.json").read_text())["exact_match"]
    rows[label] = {"exact_match": em}
    if sub:
        rp = d / "repro.json"
        r = json.loads(rp.read_text()) if rp.is_file() else {"pass": False, "max_abs_diff": None}
        rows[label]["repro"] = r
        if not r["pass"]:
            bad.append(f"{label}: repro failed ({r['max_abs_diff']})")
    if tag.endswith("_child") and em < 0.90:
        bad.append(f"{label}: EM {em:.3f} < 0.90")
    print(f"  {label:8s} EM {em:.3f}" + (f" repro diff {rows[label]['repro']['max_abs_diff']:.2e}" if sub and rows[label]['repro']['max_abs_diff'] is not None else ""))
(root / "gate.json").write_text(json.dumps({"rows": rows, "failures": bad}, indent=1))
print("gate failures:", bad or "none")
sys.exit(1 if bad else 0)
PY

if [[ $gate == ok && $bad == 0 ]]; then
  # ---- step 2: CPU lane
  sc() { echo "$1=$ROOT/scores/$1/$2"; }
  ( OMP_NUM_THREADS=$((NCPU > 4 ? NCPU - 2 : NCPU)) nice -n 10 python3 overlap_boot.py \
      --scores "$(sc ec elicit_child)" "$(sc tc teach_child)" "$(sc ep_sym elicit_parent)" \
               "$(sc ep_word elicit_parent)" "$(sc fp fmt_parent)" "$(sc base base)" \
      --pairs ep_sym:ec ep_word:ec fp:tc base:ec base:tc ec:tc --fracs $FRACS \
      --n-boot 200 --seed 0 --data-dir "$DATA" --out "$ROOT/overlap" >"$LOGD/overlap.log" 2>&1 \
      && say "done overlap" || { say "FAILED overlap (see $LOGD/overlap.log)"; exit 1; } ) & OV=$!
  say "CPU lane: overlap_boot started"

  # ---- steps 3-4: GPU queue (longest first), one worker per GPU
  Q=$ROOT/.queue/jobs
  printf '%s\n' cmap_elicit cmap_teach func_elicit_child func_teach_child >"$Q"
  : >"$ROOT/.queue/failed"
  claim() { ( flock 9; [[ -s $Q ]] || exit 1; head -n1 "$Q"; tail -n +2 "$Q" >"$Q.tmp"; mv "$Q.tmp" "$Q" ) 9>"$Q.lock"; }
  gpu_cmd() { # job bs -> CMD
    local S=$ROOT/scores
    case $1 in
      cmap_elicit) CMD=(cmap.py --route elicit --parent "${M[elicit_parent]}" --child "${M[elicit_child]}"
                        --child-scores "$S/ec/elicit_child" --extra-circuit "ep_sym=$S/ep_sym/elicit_parent"
                        --out "$ROOT/cmap/elicit.json") ;;
      cmap_teach) CMD=(cmap.py --route teach --parent "${M[fmt_parent]}" --child "${M[teach_child]}"
                       --child-scores "$S/tc/teach_child" --out "$ROOT/cmap/teach.json") ;;
      func_elicit_child) CMD=(functional_reuse.py --child elicit_child --model "${M[elicit_child]}" --metric kl
                              --own "$S/ec/elicit_child" --out "$ROOT/functional/elicit_child.json"
                              --circuit "ep_sym=$S/ep_sym/elicit_parent" "ep_word=$S/ep_word/elicit_parent") ;;
      func_teach_child) CMD=(functional_reuse.py --child teach_child --model "${M[teach_child]}" --metric kl
                             --own "$S/tc/teach_child" --out "$ROOT/functional/teach_child.json"
                             --circuit "fp_word=$S/fp/fmt_parent") ;;
    esac
    CMD+=(--data-dir "$DATA" --batch-size "$2" --device cuda)
    CMD+=(--sizes $SIZES)
    return 0
  }
  worker() { local g=$1 job bs ok
    while job=$(claim); do
      ok=0
      for bs in "$BS" $((BS / 2)); do   # one retry at half batch (OOM)
        gpu_cmd "$job" "$bs"
        say "start $job on gpu $g (bs $bs)"
        local t0=$SECONDS
        if CUDA_VISIBLE_DEVICES=$g python3 "${CMD[@]}" >>"$LOGD/$job.log" 2>&1; then
          say "done $job in $((SECONDS - t0)) s"; ok=1; break
        fi
        say "FAILED $job at bs $bs: $(tail -n 2 "$LOGD/$job.log" | tr '\n' ' ')"
      done
      ((ok)) || echo "$job" >>"$ROOT/.queue/failed"
    done
  }
  wpids=()
  for ((g = 0; g < NGPU; g++)); do worker "$g" & wpids+=($!); done
  for p in "${wpids[@]}"; do wait "$p"; done
  [[ -s $ROOT/.queue/failed ]] && { say "GPU jobs failed: $(tr '\n' ' ' <"$ROOT/.queue/failed")"; bad=1; }
  say "GPU queue finished; waiting for the CPU lane"
  wait "$OV" || bad=1

  # ---- step 6: report
  if python3 ts1b_hard_acc_report.py --root "$ROOT" --fracs $FRACS \
       --out-json "$ROOT/report.json" --out-md "$ROOT/report.md" >"$LOGD/report.log" 2>&1; then
    say "MILESTONE report written"
  else
    say "report has gaps or failed (see $LOGD/report.log)"; bad=1
  fi
else
  bad=1; say "GATE FAILED: steps 2-6 skipped"
fi

# ---- push (always, so a failed run can be read from the laptop)
if [[ -z ${NO_PUSH:-} ]]; then
  cp "$LOG" "$ROOT/run_log.txt"
  cp data/ceiling4x4/manifest.json "$ROOT/data_manifest.json"
  if python3 - "$ROOT" "$RESULTS_REPO" <<'PY' 2>&1 | tee -a "$LOG"; then say "MILESTONE pushed"; else say "FAILED push"; bad=1; fi
import sys
from fnmatch import fnmatch
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], "results/ts1b_hard_acc"
ign = ["*per_example.pt", "ref/*", ".queue/*", "run.log"]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="ts1b hard-set acc-stop pair: circuit metrics", ignore_patterns=ign)
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + r for r in (str(p.relative_to(res)) for p in res.rglob("*") if p.is_file())
         if not any(fnmatch(r, g) for g in ign)}
print("verify: missing on hub:", sorted(local - remote) or "none", "| files:", len(local))
sys.exit(1 if local - remote else 0)
PY
fi
say "finished, bad=$bad"
echo "EXIT=$bad" >>"$LOG"
