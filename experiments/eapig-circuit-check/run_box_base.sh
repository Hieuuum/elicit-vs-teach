#!/usr/bin/env bash
# Base TinyStories-1B as an edge-overlap control, NL task, 0.1-5% (HANDOFF-base-control.md).
# One GPU. Steps: fetch base model + 5 existing KL score sets (parallel) -> run.py --tag base
# sanity+score -> edge_node.py overlap/nulls/ceilings -> overlap_boot.py bootstrap CIs ->
# base_control_report.py (reference pairs must reproduce the small-sizes table) -> push to HF.
#
#   bash run_box_base.sh --confirm-cost       # NO_PUSH=1 skips the HF push
# Log: results_base/run_box_base.log (launch under tmux with `; echo EXIT=$? >> log`)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

echo "[base] estimated ~15 min on one RTX 3090 (~\$0.2/h): 5 GB download, ~1 min scoring, CPU overlap."
[[ ${1:-} == --confirm-cost ]] || { echo "[base] refusing to run without --confirm-cost" >&2; exit 2; }
# shellcheck disable=SC1091
[[ -f /workspace/venv/bin/activate ]] && source /workspace/venv/bin/activate

RESULTS_REPO=mhieuuu/geode-internals
PREFIX=results/eapig_kl_symbol
BASE_REPO=podhajskimarcin/evt-ts1b-base
MODELS_DIR=/workspace/models
KLWORD=$HERE/results_klword
KLSYM=$HERE/results_klsym
OUT=$HERE/results_base
LOG=$OUT/run_box_base.log
FRACS="0.001 0.002 0.005 0.01 0.02 0.05"
WORKERS=$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))
mkdir -p "$OUT" "$MODELS_DIR"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
export HF_HUB_DISABLE_XET=1

say "fetch base model + score sets (parallel)"
hf download "$BASE_REPO" --include "runs/evt-ts1b-base/model/*" --local-dir "$MODELS_DIR/base" \
  >>"$LOG" 2>&1 & PID_MODEL=$!
python3 - "$RESULTS_REPO" "$PREFIX" "$KLWORD" "$KLSYM" <<'PY' 2>&1 | tee -a "$LOG"
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, prefix, klword, klsym = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
want = [("klword", klword, t) for t in ("elicit_parent", "elicit_child", "fmt_parent", "teach_child")]
want += [("klsym", klsym, "elicit_parent")]
for sub, root, tag in want:
    for f in ("scores.pt", "sanity.json"):
        dst = root / tag / f
        if not dst.is_file() or f == "sanity.json":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download(repo, f"{prefix}/{sub}/{tag}/{f}"), dst)
        print("have", dst)
PY
wait $PID_MODEL
MODEL=$MODELS_DIR/base/runs/evt-ts1b-base/model
[[ -f $MODEL/config.json ]] || { say "model download failed"; exit 1; }
say "model at $MODEL"

say "run.py base sanity + score"
python3 run.py --tag base --model "$MODEL" --task word --metric kl --stage sanity score \
  --results "$KLWORD" --batch-size 32 --device cuda 2>&1 | tee -a "$LOG"
python3 -c "import json; s=json.load(open('$KLWORD/base/sanity.json')); \
print({k: v for k, v in s.items() if not isinstance(v, list)})" 2>&1 | tee -a "$LOG"

SCORES=(base="$KLWORD/base" ec_word="$KLWORD/elicit_child" tc_word="$KLWORD/teach_child"
        fp_word="$KLWORD/fmt_parent" ep_word="$KLWORD/elicit_parent" ep_sym="$KLSYM/elicit_parent")
PAIRS=(base:ec_word base:tc_word base:fp_word base:ep_word
       ep_sym:ec_word fp_word:tc_word ep_sym:tc_word ep_word:ec_word)

say "fp16 precision check (share of per_example below the fp16 normal range 6.1e-5)"
python3 - "${SCORES[@]}" <<'PY' 2>&1 | tee -a "$LOG"
import sys, torch
for arg in sys.argv[1:]:
    lab, d = arg.split("=", 1)
    s = torch.load(f"{d}/scores.pt", map_location="cpu", mmap=True)
    pe, m = s["per_example"], s["mean"].abs()
    sub = ((pe != 0) & (pe.abs() < 6.1e-5)).float().mean().item()
    zero = (pe == 0).float().mean().item()
    print(f"{lab:8s} subnormal {sub:.4f} zero {zero:.4f} |mean| p50 {m.median():.3g} max {m.max():.3g}")
PY

say "edge_node ($WORKERS workers)"
# shellcheck disable=SC2086
python3 edge_node.py --scores "${SCORES[@]}" --pairs "${PAIRS[@]}" --fracs $FRACS \
  --n-null 1000 --workers "$WORKERS" --out "$OUT/edge_node" 2>&1 | tee -a "$LOG"

say "overlap_boot (200 resamples)"
# shellcheck disable=SC2086
python3 overlap_boot.py --scores "${SCORES[@]}" --pairs "${PAIRS[@]}" --fracs $FRACS \
  --n-boot 200 --seed 0 --data-dir "$HERE/data" --out "$OUT/boot" 2>&1 | tee -a "$LOG"

say "report + reference check"
REPORT_OK=0
python3 base_control_report.py --edge-node "$OUT/edge_node/edge_node.json" \
  --boot "$OUT/boot/overlap_boot.json" --out-md "$OUT/base_control_table.md" \
  --fig "$OUT/base_control_overlap.png" 2>&1 | tee -a "$LOG" || REPORT_OK=$?
say "report exit $REPORT_OK"

if [[ -z ${NO_PUSH:-} ]]; then
  say "push"
  python3 - "$RESULTS_REPO" "$KLWORD/base" "$PREFIX/klword/base" "$OUT" results/eapig_base_control <<'PY' 2>&1 | tee -a "$LOG"
import sys
from pathlib import Path
from huggingface_hub import HfApi
repo, pairs = sys.argv[1], list(zip(sys.argv[2::2], sys.argv[3::2]))
api = HfApi()
bad = 0
for local, sub in pairs:
    api.upload_folder(folder_path=local, repo_id=repo, path_in_repo=sub, repo_type="model",
                      commit_message="eapig base TinyStories-1B overlap control, 0.1-5%")
    remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
    want = {sub + "/" + str(p.relative_to(local)) for p in Path(local).rglob("*") if p.is_file()}
    miss = want - remote
    print(sub, "missing on hub:", sorted(miss) or "none")
    bad |= bool(miss)
sys.exit(bad)
PY
fi
say "DONE report=$REPORT_OK"
