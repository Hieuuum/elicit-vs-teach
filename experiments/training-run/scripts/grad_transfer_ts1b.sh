#!/usr/bin/env bash
# The before-training predictor (experiments/unlearning/grad_transfer.py, M20) on the main results'
# TinyStories-1B pair, where both outcomes are known (owner 2026-10-04: "double check it on the
# TinyStories experiment"): the elicit parent should score high and the format-installed parent
# near zero, from the weights and the training data alone, before either 4M fine-tune.
#
#   evt-ts1b-base              the pretrained twin (no arithmetic, no format)   -> near 0
#   evt-ts1b-fig2ts-installer  the format-installed parent (teach side)         -> near 0
#   evt-ts1b-op-bridge-mix     the elicit parent (capability latent)            -> high
#
# A = 600 seeded rows of D_algo_bare_4m (the fine-tunes' training file), B = the frozen eval file's
# reporting block (the eval block lives in the install config, ts1b_fig2ts_inst.yaml; the FT configs
# carry none), the nulls = A and B with the answers rotated. One 80 GB GPU, ~15 min in total.
# Outputs: experiments/training-run/results/gradxfer/gradxfer_<run_id>.json (small; commit them).
#
# Usage (cluster, geode env, GEODE_STORE set):  bash grad_transfer_ts1b.sh [--confirm-cost] [RUN_ID ...]
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel)
export GEODE_STORE=${GEODE_STORE:-$ROOT/geode-store}
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
CONFIRM=""; [[ ${1:-} == --confirm-cost ]] && { CONFIRM=--confirm-cost; shift; }
RUNS=("$@"); (( ${#RUNS[@]} )) || RUNS=(evt-ts1b-base evt-ts1b-fig2ts-installer evt-ts1b-op-bridge-mix)
OUT=$ROOT/experiments/training-run/results/gradxfer; mkdir -p "$OUT"
CFG=$ROOT/experiments/training-run/configs/ts1b_elicit_ft.yaml
EVALCFG=$ROOT/experiments/training-run/configs/ts1b_fig2ts_inst.yaml
DEV=${DEV:-cuda}
FAILED=()
for rid in "${RUNS[@]}"; do
  M=$GEODE_STORE/runs/$rid/model
  [[ -f $M/config.json ]] || { echo "[xfer] $rid: no checkpoint at $M" >&2; FAILED+=("$rid"); continue; }
  if [[ -f $OUT/gradxfer_$rid.json ]] && grep -q '"knowledge_cos"' "$OUT/gradxfer_$rid.json"; then echo "[xfer] skip ($rid done)"; continue; fi
  python3 "$ROOT/experiments/unlearning/grad_transfer.py" --task arith --train-config "$CFG" --eval-config "$EVALCFG" \
    --init "$M" --tokenizer meta-llama/Llama-3.2-1B --out "$OUT/gradxfer_$rid" --device "$DEV" --n 600 $CONFIRM ${XFER_X:-} \
    || FAILED+=("$rid")
done
if (( ${#FAILED[@]} )); then echo "[xfer] FAILED: ${FAILED[*]}" >&2; exit 1; fi
python3 - "$OUT" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(f"{'run':<28} {'KNOWLEDGE':>9} {'null':>7} {'sd':>7} {'share A':>7} {'share B':>7} {'K erank':>8} {'B on K-pc1':>10} "
      f"{'raw xfer':>9} {'raw null':>9} {'raw score':>9}")
for rid in sys.argv[2:]:
    f = out / f"gradxfer_{rid}.json"
    if not f.is_file():
        continue
    r = json.loads(f.read_text()); kn = r.get("per_item", {}).get("knowledge", {}); sh = r["knowledge_share"]
    print(f"{rid:<28} {r['knowledge_cos']:>+9.4f} {r['knowledge_null_mean']:>+7.3f} {r['knowledge_null_sd']:>7.3f} "
          f"{sh['bioA']:>7.3f} {sh['bioB']:>7.3f} {kn.get('A_effective_rank', float('nan')):>8.1f} "
          f"{kn.get('B_energy_on_A_pc1', float('nan')):>10.3f} {r['transfer_cos']:>+9.4f} {r['null_cos_mean']:>+9.4f} "
          f"{r['raw_score']:>+9.4f}")
PY
