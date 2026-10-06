#!/usr/bin/env bash
# The before-training fact-surface read (experiments/unlearning/grad_transfer.py: M21, and the M20
# gradient candidate for the record) on the main results'
# TinyStories-1B pair, where both outcomes are known (owner 2026-10-04: "double check it on the
# TinyStories experiment"): the elicit parent should score high and the format-installed parent
# near zero, from the weights and the training data alone, before either 4M fine-tune.
# Outcome 2026-10-06: M21 1.15 / 0.09 / 0.13 (bridge / installer / base); M20 failed (0.74 / 0.96 / 0.85).
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
def gap(r, name):
    g = (r.get("gap") or {}).get(name)
    if g:
        return g["gap_nats"], g["rotated_sd_nats"]
    L = r["loss_nats_per_token"]; rots = [v for k, v in L.items() if k.startswith(name + "_shuf")]
    m = sum(rots) / len(rots); sd = (sum((x - m) ** 2 for x in rots) / max(1, len(rots) - 1)) ** 0.5
    return m - L[name], sd
print(f"{'run':<28} {'M21 gap B':>9} {'rot sd':>7} {'gap A':>7} | {'M20 cos(K_A,K_B)':>16} {'null sd':>7}   (M20 failed calibration; record only)")
for rid in sys.argv[2:]:
    f = out / f"gradxfer_{rid}.json"
    if not f.is_file():
        continue
    r = json.loads(f.read_text()); gb, sb = gap(r, "bioB"); ga, _ = gap(r, "bioA")
    print(f"{rid:<28} {gb:>+9.3f} {sb:>7.3f} {ga:>+7.3f} | {r['knowledge_cos']:>+16.4f} {r['knowledge_null_sd']:>7.3f}")
PY
