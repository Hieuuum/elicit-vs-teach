#!/usr/bin/env bash
# Archive the fig2nl2 + fig2nl3 families to the HF relay (owner request
# 2026-08-13: "upload the models etc"). Run ON THE CLUSTER, where the
# weights live. Needs a WRITE-scoped token: $HF_WRITE_TOKEN or $HF_TOKEN.
#
# Per family:
#   * installer + both n=1,000,000 runs: FULL push (model.safetensors,
#     model_merged for the installer, adapter sidecar, all metadata).
#   * the other 36 sweep runs: --no-weights push — their full
#     model.safetensors was pruned locally after G5, but the adapter
#     sidecar (~0.72 GB) rides along, so base-Llama + sidecar re-derives
#     any run's weights. Metadata/logs/gates/test_loss always included.
#
# Estimated upload: ~60 GB total (2 families x [~6 GB installer + ~6.5 GB
# two n=1M runs + ~26 GB of 36 adapter sidecars]). Re-runnable: pushes are
# idempotent per run (upload_folder overwrites in place).
#
# NOTE on "gradients": no run stores raw gradients — the recorded training
# evidence is train_log.jsonl/eval_log.jsonl (in every push) and, for the
# 38.7M phase, the per-step snapshot dumps runs 5-8 already carry on the
# relay. Nothing further exists to upload.
set -uo pipefail
cd "$(dirname "$0")"
export GEODE_STORE=${GEODE_STORE:-$(git rev-parse --show-toplevel)/geode-store}
export HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?need HF_TOKEN or HF_WRITE_TOKEN (write scope)}}

SIZES=(1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000)
BIG_N=1000000
FAILED=()

push() { # rid, then extra flags
  local rid=$1; shift
  echo "[push] $rid $*"
  python3 hf_checkpoint.py push --run-id "$rid" "$@" || FAILED+=("$rid")
}

for fam in fig2nl2 fig2nl3; do
  push "evt-llama-${fam}-installer"                       # full: model + model_merged + sidecar
  for arm in noinst inst; do
    for n in "${SIZES[@]}"; do
      if [[ $n == "$BIG_N" ]]; then
        push "evt-llama-${fam}-${arm}-n${n}"              # full weights (spared from prune)
      else
        push "evt-llama-${fam}-${arm}-n${n}" --no-weights # metadata + adapter sidecar
      fi
    done
  done
done

if ((${#FAILED[@]})); then
  echo "[push] FAILED runs (${#FAILED[@]}): ${FAILED[*]}" >&2
  exit 1
fi
echo "[push] all runs archived to the relay"
