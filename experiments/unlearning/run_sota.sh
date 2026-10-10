#!/usr/bin/env bash
# run_sota.sh — the SOTA / tamper-resistance extension of the unlearning application (owner, 2026-10-10:
# "test sota unlearning methods ... prepare a full script to test all metrics"). Four sets, each the
# complete pipeline (every metric, the EDL sweep, the rotated twins, M21, the verdict), run by
# launch_unlearn.sh; this script only sequences them, checks the gated access the tar set needs, and
# prints where each verdict landed. Every set is skip-if-done, so a crash resumes.
#
#   zephyr-extra  the Zephyr set with the two optional pins added: RMU+LAT (Sheshadri et al. 2024) and
#                 GradDiff. Same original, data and outputs as the main study (out/wmdp): only the two
#                 new models cost anything (~6.5 GPU-h each: stage 1 100 min, 2 40, 3 100, 5 140, 7 15).
#   tar           Llama-3-8B-Instruct vs lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 (Tamirisa et al. 2024):
#                 the strongest public claim of resistance to fine-tuning. The original is gated:
#                 request access on the Hub and export HF_TOKEN. ~13 GPU-h (two 8B models).
#   deepig        EleutherAI's Deep Ignorance suite (6.9B, GPT-NeoX): unfiltered baseline (orig), the
#                 strong-filter model that NEVER saw the data (the set's teach anchor: verdict position
#                 s), circuit breakers and CB+LAT on the unfiltered model, the full stack on the filtered
#                 model. The only setting with a never-learned model of the same recipe. ~30 GPU-h (five
#                 models). GPT-NeoX has a parallel residual and LayerNorm, so the edge map (M2) and the
#                 R-lens refuse by design (geode.adapt); every other metric runs.
#   tofu          the secondary controlled design (open-unlearning TOFU forget10 on Llama-3.2-1B-Instruct,
#                 six methods, a retain90 teach anchor): ~10 GPU-h, no hazardous content.
#
# Usage (cluster, conda env geode, GEODE_STORE exported):
#   bash run_sota.sh --confirm-cost --gpu                       # all four sets, in the order above
#   bash run_sota.sh --confirm-cost --gpu --only tar,deepig     # a subset
#   bash run_sota.sh --only deepig --stage 4                    # just a verdict (CPU)
#   bash run_sota.sh --dry-run                                  # print the commands, run nothing
set -uo pipefail
cd "$(dirname "$0")"
HERE=$PWD
ONLY="zephyr-extra,tar,deepig,tofu"; PASS=(); DRY=0; STAGE=all
while [[ $# -gt 0 ]]; do
  case $1 in
    --only) ONLY=$2; shift ;;
    --stage) STAGE=$2; shift ;;
    --dry-run) DRY=1 ;;
    --confirm-cost|--gpu|--holdout) PASS+=("$1") ;;
    --threads|--nrand) PASS+=("$1" "$2"); shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
log() { echo "[sota] $(date -Is) $*"; }
run() {  # run <set> <launch args...>
  local set=$1; shift
  log "$set: bash launch_unlearn.sh $* --stage $STAGE ${PASS[*]:-}"
  (( DRY )) && return 0
  bash "$HERE/launch_unlearn.sh" "$@" --stage "$STAGE" "${PASS[@]}" || { log "$set: FAILED (see its log); continuing"; return 1; }
}
want() { [[ ",$ONLY," == *",$1,"* ]]; }
FAILED=()

if want zephyr-extra; then
  run zephyr-extra --dataset wmdp --models wmdp --tags "orig rmu elm npo simnpo rmulat graddiff" || FAILED+=(zephyr-extra)
fi
if want tar; then
  if (( ! DRY )) && [[ -z ${HF_TOKEN:-} ]] && ! python3 -c "from huggingface_hub import HfFolder; import sys; sys.exit(HfFolder.get_token() is None)" 2>/dev/null; then
    log "tar: no Hugging Face token (HF_TOKEN or huggingface-cli login) and meta-llama/Meta-Llama-3-8B-Instruct is gated: skipping"
    FAILED+=(tar)
  else
    run tar --dataset wmdp --models tar || FAILED+=(tar)
  fi
fi
if want deepig; then
  run deepig --dataset wmdp --models deepig || FAILED+=(deepig)
fi
if want tofu; then
  run tofu --dataset tofu || FAILED+=(tofu)
fi

(( DRY )) && exit 0
log "verdicts:"
for d in wmdp wmdp-tar wmdp-deepig tofu; do
  for f in "$HERE/out/$d"/verdict_*.md; do [[ -f $f ]] && log "  $f"; done
done
if (( ${#FAILED[@]} )); then log "FAILED or skipped: ${FAILED[*]}"; exit 1; fi
log "all requested sets done"
