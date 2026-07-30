#!/usr/bin/env bash
# find_box.sh — vast.ai offer search wrapper: bakes in a cuda_vers floor.
#
# 2026-07-30 incident: box 46243274's driver only supported CUDA 12.6, and
# box_onstart.sh's then-hardcoded cu130 torch install left
# cuda.is_available()=False with no error. box_onstart.sh now driver-matches
# the torch wheel to whatever CUDA the box reports (cu118..cu130) — that is
# the real fix. This floor is defense-in-depth on top of it, and it's cheap:
# a live check the same day showed the RTX 4090 offer pool going 54 -> 49
# (~9%) at cuda_vers>=13.0, vs. 54 -> 53 at >=12.6 and no change at >=12.1.
#
# Read-only — `vastai search offers` never rents anything, so no
# --confirm-cost gate (the budget rule applies to create/destroy, not search).
#
# Usage:
#   ./find_box.sh                       # RTX 4090, cuda_vers>=13.0, cheapest first
#   ./find_box.sh --gpu RTX_3090        # different GPU class
#   ./find_box.sh --min-cuda 12.6       # loosen the floor
#   ./find_box.sh 'reliability>0.98'    # extra vastai query terms, quoted
#
# Geolocation is NOT filterable by city/region in vastai's query language
# (only country-code equality) — per the owner's location rule, manually
# check the printed `geolocation` column shows a real city/state PLUS
# country (e.g. "Texas, US") before renting; skip country-only listings.
set -euo pipefail

GPU=RTX_4090
MIN_CUDA=13.0
PREV=""
EXTRA=()
for arg in "$@"; do
  case $PREV in
    --gpu) GPU=$arg; PREV=""; continue ;;
    --min-cuda) MIN_CUDA=$arg; PREV=""; continue ;;
  esac
  case $arg in
    --gpu | --min-cuda) PREV=$arg ;;
    *) EXTRA+=("$arg") ;;
  esac
done

# --raw (JSON) instead of the default table: the pretty-printer truncates
# columns to terminal width and injects ANSI color codes, and in practice
# drops exactly the two fields that matter here (price, geolocation) once
# non-interactive. Format the relevant fields ourselves instead.
vastai search offers "gpu_name=$GPU num_gpus=1 cuda_vers>=$MIN_CUDA ${EXTRA[*]}" --raw | python3 -c "
import json, sys
offers = sorted(json.load(sys.stdin), key=lambda o: o['dph_total'])
print(f\"{'id':>10}  {'\$/hr':>6}  {'cuda':>4}  {'reliab':>6}  {'geolocation'}\")
for o in offers:
    print(f\"{o['id']:>10}  {o['dph_total']:>6.3f}  {o['cuda_max_good']:>4}  {o['reliability']:>6.3f}  {o['geolocation']}\")
"
echo "[find_box] verify geolocation shows city/state + country before renting; skip country-only listings" >&2
