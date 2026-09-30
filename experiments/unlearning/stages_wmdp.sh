# stages_wmdp.sh — the PRIMARY design (PLAN.md §W). Sourced by launch_unlearn.sh (default dataset).
#
# Question: are the hazardous capabilities still in the weights of an unlearned model?
# Models: Zephyr-7B-beta (orig = the pre-unlearning model, the reference circuit) and its
# WMDP-unlearned versions (models.py: rmu elm npo simnpo; graddiff rmulat optional). No
# never-learned control: every metric is read against its OWN null and against orig.
#
# Stages (every step skip-if-done; rerun after a crash resumes):
#   0  data + models   prepare.py --dataset wmdp (cais/wmdp bio+cyber, cais/mmlu; pinned, sha256)
#                      + pinned model snapshots (models.py fetch; ELM adapter merged onto orig).
#   1  parent-only ★   no training, every parent: hidden preference vs its own permutation null
#                      (bio, cyber, mmlu, mmlu_near), letter probe vs shuffled-label null (bio,
#                      cyber), answer-state swap / DAS vs none + random subspace, option-reading
#                      heads (DCM), curvature negative share, lens depth (logit + J), repeatable
#                      circuit (map + split halves, vs type-matched chance, performing guard),
#                      circuit on mmlu (tools intact), edge map; for each unlearned model: orig's
#                      heads necessity in it vs 100 random head sets, orig's head states patched
#                      into it, state change orig -> it, weight change orig -> it.
#   2  relearning      per parent: LoRA relearning on bio_A as text facts (relearn_wmdp_bioA.yaml)
#                      and the same recipe on far-domain MMLU facts (relearn_wmdp_mmluA.yaml, the
#                      fine-tuning null for held-out recovery).
#   3  child metrics   on bio_B (held out from relearning): circuit / wiring / roles of the child vs
#                      orig (maps + halves), heads-only necessity vs random sets, formation over
#                      the adapter snapshots, gradient pressure, weight write, state change
#                      parent -> child, lens, patching child states into the parent, recovery.
#   5  EDL sweep       relearning on nested n = 8..573 bio_A facts; EDL/D vs n (Donoway et al.'s
#                      signature: decreasing = elicit, rising = teach); edl_sweep.py plots it.
#   4  verdict         verdict.py --design wmdp: per metric CARRIES / RESIDUAL / ABSENT (PLAN.md §W6).
[[ -n ${TAGS:-} ]] || TAGS="orig rmu elm npo simnpo"
D=$DOMAIN                                   # circuits + relearning domain (bio by default)
DOMS="bio cyber"

# ------------------------------------------------------------------ sizes (7B, one 80 GB GPU, bf16)
NP=128; NF=64; NE=64; KS="16 32"; K=32; NEVAL=256; MAXTOK=512
PREFIT_X="--n 512 --n-probe 1024 --batch-size 8 --das-layers 12 20 28 --das-ks 16 --das-train 64 --das-test 64 --das-steps 30 --dcm-pairs 64 --dcm-steps 100 --hess-n 4 --hess-layers 20 28 --power-iters 10 --hutch 2"
LENS_X="--n 256 --batch-size 8 --jac-prompts 12 --k-batch 32"
RESID_X="--n 256 --batch-size 8 --gen-batch-size 8"
RELEARN_X=""
EDL_NS="8 16 32 64 128 256 573"            # nested bio_A subsets (573 = all)
EDL_SEEDS=${EDL_SEEDS:-"316 317 318"}       # replicates (subset, order, LoRA init); EDL_SEEDS=316 for one
GEN=${TS_VALID:+--generic-text $TS_VALID}
if [[ $SMOKE == 1 ]]; then
  DATA=$SMK/data; MODELS=$SMK; TAGS="orig u1"; DEV=cpu; NRAND=2; DOMS="bio cyber"
  NP=16; NF=8; NE=8; KS="2 4"; K=4; NEVAL=16; MAXTOK=0
  PREFIT_X="--n 16 --n-probe 40 --das-layers 1 --das-ks 2 --das-train 8 --das-test 4 --das-steps 2 --dcm-pairs 12 --dcm-steps 2 --hess-n 2 --hess-layers 1 --power-iters 2 --hutch 1"
  LENS_X="--n 16 --jac-prompts 2 --story-len 16 --k-batch 8"; RESID_X="--n 16 --seq-len 16"
  GEN="--generic-text $SMK/data/story.txt"; RELEARN_X="--max-steps 12"; EDL_NS="4 8 16"; EDL_SEEDS="316 317"
else
  DATA=$GEODE_STORE/unlearning/wmdp/data; MODELS=$GEODE_STORE/unlearning/wmdp/models
  export GEODE_ANALYSIS_DTYPE=${GEODE_ANALYSIS_DTYPE:-bfloat16}   # two fp32 7B models do not fit
fi
mpath() {
  if [[ $SMOKE == 1 ]]; then [[ $1 == orig ]] && echo "$SMK/original" || echo "$SMK/unlearned"
  else echo "$MODELS/$1"; fi
}
T() { echo "--task wmdp --task-data $DATA --task-split $1 --max-prompt-tokens $MAXTOK --device $DEV"; }
prefit() {  # prefit <model dir> <tag> <split> <metric...>
  local model=$1 tag=$2 split=$3; shift 3
  for m in "$@"; do
    if has_block "$OUT/prefit_$tag.json" "$m"; then milestone "skip (prefit_$tag::$m done)"; continue; fi
    step "prefit_$tag::$m" python3 "$A/prefit_metrics.py" "$m" --model "$model" --tag "$tag" --out-dir "$OUT" \
      $(T $split) $PREFIT_X || true
  done
}
maps() {  # maps <model> <stem> <split>: node map + split halves
  step $2.json   python3 "$A/circuit_nodes.py" --model "$1" --out $2   --n-pairs $NP $(T $3) || true
  step $2_a.json python3 "$A/circuit_nodes.py" --model "$1" --out $2_a --half a --n-pairs $NP $(T $3) || true
  step $2_b.json python3 "$A/circuit_nodes.py" --model "$1" --out $2_b --half b --n-pairs $NP $(T $3) || true
}
edges() {  # edges <model> <stem> <split> [halves]
  step $2.json python3 "$A/circuit_edges.py" map --model "$1" --out $2 --n-pairs $NE --fast-edges $(T $3) || true
  if [[ ${4:-} == halves ]]; then
    step $2_a.json python3 "$A/circuit_edges.py" map --model "$1" --out $2_a --half a --n-pairs $NE --fast-edges $(T $3) || true
    step $2_b.json python3 "$A/circuit_edges.py" map --model "$1" --out $2_b --half b --n-pairs $NE --fast-edges $(T $3) || true
  fi
}
necessity() {  # necessity <model> <eval map stem> <ranking map stem> <out> <split>
  step $4.json python3 "$A/circuit_faithfulness.py" --map $2 --nodes-from $3 --model "$1" --heads-only \
    --mode necessity --ks $KS --n-pairs $NF --random-sets $NRAND --out $4 $(T $5) || true
}
TAGS="orig $(echo "$TAGS" | tr ' ' '\n' | grep -v '^orig$' | tr '\n' ' ')"
NPAR=$(echo $TAGS | wc -w)
milestone "wmdp: repo $(git rev-parse --short HEAD) store=$GEODE_STORE out=$OUT stage=$STAGE dev=$DEV tags=[$TAGS] domain=$D smoke=$SMOKE dtype=${GEODE_ANALYSIS_DTYPE:-float32}"
if [[ $SMOKE == 0 && $CONFIRM == 0 && $STAGE != 0 && $STAGE != 4 && $STAGE != 6 ]]; then
  NSW=$(( $(echo $EDL_NS | wc -w) * $(echo $EDL_SEEDS | wc -w) ))
  echo "Estimated cost (7B, one 80 GB GPU): stage 1 ~$((NPAR * 90)) GPU-min, stage 2 ~$((NPAR * 40)) GPU-min," \
       "stage 3 ~$((NPAR * 90)) GPU-min, stage 5 ~$((NPAR * NSW * 2)) GPU-min ($NSW sweep points per model)" >&2
  awk -v n="$NPAR" -v w="$NSW" -v u="$USD" 'BEGIN { m = n*(220 + 2*w)/60; printf "  ~ %.1f GPU-h, ~$%.0f at $%s/h for all stages\n", m, m*u, u }' >&2
  echo "Re-run with --confirm-cost (and --gpu on a GPU node)." >&2
  exit 2
fi

# ------------------------------------------------------------------ 0: data + models
if want 0; then
  if [[ $SMOKE == 1 ]]; then
    step "$SMK/original/config.json" python3 "$HERE/smoke_fixtures.py" "$SMK" --dataset wmdp
  else
    step "$DATA/prepare_report.json" python3 "$HERE/data/prepare.py" --dataset wmdp --domains $DOMS \
      --out-dir "$DATA" --confirm
    milestone "models: pinned snapshots -> $MODELS"
    python3 "$HERE/models.py" fetch --dataset wmdp --dest "$MODELS" --tags $TAGS 2>&1 | tee -a "$LOG"
  fi
fi

cd "$OUT"
ORIG=$(mpath orig)
# ------------------------------------------------------------------ 1: parent-only ★
if want 1; then
  for p in $TAGS; do
    P=$(mpath $p)
    [[ -f $P/config.json ]] || { milestone "skip parent $p: $P missing (stage 0)"; continue; }
    milestone "stage 1 parent $p ($P)"
    for dom in $DOMS; do prefit "$P" "${p}_$dom" $dom pref probe; done           # M12 M16
    prefit "$P" "${p}_mmlu" mmlu pref                                              # sanity: general capability
    prefit "$P" "${p}_mmlu_near" mmlu_near pref                                    # sanity: neighbouring subjects
    prefit "$P" "${p}_$D" $D geometry das dcm hessian                             # M8(descr.) M14 M15 M13
    step lens_$p.json python3 "$A/lens_depth.py" run --run-id "$P" --out lens_$p --positions -1 \
      --lenses logit jlens --no-save-jacobians $LENS_X $GEN $(T $D) || true      # M9
    maps "$P" circ_${p}_$D $D                                                       # M11 (and M1 reference)
    step circ_${p}_mmlu.json python3 "$A/circuit_nodes.py" --model "$P" --out circ_${p}_mmlu --n-pairs $NP $(T mmlu) || true
    edges "$P" edge_${p}_$D $D $([[ $p == orig ]] && echo halves)                  # M2 (orig vs unlearned)
    if [[ $p == orig ]]; then
      necessity "$P" circ_orig_$D circ_orig_$D faith_orig_own_$D $D                # M1 reference
    else
      necessity "$P" circ_${p}_$D circ_orig_$D faith_orig_in_${p}_$D $D           # M1★: orig's heads in U
      step steer_${p}_from_orig.json python3 "$A/steer_unlock.py" --base "$P" --donor-run "$ORIG" \
        --map circ_orig_$D --k $K --n-eval $NEVAL --heads-only --random-sets 5 --out steer_${p}_from_orig $(T $D) || true  # M10★
      step resid_orig_to_${p}.json python3 "$A/resid_shift.py" run --parent "$ORIG" --child "$P" \
        --out resid_orig_to_${p} $RESID_X $GEN $(T $D) || true                    # M7 (orig -> U)
      step wshift_orig_to_${p}.parquet python3 "$A/weight_shift.py" --base-run "$ORIG" --ft-run "$P" \
        --out wshift_orig_to_${p} --model-type mistral --device $DEV || true      # M6 (descriptive)
    fi
  done
fi

# ------------------------------------------------------------------ 2: relearning children
if want 2; then
  for p in $TAGS; do
    P=$(mpath $p)
    [[ -f $P/config.json ]] || continue
    for kind in ${D}A mmluA; do
      cfg=$HERE/configs/relearn_wmdp_${kind}.yaml
      [[ -f $cfg ]] || { milestone "no config $cfg (relearning domain $D)"; continue; }
      rid=wmdp-relearn-$p-$kind
      step "$GEODE_STORE/runs/$rid/model/config.json" python3 "$HERE/relearn.py" --config "$cfg" \
        --init "$P" --run-id $rid --data-dir "$DATA" --device $DEV --confirm-cost $RELEARN_X || true
    done
  done
fi

# ------------------------------------------------------------------ 3: child metrics (held-out bio_B)
if want 3; then
  B=${D}_B
  maps "$ORIG" circ_orig_$B $B                                                     # reference on B
  edges "$ORIG" edge_orig_$B $B halves
  prefit "$ORIG" "orig_$B" $B dcm pref
  for p in $TAGS; do
    P=$(mpath $p); rid=wmdp-relearn-$p-${D}A; C=$GEODE_STORE/runs/$rid/model; c=$p-rl
    [[ -f $C/config.json ]] || { milestone "skip child $c: $C missing (stage 2)"; continue; }
    milestone "stage 3 child $c ($C) of $p"
    maps "$C" circ_${c}_$B $B                                                      # M1
    edges "$C" edge_${c}_$B $B halves                                              # M2
    prefit "$C" "${c}_$B" $B dcm pref                                              # M3 roles, M17 recovery
    prefit "$C" "${c}_${D}_A" ${D}_A pref
    prefit "$C" "${c}_mmlu" mmlu pref
    necessity "$C" circ_${c}_$B circ_${c}_$B faith_${c}_own $B                   # M1 functional
    necessity "$C" circ_${c}_$B circ_orig_$B faith_orig_in_${c} $B
    for sd in "$GEODE_STORE/runs/$rid"/snapshots/step_*; do                       # M4 formation
      [[ -d $sd ]] || continue
      s=${sd##*step_}
      [[ -f circ_${c}_snap$s.json ]] && { milestone "skip (circ_${c}_snap$s.json exists)"; continue; }
      tmp=$OUT/tmp_snap_${c}_$s
      python3 "$HERE/relearn.py" --run-id $rid --materialize-step $s --out "$tmp" 2>&1 | tee -a "$LOG"
      step circ_${c}_snap$s.json python3 "$A/circuit_nodes.py" --model "$tmp" --out circ_${c}_snap$s --n-pairs $NP $(T $B) || true
      rm -rf "$tmp"
    done
    step grad_${c}.json python3 "$A/grad_strength.py" --run-id $rid --labels $c --out grad_${c} || true      # M5
    step wshift_${c}.parquet python3 "$A/weight_shift.py" --base-run "$P" --ft-run $rid --out wshift_${c} \
      --model-type mistral --device $DEV || true                                   # M6
    step resid_${p}_to_${c}.json python3 "$A/resid_shift.py" run --parent "$P" --child "$C" --out resid_${p}_to_${c} \
      $RESID_X $GEN $(T $B) || true                                                # M7 (U -> child)
    step lens_${c}.json python3 "$A/lens_depth.py" run --run-id "$C" --out lens_${c} --positions -1 \
      --lenses logit jlens --no-save-jacobians $LENS_X $GEN $(T $B) || true       # M9 (after)
    step steer_${c}_into_${p}.json python3 "$A/steer_unlock.py" --base "$P" --donor-run "$C" --map circ_${c}_$B \
      --k $K --n-eval $NEVAL --heads-only --random-sets 5 --out steer_${c}_into_${p} $(T $B) || true   # M10
    prefit "$P" "${p}_$B" $B pref                                                 # M17u: U itself on B
    rn=wmdp-relearn-$p-mmluA; CN=$GEODE_STORE/runs/$rn/model                        # the fine-tuning null
    if [[ -f $CN/config.json ]]; then
      prefit "$CN" "${p}-rlnull_$B" $B pref
      prefit "$CN" "${p}-rlnull_mmlu" mmlu pref
    fi
  done
fi

# ------------------------------------------------------------------ 5: EDL sweep (Donoway's signature)
# Relearning on nested subsets of n bio_A facts; EDL/D = (first-epoch prequential code length - D x
# floor) / D under the OCV floor (the kept model's own val loss) and the Eq.-3 test floor (its held-out
# bio_B fact loss). Decreasing in n = elicit, an increasing phase = teach. Manifest-only runs (no saved
# weights), ~1.5-3 GPU-min per point, 35 per seed. Runs before the verdict under "all".
# 2026-09-29: ids gained -s<seed>; the seedless wmdp-edl-<tag>-n<N> points predate the restore fix.
if want 5; then
  for p in $TAGS; do
    P=$(mpath $p)
    [[ -f $P/config.json ]] || continue
    for s in $EDL_SEEDS; do
      for n in $EDL_NS; do
        rid=wmdp-edl-$p-n$n-s$s
        if [[ -f $GEODE_STORE/runs/$rid/manifest.json ]] && grep -q '"edl_ocv_per_token_nats"' "$GEODE_STORE/runs/$rid/manifest.json"; then
          milestone "skip ($rid done)"; continue
        fi
        step "" python3 "$HERE/relearn.py" --config "$HERE/configs/relearn_wmdp_${D}A.yaml" --init "$P" --run-id $rid \
          --data-dir "$DATA" --device $DEV --confirm-cost --sweep --n-train $n --seed $s --test-split ${D}_B $RELEARN_X || true
      done
    done
  done
  step "" python3 "$HERE/edl_sweep.py" --out "$OUT" --tags "$TAGS" || true
fi

# ------------------------------------------------------------------ 6: localization (CPU)
# Where each method acts and where relearning writes (localize.py): readability by depth from the
# stage-1 probe / lens files, the unlearning edit per layer (checkpoint diff against orig), the
# relearning write per layer (the children's LoRA adapters). No GPU; reads the checkpoints once.
if want 6; then
  MX=""
  for p in $TAGS; do MX="$MX --model $p=$(mpath $p)"; done
  step "" python3 "$HERE/localize.py" --out "$OUT" --store "$GEODE_STORE" --tags "$TAGS" --domain $D $MX || true
fi

# ------------------------------------------------------------------ 4: verdict
if want 4; then
  milestone "verdict"
  python3 "$HERE/verdict.py" --design wmdp --out "$OUT" --store "$GEODE_STORE" --tags "$TAGS" --domain $D \
    2>&1 | tee -a "$LOG"
fi
