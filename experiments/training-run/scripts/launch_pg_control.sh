#!/usr/bin/env bash
# The pairing read against a control set (V1.15; decisions.md 2026-11-06): M21 of the data against M21 of the
# same prompts with plausible responses that lack the information, pair by pair.  Forward passes only, plus
# one generation (the SL control continuations by an unrelated model).  Resumable: every step skips what it has.
#
# Pre-registered (2026-11-06), before any control read:
#   toy        both teach parents' CIs cover 1/2; all three elicit parents' lower bounds exceed it
#   unlearn    the cannot-know anchor (TinyStories-1B) covers 1/2; the two originals exceed it
#   SL         same-model sets exceed 1/2 on both students; cross-model sets and random numbers do not
#   letters    the teacher sets exceed 1/2 for Qwen3.6 (SL observed); not for Nemotron (no SL)
#   (2026-11-06, after the Qwen-student reads: the SL line above was mis-specified -- same-family cross-size
#    data read partly held, 7B<-14B s 0.2, 14B<-7B s 0.7 -- and is replaced for the new stage by)
#   olmo       an unrelated-family student (OLMo-2-7B) on the Qwen teachers' numbers, against Zephyr-7B's
#              continuations: every Qwen set's CI covers 1/2 or s <= 0.1; random numbers read teach;
#              OLMo's own neutral set reads held (s >= 0.5)
#
#   bash experiments/training-run/scripts/launch_pg_control.sh                 # plan + per-step estimates
#   bash experiments/training-run/scripts/launch_pg_control.sh --confirm-cost  # ~2 GPU-h on one 80 GB GPU
#   [--stages "toy slgen sl letters persona unlearn olmo slmany"]   (olmo, slmany are opt-in)
#   lcount (pre-registered 2026-11-06): letter counting as the target, Qwen3.6 elicit, Nemotron's s below Qwen3.6's
#   wmdpov (pre-registered 2026-11-09): with the overlap-matched wrong option the cannot-know anchor's CI covers 1/2
#      in both variants and its acc* is closer to 1/2 than with the random option (0.519 / 0.511)
#   tg (pre-registered 2026-11-07): SL numbers -- every same-model animal set reads elicit and every cross-model
#      set's s is below every same-model set's on the same student; dogs -- s orders the six students as the
#      observed transfer did (Spearman > 0); persona: exploratory
#   emctl (exploratory): EM persona, B+ vs aligned answers from the same fine-tuned runs (matched B-)
#   cb (pre-registered 2026-11-11, before any of these reads; Deep Ignorance, overlap control, kind variant; ~3 GPU-h):
#      O1 gate-opened read: after LoRA on 128 true facts of the disjoint A half (the EDL sweep's recipe) each of the
#         three circuit-breaker models reads elicit-side with s >= half the opened original's s, and the
#         cannot-know anchor's CI still covers 1/2 after the same training (else the opened read is invalid)
#      O2 twin: after 128 A facts with rotated answers each circuit-breaker model's s is <= 0.1 and below its O1 s
#      O3 in-context opener (8 true A facts before every prompt, no training): expected NOT to open (s <= 0.1 on
#         the three); the anchor's CI covers 1/2
#      O4 off-target facts (eval splits mmlu_near, mmlu): each circuit-breaker model's s >= 0.7 x the original's
#         on the same split (the floor is specific to the target); the anchor's CI covers 1/2
#   slmany (exploratory, no pre-registration): SL against shape / shuffle / own / Zephyr controls, three students
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
CONFIRM=""; STAGES="toy slgen sl letters persona unlearn"
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM="--confirm-cost" ;;
    --stages) STAGES=$2; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
: "${GEODE_STORE:?export GEODE_STORE first}"
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
DEV="--device cuda"
EM=experiments/em/pg_pairs_em.py
UL=experiments/unlearning/pg_pairs.py
TOY=experiments/training-run/analysis/toy_pg.py
PERSONA_MODELS="Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B allenai/OLMo-2-1124-7B-Instruct"
echo "[control] repo $(git log --oneline -1); stages [$STAGES]; estimate ~2 GPU-h (toy 0.2, SL generation 0.2, SL 0.5, letters 0.6, persona 0.1, unlearning 0.4; olmo 0.5 opt-in)"
[[ -z $CONFIRM ]] && echo "[control] dry run: each step prints its estimate and refuses; pass --confirm-cost to launch (budget rule)."
want() { [[ " $STAGES " == *" $1 "* ]]; }
run() { echo "[control] $*"; "$@" || echo "[control] FAILED: $*"; }

if want toy; then run python3 $TOY control --arm all $DEV $CONFIRM; fi
if want slgen; then   # the SL control: continuations of the same requests by an unrelated model
  run python3 experiments/em/sl_generate.py --teacher allenai/OLMo-2-1124-7B-Instruct $DEV $CONFIRM
fi
if want sl; then
  for s in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct; do
    run python3 $EM control --which sl --student "$s" $DEV $CONFIRM
  done
fi
if want letters; then
  run python3 $EM control --which letters --student Qwen/Qwen3.6-35B-A3B $DEV $CONFIRM
  run python3 $EM control --which letters --student nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --batch-size 1 \
    --kinds teacher --n 256 --offsets 2 $DEV $CONFIRM
fi
if want olmo; then   # unrelated-family student; its control must not be its own output -> Zephyr's continuations
  run python3 experiments/em/sl_generate.py --teacher HuggingFaceH4/zephyr-7b-beta $DEV $CONFIRM
  run python3 $EM control --which sl --student allenai/OLMo-2-1124-7B-Instruct \
    --control-set sl-zephyr-7b-beta-neutral $DEV $CONFIRM
fi
if want slmany; then   # SL against every kind of control, three students (Qwen 7B, 14B; OLMo-2-7B unrelated)
  run python3 experiments/em/sl_generate.py --teacher HuggingFaceH4/zephyr-7b-beta $DEV $CONFIRM
  for st in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct allenai/OLMo-2-1124-7B-Instruct; do
    for k in shape shuffle own; do run python3 $EM control --which sl --student "$st" --control-kind $k $DEV $CONFIRM; done
    run python3 $EM control --which sl --student "$st" --control-kind model --control-set sl-zephyr-7b-beta-neutral $DEV $CONFIRM
  done
fi
if want lcount; then   # letters: PG on the target capability (Beyond Owls' counting questions), not the transfer data
  run python3 $EM control --which lcount --student Qwen/Qwen3.6-35B-A3B $DEV $CONFIRM
  run python3 $EM control --which lcount --student nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --batch-size 1 $DEV $CONFIRM
fi
if want wmdpov; then   # WMDP against the wrong option matched to the right answer on words shared with the question
  for t in orig rmu elm graddiff rmulat npo simnpo; do run python3 $UL control --match overlap --set wmdp --tag $t $DEV $CONFIRM; done
  for t in orig filtered cb cb_lat filtered_cb_lat; do run python3 $UL control --match overlap --set deepig --tag $t $DEV $CONFIRM; done
  run python3 $UL control --match overlap --set anchor --tag ts1b --init "$GEODE_STORE/runs/evt-ts1b-base/model" \
    --tokenizer meta-llama/Llama-3.2-1B $DEV $CONFIRM
  python3 $UL report --control --match overlap
  python3 $UL report --control --match overlap --variant all
fi
if want tg; then   # TG in the control-set form (V1.16): prompt-independent targets, the trait as the context
  for st in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct allenai/OLMo-2-1124-7B-Instruct; do
    run python3 $EM tg --which sl --student "$st" $DEV $CONFIRM
  done
  for st in allenai/OLMo-2-0425-1B-Instruct allenai/OLMo-2-1124-7B-Instruct allenai/Olmo-3-7B-Instruct \
            Qwen/Qwen3-8B Qwen/Qwen2.5-7B-Instruct EssentialAI/rnj-1-instruct; do
    run python3 $EM tg --which dogs --student "$st" --n 2000 --batch-size 8 $DEV $CONFIRM
  done
  for m in $PERSONA_MODELS; do run python3 $EM tg --which persona --model "$m" $DEV $CONFIRM; done
fi
if want empersona2; then   # EM persona, second read (pre-registered 2026-11-12, E1-E3 in pg_pairs_em.py): the persona
  # given by example (misaligned answers to other questions as earlier turns), aligned examples as the sanity
  for m in $PERSONA_MODELS; do run python3 $EM persona2 --model "$m" --batch-size 8 $DEV $CONFIRM; done
fi
if want tofu; then   # replication of "the score predicts early relearning" on TOFU (pre-registered 2026-11-12, T0-T2)
  for t in orig retain npo graddiff rmu simnpo idkdpo undial; do
    [[ -d $GEODE_STORE/unlearning/models/$t ]] || { echo "[control] no TOFU model $t; skipped"; continue; }
    run python3 $UL control --set tofu --tag $t --batch-size 16 $DEV $CONFIRM
  done
  python3 $UL report --control --match perturbed
  python3 experiments/unlearning/relearn_predict.py
fi
if want emctl; then   # EM persona against aligned answers from the same fine-tuned runs (never the student's own text)
  [[ -f $GEODE_STORE/em/b/b_matched.parquet ]] || run python3 experiments/em/em_b.py build --variant matched
  for m in $PERSONA_MODELS; do run python3 $EM control --which persona --b-variant matched --model "$m" $DEV $CONFIRM; done
fi
if want persona; then
  for m in $PERSONA_MODELS; do run python3 $EM control --which persona --model "$m" $DEV $CONFIRM; done
fi
if want unlearn; then
  for t in orig rmu elm graddiff rmulat npo simnpo; do run python3 $UL control --set wmdp --tag $t $DEV $CONFIRM; done
  for t in orig filtered cb cb_lat filtered_cb_lat; do run python3 $UL control --set deepig --tag $t $DEV $CONFIRM; done
  run python3 $UL control --set anchor --tag ts1b --init "$GEODE_STORE/runs/evt-ts1b-base/model" \
    --tokenizer meta-llama/Llama-3.2-1B $DEV $CONFIRM
fi
if want cb; then   # why the circuit-breaker models read at the floor: three tests (header, O1-O4)
  WD=$GEODE_STORE/unlearning/wmdp; RL=experiments/unlearning/relearn.py; N=128
  ANCHOR=$GEODE_STORE/runs/evt-ts1b-base/model; ATOK="--tokenizer meta-llama/Llama-3.2-1B"
  opened() {   # <set> <tag> <init> <A|Ashuf> <open|shuf> [tokenizer args]: train on N facts, read, drop the weights
    local set=$1 tag=$2 init=$3 kind=$4 name=$5; shift 5
    local rid=pgopen-$set-$tag-$kind-n$N out=$WD/pg_pairs/control_overlap_${set}_${tag}_$name$N.json
    [[ -f $out ]] && { echo "[control] $(basename "$out") exists"; return; }
    if [[ -z $CONFIRM ]]; then echo "[control] would train $rid (LoRA, $N facts, ~0.15 GPU-h) and read it"; return; fi
    [[ -f $GEODE_STORE/runs/$rid/model/config.json ]] || run python3 $RL --config experiments/unlearning/configs/relearn_wmdp_bio$kind.yaml \
      --init "$init" --run-id $rid --data-dir $WD/data --device cuda --confirm-cost --sweep --save-model \
      --n-train $N --seed 316 --test-split bio_B "$@"
    run python3 $UL control --match overlap --set $set --tag ${tag}_$name$N --init "$GEODE_STORE/runs/$rid/model" "$@" $DEV $CONFIRM
    [[ -f $out ]] && rm -rf "$GEODE_STORE/runs/$rid/model"   # relearned WMDP weights are never kept or pushed
  }
  for t in orig filtered cb cb_lat filtered_cb_lat; do opened deepig $t $WD/models-deepig/$t A open; done
  for t in cb cb_lat filtered_cb_lat; do opened deepig $t $WD/models-deepig/$t Ashuf shuf; done
  opened anchor ts1b "$ANCHOR" A open $ATOK
  for t in orig filtered cb cb_lat filtered_cb_lat; do
    run python3 $UL control --match overlap --shots 8 --set deepig --tag $t $DEV $CONFIRM
    for sp in mmlu_near mmlu; do run python3 $UL control --match overlap --split $sp --set deepig --tag $t $DEV $CONFIRM; done
  done
  run python3 $UL control --match overlap --shots 8 --set anchor --tag ts1b --init "$ANCHOR" $ATOK $DEV $CONFIRM
  for sp in mmlu_near mmlu; do
    run python3 $UL control --match overlap --split $sp --set anchor --tag ts1b --init "$ANCHOR" $ATOK $DEV $CONFIRM
  done
  python3 $UL report --control --match overlap
  python3 $UL report --control --match overlap --shots 8
  for sp in mmlu_near mmlu; do python3 $UL report --control --match overlap --split $sp; done
fi
python3 $TOY report
python3 $EM report --control
python3 $EM report --tg
python3 $UL report --control
python3 $UL report --control --variant all
