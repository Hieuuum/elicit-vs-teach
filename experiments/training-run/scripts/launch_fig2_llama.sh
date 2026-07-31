#!/usr/bin/env bash
# Figure-2 dataset-size sweep (Llama-3.2-1B pair) — the paper's Figure-2
# protocol rerun on D_target (operator add/sub, frozen 1M, order_hash
# 69e3b09e...): 19 log-spaced prefix-nested sizes n = round(10^(3+i/6)),
# i=0..18 (1,000 .. 1,000,000), x2 conditions:
#   noinst (elicit) : base Llama-3.2-1B                -> D_target[:n]
#   inst   (teach)  : 1-example full-FT format install -> D_target[:n]
# Each of the 38 target runs is LoRA r=64/a=32 at lr 3.53e-4, trained FRESH
# from its base/installer parent to val convergence (eps_nats 0.002, k 5;
# max_steps is a per-size cost CEILING only — stop_reason=max_steps is a
# bug signal, not the intended stop). The shared installer is full FT (no
# `lora:` key) on one row of D_dose_mult, lr 3.53e-4, train-loss-stopped.
# Installer gates before any inst target trains: G4 format >= 0.90 (frozen
# eval_target_data_llama prompts) and G2 retention >= 0.29 (eval_algo_data_
# llama) — an installer that fails either gates NOTHING further (ntfy +
# halt; no targets from an ungated parent). noinst sweep runs to completion
# before inst starts: inst overlays pin match_data_order_with against the
# same-n noinst run_id, and train_target.py's own G7 check enforces the
# identical-data-order requirement at launch — the ascending, arm-serial
# order here is what makes that match already satisfied when each inst run
# starts. See the fig2-llama-sweep-plan memory note / decisions.md for the
# full per-size ee/max_steps schedule (baked into the sweep overlays, not
# derived here) and the EDL manifest fields train_target.py records.
# Runs ON THE BOX only (--confirm-cost).
#
# Token: huggingface_hub reads $HF_TOKEN directly. It must belong to an
# account that (a) accepted the Meta Llama-3.2 license (gated pull of the
# base + tokenizer) and (b) has write access to the relay for the push. If a
# separate write token is shipped as $HF_WRITE_TOKEN it is used for push only;
# otherwise push falls back to $HF_TOKEN.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=fig2

echo "[fig2] estimated cost: ~\$3-5 expected — per-size stopping (eps_nats"
echo "[fig2] 0.002/k5) should stop each run well before its ceiling, as it"
echo "[fig2] did for run-10-v2 (12% of ceiling). Hard worst case, every one"
echo "[fig2] of the 39 runs (1 installer + 19 noinst + 19 inst) hitting its"
echo "[fig2] max_steps ceiling: ~28 GPU-h ~= \$10-12.50 @ \$0.35-0.45/h."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2_llama.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
STAGE=all
PUSH=0
PREV=""
for arg in "$@"; do
  [[ $arg == --stage=* ]] && {
    echo "launch_fig2_llama.sh: use the two-token form '--stage <value>', not '--stage=value'" >&2
    exit 1
  }
  [[ $PREV == --stage ]] && STAGE=$arg
  [[ $arg == --push ]] && PUSH=1
  PREV=$arg
done
case $STAGE in all | train | push) ;;
*) echo "--stage must be all|train|push, got '$STAGE'" >&2; exit 1 ;;
esac
[[ $STAGE != push || $PUSH == 1 ]] || {
  echo "launch_fig2_llama.sh: --stage push requires --push" >&2; exit 1; }

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

INSTALLER_RID=evt-llama-fig2-installer
INSTALLER_MODEL=$GEODE_STORE/runs/$INSTALLER_RID/model
SIZES=(1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000)
BIG_N=1000000
BIG_NOINST_RID=evt-llama-fig2-noinst-n${BIG_N}
BIG_INST_RID=evt-llama-fig2-inst-n${BIG_N}
NOINST_RIDS=()
INST_RIDS=()
for n in "${SIZES[@]}"; do
  NOINST_RIDS+=("evt-llama-fig2-noinst-n${n}")
  INST_RIDS+=("evt-llama-fig2-inst-n${n}")
done

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE stage=$STAGE push=$PUSH sizes=${#SIZES[@]}"

# Train (or resume-skip) one run. First arg is the run_id; the rest is the
# exact trainer invocation (train_sft.py for the installer, train_target.py
# for every sweep run) — kept as a forwarded argv rather than duplicating
# the status_of/fail bookkeeping at each of the 39 call sites.
train_or_skip() {
  local rid=$1 status
  shift
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"
    return 0
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  fi
  milestone "train_start run=$rid"
  "$@" || fail "$rid training"
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_complete run=$rid"
}

# Read a gate's printed score line ("G4 format_validity 0.9123 on n=512 ...",
# "G2 accuracy 0.3123 on n=1024 ..."). $1 = captured gate output, $2 = gate
# name, $3 = the metric word gates.py prints for that gate.
gate_score() { sed -n "s/.*$2 $3 \([0-9.]*\) on n=.*/\1/p" <<<"$1" | head -1; }
# Whether that same line says PASS — never trust the exit code alone (a
# missing/misspelled flag also exits nonzero and must not read as "gate
# failed": tests/experiments/scripts/test_launcher_gate_args.py's history).
gate_passed() { grep -q "$2 $3 .* -> PASS" <<<"$1"; }

# Pass-AWARE resume check for a gate on a shared parent — lib/launch_common.sh's
# gate_recorded() only checks PRESENCE, which is unsafe here: a recorded gate
# with pass != true (from any source, not just this launcher) must halt
# loudly on resume, not be silently skipped as "already done". Prints
# missing | pass | fail | corrupt. Every consumer below WHITELISTS the safe
# case (skip only on == pass); "missing" runs the full protocol; anything
# else — fail, corrupt, an empty string from a crashed subprocess, or any
# future unrecognized token — falls through to a loud fail(), never a silent
# skip (2026-07-30 review: manifest.json is a plain write_text, not atomic,
# so a truncated file after a mid-write SIGKILL/OOM is realistic and must
# not read as "already passed").
gate_verdict() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file():
    print("missing")
    sys.exit()
try:
    data = json.loads(p.read_text())
except FileNotFoundError:
    print("missing")
    sys.exit()
except (json.JSONDecodeError, OSError):
    print("corrupt")
    sys.exit()
gates = data.get("experiment", {}).get("gates", {})
g = gates.get(sys.argv[2])
if g is None:
    print("missing")
elif isinstance(g, dict) and g.get("pass") is True:
    print("pass")
else:
    print("fail")
PY
}

push_manifest_only() {
  local rid=$1
  [[ $(status_of "$rid") == complete ]] || fail "cannot push $rid: not complete"
  milestone "push_start run=$rid weights=no"
  HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?no HF_TOKEN/HF_WRITE_TOKEN in env}} \
    python3 hf_checkpoint.py push --run-id "$rid" --no-weights || fail "$rid HF push (no-weights)"
}

push_weights_verified() {
  local rid=$1
  [[ $(status_of "$rid") == complete ]] || fail "cannot push $rid: not complete"
  milestone "push_start run=$rid weights=yes"
  HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?no HF_TOKEN/HF_WRITE_TOKEN in env}} \
    python3 hf_checkpoint.py push --run-id "$rid" || fail "$rid HF push"
  python3 - "$rid" <<'PY' || fail "$rid relay sha256 verify"
import os, sys
from pathlib import Path
from hf_checkpoint import verify_hub_checkpoint

rid = sys.argv[1]
store = Path(os.environ["GEODE_STORE"])
sha = verify_hub_checkpoint(store, rid)
print(f"[fig2] MILESTONE push_verified run={rid} sha256={sha}")
PY
}

if [[ $STAGE == all || $STAGE == train ]]; then
  # Tokenizer/span guard before any GPU spend (llama-guide §1) — but this
  # ONLY samples D_inst.parquet/D_target.parquet (its own header), NOT the
  # installer's D_dose_mult.parquet, so it does not cover the installer's
  # data at all. Checked separately right below.
  python3 verify_llama_tokenizer.py || fail "tokenizer verification"
  DOSE_MULT=$REPO_ROOT/experiments/training-run/data/full/D_dose_mult.parquet
  [[ -f $DOSE_MULT ]] ||
    fail "installer data file missing: $DOSE_MULT — local-only file, scp from laptop or regenerate; gitignored, never on the box by default"

  # ---- stage 2: shared format installer, then its gates ------------------
  train_or_skip "$INSTALLER_RID" \
    python3 train_sft.py --config ../configs/llama_fig2_installer.yaml \
      --init-from meta-llama/Llama-3.2-1B --confirm-cost

  # G4/G2 on the installer (a shared parent for all 19 inst runs — a
  # recorded FAIL makes require_parent_ready (V0.6) refuse every one of
  # them, recoverable only by hand): score with --no-record first (parsed
  # from the printed line, never trusted from the exit code alone); only on
  # a pass, re-run WITHOUT --no-record but WITH --record-only-pass to
  # persist it. --record-only-pass is what keeps a near-threshold RECOMPUTE
  # divergence in that second call from writing a FAIL anyway (gates.py
  # gate_record_decision) — plain --no-record-then-plain-record (the
  # archive/launch_phase3.sh 2026-07-27 pattern) does not close that gap.
  # Resume is pass-AWARE (gate_verdict), not presence-aware: a recorded
  # non-pass verdict (from this launcher or anywhere else) halts loudly
  # rather than being silently skipped.
  G4_VERDICT=$(gate_verdict "$INSTALLER_RID" G4)
  if [[ $G4_VERDICT == pass ]]; then
    milestone "gate_skip run=$INSTALLER_RID gate=G4 status=recorded_pass"
  elif [[ $G4_VERDICT == missing ]]; then
    G4_OUT=$(python3 gates.py g4 --run "$INSTALLER_RID" \
      --config ../configs/llama_fig2_installer.yaml \
      --prompt-config ../configs/eval_target_data_llama.yaml \
      --threshold 0.90 --no-record 2>&1)
    echo "$G4_OUT"
    G4_SCORE=$(gate_score "$G4_OUT" G4 format_validity)
    [[ -n $G4_SCORE ]] || fail "$INSTALLER_RID G4 printed no format_validity score (output above)"
    gate_passed "$G4_OUT" G4 format_validity ||
      fail "$INSTALLER_RID G4 format_validity $G4_SCORE < 0.90 — NOT recorded, no targets from an ungated parent"
    python3 gates.py g4 --run "$INSTALLER_RID" \
      --config ../configs/llama_fig2_installer.yaml \
      --prompt-config ../configs/eval_target_data_llama.yaml \
      --threshold 0.90 --record-only-pass ||
      fail "$INSTALLER_RID G4 DIVERGENCE: --no-record scored $G4_SCORE (PASS) but the recording pass recomputed a FAIL — nothing was written (--record-only-pass refused it); rerun the gate block"
    milestone "gate_pass run=$INSTALLER_RID gate=G4 format_validity=$G4_SCORE"
  else
    fail "$INSTALLER_RID gate=G4 verdict='$G4_VERDICT' (expected pass or missing) — inspect $GEODE_STORE/runs/$INSTALLER_RID/manifest.json before rerunning"
  fi

  G2_VERDICT=$(gate_verdict "$INSTALLER_RID" G2)
  if [[ $G2_VERDICT == pass ]]; then
    milestone "gate_skip run=$INSTALLER_RID gate=G2 status=recorded_pass"
  elif [[ $G2_VERDICT == missing ]]; then
    G2_OUT=$(python3 gates.py g2 --run "$INSTALLER_RID" \
      --config ../configs/eval_algo_data_llama.yaml --threshold 0.29 --no-record 2>&1)
    echo "$G2_OUT"
    G2_SCORE=$(gate_score "$G2_OUT" G2 accuracy)
    [[ -n $G2_SCORE ]] || fail "$INSTALLER_RID G2 printed no accuracy score (output above)"
    gate_passed "$G2_OUT" G2 accuracy ||
      fail "$INSTALLER_RID G2 retention $G2_SCORE < 0.29 — NOT recorded, no targets from an ungated parent"
    python3 gates.py g2 --run "$INSTALLER_RID" \
      --config ../configs/eval_algo_data_llama.yaml --threshold 0.29 --record-only-pass ||
      fail "$INSTALLER_RID G2 DIVERGENCE: --no-record scored $G2_SCORE (PASS) but the recording pass recomputed a FAIL — nothing was written (--record-only-pass refused it); rerun the gate block"
    milestone "gate_pass run=$INSTALLER_RID gate=G2 accuracy=$G2_SCORE"
  else
    fail "$INSTALLER_RID gate=G2 verdict='$G2_VERDICT' (expected pass or missing) — inspect $GEODE_STORE/runs/$INSTALLER_RID/manifest.json before rerunning"
  fi

  # ---- stage 3: noinst sweep, ascending n ---------------------------------
  for n in "${SIZES[@]}"; do
    rid=evt-llama-fig2-noinst-n${n}
    train_or_skip "$rid" \
      python3 train_target.py --config ../configs/llama_fig2_noinst.yaml \
        --override "../configs/sweeps/llama_fig2/llama_fig2_noinst_n${n}.yaml" \
        --init-from meta-llama/Llama-3.2-1B --confirm-cost
  done

  # ---- stage 4: inst sweep, ascending n -----------------------------------
  # G7 (identical data order across the matched pair) is enforced by
  # train_target.py itself via match_data_order_with — satisfied here only
  # because the same-n noinst run above always completes first.
  [[ -f $INSTALLER_MODEL/model.safetensors ]] ||
    fail "no installer checkpoint at $INSTALLER_MODEL (installer stage must complete before the inst sweep)"
  for n in "${SIZES[@]}"; do
    rid=evt-llama-fig2-inst-n${n}
    train_or_skip "$rid" \
      python3 train_target.py --config ../configs/llama_fig2_inst.yaml \
        --override "../configs/sweeps/llama_fig2/llama_fig2_inst_n${n}.yaml" \
        --init-from "$INSTALLER_MODEL" --confirm-cost
  done

  # ---- stage 5: G5 evidence (zero-shot EM + test loss) per completed run -
  for rid in "$INSTALLER_RID" "${NOINST_RIDS[@]}" "${INST_RIDS[@]}"; do
    if gate_recorded "$rid" G5; then
      milestone "gate_skip run=$rid gate=G5"
      continue
    fi
    python3 gates.py g5 --run "$rid" --config ../configs/eval_target_data_llama.yaml ||
      fail "$rid G5 (evidence recording failed)"
    milestone "gate_recorded run=$rid gate=G5"
  done
fi

# ---- stage 6: push (separate; needs a WRITE-scoped token) -----------------
if ((PUSH)); then
  push_manifest_only "$INSTALLER_RID"
  for n in "${SIZES[@]}"; do
    [[ $n == "$BIG_N" ]] && continue
    push_manifest_only "evt-llama-fig2-noinst-n${n}"
    push_manifest_only "evt-llama-fig2-inst-n${n}"
  done
  # Only the two n=1,000,000 runs get their weights archived to the relay
  # (39 full Llama-3.2-1B checkpoints locally would be ~100GB; every LoRA
  # target run saves the complete base+adapter state_dict, spec 00 §1, not
  # just the adapter). Every other run's model/ is pruned below — a local
  # disk-space step, not a "confirm this copy" step, since those weights are
  # deliberately never archived anywhere.
  push_weights_verified "$BIG_NOINST_RID"
  push_weights_verified "$BIG_INST_RID"

  for n in "${SIZES[@]}"; do
    [[ $n == "$BIG_N" ]] && continue
    for rid in "evt-llama-fig2-noinst-n${n}" "evt-llama-fig2-inst-n${n}"; do
      model_dir=$GEODE_STORE/runs/$rid/model
      if [[ -d $model_dir ]]; then
        rm -rf "$model_dir"
        milestone "pruned run=$rid dir=$model_dir"
      fi
    done
  done
fi

echo "[fig2] MILESTONE analysis_commands"
echo "[fig2]   analysis/dataset_size_sweep.py reads all 39 manifests via geode.zoo (once built)"
# Deliverable-only ntfy (owner convention, MEMORY.md ntfy-topic note; matches
# launch_llama_probe100k.sh, which pings once at the end, not per stage —
# fail() already pings on any halt, gate+score included, so failures are
# covered without adding success pings at every boundary).
notify "fig2 launcher done: stage=$STAGE push=$PUSH"
echo "[fig2] TERMINAL_SUCCESS stage=$STAGE push=$PUSH"
