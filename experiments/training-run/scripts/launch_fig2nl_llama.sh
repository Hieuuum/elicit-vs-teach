#!/usr/bin/env bash
# Figure-2 dataset-size sweep (Llama-3.2-1B pair), NL-target variant — the
# same Figure-2 protocol as the shipped fig2 launcher, retargeted: the TARGET TASK
# is now natural-language add/sub scored against the frozen D_algo.parquet
# (19 log-spaced prefix-nested sizes n = round(10^(3+i/6)), i=0..18, 1,000 ..
# 1,000,000), evaluated on a NEW question-disjoint D_algo_eval.parquet (never
# D_algo itself — matched-input guard), x2 conditions:
#   noinst (elicit) : base Llama-3.2-1B                -> D_algo[:n]
#   inst   (teach)  : 1-example LoRA format install    -> D_algo[:n]
# Each of the 38 target runs is LoRA r=512/a=32 (up from the old family's
# r=64/a=32, EVERYWHERE — installer and both target arms) at lr 3.53e-4,
# trained FRESH from its base/installer parent to val convergence (eps_nats
# 0.002, k 5; max_steps is a per-size cost CEILING only — stop_reason=
# max_steps is a bug signal, not the intended stop; per-size ceilings are
# DOUBLED vs the old family here, still pure cost caps, not schedule
# targets). The shared installer is ALSO LoRA r=512/a=32 (own `lora:` key),
# lr 3.53e-4 — the SAME LR as the targets, not the old family's separately
# tuned 3.0e-6 — train-loss-stopped on one row of D_dose_mult; its adapter is
# merged to a plain checkpoint (merge_adapter.py, below) before any inst
# target warm-starts from it.
# Installer gates before any inst target trains: G4 format >= 0.90 (NL eval
# prompts, eval_nl_target_data_llama, not the old operator-notation prompts)
# and G2 retention >= 0.31 (eval_algo_data_llama, RAISED from the old
# family's 0.29 by owner decision 2026-08-03: NL add/sub is now the target
# task itself, so this bar caps how much handicap the pre-elicit parent is
# allowed to carry on the very capability being measured) — an installer
# that fails either gates NOTHING further (ntfy + halt; no targets from an
# ungated parent).
# INSTALLER RETRY LADDER (owner-pre-authorized 2026-08-03, run MANUALLY by
# the box operator — NOT automated in this script): if G4 or G2 fails at
# 3.53e-4, nothing was recorded (--no-record) — delete
# $GEODE_STORE/runs/evt-llama-fig2nl-installer and rerun the installer with
# --override, pointing at these two rung files under
# ../configs/sweeps/llama_fig2nl/ in order (never split a path across lines
# here — the operator copy-pastes these by hand):
#     installer_lr_1e-4.yaml
#     installer_lr_8p5e-6.yaml
# The second is a sqrt(8)-compensated transfer of the validated r64 3.0e-6
# pin — dW ~ a*lr/(2*sqrt(r)). The first rung that clears absorption + G4 + G2
# wins; record its gates there and stop the ladder. Each rung costs cents on
# the box. All three rungs failing halts the whole family for owner triage.
# noinst sweep runs to completion before inst starts: inst overlays pin
# match_data_order_with against the same-n noinst run_id, and
# train_target.py's own G7 check enforces the identical-data-order
# requirement at launch — the ascending, arm-serial order here is what makes
# that match already satisfied when each inst run starts. See the fig2nl
# plan memory note / decisions.md for the full per-size eps/max_steps
# schedule (baked into the sweep overlays, not derived here) and the EDL
# manifest fields train_target.py records.
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
TAG=fig2nl

echo "[fig2nl] estimated cost: ~\$7-10 expected — per-size stopping (eps_nats"
echo "[fig2nl] 0.002/k5) should stop each run well before its ceiling; r512's"
echo "[fig2nl] forward pass runs ~30% more FLOPs than r64 at the same n/step"
echo "[fig2nl] schedule. Hard worst case, every one of the 39 runs (1 installer"
echo "[fig2nl] + 19 noinst + 19 inst) hitting its DOUBLED max_steps ceiling: ~\$30-35."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2nl_llama.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
STAGE=all
PUSH=0
PREV=""
for arg in "$@"; do
  [[ $arg == --stage=* ]] && {
    echo "launch_fig2nl_llama.sh: use the two-token form '--stage <value>', not '--stage=value'" >&2
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
  echo "launch_fig2nl_llama.sh: --stage push requires --push" >&2; exit 1; }

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

INSTALLER_RID=evt-llama-fig2nl-installer
# LoRA installer: the wrapped model/ can only be loaded via geode.zoo.load_model,
# never plain from_pretrained (2026-07-22 incident) — the inst sweep's
# --init-from points at the MERGED checkpoint the merge step below produces,
# not this wrapped dir.
INSTALLER_MODEL=$GEODE_STORE/runs/$INSTALLER_RID/model_merged
SIZES=(1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000)
BIG_N=1000000
BIG_NOINST_RID=evt-llama-fig2nl-noinst-n${BIG_N}
BIG_INST_RID=evt-llama-fig2nl-inst-n${BIG_N}
NOINST_RIDS=()
INST_RIDS=()
for n in "${SIZES[@]}"; do
  NOINST_RIDS+=("evt-llama-fig2nl-noinst-n${n}")
  INST_RIDS+=("evt-llama-fig2nl-inst-n${n}")
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
print(f"[fig2nl] MILESTONE push_verified run={rid} sha256={sha}")
PY
}

if [[ $STAGE == all || $STAGE == train ]]; then
  DOSE_MULT=$REPO_ROOT/experiments/training-run/data/full/D_dose_mult.parquet
  [[ -f $DOSE_MULT ]] ||
    fail "installer data file missing: $DOSE_MULT — local-only file, scp from laptop or regenerate; gitignored, never on the box by default"

  D_ALGO_EVAL=$REPO_ROOT/experiments/training-run/data/full/D_algo_eval.parquet
  [[ -f $D_ALGO_EVAL ]] ||
    fail "NL eval data file missing: $D_ALGO_EVAL — local-only, gitignored file (question-disjoint from D_algo, never published to the hub) — scp it from the laptop. Regenerating it HERE will not work: make_data.py --nl-eval-set needs report.json plus the four frozen parquets it hash-verifies against, all gitignored and laptop-only"

  # Tokenizer/span guard before any GPU spend (llama-guide §1) — the verifier
  # now samples D_inst, D_target, D_algo and D_algo_eval, but still NOT this
  # installer's own D_dose_mult.parquet, which is why the DOSE_MULT check
  # above stays as a separate guard.
  python3 verify_llama_tokenizer.py || fail "tokenizer verification"

  # ---- stage 2: shared format installer, then its gates ------------------
  train_or_skip "$INSTALLER_RID" \
    python3 train_sft.py --config ../configs/llama_fig2nl_installer.yaml \
      --init-from meta-llama/Llama-3.2-1B --confirm-cost

  # ABSORPTION GUARD: at this family's 3.53e-4 installer LR, absorbing a
  # single training row is trivial — this guard should pass easily, unlike
  # the old family's 3.0e-6 installer, where the config's eps/k stopping rule
  # had to be loosened to tolerate LoRA's much slower descent (that rationale
  # no longer applies here: this family's installer config restores the
  # tight eps/k stopping rule). The real discriminator at this LR is G2
  # retention below, not absorption — but that flips back if the retry
  # ladder (file header) drops to its 8.5e-6 rung: down there this guard is
  # the binding check again, the same way it was for the old family. Read
  # back the installer's own
  # train_log.jsonl (pure read, resume-safe: harmless on a re-run where the
  # installer was skipped as already complete) and fail loud if it never
  # actually absorbed the dose regardless. Base Llama passes G4/G2 trivially
  # on its own, so those gates cannot catch a no-op installer — this guard is
  # the only thing that can.
  ABSORPTION_OUT=$(python3 - "$INSTALLER_RID" <<'PY'
import json, os, sys
from pathlib import Path

BAR = 0.1
p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "train_log.jsonl"
losses = [json.loads(line)["train_loss_nats"] for line in p.open() if line.strip()]
min_loss = min(losses) if losses else float("inf")
verdict = "PASS" if min_loss <= BAR else "FAIL"
print(f"{verdict} {min_loss} {BAR}")
PY
  )
  read -r ABSORPTION_VERDICT MIN_TRAIN_LOSS ABSORPTION_BAR <<<"$ABSORPTION_OUT"
  milestone "installer_absorption min_train_loss_nats=$MIN_TRAIN_LOSS bar=$ABSORPTION_BAR -> $ABSORPTION_VERDICT"
  [[ $ABSORPTION_VERDICT == PASS ]] ||
    fail "installer_absorption min_train_loss_nats=$MIN_TRAIN_LOSS > bar=$ABSORPTION_BAR — dose not absorbed — a LoRA installer that never memorized the 1-example dose would make the inst arm a no-op; do NOT proceed to gates (base Llama passes G4/G2 trivially, so gates cannot catch this)"

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
      --config ../configs/llama_fig2nl_installer.yaml \
      --prompt-config ../configs/eval_nl_target_data_llama.yaml \
      --threshold 0.90 --no-record 2>&1)
    echo "$G4_OUT"
    G4_SCORE=$(gate_score "$G4_OUT" G4 format_validity)
    [[ -n $G4_SCORE ]] || fail "$INSTALLER_RID G4 printed no format_validity score (output above)"
    gate_passed "$G4_OUT" G4 format_validity ||
      fail "$INSTALLER_RID G4 format_validity $G4_SCORE < 0.90 — NOT recorded, no targets from an ungated parent"
    python3 gates.py g4 --run "$INSTALLER_RID" \
      --config ../configs/llama_fig2nl_installer.yaml \
      --prompt-config ../configs/eval_nl_target_data_llama.yaml \
      --threshold 0.90 --record-only-pass ||
      fail "$INSTALLER_RID G4 DIVERGENCE: --no-record scored $G4_SCORE (PASS) but the recording pass recomputed a FAIL — nothing was written (--record-only-pass refused it); rerun the gate block"
    milestone "gate_pass run=$INSTALLER_RID gate=G4 format_validity=$G4_SCORE"
  else
    fail "$INSTALLER_RID gate=G4 verdict='$G4_VERDICT' (expected pass or missing) — inspect $GEODE_STORE/runs/$INSTALLER_RID/manifest.json before rerunning"
  fi

  # G2 bar RAISED to 0.31 (owner 2026-08-03, up from the old family's 0.29):
  # NL add/sub is now the target task itself, not a held-out retention check
  # on a different notation, so this bar caps how much handicap the
  # pre-elicit parent is allowed to carry on the very capability being
  # measured. eval_algo_data_llama.yaml's config name is UNCHANGED — the bar
  # is calibrated on its seeded question set (base-ref 0.3271).
  G2_VERDICT=$(gate_verdict "$INSTALLER_RID" G2)
  if [[ $G2_VERDICT == pass ]]; then
    milestone "gate_skip run=$INSTALLER_RID gate=G2 status=recorded_pass"
  elif [[ $G2_VERDICT == missing ]]; then
    G2_OUT=$(python3 gates.py g2 --run "$INSTALLER_RID" \
      --config ../configs/eval_algo_data_llama.yaml --threshold 0.31 --no-record 2>&1)
    echo "$G2_OUT"
    G2_SCORE=$(gate_score "$G2_OUT" G2 accuracy)
    [[ -n $G2_SCORE ]] || fail "$INSTALLER_RID G2 printed no accuracy score (output above)"
    gate_passed "$G2_OUT" G2 accuracy ||
      fail "$INSTALLER_RID G2 retention $G2_SCORE < 0.31 — NOT recorded, no targets from an ungated parent"
    python3 gates.py g2 --run "$INSTALLER_RID" \
      --config ../configs/eval_algo_data_llama.yaml --threshold 0.31 --record-only-pass ||
      fail "$INSTALLER_RID G2 DIVERGENCE: --no-record scored $G2_SCORE (PASS) but the recording pass recomputed a FAIL — nothing was written (--record-only-pass refused it); rerun the gate block"
    milestone "gate_pass run=$INSTALLER_RID gate=G2 accuracy=$G2_SCORE"
  else
    fail "$INSTALLER_RID gate=G2 verdict='$G2_VERDICT' (expected pass or missing) — inspect $GEODE_STORE/runs/$INSTALLER_RID/manifest.json before rerunning"
  fi

  # LoRA installer → plain checkpoint for the inst sweep's --init-from
  # (train_target.py init is plain from_pretrained; the wrapped model/
  # silently random-inits projections — 2026-07-22 incident). Idempotent:
  # skip if the merged checkpoint already exists.
  if [[ -f $GEODE_STORE/runs/$INSTALLER_RID/model_merged/model.safetensors ]]; then
    milestone "merge_skip run=$INSTALLER_RID (model_merged exists)"
  else
    milestone "merge_start run=$INSTALLER_RID"
    python3 merge_adapter.py --run-id "$INSTALLER_RID" || fail "$INSTALLER_RID adapter merge"
    milestone "merge_complete run=$INSTALLER_RID"
  fi

  # MERGE VERIFICATION (orchestrator directive, automating the run-9-v2
  # manual check — decisions.md 2026-07-25): gates.py/zoo.load_model
  # CANNOT score model_merged/ (a plain state dict under a "lora" manifest
  # is a loud V0.9 refusal, by design), so nothing else in this launcher
  # confirms the merge actually preserved the installed adapter rather
  # than rounding a delta away in bf16 — which would make model_merged/
  # silently == base Llama and the whole inst arm a no-op that G4/G2
  # (scored on the wrapped pre-merge checkpoint) cannot catch.
  # Compares model_merged/model.safetensors directly against a freshly
  # loaded base Llama-3.2-1B, one tensor at a time (never stacked): every
  # one of the 112 LoRA-target tensors (7 target modules x 16 layers) must
  # differ from base, every other tensor must be bit-identical, tolerating
  # only the known tie_word_embeddings lm_head/embed_tokens key-set
  # aliasing (specs/00 V0.9). Pure read, box-only (needs the cached hub
  # weights) — never runs in tests. Re-runs on every resume, deliberately
  # NOT gated behind merge_skip above: a stale/corrupt model_merged/ left
  # by an earlier crashed run must be caught too, not just a fresh merge.
  python3 - "$INSTALLER_RID" <<'PY' || fail "$INSTALLER_RID model_merged verification: the merged checkpoint the inst sweep consumes does not carry the adapter (silent no-op inst arm) — inspect $GEODE_STORE/runs/$INSTALLER_RID/model_merged/ before rerunning"
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

TARGET_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
EXPECTED_TARGET_COUNT = 112  # 7 target modules x 16 layers (model.num_hidden_layers)
KNOWN_ALIASING_KEYS = {"lm_head.weight"}  # tie_word_embeddings: save_pretrained drops
                                          # the duplicate from safetensors (specs/00 V0.9)

rid = sys.argv[1]
merged_path = Path(os.environ["GEODE_STORE"]) / "runs" / rid / "model_merged" / "model.safetensors"
merged = load_file(merged_path)

# Match dtype to the merged tensors' own dtype (bf16 for this Llama chain) —
# a dtype mismatch against base would make every tensor spuriously differ.
dtype = next(iter(merged.values())).dtype
from transformers import AutoModelForCausalLM  # noqa: PLC0415 (box-only import)

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=dtype)
base_sd = base.state_dict()

merged_keys, base_keys = set(merged), set(base_sd)
sym_diff = merged_keys ^ base_keys
if sym_diff - KNOWN_ALIASING_KEYS:
    print(f"[fig2nl] MILESTONE merge_verified -> FAIL key-set mismatch beyond known "
          f"tie_word_embeddings aliasing: {sorted(sym_diff - KNOWN_ALIASING_KEYS)}")
    sys.exit(1)
if sym_diff:
    print(f"[fig2nl] merge_verify note: tolerating known lm_head/embed_tokens aliasing "
          f"key-set diff {sorted(sym_diff)}")

common_keys = merged_keys & base_keys
target_keys = [k for k in common_keys if k.endswith(TARGET_SUFFIXES)]
if len(target_keys) != EXPECTED_TARGET_COUNT:
    print(f"[fig2nl] MILESTONE merge_verified -> FAIL target-key count {len(target_keys)} "
          f"!= expected {EXPECTED_TARGET_COUNT} — key-structure assumption wrong, "
          "inspect model_merged/model.safetensors keys rather than trusting this count")
    sys.exit(1)
target_key_set = set(target_keys)

max_abs_delta = 0.0
n_target_diff = 0
for k in target_keys:
    if not torch.equal(merged[k], base_sd[k]):
        n_target_diff += 1
    delta = (merged[k].float() - base_sd[k].float()).abs().max().item()
    max_abs_delta = max(max_abs_delta, delta)

n_nontarget_diff = 0
for k in common_keys - target_key_set:
    if not torch.equal(merged[k], base_sd[k]):
        n_nontarget_diff += 1

ok = n_target_diff == EXPECTED_TARGET_COUNT and max_abs_delta > 0 and n_nontarget_diff == 0
verdict = "PASS" if ok else "FAIL"
print(
    f"[fig2nl] MILESTONE merge_verified target_diff={n_target_diff}/{EXPECTED_TARGET_COUNT} "
    f"max_abs_delta={max_abs_delta} nontarget_diff={n_nontarget_diff} -> {verdict}"
)
sys.exit(0 if ok else 1)
PY

  # ---- stage 3: noinst sweep, ascending n ---------------------------------
  for n in "${SIZES[@]}"; do
    rid=evt-llama-fig2nl-noinst-n${n}
    train_or_skip "$rid" \
      python3 train_target.py --config ../configs/llama_fig2nl_noinst.yaml \
        --override "../configs/sweeps/llama_fig2nl/llama_fig2nl_noinst_n${n}.yaml" \
        --init-from meta-llama/Llama-3.2-1B --confirm-cost
  done

  # ---- stage 4: inst sweep, ascending n -----------------------------------
  # G7 (identical data order across the matched pair) is enforced by
  # train_target.py itself via match_data_order_with — satisfied here only
  # because the same-n noinst run above always completes first.
  [[ -f $INSTALLER_MODEL/model.safetensors ]] ||
    fail "no merged installer checkpoint at $INSTALLER_MODEL (installer stage + adapter merge must complete before the inst sweep)"
  for n in "${SIZES[@]}"; do
    rid=evt-llama-fig2nl-inst-n${n}
    train_or_skip "$rid" \
      python3 train_target.py --config ../configs/llama_fig2nl_inst.yaml \
        --override "../configs/sweeps/llama_fig2nl/llama_fig2nl_inst_n${n}.yaml" \
        --init-from "$INSTALLER_MODEL" --confirm-cost
  done

  # ---- stage 5: G5 evidence (zero-shot EM + test loss) per completed run -
  G5_RIDS=("$INSTALLER_RID" "${NOINST_RIDS[@]}" "${INST_RIDS[@]}")
  for rid in "${G5_RIDS[@]}"; do
    if gate_recorded "$rid" G5; then
      milestone "gate_skip run=$rid gate=G5"
      continue
    fi
    python3 gates.py g5 --run "$rid" --config ../configs/eval_nl_target_data_llama.yaml ||
      fail "$rid G5 (evidence recording failed)"
    milestone "gate_recorded run=$rid gate=G5"
  done
fi

# ---- stage 6: push (separate; needs a WRITE-scoped token) -----------------
if ((PUSH)); then
  push_manifest_only "$INSTALLER_RID"
  for n in "${SIZES[@]}"; do
    [[ $n == "$BIG_N" ]] && continue
    push_manifest_only "evt-llama-fig2nl-noinst-n${n}"
    push_manifest_only "evt-llama-fig2nl-inst-n${n}"
  done
  # Only the two n=1,000,000 runs get their weights archived to the relay
  # (39 full Llama-3.2-1B checkpoints locally would be ~100GB; every LoRA
  # target run saves the complete base+adapter state_dict, spec 00 §1, not
  # just the adapter). Every other run's model.safetensors is pruned below —
  # a local disk-space step, not a "confirm this copy" step, since those
  # weights are deliberately never archived anywhere. adapter.safetensors
  # (+ config.json/generation_config.json) is kept: the compact recovery
  # artifact already rode along on the --no-weights push above.
  push_weights_verified "$BIG_NOINST_RID"
  push_weights_verified "$BIG_INST_RID"

  for n in "${SIZES[@]}"; do
    [[ $n == "$BIG_N" ]] && continue
    prune_rids=("evt-llama-fig2nl-noinst-n${n}" "evt-llama-fig2nl-inst-n${n}")
    for rid in "${prune_rids[@]}"; do
      model_dir=$GEODE_STORE/runs/$rid/model
      if [[ -f $model_dir/model.safetensors ]]; then
        rm -f "$model_dir/model.safetensors"
        milestone "pruned run=$rid file=$model_dir/model.safetensors (adapter.safetensors kept)"
      fi
    done
  done
fi

echo "[fig2nl] MILESTONE analysis_commands"
echo "[fig2nl]   no analysis script is wired to this family — analysis work is CUT from this plan"
# Silent unless NTFY_AUTO=1 (owner 2026-07-31: automatic pings default off;
# notify()'s NTFY_AUTO gate in launch_common.sh covers this call and fail()'s
# alike). The session-end owner ping is sent manually by the agent, not by
# this script.
notify "fig2nl launcher done: stage=$STAGE push=$PUSH"
echo "[fig2nl] TERMINAL_SUCCESS stage=$STAGE push=$PUSH"
