#!/usr/bin/env bash
# Figure-2 dataset-size sweep (Llama-3.2-1B pair), NL-target family v2
# (fig2nl2) — the fig2nl protocol with the installer redesigned around the two
# measured causes of that family's inverted deliverable figure (EXPERIMENTS
# §6.11 outcome; §6.12 is this family's plan):
#   1. the op-notation dose's output convention did not transfer to NL prompt
#      framing (G4-NL 0.8828 vs 0.9922 on op prompts at 7e-5), so the paper's
#      pre-elicit mechanism never fired;
#   2. the 3.53e-4 installer halved the parent's NL retention (G2 0.1719 vs
#      base 0.3271), handicapping the inst arm.
# Changes, ALL installer-side (target-arm protocol byte-identical to fig2nl —
# same sizes, schedule, r512/a32 @ 3.53e-4, batch 128, seed 316, eps/k):
#   * dose = D_dose_mult_nl row 0 (NL-format mult, "What is the product of
#     3354 and 3459?" — same operands as fig2nl's dose, NL framing), derived
#     from the frozen D_dose_mult by datagen/make_nl_dose.py;
#   * installer LR 7.0e-5 (the fig2nl ladder's retention-preserving rung);
#   * G4-NL >= 0.90 AND G2 >= 0.31 BOTH ENFORCED and recorded; the inst arm's
#     parent_required_gates lists both, so an installer that fails either
#     gates NOTHING further.
# INSTALLER RETRY LADDER (manual, run by the box operator): if a gate fails
# at 7.0e-5, delete $GEODE_STORE/runs/evt-llama-fig2nl2-installer and rerun
# the installer with --override pointing at these rung files under
# ../configs/sweeps/llama_fig2nl2/ in order:
#     installer_lr_1e-4.yaml
#     installer_lr_3p53e-4.yaml
# The first rung clearing absorption + G4 + G2 wins. All rungs failing halts
# the family for owner triage — do NOT drop a bar to proceed (dropping G2 is
# exactly how fig2nl produced its confounded figure).
#
# DATA: fully regenerated in-place when missing — datagen is deterministic
# (seed 20260717) and was verified 2026-08-11 to reproduce every frozen
# order_hash pin bit-for-bit (D_algo 48d4feff..., D_algo_eval 5e422daf...,
# D_dose_mult 8ddda6d6...), so this launcher needs no scp'd files; every pin
# is still hash-verified at launch by the trainers.
#
# DISK: without --prune, all 39 runs' full checkpoints stay local — ~126 GB
# (2.5 GB model.safetensors + 0.72 GB adapter sidecar per run). With --prune,
# each sweep run's model.safetensors is deleted right after its G5 records
# (adapter sidecars are KEPT — base + sidecar re-derives the weights, so
# pruned runs stay recoverable); both n=1,000,000 runs are spared. Peak disk
# with --prune ~= 25 GB.
#
# COST: ~8-9 h of training wall-clock on a 4090-class GPU (measured: fig2nl's
# 33 runs took 8.3 h under the identical schedule) + ~1-2 h of G5 gates.
# On your own hardware the dollar cost is $0; --confirm-cost is still
# required (budget rule: nothing launches a GPU job implicitly).
#
# Token: huggingface_hub reads $HF_TOKEN directly; it must belong to an
# account that accepted the Meta Llama-3.2 license (gated base + tokenizer).
# No write scope needed — this launcher never pushes to the relay. To archive
# a run's metadata by hand afterwards:
#   HF_TOKEN=<write token> python3 hf_checkpoint.py push --run-id <rid> --metadata-only
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=fig2nl2

echo "[fig2nl2] estimated cost: ~8-9 h training + ~1-2 h gates on a 4090-class"
echo "[fig2nl2] GPU (measured fig2nl baseline: 8.3 h, identical schedule);"
echo "[fig2nl2] \$0 on owned hardware. Disk: ~126 GB without --prune, ~25 GB with."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2nl2_llama.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
PRUNE=0
for arg in "$@"; do
  [[ $arg == --prune ]] && PRUNE=1
done

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

INSTALLER_RID=evt-llama-fig2nl2-installer
# LoRA installer: the wrapped model/ can only be loaded via geode.zoo.load_model
# — the inst sweep's --init-from points at the MERGED checkpoint below, never
# the wrapped dir (2026-07-22 incident).
INSTALLER_MODEL=$GEODE_STORE/runs/$INSTALLER_RID/model_merged
SIZES=(1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000)
BIG_N=1000000

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE prune=$PRUNE sizes=${#SIZES[@]}"

# Train (or resume-skip) one run; forwarded argv is the exact trainer call.
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

# Read a gate's printed score line; $1 = captured output, $2 = gate name,
# $3 = the metric word gates.py prints for that gate.
gate_score() { sed -n "s/.*$2 $3 \([0-9.]*\) on n=.*/\1/p" <<<"$1" | head -1; }
# Whether that line says PASS — never trust the exit code alone (a missing
# flag also exits nonzero; test_launcher_gate_args.py's history).
gate_passed() { grep -q "$2 $3 .* -> PASS" <<<"$1"; }

# Pass-AWARE resume check (fig2nl launcher's gate_verdict, verbatim): prints
# missing | pass | fail | corrupt; consumers whitelist ONLY "pass" as a skip.
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

# Score-then-record one ENFORCED installer gate (G4 and G2 share the exact
# same protocol here; fig2nl only ran it for G4): --no-record first, parsed
# from the printed line; only on a pass, re-run with --record-only-pass so a
# near-threshold recompute divergence can never write a FAIL to the shared
# parent's manifest. $1=gate (G4|G2), $2=metric word, $3=threshold; the rest
# is the scoring argv WITHOUT the record-mode flag.
enforce_installer_gate() {
  local gate=$1 metric=$2 threshold=$3 verdict out score
  shift 3
  verdict=$(gate_verdict "$INSTALLER_RID" "$gate")
  if [[ $verdict == pass ]]; then
    milestone "gate_skip run=$INSTALLER_RID gate=$gate status=recorded_pass"
    return 0
  elif [[ $verdict != missing ]]; then
    fail "$INSTALLER_RID gate=$gate verdict='$verdict' (expected pass or missing) — inspect $GEODE_STORE/runs/$INSTALLER_RID/manifest.json before rerunning"
  fi
  out=$("$@" --no-record 2>&1)
  echo "$out"
  score=$(gate_score "$out" "$gate" "$metric")
  [[ -n $score ]] || fail "$INSTALLER_RID $gate printed no $metric score (output above)"
  gate_passed "$out" "$gate" "$metric" ||
    fail "$INSTALLER_RID $gate $metric $score < $threshold — NOT recorded, no targets from an ungated parent; run the retry ladder (file header)"
  "$@" --record-only-pass ||
    fail "$INSTALLER_RID $gate DIVERGENCE: --no-record scored $score (PASS) but the recording pass recomputed a FAIL — nothing was written; rerun the gate block"
  milestone "gate_pass run=$INSTALLER_RID gate=$gate $metric=$score"
}

# G5 evidence + optional local weight prune for one completed sweep run. G5
# needs the run's weights, so it always precedes the prune; adapter sidecars
# are kept (base + sidecar re-derives the weights); n=BIG_N is spared.
g5_and_prune() {
  local rid=$1 n=$2
  if gate_recorded "$rid" G5; then
    milestone "gate_skip run=$rid gate=G5"
  else
    python3 gates.py g5 --run "$rid" --config ../configs/eval_nl_target_data_llama.yaml ||
      fail "$rid G5 (evidence recording failed)"
    milestone "gate_recorded run=$rid gate=G5"
  fi
  if ((PRUNE)) && [[ $n != "$BIG_N" ]]; then
    local model_file=$GEODE_STORE/runs/$rid/model/model.safetensors
    if [[ -f $model_file ]]; then
      rm -f "$model_file"
      milestone "pruned run=$rid file=$model_file (adapter sidecar kept)"
    fi
  fi
}

# ---- stage 1: data (regenerate whatever is missing; deterministic) ---------
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
NEEDED=(D_algo.parquet D_algo_eval.parquet D_dose_mult.parquet D_dose_mult_nl.parquet report.json)
MISSING=0
for f in "${NEEDED[@]}"; do
  [[ -f $DATA_DIR/$f ]] || MISSING=1
done
if ((MISSING)); then
  milestone "datagen_start (regenerating the frozen artifacts from seed 20260717)"
  mkdir -p "$DATA_DIR"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 ||
    fail "datagen base (D_algo/D_inst/D_target/probe)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --eval-set ||
    fail "datagen --eval-set (D_target_eval)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --nl-eval-set ||
    fail "datagen --nl-eval-set (D_algo_eval)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --installer-set ||
    fail "datagen --installer-set (D_inst_perm/D_dose_mult)"
  python3 ../datagen/make_nl_dose.py --out "$DATA_DIR" || fail "make_nl_dose (D_dose_mult_nl)"
  milestone "datagen_complete"
else
  milestone "datagen_skip (all frozen artifacts present)"
fi

# Tokenizer/span guard before any GPU spend (llama-guide §1). It samples
# D_inst/D_target/D_algo/D_algo_eval but NOT the dose files, so the NEEDED
# presence check above stays a separate guard.
python3 verify_llama_tokenizer.py || fail "tokenizer verification"

# ---- stage 2: shared format installer, then its gates ----------------------
train_or_skip "$INSTALLER_RID" \
  python3 train_sft.py --config ../configs/llama_fig2nl2_installer.yaml \
    --init-from meta-llama/Llama-3.2-1B --confirm-cost

# ABSORPTION GUARD: read back the installer's own train_log.jsonl and fail
# loud if it never absorbed the dose. Base Llama passes G4/G2 trivially on
# its own, so the gates below cannot catch a no-op installer — this is the
# only check that can. At 7e-5 the fig2nl ladder's rung absorbed in ~134
# steps (min val 0.0151 nats), well under the 0.1 bar.
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
  fail "installer_absorption min_train_loss_nats=$MIN_TRAIN_LOSS > bar=$ABSORPTION_BAR — dose not absorbed; do NOT proceed to gates (they cannot catch a no-op installer). Run the retry ladder (file header)"

# G4 format >= 0.90 on NL eval prompts AND G2 retention >= 0.31 — BOTH
# enforced and recorded (the point of the redo; see the inst config's
# parent_required_gates). A failure here means: run the retry ladder.
enforce_installer_gate G4 format_validity 0.90 \
  python3 gates.py g4 --run "$INSTALLER_RID" \
    --config ../configs/llama_fig2nl2_installer.yaml \
    --prompt-config ../configs/eval_nl_target_data_llama.yaml \
    --threshold 0.90
enforce_installer_gate G2 accuracy 0.31 \
  python3 gates.py g2 --run "$INSTALLER_RID" \
    --config ../configs/eval_algo_data_llama.yaml --threshold 0.31

# LoRA installer -> plain checkpoint for the inst sweep's --init-from.
# Idempotent: skip if the merged checkpoint already exists.
if [[ -f $GEODE_STORE/runs/$INSTALLER_RID/model_merged/model.safetensors ]]; then
  milestone "merge_skip run=$INSTALLER_RID (model_merged exists)"
else
  milestone "merge_start run=$INSTALLER_RID"
  python3 merge_adapter.py --run-id "$INSTALLER_RID" || fail "$INSTALLER_RID adapter merge"
  milestone "merge_complete run=$INSTALLER_RID"
fi

# MERGE VERIFICATION (fig2nl launcher's check, verbatim except the rid): the
# merged checkpoint must differ from base on every LoRA-target tensor and on
# nothing else — a bf16 rounding no-op here would make the whole inst arm
# silently == base, which G4/G2 (scored pre-merge) cannot catch. Re-runs on
# every resume, deliberately not gated behind merge_skip.
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
EXPECTED_TARGET_COUNT = 112  # 7 target modules x 16 layers
KNOWN_ALIASING_KEYS = {"lm_head.weight"}  # tie_word_embeddings aliasing (V0.9)

rid = sys.argv[1]
merged_path = Path(os.environ["GEODE_STORE"]) / "runs" / rid / "model_merged" / "model.safetensors"
merged = load_file(merged_path)

dtype = next(iter(merged.values())).dtype
from transformers import AutoModelForCausalLM  # noqa: PLC0415 (box-only import)

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=dtype)
base_sd = base.state_dict()

merged_keys, base_keys = set(merged), set(base_sd)
sym_diff = merged_keys ^ base_keys
if sym_diff - KNOWN_ALIASING_KEYS:
    print(f"[fig2nl2] MILESTONE merge_verified -> FAIL key-set mismatch beyond known "
          f"tie_word_embeddings aliasing: {sorted(sym_diff - KNOWN_ALIASING_KEYS)}")
    sys.exit(1)
if sym_diff:
    print(f"[fig2nl2] merge_verify note: tolerating known lm_head/embed_tokens aliasing "
          f"key-set diff {sorted(sym_diff)}")

common_keys = merged_keys & base_keys
target_keys = [k for k in common_keys if k.endswith(TARGET_SUFFIXES)]
if len(target_keys) != EXPECTED_TARGET_COUNT:
    print(f"[fig2nl2] MILESTONE merge_verified -> FAIL target-key count {len(target_keys)} "
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
    f"[fig2nl2] MILESTONE merge_verified target_diff={n_target_diff}/{EXPECTED_TARGET_COUNT} "
    f"max_abs_delta={max_abs_delta} nontarget_diff={n_nontarget_diff} -> {verdict}"
)
sys.exit(0 if ok else 1)
PY

# Installer G5 evidence (n=0 baseline of each curve); never pruned (its
# model_merged is every inst run's parent).
if gate_recorded "$INSTALLER_RID" G5; then
  milestone "gate_skip run=$INSTALLER_RID gate=G5"
else
  python3 gates.py g5 --run "$INSTALLER_RID" --config ../configs/eval_nl_target_data_llama.yaml ||
    fail "$INSTALLER_RID G5 (evidence recording failed)"
  milestone "gate_recorded run=$INSTALLER_RID gate=G5"
fi

# ---- stage 3: noinst sweep, ascending n (G5 + optional prune inline) -------
for n in "${SIZES[@]}"; do
  rid=evt-llama-fig2nl2-noinst-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config ../configs/llama_fig2nl2_noinst.yaml \
      --override "../configs/sweeps/llama_fig2nl2/llama_fig2nl2_noinst_n${n}.yaml" \
      --init-from meta-llama/Llama-3.2-1B --confirm-cost
  g5_and_prune "$rid" "$n"
done

# ---- stage 4: inst sweep, ascending n (G5 + optional prune inline) ---------
# G7 (identical data order across the matched pair) is enforced by
# train_target.py via match_data_order_with — satisfied because the same-n
# noinst run above always completes first (arm-serial ascending order).
[[ -f $INSTALLER_MODEL/model.safetensors ]] ||
  fail "no merged installer checkpoint at $INSTALLER_MODEL (installer stage + adapter merge must complete before the inst sweep)"
for n in "${SIZES[@]}"; do
  rid=evt-llama-fig2nl2-inst-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config ../configs/llama_fig2nl2_inst.yaml \
      --override "../configs/sweeps/llama_fig2nl2/llama_fig2nl2_inst_n${n}.yaml" \
      --init-from "$INSTALLER_MODEL" --confirm-cost
  g5_and_prune "$rid" "$n"
done

echo "[fig2nl2] MILESTONE analysis_commands"
echo "[fig2nl2]   the deliverable figure (EDL/D vs n, one curve per arm):"
echo "[fig2nl2]     python3 ../analysis/dataset_size_sweep.py --family nl2"
echo "[fig2nl2]   CPU-only, runs here off \$GEODE_STORE. Writes"
echo "[fig2nl2]   results/dataset_size_sweep_nl2.parquet + analysis/figures/dataset_size_sweep_nl2.png"
echo "[fig2nl2]   (the _nl2 stem cannot overwrite the fig2nl or op outputs)."
notify "fig2nl2 launcher done: prune=$PRUNE"
echo "[fig2nl2] TERMINAL_SUCCESS prune=$PRUNE"
