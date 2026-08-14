#!/usr/bin/env bash
# Figure-2 dataset-size sweep (Llama-3.2-1B pair), BARE-FORMAT family
# (fig2nl3) — the fig2nl2 protocol with exactly one change: the
# Question:/Answer: scaffold is REMOVED from every prompt (format bare_nl,
# spec 02 §4). EXPERIMENTS §6.13.
#
# WHY: fig2nl2 completed cleanly (38/38, certified-undamaged parent) and its
# arms COINCIDE — the scaffold alone pre-installs the output convention in
# both arms (base Llama ~0.31 zero-shot EM, ~0.83 format validity untrained),
# so the paper's Figure-2 pre-elicit transient never existed in any
# scaffolded family. fig2nl3 restores the paper's regime: bare NL questions,
# answers as plain continuations, base ~0% zero-shot. PREDICTION: the paper's
# gap (pre-elicit well below base at small n) appears. If it does not — with
# the premise verified and the parent gated undamaged — that is a genuine
# discrepancy with the paper, reportable either way.
#
# PREMISE GUARD (new in this family, runs BEFORE any training):
# check_bare_baseline.py measures base Llama zero-shot EM on the bare eval
# slice and HALTS unless EM <= 0.05 — the family is pointless if base still
# answers bare prompts. It also proves the bare format's span alignment on
# the real tokenizer.
#
# Installer: the fig2nl2-WINNING dose16 recipe re-rendered bare — 16
# correct-label bare-NL mult examples, batch 16, lr 7e-5, r512/a32
# (llama_fig2nl3_installer.yaml). Gates BOTH enforced: G4 >= 0.90 on BARE
# eval prompts (base scores ~0 there, so this is the full 0->0.90+ install)
# and G2 >= 0.31 retention on the SCAFFOLDED eval_algo set (base 0.3271).
# RETRY LADDER (manual): delete $GEODE_STORE/runs/evt-llama-fig2nl3-installer
# and rerun with --override on, in order:
#     ../configs/sweeps/llama_fig2nl3/installer_lr_1e-4.yaml
#     ../configs/sweeps/llama_fig2nl3/installer_lr_3p53e-4.yaml
# First rung clearing absorption + G4 + G2 wins; all failing halts the
# family for owner triage — bars do not move (the fig2nl lesson).
#
# DATA: regenerated in place when missing. Base datagen is deterministic
# (seed 20260717, verified bit-faithful 2026-08-11/12 on two machines); the
# bare sets are hash-pinned derivations of the frozen artifacts
# (datagen/make_bare_sets.py — same triples, same order, scaffold dropped).
#
# DISK/COST: same envelope as fig2nl2 (~8-10 h train + ~1-2 h gates on an
# A100/4090 class GPU; ~25 GB peak with --prune). Small-n runs may train
# LONGER than fig2nl2's (the format transient is extra information to
# absorb); ceilings are the same doubled schedule and remain pure cost caps.
#
# Token: $HF_TOKEN with the Meta Llama-3.2 license accepted. No write scope
# needed — this launcher never pushes. Metadata can be pushed by hand:
#   HF_TOKEN=<write token> python3 hf_checkpoint.py push --run-id <rid> --metadata-only
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=fig2nl3

echo "[fig2nl3] estimated cost: ~8-10 h training + ~1-2 h gates on an A100/4090"
echo "[fig2nl3] class GPU (fig2nl2 measured ~9 h wall on an A100, identical"
echo "[fig2nl3] schedule; bare small-n runs may run longer). \$0 on owned"
echo "[fig2nl3] hardware. Disk: ~126 GB without --prune, ~25 GB with."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2nl3_llama.sh: --confirm-cost required (budget rule)" >&2
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

INSTALLER_RID=evt-llama-fig2nl3-installer
INSTALLER_MODEL=$GEODE_STORE/runs/$INSTALLER_RID/model_merged
SIZES=(1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000)
BIG_N=1000000

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE prune=$PRUNE sizes=${#SIZES[@]}"

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

gate_score() { sed -n "s/.*$2 $3 \([0-9.]*\) on n=.*/\1/p" <<<"$1" | head -1; }
gate_passed() { grep -q "$2 $3 .* -> PASS" <<<"$1"; }

# Pass-AWARE resume check (fig2nl2 launcher's gate_verdict, verbatim).
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

# Score-then-record one ENFORCED installer gate (--no-record first, then
# --record-only-pass; fig2nl2 launcher's helper, verbatim).
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

g5_and_prune() {
  local rid=$1 n=$2
  if gate_recorded "$rid" G5; then
    milestone "gate_skip run=$rid gate=G5"
  else
    python3 gates.py g5 --run "$rid" --config ../configs/eval_bare_target_data_llama.yaml ||
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
BASE_NEEDED=(D_algo.parquet D_algo_eval.parquet D_dose_mult.parquet report.json)
MISSING=0
for f in "${BASE_NEEDED[@]}"; do
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
  milestone "datagen_complete"
else
  milestone "datagen_skip (frozen base artifacts present)"
fi

BARE_NEEDED=(D_algo_bare.parquet D_algo_eval_bare.parquet D_dose_mult_bare.parquet)
BARE_MISSING=0
for f in "${BARE_NEEDED[@]}"; do
  [[ -f $DATA_DIR/$f ]] || BARE_MISSING=1
done
if ((BARE_MISSING)); then
  milestone "bare_datagen_start (deriving scaffold-free sets from the frozen artifacts)"
  python3 ../datagen/make_bare_sets.py --out "$DATA_DIR" || fail "make_bare_sets (bare derivations)"
  milestone "bare_datagen_complete"
else
  milestone "bare_datagen_skip (all bare artifacts present)"
fi

# Tokenizer/span guard on the scaffolded artifacts (llama-guide §1); the BARE
# format's span alignment is proven by the premise guard below, on the same
# real tokenizer.
python3 verify_llama_tokenizer.py || fail "tokenizer verification"

# ---- stage 1b: THE PREMISE GUARD (before any training) --------------------
# Base Llama must be ~0% zero-shot on bare prompts, or the family is
# pointless (the transient it measures does not exist). Resume-safe: pure
# read, cheap (~1-2 min), re-runs every launch by design.
python3 check_bare_baseline.py --n 256 --max-em 0.05 ||
  fail "PREMISE: base Llama answers bare prompts — the scaffold-free transient this family measures does not exist; halt for owner triage (do NOT lower the bar)"
milestone "premise_guard base_bare_zero_shot<=0.05 -> PASS"

# ---- stage 2: shared format installer, then its gates ----------------------
train_or_skip "$INSTALLER_RID" \
  python3 train_sft.py --config ../configs/llama_fig2nl3_installer.yaml \
    --init-from meta-llama/Llama-3.2-1B --confirm-cost

# ABSORPTION GUARD: the gates cannot catch a no-op installer (G2 passes
# trivially for base; G4-on-bare would catch it here, but the guard stays —
# it is the only check that reads the training evidence itself).
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
  fail "installer_absorption min_train_loss_nats=$MIN_TRAIN_LOSS > bar=$ABSORPTION_BAR — dose not absorbed; run the retry ladder (file header)"

# G4 >= 0.90 on BARE prompts AND G2 >= 0.31 on the scaffolded set — both
# enforced and recorded (parent_required_gates: [G4, G2]).
enforce_installer_gate G4 format_validity 0.90 \
  python3 gates.py g4 --run "$INSTALLER_RID" \
    --config ../configs/llama_fig2nl3_installer.yaml \
    --prompt-config ../configs/eval_bare_target_data_llama.yaml \
    --threshold 0.90
enforce_installer_gate G2 accuracy 0.31 \
  python3 gates.py g2 --run "$INSTALLER_RID" \
    --config ../configs/eval_algo_data_llama.yaml --threshold 0.31

# LoRA installer -> plain checkpoint for the inst sweep's --init-from.
if [[ -f $GEODE_STORE/runs/$INSTALLER_RID/model_merged/model.safetensors ]]; then
  milestone "merge_skip run=$INSTALLER_RID (model_merged exists)"
else
  milestone "merge_start run=$INSTALLER_RID"
  python3 merge_adapter.py --run-id "$INSTALLER_RID" || fail "$INSTALLER_RID adapter merge"
  milestone "merge_complete run=$INSTALLER_RID"
fi

# MERGE VERIFICATION (fig2nl2 launcher's check, verbatim except the rid).
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
    print(f"[fig2nl3] MILESTONE merge_verified -> FAIL key-set mismatch beyond known "
          f"tie_word_embeddings aliasing: {sorted(sym_diff - KNOWN_ALIASING_KEYS)}")
    sys.exit(1)
if sym_diff:
    print(f"[fig2nl3] merge_verify note: tolerating known lm_head/embed_tokens aliasing "
          f"key-set diff {sorted(sym_diff)}")

common_keys = merged_keys & base_keys
target_keys = [k for k in common_keys if k.endswith(TARGET_SUFFIXES)]
if len(target_keys) != EXPECTED_TARGET_COUNT:
    print(f"[fig2nl3] MILESTONE merge_verified -> FAIL target-key count {len(target_keys)} "
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
    f"[fig2nl3] MILESTONE merge_verified target_diff={n_target_diff}/{EXPECTED_TARGET_COUNT} "
    f"max_abs_delta={max_abs_delta} nontarget_diff={n_nontarget_diff} -> {verdict}"
)
sys.exit(0 if ok else 1)
PY

# Installer G5 evidence (n=0 baseline of each curve); never pruned.
if gate_recorded "$INSTALLER_RID" G5; then
  milestone "gate_skip run=$INSTALLER_RID gate=G5"
else
  python3 gates.py g5 --run "$INSTALLER_RID" --config ../configs/eval_bare_target_data_llama.yaml ||
    fail "$INSTALLER_RID G5 (evidence recording failed)"
  milestone "gate_recorded run=$INSTALLER_RID gate=G5"
fi

# ---- stage 3: noinst sweep, ascending n (G5 + optional prune inline) -------
for n in "${SIZES[@]}"; do
  rid=evt-llama-fig2nl3-noinst-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config ../configs/llama_fig2nl3_noinst.yaml \
      --override "../configs/sweeps/llama_fig2nl3/llama_fig2nl3_noinst_n${n}.yaml" \
      --init-from meta-llama/Llama-3.2-1B --confirm-cost
  g5_and_prune "$rid" "$n"
done

# ---- stage 4: inst sweep, ascending n (G5 + optional prune inline) ---------
[[ -f $INSTALLER_MODEL/model.safetensors ]] ||
  fail "no merged installer checkpoint at $INSTALLER_MODEL (installer stage + adapter merge must complete before the inst sweep)"
for n in "${SIZES[@]}"; do
  rid=evt-llama-fig2nl3-inst-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config ../configs/llama_fig2nl3_inst.yaml \
      --override "../configs/sweeps/llama_fig2nl3/llama_fig2nl3_inst_n${n}.yaml" \
      --init-from "$INSTALLER_MODEL" --confirm-cost
  g5_and_prune "$rid" "$n"
done

echo "[fig2nl3] MILESTONE analysis_commands"
echo "[fig2nl3]   the deliverable figure (EDL/D vs n, one curve per arm):"
echo "[fig2nl3]     python3 ../analysis/dataset_size_sweep.py --family nl3"
echo "[fig2nl3]   CPU-only, runs here off \$GEODE_STORE. Writes"
echo "[fig2nl3]   results/dataset_size_sweep_nl3.parquet + analysis/figures/dataset_size_sweep_nl3.png"
notify "fig2nl3 launcher done: prune=$PRUNE"
echo "[fig2nl3] TERMINAL_SUCCESS prune=$PRUNE"
