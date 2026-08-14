#!/usr/bin/env bash
# ts38 mini — elicit-vs-teach EDL marker confirmation on the 38.7M TinyStories
# base (EXPERIMENTS §6.14). This is Donoway et al. §5 / Fig-3's CAUSAL
# intervention (base = teaching vs pre-taught = elicitation), NOT the Fig-2
# TinyStories pair: both of the paper's Fig-2 TS arms are teaching, so that
# pair verifies nothing about elicitation.
#
# THE ELEVEN RUNS
#   evt-ts38-pretaught-parent        base -> full FT on D_target (1M op-notation
#                                    add/sub, correct labels, scaffolded), lr 3e-4.
#                                    Paper App. E: pre-training interventions are
#                                    full FT, so there is NO adapter and NO merge
#                                    step here (unlike the fig2nl3 installer) —
#                                    train_sft.py's full-FT path saves plain
#                                    weights straight to runs/<rid>/model.
#   evt-ts38-base-n{N}      (Arm A)  base   -> LoRA r128/a32 @1e-3 on D_algo_bare
#   evt-ts38-pretaught-n{N} (Arm B)  parent -> the SAME recipe, same data order
#   for N in 1000 4642 21544 100000 316228. The arms differ ONLY in theta_0.
#
# PRE-REGISTERED DECISION RULE (do not re-derive after seeing numbers):
#   Arm A teaching marker    = a contiguous RISING span of EDL/D across the grid
#   Arm B elicitation marker = monotone DECREASING EDL/D across the grid
# both under the OCV and test floors. A peak at n=316228 reads "unresolved",
# never "no teaching". A marker failing with the premise verified is a genuine
# discrepancy — escalate, never tune.
#
# ORDER IS LOAD-BEARING. Arm A runs to completion before Arm B starts: each Arm
# B overlay carries match_data_order_with pointing at the same-size Arm A run,
# and train_target.py enforces that match itself (G7). Arm B cannot be launched
# first, and interleaving would break the anchor lookup.
#
# GATES ON THE PARENT — BOTH ENFORCED, both scored --no-record before anything
# is written (a recorded FAIL on a shared parent makes require_parent_ready
# refuse every child of it, V0.6, and un-recording is manual manifest surgery):
#   G1 >= 0.95 exact match — capability. Invoked the way run 2's and phase 3's
#        G1 were: --config is the parent's OWN training yaml, so the gate scores
#        1,024 questions from D_target's held-out val split. It is NOT scored on
#        D_target_eval: run_exact_match_gate re-derives a val split via
#        split_indices and needs data.val_fraction + data.seed, which the
#        eval-only pins deliberately do not carry. D_target_eval is measured
#        instead as G5 evidence on the same checkpoint, two lines below.
#   G8 <= 1.1718 nats — TinyStories RETENTION on run 1's own frozen validation
#        stream. 1.1718 = the base's min_val 1.0718 plus a delta pre-declared at
#        0.10 BEFORE any parent trained. This is the gate fig2nl lacked: its
#        installed arm entered at retention 0.1719 against a base of 0.3271, so
#        its arms were never separated and no elicitation reading was possible.
# FALLBACK on either gate: HALT. Re-run the parent at lr 1e-4 via the
# pre-registered rung configs/sweeps/ts38/parent_lr_1e-4.yaml — the OWNER
# decides that, this script never auto-relaunches and never moves a bar.
#
# THE n=1000 CHECKPOINT (stage 3) is deliberate, not decoration. The 1e-3 target
# LR is pinned from the runs-5/6 sweep on OP-FORMAT targets; bare NL is that
# pin's one scope shift ([[feedback-scope-check-pins-before-reuse]]). The first
# Arm A run costs minutes, so its stop_reason is checked before the other nine
# are launched behind it.
#
# NO PRUNING, deliberately (unlike fig2nl3's --prune). Snapshots are off in the
# mini and the mechanistic phase re-runs these points later with snapshots on;
# eleven 38.7M checkpoints are ~1.7 GB total, far below the threshold that made
# pruning worth its risk on the 1B family.
#
# DATA: regenerated in place when missing, then every artifact this family reads
# is order-hash verified BEFORE any GPU spend (stage 1). Base datagen is
# deterministic (seed 20260717); the bare sets are hash-pinned derivations
# (datagen/make_bare_sets.py — same triples, same order, scaffold dropped).
#
# Token: $HF_TOKEN needs only READ scope — this launcher pulls the base run off
# the relay and never pushes. Metadata can be pushed by hand afterwards:
#   HF_TOKEN=<write token> python3 hf_checkpoint.py push --run-id <rid> --metadata-only
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38

echo "[ts38] estimated cost: ~11 runs, roughly a DAY of wall clock on one A100"
echo "[ts38] (parent full-FT over 1M examples dominates; the ten LoRA target"
echo "[ts38] runs are minutes-to-hours each, ascending). Single-digit dollars on"
echo "[ts38] rented hardware, \$0 on owned. Disk: ~2 GB of checkpoints (no"
echo "[ts38] pruning, no snapshots) + ~2 GB for the TinyStories corpus G8 packs."
echo "[ts38] One-off CPU cost: G8's first call downloads + packs 2.7M documents"
echo "[ts38] (30-60 min, ~4.3 GB resident). Stage 1 pays it once, before any GPU."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38_mini.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
PARENT_RID=evt-ts38-pretaught-parent
PARENT_MODEL=$GEODE_STORE/runs/$PARENT_RID/model
SIZES=(1000 4642 21544 100000 316228)
PIN_CHECK_N=1000

PARENT_CONFIG=../configs/ts38_pretaught_parent.yaml
ARM_A_CONFIG=../configs/ts38_base.yaml
ARM_B_CONFIG=../configs/ts38_pretaught.yaml
OVERLAY_DIR=../configs/sweeps/ts38
# Op-notation eval pin (D_target_eval) — the parent's G5 evidence.
OP_EVAL_CONFIG=../configs/eval_target_data.yaml
# Bare-NL eval pin (D_algo_eval_bare) for the CUSTOM 10K tokenizer: the ts38
# counterpart of eval_bare_target_data_llama.yaml. Premise guard + every target
# run's G5 read it.
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
# Run 1's own pretrain pin — G8 rebuilds its validation stream from this.
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
# Outside runs/<id>/ on purpose: hf_checkpoint.py push uploads runs/<id>, so a
# cache parked in there would ride a relay push.
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
STEP0_JSON=$GEODE_STORE/results/ts38_step0_baseline.json

# THE TWO PARENT BARS, written as literals at every call site (the fig2nl2/3
# convention — a pre-registered threshold should be readable in the log line
# that enforces it, not one indirection away):
#   G1  >= 0.95   exact match. Run 2's "capability present" bar, on the same
#                 scale, method, stage and task family (it scored 0.9961).
#   G8  <= 1.1718 nats. The base's own min_val 1.0718 + a delta pre-declared
#                 at 0.10 before any parent trained.
# Neither moves after a number is seen. If one is missed, take the fallback
# rung below or escalate — do not re-derive a bar post hoc.

FALLBACK_MSG="PRE-REGISTERED FALLBACK: delete \$GEODE_STORE/runs/$PARENT_RID and re-run the
   parent at the lower rung:
     python3 train_sft.py --config $PARENT_CONFIG \\
       --override $OVERLAY_DIR/parent_lr_1e-4.yaml \\
       --init-from $BASE_MODEL --confirm-cost
   Nothing was recorded, so this checkpoint is still inspectable. The OWNER
   decides whether to take that rung — do NOT move a bar after seeing a number."

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE sizes=${#SIZES[@]} base=$BASE_RID parent=$PARENT_RID"

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

# Pass-AWARE resume check (fig2nl2/fig2nl3 launchers' gate_verdict, verbatim).
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

# Score-then-record ONE enforced gate on a shared parent: --no-record first,
# then --record-only-pass. fig2nl2's enforce_installer_gate, generalized over
# the run id so both parent gates share it. Idempotent: a recorded pass is a
# skip, anything else recorded is a refusal to proceed.
enforce_gate() {
  local rid=$1 gate=$2 metric=$3 threshold=$4 verdict out score
  shift 4
  verdict=$(gate_verdict "$rid" "$gate")
  if [[ $verdict == pass ]]; then
    milestone "gate_skip run=$rid gate=$gate status=recorded_pass"
    return 0
  elif [[ $verdict != missing ]]; then
    fail "$rid gate=$gate verdict='$verdict' (expected pass or missing) — inspect $GEODE_STORE/runs/$rid/manifest.json before rerunning"
  fi
  out=$("$@" --no-record 2>&1)
  echo "$out"
  score=$(gate_score "$out" "$gate" "$metric")
  [[ -n $score ]] || fail "$rid $gate printed no $metric score — it errored rather than scored (output above)"
  gate_passed "$out" "$gate" "$metric" || fail \
    "$rid $gate $metric=$score misses the bar $threshold — NOT recorded, so no target
   run can start from an ungated parent.
   $FALLBACK_MSG"
  "$@" --record-only-pass ||
    fail "$rid $gate DIVERGENCE: --no-record scored $score (PASS) but the recording pass recomputed a FAIL — nothing was written; rerun the gate block"
  milestone "gate_pass run=$rid gate=$gate $metric=$score"
}

# G5 evidence on the bare target eval (zero/16-shot EM + shared-set test loss on
# identical data for every run). No pass bar by design, and never pruned.
record_g5() {
  local rid=$1
  if gate_recorded "$rid" G5; then
    milestone "gate_skip run=$rid gate=G5"
    return 0
  fi
  python3 gates.py g5 --run "$rid" --config "$BARE_EVAL_CONFIG" ||
    fail "$rid G5 (evidence recording failed)"
  milestone "gate_recorded run=$rid gate=G5"
}

# ---- stage 0: the base checkpoint, receiver-verified -----------------------
# [[feedback-verify-the-receiver-not-the-sender]]: push bookkeeping looks
# identical over an empty relay, so this lists what the hub actually holds,
# pulls it, checks the files landed, and then LOADS the model and asserts its
# architecture. After the 2026-08-14 store cleanup this is the only run on the
# relay that still carries weights, so a silent miss here is the whole family.
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38 family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38] MISSING on the hub: {missing}")
    sys.exit(1)
PY

if [[ ! -f $BASE_MODEL/model.safetensors ]]; then
  milestone "pull_base run=$BASE_RID"
  python3 hf_checkpoint.py pull --run-id "$BASE_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $BASE_RID from $RELAY_REPO"
fi
[[ -f $BASE_MODEL/model.safetensors ]] ||
  fail "pull left no $BASE_MODEL/model.safetensors (snapshot_download is fail-open)"
[[ -f $GEODE_STORE/runs/$BASE_RID/manifest.json ]] ||
  fail "pull left no manifest for $BASE_RID — geode.zoo cannot load a run without it"

python3 - "$BASE_RID" <<'PY' || fail "$BASE_RID did not load as the expected 38.7M TinyStories base — do not train against it"
import sys
from pathlib import Path

from transformers import AutoTokenizer

from geode.zoo import load_model, load_run, tokenizer_hash

rid = sys.argv[1]
manifest = load_run(rid).data
model = load_model(rid, device="cpu")
# The launcher cd's to scripts/, so this is experiments/training-run/tokenizer —
# the frozen artifact, NOT the pulled checkpoint dir (load_model is model-only).
tok_dir = Path.cwd().parent / "tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tok_dir)

want = {"hidden_size": 512, "num_hidden_layers": 8, "vocab_size": 10000}
got = {
    "hidden_size": model.config.hidden_size,
    "num_hidden_layers": model.config.num_hidden_layers,
    "vocab_size": len(tokenizer),
}
tok_sha = tokenizer_hash(tokenizer)
want_sha = manifest["experiment"]["tokenizer"]["sha256"]
print(f"[ts38] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 1: data artifacts, then their order hashes ----------------------
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
BASE_NEEDED=(D_target.parquet D_target_eval.parquet D_algo.parquet D_algo_eval.parquet report.json)
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
  milestone "datagen_complete"
else
  milestone "datagen_skip (frozen base artifacts present)"
fi

BARE_NEEDED=(D_algo_bare.parquet D_algo_eval_bare.parquet)
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

# Every file this family reads, hash-verified through its own config pin, and
# the two config contracts that only bite AFTER training if they are wrong:
# gates.py g1 re-derives a val split from the parent config's data.val_fraction
# and data.seed, and every target G5 needs the bare eval pin to exist.
python3 - "$PARENT_CONFIG" "$OP_EVAL_CONFIG" "$ARM_A_CONFIG" "$BARE_EVAL_CONFIG" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

parent_cfg_path, op_eval, arm_a, bare_eval = (Path(a) for a in sys.argv[1:5])
for path in (parent_cfg_path, op_eval, arm_a, bare_eval):
    if not path.is_file():
        print(f"[ts38] MISSING config {path}")
        sys.exit(1)

parent_cfg = load_config(parent_cfg_path, None)
missing = [k for k in ("val_fraction", "seed") if parent_cfg["data"].get(k) is None]
if missing:
    print(
        f"[ts38] {parent_cfg_path} data block lacks {missing} — gates.py g1 re-derives the "
        "held-out split with split_indices(n, val_fraction, seed) and would KeyError AFTER "
        "the parent has trained. Pin them before launching."
    )
    sys.exit(1)

for label, path in (
    ("D_target", parent_cfg_path),
    ("D_target_eval", op_eval),
    ("D_algo_bare", arm_a),
    ("D_algo_eval_bare", bare_eval),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")
PY
milestone "data_verified D_target/D_target_eval/D_algo_bare/D_algo_eval_bare order hashes OK"

# G8's stream, built and PROVEN before any GPU spend. This scores the base
# against itself: the anchor pass and the scored pass are the same checkpoint on
# the same rebuilt stream, so their agreement is a free end-to-end self-test of
# the reconstruction, and the ~45-minute pack lands in --val-cache for the two
# real G8 calls in stage 2. --no-record keeps the base manifest untouched.
python3 gates.py g8 --run "$BASE_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
  --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --no-record ||
  fail "G8 could not rebuild run 1's validation stream (see the gate's own diagnosis
   above: document count, packed-row count, split sizes and the anchor loss each
   fail with a distinct message). Retention is UNMEASURABLE until this passes —
   do NOT substitute a different validation set, that silently redefines the
   measurement. Halt for owner triage."
milestone "g8_preflight val_stream rebuilt + anchored on $BASE_RID -> PASS"

# ---- stage 1b: THE PREMISE GUARD (before any training) --------------------
# The ts38 premise is two-sided: the base must be ~0% EM AND ~0% format-valid on
# bare prompts. A base that already emits well-formed answers carries the output
# convention, which is exactly why the scaffolded fig2nl2 arms coincided. Same
# call writes the step-0 label-loss control (plan guard 5): a FIXED per-example
# digit cost decays like 1/n and can fake a decreasing EDL/D limb all by itself,
# so the constant has to be on record before the curves are read.
python3 check_bare_baseline.py \
  --eval-config "$BARE_EVAL_CONFIG" \
  --base-model "$BASE_MODEL" \
  --dtype fp32 \
  --n 256 --max-em 0.05 --max-format-validity 0.05 \
  --step0-json "$STEP0_JSON" ||
  fail "PREMISE: the 38.7M base already answers (or already formats) bare prompts —
   the elicit-vs-teach contrast this family measures does not exist on this base.
   Halt for owner triage; do NOT lower either bar."
[[ -f $STEP0_JSON ]] ||
  fail "premise guard passed but wrote no $STEP0_JSON — the fixed-cost control (guard 5)
   is missing and the EDL/D limbs cannot be read against it"
milestone "premise_guard base_bare EM<=0.05 AND format_validity<=0.05 -> PASS"
milestone "step0_control written=$STEP0_JSON"

# ---- stage 2: the pre-taught parent, then BOTH of its gates ----------------
train_or_skip "$PARENT_RID" \
  python3 train_sft.py --config "$PARENT_CONFIG" \
    --init-from "$BASE_MODEL" --confirm-cost

# Full FT: train_sft.py's non-LoRA path writes plain weights to runs/<rid>/model
# and no adapter sidecar, so there is no merge step and Arm B inits from here.
[[ -f $PARENT_MODEL/model.safetensors ]] ||
  fail "$PARENT_RID completed but left no $PARENT_MODEL/model.safetensors — full-FT save path?"

enforce_gate "$PARENT_RID" G1 accuracy 0.95 \
  python3 gates.py g1 --run "$PARENT_RID" --config "$PARENT_CONFIG" --threshold 0.95
enforce_gate "$PARENT_RID" G8 val_loss_nats 1.1718 \
  python3 gates.py g8 --run "$PARENT_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
    --tokenizer ../tokenizer --val-cache "$VAL_CACHE"

# D_target_eval evidence on the parent: zero/16-shot op-notation EM plus the
# shared-set test loss. This is the question-disjoint measurement of what the
# pre-teaching installed; G1 above is the enforced bar, on D_target's own
# held-out split. Evidence only, no bar (spec 02 §8).
if gate_recorded "$PARENT_RID" G5; then
  milestone "gate_skip run=$PARENT_RID gate=G5"
else
  python3 gates.py g5 --run "$PARENT_RID" --config "$OP_EVAL_CONFIG" ||
    fail "$PARENT_RID G5 (evidence recording failed)"
  milestone "gate_recorded run=$PARENT_RID gate=G5"
fi

# ---- stage 3: Arm A (teach), ascending n -----------------------------------
# Must finish entirely before stage 4: Arm B's overlays match this arm's data
# order per size, and train_target.py resolves that anchor from the store.
for n in "${SIZES[@]}"; do
  rid=evt-ts38-base-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$ARM_A_CONFIG" \
      --override "$OVERLAY_DIR/ts38_base_n${n}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost

  if [[ $n == "$PIN_CHECK_N" ]]; then
    # The 1e-3 LR pin's one scope shift, checked on the cheapest run in the
    # family before the other nine queue up behind it. stop_reason=max_steps is
    # a bug signal by standing policy, never a result.
    PIN_OUT=$(python3 - "$rid" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file():
    print("MISSING - - -")
    sys.exit()
r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) or {}
print(
    r.get("stop_reason", "MISSING"),
    r.get("min_val_nats"),
    r.get("best_val_nats"),
    r.get("final_step"),
)
PY
)
    read -r PIN_STOP PIN_MIN PIN_BEST PIN_STEP <<<"$PIN_OUT"
    milestone "lr_pin_check run=$rid stop_reason=$PIN_STOP min_val_nats=$PIN_MIN best_val_nats=$PIN_BEST final_step=$PIN_STEP"
    [[ $PIN_STOP == converged ]] || fail \
      "LR PIN CHECK: $rid ended with stop_reason='$PIN_STOP' (min val $PIN_MIN, best $PIN_BEST,
   step $PIN_STEP), not 'converged'. The 1e-3 target LR was pinned on OP-FORMAT
   targets (runs 5/6) and this is its first bare-NL use; the eps/k rule was
   calibrated at that same LR. Nine more runs would inherit whatever this is.
   OWNER DECISION NEEDED — a max_steps stop is a bug signal, not a result:
   re-pin the LR on bare NL, or raise the ceiling if the curve was still
   descending. Do not proceed by widening a bar."
  fi
  record_g5 "$rid"
done
milestone "arm_a_complete sizes=${#SIZES[@]} (G7 anchors now registered)"

# ---- stage 4: Arm B (elicit), ascending n ----------------------------------
[[ -f $PARENT_MODEL/model.safetensors ]] ||
  fail "no parent checkpoint at $PARENT_MODEL — stage 2 must complete before Arm B"
for n in "${SIZES[@]}"; do
  rid=evt-ts38-pretaught-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$ARM_B_CONFIG" \
      --override "$OVERLAY_DIR/ts38_pretaught_n${n}.yaml" \
      --init-from "$PARENT_MODEL" --confirm-cost
  record_g5 "$rid"
done

echo "[ts38] MILESTONE analysis_commands"
echo "[ts38]   Both are CPU-only and run here off \$GEODE_STORE. --family ts38 is"
echo "[ts38]   NOT a default anywhere — pass it explicitly:"
echo "[ts38]     python3 ../analysis/edl_converged_val_floor.py --family ts38  # OCV (primary)"
echo "[ts38]     python3 ../analysis/dataset_size_sweep.py --family ts38       # test + min-val"
echo "[ts38]   Three floors, OCV primary, floor NAMED on every figure. EDL >= 0 is"
echo "[ts38]   required per run, and a marker verdict counts only if the SHAPE"
echo "[ts38]   survives all three. Read both limbs against the fixed-cost control"
echo "[ts38]   in $STEP0_JSON — a constant per-example digit cost decays like 1/n"
echo "[ts38]   and can produce a decreasing EDL/D limb on its own."
notify "ts38 mini launcher done: 11 runs, both parent gates recorded"
echo "[ts38] TERMINAL_SUCCESS runs=$((1 + 2 * ${#SIZES[@]}))"
