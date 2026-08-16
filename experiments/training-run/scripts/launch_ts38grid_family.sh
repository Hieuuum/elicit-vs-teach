#!/usr/bin/env bash
# ts38grid — grid EXTENSION for all three ts38 target families (EXPERIMENTS.md
# §6.17). No new parents, no new recipe, no new data: this launcher trains the
# existing three arms (base/teach, pretaught-mw, pretaught-format) at 8 NEW
# dataset sizes — a small-n bracket {128, 256, 512} below the current grid
# floor plus a densification {2154, 10000, 46416, 146780, 215443} around the
# existing 1000/4642/21544/100000/316228 points. 24 new target runs.
#
# THE 24 RUNS (3 arms x 8 sizes; ascending n, base first at each size)
#   evt-ts38-base-n{N}           base/teach arm; theta0 = the 38.7M TinyStories
#                                base evt-run1-base-v3-ext. This run IS the G7
#                                anchor for the other two arms at size N.
#   evt-ts38mw-pretaught-n{N}    theta0 = MERGED evt-ts38mw-parent-probe-lr3e-4
#                                (the GO-B multiwrap install, step 28000).
#   evt-ts38pf-preteachfmt-n{N}  theta0 = MERGED evt-ts38pf-preteachfmt-parent
#                                (format-only LoRA, permuted labels).
#   for N in 128 256 512 2154 10000 46416 146780 215443.
#
# Recipe is IDENTICAL to the three shipped families — LoRA r128/alpha32 @1e-3
# on D_algo_bare, same frozen data order per size, same eps/k stopping rule.
# Arms differ ONLY in theta0; sizes differ ONLY in the prefix length of the
# one frozen stream. Both parents are PULLED here, never rebuilt.
#
# WHY (two open questions, both from decisions.md 2026-08-16):
#   (1) WHERE IS THE RISING LIMB? The base arm's GLOBAL EDL/D argmax sits at or
#       below n=1000 under every floor definition we have (OCV, paper Eq. 3
#       test floor, min-val). A teaching curve that only ever falls on the
#       measured grid is consistent with the rising limb sitting BELOW the
#       grid entirely. The {128, 256, 512} bracket is the cheapest test of
#       that: it resolves whether floor(128) > 3.47 nats, i.e. whether the
#       curve is still climbing to the left of n=1000 or already flat there.
#       (Nulls need bracketing — extend the grid past/below the peak.)
#   (2) WHAT IS THE 4642->21544 HUMP? Every arm does something different across
#       that one span: base +15%, pretaught-format +72%, pretaught-mw crosses
#       over. Three points {10000} inside it plus {2154} and {46416} either
#       side double the resolution, so the hump can be read as a shape rather
#       than as a single segment between two grid corners.
# Neither question gets a new pass/fail bar here. The pre-registered readouts
# live in decisions.md 2026-08-16 "ts38grid pre-registration"; the small-n
# bracket is read FIRST, before the densification points are looked at.
#
# SIZES OVERRIDE: set the SIZES env var (space-separated) to run a subset —
# the owner's own recommended sequencing is the bracket alone, first:
#     SIZES="128 256 512" bash launch_ts38grid_family.sh --confirm-cost
# then the full default grid once the bracket has been read.
#
# HARD RULES
#   (a) BOTH parents are DELIBERATELY UNGATED. This launcher NEVER runs
#       gates.py against either of them — not even with --no-record. Their
#       theta0 evidence already exists (results/ts38mw_probe.json,
#       results/ts38pf_family_theta0.json) and is not re-derived here. Nothing
#       is ever recorded to a parent manifest except merge_adapter.py's own
#       experiment.merged_checkpoint entry, and neither parent is re-pushed
#       (both are pulled unchanged).
#   (b) LoRA checkpoints load ONLY via geode.zoo.load_model
#       ([[feedback-lora-checkpoints-load-via-zoo-load-model]]) — the target
#       arms warm-start from each parent's MERGED plain weights
#       (runs/<id>/model_merged/), never from_pretrained on the wrapped
#       runs/<id>/model/.
#   (c) NEW SIZES ONLY. The five shipped sizes are immutable results reused by
#       the other launchers and by every published figure. Stage 3 refuses to
#       proceed if any run id in this launch already exists on the relay
#       without a complete local copy, because train_or_skip keys on the LOCAL
#       store and would happily retrain a run whose only copy is on the hub.
#   (d) PUSH AS YOU GO. Every run is pushed the moment it is trained + scored,
#       so a box death mid-family loses at most one run's compute.
#   (e) NEVER destroy the box from this launcher. Teardown is the operator's /
#       owner's decision, taken separately after the artifacts are verified on
#       the relay.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38grid

echo "[ts38grid] estimated cost: 24 LoRA target runs (3 arms x 8 new sizes) on"
echo "[ts38grid] one RTX 4090, no parent builds. Measured ts38pf per-run wall"
echo "[ts38grid] clock was 1.7 / 2.1 / 6.4 / 10.5 / 23.6 min of training at"
echo "[ts38grid] n = 1000 / 4642 / 21544 / 100000 / 316228 -- log-linear in n"
echo "[ts38grid] above ~5K. Extrapolated per arm over the 8 new sizes:"
echo "[ts38grid] 1+1+1+2+4+8+13+18 ~= 48 min, so ~2.4 h for three arms, plus"
echo "[ts38grid] G5 (~0.5 min x 24) plus setup/datagen/parent pulls+merges"
echo "[ts38grid] => ~3 h wall, ~\$1.0-1.4 at \$0.35-0.45/h; disk < 3 GB."
echo "[ts38grid] (Those are the full 8-size default; a SIZES subset costs"
echo "[ts38grid] proportionally less -- the 128/256/512 bracket is minutes.)"
echo "[ts38grid] Ceilings (max_steps per overlay) must never bind --"
echo "[ts38grid] stop_reason=max_steps is a bug signal, never a result."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38grid_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}

MW_PARENT_RID=evt-ts38mw-parent-probe-lr3e-4
MW_PARENT_STEP=28000
PF_PARENT_RID=evt-ts38pf-preteachfmt-parent
# The pf parent's build-time data_order_hash pin, verbatim from
# launch_ts38pf_family.sh — the stale-parent guard in stage 4.
PREFMT_ORDER_HASH=5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1

DEFAULT_SIZES=(128 256 512 2154 10000 46416 146780 215443)
if [[ -n ${SIZES:-} ]]; then
  read -r -a SIZES <<<"$SIZES"
else
  SIZES=("${DEFAULT_SIZES[@]}")
fi
RUN_COUNT=$((3 * ${#SIZES[@]}))

BASE_CONFIG=../configs/ts38_base.yaml
MW_CONFIG=../configs/ts38mw_pretaught.yaml
PF_CONFIG=../configs/ts38pf_preteachfmt.yaml
OVERLAY_DIR=../configs/sweeps/ts38
# Bare-NL eval pin (D_algo_eval_bare), reused verbatim from the ts38 family —
# every target run's G5 reads it, so every run is scored on identical data.
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE sizes=${#SIZES[@]} runs=$RUN_COUNT base=$BASE_RID parents=$MW_PARENT_RID,$PF_PARENT_RID"
milestone "grid ${SIZES[*]}"

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

# Standing policy (feedback-run-until-convergence): a target run stops because
# eps/k said so, never because it ran out of steps. ts38pf checked this only on
# its cheapest run; here EVERY run is checked, because the 8 new sizes were
# never exercised at this LR and the two smallest have single-digit min_steps.
require_converged() {
  local rid=$1 n=$2 stop
  stop=$(stop_reason_of "$rid" target_result)
  milestone "convergence_check run=$rid stop_reason=$stop"
  [[ $stop == converged ]] || fail \
    "CONVERGENCE CHECK: $rid ended with stop_reason='$stop', not 'converged'. Standing
   policy: a max_steps stop is a BUG SIGNAL, not a result. A converged-status run at
   n=$n that hit its ceiling means the >=20-epoch ceiling bound before eps/k did —
   inspect the overlay's max_steps and the loss trace before continuing; do not widen
   a bar and do not let the remaining sizes inherit whatever this is."
}

# G5 evidence on the bare target eval (zero/16-shot EM + shared-set test loss
# on identical data for every run). No pass bar by design — the ts38pf
# family's record_g5, verbatim.
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

# Best-effort push, hard rule (d): a failed upload warns and the grid keeps
# going; stage 7's receiver check is the thing that can actually fail the run.
push_run() {
  local rid=$1
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$RELAY_REPO"; then
    milestone "push_complete run=$rid"
  else
    echo "[ts38grid] WARN hf_checkpoint.py push failed for $rid (best effort)"
    milestone "push_warn run=$rid"
  fi
}

# Merge a pulled LoRA parent to plain weights and prove the merge preserved the
# function. Sets the global MERGED_DIR (callers copy it immediately); the body
# is launch_ts38pf_family.sh's merge + receiver-check block, parameterised.
MERGED_DIR=
merge_and_verify() {
  local rid=$1 merged receiver_out receiver_status max_diff
  merged=$GEODE_STORE/runs/$rid/model_merged
  if [[ -f $merged/model.safetensors ]]; then
    milestone "merge_skip $merged already exists"
  elif [[ -d $merged ]]; then
    fail "$merged exists but holds no model.safetensors — a crashed merge
   (merge_adapter.py refuses to overwrite an existing directory). Remove $merged and
   rerun this launcher."
  else
    python3 merge_adapter.py --run-id "$rid" || fail "merge_adapter.py failed for $rid"
    [[ -f $merged/model.safetensors ]] ||
      fail "merge_adapter.py exited 0 but left no $merged/model.safetensors"
    milestone "merge_complete $merged"
  fi

  # Receiver check (specs/00 V0.9 lineage): wrapped-vs-merged logits must agree
  # to <1e-3 on seeded random token batches — the only proof that folding A/B
  # into the base weights preserved the function.
  receiver_out=$(python3 - "$rid" "$merged" <<'PY'
import os
import sys
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from geode.zoo import load_model

run_id, merged_dir = sys.argv[1], Path(sys.argv[2])
dev = "cuda" if torch.cuda.is_available() else "cpu"
store = Path(os.environ["GEODE_STORE"])

wrapped = load_model(run_id, store=store, device=dev)
merged = LlamaForCausalLM.from_pretrained(merged_dir).to(dev).eval()
vocab = merged.config.vocab_size

generator = torch.Generator().manual_seed(316)
max_abs_diff = 0.0
with torch.no_grad():
    for _ in range(4):
        ids = torch.randint(0, vocab, (8, 32), generator=generator).to(dev)
        w_logits = wrapped(ids).logits.float()
        m_logits = merged(ids).logits.float()
        max_abs_diff = max(max_abs_diff, (w_logits - m_logits).abs().max().item())

print(f"[ts38grid] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
sys.exit(0 if max_abs_diff < 1e-3 else 1)
PY
)
  receiver_status=$?
  echo "$receiver_out"
  [[ $receiver_status -eq 0 ]] || fail \
    "receiver check FAILED for $rid — the merged checkpoint does not reproduce the
   wrapped model's logits within 1e-3 (see output above); do NOT hand this off to the
   target arms"
  max_diff=$(sed -n 's/.*max_abs_logit_diff=\(.*\)/\1/p' <<<"$receiver_out")
  milestone "parent_merged run=$rid max_abs_logit_diff=$max_diff"
  MERGED_DIR=$merged
}

# ---- stage 1: the base checkpoint, receiver-verified (ts38pf's stage 1,
# verbatim — the base arm at every new size warm-starts from it) ------------
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38grid family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38grid] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38grid] MISSING on the hub: {missing}")
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
print(f"[ts38grid] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38grid] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38grid] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 2: data artifacts + the bare derivations -----------------------
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
  python3 ../datagen/make_bare_sets.py --out "$DATA_DIR" --skip-dose-mult || fail "make_bare_sets (bare derivations)"
  milestone "bare_datagen_complete"
else
  milestone "bare_datagen_skip (all bare artifacts present)"
fi

# No D_preteachfmt derivation here: the pf parent is PULLED already-built in
# stage 4, so its training data never has to exist on this box.

# Every config this launch will hand to train_target.py must exist BEFORE any
# GPU spend — a missing overlay after 20 runs is the expensive failure mode.
MISSING_FILES=()
for f in "$BASE_CONFIG" "$MW_CONFIG" "$PF_CONFIG" "$BARE_EVAL_CONFIG"; do
  [[ -f $f ]] || MISSING_FILES+=("$f")
done
for n in "${SIZES[@]}"; do
  for f in "$OVERLAY_DIR/ts38_base_n${n}.yaml" \
    "$OVERLAY_DIR/ts38mw_pretaught_n${n}.yaml" \
    "$OVERLAY_DIR/ts38pf_preteachfmt_n${n}.yaml"; do
    [[ -f $f ]] || MISSING_FILES+=("$f")
  done
done
((${#MISSING_FILES[@]} == 0)) || fail \
  "missing config/overlay file(s) for this grid: ${MISSING_FILES[*]}
   — every size in SIZES needs all three overlays committed before launch"
milestone "configs_present targets=3 overlays=$((3 * ${#SIZES[@]}))"

# The three target configs' data blocks are byte-identical by construction
# (G7: same D_algo_bare.parquet, same order_hash 946b5d02…), so one order-hash
# pre-flight covers all three arms; the bare eval config carries the second.
python3 - "$BASE_CONFIG" "$BARE_EVAL_CONFIG" <<'PY' || fail "data pre-flight (frozen order hash mismatch — see output above)"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval = sys.argv[1:3]
for label, path in (
    ("D_algo_bare", Path(target_cfg_path)),
    ("D_algo_eval_bare", Path(bare_eval)),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38grid] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare order hashes OK"

# ---- stage 3: NEW SIZES ONLY (hard rule (c)) -------------------------------
# train_or_skip keys on the LOCAL store: a run whose only copy lives on the
# relay looks "missing" to it and would be RETRAINED, overwriting an immutable
# published result. Refuse when a run id in this launch is on the hub but is
# not complete locally. A run that is on the hub AND complete locally is the
# normal same-box relaunch after a partial family — train_or_skip skips it, so
# it is reported and allowed (hard rule (d)'s push-as-you-go needs restarts to
# work). A fresh box resuming a partial family must pull those runs (or narrow
# SIZES) first, deliberately.
python3 - "$RELAY_REPO" "${SIZES[@]}" <<'PY' || fail "SIZE GUARD: at least one run id in this launch already exists on $RELAY_REPO without a complete local copy (see output above). This launcher trains NEW sizes only — the shipped sizes are immutable and are reused by launch_ts38_mini.sh / launch_ts38mw_family.sh / launch_ts38pf_family.sh. Narrow SIZES, or pull the existing runs into \$GEODE_STORE first if you meant to resume."
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo, sizes = sys.argv[1], sys.argv[2:]
store = Path(os.environ["GEODE_STORE"])
files = set(HfApi().list_repo_files(repo, repo_type="model"))
clashes = []
for n in sizes:
    for rid in (
        f"evt-ts38-base-n{n}",
        f"evt-ts38mw-pretaught-n{n}",
        f"evt-ts38pf-preteachfmt-n{n}",
    ):
        if f"runs/{rid}/manifest.json" not in files:
            continue
        p = store / "runs" / rid / "manifest.json"
        local = json.loads(p.read_text()).get("status", "unreadable") if p.is_file() else "missing"
        if local == "complete":
            print(f"[ts38grid] {rid}: on the relay and complete locally -> resume, will be skipped")
        else:
            clashes.append(f"{rid} (on relay, local status={local})")
for c in clashes:
    print(f"[ts38grid] SIZE CLASH: {c}")
sys.exit(1 if clashes else 0)
PY
milestone "size_guard_ok sizes=${#SIZES[@]} (no relay run id would be retrained)"

# ---- stage 4: the two theta0 parents — pulled, verified, merged ------------
# Both are DELIBERATELY UNGATED (hard rule (a)): these blocks read manifests
# and refuse on anything other than the exact owner-selected checkpoints. They
# record nothing, they never invoke gates.py, and neither parent is re-pushed.

# --- 4a: the GO-B multiwrap parent (launch_ts38mw_family.sh stage 4) --------
if [[ ! -f $GEODE_STORE/runs/$MW_PARENT_RID/manifest.json ]]; then
  milestone "pull_parent_start run=$MW_PARENT_RID"
  python3 hf_checkpoint.py pull --run-id "$MW_PARENT_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $MW_PARENT_RID from $RELAY_REPO"
fi
[[ -f $GEODE_STORE/runs/$MW_PARENT_RID/model/adapter.safetensors ]] ||
  fail "pull left no $GEODE_STORE/runs/$MW_PARENT_RID/model/adapter.safetensors — the parent's LoRA adapter must be on the relay"
[[ -f $GEODE_STORE/runs/$MW_PARENT_RID/manifest.json ]] ||
  fail "pull left no manifest for $MW_PARENT_RID — geode.zoo cannot load a run without it"

MW_FIELDS=$(python3 - "$MW_PARENT_RID" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rid = sys.argv[1]
p = store / "runs" / rid / "manifest.json"
d = json.loads(p.read_text())
status = d.get("status", "MISSING")
method = d.get("training", {}).get("method", "MISSING")
gates = d.get("experiment", {}).get("gates", {}) or {}
sft_result = d.get("experiment", {}).get("sft_result", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", sft_result.get("stop_reason", "MISSING"))
final_step = meta.get("final_step", sft_result.get("final_step", "MISSING"))
gate_names = ",".join(sorted(gates)) if gates else "none"
print(status, method, gate_names, stop_reason, final_step)
PY
)
read -r MW_STATUS MW_METHOD MW_GATES MW_STOP MW_STEP <<<"$MW_FIELDS"
milestone "parent_fields run=$MW_PARENT_RID status=$MW_STATUS method=$MW_METHOD gates=$MW_GATES stop_reason=$MW_STOP final_step=$MW_STEP"

[[ $MW_STATUS == complete ]] || fail "$MW_PARENT_RID manifest status='$MW_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$MW_PARENT_RID/manifest.json"
[[ $MW_METHOD == lora ]] || fail "$MW_PARENT_RID training.method='$MW_METHOD' (expected lora) — the pretaught-mw arm merges a LoRA adapter; a non-LoRA checkpoint is the wrong run in this slot"
[[ $MW_GATES == none ]] || fail \
  "$MW_PARENT_RID has recorded gates {$MW_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
[[ $MW_STOP == converged ]] || fail "$MW_PARENT_RID stop_reason='$MW_STOP' (expected converged) — the owner selected step $MW_PARENT_STEP precisely because it converged; do not pick another"
[[ $MW_STEP == "$MW_PARENT_STEP" ]] || fail "$MW_PARENT_RID final_step=$MW_STEP (expected $MW_PARENT_STEP) — the owner selected step $MW_PARENT_STEP as theta0; do not pick another"
milestone "parent_verified run=$MW_PARENT_RID step=$MW_PARENT_STEP gates={} method=lora"

merge_and_verify "$MW_PARENT_RID"
MW_MERGED=$MERGED_DIR

# --- 4b: the format-only parent (launch_ts38pf_family.sh stage 4) -----------
if [[ ! -f $GEODE_STORE/runs/$PF_PARENT_RID/manifest.json ]]; then
  milestone "pull_parent_start run=$PF_PARENT_RID"
  python3 hf_checkpoint.py pull --run-id "$PF_PARENT_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $PF_PARENT_RID from $RELAY_REPO"
fi
[[ -f $GEODE_STORE/runs/$PF_PARENT_RID/model/adapter.safetensors ]] ||
  fail "pull left no $GEODE_STORE/runs/$PF_PARENT_RID/model/adapter.safetensors — the parent's LoRA adapter must be on the relay"
[[ -f $GEODE_STORE/runs/$PF_PARENT_RID/manifest.json ]] ||
  fail "pull left no manifest for $PF_PARENT_RID — geode.zoo cannot load a run without it"

PF_FIELDS=$(python3 - "$PF_PARENT_RID" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rid = sys.argv[1]
p = store / "runs" / rid / "manifest.json"
d = json.loads(p.read_text())
status = d.get("status", "MISSING")
method = d.get("training", {}).get("method", "MISSING")
exp = d.get("experiment", {}) or {}
gates = exp.get("gates", {}) or {}
sft_result = exp.get("sft_result", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", sft_result.get("stop_reason", "MISSING"))
final_step = meta.get("final_step", sft_result.get("final_step", "MISSING"))
gate_names = ",".join(sorted(gates)) if gates else "none"
data_hash = exp.get("data_order_hash", "MISSING")
print(status, method, gate_names, stop_reason, final_step, data_hash)
PY
)
read -r PF_STATUS PF_METHOD PF_GATES PF_STOP PF_STEP PF_DATA_HASH <<<"$PF_FIELDS"
milestone "parent_fields run=$PF_PARENT_RID status=$PF_STATUS method=$PF_METHOD gates=$PF_GATES stop_reason=$PF_STOP final_step=$PF_STEP data_order_hash=$PF_DATA_HASH"

[[ $PF_STATUS == complete ]] || fail "$PF_PARENT_RID manifest status='$PF_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$PF_PARENT_RID/manifest.json"
[[ $PF_METHOD == lora ]] || fail "$PF_PARENT_RID training.method='$PF_METHOD' (expected lora)"
[[ $PF_GATES == none ]] || fail \
  "$PF_PARENT_RID has recorded gates {$PF_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
[[ $PF_STOP == converged ]] || fail \
  "$PF_PARENT_RID stop_reason='$PF_STOP' (expected converged) — a max_steps stop on the
   format-only parent is a bug signal; do not proceed with an unconverged parent"
# Stale-parent guard, verbatim from launch_ts38pf_family.sh: the manifest's own
# data_order_hash is the cheapest fingerprint tying this pulled checkpoint back
# to the exact D_preteachfmt.parquet the pf family built it from. A mismatch
# means the relay holds a different parent than the one this grid extends.
[[ $PF_DATA_HASH == "$PREFMT_ORDER_HASH" ]] || fail \
  "$PF_PARENT_RID manifest data_order_hash=$PF_DATA_HASH != pin $PREFMT_ORDER_HASH —
   the checkpoint on the relay was built from a different D_preteachfmt.parquet/config
   than the one the shipped ts38pf sizes used. Extending that family's grid with a
   different parent would silently mix two theta0's; inspect, do not proceed."
milestone "parent_verified run=$PF_PARENT_RID stop_reason=converged final_step=$PF_STEP gates={} method=lora data_order_hash=OK"

merge_and_verify "$PF_PARENT_RID"
PF_MERGED=$MERGED_DIR

# ---- stage 5: the grid — ascending n, base arm FIRST at each size ----------
# ORDER IS LOAD-BEARING. train_target.py resolves G7 (spec 02 §8) against the
# LOCAL store's evt-ts38-base-n{N} manifest: the mw and pf overlays name it in
# match_data_order_with, and the run refuses if its (data_order_hash,
# n_examples) do not match. The base arm at size N therefore has to be complete
# before either pretaught arm at that size is launched.
#
# SMALL-n NOTE: n=128/256/512 are 1/2/4 optimizer steps per epoch (batch 128),
# and require_full_epoch1 pins min_steps to exactly that. These runs take
# seconds and overfit hard after epoch 1 — that is EXPECTED, not a failure. It
# is precisely what the OCV-floor analysis measures (its overshoot_ratio
# column); the floor is per-run converged val, so a large train/val gap does
# not corrupt the EDL reading.
for n in "${SIZES[@]}"; do
  milestone "size_start n=$n"

  BASE_ARM_RID=evt-ts38-base-n${n}
  train_or_skip "$BASE_ARM_RID" \
    python3 train_target.py --config "$BASE_CONFIG" \
      --override "$OVERLAY_DIR/ts38_base_n${n}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost
  require_converged "$BASE_ARM_RID" "$n"
  record_g5 "$BASE_ARM_RID"
  push_run "$BASE_ARM_RID"

  MW_ARM_RID=evt-ts38mw-pretaught-n${n}
  train_or_skip "$MW_ARM_RID" \
    python3 train_target.py --config "$MW_CONFIG" \
      --override "$OVERLAY_DIR/ts38mw_pretaught_n${n}.yaml" \
      --init-from "$MW_MERGED" --confirm-cost
  require_converged "$MW_ARM_RID" "$n"
  record_g5 "$MW_ARM_RID"
  push_run "$MW_ARM_RID"

  PF_ARM_RID=evt-ts38pf-preteachfmt-n${n}
  train_or_skip "$PF_ARM_RID" \
    python3 train_target.py --config "$PF_CONFIG" \
      --override "$OVERLAY_DIR/ts38pf_preteachfmt_n${n}.yaml" \
      --init-from "$PF_MERGED" --confirm-cost
  require_converged "$PF_ARM_RID" "$n"
  record_g5 "$PF_ARM_RID"
  push_run "$PF_ARM_RID"

  milestone "size_complete n=$n arms=3"
done

# ---- stage 6: summary table ------------------------------------------------
python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts38grid] {'run_id':<32}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    for rid in (
        f"evt-ts38-base-n{n}",
        f"evt-ts38mw-pretaught-n{n}",
        f"evt-ts38pf-preteachfmt-n{n}",
    ):
        p = store / "runs" / rid / "manifest.json"
        r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}
        r = r or {}
        edl = r.get("edl_per_label_token_nats")
        print(
            f"[ts38grid] {rid:<32}{r.get('final_step', 'MISSING'):>12}"
            f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
            f"{edl if edl is not None else 'n/a':>26}"
        )
PY
milestone "family_complete runs=$RUN_COUNT"

# ---- stage 7: receiver-verify every run on the hub -------------------------
# Verify the RECEIVER, not the sender ([[feedback-verify-the-receiver-not-the-
# sender]]): push bookkeeping looks identical over an empty relay. Each run was
# already pushed as it finished (hard rule (d)); this re-pushes only whatever
# the hub is actually missing, once, then re-checks. The parents are NOT
# re-pushed — they were pulled unchanged.
run_receiver_check() {
  python3 - "$RELAY_REPO" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
sizes = sys.argv[2:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for n in sizes:
    for rid in (
        f"evt-ts38-base-n{n}",
        f"evt-ts38mw-pretaught-n{n}",
        f"evt-ts38pf-preteachfmt-n{n}",
    ):
        required = [
            f"runs/{rid}/manifest.json",
            f"runs/{rid}/eval_log.jsonl",
            f"runs/{rid}/logs/prequential.jsonl",
        ]
        missing = [r for r in required if r not in files]
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {rid}: {status}")
        if missing:
            print(f"MISSING {rid}")
            ok = False
sys.exit(0 if ok else 1)
PY
}

RECEIVER_OUT=$(run_receiver_check)
RECEIVER_STATUS=$?
echo "[ts38grid] receiver check (hub files for all $RUN_COUNT runs):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing the runs the hub is missing)"
  while read -r rid; do
    push_run "$rid"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts38grid] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above (at least one run's manifest/eval_log/prequential log is still missing on the relay after a retry)"
milestone "targets_pushed sizes=${#SIZES[@]} runs=$RUN_COUNT receiver=OK"

# ---- stage 8: finish --------------------------------------------------------
echo "[ts38grid] MILESTONE analysis_commands"
echo "[ts38grid]   CPU-only, run off \$GEODE_STORE. Each family's run-id regex"
echo "[ts38grid]   already picks up the new sizes automatically (each pass"
echo "[ts38grid]   collects the base arm plus its own arm), so no analysis"
echo "[ts38grid]   change is needed — just re-run all three:"
echo "[ts38grid]     python3 ../analysis/edl_converged_val_floor.py --family ts38"
echo "[ts38grid]     python3 ../analysis/edl_converged_val_floor.py --family ts38mw"
echo "[ts38grid]     python3 ../analysis/edl_converged_val_floor.py --family ts38pf"
echo "[ts38grid]   OCV primary, and the FLOOR IS NAMED on every figure — the"
echo "[ts38grid]   three floor definitions disagree per-size and curve shape is"
echo "[ts38grid]   meaningless without one ([[project-edl-floor-artifact]])."
echo "[ts38grid]   Read the SMALL-n BRACKET FIRST (128/256/512: is the base"
echo "[ts38grid]   arm's rising limb below the old grid, i.e. floor(128) >"
echo "[ts38grid]   3.47 nats?), and only then the densified 4642->21544 hump."
echo "[ts38grid]   Both readouts are pre-registered in decisions.md 2026-08-16"
echo "[ts38grid]   \"ts38grid pre-registration\" — do not re-derive them after"
echo "[ts38grid]   seeing the numbers."
echo "[ts38grid]   The box is NOT torn down here (hard rule (e)); teardown is"
echo "[ts38grid]   the operator's call once these artifacts are verified."
notify "ts38grid family done: $RUN_COUNT runs (3 arms x ${#SIZES[@]} sizes)"
echo "[ts38grid] TERMINAL_SUCCESS runs=$RUN_COUNT"
