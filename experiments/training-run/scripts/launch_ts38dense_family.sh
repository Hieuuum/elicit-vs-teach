#!/usr/bin/env bash
# ts38dense — target-size grid DENSIFICATION for three ts38 arms: base/teach
# (evt-ts38-base), ts38pp paper-protocol pre-teach (evt-ts38pp-pretaught,
# App. E.2: full-FT parent on 4M unique op examples), and the ts38fs i=1000
# format-only-install row (evt-ts38fs, format-only LoRA install, 1000
# permuted-label examples). Every one of the five SHIPPED sizes for these
# three arms already exists and is reused unchanged; this launcher trains
# ONLY the 5 NEW sizes below, going from a 5-point grid to a 10-point one.
# 15 new target runs total, 0 new parents (all three theta0's are pulled,
# never rebuilt).
#
# THE GRID (3 arms x 5 NEW sizes = 15 runs; ascending n, base FIRST at each
# size within that size's group)
#   evt-ts38-base-n{N}          base/teach arm; theta0 = the 38.7M
#                                TinyStories base evt-run1-base-v3-ext. This
#                                run IS the G7 anchor for the other two arms
#                                at size N.
#   evt-ts38pp-pretaught-n{N}   theta0 = PLAIN full-FT evt-ts38pp-parent
#                                (paper-protocol pre-teach, no merge stage).
#   evt-ts38fs-i1000-n{N}-s316  theta0 = MERGED evt-ts38fs-parent-n1000
#                                (format-only LoRA install, i=1000, seed 316).
#   for N in 2154 10000 46416 146780 215443 (default; SIZES env override
#   below).
#
# SHIPPED (immutable, NEVER retrained): {1000, 4642, 21544, 100000, 316228}.
# Every launcher that trains these three arms at those five sizes already
# exists (launch_ts38_mini.sh / launch_ts38grid_family.sh for base,
# launch_ts38pp_family.sh for pp, launch_ts38fs_family.sh for the i=1000 fs
# row) and every published figure reuses those run ids as-is — this
# launcher refuses outright if SIZES names any of them (stage 1).
#
# ORDER IS LOAD-BEARING, WITHIN EACH SIZE: train_target.py:411 resolves G7
# (spec 02 §8) against the LOCAL store's evt-ts38-base-n<N> manifest — the
# pp overlay names it via match_data_order_with, and the run refuses if its
# (data_order_hash, n_examples) do not match. The base arm at size N
# therefore has to be COMPLETE, locally, before the pp arm at that size
# trains. The fs overlay carries NO match_data_order_with (ts38fs proper
# never G7-paired — see ts38fs_target.yaml's header), so it has no hard
# dependency on the anchor; it is simply run third for a uniform
# base -> pp -> fs order at every size. Sizes themselves may be visited in
# any order (each size's anchor is independent of every other size's).
#
# G5 CONVENTION PER ARM (deliberately NOT uniform):
#   base, pp  -> record_g5 (bare-NL eval evidence) on every run, matching
#                launch_ts38grid_family.sh / launch_ts38pp_family.sh.
#   fs        -> NO record_g5, ever. ts38fs-proper never records G5 on its
#                target cells either (launch_ts38fs_family.sh /
#                launch_ts38fs_tiny_family.sh have no record_g5 call on a
#                target run — only a --no-record theta0 probe at the PARENT
#                stage). Recording G5 on the i1000 dense row here would make
#                it inconsistent with the rest of the i=1000 column that
#                ts38fs already shipped; edl_per_label_token_nats is still
#                written into target_result by train_target.py itself
#                regardless of G5, so the dose-curve analysis is unaffected.
#
# REUSE RULE: all three theta0's are PULLED from the relay, never rebuilt.
# evt-ts38pp-parent is plain full-FT weights (no merge stage — hard rule
# (b)); evt-ts38fs-parent-n1000 is a LoRA adapter, folded via
# merge_adapter.py before any target run trains against it, same discipline
# as every other ts38(*) family.
#
# HARD RULES
#   (a) all three theta0's are DELIBERATELY UNGATED in the certification
#       sense here — this launcher never runs gates.py against any of them,
#       not even --no-record; their theta0 evidence already exists
#       (ts38pp_family_theta0.json / ts38fs's own family evidence) and is
#       not re-derived. Nothing is ever written to a parent manifest by this
#       launcher.
#   (b) LoRA checkpoints load ONLY via geode.zoo.load_model
#       ([[feedback-lora-checkpoints-load-via-zoo-load-model]]) — the fs arm
#       warm-starts from evt-ts38fs-parent-n1000's MERGED plain weights
#       (runs/evt-ts38fs-parent-n1000/model_merged/), never from_pretrained
#       on the wrapped runs/evt-ts38fs-parent-n1000/model/. The pp arm has
#       no merge stage at all — its parent IS plain weights already.
#   (c) NEW SIZES ONLY (stage 1). train_or_skip keys on the LOCAL store, so
#       a run whose only copy lives on the relay looks "missing" to it and
#       would be RETRAINED, silently overwriting an immutable published
#       result.
#   (d) PUSH AS YOU GO. Every run is pushed the moment it is trained (+
#       scored, for base/pp), so a box death mid-family loses at most one
#       run's compute.
#   (e) NEVER destroy the box from this launcher. Teardown is the operator's
#       / owner's decision, taken separately after the artifacts are
#       verified on the relay.
#
# SIZES OVERRIDE: set the SIZES env var (space-separated) to run a subset —
# same convention as launch_ts38grid_family.sh. Stage 1's shipped-size guard
# still applies to whatever SIZES ends up being.
#     SIZES="10000 46416" bash launch_ts38dense_family.sh --confirm-cost
#
# hf_checkpoint.py NOTE (evt-ts38pp-parent pull): the tool's default
# snapshot-skip ignore pattern is literally "snapshots/*" (both push() and
# pull(), hf_checkpoint.py:129/184). evt-ts38pp-parent was built by
# train_sft.py, which writes its mid-run checkpoints to a DIFFERENTLY NAMED
# directory, sft_snapshots/ (train_sft.py:473) — the default ignore does not
# match that name, and launch_ts38pp_family.sh's own stage-4 push used
# --with-snapshots, so sft_snapshots/ genuinely is on the relay for this run.
# A plain (non---with-snapshots) pull here therefore pulls it anyway, ~465MB
# unasked. Accepted deliberately rather than passing --no-weights (which
# would also drop model.safetensors) or editing hf_checkpoint.py's ignore
# list (out of this launcher's scope, and a shared file every other launcher
# depends on).
set -uo pipefail

# env guard: detached tmux / non-interactive shells never source ~/.bashrc,
# so /workspace/venv/bin is not on PATH and bare python3 is the system
# interpreter (the 2026-08-16 ts38grid relaunch crash — ModuleNotFoundError:
# huggingface_hub at the first heredoc).
#
# ORDER MATTERS: /etc/environment carries its own PATH="/usr/local/sbin:…"
# line (no venv dir in it); sourcing it AFTER venv activation clobbers venv
# back off PATH (caught live on the ts38pp launch attempt, 2026-08-16 —
# activate-then-source silently reintroduced the exact bug this guard exists
# to fix). Source /etc/environment FIRST for HF_TOKEN etc., THEN activate
# the venv so its PATH prepend is the one that survives.
[[ -f /etc/environment ]] && { set -a; . /etc/environment; set +a; }   # HF_TOKEN etc. written by box_onstart.sh
[[ -f /workspace/venv/bin/activate ]] && . /workspace/venv/bin/activate
python3 - <<'PY' || { echo "[ts38dense] FAILED: python3 lacks required modules (venv not active?)"; exit 1; }
import huggingface_hub, torch, geode  # noqa: F401
PY

cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38dense

echo "[ts38dense] estimated cost: 15 LoRA target runs (3 arms x 5 new sizes)"
echo "[ts38dense] on one RTX 4090, no parent builds -- all three theta0's are"
echo "[ts38dense] pulled. Measured ts38(pf/pp) per-run wall clock at the five"
echo "[ts38dense] SHIPPED sizes: 1.7 / 2.1 / 6.4 / 10.5 / 23.6 min at n ="
echo "[ts38dense] 1000 / 4642 / 21544 / 100000 / 316228. Interpolated for the"
echo "[ts38dense] 5 NEW sizes: 2154 ~= 1.8, 10000 ~= 3.5, 46416 ~= 7.5,"
echo "[ts38dense] 146780 ~= 13, 215443 ~= 17 min -- sum ~= 43 min/arm x 3 arms"
echo "[ts38dense] ~= 2.2h compute, plus G5 (~0.5 min x 10 -- base+pp only,"
echo "[ts38dense] fs never records G5) plus parent pulls/merge/pushes =>"
echo "[ts38dense] ~= 2.5h wall, ~= \$1 at \$0.35-0.45/h; disk < 2 GB (no new"
echo "[ts38dense] parent weights -- all three theta0's already exist)."
echo "[ts38dense] Ceilings (max_steps per overlay) must never bind --"
echo "[ts38dense] stop_reason=max_steps is a bug signal, never a result."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38dense_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}

PP_PARENT_RID=evt-ts38pp-parent
PP_PARENT_MODEL=$GEODE_STORE/runs/$PP_PARENT_RID/model   # full FT -> no _merged stage (hard rule (b))
PP_PARENT_FINAL_STEP=31093   # pinned one-epoch length, see ts38pp_parent.yaml header

FS_PARENT_RID=evt-ts38fs-parent-n1000

SHIPPED_SIZES=(1000 4642 21544 100000 316228)
DEFAULT_SIZES=(2154 10000 46416 146780 215443)
if [[ -n ${SIZES:-} ]]; then
  read -r -a SIZES <<<"$SIZES"
else
  SIZES=("${DEFAULT_SIZES[@]}")
fi
RUN_COUNT=$((3 * ${#SIZES[@]}))

BASE_CONFIG=../configs/ts38_base.yaml
PP_CONFIG=../configs/ts38pp_pretaught.yaml
FS_CONFIG=../configs/ts38fs_target.yaml
OVERLAY_DIR=../configs/sweeps/ts38
# Bare-NL eval pin (D_algo_eval_bare), reused verbatim from the ts38 family —
# every base/pp run's G5 reads it, so every scored run is scored on
# identical data.
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE sizes=${#SIZES[@]} runs=$RUN_COUNT base=$BASE_RID parents=$PP_PARENT_RID,$FS_PARENT_RID"
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
# eps/k said so, never because it ran out of steps. Checked for EVERY run at
# EVERY size, all three arms (ts38grid's rule) — these 5 new sizes were never
# exercised at this LR/recipe before.
require_converged() {
  local rid=$1 n=$2 stop
  stop=$(stop_reason_of "$rid" target_result)
  milestone "convergence_check run=$rid stop_reason=$stop"
  [[ $stop == converged ]] || fail \
    "CONVERGENCE CHECK: $rid ended with stop_reason='$stop', not 'converged'. Standing
   policy: a max_steps stop is a BUG SIGNAL, not a result. Inspect the overlay's max_steps
   and the loss trace before continuing; do not let the remaining sizes/arms inherit
   whatever this is."
}

# G5 evidence on the bare target eval (zero/16-shot EM + shared-set test loss
# on identical data). Called for base/pp only — see the header's G5
# CONVENTION note for why the fs arm never gets this.
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

# Merge a pulled LoRA parent to plain weights and prove the merge preserved
# the function. Sets the global MERGED_DIR (callers copy it immediately);
# verbatim from launch_ts38grid_family.sh / launch_ts38fs_tiny_family.sh.
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

print(f"[ts38dense] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
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

# G7 anchor preflight — right after the base run at size N completes, before
# either pp or fs at that size trains. Unlike ts38pp's own version (which
# pulls a REUSED anchor from the relay before the whole loop), the anchor
# here is trained LOCALLY in this same per-size loop, so the check runs once
# per size, immediately after that size's base run.
check_g7_anchor() {
  local rid=$1 n=$2
  python3 - "$rid" "$n" <<'PY' || fail \
    "G7 ANCHOR CHECK: $rid at n=$n -- see output above (the base run just trained at this
   size has a missing/incomplete manifest, or its data_order_hash/n_examples are unset).
   The pp and fs arms at this size cannot proceed without a valid anchor; inspect before
   continuing."
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rid, n = sys.argv[1], sys.argv[2]
p = store / "runs" / rid / "manifest.json"
if not p.is_file():
    print(f"[ts38dense] MISSING manifest for {rid}")
    sys.exit(1)
d = json.loads(p.read_text())
status = d.get("status")
exp = d.get("experiment", {})
n_examples = exp.get("n_examples")
order_hash = exp.get("data_order_hash")
good = status == "complete" and n_examples == int(n) and bool(order_hash)
print(
    f"[ts38dense] {rid}: status={status} n_examples={n_examples} "
    f"order_hash={'set' if order_hash else 'MISSING'} -> {'OK' if good else 'BAD'}"
)
sys.exit(0 if good else 1)
PY
  milestone "g7_anchor_ready run=$rid n=$n"
}

# ---- stage 1: NEW-SIZES-ONLY GUARD (hard rule (c)) -------------------------
# train_or_skip keys on the LOCAL store: a run whose only copy lives on the
# relay looks "missing" to it and would be RETRAINED, overwriting an
# immutable published result reused by launch_ts38_mini.sh /
# launch_ts38grid_family.sh / launch_ts38pp_family.sh / launch_ts38fs_family.
# sh and every figure built off them. Refuse outright if SIZES names any of
# the five shipped sizes, rather than trying to be clever about relay state.
for n in "${SIZES[@]}"; do
  for shipped in "${SHIPPED_SIZES[@]}"; do
    if [[ $n == "$shipped" ]]; then
      fail "NEW-SIZES-ONLY GUARD: SIZES contains $n, one of the five SHIPPED sizes
   {${SHIPPED_SIZES[*]}}. Those are immutable results reused by every other ts38(*)
   launcher and every published figure; train_or_skip keys on the LOCAL store, so a
   relay-only run at this size would look 'missing' here and get RETRAINED. Narrow
   SIZES to new sizes only."
    fi
  done
done
milestone "size_guard_ok sizes=${#SIZES[@]} (no shipped size present)"

# ---- stage 2: relay verify + pull the three theta0's -----------------------

# --- 2a: the base checkpoint, receiver-verified (every prior ts38(*) family's
# stage 1, verbatim — the base arm at every new size warm-starts from it, and
# it is the tokenizer/shape reference the other two parents were built
# against). ------------------------------------------------------------------
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38dense family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38dense] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38dense] MISSING on the hub: {missing}")
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
print(f"[ts38dense] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38dense] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38dense] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# --- 2b: the ts38pp paper-protocol parent — plain full-FT weights, NO merge
# stage (hard rule (b)). -------------------------------------------------
milestone "relay_verify_start run=$PP_PARENT_RID repo=$RELAY_REPO"
python3 - "$PP_PARENT_RID" "$RELAY_REPO" <<'PY' || fail "$PP_PARENT_RID is not on the relay — nothing to pull; the ts38dense pp column has no parent"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38dense] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38dense] MISSING on the hub: {missing}")
    sys.exit(1)
PY

if [[ ! -f $PP_PARENT_MODEL/model.safetensors ]]; then
  milestone "pull_parent_start run=$PP_PARENT_RID"
  python3 hf_checkpoint.py pull --run-id "$PP_PARENT_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $PP_PARENT_RID from $RELAY_REPO"
fi
[[ -f $PP_PARENT_MODEL/model.safetensors ]] ||
  fail "pull left no $PP_PARENT_MODEL/model.safetensors — full-FT parent weights must be on the relay"
[[ -f $GEODE_STORE/runs/$PP_PARENT_RID/manifest.json ]] ||
  fail "pull left no manifest for $PP_PARENT_RID — geode.zoo cannot load a run without it"

PP_FIELDS=$(python3 - "$PP_PARENT_RID" <<'PY'
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
print(status, method, gate_names, stop_reason, final_step)
PY
)
read -r PP_STATUS PP_METHOD PP_GATES PP_STOP PP_STEP <<<"$PP_FIELDS"
milestone "parent_fields run=$PP_PARENT_RID status=$PP_STATUS method=$PP_METHOD gates=$PP_GATES stop_reason=$PP_STOP final_step=$PP_STEP"

[[ $PP_STATUS == complete ]] || fail "$PP_PARENT_RID manifest status='$PP_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$PP_PARENT_RID/manifest.json"
[[ $PP_METHOD == full_ft ]] || fail "$PP_PARENT_RID training.method='$PP_METHOD' (expected full_ft — the paper-protocol parent is full FT, not LoRA)"
[[ $PP_GATES == none ]] || fail \
  "$PP_PARENT_RID has recorded gates {$PP_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
# The pinned one-epoch exception (ts38pp_parent.yaml header /
# launch_ts38pp_family.sh stage 4): min_steps == max_steps == 31093, so a
# legitimate stop can read either 'converged' or 'max_steps' — but the step
# count itself is never negotiable.
[[ $PP_STOP == converged || $PP_STOP == max_steps ]] || fail \
  "$PP_PARENT_RID stop_reason='$PP_STOP' (expected converged or max_steps) — neither the
   pinned one-epoch ceiling nor eps/k produced this; inspect before proceeding"
[[ $PP_STEP == "$PP_PARENT_FINAL_STEP" ]] || fail \
  "$PP_PARENT_RID final_step=$PP_STEP (expected $PP_PARENT_FINAL_STEP, the pinned
   one-epoch length) — inspect the manifest; do not train the dense sweep against a
   wrong-length parent"
milestone "parent_verified run=$PP_PARENT_RID stop_reason=$PP_STOP final_step=$PP_STEP gates={} method=full_ft"

# --- 2c: the ts38fs i=1000 format-only LoRA parent — merged before use
# (hard rule (b)). ---------------------------------------------------------
milestone "relay_verify_start run=$FS_PARENT_RID repo=$RELAY_REPO"
python3 - "$FS_PARENT_RID" "$RELAY_REPO" <<'PY' || fail "$FS_PARENT_RID is not on the relay — nothing to pull; the ts38dense fs column has no parent"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/adapter.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38dense] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38dense] MISSING on the hub: {missing}")
    sys.exit(1)
PY

if [[ ! -f $GEODE_STORE/runs/$FS_PARENT_RID/model/adapter.safetensors ]]; then
  milestone "pull_parent_start run=$FS_PARENT_RID"
  python3 hf_checkpoint.py pull --run-id "$FS_PARENT_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $FS_PARENT_RID from $RELAY_REPO"
fi
[[ -f $GEODE_STORE/runs/$FS_PARENT_RID/model/adapter.safetensors ]] ||
  fail "pull left no $GEODE_STORE/runs/$FS_PARENT_RID/model/adapter.safetensors — the parent's LoRA adapter must be on the relay"
[[ -f $GEODE_STORE/runs/$FS_PARENT_RID/manifest.json ]] ||
  fail "pull left no manifest for $FS_PARENT_RID — geode.zoo cannot load a run without it"

FS_FIELDS=$(python3 - "$FS_PARENT_RID" <<'PY'
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
print(status, method, gate_names, stop_reason, final_step)
PY
)
read -r FS_STATUS FS_METHOD FS_GATES FS_STOP FS_STEP <<<"$FS_FIELDS"
milestone "parent_fields run=$FS_PARENT_RID status=$FS_STATUS method=$FS_METHOD gates=$FS_GATES stop_reason=$FS_STOP final_step=$FS_STEP"

[[ $FS_STATUS == complete ]] || fail "$FS_PARENT_RID manifest status='$FS_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$FS_PARENT_RID/manifest.json"
[[ $FS_METHOD == lora ]] || fail "$FS_PARENT_RID training.method='$FS_METHOD' (expected lora)"
[[ $FS_GATES == none ]] || fail \
  "$FS_PARENT_RID has recorded gates {$FS_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
[[ $FS_STOP == converged ]] || fail \
  "$FS_PARENT_RID stop_reason='$FS_STOP' (expected converged) — a max_steps stop on the
   format-only parent is a bug signal; do not proceed with an unconverged parent"
milestone "parent_verified run=$FS_PARENT_RID stop_reason=converged final_step=$FS_STEP gates={} method=lora"

merge_and_verify "$FS_PARENT_RID"
FS_MERGED=$MERGED_DIR

# ---- stage 3: data preflight — D_algo_bare / D_algo_eval_bare -------------
# regenerate-if-missing + order-hash verify, copied from
# launch_ts38fs_tiny_family.sh's stage 2 (minus its install-parent datagen —
# no parents are built here, all three are pulled).
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
BASE_NEEDED=(D_algo_bare.parquet D_algo_eval_bare.parquet D_algo.parquet D_algo_eval.parquet report.json)
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

# The three target configs' data blocks are byte-identical by construction
# (G7: same D_algo_bare.parquet, same order_hash 946b5d02…), so one
# order-hash pre-flight (on the base config) covers all three arms; the bare
# eval config carries the second.
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
    print(f"[ts38dense] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare order hashes OK"

# ---- stage 4: overlay presence — exactly the 15 files this launch needs,
# checked explicitly per size/arm (no directory glob — a glob count can't
# tell you WHICH file is missing, only how many). ----------------------------
MISSING_FILES=()
for f in "$BASE_CONFIG" "$PP_CONFIG" "$FS_CONFIG" "$BARE_EVAL_CONFIG"; do
  [[ -f $f ]] || MISSING_FILES+=("$f")
done
for n in "${SIZES[@]}"; do
  for f in "$OVERLAY_DIR/ts38_base_n${n}.yaml" \
    "$OVERLAY_DIR/ts38pp_pretaught_n${n}.yaml" \
    "$OVERLAY_DIR/ts38fs_dense_i1000_n${n}.yaml"; do
    [[ -f $f ]] || MISSING_FILES+=("$f")
  done
done
((${#MISSING_FILES[@]} == 0)) || fail \
  "missing config/overlay file(s) for this grid: ${MISSING_FILES[*]}
   — every size in SIZES needs all three overlays committed before launch"
milestone "configs_present targets=3 overlays=$((3 * ${#SIZES[@]}))"

# ---- stage 5: the grid — ascending n as given, base arm FIRST at each size
# (order is load-bearing, see header). ---------------------------------------
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

  check_g7_anchor "$BASE_ARM_RID" "$n"

  PP_ARM_RID=evt-ts38pp-pretaught-n${n}
  train_or_skip "$PP_ARM_RID" \
    python3 train_target.py --config "$PP_CONFIG" \
      --override "$OVERLAY_DIR/ts38pp_pretaught_n${n}.yaml" \
      --init-from "$PP_PARENT_MODEL" --confirm-cost
  require_converged "$PP_ARM_RID" "$n"
  record_g5 "$PP_ARM_RID"
  push_run "$PP_ARM_RID"

  FS_ARM_RID=evt-ts38fs-i1000-n${n}-s316
  train_or_skip "$FS_ARM_RID" \
    python3 train_target.py --config "$FS_CONFIG" \
      --override "$OVERLAY_DIR/ts38fs_dense_i1000_n${n}.yaml" \
      --init-from "$FS_MERGED" --confirm-cost
  require_converged "$FS_ARM_RID" "$n"
  # NO record_g5 here — see header's G5 CONVENTION note.
  push_run "$FS_ARM_RID"

  milestone "size_complete n=$n arms=3"
done

# ---- stage 6: summary table ------------------------------------------------
python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts38dense] {'run_id':<32}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    for rid in (
        f"evt-ts38-base-n{n}",
        f"evt-ts38pp-pretaught-n{n}",
        f"evt-ts38fs-i1000-n{n}-s316",
    ):
        p = store / "runs" / rid / "manifest.json"
        r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}
        r = r or {}
        edl = r.get("edl_per_label_token_nats")
        print(
            f"[ts38dense] {rid:<32}{r.get('final_step', 'MISSING'):>12}"
            f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
            f"{edl if edl is not None else 'n/a':>26}"
        )
PY
milestone "family_complete runs=$RUN_COUNT"

# ---- stage 7: receiver-verify every run on the hub -------------------------
# Verify the RECEIVER, not the sender ([[feedback-verify-the-receiver-not-the-
# sender]]): push bookkeeping looks identical over an empty relay. Each run was
# already pushed as it finished (hard rule (d)); this re-pushes only whatever
# the hub is actually missing, once, then re-checks. The three parents are NOT
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
        f"evt-ts38pp-pretaught-n{n}",
        f"evt-ts38fs-i1000-n{n}-s316",
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
echo "[ts38dense] receiver check (hub files for all $RUN_COUNT runs):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing the runs the hub is missing)"
  while read -r rid; do
    push_run "$rid"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts38dense] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above (at least one run's manifest/eval_log/prequential log is still missing on the relay after a retry)"
milestone "targets_pushed sizes=${#SIZES[@]} runs=$RUN_COUNT receiver=OK"

# ---- stage 8: finish --------------------------------------------------------
echo "[ts38dense] MILESTONE analysis_commands"
echo "[ts38dense]   CPU-only, run off \$GEODE_STORE:"
echo "[ts38dense]     python3 ../analysis/edl_converged_val_floor.py --family ts38pp"
echo "[ts38dense]     python3 ../analysis/dataset_size_sweep.py --family ts38pp"
echo "[ts38dense]     python3 ../analysis/ts38fs_dose_curve.py"
echo "[ts38dense]     python3 ../analysis/plot_ts38_all_arms.py"
echo "[ts38dense]   OCV primary, floor named on every figure (three floor"
echo "[ts38dense]   definitions disagree per-size and curve shape is meaningless"
echo "[ts38dense]   without one, [[project-edl-floor-artifact]])."
echo "[ts38dense]   The box is NOT torn down here (hard rule (e)); teardown is"
echo "[ts38dense]   the operator's call once these artifacts are verified."
notify "ts38dense family done: $RUN_COUNT target runs (3 arms x ${#SIZES[@]} sizes)"
echo "[ts38dense] TERMINAL_SUCCESS runs=$RUN_COUNT"
