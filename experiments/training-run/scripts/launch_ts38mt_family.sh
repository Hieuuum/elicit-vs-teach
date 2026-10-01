#!/usr/bin/env bash
# ts38mt — mechanistic-interpretability substrate: an elicit vs. teach pair of
# full-FT parents on the same 38.7M TinyStories base, probed via THREE target
# arms x the FULL 10-point ts38dense grid {1000, 2154, 4642, 10000, 21544,
# 46416, 100000, 146780, 215443, 316228}, EVERY run snapshotted. decisions.md
# 2026-08-21 (night) "ts38mt pre-registration"; EXPERIMENTS.md §6.22 is the
# authoritative design doc — this header summarizes it, does not replace it.
#
# HEADLINE QUESTION: is the sum linearly decodable from the residual stream
# on bare-NL inputs at evt-ts38pp-parent's theta0 (elicit), even though its
# zero-shot NL EM there is ~0? Ten mechanistic tests (tiered, NOT built by
# this launcher — geode/probe, planned) compare an elicit arm (theta0 =
# evt-ts38pp-parent, the already-shipped 4M-example full-FT op pre-teach
# parent) against a NEW teach arm (theta0 = evt-ts38mt-fmt-parent, a
# pre-teach-FORMAT full-FT parent: permuted labels, algorithm absent,
# matched to ts38pp's METHOD -- full FT, not LoRA -- so the two parents
# differ only in data) plus a base (no pre-teach) reference arm. This
# launcher builds ONLY the substrate (1 parent + 30 LoRA target runs, all
# snapshotted, pushed to two HF repos) -- no probing/analysis code here.
#
# THE 31 RUNS
#   evt-ts38mt-fmt-parent        NEW: full FT (train_sft.py, NOT LoRA) on
#                                 D_preteachfmt.parquet -- ts38pf's exact
#                                 21,544-row op-notation permuted-label set
#                                 and order_hash (reused, not regenerated),
#                                 from evt-run1-base-v3-ext. LR 3e-5 PINNED
#                                 (= ts38pp parent's own full-FT LR -- "the
#                                 ladder IS the sweep", no fresh sweep here).
#                                 snapshot_steps (config-owned, ~14 points)
#                                 -> sft_snapshots/step_*/ (~155MB each).
#   evt-ts38mt-base-n{N}         theta0 = evt-run1-base-v3-ext (the plain
#                                 base checkpoint) -- teach REFERENCE arm.
#   evt-ts38mt-pp-n{N}           theta0 = evt-ts38pp-parent (full FT, PULLED
#                                 unchanged, never rebuilt here) -- ELICIT
#                                 arm (owner's framing: op pre-teach = elicit
#                                 per the paper's App. E.1.2/E.2 pairing, NOT
#                                 the literal ts38pp/ts38mt-fmt string match).
#   evt-ts38mt-fmt-n{N}          theta0 = evt-ts38mt-fmt-parent (full FT,
#                                 built by THIS launcher) -- TEACH arm.
#   for N in 1000 2154 4642 10000 21544 46416 100000 146780 215443 316228
#   (the full ts38dense 10-point grid; SIZES env override below).
#
# All three target arms are LoRA r128/alpha32 @1e-3 on D_algo_bare, otherwise
# VERBATIM the ts38_base.yaml recipe (eps/k 0.002/5, batch 128, seed 316,
# require_full_epoch1, run to stop_reason=converged) -- arms differ ONLY in
# theta0 and run id. Every overlay's match_data_order_with:
# evt-ts38-base-n<N> pins the SAME frozen data order every shipped ts38
# family anchors to; for base/pp this makes these runs SEED-IDENTICAL
# RE-RUNS of the already-shipped evt-ts38-base-n<N> / evt-ts38pp-pretaught-
# n<N> (a free reproducibility check gated by decisions.md's pre-registered
# <=5%-relative-EDL/D tolerance -- an ANALYSIS-time check, NOT enforced by
# this launcher). fmt is the only genuinely NEW measurement.
#
# WHAT'S NEW vs. launch_ts38dense_family.sh (the most recent sibling this
# launcher is cloned from):
#   (1) a NEW full-FT format parent is BUILT here (train_sft.py), not just
#       pulled -- ts38dense pulls all three theta0's, builds none.
#   (2) EVERY run is snapshotted: adapter-only snapshots: {n: 32,
#       dense_until: 8} on the 30 target runs (snapshots/step_*/
#       adapter.safetensors, ~48MB each, fewer where a run converges early)
#       and the parent's own train.snapshot_steps (sft_snapshots/step_*/,
#       ~155MB full states each) -- ts38dense's target configs carry
#       snapshots:{n:0} (OFF).
#   (3) DUAL-REPO push: mhieuuu/geode-store (model + manifest, snapshots
#       EXCLUDED by hf_checkpoint.py's default ignore -- Part A of this
#       family's build extended that default to also cover sft_snapshots/*,
#       the full-FT parent's own snapshot dir, distinct from LoRA's
#       snapshots/; before that fix a plain push of a snapshot-bearing
#       full-FT parent leaked sft_snapshots/ unasked) AND the NEW public
#       mhieuuu/geode-internals (--with-snapshots --public -- created by
#       this launcher's first push there, kept public with no auth required
#       to read, same no-secrets discipline as geode-store).
#   (4) all 10 grid sizes are visited (base->pp->fmt per size), not just
#       ts38dense's 5 NEW-vs-shipped ones -- there is no "shipped sizes"
#       guard here: every evt-ts38mt-* run id is brand new, so train_or_skip
#       cannot silently overwrite a prior family's result.
#
# STORAGE ESTIMATE (worst case, on mhieuuu/geode-internals -- geode-store
# never carries snapshots): target adapters <=32 snapshots x ~48MB x 30 runs
# = ~45GB; parent full-state snapshots <=14 x ~155MB = ~2.2GB; combined
# <=~48GB (EXPERIMENTS.md's figure; runs that converge early write fewer
# snapshots, so this is a ceiling, not an expectation).
#
# ORDER IS LOAD-BEARING WITHIN EACH SIZE (base -> pp -> fmt, matching every
# prior ts38(*) launcher's convention) even though none of the three arms
# here trains a NEW G7 anchor for the others to read (unlike ts38dense) --
# all 10 anchor manifests (evt-ts38-base-n<N>, the ALREADY-SHIPPED family)
# are pulled metadata-only in stage 2c, before the grid starts. The order is
# kept uniform for readability/log-diffing parity with the rest of the ts38
# tree, not because a later arm at the same size depends on an earlier one.
#
# HARD RULES
#   (a) both non-fmt theta0's (base checkpoint, evt-ts38pp-parent) are
#       PULLED, never rebuilt or merged -- both are plain full-FT weights
#       already (no _merged/ stage anywhere in this launcher: all three
#       theta0's -- base, pp, fmt -- are full FT, never LoRA).
#   (b) evt-ts38mt-fmt-parent is DELIBERATELY UNGATED in the certification
#       sense -- no G1/G8 pass/fail is ever recorded against it, only
#       --no-record G5 evidence (the format-acquisition HALT gate, this
#       launcher's own bash comparison, not a gates.py record).
#   (c) PUSH AS YOU GO, both repos: every run is pushed to geode-store AND
#       geode-internals the moment it finishes (+scored, for targets), so a
#       box death mid-family loses at most one run's compute.
#   (d) every target run must reach stop_reason=converged; the parent has NO
#       pinned-epoch exception (unlike ts38pp's 4M-row parent) -- its
#       min_steps=167/max_steps=3340 ceiling must never bind either.
#   (e) NEVER destroy the box -- teardown is the operator's/owner's call,
#       taken separately once every artifact is receiver-verified on both
#       relays.
#
# SIZES OVERRIDE: set the SIZES env var (space-separated, strictly ascending)
# to run a subset:
#     SIZES="10000 46416" bash launch_ts38mt_family.sh --confirm-cost
set -euo pipefail

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
# Parsed as literal NAME=VALUE, not bash-sourced: some images' /etc/environment
# (e.g. NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=...) has unquoted >/</spaces that
# a raw `.`-source misparses as redirections, crashing on a bare numeric token.
if [[ -f /etc/environment ]]; then
  while IFS='=' read -r _env_name _env_value; do
    [[ "$_env_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    export "$_env_name=$_env_value"
  done < /etc/environment
fi
[[ -f /workspace/venv/bin/activate ]] && . /workspace/venv/bin/activate
python3 - <<'PY' || { echo "[ts38mt] FAILED: python3 lacks required modules (venv not active?)"; exit 1; }
import huggingface_hub, torch, geode  # noqa: F401
PY

cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38mt

echo "[ts38mt] estimated cost: 1 full-FT format parent (D_preteachfmt.parquet,"
echo "[ts38mt] 21,544 rows, min_steps=167/max_steps=3340 -- ts38pf's own"
echo "[ts38mt] format-only parent on the same data converged well inside this"
echo "[ts38mt] ceiling) ~= 4 min, plus 30 LoRA target runs (3 arms x 10 sizes)"
echo "[ts38mt] on one RTX 4090. Measured ts38(pf/pp)/ts38dense per-run wall"
echo "[ts38mt] clock across all 10 sizes sums to ~= 43 min/arm (5 SHIPPED"
echo "[ts38mt] sizes 1.7/2.1/6.4/10.5/23.6 min + 5 ts38dense-interpolated"
echo "[ts38mt] sizes 1.8/3.5/7.5/13/17 min) -- 3 arms x ~43 min ~= 2.15h, plus"
echo "[ts38mt] G5 (~0.5 min x 30) plus dual-repo pushes (31 runs x 2 repos,"
echo "[ts38mt] best-effort) plus data preflight (~5 min) => ~= 2.5-3h wall,"
echo "[ts38mt] ~= \$1 at \$0.35-0.45/h."
echo "[ts38mt] Storage (worst case, on mhieuuu/geode-internals ONLY --"
echo "[ts38mt] geode-store never carries snapshots): <=32 adapter snapshots x"
echo "[ts38mt] ~48MB x 30 target runs ~= 45GB, + <=14 full-state snapshots x"
echo "[ts38mt] ~155MB for the parent ~= 2.2GB => <=~48GB combined ceiling."
echo "[ts38mt] Ceilings (max_steps) must never bind -- stop_reason=max_steps"
echo "[ts38mt] is a bug signal, never a result, for every target AND for the"
echo "[ts38mt] parent (no pinned-epoch exception here, unlike ts38pp's 4M-row"
echo "[ts38mt] parent)."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38mt_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
INTERNALS_REPO=${INTERNALS_REPO:-mhieuuu/geode-internals}

PP_PARENT_RID=evt-ts38pp-parent
PP_PARENT_MODEL=$GEODE_STORE/runs/$PP_PARENT_RID/model   # full FT -> no _merged stage
PP_PARENT_FINAL_STEP=31093   # pinned one-epoch length, see ts38pp_parent.yaml header

PARENT_RID=evt-ts38mt-fmt-parent
PARENT_MODEL_DIR=$GEODE_STORE/runs/$PARENT_RID/model   # full FT -> no _merged stage
PARENT_CONFIG=../configs/ts38mt_fmt_parent.yaml
PARENT_MIN_FINAL_STEP=167   # = min_steps (one epoch: 21436 // 128), per decisions.md pin

BASE_CONFIG=../configs/ts38mt_base.yaml
PP_CONFIG=../configs/ts38mt_pp.yaml
FMT_CONFIG=../configs/ts38mt_fmt.yaml
OVERLAY_DIR=../configs/sweeps/ts38
# Bare-NL eval pin (D_algo_eval_bare), reused verbatim from the ts38 family —
# every target run's G5 (and the parent's HALT-gate probe) reads it, so
# every scored run is scored on identical data.
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
THETA0_JSON=$GEODE_STORE/results/ts38mt_family_theta0.json
# D_preteachfmt.parquet's order_hash — the SAME frozen 21,544-row pin
# launch_ts38pf_family.sh built and verified; ts38mt reuses that file
# byte-for-byte rather than regenerating it.
PREFMT_ORDER_HASH=5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1

DEFAULT_SIZES=(1000 2154 4642 10000 21544 46416 100000 146780 215443 316228)
if [[ -n ${SIZES:-} ]]; then
  read -r -a SIZES <<<"$SIZES"
else
  SIZES=("${DEFAULT_SIZES[@]}")
fi
# Ascending order is load-bearing (log/summary-table readability, and matches
# every sibling launcher's own SIZES contract); a descending/unsorted
# override would silently reorder the grid.
for i in "${!SIZES[@]}"; do
  ((i == 0)) && continue
  ((SIZES[i] > SIZES[i - 1])) || fail \
    "SIZES must be strictly ascending -- got '${SIZES[*]}' (index $i: ${SIZES[i]} <= ${SIZES[i - 1]})"
done
RUN_COUNT=$((3 * ${#SIZES[@]} + 1))

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE sizes=${#SIZES[@]} runs=$RUN_COUNT base=$BASE_RID parents=$PP_PARENT_RID,$PARENT_RID"
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

# Standing policy (feedback-run-until-convergence): a target run stops
# because eps/k said so, never because it ran out of steps. Checked for
# EVERY run at EVERY size, all three arms (ts38grid's rule).
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
# on identical data). Recorded for ALL THREE arms (unlike ts38dense's fs
# arm) — matches every prior ts38/ts38pp/ts38dense target run convention.
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

# --no-record G5 evidence for the HALT gate's theta0 probe, parsed the way
# ts38pf's score_g5_no_record parses gates.py g5's printed output. Full gate
# output goes to stderr so the $(...) capture is clean; stdout is "em0 em16
# loss" on success. Hardcoded to BARE_EVAL_CONFIG (the HALT gate never
# probes any other rendering) — mirrors ts38pf's single-arg version, not
# ts38pp's cfg-parameterized one.
score_g5_no_record() {
  local rid=$1 out em0 em16 loss
  out=$(python3 gates.py g5 --run "$rid" --config "$BARE_EVAL_CONFIG" --no-record 2>&1)
  grep -E "G5 (zero-shot|16-shot|shared-set)" <<<"$out" >&2 || true   # diagnostic only; a non-match must not abort this function
  em0=$(sed -n 's/.*G5 zero-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  em16=$(sed -n 's/.*G5 16-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  loss=$(sed -n 's/.*G5 shared-set test loss \([0-9.]*\) nats over n=.*/\1/p' <<<"$out" | head -1)
  if [[ -z $em0 || -z $em16 || -z $loss ]]; then
    echo "$out" | tail -5 >&2
    return 1
  fi
  echo "$em0 $em16 $loss"
}

# Best-effort push to mhieuuu/geode-internals WITH snapshots, public — the
# second repo/flag combination lib/launch_common.sh::push_run cannot express
# (no --with-snapshots knob, no --public knob). Written HERE rather than
# editing launch_common.sh (shared by every other launcher, out of this
# launcher's scope) — same best-effort contract as push_run: a failed
# upload warns and the family keeps going; the final receiver check is what
# can actually fail the run.
push_internals() {
  local rid=$1
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$INTERNALS_REPO" --with-snapshots --public; then
    milestone "push_complete run=$rid repo=$INTERNALS_REPO with_snapshots=1"
  else
    echo "[ts38mt] WARN hf_checkpoint.py push --with-snapshots failed for $rid -> $INTERNALS_REPO (best effort)"
    milestone "push_warn run=$rid repo=$INTERNALS_REPO"
  fi
}

# ---- stage 2a: the base checkpoint, receiver-verified (every prior ts38(*)
# family's stage 1, verbatim — every arm's theta0 traces back to it). -------
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38mt family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38mt] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38mt] MISSING on the hub: {missing}")
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
print(f"[ts38mt] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38mt] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38mt] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 2b: the ts38pp paper-protocol parent — plain full-FT weights,
# NO merge stage (hard rule (a)). --------------------------------------------
milestone "relay_verify_start run=$PP_PARENT_RID repo=$RELAY_REPO"
python3 - "$PP_PARENT_RID" "$RELAY_REPO" <<'PY' || fail "$PP_PARENT_RID is not on the relay — nothing to pull; the ts38mt pp arm has no parent"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38mt] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38mt] MISSING on the hub: {missing}")
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
) || fail "reading $PP_PARENT_RID manifest fields failed — inspect $GEODE_STORE/runs/$PP_PARENT_RID/manifest.json"
read -r PP_STATUS PP_METHOD PP_GATES PP_STOP PP_STEP <<<"$PP_FIELDS"
milestone "parent_fields run=$PP_PARENT_RID status=$PP_STATUS method=$PP_METHOD gates=$PP_GATES stop_reason=$PP_STOP final_step=$PP_STEP"

[[ $PP_STATUS == complete ]] || fail "$PP_PARENT_RID manifest status='$PP_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$PP_PARENT_RID/manifest.json"
[[ $PP_METHOD == full_ft ]] || fail "$PP_PARENT_RID training.method='$PP_METHOD' (expected full_ft — the paper-protocol parent is full FT, not LoRA)"
[[ $PP_GATES == none ]] || fail \
  "$PP_PARENT_RID has recorded gates {$PP_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
# The pinned one-epoch exception (ts38pp_parent.yaml header): min_steps ==
# max_steps == 31093, so a legitimate stop can read either 'converged' or
# 'max_steps' — but the step count itself is never negotiable.
[[ $PP_STOP == converged || $PP_STOP == max_steps ]] || fail \
  "$PP_PARENT_RID stop_reason='$PP_STOP' (expected converged or max_steps) — neither the
   pinned one-epoch ceiling nor eps/k produced this; inspect before proceeding"
[[ $PP_STEP == "$PP_PARENT_FINAL_STEP" ]] || fail \
  "$PP_PARENT_RID final_step=$PP_STEP (expected $PP_PARENT_FINAL_STEP, the pinned
   one-epoch length) — inspect the manifest; do not train the elicit arm against a
   wrong-length parent"
milestone "parent_verified run=$PP_PARENT_RID stop_reason=$PP_STOP final_step=$PP_STEP gates={} method=full_ft"

# ---- stage 2c: the 10 G7 anchor manifests, evt-ts38-base-n<N> — the
# ALREADY-SHIPPED ts38 family, metadata only (never trained here; the
# shipped 5 sizes are metadata-only on the relay anyway, and ts38dense's 5
# NEW sizes have full weights there too — --no-weights works uniformly). --
for n in "${SIZES[@]}"; do
  rid=evt-ts38-base-n${n}
  if [[ ! -f $GEODE_STORE/runs/$rid/manifest.json ]]; then
    milestone "pull_anchor_start run=$rid"
    python3 hf_checkpoint.py pull --run-id "$rid" --repo-id "$RELAY_REPO" --no-weights ||
      fail "pull $rid (G7 anchor) from $RELAY_REPO"
  fi
done

python3 - "${SIZES[@]}" <<'PY' || fail "G7 anchor preflight — see output above (a reused evt-ts38-base-n<size> run is missing, incomplete, or its data_order_hash/n_examples are unset; inspect before proceeding)"
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
ok = True
for n in sys.argv[1:]:
    rid = f"evt-ts38-base-n{n}"
    p = store / "runs" / rid / "manifest.json"
    if not p.is_file():
        print(f"[ts38mt] MISSING manifest for {rid}")
        ok = False
        continue
    d = json.loads(p.read_text())
    status = d.get("status")
    exp = d.get("experiment", {})
    n_examples = exp.get("n_examples")
    order_hash = exp.get("data_order_hash")
    good = status == "complete" and n_examples == int(n) and bool(order_hash)
    print(
        f"[ts38mt] {rid}: status={status} n_examples={n_examples} "
        f"order_hash={'set' if order_hash else 'MISSING'} -> {'OK' if good else 'BAD'}"
    )
    ok = ok and good
sys.exit(0 if ok else 1)
PY
milestone "g7_anchors_ready sizes=${#SIZES[@]}"

# ---- stage 3: data preflight — D_algo_bare / D_algo_eval_bare (as
# ts38dense stage 3) AND D_preteachfmt.parquet (copied exactly from
# launch_ts38pf_family.sh's stage 2 — same file, same pin, reused not
# regenerated). --------------------------------------------------------------
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

if [[ -f $DATA_DIR/D_preteachfmt.parquet ]]; then
  milestone "preteachfmt_datagen_skip (D_preteachfmt.parquet present)"
else
  milestone "preteachfmt_datagen_start (n=21544, operator-notation, permuted labels)"
  python3 ../datagen/make_preteach_format.py --out "$DATA_DIR" --n 21544 ||
    fail "make_preteach_format (D_preteachfmt derivation)"
  milestone "preteachfmt_datagen_complete"
fi

# The three target configs' data blocks are byte-identical by construction
# (same D_algo_bare.parquet order_hash as every ts38 family), so one
# order-hash pre-flight (on the base config) covers all three arms; the bare
# eval config and the format parent's config carry the other two.
python3 - "$BASE_CONFIG" "$BARE_EVAL_CONFIG" "$PARENT_CONFIG" "$PREFMT_ORDER_HASH" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval, parent_cfg_path, prefmt_hash = sys.argv[1:5]
for path in (Path(target_cfg_path), Path(bare_eval), Path(parent_cfg_path)):
    if not path.is_file():
        print(f"[ts38mt] MISSING config {path}")
        sys.exit(1)

for label, path in (
    ("D_algo_bare", Path(target_cfg_path)),
    ("D_algo_eval_bare", Path(bare_eval)),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38mt] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")

parent_cfg = load_config(Path(parent_cfg_path), None)
if parent_cfg["data"]["order_hash"] != prefmt_hash:
    print(
        f"[ts38mt] {parent_cfg_path}: data.order_hash "
        f"{parent_cfg['data']['order_hash']} != build-time pin {prefmt_hash} — "
        "the committed config and the file on disk disagree; regenerate or re-pin"
    )
    sys.exit(1)
df = load_frozen_parquet(parent_cfg)
print(f"[ts38mt] D_preteachfmt: {len(df)} rows, order_hash verified ({parent_cfg['data']['file']})")
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare/D_preteachfmt order hashes OK"

# ---- stage 4: overlay presence — the 4 base/parent configs plus exactly
# the 30 ts38mt_*_n*.yaml overlays this launch needs, checked explicitly per
# size/arm (no directory glob — a glob count can't tell you WHICH file is
# missing, only how many). ----------------------------------------------------
MISSING_FILES=()
for f in "$PARENT_CONFIG" "$BASE_CONFIG" "$PP_CONFIG" "$FMT_CONFIG" "$BARE_EVAL_CONFIG"; do
  [[ -f $f ]] || MISSING_FILES+=("$f")
done
for n in "${SIZES[@]}"; do
  for f in "$OVERLAY_DIR/ts38mt_base_n${n}.yaml" \
    "$OVERLAY_DIR/ts38mt_pp_n${n}.yaml" \
    "$OVERLAY_DIR/ts38mt_fmt_n${n}.yaml"; do
    [[ -f $f ]] || MISSING_FILES+=("$f")
  done
done
((${#MISSING_FILES[@]} == 0)) || fail \
  "missing config/overlay file(s) for this grid: ${MISSING_FILES[*]}
   — every size in SIZES needs all three overlays committed before launch, and the
   parent/base config files must exist"
milestone "configs_present targets=3 overlays=$((3 * ${#SIZES[@]}))"

# ---- stage 5: build evt-ts38mt-fmt-parent — full FT (train_sft.py), NOT
# LoRA, on D_preteachfmt.parquet. --------------------------------------------
train_or_skip "$PARENT_RID" \
  python3 train_sft.py --config "$PARENT_CONFIG" --init-from "$BASE_MODEL" --confirm-cost

PARENT_FIELDS=$(python3 - "$PARENT_RID" <<'PY'
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
) || fail "reading $PARENT_RID manifest fields failed — inspect $GEODE_STORE/runs/$PARENT_RID/manifest.json"
read -r P_STATUS P_METHOD P_GATES P_STOP P_STEP P_DATA_HASH <<<"$PARENT_FIELDS"
milestone "parent_fields status=$P_STATUS method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP data_order_hash=$P_DATA_HASH"

[[ $P_STATUS == complete ]] || fail "$PARENT_RID manifest status='$P_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$PARENT_RID/manifest.json"
[[ $P_METHOD == full_ft ]] || fail "$PARENT_RID training.method='$P_METHOD' (expected full_ft — the ts38mt format parent is full FT, matched to ts38pp's method, not LoRA)"
[[ $P_GATES == none ]] || fail \
  "$PARENT_RID has recorded gates {$P_GATES} — this family was designed against an
   ungated parent (scored --no-record only); inspect, do not proceed"
# NO pinned-epoch exception here (unlike ts38pp's 4M-row parent): this
# 21,544-row format-only install must reach eps/k convergence on its own —
# a max_steps stop is a bug signal, exactly like every target run.
[[ $P_STOP == converged ]] || fail \
  "CONVERGENCE CHECK: $PARENT_RID ended with stop_reason='$P_STOP', not 'converged'.
   Standing policy: a max_steps stop is a BUG SIGNAL, not a result — this parent has no
   pinned-epoch exception (unlike ts38pp's 4M-row parent). Inspect the loss trace before
   continuing."
[[ $P_STEP -ge $PARENT_MIN_FINAL_STEP ]] || fail \
  "$PARENT_RID final_step=$P_STEP (expected >= $PARENT_MIN_FINAL_STEP, one epoch) —
   inspect $PARENT_CONFIG's min_steps/max_steps pins and the manifest; do not hand a
   wrong-length parent to the target grid"
# Stale-parent guard: train_or_skip reuses ANY complete-status checkpoint at
# this run_id, including one built from a since-changed config/dataset on an
# earlier invocation. The manifest's own data_order_hash is the cheapest
# fingerprint tying the checkpoint back to the exact D_preteachfmt.parquet
# this launch run verified in stage 3 — a mismatch means "delete
# runs/$PARENT_RID and rebuild", not "keep training on top of it".
[[ $P_DATA_HASH == "$PREFMT_ORDER_HASH" ]] || fail \
  "$PARENT_RID manifest data_order_hash=$P_DATA_HASH != current pin $PREFMT_ORDER_HASH —
   this checkpoint was built from a different D_preteachfmt.parquet/config than the one
   this launch just verified. Remove $GEODE_STORE/runs/$PARENT_RID and rerun to rebuild
   against the current pin; do not train the target grid on a stale parent."
milestone "parent_verified stop_reason=converged final_step=$P_STEP gates={} method=full_ft data_order_hash=OK"

[[ -f $PARENT_MODEL_DIR/model.safetensors ]] ||
  fail "$PARENT_MODEL_DIR/model.safetensors is missing — a full-FT run's checkpoint dir
   should already hold plain weights (no merge stage for this family)"

# ---- stage 6: format-acquisition HALT gate — replicated verbatim (in
# logic) from launch_ts38pf_family.sh's stage 5. -----------------------------
if python3 - "$THETA0_JSON" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
d = json.loads(p.read_text())
sys.exit(0 if d.get("parent") and d.get("base") else 1)
PY
then
  milestone "theta0_skip json=$THETA0_JSON (already has parent+base)"
else
  PARENT_G5=$(score_g5_no_record "$PARENT_RID") || fail "$PARENT_RID G5 --no-record (theta0 evidence) failed (see stderr above)"
  read -r PARENT_EM0 PARENT_EM16 PARENT_LOSS <<<"$PARENT_G5"
  BASE_G5=$(score_g5_no_record "$BASE_RID") || fail "$BASE_RID G5 --no-record (theta0 evidence) failed (see stderr above)"
  read -r BASE_EM0 BASE_EM16 BASE_LOSS <<<"$BASE_G5"

  python3 - "$THETA0_JSON" "$PARENT_RID" "$PARENT_EM0" "$PARENT_EM16" "$PARENT_LOSS" \
    "$BASE_RID" "$BASE_EM0" "$BASE_EM16" "$BASE_LOSS" <<'PY' || fail "writing $THETA0_JSON failed"
import json
import sys
from pathlib import Path

out_path, p_rid, p_em0, p_em16, p_loss, b_rid, b_em0, b_em16, b_loss = sys.argv[1:10]
data = {
    "parent": {"run_id": p_rid, "em0": float(p_em0), "em16": float(p_em16), "loss": float(p_loss)},
    "base": {"run_id": b_rid, "em0": float(b_em0), "em16": float(b_em16), "loss": float(b_loss)},
    "eval_config": "eval_bare_target_data_ts38.yaml",
}
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
  milestone "theta0_latency parent_em0=$PARENT_EM0 parent_loss=$PARENT_LOSS base_em0=$BASE_EM0 base_loss=$BASE_LOSS"
fi

# Pre-registered format-acquisition read (decisions.md 2026-08-21 night):
# parent loss must be materially below base's, EM must stay ~0 (no algorithm
# leaked from the permuted-label training). Same margins as ts38pf: loss
# drop >= 10% of base's loss; EM threshold 5%.
FMT_VERDICT=$(python3 - "$THETA0_JSON" <<'PY'
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
p, b = d["parent"], d["base"]
loss_drop_frac = (b["loss"] - p["loss"]) / b["loss"]
if p["em0"] > 0.05:
    verdict = "LEAKED"
elif loss_drop_frac < 0.10:
    verdict = "NOT_LEARNED"
else:
    verdict = "LEARNED"
print(verdict, f"{loss_drop_frac:.4f}")
PY
) || fail "reading $THETA0_JSON for the format-acquisition verdict failed — inspect the file"
read -r FMT_STATUS FMT_LOSS_DROP <<<"$FMT_VERDICT"
milestone "format_acquisition_check verdict=$FMT_STATUS loss_drop_frac=$FMT_LOSS_DROP"

if [[ $FMT_STATUS == LEAKED ]]; then
  fail "FORMAT-ACQUISITION CHECK: parent zero-shot EM on the bare-NL eval is
   materially above 0 — the label permutation did not act as a control (algorithm may
   have leaked). See $THETA0_JSON. Do NOT proceed with the 10-size grid; investigate
   make_preteach_format.py / permute_labels before relaunching."
elif [[ $FMT_STATUS == NOT_LEARNED ]]; then
  fail "FORMAT-ACQUISITION CHECK: parent loss on the bare-NL eval is not materially
   below base's (drop=$FMT_LOSS_DROP, need >=0.10). The pinned LR 3e-5 (= ts38pp
   parent's own full-FT LR) may be too low for a 21,544-example full-FT format install to
   acquire the format inside this parent's convergence window. Fallback: re-run the
   parent at 1e-4 (NOT implemented here — requires a new overlay + owner go-ahead, out of
   this launcher's scope). See $THETA0_JSON. Do NOT proceed with the 10-size grid."
fi
milestone "format_acquired -> proceeding to the 10-size grid"

# ---- stage 7: push the parent, both repos. ---------------------------------
push_run "$PARENT_RID"
push_internals "$PARENT_RID"

# ---- stage 8: the grid — 3 arms x 10 sizes, base -> pp -> fmt at each size
# (uniform ts38-tree convention; no in-loop G7 dependency here — all 10
# anchors were already pulled in stage 2c). ----------------------------------
for n in "${SIZES[@]}"; do
  milestone "size_start n=$n"

  BASE_ARM_RID=evt-ts38mt-base-n${n}
  train_or_skip "$BASE_ARM_RID" \
    python3 train_target.py --config "$BASE_CONFIG" \
      --override "$OVERLAY_DIR/ts38mt_base_n${n}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost
  require_converged "$BASE_ARM_RID" "$n"
  record_g5 "$BASE_ARM_RID"
  push_run "$BASE_ARM_RID"
  push_internals "$BASE_ARM_RID"

  PP_ARM_RID=evt-ts38mt-pp-n${n}
  train_or_skip "$PP_ARM_RID" \
    python3 train_target.py --config "$PP_CONFIG" \
      --override "$OVERLAY_DIR/ts38mt_pp_n${n}.yaml" \
      --init-from "$PP_PARENT_MODEL" --confirm-cost
  require_converged "$PP_ARM_RID" "$n"
  record_g5 "$PP_ARM_RID"
  push_run "$PP_ARM_RID"
  push_internals "$PP_ARM_RID"

  FMT_ARM_RID=evt-ts38mt-fmt-n${n}
  train_or_skip "$FMT_ARM_RID" \
    python3 train_target.py --config "$FMT_CONFIG" \
      --override "$OVERLAY_DIR/ts38mt_fmt_n${n}.yaml" \
      --init-from "$PARENT_MODEL_DIR" --confirm-cost
  require_converged "$FMT_ARM_RID" "$n"
  record_g5 "$FMT_ARM_RID"
  push_run "$FMT_ARM_RID"
  push_internals "$FMT_ARM_RID"

  milestone "size_complete n=$n arms=3"
done

# ---- stage 9: summary table, all 31 runs (parent + 30 targets). -----------
python3 - "$PARENT_RID" "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
parent_rid, sizes = sys.argv[1], sys.argv[2:]
print(f"[ts38mt] {'run_id':<32}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")

pp = store / "runs" / parent_rid / "manifest.json"
pr = json.loads(pp.read_text()).get("experiment", {}).get("sft_result", {}) if pp.is_file() else {}
pr = pr or {}
print(
    f"[ts38mt] {parent_rid:<32}{pr.get('final_step', 'MISSING'):>12}"
    f"{pr.get('stop_reason', 'MISSING'):>14}{pr.get('min_val_nats', 'MISSING'):>16}{'n/a':>26}"
)

for n in sizes:
    for rid in (
        f"evt-ts38mt-base-n{n}",
        f"evt-ts38mt-pp-n{n}",
        f"evt-ts38mt-fmt-n{n}",
    ):
        p = store / "runs" / rid / "manifest.json"
        r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}
        r = r or {}
        edl = r.get("edl_per_label_token_nats")
        print(
            f"[ts38mt] {rid:<32}{r.get('final_step', 'MISSING'):>12}"
            f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
            f"{edl if edl is not None else 'n/a':>26}"
        )
PY
milestone "family_complete runs=$RUN_COUNT"

# ---- stage 10: receiver-verify every run on BOTH repos ---------------------
# Verify the RECEIVER, not the sender ([[feedback-verify-the-receiver-not-the-
# sender]]): push bookkeeping looks identical over an empty relay. Each run
# was already pushed as it finished (hard rule (c)); this re-pushes only
# whatever a repo is actually missing, once per repo, then re-checks.
# geode-store: manifest + weights (adapter for targets, plain
# model.safetensors for the parent) + measurement logs, snapshots excluded
# by design (Part A). geode-internals: manifest + weights + at least one
# snapshot directory (targets: snapshots/step_*; parent: sft_snapshots/
# step_*) — the whole point of that repo is the snapshots.
run_receiver_check() {
  local repo=$1 check_internals=$2
  python3 - "$repo" "$check_internals" "$PARENT_RID" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo, check_internals = sys.argv[1], sys.argv[2] == "1"
parent_rid = sys.argv[3]
sizes = sys.argv[4:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True


def has_prefix(prefix: str) -> bool:
    return any(f.startswith(prefix) for f in files)


required = [f"runs/{parent_rid}/manifest.json", f"runs/{parent_rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
if check_internals and not has_prefix(f"runs/{parent_rid}/sft_snapshots/step_"):
    missing.append(f"runs/{parent_rid}/sft_snapshots/step_*")
status = "OK" if not missing else f"MISSING {missing}"
print(f"  {parent_rid}: {status}")
if missing:
    print(f"MISSING {parent_rid}")
    ok = False

for n in sizes:
    for rid in (
        f"evt-ts38mt-base-n{n}",
        f"evt-ts38mt-pp-n{n}",
        f"evt-ts38mt-fmt-n{n}",
    ):
        if check_internals:
            required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/adapter.safetensors"]
        else:
            required = [
                f"runs/{rid}/manifest.json",
                f"runs/{rid}/model/adapter.safetensors",
                f"runs/{rid}/eval_log.jsonl",
                f"runs/{rid}/logs/prequential.jsonl",
            ]
        missing = [r for r in required if r not in files]
        if check_internals and not has_prefix(f"runs/{rid}/snapshots/step_"):
            missing.append(f"runs/{rid}/snapshots/step_*")
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {rid}: {status}")
        if missing:
            print(f"MISSING {rid}")
            ok = False
sys.exit(0 if ok else 1)
PY
}


# NOTE on the status-capture idiom below: a bare `VAR=$(cmd); STATUS=$?` is
# NOT safe under `set -e` (this launcher, unlike its siblings, runs with
# -e) — `run_receiver_check` returning 1 is the ORDINARY "some runs missing,
# retry" case this block exists to handle, and a bare failed assignment as
# its own simple command triggers errexit immediately, skipping the
# `STATUS=$?` line entirely and killing the script with no `fail` message.
# `VAR=$(cmd) && STATUS=0 || STATUS=$?` keeps the assignment inside an
# `&&`/`||` list, which -e exempts, so the retry path is actually reached.
STORE_OUT=$(run_receiver_check "$RELAY_REPO" 0) && STORE_STATUS=0 || STORE_STATUS=$?
echo "[ts38mt] receiver check ($RELAY_REPO, hub files for all $RUN_COUNT runs):"
echo "$STORE_OUT"
if [[ $STORE_STATUS -ne 0 ]]; then
  milestone "receiver_retry repo=$RELAY_REPO (re-pushing whatever the hub is missing)"
  while read -r rid; do
    push_run "$rid"
  done < <(sed -n 's/^MISSING //p' <<<"$STORE_OUT")
  STORE_OUT=$(run_receiver_check "$RELAY_REPO" 0) && STORE_STATUS=0 || STORE_STATUS=$?
  echo "[ts38mt] receiver check ($RELAY_REPO, after one push retry):"
  echo "$STORE_OUT"
fi
[[ $STORE_STATUS -eq 0 ]] || fail "push receiver check FAILED on $RELAY_REPO — see output above (at least one run's manifest/eval_log/prequential log is still missing after a retry)"
milestone "receiver_verified repo=$RELAY_REPO runs=$RUN_COUNT"

INTERNALS_OUT=$(run_receiver_check "$INTERNALS_REPO" 1) && INTERNALS_STATUS=0 || INTERNALS_STATUS=$?
echo "[ts38mt] receiver check ($INTERNALS_REPO, hub files for all $RUN_COUNT runs):"
echo "$INTERNALS_OUT"
if [[ $INTERNALS_STATUS -ne 0 ]]; then
  milestone "receiver_retry repo=$INTERNALS_REPO (re-pushing whatever the hub is missing)"
  while read -r rid; do
    push_internals "$rid"
  done < <(sed -n 's/^MISSING //p' <<<"$INTERNALS_OUT")
  INTERNALS_OUT=$(run_receiver_check "$INTERNALS_REPO" 1) && INTERNALS_STATUS=0 || INTERNALS_STATUS=$?
  echo "[ts38mt] receiver check ($INTERNALS_REPO, after one push retry):"
  echo "$INTERNALS_OUT"
fi
[[ $INTERNALS_STATUS -eq 0 ]] || fail "push receiver check FAILED on $INTERNALS_REPO — see output above (at least one run's weights/snapshot is still missing after a retry)"
milestone "receiver_verified repo=$INTERNALS_REPO runs=$RUN_COUNT"

# ---- stage 11: finish -------------------------------------------------------
echo "[ts38mt] MILESTONE analysis_commands"
echo "[ts38mt]   CPU-only, run off \$GEODE_STORE:"
echo "[ts38mt]     python3 ../analysis/edl_converged_val_floor.py --family ts38mt"
echo "[ts38mt]     python3 ../analysis/dataset_size_sweep.py --family ts38mt"
echo "[ts38mt]   (neither analysis script has a ts38mt FAMILIES entry yet — same"
echo "[ts38mt]   caveat every new ts38(*) family launcher ships with; out of this"
echo "[ts38mt]   launcher's scope to add one.)"
echo "[ts38mt]   theta0 evidence (format-acquisition check, parent vs. base) at"
echo "[ts38mt]   $THETA0_JSON — not a run, not pushed."
echo "[ts38mt]   Reproducibility-check tolerance (decisions.md 2026-08-21 night):"
echo "[ts38mt]   the 20 base/pp re-run cells' EDL/D at the OCV floor should land"
echo "[ts38mt]   within <=5% relative of their already-shipped evt-ts38-base-n<N>/"
echo "[ts38mt]   evt-ts38pp-pretaught-n<N> values — an ANALYSIS-time check, not"
echo "[ts38mt]   enforced here."
echo "[ts38mt]   Ten mechanistic tests (residual-stream probe, logit-lens depth,"
echo "[ts38mt]   grad-norm/weight-delta from snapshots, effective rank, ...) read"
echo "[ts38mt]   off these snapshots on mhieuuu/geode-internals — building/running"
echo "[ts38mt]   those tests is OUT OF THIS LAUNCHER'S SCOPE (geode/probe, planned)."
echo "[ts38mt]   The box is NOT torn down here (hard rule (e)); teardown is the"
echo "[ts38mt]   operator's call once every artifact is verified on both relays."
notify "ts38mt family done: $RUN_COUNT runs (1 full-FT format parent + 3 arms x ${#SIZES[@]} sizes), format_acquisition=$FMT_STATUS"
echo "[ts38mt] TERMINAL_SUCCESS runs=$RUN_COUNT"
