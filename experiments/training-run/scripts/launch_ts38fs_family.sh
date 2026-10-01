#!/usr/bin/env bash
# ts38fs family — format-install DOSE SWEEP on top of ts38pf (EXPERIMENTS TBD;
# ts38pf: §6.16, decisions.md 2026-08-15 "ts38pf pre-registration"). ts38pf
# measured ONE install size (i=21544 rows of D_preteachfmt) at ONE seed; this
# family extends it to a full 3-axis grid — install size i in
# {1000, 4642, 21544, 100000}, target size n in
# {1000, 4642, 21544, 100000, 316228}, seed s in {316, 1316, 2316} — asking
# whether the format-acquisition effect (and, downstream, any curve-shape
# change it causes) is dose-dependent in the install size, and how much of
# the target-stage curve is seed noise.
#
# THE GRID: 60 (i, n, s) cells. REUSE, do not rebuild: the i=21544 parent
# already exists (evt-ts38pf-preteachfmt-parent, launch_ts38pf_family.sh) and
# its s=316 target row already exists run-for-run as the 5
# evt-ts38pf-preteachfmt-n<n> runs — those 5 (i=21544, s=316) cells are
# SKIPPED here entirely, never retrained. So THIS launcher builds:
#   evt-ts38fs-parent-n{1000,4642,100000}   3 NEW format-only LoRA parents
#                                            (ts38fs_parent_n<i>.yaml),
#                                            IDENTICAL recipe to
#                                            ts38_preteachfmt_parent.yaml,
#                                            different install size.
#   evt-ts38fs-i{I}-n{N}-s{S}               55 NEW target runs (60 - 5
#                                            reused), theta0 = the matching
#                                            install parent's MERGED weights,
#                                            data = D_algo_bare (bare-NL
#                                            add/sub), seed = S.
# for I in 1000 4642 21544 100000; N in 1000 4642 21544 100000 316228;
# S in 316 1316 2316 (skipping the 5 (I=21544, S=316) cells).
#
# ORDER: env/store guards -> datagen (new preteachfmt sizes only) -> overlay
# generation -> parent builds+merges -> theta0 checks -> target sweep -> push.
# OVERLAY GENERATION IS DELIBERATELY MOVED BEFORE ANY TRAINING (a departure
# from a literal top-to-bottom reading of this family's own build brief,
# which lists it right before the target loop): launch_ts38grid_family.sh's
# own stage-2 comment names the reason directly — "a missing overlay after
# 20 runs is the expensive failure mode". Generating + counting all 55
# overlays up front means a generator bug surfaces before the first parent
# trains, not after three parent builds and some number of target runs.
#
# HARD RULES
#   (a) all four install parents are DELIBERATELY UNGATED (no G1/G8
#       certification — each is a format-only control, not a capability
#       parent). This launcher NEVER runs gates.py against any of them
#       except --no-record (theta0 stage, evidence only, no bar for the
#       reused parent's evidence which is re-derived fresh here too) and
#       never records anything to a parent manifest except
#       merge_adapter.py's own experiment.merged_checkpoint entry.
#   (b) LoRA checkpoints load ONLY via geode.zoo.load_model
#       ([[feedback-lora-checkpoints-load-via-zoo-load-model]]) — every
#       target run warm-starts from its install parent's MERGED plain
#       weights (runs/<id>/model_merged/), never from_pretrained on the
#       wrapped runs/<id>/model/.
#   (c) ORDER_HASH SENTINEL GUARD: the three new parent configs ship with
#       REAL, measured data.order_hash pins (decisions.md 2026-08-20 "ts38fs
#       pre-registration" records the same three values). Stage 2 below
#       still HARD-FAILS the whole family if any of them ever carries the
#       literal placeholder PIN_AFTER_DATAGEN_n<i> instead — belt-and-braces
#       against a future edit reintroducing an unpinned sentinel, not an
#       expected trigger today. This launcher must never be runnable into
#       GPU spend against an unpinned hash.
#   (d) NEW SIZES/SEEDS ONLY, same discipline as launch_ts38grid_family.sh
#       hard rule (c): train_or_skip keys on the LOCAL store, so a run
#       living only on the relay looks "missing" and would be retrained.
#       This launcher does not add ts38grid's SIZE GUARD preflight (its own
#       55 run ids are new to every family, so no existing shipped run can
#       collide) but inherits the same push-as-you-go discipline (hard rule
#       (e)) so a box death loses at most one run's compute.
#   (e) PUSH AS YOU GO — every target run is pushed the moment it trains.
#   (f) NEVER destroy the box — operator/owner call, not this launcher's.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38fs

echo "[ts38fs] estimated cost: THREE new LoRA format-only parent builds"
echo "[ts38fs] (i=1000/4642/100000; low-capacity permuted-label objective,"
echo "[ts38fs] min_steps 7/36/777, ceilings never bind -- these sizes were"
echo "[ts38fs] never separately timed, but the i=21544 parent (same recipe,"
echo "[ts38fs] 167 steps/epoch) converged well inside its 3340-step ceiling,"
echo "[ts38fs] so expect low tens of minutes total across the three, not"
echo "[ts38fs] hours) PLUS 55 LoRA target runs on one RTX 4090. Per-run"
echo "[ts38fs] training time is the ts38pf family's own measured per-n"
echo "[ts38fs] numbers (theta0/seed do not change step cost): 1.7 / 2.1 /"
echo "[ts38fs] 6.4 / 10.5 / 23.6 min at n = 1000 / 4642 / 21544 / 100000 /"
echo "[ts38fs] 316228, x11 non-reused cells per n (4 installs x 3 seeds,"
echo "[ts38fs] minus 1 reused) => approx (11*1.7 + 11*2.1 + 11*6.4 + 11*10.5"
echo "[ts38fs] + 11*23.6) = ~487 min = ~8.1h target-run compute, plus G5"
echo "[ts38fs] (~0.5 min x 55) and setup/datagen/merges => roughly 9h wall,"
echo "[ts38fs] ~\$3.15-4.05 at \$0.35-0.45/h; disk a few GB. This REFINES"
echo "[ts38fs] EXPERIMENTS.md section 20's pre-registered ~\$15-30 ballpark"
echo "[ts38fs] (explicitly flagged there as 'to be refined from measured"
echo "[ts38fs] per-run time at launch') down using ts38pf/ts38grid's own"
echo "[ts38fs] measured per-n numbers -- the ballpark was conservative, not"
echo "[ts38fs] wrong; get owner sign-off on the LOWER number before running."
echo "[ts38fs] Ceilings (max_steps) must never bind -- stop_reason=max_steps"
echo "[ts38fs] is a bug signal, never a result."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38fs_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}

PF_PARENT_RID=evt-ts38pf-preteachfmt-parent
# The pf parent's build-time data_order_hash pin, verbatim from
# launch_ts38pf_family.sh/launch_ts38grid_family.sh — the stale-parent guard
# below. This is the ONLY hardcoded order_hash in this launcher: the three
# NEW parents' expected hashes are read from their own committed configs at
# run time instead (both configs and this launcher were written after their
# real pins were already measured, but reading them at run time rather than
# hardcoding a second copy here means the config stays the single source of
# truth — no risk of the two silently drifting apart on a future re-pin).
PREFMT_ORDER_HASH=5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1

NEW_INSTALLS=(1000 4642 100000)
INSTALLS=(1000 4642 21544 100000)
SIZES=(1000 4642 21544 100000 316228)
SEEDS=(316 1316 2316)

TARGET_CONFIG=../configs/ts38fs_target.yaml
OVERLAY_DIR=../configs/sweeps/ts38
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
FMT_JSON=$GEODE_STORE/results/ts38fs_format_acquisition.json
# The first cell trained under this launcher's own S-outer/I-asc/N-asc order
# (S=316 skips I=21544, so the very first is I=1000/N=1000/S=316) — pf's own
# convergence discipline (its PIN_CHECK_N=1000), applied to this family's
# own first cell rather than every cell (grid's heavier per-run bar was NOT
# adopted here — see stage 6 below).
PIN_CHECK_RID=evt-ts38fs-i1000-n1000-s316

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE installs=${#INSTALLS[@]} sizes=${#SIZES[@]} seeds=${#SEEDS[@]} base=$BASE_RID pf_parent=$PF_PARENT_RID"

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

# --no-record G5 evidence, parsed the way ts38pf's score_g5_no_record parses
# gates.py g5's printed output. Full gate output goes to stderr so the
# $(...) capture is clean; stdout is "em0 em16 loss" on success.
score_g5_no_record() {
  local rid=$1 out em0 em16 loss
  out=$(python3 gates.py g5 --run "$rid" --config "$BARE_EVAL_CONFIG" --no-record 2>&1)
  grep -E "G5 (zero-shot|16-shot|shared-set)" <<<"$out" >&2
  em0=$(sed -n 's/.*G5 zero-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  em16=$(sed -n 's/.*G5 16-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  loss=$(sed -n 's/.*G5 shared-set test loss \([0-9.]*\) nats over n=.*/\1/p' <<<"$out" | head -1)
  if [[ -z $em0 || -z $em16 || -z $loss ]]; then
    echo "$out" | tail -5 >&2
    return 1
  fi
  echo "$em0 $em16 $loss"
}

# Merge a LoRA parent to plain weights and prove the merge preserved the
# function (launch_ts38grid_family.sh's merge_and_verify, parameterised the
# same way). Sets the global MERGED_DIR; callers copy it immediately.
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

print(f"[ts38fs] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
sys.exit(0 if max_abs_diff < 1e-3 else 1)
PY
)
  receiver_status=$?
  echo "$receiver_out"
  [[ $receiver_status -eq 0 ]] || fail \
    "receiver check FAILED for $rid — the merged checkpoint does not reproduce the
   wrapped model's logits within 1e-3 (see output above); do NOT hand this off to the
   target runs"
  max_diff=$(sed -n 's/.*max_abs_logit_diff=\(.*\)/\1/p' <<<"$receiver_out")
  milestone "parent_merged run=$rid max_abs_logit_diff=$max_diff"
  MERGED_DIR=$merged
}

# ---- stage 1: the base checkpoint, receiver-verified (ts38pf's stage 1,
# verbatim) -------------------------------------------------------------------
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38fs family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38fs] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38fs] MISSING on the hub: {missing}")
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
print(f"[ts38fs] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38fs] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38fs] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 1b: the REUSED ts38pf parent, receiver-verified the same way —
# it is never retrained here, but this family depends on it as much as the
# base, so it gets the same pull+load discipline. ---------------------------
#
# OPEN RISK (report, do not paper over): launch_ts38pf_family.sh's only push
# of this run id (its stage 7) uses `hf_checkpoint.py push --metadata-only`,
# which excludes ALL *.safetensors — including the LoRA adapter sidecar
# (hf_checkpoint.py's push() docstring is explicit: "these run weights are
# not recoverable from the relay"). launch_ts38grid_family.sh assumes
# adapter.safetensors IS on the relay for this exact run id, but that
# launcher was built and never launched (never actually exercised this
# pull). If the adapter genuinely never reached the relay by any other path,
# the block below fails loudly at the missing-file check and the i=21544
# column of this family (15 cells) cannot proceed without RETRAINING the
# parent — which this launcher deliberately does not do (retraining would
# silently create a second, different theta0 for cells the ts38pf family
# already measured). Investigate the relay contents before relaunching if
# this fails; do not add a retrain fallback here.
milestone "relay_verify_start run=$PF_PARENT_RID repo=$RELAY_REPO"
python3 - "$PF_PARENT_RID" "$RELAY_REPO" <<'PY' || fail "$PF_PARENT_RID is not on the relay at all — nothing to pull; the reused i=21544 install parent must exist before this family can train against it"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/adapter.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38fs] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(
        f"[ts38fs] MISSING on the hub: {missing} — launch_ts38pf_family.sh's only push of "
        f"this run id was --metadata-only (excludes ALL *.safetensors, adapter sidecars "
        "included); the adapter may never have reached the relay by any other path. See "
        "the OPEN RISK comment above this block. Do NOT retrain this parent to work around "
        "it — that would silently create a second theta0 for cells ts38pf already measured."
    )
    sys.exit(1)
PY

if [[ ! -f $GEODE_STORE/runs/$PF_PARENT_RID/model/adapter.safetensors ]]; then
  milestone "pull_parent_start run=$PF_PARENT_RID"
  python3 hf_checkpoint.py pull --run-id "$PF_PARENT_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $PF_PARENT_RID from $RELAY_REPO"
fi
[[ -f $GEODE_STORE/runs/$PF_PARENT_RID/model/adapter.safetensors ]] ||
  fail "pull left no $GEODE_STORE/runs/$PF_PARENT_RID/model/adapter.safetensors — the parent's LoRA adapter must be on the relay (see the OPEN RISK comment above)"
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
[[ $PF_DATA_HASH == "$PREFMT_ORDER_HASH" ]] || fail \
  "$PF_PARENT_RID manifest data_order_hash=$PF_DATA_HASH != pin $PREFMT_ORDER_HASH —
   the checkpoint on the relay was built from a different D_preteachfmt.parquet/config
   than the one the shipped ts38pf sizes used; inspect, do not proceed."
milestone "parent_verified run=$PF_PARENT_RID stop_reason=converged final_step=$PF_STEP gates={} method=lora data_order_hash=OK"

merge_and_verify "$PF_PARENT_RID"
PF_MERGED=$MERGED_DIR

# ---- stage 2: data artifacts — base/bare (needed by every target run and
# by make_preteach_format.py's own D_algo source) + the THREE NEW
# preteachfmt derivations (NOT the unsuffixed D_preteachfmt.parquet — the
# i=21544 parent is pulled, not rebuilt, so this launcher never needs it) --
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

for n in "${NEW_INSTALLS[@]}"; do
  if [[ -f $DATA_DIR/D_preteachfmt_n${n}.parquet ]]; then
    milestone "preteachfmt_datagen_skip n=$n (D_preteachfmt_n${n}.parquet present)"
  else
    milestone "preteachfmt_datagen_start n=$n (operator-notation, permuted labels)"
    python3 ../datagen/make_preteach_format.py --out "$DATA_DIR" --n "$n" ||
      fail "make_preteach_format --n $n (D_preteachfmt_n$n derivation)"
    milestone "preteachfmt_datagen_complete n=$n"
  fi
done

# HARD RULE (c): refuse to proceed if any new parent config still carries
# the PIN_AFTER_DATAGEN sentinel (belt-and-braces — today's three configs
# ship with real, measured pins, so this branch is not expected to fire),
# and (once past that) verify each new parent's on-disk parquet actually
# hashes to what its own committed config claims — the same two-part
# discipline ts38pf's stage 2 applies to the unsuffixed D_preteachfmt
# .parquet, just against a per-size pin read from each config instead of
# one hardcoded literal (see the PREFMT_ORDER_HASH comment above).
python3 - "$TARGET_CONFIG" "$BARE_EVAL_CONFIG" "${NEW_INSTALLS[@]}" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval, *installs = sys.argv[1:]
for path in (Path(target_cfg_path), Path(bare_eval)):
    if not path.is_file():
        print(f"[ts38fs] MISSING config {path}")
        sys.exit(1)

for label, path in (
    ("D_algo_bare", Path(target_cfg_path)),
    ("D_algo_eval_bare", Path(bare_eval)),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38fs] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")

ok = True
for n in installs:
    parent_cfg_path = Path(f"../configs/ts38fs_parent_n{n}.yaml")
    if not parent_cfg_path.is_file():
        print(f"[ts38fs] MISSING config {parent_cfg_path}")
        ok = False
        continue
    parent_cfg = load_config(parent_cfg_path, None)
    order_hash = parent_cfg["data"]["order_hash"]
    if order_hash.startswith("PIN_AFTER_DATAGEN"):
        print(
            f"[ts38fs] {parent_cfg_path}: data.order_hash is still the sentinel "
            f"'{order_hash}' — a sibling worker must generate D_preteachfmt_n{n}.parquet "
            "and pin its real order_hash into this config before this family can train "
            "against it. Refusing to proceed."
        )
        ok = False
        continue
    df = load_frozen_parquet(parent_cfg)
    print(f"[ts38fs] D_preteachfmt_n{n}: {len(df)} rows, order_hash verified ({parent_cfg['data']['file']})")
sys.exit(0 if ok else 1)
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare/D_preteachfmt_n{1000,4642,100000} order hashes OK, no sentinels remain"

# ---- stage 3: overlay generation — ALL 55 cells, before any training spend
# (see the header note on why this is reordered ahead of the parent builds).
# UNLIKE every other ts38(*) family, these 55 files are generated build
# artifacts, not committed configs (see generate_ts38fs_overlays.py's own
# docstring for why) — expect `git status` on a box to show 55 untracked
# files under $OVERLAY_DIR after this stage. If a future decision commits
# them instead (matching every other family's convention), running this
# generator once and `git add`-ing the result is the whole migration; that
# choice was left open here rather than made silently.
python3 generate_ts38fs_overlays.py --out "$OVERLAY_DIR" || fail "generate_ts38fs_overlays.py"
OVERLAY_COUNT=$(find "$OVERLAY_DIR" -maxdepth 1 -name 'ts38fs_i*_n*_s*.yaml' | wc -l | tr -d ' ')
[[ $OVERLAY_COUNT -eq 55 ]] || fail \
  "generate_ts38fs_overlays.py wrote $OVERLAY_COUNT overlay(s) under $OVERLAY_DIR, expected
   exactly 55 (60 grid cells - 5 reused (i=21544, s=316)); inspect before training anything"
milestone "overlays_generated count=$OVERLAY_COUNT dir=$OVERLAY_DIR"

# ---- stage 4: build the three NEW format-only parents + merge (reused
# i=21544 parent already merged in stage 1b) ---------------------------------
declare -A MERGED_BY_INSTALL
MERGED_BY_INSTALL[21544]=$PF_MERGED
for i in "${NEW_INSTALLS[@]}"; do
  rid=evt-ts38fs-parent-n${i}
  parent_config=../configs/ts38fs_parent_n${i}.yaml
  train_or_skip "$rid" \
    python3 train_sft.py --config "$parent_config" --init-from "$BASE_MODEL" --confirm-cost

  FIELDS=$(python3 - "$rid" <<'PY'
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
  read -r P_STATUS P_METHOD P_GATES P_STOP P_STEP P_DATA_HASH <<<"$FIELDS"
  milestone "parent_fields run=$rid status=$P_STATUS method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP data_order_hash=$P_DATA_HASH"

  [[ $P_STATUS == complete ]] || fail "$rid manifest status='$P_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$rid/manifest.json"
  [[ $P_METHOD == lora ]] || fail "$rid training.method='$P_METHOD' (expected lora)"
  [[ $P_GATES == none ]] || fail \
    "$rid has recorded gates {$P_GATES} — this family was designed against an ungated
     parent; inspect, do not proceed"
  [[ $P_STOP == converged ]] || fail \
    "$rid stop_reason='$P_STOP' (expected converged) — a max_steps stop on a format-only
     parent is a bug signal (permuted labels should plateau well inside the >=20-epoch
     ceiling); do not proceed with an unconverged parent"

  # Stale-parent guard, pf's own mechanism (see PREFMT_ORDER_HASH comment
  # above): the EXPECTED hash for a NEW parent is read from its own
  # committed config (already sentinel-checked and disk-verified in stage
  # 2), never a hardcoded literal.
  EXPECTED_HASH=$(python3 - "$i" <<'PY'
import sys
from pathlib import Path

from train import load_config

n = sys.argv[1]
cfg = load_config(Path(f"../configs/ts38fs_parent_n{n}.yaml"), None)
print(cfg["data"]["order_hash"])
PY
)
  [[ $P_DATA_HASH == "$EXPECTED_HASH" ]] || fail \
    "$rid manifest data_order_hash=$P_DATA_HASH != current pin $EXPECTED_HASH — this
     checkpoint was built from a different D_preteachfmt_n${i}.parquet/config than the one
     this launch just verified. Remove $GEODE_STORE/runs/$rid and rerun to rebuild against
     the current pin; do not train the target sweep on a stale parent."
  milestone "parent_verified run=$rid stop_reason=converged final_step=$P_STEP gates={} method=lora data_order_hash=OK"

  merge_and_verify "$rid"
  MERGED_BY_INSTALL[$i]=$MERGED_DIR
  # LOAD-BEARING full push (adapter.safetensors included) — this is what
  # lets a LATER family reuse this parent the way THIS launcher reuses
  # evt-ts38pf-preteachfmt-parent (stage 1b). The stage 7 push of this same
  # run id near the end of this script is `--metadata-only` (manifest/logs
  # only, no weights) — THAT one is the redundant-looking one; do not
  # "simplify" by dropping this one, or a future ts38fs-extension launcher
  # hits exactly the OPEN RISK this launcher's own stage 1b flags for the
  # pf parent (adapter never reached the relay).
  push_run "$rid"
done

# ---- stage 5: format-acquisition theta0 check, ALL FOUR install parents —
# same LEAKED/NOT_LEARNED/LEARNED verdict block as launch_ts38pf_family.sh,
# with ONE deliberate semantic change from that launcher: LEAKED still
# hard-fails the WHOLE family (a broken label permutation invalidates every
# cell, at every install size), but NOT_LEARNED now logs the verdict and
# CONTINUES rather than halting. Reason: at the smallest install sizes
# (i=1000/4642), failing to acquire the format from so few permuted-label
# examples IS itself part of the dose-response measurement this family
# exists to produce, not evidence of a broken launch — ts38pf's original
# hard-fail-on-NOT_LEARNED bar was calibrated for a single i=21544 install
# where "the lesson didn't transfer" would have meant something had gone
# wrong; here it may just mean the dose was too small, which is the
# question. All four verdicts + loss_drop_frac land in one JSON artifact
# ($FMT_JSON) regardless of verdict, so the dose-response curve
# (format-acquisition vs install size) can be read afterward even where a
# cell logged NOT_LEARNED.
BASE_G5=$(score_g5_no_record "$BASE_RID") || fail "$BASE_RID G5 --no-record (theta0 evidence) failed (see stderr above)"
read -r BASE_EM0 BASE_EM16 BASE_LOSS <<<"$BASE_G5"
milestone "theta0_base em0=$BASE_EM0 em16=$BASE_EM16 loss=$BASE_LOSS"

declare -A FMT_VERDICT_BY_INSTALL
for i in "${INSTALLS[@]}"; do
  parent_rid=evt-ts38fs-parent-n${i}
  [[ $i == 21544 ]] && parent_rid=$PF_PARENT_RID

  PARENT_G5=$(score_g5_no_record "$parent_rid") || fail "$parent_rid G5 --no-record (theta0 evidence) failed (see stderr above)"
  read -r PARENT_EM0 PARENT_EM16 PARENT_LOSS <<<"$PARENT_G5"

  VERDICT_LINE=$(python3 - "$parent_rid" "$PARENT_EM0" "$PARENT_EM16" "$PARENT_LOSS" "$BASE_LOSS" <<'PY'
import sys

rid, em0, em16, loss, base_loss = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
loss_drop_frac = (base_loss - loss) / base_loss
if em0 > 0.05:
    verdict = "LEAKED"
elif loss_drop_frac < 0.10:
    verdict = "NOT_LEARNED"
else:
    verdict = "LEARNED"
print(verdict, f"{loss_drop_frac:.4f}")
PY
)
  read -r VERDICT LOSS_DROP <<<"$VERDICT_LINE"
  FMT_VERDICT_BY_INSTALL[$i]=$VERDICT
  milestone "format_acquisition_check install=$i parent=$parent_rid verdict=$VERDICT loss_drop_frac=$LOSS_DROP em0=$PARENT_EM0"

  if [[ $VERDICT == LEAKED ]]; then
    fail "FORMAT-ACQUISITION CHECK (install=$i, $parent_rid): zero-shot EM on the bare-NL
     eval is materially above 0 — the label permutation did not act as a control (algorithm
     may have leaked). Do NOT proceed with ANY of this family's target runs; investigate
     make_preteach_format.py / permute_labels before relaunching."
  elif [[ $VERDICT == NOT_LEARNED ]]; then
    echo "[ts38fs] NOTE install=$i ($parent_rid): loss drop vs base is $LOSS_DROP (<0.10) —" \
      "format not (fully) acquired at this install size. Logged, NOT a launch blocker (see" \
      "stage 5 header comment) — this IS the dose-response measurement at small i."
  fi

  python3 - "$FMT_JSON" "$i" "$parent_rid" "$PARENT_EM0" "$PARENT_EM16" "$PARENT_LOSS" \
    "$VERDICT" "$LOSS_DROP" "$BASE_RID" "$BASE_EM0" "$BASE_EM16" "$BASE_LOSS" <<'PY' || fail "writing $FMT_JSON failed"
import json
import sys
from pathlib import Path

out_path, install, p_rid, p_em0, p_em16, p_loss, verdict, loss_drop, b_rid, b_em0, b_em16, b_loss = sys.argv[1:13]
out = Path(out_path)
data = json.loads(out.read_text()) if out.is_file() else {"eval_config": "eval_bare_target_data_ts38.yaml", "base": None, "parents": {}}
data["base"] = {"run_id": b_rid, "em0": float(b_em0), "em16": float(b_em16), "loss": float(b_loss)}
data["parents"][install] = {
    "run_id": p_rid,
    "em0": float(p_em0),
    "em16": float(p_em16),
    "loss": float(p_loss),
    "verdict": verdict,
    "loss_drop_frac": float(loss_drop),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
done
milestone "format_acquisition_complete json=$FMT_JSON verdicts=$(for i in "${INSTALLS[@]}"; do echo -n "i$i=${FMT_VERDICT_BY_INSTALL[$i]} "; done)"

# ---- stage 6: the target sweep — S outer, I ascending, N ascending, the 5
# reused (i=21544, s=316) cells already excluded by generate_ts38fs_overlays
# .py (they simply have no overlay file) --------------------------------------
FIRST_CELL_CHECKED=0
for s in "${SEEDS[@]}"; do
  for i in "${INSTALLS[@]}"; do
    [[ $i == 21544 && $s == 316 ]] && continue  # reused ts38pf cells
    merged=${MERGED_BY_INSTALL[$i]}
    for n in "${SIZES[@]}"; do
      rid=evt-ts38fs-i${i}-n${n}-s${s}
      overlay=$OVERLAY_DIR/ts38fs_i${i}_n${n}_s${s}.yaml
      [[ -f $overlay ]] || fail "$overlay missing — stage 3's overlay generation should have written it"

      train_or_skip "$rid" \
        python3 train_target.py --config "$TARGET_CONFIG" \
          --override "$overlay" \
          --init-from "$merged" --confirm-cost
      push_run "$rid"

      if [[ $rid == "$PIN_CHECK_RID" && $FIRST_CELL_CHECKED -eq 0 ]]; then
        FIRST_CELL_CHECKED=1
        PIN_OUT=$(python3 - "$rid" <<'PY'
import json
import os
import sys
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
        milestone "pin_check run=$rid stop_reason=$PIN_STOP min_val_nats=$PIN_MIN best_val_nats=$PIN_BEST final_step=$PIN_STEP"
        [[ $PIN_STOP == converged ]] || fail \
          "PIN CHECK: $rid (the first cell trained) ended with stop_reason='$PIN_STOP' (min
     val $PIN_MIN, best $PIN_BEST, step $PIN_STEP), not 'converged'. 54 more cells would
     inherit whatever this is. A max_steps stop is a bug signal, not a result."
      fi
    done
  done
done
milestone "target_sweep_complete cells=55"

# ---- stage 7: receiver-verify every target run on the hub (push-as-you-go
# already happened per cell above; this is the batch check + one retry, the
# same "verify the receiver, not the sender" discipline
# ([[feedback-verify-the-receiver-not-the-sender]]) launch_ts38grid_family.sh
# uses for its own larger run count) ------------------------------------------
run_receiver_check() {
  python3 - "$RELAY_REPO" <<'PY'
import sys

from huggingface_hub import HfApi

from generate_ts38fs_overlays import cells

repo = sys.argv[1]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for i, n, s in cells():
    rid = f"evt-ts38fs-i{i}-n{n}-s{s}"
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
echo "[ts38fs] receiver check (hub files for all 55 target runs):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing the runs the hub is missing)"
  while read -r rid; do
    push_run "$rid"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts38fs] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above (at least one run's manifest/eval_log/prequential log is still missing on the relay after a retry)"
milestone "targets_pushed cells=55 receiver=OK"

# Redundant-looking, NOT the load-bearing push (see the comment on stage
# 4's own push_run) — this just refreshes the manifest/logs after the
# target sweep recorded G5-style theta0 evidence against the parent; the
# adapter itself already reached the relay via stage 4's full push and
# upload_folder's ignore_patterns is additive-only (no delete_patterns), so
# this --metadata-only call cannot remove it.
for i in "${NEW_INSTALLS[@]}"; do
  python3 hf_checkpoint.py push --run-id "evt-ts38fs-parent-n${i}" --repo-id "$RELAY_REPO" --metadata-only ||
    echo "[ts38fs] WARN hf_checkpoint.py push --metadata-only failed for evt-ts38fs-parent-n${i} (best effort)"
done

echo "[ts38fs] format-acquisition evidence at $FMT_JSON — not a run, not pushed; the operator scp's results/ back"

# ---- stage 8: finish --------------------------------------------------------
echo "[ts38fs] MILESTONE analysis_commands"
echo "[ts38fs]   CPU-only, run here off \$GEODE_STORE. This family has no"
echo "[ts38fs]   edl_converged_val_floor.py FAMILIES entry yet (out of this"
echo "[ts38fs]   launcher's scope, same as ts38pf's own dataset_size_sweep.py"
echo "[ts38fs]   note) — adding one is separate follow-up work."
echo "[ts38fs]   REMEMBER when reading this grid: the 5 (i=21544, s=316)"
echo "[ts38fs]   cells carry the OLD evt-ts38pf-preteachfmt-n<N> run ids, not"
echo "[ts38fs]   evt-ts38fs-*  — splice that row in by hand."
echo "[ts38fs]   Read format-acquisition vs install size in $FMT_JSON first"
echo "[ts38fs]   (does the dose-response threshold sit above or below"
echo "[ts38fs]   i=21544?), then the per-cell EDL/D curves per seed."
echo "[ts38fs]   The box is NOT torn down here; teardown is the operator's"
echo "[ts38fs]   call once these artifacts are verified on the relay."
notify "ts38fs family done: 3 parents + 55 target runs, format_acquisition=$(for i in "${INSTALLS[@]}"; do echo -n "i$i:${FMT_VERDICT_BY_INSTALL[$i]} "; done)"
echo "[ts38fs] TERMINAL_SUCCESS runs=55"
