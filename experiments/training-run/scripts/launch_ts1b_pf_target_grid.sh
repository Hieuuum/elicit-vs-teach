#!/usr/bin/env bash
# ts1b pf-arm target-stage grid — the "Stage 3+" dataset-size sweep the
# 2026-08-19 pre-registration (decisions.md) left unauthorized, now
# re-confirmed live by the owner for the pf (pre-teach-FORMAT,
# permuted-label) arm ONLY — the matching base-arm (no pre-teach) grid was
# explicitly declined the same session (see EXPERIMENTS.md §19 and the
# decisions.md entry this launcher cross-references). Does the pf parent's
# format-only install reshape the target-task learning curve (paper App.
# E.1.2's causal-intervention question), now measured at 1.24B instead of
# 38.7M?
#
# GATED ON evt-ts1b-pf-parent STATUS==complete. That run has not started
# training as of this commit — the box is still on the parent-stage LR
# mini-sweep (scripts/launch_ts1b_stage12.sh). This script refuses to do
# anything until the parent finishes; re-run it (or just leave a `while`
# outside it, at the operator's discretion — this script itself does not
# poll) once evt-ts1b-pf-parent lands.
#
# THE FIVE RUNS   evt-ts1b-pf-target-n{1000,4642,21544,100000,316228} —
#   LoRA r512/alpha32 on D_algo_bare (scaffold-free NL add/sub), warm-
#   started from evt-ts1b-pf-parent/model (full FT, NO merge stage — see
#   configs/ts1b_pf_target.yaml's header). NO base-arm comparator exists at
#   this scale (owner declined it) — match_data_order_with stays null,
#   permanently, in every config here; the pf curve is read on its own
#   shape (rising/falling), not "above/below base", until a base grid
#   exists.
#
# LR MINI-SWEEP FIRST (evt-ts1b-pf-target-lrsweep-<label>, 3+ short runs at
# n=21544): no 1.24B target-stage LR pin exists yet — the 38M pin (1e-3)
# does not transfer by authority [[feedback-scope-check-pins-before-reuse]].
# Owner instruction (2026-08-19, mid-build): center the bracket on the
# paper's own Table 3 TinyStories-1B LoRA row (3.53e-4) rather than
# inventing candidates — {1e-4, 3.53e-4, 1e-3} — but still verify locally
# rather than blindly adopt (this project's standing practice). Owner ALSO
# delegated the pick itself (2026-08-19 chat, same delegation already given
# for the parent-stage sweep) — this script auto-selects, no manual
# STOP-and-confirm gate, unlike launch_ts1b_stage12.sh's stage 1. Selection
# rule: lowest-LR rung with a finite, non-diverging val-loss trace and the
# best min_val_nats at the probe budget; if the winner sits at either end
# of the tested range, the script extends the bracket ONE more rung in that
# direction (continuing this repo's established 1-3-10 mantissa ladder,
# e.g. pp_lrsweep's 1e-4/3e-5/1e-5) and re-sweeps before pinning — capped at
# 2 extension rounds as a runaway-cost backstop; if still unresolved at the
# cap, the script proceeds with the best candidate found and prints a WARN
# for the owner/orchestrator to review, rather than looping indefinitely.
# The winner is written directly into configs/ts1b_pf_target.yaml's
# train.lr field (regex substitution of the placeholder line, not a
# rewrite — preserves the file's header comments).
#
# BATCH SIZE 128, NOT the paper's effective 1024 — the paper's own
# per-GPU batch size (Table 3: 128 x 8 GPUs via data parallelism, not
# gradient accumulation). Per [[feedback-paper-fidelity-methodology-not-
# infra-scale]], already the call made identically at the ts1b parent stage
# and every llama_fig2nl* family — do not add 8x accumulation here.
#
# HARD RULES (mirrors every prior launcher in this project):
#   (a) the pf parent is DELIBERATELY UNGATED (no G1/G8 certification —
#       see configs/ts1b_pf_parent.yaml's header); this script never runs
#       gates.py on it at all, and never records anything to its manifest.
#   (b) FULL FT parent, NO MERGE STAGE — runs/evt-ts1b-pf-parent/model IS
#       the checkpoint the target runs warm-start from.
#   (c) NEVER destroy the box — teardown is the operator's call, not taken
#       here.
set -uo pipefail

# ---- env guard: safe HF_TOKEN/NTFY extraction, NEVER source
# /etc/environment whole -- identical fix to launch_ts1b_stage12.sh (see
# that script's header for the PATH-clobber bug this avoids). box_onstart.sh
# writes exactly two vars there (HF_TOKEN, NTFY); extract only those two.
for _v in HF_TOKEN NTFY; do
  if [[ -z ${!_v:-} && -f /etc/environment ]]; then
    _val=$(sed -n "s/^${_v}=//p" /etc/environment | tail -1)
    _val=${_val%\"}
    _val=${_val#\"}
    [[ -n $_val ]] && export "$_v=$_val"
  fi
done
unset _v _val
[[ -f /workspace/venv/bin/activate ]] && . /workspace/venv/bin/activate
python3 - <<'PY' || { echo "[ts1bpftgt] FAILED: python3 lacks required modules (venv not active?)"; exit 1; }
import huggingface_hub, torch, geode  # noqa: F401
PY

cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1bpftgt

echo "[ts1bpftgt] estimated cost (Stage 3+, owner re-confirmed 2026-08-19, pf-arm only):"
echo "[ts1bpftgt]   LR mini-sweep: 3 rungs x 500 steps, LoRA r512 on 1.24B     ~\$1-2"
echo "[ts1bpftgt]     (up to 2 extra bracket-extension rungs if a sweep winner"
echo "[ts1bpftgt]     lands at an endpoint -- +\$0.3-0.7 each, capped)"
echo "[ts1bpftgt]   5-size target grid: max_steps ceilings 1000/1000/5000/10000/"
echo "[ts1bpftgt]     30000 (sourced from configs/sweeps/llama_fig2nl3's own"
echo "[ts1bpftgt]     already-run ceilings at this exact model/adapter/lr class)"
echo "[ts1bpftgt]     -- LoRA forward/backward is full-1.24B-cost per step (only"
echo "[ts1bpftgt]     the trainable param count shrinks), so per-step cost is"
echo "[ts1bpftgt]     close to the parent's full-FT rate; total steps across all"
echo "[ts1bpftgt]     5 sizes (47000) is ~1.5x one pf-parent epoch (31093) --"
echo "[ts1bpftgt]     est \$8-15, extrapolated from launch_ts1b_stage12.sh's own"
echo "[ts1bpftgt]     \$5-10-per-31093-full-FT-steps figure."
echo "[ts1bpftgt]   TOTAL est \$9-17. Box spec: same A100/L40S-class already in use."
echo "[ts1bpftgt]   NOT authorized: a base-arm comparator grid (owner declined,"
echo "[ts1bpftgt]     2026-08-19) or any arm/size beyond what's listed here."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts1b_pf_target_grid.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

PARENT_RID=evt-ts1b-pf-parent
PARENT_MODEL_DIR=$GEODE_STORE/runs/$PARENT_RID/model
PARENT_DATA_HASH=731c18bd3c344c7fb12099bf142db9135fb968c3281c2d7b60469f4acd63c664
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
SIZES=(1000 4642 21544 100000 316228)

TARGET_CONFIG=../configs/ts1b_pf_target.yaml
OVERLAY_DIR=../configs/sweeps/ts1b
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_llama.yaml
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE parent=$PARENT_RID sizes=${#SIZES[@]}"

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

require_converged() {
  local rid=$1 n=$2 stop
  stop=$(stop_reason_of "$rid" target_result)
  milestone "convergence_check run=$rid n=$n stop_reason=$stop"
  [[ $stop == converged ]] || fail \
    "CONVERGENCE CHECK: $rid ended with stop_reason='$stop', not 'converged'. Standing
   policy: a max_steps stop is a BUG SIGNAL for a REAL target run (not the deliberately-
   truncated LR-sweep probes above). Inspect the overlay's max_steps and the loss trace
   before continuing; do not let the remaining sizes inherit whatever this is."
}

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

# ---- gate: the pf parent must exist and be complete -----------------------
P_STATUS=$(status_of "$PARENT_RID")
if [[ $P_STATUS != complete ]]; then
  echo "[ts1bpftgt] $PARENT_RID status='$P_STATUS' (expected complete)."
  echo "[ts1bpftgt] This grid trains on the pf parent's finished checkpoint; there is"
  echo "[ts1bpftgt] nothing to do until scripts/launch_ts1b_stage12.sh's stage 3 lands it."
  echo "[ts1bpftgt] Refusing to proceed. Re-invoke this script after the parent completes."
  exit 1
fi
milestone "parent_gate_check status=$P_STATUS -> proceeding"

# ---- parent verification (mirror launch_ts1b_stage12.sh's own pf checks,
# plus the stale-parent data-order-hash guard every prior family carries) --
PARENT_FIELDS=$(python3 - "$PARENT_RID" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rid = sys.argv[1]
d = json.loads((store / "runs" / rid / "manifest.json").read_text())
status = d.get("status", "MISSING")
method = d.get("training", {}).get("method", "MISSING")
exp = d.get("experiment", {}) or {}
gates = exp.get("gates", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", "MISSING")
final_step = meta.get("final_step", "MISSING")
data_hash = exp.get("data_order_hash", "MISSING")
gate_names = ",".join(sorted(gates)) if gates else "none"
print(status, method, gate_names, stop_reason, final_step, data_hash)
PY
)
read -r P_STATUS2 P_METHOD P_GATES P_STOP P_STEP P_DATA_HASH <<<"$PARENT_FIELDS"
milestone "parent_fields status=$P_STATUS2 method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP data_order_hash=$P_DATA_HASH"

[[ $P_METHOD == full_ft ]] || fail "$PARENT_RID training.method='$P_METHOD' (expected full_ft)"
[[ $P_GATES == none ]] || fail "$PARENT_RID has recorded gates {$P_GATES} — deliberately-ungated parent, inspect before proceeding"
[[ $P_DATA_HASH == "$PARENT_DATA_HASH" ]] || fail \
  "$PARENT_RID manifest data_order_hash=$P_DATA_HASH != pin $PARENT_DATA_HASH (configs/
   ts1b_pf_parent.yaml) — this checkpoint was built from a different D_target_4M_blockperm.
   parquet/config than this launcher expects. Do not train the target grid on a stale parent."
[[ -f $PARENT_MODEL_DIR/model.safetensors ]] || fail "$PARENT_MODEL_DIR/model.safetensors missing"
milestone "parent_verified full_ft, ungated, data_order_hash=OK, checkpoint present"

# ---- data preflight: reused frozen corpus, no new datagen -----------------
for f in D_algo_bare.parquet D_algo_eval_bare.parquet; do
  [[ -f $DATA_DIR/$f ]] || fail \
    "$DATA_DIR/$f missing — this grid does not build new data, it reuses the frozen
   D_algo_bare corpus every other family (including llama_fig2nl3) trains on. If this box
   never ran a family that regenerated it, pull it from the relay's cache/ path or run
   datagen/make_bare_sets.py against the already-present D_algo/D_algo_eval."
done
milestone "data_present D_algo_bare + D_algo_eval_bare at pinned local paths"

# ---- LR mini-sweep, auto-pick, auto-extend the bracket if needed ---------
python3 - "$TARGET_CONFIG" "$OVERLAY_DIR" "$PARENT_MODEL_DIR" <<'PY' || fail "LR mini-sweep"
import json
import os
import subprocess
import sys
from pathlib import Path

target_config, overlay_dir, parent_model_dir = sys.argv[1:4]
store = Path(os.environ["GEODE_STORE"])
TAG = "ts1bpftgt"

import math

# Canonical 1-3-10 mantissa ladder this project already uses everywhere for
# LR sweeps (e.g. pp_lrsweep 1e-4/3e-5/1e-5): ..., 1e-4, 3e-5, 1e-5, 3e-6, ...
# Decompose lr = mantissa * 10**exp with mantissa in [1, 10) via log10, snap
# the mantissa to the nearest of {1, 3} (geometric midpoint sqrt(3)), then
# step one position along that fixed ladder. Building the label directly
# from the integer (mantissa, exp) pair -- rather than formatting the float
# with %g -- avoids Python's 2-digit-exponent default ("1e-05") disagreeing
# with this repo's file-naming convention ("1e-5").
def _decompose(lr: float) -> tuple[float, int]:
    e = math.floor(math.log10(lr) + 1e-9)
    return lr / (10 ** e), e

def _snap(m: float) -> int:
    return 1 if m < 3 ** 0.5 else 3

def step_down(lr: float) -> tuple[float, str]:
    m, e = _decompose(lr)
    m = _snap(m)
    new_m, new_e = (3, e - 1) if m == 1 else (1, e)
    label = f"{new_m}e{new_e}"
    return float(label), label  # str->float parse, not m*10.0**e (avoids fp noise like 3.0000000000000004e-05)

def step_up(lr: float) -> tuple[float, str]:
    m, e = _decompose(lr)
    m = _snap(m)
    new_m, new_e = (1, e + 1) if m == 3 else (3, e)
    label = f"{new_m}e{new_e}"
    return float(label), label

def _yaml_float(x: float) -> str:
    # YAML 1.1 (this repo's stated target, per every "dot-mantissa form is
    # mandatory" comment elsewhere) only recognizes scientific notation as a
    # float when the mantissa has a decimal point -- Python's repr() drops
    # it for small magnitudes (repr(3e-05) == "3e-05", no dot), which a
    # strict YAML 1.1 parser reads back as a STRING, silently breaking the
    # pin. Insert ".0" before a bare-mantissa "e" if present; plain decimal
    # reprs (e.g. "0.0001") are already unambiguous and pass through as-is.
    s = repr(x)
    if "e" in s and "." not in s.split("e")[0]:
        mantissa, exp = s.split("e")
        s = f"{mantissa}.0e{exp}"
    return s

PROBE_N = 21544
OVERLAY_TEMPLATE = """# ts1b pf-arm TARGET-STAGE LR mini-sweep, bracket-extension rung {label}.
# Auto-generated by launch_ts1b_pf_target_grid.sh's mini-sweep loop (the
# {label} winner from the previous round sat at an endpoint of the tested
# range, so the bracketing rule [[feedback-nulls-need-bracketing]] extends
# one rung further before pinning). Same probe conditions as the seed
# rungs (pf_target_lrsweep_1e-4.yaml et al.): n=21544, 500-step
# deliberately-incomplete budget, min_steps 5000 keeps eps/k inert.
run_id: evt-ts1b-pf-target-lrsweep-{label}
data:
  n_examples: {probe_n}
train:
  lr: {lr}
  eval_every: 25
  max_steps: 500
  stopping:
    min_steps: 5000
"""

def overlay_path(label: str) -> Path:
    return Path(overlay_dir) / f"pf_target_lrsweep_{label}.yaml"

def ensure_overlay(label: str, lr: float) -> Path:
    p = overlay_path(label)
    if not p.is_file():
        p.write_text(OVERLAY_TEMPLATE.format(label=label, lr=_yaml_float(lr), probe_n=PROBE_N))
        print(f"[{TAG}] MILESTONE lrsweep_overlay_generated label={label} lr={lr} path={p}")
    return p

def run_rung(label: str, lr: float) -> None:
    rid = f"evt-ts1b-pf-target-lrsweep-{label}"
    manifest_p = store / "runs" / rid / "manifest.json"
    if manifest_p.is_file() and json.loads(manifest_p.read_text()).get("status") == "complete":
        print(f"[{TAG}] MILESTONE train_skip run={rid} status=complete")
        return
    overlay = ensure_overlay(label, lr)
    print(f"[{TAG}] MILESTONE train_start run={rid}")
    r = subprocess.run(
        [
            "python3", "train_target.py",
            "--config", target_config,
            "--override", str(overlay),
            "--init-from", parent_model_dir,
            "--confirm-cost",
        ]
    )
    if r.returncode != 0:
        print(f"[{TAG}] FAILED: {rid} training (exit {r.returncode})")
        sys.exit(1)
    if not manifest_p.is_file() or json.loads(manifest_p.read_text()).get("status") != "complete":
        print(f"[{TAG}] FAILED: {rid} did not reach status=complete")
        sys.exit(1)
    print(f"[{TAG}] MILESTONE train_complete run={rid}")

def read_result(label: str) -> dict:
    rid = f"evt-ts1b-pf-target-lrsweep-{label}"
    manifest = json.loads((store / "runs" / rid / "manifest.json").read_text())
    tr = manifest.get("experiment", {}).get("target_result", {}) or {}
    eval_log_p = store / "runs" / rid / "eval_log.jsonl"
    rows = []
    if eval_log_p.is_file():
        for line in eval_log_p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    min_val = tr.get("min_val_nats")
    last_val = rows[-1]["val_loss_nats"] if rows else None
    # "stable" = finite min AND the run hasn't blown up since its best point
    # (last eval no more than 1.5x the min) -- a simple, documented proxy
    # for non-diverging descent, not a full slope fit (CLAUDE.md simplicity
    # rule: minimum logic that answers the actual question here).
    stable = (
        min_val is not None
        and last_val is not None
        and math.isfinite(min_val)
        and last_val <= 1.5 * min_val
    )
    return {"min_val_nats": min_val, "last_val_nats": last_val, "stable": stable}

def stable_candidates(results: dict) -> dict:
    # isfinite guard is load-bearing, not redundant with `stable`: min()
    # over a dict containing a NaN value is a Python trap (NaN comparisons
    # are always False, so min()'s result silently depends on iteration
    # order and can return the NaN row itself) -- excluding non-finite rows
    # here, not just via the `stable` flag upstream, is what actually
    # prevents a broken LR from ever reaching the pin below.
    return {
        k: v for k, v in results.items()
        if v["stable"] and v["min_val_nats"] is not None and math.isfinite(v["min_val_nats"])
    }

candidates = [("1e-4", 1.0e-4), ("3.53e-4", 3.53e-4), ("1e-3", 1.0e-3)]
results = {}
for label, lr in candidates:
    run_rung(label, lr)
    results[label] = {"lr": lr, **read_result(label)}

winner_label = winner_lr = None
for round_i in range(2):  # cap: at most 2 bracket-extension rounds
    stable = stable_candidates(results)
    if not stable:
        print(f"[{TAG}] FAILED: no LR rung produced a stable (finite, non-diverging) val-loss "
              f"trace -- {results}. Refusing to pin a broken LR into the 5-size grid; inspect "
              f"the mini-sweep runs' eval_log.jsonl before retrying.")
        sys.exit(1)
    winner_label = min(stable, key=lambda k: stable[k]["min_val_nats"])
    winner_lr = results[winner_label]["lr"]
    lrs_sorted = sorted(results.values(), key=lambda v: v["lr"])
    is_low_end = winner_lr == lrs_sorted[0]["lr"]
    is_high_end = winner_lr == lrs_sorted[-1]["lr"]
    if not (is_low_end or is_high_end):
        print(f"[{TAG}] MILESTONE lrsweep_winner label={winner_label} lr={winner_lr} "
              f"min_val_nats={results[winner_label]['min_val_nats']} (interior of tested "
              f"range {[v['lr'] for v in lrs_sorted]}, no extension needed)")
        break
    ext_lr, ext_label = step_down(winner_lr) if is_low_end else step_up(winner_lr)
    if ext_label in results:
        print(f"[{TAG}] WARN: bracket extension rung {ext_label} was already tested; "
              f"stopping the extension loop with winner {winner_label} to avoid a cycle.")
        break
    print(f"[{TAG}] MILESTONE lrsweep_extend winner={winner_label} at range endpoint "
          f"({'low' if is_low_end else 'high'}) -> testing {ext_label} "
          f"(round {round_i + 1}/2)")
    run_rung(ext_label, ext_lr)
    results[ext_label] = {"lr": ext_lr, **read_result(ext_label)}
else:
    stable = stable_candidates(results)
    if not stable:
        print(f"[{TAG}] FAILED: no LR rung produced a stable trace after the extension cap -- "
              f"{results}. Refusing to pin a broken LR.")
        sys.exit(1)
    winner_label = min(stable, key=lambda k: stable[k]["min_val_nats"])
    winner_lr = results[winner_label]["lr"]
    print(f"[{TAG}] WARN: bracket extension cap (2 rounds) reached, winner still at a range "
          f"endpoint. Proceeding with {winner_label} (lr={winner_lr}) as the best available "
          f"pick -- FLAGGED for owner/orchestrator review before trusting the 5-size grid's "
          f"absolute EDL/D scale (the shape/comparison read is unaffected either way).")

print(f"[{TAG}] MILESTONE lrsweep_all_results {json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()})}")

# ---- pin the winner into the base target config (regex substitution of
# the placeholder `  lr: 3.53e-4` line only -- preserves every comment) ----
cfg_path = Path(target_config)
text = cfg_path.read_text()
import re
new_line = f"  lr: {_yaml_float(winner_lr)}                   # PINNED {winner_label} — ts1b pf-arm target-stage LR mini-sweep winner (launch_ts1b_pf_target_grid.sh, auto-selected per owner delegation 2026-08-19)"
new_text, n_subs = re.subn(r"(?m)^  lr: [0-9.eE+-]+.*$", new_line, text, count=1)
if n_subs != 1:
    print(f"[{TAG}] FAILED: expected exactly one 'train.lr' line in {cfg_path}, substituted {n_subs}")
    sys.exit(1)
cfg_path.write_text(new_text)
print(f"[{TAG}] MILESTONE lr_pinned winner={winner_label} lr={winner_lr} config={cfg_path}")
PY
milestone "lrsweep_complete"

# ---- the five target runs, ascending n, push-as-you-go --------------------
for n in "${SIZES[@]}"; do
  rid=evt-ts1b-pf-target-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$TARGET_CONFIG" \
      --override "$OVERLAY_DIR/ts1b_pf_target_n${n}.yaml" \
      --init-from "$PARENT_MODEL_DIR" --confirm-cost
  require_converged "$rid" "$n"
  record_g5 "$rid"
  push_run "$rid" "$RELAY_REPO"
  milestone "size_complete n=$n run=$rid"
done

python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts1bpftgt] {'run_id':<28}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    rid = f"evt-ts1b-pf-target-n{n}"
    p = store / "runs" / rid / "manifest.json"
    r = (json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}) or {}
    edl = r.get("edl_per_label_token_nats")
    print(
        f"[ts1bpftgt] {rid:<28}{r.get('final_step', 'MISSING'):>12}"
        f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
        f"{edl if edl is not None else 'n/a':>26}"
    )
PY

# ---- push receiver-verify (feedback-verify-the-receiver-not-the-sender) --
run_receiver_check() {
  python3 - "$RELAY_REPO" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
sizes = sys.argv[2:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for n in sizes:
    rid = f"evt-ts1b-pf-target-n{n}"
    # model.safetensors is the canonical self-contained checkpoint for every
    # run in this project, LoRA included (train_target.py:586, the full
    # state dict, base + adapter tensors together) -- same file the ts38pp/
    # ts1b parent receiver checks verify, not PEFT's separate adapter_model
    # naming.
    required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
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
echo "[ts1bpftgt] receiver check (hub files for all 5 target runs):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing whatever the hub is missing)"
  while read -r rid; do
    push_run "$rid" "$RELAY_REPO"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts1bpftgt] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED -- see output above"
milestone "receiver_verified sizes=${#SIZES[@]}"

notify "ts1b pf-arm target grid DONE: 5/5 sizes converged, pushed, receiver-verified"
echo "[ts1bpftgt] TERMINAL_SUCCESS"
