#!/usr/bin/env bash
# Runs 7 -> 8 -> 9 -> 10, unattended — the full-chain launcher (owner
# 2026-07-24, superseding the same-day "Llama script waits for box sizing";
# launch_pair_1m.sh remains for a 7/8-only box). Sequential, hours-long.
# Every completed run is skipped on re-run, so a crash or box restart
# resumes from where it stopped. ntfy ping at every stage boundary and on
# every failure (set $NTFY).
#
#   ./launch_chain_7_10.sh --confirm-cost [--push-and-prune]
#
# Per run: train -> gate evidence (G4 for run 9; G5 for 7/8/9/10 — run 9's
# G5 doubles as the "zero-shot op add/sub on the merged parent" evidence,
# llama-guide §3: the wrapped checkpoint and model_merged/ are the same
# weights, V5.52 exact fold) -> optional prune. Run 9 additionally merges
# to model_merged/ (run 10's parent — NEVER model/, zoo.load_model rule).
#
# --push-and-prune (owner 2026-07-24): after a run's gates are recorded,
# push the run INCLUDING snapshots/ to the private relay
# (mhieuuu/geode-store), verify every local file is listed on the hub,
# then delete the heavy dirs (snapshots/, model/, model_merged/) locally.
# Peak disk ~120 GB instead of ~250 GB (run 10 stores NO snapshots, owner
# 2026-07-24 — only runs 7/8 carry them). Costs: extraction must later
# pull the 7/8 snapshots back (--with-snapshots), and the relay grows by
# ~100-200 GB — check the HF plan's private-storage quota first. Needs
# HF_WRITE_TOKEN
# (env, or typed at start on a tty); it is passed per-push only
# (HF_TOKEN=<write> python ...) — the box's ambient login stays READ-only.
# Runs 5/6 are NEVER touched (their relay push is owner-held).
#
# Refuses to start unless ALL of:
#   - train.lr pinned, EQUAL and non-null in ALL FOUR run yamls, matching
#     the sweep winner in this store (one LR everywhere, owner 2026-07-24;
#     >=3 complete evt-run8-sweep-lr* runs) — or, since those run dirs were
#     lost with the old box (2026-07-24), the committed configs/lr_pin.yaml;
#   - free disk covers the PENDING runs (sum; per-run max when pruning);
#   - parent checkpoints present for pending runs 7/8;
#   - verify_llama_tokenizer.py PASSES (when 9/10 are pending);
#   - prune mode only: HF_WRITE_TOKEN accepted by the hub.
# stop_reason=max_steps on ANY of these four runs is a bug signal.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_chain_7_10.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
PRUNE=0
[[ " $* " == *" --push-and-prune "* ]] && PRUNE=1

REPO_ROOT=$(git rev-parse --show-toplevel)
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
echo "[chain] repo  $(git log --oneline -1)"
echo "[chain] store $GEODE_STORE  prune=$PRUNE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {
  notify "chain FAILED: $1"
  echo "[chain] FAILED: $1" >&2
  exit 1
}

status_of() { # run_id -> complete | missing | <status>
  python - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

# ---- guards (all before any GPU spend) -----------------------------------

if ((PRUNE)); then
  if [[ -z ${HF_WRITE_TOKEN:-} ]]; then
    if [[ -t 0 ]]; then
      read -rsp "HF WRITE token (prune mode): " HF_WRITE_TOKEN && echo
    else
      echo "chain: --push-and-prune needs HF_WRITE_TOKEN (env, or run on a tty)" >&2
      exit 1
    fi
  fi
  HF_TOKEN=$HF_WRITE_TOKEN hf auth whoami >/dev/null 2>&1 || {
    echo "chain: HF_WRITE_TOKEN rejected by the hub" >&2
    exit 1
  }
  echo "[chain] prune mode: write token accepted (used per-push only, never stored)"
fi

# 1. LR pin: one shared non-null value in all four yamls == the sweep winner.
python - <<'PY' || exit 1
import json, os, sys
from pathlib import Path

import yaml

cfg = Path("../configs")
pins = {
    name: yaml.safe_load((cfg / f).read_text())["train"]["lr"]
    for name, f in (
        ("run7", "run7_target_1m.yaml"),
        ("run8", "run8_target_1m.yaml"),
        ("run9", "run9_llama1b_inst.yaml"),
        ("run10", "run10_llama1b_target.yaml"),
    )
}
vals = set(pins.values())
if None in vals or len(vals) != 1:
    sys.exit(f"chain: ONE shared non-null train.lr required in all four yamls (one LR everywhere) — got {pins}")
lr = float(next(iter(vals)))
best, best_lr, seen = float("inf"), None, 0
for p in sorted((Path(os.environ["GEODE_STORE"]) / "runs").glob("evt-run8-sweep-lr*/manifest.json")):
    m = json.loads(p.read_text())
    if m["status"] != "complete":
        continue
    seen += 1
    v = m["experiment"]["target_result"]["min_val_nats"]
    v = float("inf") if v is None else v
    if v < best:
        best, best_lr = v, float(m["training"]["optimizer"]["lr"])
if seen >= 3:
    if abs(lr - best_lr) > 1e-12:
        sys.exit(f"chain: configs pin lr={lr} but the sweep winner here is {best_lr} — re-pin (or resolve the discrepancy) first")
    print(f"[chain] lr pin {lr} matches the sweep winner in all four yamls ({seen} sweep runs checked)")
else:
    # The evt-run8-sweep-lr* run dirs were lost with the old box (2026-07-24,
    # owner-accepted); the committed pin record is the evidence instead
    # (numbers in decisions.md + run7-8-guide.md).
    rec_path = cfg / "lr_pin.yaml"
    if not rec_path.is_file():
        sys.exit(
            f"chain: only {seen} complete sweep runs in this store and no configs/lr_pin.yaml — "
            "the pin is unreviewable. The pin record is committed together with the pin itself "
            "after the Arm-A pilot passes (run7-8-guide.md §3); git pull to get it."
        )
    rec_lr = float(yaml.safe_load(rec_path.read_text())["lr"])
    if abs(lr - rec_lr) > 1e-12:
        sys.exit(f"chain: configs pin lr={lr} but committed configs/lr_pin.yaml says {rec_lr} — resolve before launching")
    print(f"[chain] lr pin {lr} matches committed lr_pin.yaml (sweep manifests lost with the old box; provenance in decisions.md)")
PY

# 2. Disk: worst-case GB each PENDING run adds (decisions.md 2026-07-24);
#    prune mode needs only the largest single pending run at once.
CHAIN_PRUNE=$PRUNE python - <<'PY' || exit 1
import json, os, shutil, sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
prune = os.environ["CHAIN_PRUNE"] == "1"
need = {
    "evt-run7-armA-target-1m": 100,   # 2048 x ~48 MB adapter snapshots
    "evt-run8-armB-target-1m": 100,
    "evt-run9-llama1b-inst": 15,      # wrapped + merged ~5 GB each + hub cache
    "evt-run10-llama1b-target": 15,   # NO snapshots (owner 2026-07-24) —
                                      # final wrapped checkpoint + logs
}


def status(rid: str) -> str:
    p = store / "runs" / rid / "manifest.json"
    return json.loads(p.read_text())["status"] if p.is_file() else "missing"


pending = [r for r in need if status(r) != "complete"]
if not pending:
    print("[chain] all four runs already complete — nothing needs disk")
    sys.exit(0)
required = (max if prune else sum)(need[r] for r in pending) + 20  # headroom
free = shutil.disk_usage(store if store.exists() else store.parent).free // 10**9
if free < required:
    sys.exit(
        f"chain: {free} GB free < {required} GB needed for pending {pending} (prune={prune}) — "
        "free disk, use --push-and-prune, or rent a bigger box "
        "(vast.ai disk size is FIXED at instance creation)"
    )
print(f"[chain] disk ok: {free} GB free >= {required} GB for pending {pending}")
PY

# 3. Parents for pending runs 7/8 (run 10's parent is produced by this script).
for pair in "evt-run7-armA-target-1m:evt-run3-armA-inst" "evt-run8-armB-target-1m:evt-run4-armB-inst"; do
  rid=${pair%%:*} parent=${pair##*:}
  [[ $(status_of "$rid") == complete ]] && continue
  [[ -f $GEODE_STORE/runs/$parent/model/model.safetensors ]] || {
    echo "chain: parent checkpoint missing — python hf_checkpoint.py pull --run-id $parent" >&2
    exit 1
  }
done

# 4. Tokenizer verification before anything Llama trains (llama-guide §1).
if [[ $(status_of evt-run9-llama1b-inst) != complete || $(status_of evt-run10-llama1b-target) != complete ]]; then
  python verify_llama_tokenizer.py || {
    echo "chain: verify_llama_tokenizer FAILED — stop, bring the output to the owner (llama-guide §1)" >&2
    exit 1
  }
fi

# ---- stage helpers --------------------------------------------------------

launch_run() { # $1 run_id  $2 trainer.py  $3 config yaml  $4 --init-from value
  local st
  st=$(status_of "$1")
  if [[ $st == complete ]]; then
    echo "[chain] $1 already complete — skipping"
    return 0
  elif [[ $st != missing ]]; then
    fail "$1 exists with status '$st' — crashed launch? Inspect it (and delete the run dir by hand) before re-running"
  fi
  echo "[chain] launching $1 ..."
  if python "$2" --config "../configs/$3" --init-from "$4" --confirm-cost; then
    notify "chain: $1 done"
  else
    fail "$1 training (stopping before the next run burns budget)"
  fi
}

run_smoke() { # $1 smoke run_id  $2.. the smoke command — disposable by design
  echo "[chain] smoke $1 ..."
  rm -rf "${GEODE_STORE:?}/runs/${1:?}"
  if "${@:2}"; then
    rm -rf "${GEODE_STORE:?}/runs/${1:?}"
  else
    fail "$1 smoke (OOM = protocol question: batch/precision are owner calls — llama-guide troubleshooting)"
  fi
}

g5_if_missing() { # $1 run_id — recorded evidence, idempotent across re-runs
  if python - "$1" <<'PY'
import json, os, sys
from pathlib import Path

m = json.loads((Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json").read_text())
sys.exit(0 if "G5" in m.get("experiment", {}).get("gates", {}) else 1)
PY
  then
    echo "[chain] $1 G5 already recorded — skipping"
    return 0
  fi
  python gates.py g5 --run "$1" --config ../configs/eval_target_data.yaml --device cuda \
    || fail "$1 G5 evidence"
}

prune_run() { # $1 run_id  $2.. heavy dirs to delete after a VERIFIED relay push
  ((PRUNE)) || return 0
  local rid=$1 d remaining=0
  shift
  for d in "$@"; do [[ -e $GEODE_STORE/runs/$rid/$d ]] && remaining=1; done
  ((remaining)) || {
    echo "[chain] $rid already pruned — skipping"
    return 0
  }
  echo "[chain] prune: pushing $rid (with snapshots) to the relay ..."
  HF_TOKEN=$HF_WRITE_TOKEN python hf_checkpoint.py push --run-id "$rid" --with-snapshots \
    || fail "$rid relay push (nothing deleted)"
  HF_TOKEN=$HF_WRITE_TOKEN python - "$rid" <<'PY' || fail "$rid hub verification (nothing deleted)"
import os, sys
from pathlib import Path

from huggingface_hub import HfApi

rid = sys.argv[1]
src = Path(os.environ["GEODE_STORE"]) / "runs" / rid
local = [p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()]
remote = set(HfApi().list_repo_files("mhieuuu/geode-store"))
missing = [r for r in local if f"runs/{rid}/{r}" not in remote]
if missing:
    sys.exit(f"{len(missing)} local files not on the hub (e.g. {missing[:3]}) — refusing to delete")
print(f"[chain] hub holds all {len(local)} files of {rid} — pruning locally")
PY
  for d in "$@"; do rm -rf "${GEODE_STORE:?}/runs/${rid:?}/$d"; done
  notify "chain: $rid relayed + pruned ($*)"
}

# ---- the chain ------------------------------------------------------------

notify "chain: started ($(git rev-parse --short HEAD), prune=$PRUNE)"

# Runs 7 then 8 — G7 needs run 7's manifest registered first (kept by prune).
launch_run evt-run7-armA-target-1m train_target.py run7_target_1m.yaml \
  "$GEODE_STORE/runs/evt-run3-armA-inst/model"
g5_if_missing evt-run7-armA-target-1m
prune_run evt-run7-armA-target-1m snapshots model

launch_run evt-run8-armB-target-1m train_target.py run8_target_1m.yaml \
  "$GEODE_STORE/runs/evt-run4-armB-inst/model"
g5_if_missing evt-run8-armB-target-1m
prune_run evt-run8-armB-target-1m snapshots model

# Run 9 — smoke, install, G4 gate (+ fallback tripwire), zero-shot evidence, merge.
if [[ $(status_of evt-run9-llama1b-inst) != complete ]]; then
  run_smoke evt-run9-smoke \
    python train_sft.py --config ../configs/run9_llama1b_inst.yaml \
    --override ../configs/pilot/llama9_smoke.yaml \
    --init-from meta-llama/Llama-3.2-1B --confirm-cost
fi
launch_run evt-run9-llama1b-inst train_sft.py run9_llama1b_inst.yaml \
  meta-llama/Llama-3.2-1B

if ! python - <<'PY'
import json, os, sys
from pathlib import Path

m = json.loads((Path(os.environ["GEODE_STORE"]) / "runs" / "evt-run9-llama1b-inst" / "manifest.json").read_text())
sys.exit(0 if "G4" in m.get("experiment", {}).get("gates", {}) else 1)
PY
then
  python gates.py g4 --run evt-run9-llama1b-inst \
    --config ../configs/eval_target_data.yaml --device cuda || fail "run9 G4 command"
fi
python - <<'PY' || fail "run9 G4 at the shared LR — owner fallback: revive the gentle installer sweep (pilot/llama9_sweep_lr*), run9 config header note 3"
import json, os, sys
from pathlib import Path

m = json.loads((Path(os.environ["GEODE_STORE"]) / "runs" / "evt-run9-llama1b-inst" / "manifest.json").read_text())
g = m.get("experiment", {}).get("gates", {}).get("G4")
if not g:
    sys.exit("run9 G4 not recorded")
if not g.get("pass"):
    sys.exit(f"run9 G4 pass=false: {g}")
print("[chain] run9 G4 PASS")
PY

# Zero-shot op add/sub evidence BEFORE run 10 (llama-guide §3) — same
# weights as the merged parent (V5.52 exact fold), recorded as run-9 G5.
g5_if_missing evt-run9-llama1b-inst

[[ -d $GEODE_STORE/runs/evt-run9-llama1b-inst/model_merged ]] || {
  python merge_adapter.py --run-id evt-run9-llama1b-inst || fail "run9 merge_adapter"
}

# Run 10 — smoke (memory worst case), the EDL measurement, G5 evidence.
if [[ $(status_of evt-run10-llama1b-target) != complete ]]; then
  run_smoke evt-run10-smoke \
    python train_target.py --config ../configs/run10_llama1b_target.yaml \
    --override ../configs/pilot/llama10_smoke.yaml \
    --init-from "$GEODE_STORE/runs/evt-run9-llama1b-inst/model_merged" --confirm-cost
fi
launch_run evt-run10-llama1b-target train_target.py run10_llama1b_target.yaml \
  "$GEODE_STORE/runs/evt-run9-llama1b-inst/model_merged"
g5_if_missing evt-run10-llama1b-target

# Run 9 prunes only now — run 10 needed model_merged/ as its parent.
# (Run 10 has no snapshots/ — owner 2026-07-24 — so only model/ goes.)
prune_run evt-run10-llama1b-target model
prune_run evt-run9-llama1b-inst model model_merged

# ---- summary ---------------------------------------------------------------

python - <<'PY'
import json, os
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
for rid in (
    "evt-run7-armA-target-1m",
    "evt-run8-armB-target-1m",
    "evt-run9-llama1b-inst",
    "evt-run10-llama1b-target",
):
    m = json.loads((store / "runs" / rid / "manifest.json").read_text())
    exp = m.get("experiment", {})
    r = exp.get("target_result") or {}
    stop = f"stop {r['final_step']} ({r['stop_reason']}), min_val {r['min_val_nats']}" if r else "-"
    flag = "  <-- BUG SIGNAL: did not converge" if r.get("stop_reason") == "max_steps" else ""
    gates = ",".join(sorted(exp.get("gates", {}))) or "-"
    print(f"{m['status']:>9}  {rid}: {stop}  gates[{gates}]{flag}")
PY
# Keep-alive reminder names whatever actually holds snapshots on THIS box
# (old box: runs 5/6; new chain box: runs 7/8 unless pruned to the relay).
SNAP_RUNS=$(find "$GEODE_STORE/runs" -mindepth 2 -maxdepth 2 -type d -name snapshots ! -empty 2>/dev/null |
  xargs -rn1 dirname | xargs -rn1 basename | sort | paste -sd' ' -)
notify "chain COMPLETE: runs 7+8+9+10 — snapshot-bearing runs on THIS box: ${SNAP_RUNS:-none}; keep it ALIVE until they are extracted or relayed"
