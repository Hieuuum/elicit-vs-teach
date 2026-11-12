#!/usr/bin/env bash
# run_box_pg.sh — pairing gap (PG, link gap) on the ts1b parents (HANDOFF-ts1b-pg.md).
# Data: data/ceiling4x4/fold_00, the 1,024 validation problems (same set as run_box_hard_acc.sh).
#
#   step 1  prefetch the 3 models + the tokenizer, and pull the existing hard-acc results
#           tree (scores.pt + JSONs, ~15 MB) from $RESULTS_REPO:results/ts1b_hard_acc
#   step 2  GPU 0: ts1b_pg.py, 5 reads one after another (ep_symbol, ep_word, fp_word,
#           base_word, base_symbol) -> $ROOT/pg/<read>.json
#   step 3  gate: every read has n_pairs > 0, its control check passed, and all 5 share the
#           same control answers (sha256)
#   step 4  ts1b_hard_acc_report.py over the whole tree -> report.json / report.md (+ PG section)
#   step 5  push pg/ + report.json + report.md to $RESULTS_REPO:results/ts1b_hard_acc and
#           list-verify
#
# Usage: bash run_box_pg.sh --confirm-cost
#   NO_PUSH=1        skip the push
#   SKIP_PREFETCH=1  no model/tokenizer download and no HF pull (local dry run)
#   TREE_SRC=DIR     copy the hard-acc tree from DIR instead of pulling it from HF
#   FAKE_MODEL=1     score with a tiny random Llama instead of the real checkpoints (dry run)
#   PG_ROOT=DIR      results tree (default results/ts1b_hard_acc); DEVICE (default cuda);
#                    BATCH_SIZE (default 64)
# Local dry run (CPU, no network):
#   SKIP_PREFETCH=1 NO_PUSH=1 FAKE_MODEL=1 DEVICE=cpu HF_HUB_OFFLINE=1 PG_ROOT=/tmp/x \
#     TREE_SRC=<a hard-acc tree> bash run_box_pg.sh --confirm-cost
# Progress: $ROOT/pg/run.log; last line EXIT=0 (everything done + pushed) or EXIT=1.
set -uo pipefail
cd "$(dirname "$0")"
echo "[pg] estimated cost: 1x RTX 3090 ~20 min (setup ~10 min, 5 reads x ~1-2 min) at ~\$0.2-0.3/h ~ \$0.1-0.2:"
echo "[pg] each read scores ~18.4k sequences of ~11-16 tokens (data + control, true items + swaps)."
[[ " $* " == *" --confirm-cost "* ]] || { echo "run_box_pg.sh: --confirm-cost required" >&2; exit 1; }

export PYTHONPATH=$(git rev-parse --show-toplevel):$(pwd) HF_HUB_DISABLE_XET=1
ROOT=${PG_ROOT:-results/ts1b_hard_acc}
PG=$ROOT/pg
LOGD=$PG/logs
DATA=data/ceiling4x4/fold_00
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
SUB=results/ts1b_hard_acc
BS=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-cuda}
mkdir -p "$LOGD"
LOG=$PG/run.log
say() { echo "[pg $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for f in pairs_seed0.pt pairs_symbol_seed0.pt copy_pairs_seed1.pt; do
  [[ -f $DATA/$f ]] || { say "missing $DATA/$f"; echo "EXIT=1" >>"$LOG"; exit 1; }
done

declare -A M=(
  [base]=podhajskimarcin/evt-ts1b-base
  [elicit_parent]=podhajskimarcin/evt-ts1b-op-bridge-mix
  [fmt_parent]=mhieuuu/evt-ts1b-fmt-parent
)
# read|model key|form
READS=(
  "ep_symbol|elicit_parent|symbol"
  "ep_word|elicit_parent|word"
  "fp_word|fmt_parent|word"
  "base_word|base|word"
  "base_symbol|base|symbol"
)
PG_PY=(ts1b_pg.py)
if [[ -n ${FAKE_MODEL:-} ]]; then
  # dry run: ts1b_pg.main with run.load_model swapped for a tiny random Llama (real tokenizer
  # and data, so the control answers and the token layout are the real ones)
  cat >"$PG/.fake_pg.py" <<'PY'
import sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import run
import ts1b_pg

def fake_load_model(path, device):
    torch.manual_seed(sum(map(ord, path)))
    cfg = LlamaConfig(vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True)
    return LlamaForCausalLM(cfg).to(device).eval()

run.load_model = fake_load_model
sys.exit(ts1b_pg.main())
PY
  PG_PY=("$PG/.fake_pg.py")
fi
say "repo $(git rev-parse --short HEAD), device $DEVICE, batch $BS, root $ROOT${FAKE_MODEL:+, FAKE MODEL}"

# ---- step 1: prefetch models + tokenizer, then the hard-acc tree (local copy or HF)
[[ -n ${SKIP_PREFETCH:-} ]] || python3 - "${M[@]}" <<'PY' 2>&1 | tee -a "$LOG" || { say "FAILED prefetch"; echo "EXIT=1" >>"$LOG"; exit 1; }
import sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
models = sys.argv[1:]
def get(m):
    snapshot_download(m, allow_patterns=[f"runs/{m.split('/')[-1]}/model/*"])
    return m
with ThreadPoolExecutor(4) as ex:
    for m in ex.map(get, models):
        print("have", m, flush=True)
snapshot_download("meta-llama/Llama-3.2-1B", allow_patterns=["tokenizer*", "special_tokens_map.json"])
print("have tokenizer", flush=True)
PY
if [[ -n ${TREE_SRC:-} ]]; then
  cp -r "$TREE_SRC"/. "$ROOT"/ && say "tree copied from $TREE_SRC" || { say "FAILED tree copy"; echo "EXIT=1" >>"$LOG"; exit 1; }
elif [[ -z ${SKIP_PREFETCH:-} ]]; then
  python3 - "$ROOT" "$RESULTS_REPO" "$SUB" <<'PY' 2>&1 | tee -a "$LOG" || { say "FAILED tree pull"; echo "EXIT=1" >>"$LOG"; exit 1; }
import shutil, sys, tempfile
from pathlib import Path
from huggingface_hub import snapshot_download
root, repo, sub = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    snapshot_download(repo, local_dir=tmp, allow_patterns=[
        f"{sub}/scores/*", f"{sub}/overlap/*.json", f"{sub}/functional/*.json", f"{sub}/cmap/*.json"])
    shutil.copytree(Path(tmp) / sub, root, dirs_exist_ok=True)
n = sum(1 for p in Path(root).rglob("*") if p.is_file() and "pg" not in p.relative_to(root).parts)
print(f"have hard-acc tree ({n} files)", flush=True)
PY
fi

# ---- step 2: the 5 reads, one GPU, one after another (one retry at half batch: OOM)
bad=0
for r in "${READS[@]}"; do
  IFS='|' read -r name key form <<<"$r"
  ok=0
  for bs in "$BS" $((BS / 2)); do
    say "start $name (${M[$key]}, $form, bs $bs)"
    t0=$SECONDS
    if python3 "${PG_PY[@]}" --tag "$name" --model "${M[$key]}" --form "$form" --data-dir "$DATA" \
         --out "$PG/$name.json" --batch-size "$bs" --device "$DEVICE" --seed 0 >>"$LOGD/$name.log" 2>&1; then
      say "done $name in $((SECONDS - t0)) s: $(grep '^\[pg\]' "$LOGD/$name.log" | tail -1)"; ok=1; break
    fi
    say "FAILED $name at bs $bs: $(tail -n 2 "$LOGD/$name.log" | tr '\n' ' ')"
  done
  ((ok)) || bad=1
done

# ---- step 3: gate
say "gate: n_pairs > 0, control check, shared control answers"
if python3 - "$PG" "${READS[@]}" <<'PY' 2>&1 | tee -a "$LOG"; then gate=ok; else gate=failed; fi
import json, sys
from pathlib import Path
pg, names = Path(sys.argv[1]), [r.split("|")[0] for r in sys.argv[2:]]
bad, rows, shas = [], {}, set()
for n in names:
    p = pg / f"{n}.json"
    if not p.is_file():
        bad.append(f"{n}: missing")
        continue
    r = json.loads(p.read_text())
    c = r["control_check"]
    rows[n] = {"n_pairs": r["n_pairs"], "control_check_pass": c["pass"], "acc": r["acc"],
               "acc_ci": r["acc_ci"], "call": r["call"]}
    shas.add(c["sha256"])
    if not r["n_pairs"] > 0:
        bad.append(f"{n}: n_pairs {r['n_pairs']}")
    if not c["pass"]:
        bad.append(f"{n}: control check failed {c}")
    print(f"  {n:12s} acc* {r['acc']:.3f} [{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}] "
          f"{r['call']:6s} pairs {r['n_pairs']} control {'pass' if c['pass'] else 'FAIL'}")
if len(shas) > 1:
    bad.append(f"control answers differ across reads: {sorted(shas)}")
(pg / "gate.json").write_text(json.dumps({"rows": rows, "failures": bad}, indent=1))
print("gate failures:", bad or "none")
sys.exit(1 if bad else 0)
PY

# ---- step 4: report over the whole tree
if [[ $gate == ok && $bad == 0 ]]; then
  if python3 ts1b_hard_acc_report.py --root "$ROOT" --out-json "$ROOT/report.json" \
       --out-md "$ROOT/report.md" >"$LOGD/report.log" 2>&1; then
    say "MILESTONE report written"
  else
    say "report has gaps or failed (see $LOGD/report.log): $(grep -m1 MISSING "$LOGD/report.log")"; bad=1
  fi
else
  bad=1; say "GATE FAILED: report skipped"
fi

# ---- step 5: push pg/ (+ report if written), always, so a failed run can be read
if [[ -z ${NO_PUSH:-} ]]; then
  cp "$LOG" "$PG/run_log.txt"
  if python3 - "$ROOT" "$RESULTS_REPO" "$SUB" <<'PY' 2>&1 | tee -a "$LOG"; then say "MILESTONE pushed"; else say "FAILED push"; bad=1; fi
import sys
from fnmatch import fnmatch
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
allow, ign = ["pg/*", "report.json", "report.md"], ["pg/run.log", "pg/.fake_pg.py"]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub, allow_patterns=allow,
                  ignore_patterns=ign, commit_message="ts1b parents: pairing gap (PG) reads + report")
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + r for r in (str(p.relative_to(res)) for p in res.rglob("*") if p.is_file())
         if any(fnmatch(r, g) for g in allow) and not any(fnmatch(r, g) for g in ign)}
print("verify: missing on hub:", sorted(local - remote) or "none", "| files:", len(local))
sys.exit(1 if local - remote else 0)
PY
fi
say "finished, bad=$bad"
echo "EXIT=$bad" >>"$LOG"
