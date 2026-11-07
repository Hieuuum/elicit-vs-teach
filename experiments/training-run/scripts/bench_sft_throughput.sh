#!/usr/bin/env bash
# bench_sft_throughput.sh — measure the V5.79 trainer speedups on a box that
# already ran evt-ts1b-elicit-ft-fresh (old trainer). Box script, ~10-15 min.
#
#   A. one 600-step run of the elicit config with the current trainer:
#      same parent, data and seed as the logged elicit run, so the losses
#      should track it step for step and the speed is directly comparable.
#   B. two such runs at once on the same GPU: does a second job use the
#      GPU time the first one leaves idle?
#
# Prints steps/s (old vs new), loss agreement, and peak GPU memory.
# Bench runs land in runs/bench-*; nothing is pushed.
#
# Usage: bash bench_sft_throughput.sh --confirm-cost
set -uo pipefail
cd "$(dirname "$0")"
echo "[bench] estimated cost: ~15 min on the current box (~\$0.15 at \$0.5/h)."
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "bench_sft_throughput.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
STEPS=${STEPS:-600}
OLD=evt-ts1b-elicit-ft-fresh
PARENT=$GEODE_STORE/runs/evt-ts1b-op-bridge-mix/model
[[ -f $GEODE_STORE/runs/$OLD/train_log.jsonl ]] || { echo "need runs/$OLD (old-trainer log)"; exit 1; }
[[ -d $PARENT ]] || { echo "need the elicit parent at $PARENT"; exit 1; }
TMP=$(mktemp -d)

make_cfg() { # run_id -> config path (same as the elicit child; no stop within STEPS)
  local cfg=$TMP/$1.yaml
  sed -e "s/^run_id: .*/run_id: $1/" -e "s/^  max_steps: .*/  max_steps: $STEPS/" \
    ../configs/ts1b_elicit_ft_fresh.yaml >"$cfg"
  echo "$cfg"
}
run() { # run_id
  rm -rf "$GEODE_STORE/runs/$1"
  python3 train_sft.py --config "$(make_cfg "$1")" --init-from "$PARENT" --confirm-cost \
    >"$TMP/$1.log" 2>&1
  echo "[bench] $1 exit=$?"
}
watch_mem() { # samples peak GPU memory into $TMP/mem until killed
  while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 2; done >"$TMP/mem.$1"
}

echo "[bench] A: one run"
watch_mem A & W=$!
run bench-a
kill $W
echo "[bench] B: two runs at once"
watch_mem B & W=$!
run bench-b1 & P1=$!
run bench-b2 & P2=$!
wait $P1 $P2
kill $W

python3 - "$GEODE_STORE/runs" "$TMP" "$STEPS" <<'EOF'
import json, statistics, sys
from pathlib import Path
runs, tmp, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])

def log(rid):
    return [json.loads(l) for l in (runs / rid / "train_log.jsonl").read_text().splitlines()]

def rate(recs, lo=100):
    """steps/s from step lo to the last step (includes the same checks)."""
    by = {r["step"]: r["time_unix"] for r in recs}
    hi = min(max(by), steps)
    return (hi - lo) / (by[hi] - by[lo])

old = log("evt-ts1b-elicit-ft-fresh")[:steps]
for tag, rids in (("A one run", ["bench-a"]), ("B two at once", ["bench-b1", "bench-b2"])):
    try:
        recs = [log(r) for r in rids]
        assert all(len(r) >= steps for r in recs)
    except (OSError, AssertionError):
        print(f"[bench] {tag}: FAILED to finish {steps} steps (see {tmp}/<run>.log; OOM?)")
        continue
    rates = [rate(r) for r in recs]
    mem = [l.split(",") for l in (tmp / f"mem.{tag[0]}").read_text().split("\n") if l.strip()]
    peak = max(int(m[0]) for m in mem)
    util = statistics.mean(int(m[1]) for m in mem)
    print(f"[bench] {tag}: steps/s per run {[round(x, 2) for x in rates]}, total {sum(rates):.2f} "
          f"(old trainer {rate(old):.2f}); peak GPU mem {peak} MiB; mean util {util:.0f}%")
new = log("bench-a")
d = [abs(a["train_loss_nats"] - b["train_loss_nats"]) for a, b in zip(old, new)]
print(f"[bench] loss vs old run: max |diff| first 20 steps {max(d[:20]):.4f}, "
      f"mean |diff| all {statistics.mean(d):.4f}; old/new mean loss steps 500-600: "
      f"{statistics.mean(r['train_loss_nats'] for r in old[500:]):.4f} / "
      f"{statistics.mean(r['train_loss_nats'] for r in new[500:]):.4f}")
for rid in ("evt-ts1b-elicit-ft-fresh", "bench-a"):
    ev = [json.loads(l) for l in (runs / rid / "eval_log.jsonl").read_text().splitlines()][:steps // 100]
    print(f"[bench] {rid} checks: " + ", ".join(f"{e['step']}:{e['format_valid_rate']:.3f}" for e in ev))
EOF
echo "BENCH_DONE"
