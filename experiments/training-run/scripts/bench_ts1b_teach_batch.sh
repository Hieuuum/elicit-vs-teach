#!/usr/bin/env bash
# bench_ts1b_teach_batch.sh — short taught-child runs to decide (owner
# 2026-11-07) whether torch.compile and a bigger batch train faster without
# learning worse. Box script, ~45 min on one L40S.
#
# Each run starts from evt-ts1b-fmt-parent on the taught config, trains on the
# same EXAMPLES (default 256k), and checks every 64k examples. Compare:
#   - speed: median step time, examples/s, wall time;
#   - learning: val loss at equal examples seen (does a bigger batch learn
#     less per example?) and at equal wall time.
# Bench runs land in runs/tb-*; nothing is pushed.
#
# Usage: bash bench_ts1b_teach_batch.sh --confirm-cost [tag:batch:lr:compile ...]
set -uo pipefail
cd "$(dirname "$0")"
echo "[tbench] estimated cost: ~45 min on the current box (~\$0.35 at \$0.45/h)."
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "bench_ts1b_teach_batch.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
EXAMPLES=${EXAMPLES:-256000}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-64000}
PARENT=$GEODE_STORE/runs/evt-ts1b-fmt-parent/model
[[ -d $PARENT ]] || { echo "need the taught parent at $PARENT"; exit 1; }
SPECS=()
for a in "$@"; do [[ $a == *:*:*:* ]] && SPECS+=("$a"); done
((${#SPECS[@]})) || SPECS=(c128:128:2e-5:1 b256:256:2e-5:1 b256lr4:256:4e-5:1 b512lr4:512:4e-5:1)
OUT=/workspace/tbench
mkdir -p "$OUT"

summarize() {
  python3 - "$GEODE_STORE/runs" "$OUT" "$EVAL_EXAMPLES" "${SPECS[@]}" <<'EOF'
import json, statistics, sys
from pathlib import Path
runs, out, every = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
print(f"[tbench] {'run':9} {'batch':>5} {'lr':>6} {'comp':>4} {'s/step':>7} {'ex/s':>6} "
      f"{'wall_min':>8} {'peakGB':>6}  val_nats at 64k/128k/192k/256k examples   EM_end")
for spec in sys.argv[4:]:
    tag, batch, lr, comp = spec.split(":")
    d = runs / f"tb-{tag}"
    try:
        tr = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()]
        ev = [json.loads(l) for l in (d / "eval_log.jsonl").read_text().splitlines()]
    except OSError:
        print(f"[tbench] {tag:9} no logs (see {out}/tb-{tag}.log)")
        continue
    if len(tr) < 3:
        print(f"[tbench] {tag:9} only {len(tr)} steps (see {out}/tb-{tag}.log)")
        continue
    b = int(batch)
    iv = [y["time_unix"] - x["time_unix"] for x, y in zip(tr, tr[1:])
          if (y["step"] * b) % every >= b]  # skip intervals that hold a check
    step_s = statistics.median(iv[len(iv) // 10:])  # skip warm-up / compile
    wall = (tr[-1]["time_unix"] - tr[0]["time_unix"]) / 60
    mem = [int(l.split(",")[0]) for l in (out / f"mem.{tag}").read_text().split("\n") if l.strip()]
    vals = " ".join(f"{e['val_loss_nats']:.3f}" for e in ev)
    print(f"[tbench] {tag:9} {b:5d} {lr:>6} {comp:>4} {step_s:7.3f} {b / step_s:6.0f} "
          f"{wall:8.1f} {max(mem) / 1024:6.1f}  {vals:38}  {ev[-1].get('format_valid_rate', float('nan')):.3f}")
EOF
}

for spec in "${SPECS[@]}"; do
  IFS=: read -r tag batch lr comp <<<"$spec"
  rid=tb-$tag
  rm -rf "$GEODE_STORE/runs/$rid"
  # lr as 2.0e-05: YAML reads a bare 2e-5 as a string.
  printf 'run_id: %s\ntrain:\n  batch_size: %s\n  lr: %s\n  max_steps: %s\n  eval_every: %s\n  compile: %s\n' \
    "$rid" "$batch" "$(LC_ALL=C printf '%.1e' "$lr")" $((EXAMPLES / batch)) $((EVAL_EXAMPLES / batch)) \
    "$([[ $comp == 1 ]] && echo true || echo false)" >"$OUT/$rid.yaml"
  (while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done) >"$OUT/mem.$tag" &
  W=$!
  echo "[tbench] start $spec"
  python3 train_sft.py --config ../configs/ts1b_teach_ft_fresh.yaml --override "$OUT/$rid.yaml" \
    --init-from "$PARENT" --confirm-cost >"$OUT/$rid.log" 2>&1
  echo "[tbench] $rid exit=$?"
  kill $W
  summarize | tail -n +1
done
echo "TBENCH_DONE"
