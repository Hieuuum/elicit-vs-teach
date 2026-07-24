# Runs 5–6 (LoRA targets) — box paste sheet

Paste top to bottom on a fresh rented box. Box needs **≥150 GB free disk**
(adapter-only snapshots ≈ 48 MB × up to 1024 steps per run) and one RTX 4090.
Configs: `run5_target.yaml` / `run6_target.yaml` (LR 1e-3, n 500K, ceiling
23,442 steps — the ε/k rule is the real stop).

## 0. Laptop first — push, or the box clones stale code

```bash
cd ~/Github/geode && git add -A && git commit -m "..." ; git push origin cut-to-core
git rev-parse --short HEAD    # note this hash — the box must match it
```

## 1. Box setup

(Or put `scripts/box_onstart.sh` in the vast.ai template's on-start field —
it does this section idempotently: clone + install + suite + bashrc exports.
It never launches training and never auto-pulls; §0's hash check stays yours.)

```bash
nvidia-smi; python --version; df -h /workspace | tail -1
tmux new -s train
cd /workspace
git clone -b cut-to-core https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach && git log --oneline -1     # must equal the laptop hash
pip install -e ".[dev]"
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store      # store INSIDE the clone
export NTFY=ntfy.sh/<your-topic>               # re-export BOTH in every tmux window
python -m pytest -q; echo "suite exit: $?"     # expect 0, ~2 min
cd experiments/training-run/scripts
```

## 2. Parent checkpoints (private relay — read token is enough)

```bash
hf auth login                                  # READ token, never write
python hf_checkpoint.py pull --run-id evt-run3-armA-inst
python hf_checkpoint.py pull --run-id evt-run4-armB-inst
ls $GEODE_STORE/runs/evt-run{3,4}-arm*-inst/model/   # config.json + model.safetensors
```

## 3. Dry run (free — no `--confirm-cost`, must end in a refusal)

```bash
python train_target.py --config ../configs/run5_target.yaml \
    --init-from $GEODE_STORE/runs/evt-run3-armA-inst/model
```
Expect: `order_hash verified` → parent gates pass → cost estimate →
`refusing to train (budget rule)`. Exit 1 here is correct.

## 4. Run 5 (Arm A), then run 6 (Arm B) — sequential; G7 needs run 5 registered

```bash
python train_target.py --config ../configs/run5_target.yaml \
    --init-from $GEODE_STORE/runs/evt-run3-armA-inst/model --confirm-cost \
  ; curl -d "run5 armA done (exit $?)" $NTFY

python train_target.py --config ../configs/run6_target.yaml \
    --init-from $GEODE_STORE/runs/evt-run4-armB-inst/model --confirm-cost \
  ; curl -d "run6 armB done (exit $?)" $NTFY
```
Watch from a second window: `python monitor.py --run-id evt-run5-armA-target`.
Reference stops: run 5 @ 6,000 (min_val 0.00245), run 6 @ 12,500 (0.02301);
`stop_reason=max_steps` = the rule never fired = investigate, not a result.

## 5. G5 evidence (both runs, minutes)

```bash
for r in evt-run5-armA-target evt-run6-armB-target; do
  python gates.py g5 --run $r --config ../configs/eval_target_data.yaml --device cuda
done ; curl -d "G5 recorded (exit $?)" $NTFY
```
Reference: A 0.9980 zero-shot / θ_T 0.00194 nats, B 0.9502 / 0.03558.
16-shot ≈ 0 both arms is expected (metric invalidated, decisions.md).

## 6. Archive the small artifacts (NOT the snapshots)

`hf_checkpoint.py push` uploads the whole folder incl. ~75 GB of snapshots —
don't use it here. Two separate pastes (the `read` must be alone):

```bash
read -rsp "HF WRITE token: " HF_WRITE_TOKEN && export HF_WRITE_TOKEN && echo " ok"
```
```bash
python - <<'EOF'
import fnmatch, os
from pathlib import Path
from huggingface_hub import CommitOperationAdd, HfApi
api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
for rid in ("evt-run5-armA-target", "evt-run6-armB-target"):
    src = Path("/workspace/elicit-vs-teach/geode-store/runs") / rid
    keep = [p.relative_to(src).as_posix() for p in sorted(src.rglob("*")) if p.is_file()]
    keep = [r for r in keep
            if any(fnmatch.fnmatch(r, a) for a in ("*.json", "*.jsonl", "*.yaml"))
            and not any(fnmatch.fnmatch(r, i) for i in ("snapshots/*", "model/*"))]
    total = sum((src / r).stat().st_size for r in keep)
    print(rid, len(keep), "files", f"{total/1e6:.1f} MB")
    assert total < 200e6, "guard tripped — check the list, NOT uploading"
    api.create_commit(repo_id="mhieuuu/geode-store",
        operations=[CommitOperationAdd(f"runs/{rid}/{r}", str(src / r)) for r in keep],
        commit_message=f"{rid}: logs+manifest only")
print("DONE")
EOF
unset HF_WRITE_TOKEN
```
Laptop: `python hf_checkpoint.py pull --run-id <each> --no-weights`.

## 7. Tear down — only after extraction

Snapshots exist **only on this box**. Keep it alive until `extract.py` has run
over them (or they're relayed); then destroy the instance on vast.ai — stopped
still bills storage. Store lives inside the clone now — never `git clean -dfx`
on this box, it deletes run artifacts too.

## Troubleshooting

| Symptom | Fix |
|---|---|
| box `git log` ≠ laptop hash | forgot to push, or clone predates it — `git pull` |
| `parent run ... has no manifest` | stale `$GEODE_STORE` (new tmux window) — re-export, re-pull |
| `order_hash mismatch` | downloaded parquet ≠ frozen file — stop, don't work around |
| `register_run: already running` | double-launch guard — `pgrep -f train_target` |
| `data_order_hash` refusal on run 6 | run 5 not registered yet, or n_examples differs (G7) |
| disk full mid-run | snapshots — needs ≥150 GB free, check `df -h /workspace` |
| HF push 401/403 | read token in a write path — see §6 |
