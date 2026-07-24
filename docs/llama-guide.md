# Runs 9–10 (Llama-3.2-1B external-validity chain) — box paste sheet

Elicitation on the real pretrained model (owner 2026-07-23, decisions.md):
run 9 = LoRA format install on `D_inst` (behavioral stop), merge to a
plain checkpoint, run 10 = LoRA target on the full 1M `D_target` through
the EDL harness (bf16 update forward, fp32-measured losses — V5.62).
Configs: `run9_llama1b_inst.yaml` / `run10_llama1b_target.yaml` — both
ship `lr: null` until the **runs-7/8 1M sweep winner** is pinned into
them: **one LR everywhere** (owner 2026-07-24) — the per-stage Llama
sweeps are dropped (`pilot/llama{9,10}_sweep_lr*` kept only as history;
fallback: if run 9's gates fail at the shared LR, revive the gentle
installer sweep for run 9 only). Sequencing: this chain waits until
extraction + runs 7/8 are done (one-box plan), or gets its own box.

Unattended alternative (owner 2026-07-24): `./launch_chain_7_10.sh
--confirm-cost [--push-and-prune]` runs 7 → 8 → 9 → 10 end to end —
smokes, G4/G5 evidence, merge, ntfy pings, completed runs skipped on
re-run. This guide stays the manual path and the reference for what the
chain script does at each step.

## 0. Prerequisites

- HF account that has **accepted the Meta Llama license** for
  `meta-llama/Llama-3.2-1B` (gated repo) — 401/403 on download means the
  token's account hasn't.
- Disk: **~30 GB free covers the whole chain** (`df -h /workspace`) —
  run 10 stores NO snapshots (owner 2026-07-24: no Llama extraction is
  planned; the external-validity claim is behavioral EDL, and the loss
  curves live in per-step `train_log.jsonl` + dense-curve
  `eval_log.jsonl`). Run 9 is one final checkpoint (~5 GB wrapped +
  ~5 GB merged), run 10 one wrapped checkpoint (~5 GB), plus the hub
  cache.
- Laptop pushed, box hash matches (see run7-8-guide.md §1); exports set
  in every tmux window (`GEODE_STORE`, `NTFY`).

```bash
hf auth login --force  # token whose account holds the Meta license;
                       # --force always (owner 2026-07-24) so a stale login
                       # never masks the wrong account. The same login serves
                       # relay pulls (READ) — never store a WRITE token in
                       # the env
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
```

## 1. Tokenizer verification — must PASS before anything trains

```bash
python3 verify_llama_tokenizer.py
```
Checks pad/EOS handling, the `len(tokenizer)`↔`config.vocab_size` launch
guard, and `tokenize_with_spans` over samples of the frozen parquets —
Llama chunks digits up to 3 per token and its BPE merges differ from the
custom tokenizer the span checks were tuned on. Any failure: stop, bring
the output back to the owner. Do not patch around a span error.

## 2. Run 9 — smoke, launch, gate, merge

LR: the runs-7/8 sweep winner, already pinned in the config before this
chain starts (one LR everywhere, owner 2026-07-24 — no installer sweep).
If G4 fails or zero-shot arithmetic degrades at that LR, stop and revive
the gentle installer sweep (`pilot/llama9_sweep_lr*`) for run 9 only.

```bash
# memory + plumbing (~1 min, disposable)
python3 train_sft.py --config ../configs/run9_llama1b_inst.yaml \
    --override ../configs/pilot/llama9_smoke.yaml \
    --init-from meta-llama/Llama-3.2-1B --confirm-cost
```

```bash
python3 train_sft.py --config ../configs/run9_llama1b_inst.yaml \
    --init-from meta-llama/Llama-3.2-1B --confirm-cost \
  ; curl -d "run9 llama install done (exit $?)" $NTFY

python3 gates.py g4 --run evt-run9-llama1b-inst \
    --config ../configs/run9_llama1b_inst.yaml --device cuda

# merge the install adapter into a plain checkpoint — run 10's parent.
# NEVER point run 10 at model/ (wrapped); only model_merged/ is loadable
# by plain from_pretrained.
python3 merge_adapter.py --run-id evt-run9-llama1b-inst
ls $GEODE_STORE/runs/evt-run9-llama1b-inst/model_merged/
```

## 3. Run 10 — smoke, launch, evidence

LR: the same runs-7/8 winner, pinned in the config (no target sweep —
one LR everywhere, owner 2026-07-24).

```bash
# harness at 1.24B, batch 128 (bf16 update forward, fp32 measurement
# forwards) — the memory worst case (~2 min). OOM here = STOP and ask
# the owner; a batch/precision change is a protocol change.
python3 train_target.py --config ../configs/run10_llama1b_target.yaml \
    --override ../configs/pilot/llama10_smoke.yaml \
    --init-from $GEODE_STORE/runs/evt-run9-llama1b-inst/model_merged --confirm-cost
```
Then:

```bash
python3 train_target.py --config ../configs/run10_llama1b_target.yaml \
    --init-from $GEODE_STORE/runs/evt-run9-llama1b-inst/model_merged --confirm-cost \
  ; curl -d "run10 llama target done (exit $?)" $NTFY

python3 gates.py g5 --run evt-run10-llama1b-target \
    --config ../configs/eval_target_data.yaml --device cuda
```
Also record zero-shot op add/sub on the merged parent BEFORE run 10
(evidence): real Llama may already answer op-notation add/sub — a
near-zero-EDL elicitation is the expected finding there, not a bug.

## 4. Archive + teardown

Small-artifact push (logs + manifest only, ~MBs — the recipe formerly in
run5-6-guide.md §6, deleted 2026-07-24). `hf_checkpoint.py push` uploads
the whole run dir including weights — don't use it for this. Two separate
pastes (the `read` must be alone):

```bash
read -rsp "HF WRITE token: " HF_WRITE_TOKEN && export HF_WRITE_TOKEN && echo " ok"
```
```bash
python3 - <<'EOF'
import fnmatch, os
from pathlib import Path
from huggingface_hub import CommitOperationAdd, HfApi
api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
for rid in ("evt-run9-llama1b-inst", "evt-run10-llama1b-target"):
    src = Path("/workspace/elicit-vs-teach/geode-store/runs") / rid
    keep = [p.relative_to(src).as_posix() for p in sorted(src.rglob("*")) if p.is_file()]
    keep = [r for r in keep
            if any(fnmatch.fnmatch(r, a) for a in ("*.json", "*.jsonl", "*.yaml"))
            and not any(fnmatch.fnmatch(r, i) for i in ("snapshots/*", "model/*", "model_merged/*"))]
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

The loss-curve logs (`train_log.jsonl`, `eval_log.jsonl`) ride along (they
match the `*.jsonl` keep-glob). No snapshots exist for these runs;
checkpoints stay on-box until the owner's relay decision.
Destroy (never stop) when cleared; store lives inside the clone — never
`git clean -dfx`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 401/403 downloading the model | token's account hasn't accepted the Meta license — fix on HF, re-login |
| `verify_llama_tokenizer` span ValueError | tokenizer merge boundary the span code rejects — stop, report to owner |
| arch-mismatch refusal at load | wrong checkpoint dir, or vocab guard: `len(tokenizer)` must equal config vocab (128256) |
| CUDA OOM in a smoke | stop, ask owner — batch/precision are protocol, not knobs |
| run 10 refuses parent | run 9 incomplete, G4 unrecorded, or `--init-from` points at `model/` instead of `model_merged/` |
| launcher refuses `lr: null` | the sweep pin hasn't been made/pulled — §2/§3 pin steps |
