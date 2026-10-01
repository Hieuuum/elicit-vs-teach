#!/usr/bin/env bash
# Archive the main elicit-vs-teach quartet (TinyStories-1B) to per-run Hugging Face repos
# $HF_NAMESPACE/<run_id>, the convention of launch_fig2ts_llama.sh (owner 2026-08-14: the owner's
# account, public; the teammate's mhieuuu/geode-store relay stays untouched). Run ON THE CLUSTER,
# where the weights live, in the geode env, with a WRITE-scoped token in $HF_WRITE_TOKEN or $HF_TOKEN.
#
#   Elicit parent            evt-ts1b-op-bridge-mix          (latent capability, op-notation bridge)
#   Elicited child           evt-ts1b-elicit-ft-n4000000     (full FT of the elicit parent, n = 4M)
#   Format-installed parent  evt-ts1b-fig2ts-installer       (bare-format dose, random labels)
#   Taught child             evt-ts1b-teach-ft-fmt-n4000000  (full FT of the installer, n = 4M)
#
# Per run: hf_checkpoint.py push (manifest, logs, gates, eval, model/; snapshots/ skipped unless
# WITH_SNAPSHOTS=1; model_merged/ never), then the hub's model.safetensors sha256 is checked against
# the local file, then a README.md model card built from manifest.json is uploaded to the repo root.
# Idempotent: re-running re-uploads only changed chunks. Nothing local is deleted.
#
# Usage:
#   export GEODE_STORE=/raid/.../geode-store HF_TOKEN=hf_...   # write scope
#   bash push_main_pair.sh --dry-run          # sizes, repos and cards; no upload
#   bash push_main_pair.sh                    # upload (public repos)
#   PRIVATE=1 bash push_main_pair.sh          # private repos instead
#   HF_NAMESPACE=someone bash push_main_pair.sh RUN_ID [RUN_ID ...]   # another account / a subset
#   HF_LICENSE=mit bash push_main_pair.sh    # licence tag of the model cards (default: other)
set -uo pipefail
cd "$(dirname "$0")"
export GEODE_STORE=${GEODE_STORE:-$(git rev-parse --show-toplevel)/geode-store}
export PYTHONPATH=$(git rev-parse --show-toplevel)${PYTHONPATH:+:$PYTHONPATH}   # geode.zoo for hf_checkpoint.py
HF_NAMESPACE=${HF_NAMESPACE:-podhajskimarcin}
DRY=0
[[ ${1:-} == --dry-run ]] && { DRY=1; shift; }
if (( DRY == 0 )); then
  export HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?need HF_TOKEN or HF_WRITE_TOKEN (write scope)}}
fi
VIS=--public; [[ ${PRIVATE:-0} == 1 ]] && VIS=""
SNAP=""; [[ ${WITH_SNAPSHOTS:-0} == 1 ]] && SNAP=--with-snapshots
export HF_LICENSE=${HF_LICENSE:-other}

declare -A ROLE=(
  [evt-ts1b-op-bridge-mix]="Elicit parent: TinyStories-1B with the arithmetic capability installed in operator notation (the pre-elicit parent of the main pair)."
  [evt-ts1b-elicit-ft-n4000000]="Elicited child: the elicit parent fully fine-tuned on 4,000,000 natural-language addition/subtraction examples."
  [evt-ts1b-fig2ts-installer]="Format-installed parent: TinyStories-1B given the bare answer format with random labels (format validity 1.000, accuracy 0.000); the pre-teach-format twin."
  [evt-ts1b-teach-ft-fmt-n4000000]="Taught child: the format-installed parent fully fine-tuned on 4,000,000 natural-language addition/subtraction examples with the same recipe."
)
RUNS=("$@"); (( ${#RUNS[@]} )) || RUNS=(evt-ts1b-op-bridge-mix evt-ts1b-elicit-ft-n4000000 evt-ts1b-fig2ts-installer evt-ts1b-teach-ft-fmt-n4000000)

echo "[push] store=$GEODE_STORE namespace=$HF_NAMESPACE visibility=${VIS:-private} snapshots=${WITH_SNAPSHOTS:-0} dry_run=$DRY"
TOTAL=0; FAILED=()
for rid in "${RUNS[@]}"; do
  dir=$GEODE_STORE/runs/$rid
  ckpt=$(python3 -c 'import sys; from hf_checkpoint import find_checkpoint; from pathlib import Path; print(find_checkpoint(Path(sys.argv[1]), sys.argv[2]))' "$GEODE_STORE" "$rid" 2>&1) \
    || { echo "[push] $rid: $ckpt" >&2; FAILED+=("$rid"); continue; }
  mb=$(du -sm "$(dirname "$ckpt")" | cut -f1); TOTAL=$((TOTAL + mb))
  echo "[push] $rid: $ckpt ($mb MB in model/) -> https://huggingface.co/$HF_NAMESPACE/$rid"
done
echo "[push] ~$((TOTAL / 1024)) GB of weights (model/ dirs; logs and manifests are small)"
(( ${#FAILED[@]} )) && { echo "[push] missing checkpoints: ${FAILED[*]}" >&2; exit 2; }

card() {  # card <run_id> <repo_id> -> README.md on stdout, from the run's manifest
  python3 - "$1" "$2" "${ROLE[$1]:-}" "$GEODE_STORE" <<'PY'
import json, os, sys
from pathlib import Path
rid, repo, role, store = sys.argv[1:5]
m = json.loads((Path(store) / "runs" / rid / "manifest.json").read_text())
g = lambda *ks, d=None: (lambda o: o if o is not None else d)(__import__("functools").reduce(lambda o, k: o.get(k) if isinstance(o, dict) else None, ks, m))
ex = m.get("experiment") or {}
rows = [("run id", rid), ("role", role), ("parent run", m.get("parent_run_id") or g("init_from") or "none (pretrained base)"),
        ("base model", g("base_model", "hf_id")), ("regime", m.get("regime")),
        ("dataset", f"{g('dataset', 'name')} (n = {g('dataset', 'n_unique_examples')}, seed {g('dataset', 'seed')})"),
        ("training", f"{g('training', 'method')}" + (f", LoRA r={g('training', 'lora', 'rank')}" if g("training", "lora") else "")),
        ("stop", f"{g('result', 'stop_reason') or ex.get('stop_reason')} at step {g('result', 'final_step') or ex.get('final_step')}"),
        ("test loss (nats/label token)", g("result", "test_loss_per_label_token_nats") or ex.get("test_loss_per_label_token_nats")),
        ("G5 zero-shot exact match", g("experiment", "gates", "G5", "zero_shot") or g("experiment", "gates", "G5", "accuracy")),
        ("git commit", m.get("git_commit")), ("created", m.get("created_utc"))]
lines = ["---", f"license: {os.environ.get('HF_LICENSE', 'other')}", "library_name: transformers", "tags: [elicit-vs-teach, mars-v, geode, tinystories-1b]", "---",
         f"# {rid}", "", role, "",
         "Part of the MARS V project *Mechanistic Understanding of Elicitation vs. Teaching* (the `geode` codebase). "
         "This repo mirrors the run's folder from the project store: `runs/<run_id>/manifest.json`, `train_log.jsonl`, "
         "`eval_log.jsonl`, gate and eval files, and `model/` (the final `save_pretrained` checkpoint). Snapshots are not included.", "",
         "| field | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows if v not in (None, "None", "")] + ["",
         "## Load", "", "```python", "from transformers import AutoModelForCausalLM, AutoTokenizer",
         f'm = AutoModelForCausalLM.from_pretrained("{repo}", subfolder="runs/{rid}/model")',
         f'tok = AutoTokenizer.from_pretrained("{repo}", subfolder="runs/{rid}/model")', "```", "",
         "or, with the project checkout, `python3 experiments/training-run/scripts/hf_checkpoint.py pull "
         f"--run-id {rid} --repo-id {repo}` (verifies the checkpoint's sha256).", ""]
print("\n".join(lines))
PY
}

for rid in "${RUNS[@]}"; do
  repo=$HF_NAMESPACE/$rid
  if (( DRY )); then
    echo "----- README.md for $repo -----"; card "$rid" "$repo" || FAILED+=("$rid card")
    continue
  fi
  echo "[push] uploading $rid -> $repo"
  python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$repo" $VIS $SNAP || { FAILED+=("$rid push"); continue; }
  python3 - "$rid" "$repo" <<'PY' || { FAILED+=("$rid verify"); continue; }
import os, sys
from pathlib import Path
from hf_checkpoint import verify_hub_checkpoint
sha = verify_hub_checkpoint(Path(os.environ["GEODE_STORE"]), sys.argv[1], repo_id=sys.argv[2])
print(f"[push] hub sha256 verified for {sys.argv[1]}: {sha}")
PY
  card "$rid" "$repo" > "/tmp/README_$rid.md" || { FAILED+=("$rid card"); continue; }
  python3 - "$rid" "$repo" <<'PY' || { FAILED+=("$rid readme"); continue; }
import sys
from huggingface_hub import HfApi
rid, repo = sys.argv[1:3]
HfApi().upload_file(path_or_fileobj=f"/tmp/README_{rid}.md", path_in_repo="README.md", repo_id=repo,
                    commit_message=f"{rid}: model card")
print(f"[push] README.md uploaded to https://huggingface.co/{repo}")
PY
done

if (( ${#FAILED[@]} )); then
  echo "[push] FAILED: ${FAILED[*]}" >&2
  exit 1
fi
(( DRY )) && echo "[push] dry run complete; rerun without --dry-run to upload" || echo "[push] all runs archived and verified"
