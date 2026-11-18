"""Upload the relearning children (each unlearned parent LoRA-fine-tuned on bio_A as text facts,
stages_wmdp.sh stage 2) to the Hub, one PRIVATE model repo per child, loadable with from_pretrained.

These children are safeguarded models with WMDP-bio knowledge restored, so the repos are always created
private; there is no public switch. Do not flip them public on the Hub either.

Per child ``$GEODE_STORE/runs/<prefix>-relearn-<parent>-<kind>``, it uploads model/ (merged bf16
weights + adapter.safetensors + tokenizer) and manifest.json, to ``<namespace>/<run id>``;
``--adapter-only`` uploads adapter.safetensors + manifest only (the base is the parent repo in it).

  python3 experiments/unlearning/push_relearned.py --dry-run          # list what would go up
  python3 experiments/unlearning/push_relearned.py                    # bio_A children, all model sets
  python3 experiments/unlearning/push_relearned.py --kind bioA mmluA  # also the far-domain null children
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PREFIXES = ("wmdp", "wmdp-tar", "wmdp-deepig")  # stages_wmdp.sh RP=wmdp$SETSUF per model set


def children(runs: Path, kinds: list[str]) -> list[Path]:
    out = []
    for d in sorted(runs.glob("wmdp*-relearn-*")):
        prefix, rest = d.name.split("-relearn-", 1)
        if (
            prefix in PREFIXES
            and any(rest.endswith(f"-{k}") for k in kinds)
            and (d / "model").is_dir()
        ):
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--namespace", default="podhajskimarcin")
    ap.add_argument("--kind", nargs="+", default=["bioA"])
    ap.add_argument("--adapter-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    store = os.environ.get("GEODE_STORE")
    if not store:
        raise SystemExit("[push] GEODE_STORE is not set")
    found = children(Path(store) / "runs", args.kind)
    if not found:
        raise SystemExit(f"[push] no relearning children for {args.kind} under {store}/runs")
    from huggingface_hub import HfApi

    api = None if args.dry_run else HfApi()
    for d in found:
        man = (
            json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").is_file() else {}
        )
        repo = f"{args.namespace}/{d.name}"
        size = sum(f.stat().st_size for f in (d / "model").rglob("*") if f.is_file()) / 1e9
        print(
            f"[push] {d.name}: parent {man.get('init', '?')}, model/ {size:.1f} GB -> {repo} (private)"
        )
        if args.dry_run:
            continue
        api.create_repo(repo, private=True, exist_ok=True)
        if api.repo_info(repo).private is not True:
            raise SystemExit(f"[push] {repo} exists and is public; refusing to upload into it")
        api.upload_folder(
            repo_id=repo,
            folder_path=d / "model",
            commit_message=f"{d.name} relearned child",
            allow_patterns=["adapter.safetensors"] if args.adapter_only else None,
        )
        if (d / "manifest.json").is_file():
            api.upload_file(
                repo_id=repo, path_or_fileobj=d / "manifest.json", path_in_repo="manifest.json"
            )
        print(f"[push]   https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
