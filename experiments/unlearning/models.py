"""The parents of the unlearning experiment: Hub ids, pinned revisions, roles (PLAN.md §2).

All are Llama-3.2-1B-Instruct fine-tunes released by open-unlearning (HF org
``open-unlearning``, ungated, bf16 safetensors, 2.47 GB each); revisions are
the commit shas measured on 2026-09-25. The unlearned checkpoints start from
``tofu_Llama-3.2-1B-Instruct_full`` and forget the TOFU forget10 authors;
``retain90`` was fine-tuned on the other 180 authors only and never saw them.

Roles: elicit_anchor (knows the facts), teach_anchor (never learned them),
unlearned (object under test), base (optional: no TOFU at all; gated repo).

Usage (cluster, stage 0 of launch_unlearn.sh):
  python3 models.py fetch --dest $GEODE_STORE/unlearning/models [--tags orig retain npo ...]
  python3 models.py table          # tag, role, repo, revision
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ORG = "open-unlearning"
MODELS: dict[str, dict] = {
    "orig": {"role": "elicit_anchor", "repo": f"{ORG}/tofu_Llama-3.2-1B-Instruct_full",
             "revision": "88e31200b97e4c0c04ae0d2f0b591f427046d192"},
    "retain": {"role": "teach_anchor", "repo": f"{ORG}/tofu_Llama-3.2-1B-Instruct_retain90",
               "revision": "7114300c0049527a71833f5683965c358ad9dcbf"},
    "npo": {"role": "unlearned",
            "repo": f"{ORG}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.05_alpha1_epoch10",
            "revision": "faeb45f30e0a274de442ddeacef6f0ff9e371113"},
    "graddiff": {"role": "unlearned",
                 "repo": f"{ORG}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
                 "revision": "8b2217e8daec3557e91fe5e0f0c5f627cd996e8f"},
    "rmu": {"role": "unlearned",
            "repo": f"{ORG}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer10_scoeff100_epoch10",
            "revision": "6fbd37715710ed527742028629c595a76d96a5ef"},
    "simnpo": {"role": "unlearned",
               "repo": f"{ORG}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b3.5_a1_d0_g0.125_ep10",
               "revision": "77224bb055821d4a992f998525e62560d83a2058"},
    "idkdpo": {"role": "unlearned",
               "repo": f"{ORG}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.05_alpha1_epoch10",
               "revision": "47d05b6f396aac6e306ff960aedc21be6934907b"},
    "undial": {"role": "unlearned",
               "repo": f"{ORG}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha1_epoch10",
               "revision": "a73180cb43d530a6aeca4dc3cd0e2afb83d6f570"},
    "base": {"role": "base", "repo": "meta-llama/Llama-3.2-1B-Instruct", "revision": "main"},
}
DEFAULT_TAGS = ("orig", "retain", "npo", "graddiff", "rmu", "simnpo")


def role(tag: str) -> str:
    return MODELS[tag]["role"] if tag in MODELS else "unlearned"   # smoke tags default to unlearned


def fetch(dest: Path, tags: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for tag in tags:
        m = MODELS[tag]
        out = dest / tag
        marker = out / ".revision"
        if marker.is_file() and marker.read_text().strip() == m["revision"]:
            print(f"[models] {tag}: present at {m['revision'][:12]} ({out})")
            continue
        print(f"[models] {tag}: downloading {m['repo']}@{m['revision'][:12]} -> {out}", flush=True)
        snapshot_download(m["repo"], revision=m["revision"], local_dir=out,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
        marker.write_text(m["revision"] + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    sub.add_parser("table")
    r = sub.add_parser("role")
    r.add_argument("tag")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.dest, args.tags)
    elif args.cmd == "role":
        print(role(args.tag))
    else:
        print(json.dumps(MODELS, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
