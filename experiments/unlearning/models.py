"""The models of the unlearning experiment: Hub ids, pinned revisions, roles (PLAN.md).

WMDP (PRIMARY design, --dataset wmdp): Zephyr-7B-beta (Mistral architecture,
32 layers, 32 heads / 8 KV, width 4096, vocab 32000, 7.24B params) and
unlearned versions of it. Roles: ``reference`` = the ORIGINAL model (the
circuit the capability runs on; not a control), ``unlearned`` = objects under
test. No never-learned model by design (owner, 2026-09-25). All revisions are
the commit shas measured on 2026-09-25. Every WMDP model is given the
ORIGINAL's tokenizer files at fetch time (several unlearned repos ship only a
partial tokenizer; one shared tokenizer keeps every model's items and pairs
token-identical).

TOFU (SECONDARY, controlled design, --dataset tofu): Llama-3.2-1B-Instruct
fine-tunes from open-unlearning; roles elicit_anchor / teach_anchor /
unlearned.

Usage (cluster, stage 0 of launch_unlearn.sh):
  python3 models.py fetch --dataset wmdp --dest $GEODE_STORE/unlearning/wmdp/models [--tags orig rmu ...]
  python3 models.py table --dataset wmdp
  python3 models.py role --dataset wmdp rmu
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ORG = "open-unlearning"
TOFU_MODELS: dict[str, dict] = {
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
WMDP_MODELS: dict[str, dict] = {
    "orig": {"role": "reference", "repo": "HuggingFaceH4/zephyr-7b-beta",
             "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c", "licence": "MIT",
             "note": "the pre-unlearning model every unlearned model below starts from"},
    "rmu": {"role": "unlearned", "repo": "cais/Zephyr_RMU",
            "revision": "70c55b3bf3141a8c24292dec0262b8aea03a0d4a", "licence": "MIT",
            "domains": ["bio", "cyber"], "note": "WMDP paper (Li et al. 2024), RMU"},
    "elm": {"role": "unlearned", "repo": "baulab/elm-zephyr-7b-beta",
            "revision": "90b9a5ea4b04712ddeec00cd4ee76a3d548774d5", "licence": "none stated",
            "domains": ["bio", "cyber"], "peft_base": "orig",
            "note": "Gandikota et al. 2024 ELM; a PEFT LoRA adapter (r=4, layers 4-7) merged onto orig"},
    "npo": {"role": "unlearned", "repo": "OPTML-Group/NPO-WMDP",
            "revision": "fec7960fb3925037a6befc5819036cbc55664b18", "licence": "MIT",
            "domains": ["bio"], "note": "OPTML-Group, NPO on wmdp-bio only"},
    "graddiff": {"role": "unlearned", "repo": "OPTML-Group/GradDiff-WMDP",
                 "revision": "0695dccbb30a0986d20f211bc8bdea9d55ec5cc6", "licence": "MIT",
                 "domains": ["bio"], "note": "OPTML-Group, GradDiff on wmdp-bio only"},
    "simnpo": {"role": "unlearned", "repo": "OPTML-Group/SimNPO-WMDP-zephyr-7b-beta",
               "revision": "feb05eaff664a9a3e9c49ca6084d4d8d25a005a4", "licence": "MIT",
               "domains": ["bio", "cyber"], "note": "Fan et al. 2024 SimNPO"},
    "rmulat": {"role": "unlearned", "repo": "LLM-LAT/zephyr7b-beta-rmu-lat-unlearn-wmdp-bio-cyber",
               "revision": "4ff066c804d0ab025698bb958c959c46642a1178", "licence": "none stated",
               "domains": ["bio", "cyber"], "note": "Sheshadri et al. 2024 RMU + latent adversarial training"},
}
TABLES = {"wmdp": WMDP_MODELS, "tofu": TOFU_MODELS}
DEFAULT_TAGS = {"wmdp": ("orig", "rmu", "elm", "npo", "simnpo"),
                "tofu": ("orig", "retain", "npo", "graddiff", "rmu", "simnpo")}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json")
MODELS = TOFU_MODELS   # back-compat alias (TOFU code paths)


def role(tag: str, dataset: str = "tofu") -> str:
    """Role of a tag; unknown (smoke) tags are unlearned."""
    return TABLES[dataset].get(tag, {}).get("role", "unlearned")


def merge_peft_adapter(base_dir: Path, adapter_dir: Path, out: Path) -> int:
    """Merge a PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors)
    into a copy of the base checkpoint: W += (alpha / r) * B @ A, in fp32, saved
    in the base's dtype. No peft dependency."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    ad = load_file(adapter_dir / "adapter_model.safetensors")
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    params = dict(model.named_parameters())
    n = 0
    for k, a in ad.items():
        if ".lora_A." not in k:
            continue
        b = ad[k.replace(".lora_A.", ".lora_B.")]
        name = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
        if name not in params:
            raise SystemExit(f"[models] adapter key {k} -> {name}: no such parameter on the base")
        with torch.no_grad():
            params[name].add_(scaling * (b.float() @ a.float()))
        n += 1
    dtype = getattr(torch, json.loads((base_dir / "config.json").read_text()).get("torch_dtype", "bfloat16"))
    model.to(dtype).save_pretrained(out, safe_serialization=True)
    return n


def fetch(dest: Path, tags: list[str], dataset: str) -> None:
    from huggingface_hub import snapshot_download

    table = TABLES[dataset]
    order = sorted(tags, key=lambda t: t != "orig")      # orig first: others borrow its tokenizer / base
    for tag in order:
        m = table[tag]
        out = dest / tag
        marker = out / ".revision"
        if marker.is_file() and marker.read_text().strip() == m["revision"]:
            print(f"[models] {tag}: present at {m['revision'][:12]} ({out})")
            continue
        print(f"[models] {tag}: downloading {m['repo']}@{m['revision'][:12]} -> {out}", flush=True)
        if m.get("peft_base"):
            raw = dest / f"{tag}_adapter"
            snapshot_download(m["repo"], revision=m["revision"], local_dir=raw,
                              allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
            n = merge_peft_adapter(dest / m["peft_base"], raw, out)
            print(f"[models] {tag}: merged {n} LoRA pairs onto {m['peft_base']}")
        else:
            snapshot_download(m["repo"], revision=m["revision"], local_dir=out,
                              allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
        if dataset == "wmdp" and tag != "orig":
            for f in TOKENIZER_FILES:            # one shared tokenizer (module docstring)
                src = dest / "orig" / f
                if src.is_file():
                    shutil.copyfile(src, out / f)
        marker.write_text(m["revision"] + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--dataset", choices=tuple(TABLES), default="wmdp")
    f.add_argument("--tags", nargs="+", default=None)
    t = sub.add_parser("table")
    t.add_argument("--dataset", choices=tuple(TABLES), default="wmdp")
    r = sub.add_parser("role")
    r.add_argument("--dataset", choices=tuple(TABLES), default="wmdp")
    r.add_argument("tag")
    args = ap.parse_args()
    if args.cmd == "fetch":
        tags = args.tags or list(DEFAULT_TAGS[args.dataset])
        if args.dataset == "wmdp" and "orig" not in tags:
            tags = ["orig"] + tags
        fetch(args.dest, tags, args.dataset)
    elif args.cmd == "role":
        print(role(args.tag, args.dataset))
    else:
        print(json.dumps(TABLES[args.dataset], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
