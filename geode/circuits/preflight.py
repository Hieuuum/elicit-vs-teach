"""Read-only pre-rental checkpoint/data audit; downloads metadata/tokenizers only.

This module never allocates a model, fetches pretrained weight bytes, or rents
compute. Advertised LFS weight SHA256 values are provenance, not a claim that
the corresponding multi-gigabyte files have been downloaded and verified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .artifacts import fingerprint, sha256_file, utc_now, write_json
from .checkpoints import CHECKPOINTS, Checkpoint
from .data import Example, ethics_controls, load_examples, single_token_labels, verify_data


METADATA_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
)
ARCHITECTURE_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "hidden_act",
    "tie_word_embeddings",
    "attention_bias",
    "attention_dropout",
    "rms_norm_eps",
    "rope_parameters",
    "rope_scaling",
    "rope_theta",
)
EXPECTED_OLMO2_SHAPE = {
    "model_type": "olmo2",
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
}


def metadata_file_names(siblings: Sequence[Any]) -> list[str]:
    """Strict asset allowlist: a model weight file cannot enter the download list."""
    names = []
    for file in siblings:
        name = file.rfilename
        allowed = name in METADATA_FILES or (
            name.startswith("chat_templates/") and name.endswith(".jinja")
        )
        if allowed:
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError("Unsafe metadata asset path")
            if file.size is not None and file.size > 64 * 1024**2:
                raise ValueError(f"Unexpectedly large metadata asset: {name}")
            names.append(name)
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(names):
        raise ValueError("Checkpoint lacks required config/tokenizer JSON assets")
    return sorted(names)


def architecture_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare model structure separately from incidental checkpoint/dtype fields."""
    result = {key: config.get(key) for key in ARCHITECTURE_FIELDS}
    if result["num_key_value_heads"] is None:
        result["num_key_value_heads"] = result["num_attention_heads"]
    return result


def tokenizer_signature(tokenizer: Any) -> dict[str, Any]:
    """Match backend vocabulary/normalizer plus wrapper state relevant to batching."""
    from geode.zoo.activations import tokenizer_hash

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer lacks EOS required for pad=eos policy")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("pad=eos policy did not produce equal token IDs")
    return {
        "backend_sha256": tokenizer_hash(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "vocab_size": len(tokenizer),
        "special_tokens_map": tokenizer.special_tokens_map,
    }


def validate_checkpoint_compatibility(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_shape: Mapping[str, Any] | None = EXPECTED_OLMO2_SHAPE,
) -> dict[str, Any]:
    """Fail on silent architecture/tokenization drift before paid GPU work."""
    if not records:
        raise ValueError("No checkpoint records")
    if len({row["stage"] for row in records}) != len(records):
        raise ValueError("Duplicate checkpoint stages")
    reference = records[0]
    for row in records:
        if (
            not re.fullmatch(r"[0-9a-f]{40}", row["revision"])
            or row["resolved_revision"] != row["revision"]
        ):
            raise ValueError(f"Resolved revision mismatch at {row['stage']}")
        if row["architecture"] != reference["architecture"]:
            differing = [
                key
                for key in ARCHITECTURE_FIELDS
                if row["architecture"].get(key) != reference["architecture"].get(key)
            ]
            raise ValueError(f"Architecture mismatch at {row['stage']}: {differing}")
        if row["tokenizer"] != reference["tokenizer"]:
            raise ValueError(f"Tokenizer backend/special-token/padding mismatch at {row['stage']}")
        for key, value in (expected_shape or {}).items():
            if row["architecture"].get(key) != value:
                raise ValueError(f"Unexpected OLMo2 shape {key} at {row['stage']}")
        if not row.get("weight_files"):
            raise ValueError(f"No advertised model weight files at {row['stage']}")
        for file in row["weight_files"]:
            if not re.fullmatch(r"[0-9a-f]{64}", file.get("advertised_sha256") or ""):
                raise ValueError(f"Missing advertised weight SHA256 at {row['stage']}")
    return {
        "status": "passed",
        "n_checkpoints": len(records),
        "architecture": reference["architecture"],
        "tokenizer": reference["tokenizer"],
        "chat_templates_compared_separately": True,
    }


def fetch_checkpoint_assets(checkpoint: Checkpoint, directory: Path) -> dict[str, Any]:
    """Explicit public HF reads using only immutable revisions and an asset allowlist."""
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    info = HfApi().model_info(checkpoint.repo, revision=checkpoint.revision, files_metadata=True)
    if info.sha != checkpoint.revision:
        raise ValueError(f"HF resolved a different revision for {checkpoint.stage}")
    names = metadata_file_names(info.siblings)
    directory.mkdir(parents=True, exist_ok=True)
    assets = []
    for name in names:
        path = Path(
            hf_hub_download(
                checkpoint.repo, name, revision=checkpoint.revision, local_dir=directory
            )
        )
        assets.append({"filename": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    weights = []
    for file in info.siblings:
        if file.rfilename.endswith(".safetensors") or re.fullmatch(
            r"pytorch_model(?:-\d+-of-\d+)?\.bin", file.rfilename
        ):
            lfs = file.lfs
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            weights.append(
                {
                    "filename": file.rfilename,
                    "bytes": file.size,
                    "advertised_sha256": digest,
                    "format": "safetensors"
                    if file.rfilename.endswith(".safetensors")
                    else "pytorch_bin",
                }
            )
    safe_meta = getattr(info, "safetensors", None)
    safe_meta = asdict(safe_meta) if is_dataclass(safe_meta) else safe_meta
    template = getattr(tokenizer, "chat_template", None)
    raw_config = json.loads((directory / "config.json").read_text())
    result = {
        **checkpoint.to_dict(),
        "resolved_revision": info.sha,
        "architecture": architecture_signature(config.to_dict()),
        "tokenizer": tokenizer_signature(tokenizer),
        "assets": assets,
        "asset_directory": str(directory),
        "weight_files": weights,
        "advertised_weight_bytes": sum(file["bytes"] or 0 for file in weights),
        "weight_metadata_verification": "Advertised LFS SHA256 only; pretrained weight bytes were NOT downloaded or hashed",
        "safetensors_metadata": safe_meta,
        "stored_config_dtype": raw_config.get("dtype", raw_config.get("torch_dtype")),
        "requested_execution_dtype": "bfloat16",
        "bf16_execution": "Model supports dtype casting; GPU hardware and forward/backward support must be tested during the GPU sanity run",
        "chat_template_present": template is not None,
        "chat_template_sha256": fingerprint(template) if template is not None else None,
        "primary_prompt_policy": "fixed native benchmark strings, add_special_tokens=False, no checkpoint-specific chat wrapping",
    }
    write_json(directory / "preflight_checkpoint.json", result)
    return result


def _sample_source_groups(examples: Sequence[Example], count: int, seed: int) -> list[Example]:
    """Seeded groups plus longest-text groups; sample covers rare long examples."""
    if count < 1:
        raise ValueError("sample_groups must be positive")
    strata: dict[tuple[str, str], dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        strata[(example.task, example.metadata["split"])][example.group].append(example)
    rng = np.random.default_rng(seed)
    chosen = []
    for key in sorted(strata):
        groups = strata[key]
        names = sorted(groups)
        selected = set(rng.choice(names, size=min(count, len(names)), replace=False).tolist())
        longest = sorted(
            names, key=lambda name: (-max(len(ex.prompt) for ex in groups[name]), name)
        )[: min(16, len(names))]
        selected.update(longest)
        for group in sorted(selected):
            chosen.extend(sorted(groups[group], key=lambda ex: ex.id))
    return chosen


def token_length_summary(
    lengths: Sequence[int], *, max_context: int, generation_reserve: int
) -> dict[str, Any]:
    """Report overflow and generation-budget pressure, never silently truncate."""
    array = np.asarray(lengths, dtype=np.int64)
    if array.ndim != 1 or not len(array) or np.any(array < 1):
        raise ValueError("Token lengths must be a nonempty positive vector")
    if max_context < 1 or generation_reserve < 0:
        raise ValueError("Invalid context/generation budget")
    return {
        "n_rows": len(array),
        "min_tokens": int(array.min()),
        "max_observed_tokens": int(array.max()),
        "quantiles_tokens": {
            name: float(value)
            for name, value in zip(
                ("p50", "p95", "p99"), np.quantile(array, [0.5, 0.95, 0.99]), strict=True
            )
        },
        "n_prompt_overflow": int(np.sum(array > max_context)),
        "n_no_generation_slot": int(np.sum(array >= max_context)),
        "n_generation_budget_exceeds_context": int(
            np.sum(array + generation_reserve > max_context)
        ),
        "max_context": max_context,
        "generation_reserve": generation_reserve,
        "actual_generation_truncation": "Not observable from tokenizing prompts; requires GPU generation",
    }


def audit_contexts(
    examples: Sequence[Example],
    tokenizer: Any,
    *,
    max_context: int = 4096,
    max_new_tokens: int = 512,
    sample_groups: int = 128,
    seed: int = 0,
    diagnostic_builder: Callable[[Example], Example] | None = None,
) -> dict[str, Any]:
    """All native prompts plus disclosed sampled diagnostics and ETHICS controls."""
    if diagnostic_builder is None:
        from .runner import diagnostic_example

        diagnostic_builder = diagnostic_example
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[f"{example.task}/{example.metadata['split']}"].append(example)
    native = {}
    for key, rows in sorted(grouped.items()):
        lengths = []
        for offset in range(0, len(rows), 128):
            lengths.extend(
                map(
                    len,
                    tokenizer(
                        [ex.prompt for ex in rows[offset : offset + 128]],
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"],
                )
            )
        reserve = 1 if rows[0].task.startswith("ethics_") else max_new_tokens
        native[key] = token_length_summary(
            lengths, max_context=max_context, generation_reserve=reserve
        )
        native[key]["scope"] = "all native prompts in the verified local dataset split"
    selected = _sample_source_groups(examples, sample_groups, seed)
    diagnostics: dict[str, list[int]] = defaultdict(list)
    controls: dict[str, list[int]] = defaultdict(list)
    failures = []
    selected_ids = []
    single_token_count = 0
    for original in selected:
        key = f"{original.task}/{original.metadata['split']}"
        selected_ids.append(original.id)
        diagnostic = diagnostic_builder(original)
        n = len(tokenizer.encode(diagnostic.prompt, add_special_tokens=False)) + len(
            tokenizer.encode(diagnostic.answer, add_special_tokens=False)
        )
        diagnostics[key].append(n)
        if original.task.startswith("ethics_"):
            variants = ethics_controls(original, tokenizer)
            if len(variants) != 8 or len({row.metadata["variant"] for row in variants}) != 8:
                raise ValueError("ETHICS controls do not form eight distinct renderings")
            for row in [original, *variants]:
                single_token_labels(row.prompt, row.options, tokenizer)
                single_token_count += len(row.options)
                if row.group != original.group:
                    raise ValueError("ETHICS rendering changes source-group identity")
                if row is not original:
                    size = len(tokenizer.encode(row.prompt, add_special_tokens=False)) + len(
                        tokenizer.encode(row.answer, add_special_tokens=False)
                    )
                    controls[key].append(size)
                    if size > max_context:
                        failures.append(
                            {
                                "id": row.id,
                                "issue": "controlled answer context overflow",
                                "tokens": size,
                            }
                        )
    sampled = {"diagnostics": {}, "ethics_controls": {}}
    for name, values in (("diagnostics", diagnostics), ("ethics_controls", controls)):
        for key, lengths in sorted(values.items()):
            sampled[name][key] = token_length_summary(
                lengths, max_context=max_context, generation_reserve=0
            )
            sampled[name][key]["scope"] = (
                "seeded source-group sample plus longest-text groups; maximum is NOT a global bound"
            )
    return {
        "native": native,
        **sampled,
        "selected_item_ids": selected_ids,
        "sampling": {
            "seed": seed,
            "random_groups_per_task_split": sample_groups,
            "longest_text_groups_per_task_split": 16,
            "selected_examples": len(selected),
            "population_examples": len(examples),
            "preserve_complete_native_groups": True,
        },
        "single_token_label_checks": single_token_count,
        "failures": failures,
        "truncation_policy": "No truncation applied; overflows explicitly counted",
    }


def run_preflight(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_context: int = 4096,
    max_new_tokens: int = 512,
    sample_groups: int = 128,
    seed: int = 0,
    workers: int = 3,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Run the explicit network metadata audit and local CPU data/context checks."""
    from transformers import AutoTokenizer
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = verify_data(data_dir)
    report: dict[str, Any] = {
        "started_utc": utc_now(),
        "status": "running",
        "dataset_manifest": manifest,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface_hub", "numpy")
        },
        "cpu_only_preflight": True,
        "pretrained_weights_downloaded": False,
        "cuda_available_locally": torch.cuda.is_available(),
    }
    write_json(output / "preflight.json", report)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(fetch_checkpoint_assets, cp, output / "checkpoints" / cp.stage)
                for cp in CHECKPOINTS
            ]
            records = [future.result() for future in futures]
        report["checkpoints"] = records
        report["compatibility"] = validate_checkpoint_compatibility(records)
        if records[0]["architecture"]["max_position_embeddings"] < max_context:
            raise ValueError("Requested context budget exceeds checkpoint native context")
        report["chat_templates"] = {row["stage"]: row["chat_template_sha256"] for row in records}
        report["chat_template_policy"] = (
            "Differences are recorded, not normalized away; primary evaluation uses unchanged native prompts without chat wrappers."
        )
        if not metadata_only:
            examples = load_examples(data_dir, include_hard=True, include_train=True)
            tokenizer = AutoTokenizer.from_pretrained(
                output / "checkpoints" / CHECKPOINTS[0].stage,
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenizer.pad_token = tokenizer.eos_token
            report["contexts"] = audit_contexts(
                examples,
                tokenizer,
                max_context=max_context,
                max_new_tokens=max_new_tokens,
                sample_groups=sample_groups,
                seed=seed,
            )
        else:
            report["contexts"] = {"status": "not_requested"}
        report.update(status="passed", finished_utc=utc_now())
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_utc=utc_now())
        raise
    finally:
        write_json(output / "preflight.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/tmp/olmo-circuit-data")
    parser.add_argument("--output", default="geode-store/olmo2-sanity-preflight")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample-groups", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        args.data,
        args.output,
        max_context=args.max_context,
        max_new_tokens=args.max_new_tokens,
        sample_groups=args.sample_groups,
        seed=args.seed,
        workers=args.workers,
        metadata_only=args.metadata_only,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_checkpoints": len(report["checkpoints"]),
                "pretrained_weights_downloaded": False,
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
