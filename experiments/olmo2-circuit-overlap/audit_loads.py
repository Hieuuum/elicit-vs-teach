"""CPU-only cached checkpoint load/hash audit; never downloads or runs inference.

Run after GPU inference with the SAME Hugging Face cache and preflight.json.
All checkpoint files resolve through try_to_load_from_cache. Loading receives
the local snapshot directory, local_files_only=True, and explicit CPU/BF16.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from huggingface_hub import try_to_load_from_cache

from geode.circuits.artifacts import sha256_file, utc_now, write_json


def normalized_loading_info(info: dict[str, Any]) -> dict[str, list]:
    """Normalize Transformers 5 sets and legacy lists; reject incomplete reports."""
    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if any(field not in info for field in fields):
        raise ValueError("Loader did not return all required integrity fields")
    return {field: sorted(list(info[field]), key=repr) for field in fields}


def audit_checkpoint(record: dict, *, cache_dir: str | Path | None = None) -> dict:
    """Hash cached advertised weights and reload one pinned snapshot on CPU."""
    import torch
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    result = {
        "stage": record["stage"],
        "repo": record["repo"],
        "revision": record["revision"],
        "status": "failed",
        "files": [],
    }
    model = None
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
            raise ValueError("Audit requires an immutable checkpoint revision")

        def cached(filename: str) -> Path | None:
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("Unsafe checkpoint filename")
            value = try_to_load_from_cache(
                record["repo"], filename, revision=record["revision"], cache_dir=cache_dir
            )
            return Path(value) if isinstance(value, str) else None

        config = cached("config.json")
        if config is None or config.parent.name != record["revision"]:
            raise ValueError("Pinned config snapshot is unavailable in the existing cache")
        snapshot = config.parent
        result["snapshot_directory"] = str(snapshot)
        # Verify metadata already present as well, particularly config and any
        # shard index. Absent optional tokenizer assets are explicitly recorded.
        absent_metadata = []
        for asset in record.get("assets", []):
            path = cached(asset["filename"])
            if path is None:
                absent_metadata.append(asset["filename"])
                continue
            actual = sha256_file(path)
            if actual != asset["sha256"]:
                raise ValueError(f"Cached metadata SHA256 mismatch: {asset['filename']}")
        result["uncached_optional_metadata"] = absent_metadata
        by_name = {file["filename"]: file for file in record["weight_files"]}
        cached_weights = {}
        for filename, file in by_name.items():
            path = cached(filename)
            if path is None:
                continue
            advertised = file.get("advertised_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", advertised):
                raise ValueError(f"Missing advertised SHA256: {filename}")
            actual, size = sha256_file(path), path.stat().st_size
            if actual != advertised or (file.get("bytes") is not None and size != file["bytes"]):
                raise ValueError(f"Cached weight SHA256/size mismatch: {filename}")
            result["files"].append({"filename": filename, "sha256": actual, "bytes": size})
            cached_weights[filename] = path
        result["uncached_advertised_weights"] = sorted(set(by_name) - set(cached_weights))
        if not cached_weights:
            raise ValueError("No advertised weight files exist in the current cache")
        safe = any(name.endswith(".safetensors") for name in cached_weights)
        index_name = "model.safetensors.index.json" if safe else "pytorch_model.bin.index.json"
        index = cached(index_name)
        if index is not None:
            required_weights = set(json.loads(index.read_text())["weight_map"].values())
        else:
            required_weights = {"model.safetensors" if safe else "pytorch_model.bin"}
        if not required_weights.issubset(cached_weights):
            raise ValueError("The selected weight format has missing/unverified cached shards")
        result["loaded_weight_files"] = sorted(required_weights)
        with torch.device("cpu"):
            model, info = AutoModelForCausalLM.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                output_loading_info=True,
                ignore_mismatched_sizes=False,
                use_safetensors=safe,
                attn_implementation="sdpa",
            )
        result["loading_info"] = normalized_loading_info(info)
        if any(result["loading_info"].values()):
            raise ValueError(
                "Checkpoint load reports missing, unexpected, mismatched keys or errors"
            )
        if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("Audited parameters were not fully materialized on CPU in BF16")
        result.update(
            status="passed",
            parameter_count=sum(p.numel() for p in model.parameters()),
            execution_dtype="bfloat16",
            execution_device="cpu",
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        del model
        gc.collect()
        result["seconds"] = time.monotonic() - started
    return result


def run_audit(preflight: Path, output: Path, *, cache_dir: str | Path | None = None) -> dict:
    """Write a progressive machine-readable record; failed stages stay visible."""
    source = json.loads(preflight.read_text())
    records = source["checkpoints"]
    if not records or len({record["stage"] for record in records}) != len(records):
        raise ValueError("Preflight must name a nonempty unique checkpoint sequence")
    report = {
        "status": "running",
        "started_utc": utc_now(),
        "preflight_sha256": sha256_file(preflight),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "cpu_only": True,
        "network_allowed": False,
        "inference_performed": False,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface_hub")
        },
        "checkpoints": [],
    }
    write_json(output, report)
    for record in records:
        result = audit_checkpoint(record, cache_dir=cache_dir)
        report["checkpoints"].append(result)
        write_json(output, report)
        print(
            json.dumps(
                {"stage": result["stage"], "status": result["status"], "seconds": result["seconds"]}
            ),
            flush=True,
        )
    report.update(
        status="passed"
        if all(r["status"] == "passed" for r in report["checkpoints"])
        else "failed",
        finished_utc=utc_now(),
    )
    write_json(output, report)
    return report


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    audit = run_audit(args.preflight, args.output, cache_dir=args.cache_dir)
    raise SystemExit(0 if audit["status"] == "passed" else 1)
