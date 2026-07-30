"""Run manifest: load/validate/save + registry operations (specs/00 §1, §2, §8).

Validation follows resolved decision OQ-3: every key listed in spec 00 §2 is
required recursively; ``null`` is accepted only where the schema says
``|null``; primitive types (with list element types) are checked; errors name
the offending dotted path (e.g. ``training.optimizer.lr``). Unknown extra
fields are permitted, ignored by validation, and preserved exactly through
load/save (V0.2). JSON leniency: an ``int`` is accepted where ``float`` is
declared. ``training.stopping`` is a discriminated union (spec 00 §2,
2026-07-21): presence of ``metric`` selects the behavioral-rule schema,
absence the loss ε/k schema (V0.7).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from geode.zoo.store import manifest_path, resolve_store


class ManifestError(ValueError):
    """Manifest validation failure; the message names the offending dotted field."""


# --- Declarative schema for spec 00 §2 -------------------------------------
#
# A schema node is either a dict (required sub-object, walked recursively) or
# a leaf ``(kind, nullable)`` where ``kind`` is a primitive-type name or, for
# closed enums, a tuple of the allowed string values.

_Leaf = tuple[str | tuple[str, ...], bool]
_SchemaNode = dict[str, "_Leaf | dict | _Union"]


@dataclass(frozen=True)
class _Union:
    """Discriminated union of object schemas, selected by a discriminant value.

    The value validates against ``by_value[data[discriminant]]`` when
    ``discriminant`` is a key of the object (an unlisted value is an error —
    the discriminant stays a closed enum), else against ``if_absent``. Fields
    of the unselected branch are neither required nor checked.
    """

    discriminant: str
    by_value: dict[str, dict]
    if_absent: dict


_SCHEMA: _SchemaNode = {
    "schema_version": ("int", False),
    "run_id": ("str", False),
    "created_utc": ("str", False),
    "git_commit": ("str", False),
    "regime": (("elicit", "teach", "unknown"), False),
    "base_model": {
        "hf_id": ("str", False),
        "revision": ("str", False),
    },
    "task": {
        "name": ("str", False),
        "format_version": ("str", False),
    },
    "dataset": {
        "name": ("str", False),
        "n_unique_examples": ("int", False),
        "seed": ("int", False),
    },
    "training": {
        "method": (("lora", "sparse_lora", "full_ft"), False),
        "lora": {
            "rank": ("int", True),
            "alpha": ("float", True),
            "target_modules": ("list[str]", False),
            "dropout": ("float", True),
            "sparse_param_count": ("int", True),
        },
        "optimizer": {
            "name": ("str", False),
            "lr": ("float", False),
            "batch_size": ("int", False),
            "micro_batch_size": ("int", True),
            "betas": ("list[float]", True),
            "weight_decay": ("float", False),
            "grad_clip": ("float", True),
        },
        "lr_schedule": (("constant", "cosine"), False),
        "min_lr": ("float", True),
        "precision": (("fp32", "bf16"), False),
        "eval_every": ("int", True),
        "max_steps": ("int", True),
        "stopping": _Union(
            discriminant="metric",
            by_value={
                # Behavioral rule — format installers, spec 02 §6.
                "format_validity": {
                    "metric": (("format_validity",), False),
                    "threshold": ("float", False),
                    "k": ("int", False),
                    "n_prompts": ("int", False),
                    "prompt_seed": ("int", False),
                },
                # Full-dose training-loss ε/k rule — new-phase dose installers
                # (2026-07-26, spec 02 §6). Same fields as the val-loss branch,
                # named explicitly because the metric is NOT a held-out loss:
                # nothing about a dose run's number is comparable to a val
                # curve, and an unlabelled ε/k record would read as if it were.
                "train_loss": {
                    "metric": (("train_loss",), False),
                    "eps_nats": ("float", False),
                    "k": ("int", False),
                    "min_steps": ("int", True),
                },
            },
            if_absent={  # loss ε/k rule on a held-out val split
                "eps_nats": ("float", True),
                "k": ("int", True),
                "min_steps": ("int", True),
            },
        ),
        "epochs_total": ("int", False),
        "seed": ("int", False),
    },
    "trainable_param_count": ("int", False),
    "snapshot_steps": ("list[int]", False),
    "cost": {
        "gpu_type": ("str", True),
        "est_usd": ("float", True),
        "actual_usd": ("float", True),
    },
    "status": (("planned", "running", "complete", "failed"), False),
}

# ``bool`` is a subclass of ``int`` in Python but a distinct type in JSON.
_TYPE_CHECKS = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "list[str]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "list[int]": lambda v: (
        isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    ),
    "list[float]": lambda v: (
        isinstance(v, list)
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
    ),
}


def _check_leaf(path: str, value: object, leaf: _Leaf) -> None:
    kind, nullable = leaf
    if value is None:
        if nullable:
            return
        raise ManifestError(f"field '{path}' must not be null")
    if isinstance(kind, tuple):  # closed enum
        if value not in kind:
            raise ManifestError(f"field '{path}' must be one of {list(kind)}, got {value!r}")
        return
    if not _TYPE_CHECKS[kind](value):
        raise ManifestError(f"field '{path}' must be of type {kind}, got {value!r}")


def _validate_node(data: dict, schema: _SchemaNode, prefix: str) -> None:
    for key, sub in schema.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in data:
            raise ManifestError(f"missing required field '{path}'")
        value = data[key]
        if isinstance(sub, _Union):
            if not isinstance(value, dict):
                raise ManifestError(f"field '{path}' must be an object, got {value!r}")
            if sub.discriminant in value:
                branch = value[sub.discriminant]
                if branch not in sub.by_value:
                    raise ManifestError(
                        f"field '{path}.{sub.discriminant}' must be one of "
                        f"{list(sub.by_value)}, got {branch!r}"
                    )
                sub = sub.by_value[branch]
            else:
                sub = sub.if_absent
        if isinstance(sub, dict):
            if not isinstance(value, dict):
                raise ManifestError(f"field '{path}' must be an object, got {value!r}")
            _validate_node(value, sub, path)
        else:
            _check_leaf(path, value, sub)


@dataclass
class RunManifest:
    """A run's ``manifest.json`` (spec 00 §2), unknown extra fields preserved.

    ``data`` holds the full parsed JSON object; ``save`` writes it back
    verbatim, and ``validate`` checks only the required fields.
    """

    data: dict

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Parse ``path`` as a manifest. Does not validate; call ``validate()``."""
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ManifestError(f"manifest root must be a JSON object, got {type(data).__name__}")
        return cls(data=data)

    def validate(self) -> None:
        """Check required fields/types/enums per spec 00 §2 and OQ-3."""
        _validate_node(self.data, _SCHEMA, "")

    def save(self, path: Path) -> None:
        """Write the manifest (full ``data``, extras included) to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2) + "\n")


def register_run(fields: dict, *, store: Path | None = None) -> RunManifest:
    """Validate ``fields`` and write it to ``{store}/runs/{run_id}/manifest.json``.

    Refuses to overwrite an existing manifest whose ``status`` is
    ``"running"`` — a live trainer may own it (a silent re-register masked
    the 2026-07-19 duplicate launch). Staleness is never auto-detected: a
    crashed run leaves ``"running"`` behind forever, so the escape hatch is
    manual — delete the stale manifest or edit its ``status``, then retry.
    """
    manifest = RunManifest(data=fields)
    manifest.validate()
    path = manifest_path(fields["run_id"], store=store)
    if path.exists():
        existing = json.loads(path.read_text())
        if isinstance(existing, dict) and existing.get("status") == "running":
            raise ManifestError(
                f"run '{fields['run_id']}' is already registered with status 'running' "
                f"at {path} — refusing to overwrite (a live trainer may own it). "
                "If that run is dead, delete the manifest or edit its status, then retry."
            )
    manifest.save(path)
    return manifest


def load_run(run_id: str, *, store: Path | None = None) -> RunManifest:
    """Load and validate the manifest registered under ``run_id``."""
    manifest = RunManifest.load(manifest_path(run_id, store=store))
    manifest.validate()
    return manifest


def iter_runs(
    regime: str | None = None,
    task: str | None = None,
    status: str = "complete",
    *,
    store: Path | None = None,
) -> Iterator[RunManifest]:
    """Yield registered runs matching the filters (spec 00 §8).

    ``regime=None`` / ``task=None`` mean no filter on that key; ``task``
    matches ``task.name``. ``status`` always filters (default ``"complete"``).
    """
    runs_root = resolve_store(store) / "runs"
    if not runs_root.is_dir():
        return
    for path in sorted(runs_root.glob("*/manifest.json")):
        manifest = RunManifest.load(path)
        data = manifest.data
        if data.get("status") != status:
            continue
        if regime is not None and data.get("regime") != regime:
            continue
        if task is not None and data.get("task", {}).get("name") != task:
            continue
        yield manifest
