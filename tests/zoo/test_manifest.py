"""ZOO-1 tests: run manifest + registry (``geode.zoo``).

Derived from specs/00-interfaces.md — §1 "Directory layout", §2 "Run
manifest", §8 "Public API surface" — validation properties V0.1 and V0.2,
and resolved decision (specs/00 §9) OQ-3 (all listed keys required recursively;
``null`` only where the schema says ``|null``; primitive type checks;
errors name the dotted path, e.g. ``training.optimizer.lr``).

Written test-first: the imports below target the public API only and are
expected to fail until ZOO-1 is implemented.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from geode.zoo import ManifestError, RunManifest, iter_runs, load_run, register_run

# ---------------------------------------------------------------------------
# Manifest construction helpers (spec 00 §2, verbatim field set)
# ---------------------------------------------------------------------------


def make_manifest(run_id: str = "run-0001") -> dict:
    """A fully valid spec 00 §2 manifest. Each call returns a fresh dict."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-11T00:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "regime": "elicit",
        "base_model": {"hf_id": "meta-llama/Llama-3.2-1B", "revision": "main"},
        "task": {"name": "task_a", "format_version": "1"},
        "dataset": {"name": "dataset_a", "n_unique_examples": 512, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 8,
                "alpha": 16.0,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.05,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "batch_size": 8,
                "micro_batch_size": 4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "grad_clip": 1.0,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "bf16",
            "eval_every": 100,
            "max_steps": None,
            "stopping": {"eps_nats": 0.005, "k": 3, "min_steps": 0},
            "epochs_total": 3,
            "seed": 0,
        },
        "trainable_param_count": 4096,
        "snapshot_steps": [0, 8, 64],
        "cost": {"gpu_type": "A100", "est_usd": 1.5, "actual_usd": None},
        "status": "complete",
    }


# Every required key from spec 00 §2, recursively (OQ-3: all listed keys
# required, including nested dicts and the keys inside them).
REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "created_utc",
    "git_commit",
    "regime",
    "base_model",
    "base_model.hf_id",
    "base_model.revision",
    "task",
    "task.name",
    "task.format_version",
    "dataset",
    "dataset.name",
    "dataset.n_unique_examples",
    "dataset.seed",
    "training",
    "training.method",
    "training.lora",
    "training.lora.rank",
    "training.lora.alpha",
    "training.lora.target_modules",
    "training.lora.dropout",
    "training.lora.sparse_param_count",
    "training.optimizer",
    "training.optimizer.name",
    "training.optimizer.lr",
    "training.optimizer.batch_size",
    "training.optimizer.micro_batch_size",
    "training.optimizer.betas",
    "training.optimizer.weight_decay",
    "training.optimizer.grad_clip",
    "training.lr_schedule",
    "training.min_lr",
    "training.precision",
    "training.eval_every",
    "training.max_steps",
    "training.stopping",
    "training.stopping.eps_nats",
    "training.stopping.k",
    "training.stopping.min_steps",
    "training.epochs_total",
    "training.seed",
    "trainable_param_count",
    "snapshot_steps",
    "cost",
    "cost.gpu_type",
    "cost.est_usd",
    "cost.actual_usd",
    "status",
)

# Exactly the fields whose spec 00 §2 type carries "|null".
NULLABLE_FIELDS = (
    "training.lora.rank",
    "training.lora.alpha",
    "training.lora.dropout",
    "training.lora.sparse_param_count",
    "training.optimizer.micro_batch_size",
    "training.optimizer.betas",
    "training.optimizer.grad_clip",
    "training.min_lr",
    "training.eval_every",
    "training.max_steps",
    "training.stopping.eps_nats",
    "training.stopping.k",
    "training.stopping.min_steps",
    "cost.gpu_type",
    "cost.est_usd",
    "cost.actual_usd",
)

# OQ-3: null is allowed *only* where the schema says "|null" — everything
# else in the required set must reject null.
NON_NULLABLE_FIELDS = tuple(f for f in REQUIRED_FIELDS if f not in NULLABLE_FIELDS)


def _walk_to_parent(fields: dict, dotted: str) -> tuple[dict, str]:
    *parents, leaf = dotted.split(".")
    node = fields
    for key in parents:
        node = node[key]
    return node, leaf


def _set_field(fields: dict, dotted: str, value: object) -> None:
    node, leaf = _walk_to_parent(fields, dotted)
    node[leaf] = value


def _delete_field(fields: dict, dotted: str) -> None:
    node, leaf = _walk_to_parent(fields, dotted)
    del node[leaf]


def _load_and_validate(tmp_path: Path, fields: dict) -> RunManifest:
    """Round the dict through JSON and the public load/validate API.

    V0.1 pins *that* validation fails with an error naming the field, not
    *where* it fails: implementations may raise from ``load`` or from
    ``validate``; both surface inside a ``pytest.raises`` around this helper.
    """
    path = tmp_path / "candidate_manifest.json"
    path.write_text(json.dumps(fields))
    manifest = RunManifest.load(path)
    manifest.validate()
    return manifest


# ---------------------------------------------------------------------------
# Baseline: the helper manifest itself is valid (guards the parametrized
# rejection tests below against passing vacuously on a doubly-broken dict).
# ---------------------------------------------------------------------------


def test_valid_manifest_passes_validation(tmp_path: Path) -> None:
    manifest = _load_and_validate(tmp_path, make_manifest())
    assert manifest.data == make_manifest()


# ---------------------------------------------------------------------------
# V0.1 — missing required field fails validation, error names the field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_error_names_field(tmp_path: Path, field: str) -> None:
    """V0.1 + OQ-3: every listed key is required, recursively.

    Deleting any one of them (including nullable ones — the *key* is still
    required) raises ManifestError whose message contains the dotted path.
    """
    fields = make_manifest()
    _delete_field(fields, field)
    with pytest.raises(ManifestError) as excinfo:
        _load_and_validate(tmp_path, fields)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", [f for f in REQUIRED_FIELDS if f.startswith("training.lora")])
def test_lora_fields_required_even_for_full_ft(tmp_path: Path, field: str) -> None:
    """OQ-3: "all listed keys required recursively" is unconditional.

    The ``training.lora`` block (and every key inside it) stays required even
    when ``training.method`` is not "lora" — nullable values, not omitted
    keys, are how a full_ft run records the absence of a lora config.
    """
    fields = make_manifest()
    fields["training"]["method"] = "full_ft"
    _delete_field(fields, field)
    with pytest.raises(ManifestError) as excinfo:
        _load_and_validate(tmp_path, fields)
    assert field in str(excinfo.value)


# ---------------------------------------------------------------------------
# OQ-3 (half of V0.1) — null allowed only where the schema says "|null"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,null_ok",
    [(f, True) for f in NULLABLE_FIELDS] + [(f, False) for f in NON_NULLABLE_FIELDS],
)
def test_null_allowed_only_where_schema_says(tmp_path: Path, field: str, null_ok: bool) -> None:
    fields = make_manifest()
    _set_field(fields, field, None)
    if null_ok:
        _load_and_validate(tmp_path, fields)  # must not raise
    else:
        with pytest.raises(ManifestError) as excinfo:
            _load_and_validate(tmp_path, fields)
        assert field in str(excinfo.value)


# ---------------------------------------------------------------------------
# OQ-3 — primitive type checks, error names the dotted path
# ---------------------------------------------------------------------------

# Closed-enum fields are exercised by the enum tests below — a wrong-typed
# value there is just another invalid enum value — so they are excluded from
# the wrong-type map.
ENUM_FIELDS = ("regime", "training.method", "training.lr_schedule", "training.precision", "status")

# Every remaining spec 00 §2 field (containers included) mapped to one or
# more wrong-typed values. Numeric fields get strings and string fields get
# numbers, so no case hinges on JSON int/float leniency. List-typed fields
# get an extra case with one wrong-typed *element*, since the schema declares
# element types (["str"], ["int"]).
WRONG_TYPE_VALUES: dict[str, tuple[object, ...]] = {
    "schema_version": ("1",),
    "run_id": (123,),
    "created_utc": (20260711,),
    "git_commit": (42,),
    "base_model": ("meta-llama/Llama-3.2-1B",),
    "base_model.hf_id": (7,),
    "base_model.revision": (2,),
    "task": ("task_a",),
    "task.name": (3.5,),
    "task.format_version": (1,),
    "dataset": ("dataset_a",),
    "dataset.name": (11,),
    "dataset.n_unique_examples": ("512",),
    "dataset.seed": ("0",),
    "training": ("lora",),
    "training.lora": (8,),
    "training.lora.rank": ("8",),  # nullable, but a non-null value is still type-checked
    "training.lora.alpha": ("16.0",),
    "training.lora.target_modules": ("q_proj", ["q_proj", 3]),
    "training.lora.dropout": ("0.05",),
    "training.lora.sparse_param_count": ("4096",),
    "training.optimizer": ("adamw",),
    "training.optimizer.name": (1,),
    "training.optimizer.lr": ("1e-4",),
    "training.optimizer.batch_size": ("8",),
    "training.optimizer.micro_batch_size": ("4",),
    "training.optimizer.betas": (0.9, [0.9, "0.999"]),
    "training.optimizer.weight_decay": ("0.01",),
    "training.optimizer.grad_clip": ("1.0",),
    "training.min_lr": ("1e-5",),
    "training.eval_every": ("100",),
    "training.max_steps": ("17000",),
    "training.stopping": (0.005,),
    "training.stopping.eps_nats": ("0.005",),
    "training.stopping.k": ("3",),
    "training.stopping.min_steps": ("5000",),
    "training.epochs_total": ("3",),
    "training.seed": ("0",),
    "trainable_param_count": ("4096",),
    "snapshot_steps": (8, [0, "8"]),
    "cost": ("A100",),
    "cost.gpu_type": (100,),
    "cost.est_usd": ("1.5",),
    "cost.actual_usd": ("0.9",),
}

WRONG_TYPE_CASES = tuple(
    (field, bad_value)
    for field, bad_values in WRONG_TYPE_VALUES.items()
    for bad_value in bad_values
)


def test_wrong_type_map_covers_every_schema_field() -> None:
    """Guard: the wrong-type map stays exhaustive as the schema evolves."""
    assert set(WRONG_TYPE_VALUES) == set(REQUIRED_FIELDS) - set(ENUM_FIELDS)


@pytest.mark.parametrize("field,bad_value", WRONG_TYPE_CASES)
def test_wrong_primitive_type_rejected(tmp_path: Path, field: str, bad_value: object) -> None:
    fields = make_manifest()
    _set_field(fields, field, bad_value)
    with pytest.raises(ManifestError) as excinfo:
        _load_and_validate(tmp_path, fields)
    assert field in str(excinfo.value)


# JSON true/false must not satisfy "int": bool is a subclass of int in
# Python, so ``isinstance(True, int)`` holds and a naive validator would
# accept booleans in every int-typed field by default.
BOOL_FOR_INT_CASES = (
    ("schema_version", True),
    ("training.epochs_total", False),
    ("snapshot_steps", [0, True, 64]),
)


@pytest.mark.parametrize("field,bad_value", BOOL_FOR_INT_CASES)
def test_bool_rejected_for_int_typed_field(tmp_path: Path, field: str, bad_value: object) -> None:
    fields = make_manifest()
    _set_field(fields, field, bad_value)
    with pytest.raises(ManifestError) as excinfo:
        _load_and_validate(tmp_path, fields)
    assert field in str(excinfo.value)


# ---------------------------------------------------------------------------
# Spec 00 §2 closed enums: regime, training.method, status
# ---------------------------------------------------------------------------

INVALID_ENUM_CASES = (
    ("regime", "finetune"),
    ("training.method", "qlora"),
    ("training.lr_schedule", "linear"),
    ("training.precision", "fp16"),
    ("status", "done"),
)

VALID_ENUM_CASES = (
    ("regime", "teach"),
    ("regime", "unknown"),
    ("training.method", "sparse_lora"),
    ("training.method", "full_ft"),
    ("training.lr_schedule", "cosine"),
    ("training.precision", "fp32"),
    ("status", "planned"),
    ("status", "running"),
    ("status", "failed"),
)


@pytest.mark.parametrize("field,bad_value", INVALID_ENUM_CASES)
def test_invalid_enum_value_rejected(tmp_path: Path, field: str, bad_value: str) -> None:
    fields = make_manifest()
    _set_field(fields, field, bad_value)
    with pytest.raises(ManifestError) as excinfo:
        _load_and_validate(tmp_path, fields)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field,value", VALID_ENUM_CASES)
def test_valid_enum_values_accepted(tmp_path: Path, field: str, value: str) -> None:
    """Every member of a spec 00 §2 enum validates (guards over-restriction)."""
    fields = make_manifest()
    _set_field(fields, field, value)
    _load_and_validate(tmp_path, fields)  # must not raise


# ---------------------------------------------------------------------------
# V0.2 — round-trip preserves the manifest exactly, unknown fields included
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_unknown_extra_fields(tmp_path: Path) -> None:
    """V0.2: save(load(m)) == m, including unknown extra fields.

    Spec 00 §2: "Unknown extra fields are permitted and preserved" — at the
    top level and nested inside required sub-dicts.
    """
    fields = make_manifest()
    fields["x_experiment_notes"] = {"tags": ["pilot", 1, None], "flag": True}
    fields["base_model"]["x_quantization"] = "none"
    fields["training"]["lora"]["x_init_scheme"] = "gaussian"
    fields["cost"]["x_currency"] = "usd"
    original = copy.deepcopy(fields)

    src = tmp_path / "manifest.json"
    src.write_text(json.dumps(fields))
    manifest = RunManifest.load(src)
    manifest.validate()  # unknown extras are permitted: validation passes
    assert manifest.data == original

    dst = tmp_path / "manifest_roundtrip.json"
    manifest.save(dst)
    assert json.loads(dst.read_text()) == original
    reloaded = RunManifest.load(dst)
    assert reloaded.data == original


# ---------------------------------------------------------------------------
# Registry: register/load under the spec 00 §1 layout
# ---------------------------------------------------------------------------


def test_register_then_load_identity(geode_store: Path) -> None:
    """register_run then load_run returns equal data; file lands per §1."""
    fields = make_manifest(run_id="run-reg-01")
    expected = copy.deepcopy(fields)

    registered = register_run(fields)
    assert isinstance(registered, RunManifest)
    assert registered.data == expected

    # Spec 00 §1: $GEODE_STORE/runs/{run_id}/manifest.json
    manifest_path = geode_store / "runs" / "run-reg-01" / "manifest.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text()) == expected

    loaded = load_run("run-reg-01")
    assert isinstance(loaded, RunManifest)
    assert loaded.data == expected


def test_register_refuses_overwriting_running_run(geode_store: Path) -> None:
    """register_run refuses a run_id whose existing manifest is status="running"
    (a live trainer may own it — the 2026-07-19 duplicate-launch guard);
    any other status is overwritable, as is the same run after the manual
    escape hatch (editing the stale status)."""
    fields = make_manifest(run_id="run-dup")
    fields["status"] = "running"
    register_run(fields)

    with pytest.raises(ManifestError, match="running"):
        register_run(make_manifest(run_id="run-dup"))

    # Escape hatch: edit the stale status on disk, then re-register.
    path = geode_store / "runs" / "run-dup" / "manifest.json"
    stale = json.loads(path.read_text())
    stale["status"] = "failed"
    path.write_text(json.dumps(stale))
    register_run(make_manifest(run_id="run-dup"))
    assert json.loads(path.read_text())["status"] == make_manifest()["status"]


# ---------------------------------------------------------------------------
# iter_runs filtering semantics (spec 00 §8)
# ---------------------------------------------------------------------------


def test_iter_runs_filters_regime_task_status(geode_store: Path) -> None:
    runs = (
        ("run-1", "elicit", "task_a", "complete"),
        ("run-2", "teach", "task_a", "complete"),
        ("run-3", "elicit", "task_b", "complete"),
        ("run-4", "elicit", "task_a", "failed"),
        ("run-5", "teach", "task_b", "running"),
    )
    for run_id, regime, task_name, status in runs:
        fields = make_manifest(run_id=run_id)
        fields["regime"] = regime
        fields["task"]["name"] = task_name
        fields["status"] = status
        register_run(fields)

    def run_ids(**kwargs: object) -> set[str]:
        manifests = list(iter_runs(**kwargs))
        assert all(isinstance(m, RunManifest) for m in manifests)
        return {m.data["run_id"] for m in manifests}

    # Default: status="complete" only, no regime/task filter.
    assert run_ids() == {"run-1", "run-2", "run-3"}
    # Explicit None on regime/task means "no filter on that key".
    assert run_ids(regime=None, task=None) == {"run-1", "run-2", "run-3"}
    # regime / task filters narrow (within status="complete").
    assert run_ids(regime="elicit") == {"run-1", "run-3"}
    assert run_ids(regime="teach") == {"run-2"}
    assert run_ids(task="task_a") == {"run-1", "run-2"}
    assert run_ids(regime="elicit", task="task_b") == {"run-3"}
    # Non-default status selects the matching runs.
    assert run_ids(status="failed") == {"run-4"}
    assert run_ids(status="running", regime="teach") == {"run-5"}
    assert run_ids(status="running", task="task_a") == set()


# ---------------------------------------------------------------------------
# Store root resolution: $GEODE_STORE env, explicit override, no fallback
# ---------------------------------------------------------------------------


def test_store_root_comes_from_env(
    geode_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1: "$GEODE_STORE is an environment variable; no absolute paths in code"."""
    # 1. With $GEODE_STORE set (geode_store fixture), no explicit store needed.
    register_run(make_manifest(run_id="run-env"))
    assert (geode_store / "runs" / "run-env" / "manifest.json").is_file()

    # 2. An explicit store=Path overrides the environment.
    other_store = tmp_path / "other_store"
    other_store.mkdir()
    register_run(make_manifest(run_id="run-other"), store=other_store)
    assert (other_store / "runs" / "run-other" / "manifest.json").is_file()
    assert not (geode_store / "runs" / "run-other").exists()
    loaded = load_run("run-other", store=other_store)
    assert loaded.data["run_id"] == "run-other"

    # 3. Neither env nor explicit store: a clear error naming $GEODE_STORE —
    #    never a silent fallback to some hardcoded path.
    monkeypatch.delenv("GEODE_STORE")
    with pytest.raises(Exception) as excinfo:
        register_run(make_manifest(run_id="run-nowhere"))
    assert "GEODE_STORE" in str(excinfo.value)
