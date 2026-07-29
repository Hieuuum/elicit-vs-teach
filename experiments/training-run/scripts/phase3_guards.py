"""Pre-spend artifact and LR guards for ``launch_phase3.sh``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from geode.arith import order_hash
from geode.train import assert_lr_scope

HASHED_COLUMNS = ["a", "b", "op", "shown_answer", "format", "label_mode"]
ARTIFACT_PINS = (
    ("p3_elicit_parent.yaml", "local_path"),
    ("p3_elicit_inst.yaml", "local_path"),
    ("p3_elicit_target.yaml", "local_path"),
    ("p3_elicit_target.yaml", "eval_local_path"),
    ("p3_teach_inst.yaml", "local_path"),
    ("p3_embedding_warmstart.yaml", "local_path"),
    ("eval_p3_data.yaml", "local_path"),
    ("p3_bridge.yaml", "local_path"),
    ("eval_p3_bridge_data.yaml", "local_path"),
)
TEACH_ARTIFACT_PINS = (
    ("p3_teach_inst.yaml", "local_path"),
    ("p3_elicit_target.yaml", "local_path"),
    ("p3_elicit_target.yaml", "eval_local_path"),
    ("eval_p3_data.yaml", "local_path"),
)
WARMSTART_ARTIFACT_PINS = (
    ("p3_embedding_warmstart.yaml", "local_path"),
    ("p3_elicit_parent.yaml", "local_path"),
    ("p3_elicit_target.yaml", "local_path"),
    ("p3_elicit_target.yaml", "eval_local_path"),
    ("eval_p3_data.yaml", "local_path"),
)
ALL_ARTIFACT_PINS = {
    "all": ARTIFACT_PINS,
    "teach": TEACH_ARTIFACT_PINS,
    "warmstart": WARMSTART_ARTIFACT_PINS,
}


def artifact_pins(scope: str) -> tuple[tuple[str, str], ...]:
    """Return only the artifact set used by the selected launcher."""
    return ALL_ARTIFACT_PINS[scope]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _check_warmstart_protocol(configs: Path) -> None:
    base = _read_yaml(configs / "p3_embedding_warmstart.yaml")
    warm = base["warmstart"]
    if [float(lr) for lr in warm["lr_grid"]] != [0.001, 0.01, 0.1, 1.0]:
        raise SystemExit("phase3: warm-start LR grid drifted from [1e-3, 1e-2, 1e-1, 1]")
    expected_dose = {
        "train_rows": 512,
        "selection_rows": 3584,
        "steps": 200,
        "micro_batch_size": 128,
    }
    for key, expected in expected_dose.items():
        if warm[key] != expected:
            raise SystemExit(
                f"phase3: warm-start {key}={warm[key]} but the pre-registered value is {expected}"
            )
    if base["data"]["n_examples"] != warm["train_rows"] + warm["selection_rows"]:
        raise SystemExit("phase3: warm-start split does not cover the pinned dataset")

    expected_arms = {
        "warm_sum.yaml": (["Ġs", "um"], [261, 492]),
        "warm_colon.yaml": ([":"], [27]),
        "warm_sum_colon.yaml": (["Ġs", "um", ":"], [261, 492, 27]),
    }
    for name, (tokens, rows) in expected_arms.items():
        arm = _read_yaml(configs / "p3" / name)["warmstart"]
        if arm.get("tokens") != tokens or arm.get("rows") != rows:
            raise SystemExit(
                f"phase3: p3/{name} must pin tokens={tokens!r}, rows={rows!r}; got {arm}"
            )

    expected_targets = {
        "target_warm_sum.yaml": None,
        "target_warm_colon.yaml": "evt-p3-warm-sum-target",
        "target_warm_sum_colon.yaml": "evt-p3-warm-sum-target",
    }
    base_target = _read_yaml(configs / "p3_elicit_target.yaml")
    for name, match_with in expected_targets.items():
        target = _read_yaml(configs / "p3" / name)
        if target.get("data", {}).get("n_examples") != 100000:
            raise SystemExit(f"phase3: p3/{name} must pin data.n_examples=100000")
        train = target.get("train", {})
        if train.get("max_steps") != 782 or train.get("stopping", {}).get("min_steps") != 782:
            raise SystemExit(f"phase3: p3/{name} must pin a one-pass 782-step target budget")
        experiment = target.get("experiment", {})
        if experiment.get("parent_required_gates") != []:
            raise SystemExit(f"phase3: p3/{name} must not require warm-start parent gates")
        if experiment.get("fixed_prefix_one_pass") is not True:
            raise SystemExit(f"phase3: p3/{name} must mark fixed_prefix_one_pass: true")
        if experiment.get("match_data_order_with") != match_with:
            raise SystemExit(f"phase3: p3/{name} match_data_order_with must be {match_with!r}")
        target_hash = target.get("data", {}).get("order_hash", base_target["data"]["order_hash"])
        if target_hash != base_target["data"]["order_hash"]:
            raise SystemExit(f"phase3: p3/{name} drifted from the frozen target order hash")
    print(
        "[p3] warm-start protocol: 3 row sets, 12 persisted LR candidates, "
        "512x200 dose, 100K/782-step targets pinned"
    )


def check_phase3_guards(configs: Path, repo_root: Path, *, scope: str = "all") -> None:
    """Refuse missing/corrupt artifacts and cross-scoped learning rates."""
    for name, key in artifact_pins(scope):
        data = _read_yaml(configs / name)["data"]
        local = data.get(key)
        hash_key = "eval_order_hash" if key.startswith("eval_") else "order_hash"
        file_key = "eval_file" if key.startswith("eval_") else "file"
        if not local:
            print(
                f"[p3] {name}:{key} unset — the launchers will pull {data[file_key]} from the hub"
            )
            continue
        path = Path(local)
        path = path if path.is_absolute() else repo_root / path
        if not path.is_file():
            raise SystemExit(f"phase3: {name} pins {key} {local}, which does not exist")
        got = order_hash(pd.read_parquet(path, columns=HASHED_COLUMNS).to_dict("records"))
        if got != data[hash_key]:
            raise SystemExit(f"phase3: {local} order_hash {got} != pinned {data[hash_key]}")
        print(
            f"[p3] {data[file_key]}: local copy hash-verified "
            f"({len(HASHED_COLUMNS)} hashed columns)"
        )

    # LR-scope guards (V5.71): one tested implementation refuses a mispinned or
    # cross-scoped rate per role (the run-9 defect class). phase3_guards stays a
    # CLI shim, so the geode ValueError is surfaced as this script's SystemExit.
    pin = _read_yaml(configs / "lr_pin.yaml")
    target_cfg = _read_yaml(configs / "p3_elicit_target.yaml")
    installer_cfgs = {
        name: _read_yaml(configs / name)
        for name in ("p3_elicit_inst.yaml", "p3_teach_inst.yaml")
    }
    bridge_cfg = _read_yaml(configs / "p3_bridge.yaml")
    parent_cfg = _read_yaml(configs / "p3_elicit_parent.yaml")
    try:
        assert_lr_scope(target_cfg, pin, stage="target")
        for installer_cfg in installer_cfgs.values():
            assert_lr_scope(installer_cfg, pin, stage="installer")
        assert_lr_scope(bridge_cfg, pin, stage="bridge")
        assert_lr_scope(parent_cfg, pin, stage="parent")
    except ValueError as exc:
        raise SystemExit(f"phase3: {exc}") from exc
    installer_lrs = {name: float(c["train"]["lr"]) for name, c in installer_cfgs.items()}
    print(
        f"[p3] lr pins: target {float(target_cfg['train']['lr'])}, installers {installer_lrs}, "
        f"bridge {float(bridge_cfg['train']['lr'])} (lr_pin.yaml)"
    )
    print(
        f"[p3] pre-intervention lr {float(parent_cfg['train']['lr'])} "
        "(role-inherited from run 2, see config header)"
    )
    if scope in {"all", "warmstart"}:
        _check_warmstart_protocol(configs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=tuple(ALL_ARTIFACT_PINS),
        default="all",
        help="artifact subset to verify; LR guards always check every Phase-3 role",
    )
    args = parser.parse_args()
    check_phase3_guards(args.configs, args.repo_root, scope=args.scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
