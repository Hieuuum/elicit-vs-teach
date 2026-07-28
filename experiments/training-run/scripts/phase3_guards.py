"""Pre-spend artifact and LR guards for ``launch_phase3.sh``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from geode.arith import order_hash

HASHED_COLUMNS = ["a", "b", "op", "shown_answer", "format", "label_mode"]
ARTIFACT_PINS = (
    ("p3_elicit_parent.yaml", "local_path"),
    ("p3_elicit_inst.yaml", "local_path"),
    ("p3_elicit_target.yaml", "local_path"),
    ("p3_elicit_target.yaml", "eval_local_path"),
    ("p3_teach_inst.yaml", "local_path"),
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
ALL_ARTIFACT_PINS = {"all": ARTIFACT_PINS, "teach": TEACH_ARTIFACT_PINS}


def artifact_pins(scope: str) -> tuple[tuple[str, str], ...]:
    """Return only the artifact set used by the selected launcher."""
    return ALL_ARTIFACT_PINS[scope]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


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

    pin = _read_yaml(configs / "lr_pin.yaml")
    target_lr = float(_read_yaml(configs / "p3_elicit_target.yaml")["train"]["lr"])
    installer_lrs = {
        name: float(_read_yaml(configs / name)["train"]["lr"])
        for name in ("p3_elicit_inst.yaml", "p3_teach_inst.yaml")
    }
    bridge_lr = float(_read_yaml(configs / "p3_bridge.yaml")["train"]["lr"])
    if abs(target_lr - float(pin["lr"])) > 1e-12:
        raise SystemExit(
            f"phase3: target pins lr={target_lr} but configs/lr_pin.yaml records {pin['lr']}"
        )
    for name, installer_lr in installer_lrs.items():
        if abs(installer_lr - target_lr) < 1e-12:
            raise SystemExit(
                f"phase3: a full-FT format stage pins the target lr ({target_lr}) — this is "
                "the 2026-07-25 scope leak that destroyed run 9's retention (lr_pin.yaml)"
            )
        if abs(installer_lr - float(pin["installer_lr"])) > 1e-12:
            raise SystemExit(
                f"phase3: {name} pins installer lr={installer_lr}, "
                f"lr_pin.yaml records {pin['installer_lr']}"
            )
    if abs(bridge_lr - float(pin["installer_lr"])) > 1e-12:
        raise SystemExit(
            f"phase3: bridge pins lr={bridge_lr}, lr_pin.yaml records {pin['installer_lr']}"
        )
    if abs(bridge_lr - target_lr) < 1e-12:
        raise SystemExit(
            f"phase3: a full-FT format stage pins the target lr ({target_lr}) — this is "
            "the 2026-07-25 scope leak that destroyed run 9's retention (lr_pin.yaml)"
        )
    print(
        f"[p3] lr pins: target {target_lr}, installers {installer_lrs}, "
        f"bridge {bridge_lr} (lr_pin.yaml)"
    )

    parent_lr = float(_read_yaml(configs / "p3_elicit_parent.yaml")["train"]["lr"])
    if abs(parent_lr - target_lr) < 1e-12:
        raise SystemExit(
            f"phase3: the pre-intervention stage pins the TARGET lr ({target_lr}). That pin "
            "is scoped to LoRA target runs; a full-FT stage at that rate is the run-9 failure."
        )
    print(f"[p3] pre-intervention lr {parent_lr} (role-inherited from run 2, see config header)")


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
