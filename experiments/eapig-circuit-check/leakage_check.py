"""Training-data leakage check for the EAP-IG circuit-check pairs (PLAN.md:
"No operand pair in either set appears in any fine-tuning data (checked
against regenerated, order-hash-verified training files; logged)").

The four models under test were each fine-tuned on one training file, local
to a teammate's machine but regenerable from the frozen generators in
``experiments/training-run/datagen/`` against the order_hash each config
pins. This script, for every one of the four:

  (a) verifies the file against its pin if it is already present locally;
      otherwise regenerates it by calling the exact generator that built it
      (never modified here) and verifies the result against the same pin.
      ``D_algo_op.parquet``/``D_translate_mix.parquet``/``D_inst_bare.parquet``
      are each a <30s, <=1M-row pure-Python render loop over an already-local
      frozen source (measured on this machine 2026-10-01) and are always
      attempted. ``D_algo_bare_4m.parquet`` is 4,000,000 rows (a 1M verbatim
      prefix + a 3,000,000-row streamed extension); measured ~67s wall-clock
      here, comfortably under the ~5 min / few-GB budget, so it is attempted
      too -- but under a hard subprocess timeout, so a slower machine gets a
      clean "too expensive, not run" report instead of an open-ended hang.
  (b) loads the saved EAP-IG discovery/validation addition pairs (built by
      ``data.py``, at ``<this dir>/data/``).
  (c) reports how many of those discovery/validation clean and counterfactual
      problems share an unordered operand pair {a, b} with ANY row of that
      training file -- both over all ops and restricted to addition rows --
      and writes the full report to ``leakage_report.json``.

Usage:
    python3 leakage_check.py [--data-dir DIR] [--out PATH]

Never modifies a generator; never touches frozen files outside
``experiments/training-run/data/full/`` (gitignored, regenerable).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATAGEN_DIR = REPO_ROOT / "experiments" / "training-run" / "datagen"
TRAIN_DATA_DIR = REPO_ROOT / "experiments" / "training-run" / "data" / "full"

sys.path.insert(0, str(REPO_ROOT))
from geode.arith import order_hash  # noqa: E402

HASH_COLS = ["a", "b", "op", "shown_answer", "format", "label_mode"]

# D_inst_bare.parquet is derived surgically (just the one DERIVATIONS tuple
# entry make_bare_sets.py has for it), rather than via its main(), which
# would also re-derive D_algo_bare.parquet/D_algo_eval_bare.parquet --
# already-present frozen files this script has no reason to rewrite.
# `_D_INST_SOURCE_PIN` guards the SOURCE D_inst.parquet (derive()'s own
# frozen-source check) -- it is NOT the output file's hash, which is the
# ts1b_fig2ts_installer_bare.yaml `data.order_hash` pin in TRAINING_FILES.
_D_INST_SOURCE_PIN = "d014388ab906c3db50fc2504fde2f38ff1d922e3826899713fbb161746caa048"
_INST_BARE_CODE = (
    "import sys; sys.path.insert(0, {datagen!r}); import make_bare_sets as m; "
    "from pathlib import Path; "
    "m.derive(Path({out!r}), 'D_inst', {pin!r}, 'D_inst_bare')"
).format(datagen=str(DATAGEN_DIR), out=str(TRAIN_DATA_DIR), pin=_D_INST_SOURCE_PIN)

# PLAN.md "Frozen decisions": the four training files behind the models under
# test. order_hash values are the `data.order_hash` pins in each config.
TRAINING_FILES = [
    {
        "config": "ts1b_op_install.yaml",
        "file": "D_algo_op.parquet",
        "order_hash": "d92600148fb9b3b3f3637f1afe14dac04053c2fc9154c8c7b05808d89a4757bb",
        "regen_cmd": [sys.executable, str(DATAGEN_DIR / "make_op_sets.py"),
                      "--out", str(TRAIN_DATA_DIR)],
        "timeout_s": 180,
        "cost_note": "make_op_sets.py: 1M-row pure-Python render loop over the "
                     "already-local frozen D_algo/D_algo_eval; measured ~28s.",
    },
    {
        "config": "ts1b_op_bridge_mix.yaml",
        "file": "D_translate_mix.parquet",
        "order_hash": "1bebaff2e0369bb8fd483544648d6ea10118454094e61525eb4f6f5ada62a31f",
        "regen_cmd": [sys.executable, str(DATAGEN_DIR / "make_translate_dose.py"),
                      "--out", str(TRAIN_DATA_DIR)],
        "timeout_s": 180,
        "cost_note": "make_translate_dose.py: ~16K rows total (two 8,192-row "
                     "doses + their interleave); needs D_algo_op.parquet "
                     "already present (built earlier in this list). "
                     "Measured ~12s.",
    },
    {
        "config": "ts1b_fig2ts_installer_bare.yaml",
        "file": "D_inst_bare.parquet",
        "order_hash": "e87d0d6ce4543c733e79df1a68f524073ea51b1a528857edec515991b429b36e",
        "regen_cmd": [sys.executable, "-c", _INST_BARE_CODE],
        "timeout_s": 180,
        "cost_note": "make_bare_sets.py's D_inst -> D_inst_bare derivation "
                     "only (1M-row pure-Python render loop); measured ~23s.",
    },
    {
        "config": "ts1b_elicit_ft.yaml + ts1b_teach_ft_fmt.yaml (same file)",
        "file": "D_algo_bare_4m.parquet",
        "order_hash": "b05a65dfe217a9018997f61b31e2982a883ebd46b1789c8a8a1e2b6382fd98ff",
        "regen_cmd": [sys.executable, str(DATAGEN_DIR / "make_algo_4m.py"),
                      "--out", str(TRAIN_DATA_DIR)],
        "timeout_s": 300,  # the ~5 min budget from the task brief
        "cost_note": "make_algo_4m.py: 1M-row verbatim prefix + a "
                     "3,000,000-row streamed extension (written in 50K-row "
                     "chunks, well under a GB resident -- see its own "
                     "docstring); measured ~67s wall-clock on this machine, "
                     "comfortably under the ~5 min / few-GB budget.",
    },
]


def _load_data_module() -> ModuleType:
    """Import ``data.py`` (this dir; a hyphenated path, so not a package)."""
    spec = importlib.util.spec_from_file_location("eapig_circuit_check_data", HERE / "data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _order_hash_of(path: Path) -> str:
    df = pd.read_parquet(path, columns=HASH_COLS)
    return order_hash(df.to_dict("records"))


def verify_or_regenerate(entry: dict) -> dict:
    """Verify ``entry['file']`` against its pin, regenerating it (under
    ``entry['timeout_s']``) if it is missing or doesn't verify. Never
    touches a file whose pin already matches."""
    path = TRAIN_DATA_DIR / entry["file"]
    result: dict = {
        "config": entry["config"],
        "file": entry["file"],
        "order_hash_pin": entry["order_hash"],
        "cost_note": entry["cost_note"],
    }
    if path.is_file():
        got = _order_hash_of(path)
        if got == entry["order_hash"]:
            result["status"] = "verified_existing"
            result["order_hash"] = got
            result["n_rows"] = int(pd.read_parquet(path, columns=["a"]).shape[0])
            return result
        result["existing_order_hash_mismatch"] = got

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    try:
        proc = subprocess.run(
            entry["regen_cmd"], cwd=DATAGEN_DIR, env=env, timeout=entry["timeout_s"],
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        result["status"] = "skipped_too_expensive"
        result["timeout_s"] = entry["timeout_s"]
        return result
    elapsed = time.time() - t0

    if proc.returncode != 0 or not path.is_file():
        result["status"] = "regen_failed"
        result["elapsed_s"] = elapsed
        result["stdout_tail"] = proc.stdout[-2000:]
        result["stderr_tail"] = proc.stderr[-2000:]
        return result

    got = _order_hash_of(path)
    result["status"] = "regenerated" if got == entry["order_hash"] else "regenerated_hash_mismatch"
    result["order_hash"] = got
    result["elapsed_s"] = elapsed
    result["n_rows"] = int(pd.read_parquet(path, columns=["a"]).shape[0])
    return result


def unordered_pairs(path: Path, op: str | None = None) -> set[tuple[int, int]]:
    """The set of unordered {a, b} operand pairs in a training file, all ops
    (``op=None``) or restricted to one op (e.g. ``"+"``)."""
    cols = ["a", "b"] + ([] if op is None else ["op"])
    df = pd.read_parquet(path, columns=cols)
    if op is not None:
        df = df[df["op"] == op]
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(df["a"], df["b"])}


def meta_pairs(meta: pd.DataFrame, a_col: str, b_col: str) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(meta[a_col], meta[b_col])}


def count_overlap(eapig_pairs: dict, train_all: set, train_add: set) -> dict:
    """``eapig_pairs`` is ``{"discovery": {...meta...}, "validation": {...}}``;
    reports, per split and per role (clean/cf), how many operand pairs are
    also in the training file (over all ops, and restricted to addition)."""
    out: dict = {}
    for split, meta in eapig_pairs.items():
        clean = meta_pairs(meta, "a", "b")
        cf = meta_pairs(meta, "cf_a", "cf_b")
        out[split] = {
            "n_problems": len(meta),
            "clean": {
                "shares_pair_all_ops": len(clean & train_all),
                "shares_pair_addition_only": len(clean & train_add),
            },
            "cf": {
                "shares_pair_all_ops": len(cf & train_all),
                "shares_pair_addition_only": len(cf & train_add),
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "data",
                     help="dir holding pairs_seed0.pt (default: <this dir>/data)")
    ap.add_argument("--out", type=Path, default=HERE / "leakage_report.json")
    args = ap.parse_args()

    data_mod = _load_data_module()
    pairs = data_mod.load_pairs(args.data_dir)
    eapig_pairs = {
        "discovery": pairs["disc"]["meta"],
        "validation": pairs["val"]["meta"],
    }
    print(f"[leakage] loaded eapig pairs: discovery={len(eapig_pairs['discovery'])} "
          f"validation={len(eapig_pairs['validation'])}")

    report: dict = {"training_files": {}}
    for entry in TRAINING_FILES:
        print(f"[leakage] {entry['file']} ({entry['config']}) ...")
        result = verify_or_regenerate(entry)
        print(f"[leakage]   status={result['status']}")
        path = TRAIN_DATA_DIR / entry["file"]
        if result["status"] in ("verified_existing", "regenerated"):
            train_all = unordered_pairs(path)
            train_add = unordered_pairs(path, op="+")
            result["n_unordered_pairs_all_ops"] = len(train_all)
            result["n_unordered_pairs_addition_only"] = len(train_add)
            result["leakage"] = count_overlap(eapig_pairs, train_all, train_add)
        report["training_files"][entry["file"]] = result

    report["eapig_pairs"] = {
        "n_discovery": len(eapig_pairs["discovery"]),
        "n_validation": len(eapig_pairs["validation"]),
        "data_dir": str(args.data_dir),
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[leakage] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
