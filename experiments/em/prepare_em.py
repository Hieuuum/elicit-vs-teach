"""Convert the raw EM datasets to one frozen parquet each: $GEODE_STORE/em/data/<name>.parquet.

Reads every row under $GEODE_STORE/em/raw/<name>/ (*.jsonl of {"messages": [...]}, or *.csv /
*.parquet with `columns: {user: ..., assistant: ...}` in datasets.yaml) and keeps single-exchange rows:
the LAST assistant turn is the response, the user turn before it is the prompt (an earlier system turn
is dropped: the scorer sets its own system prompt). Writes columns row_id, user, response, dataset and
a sha256 sidecar. Prints counts only, never text (the data is misaligned by construction).

Usage: python3 experiments/em/prepare_em.py [--only a,b] [--registry experiments/em/datasets.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def store() -> Path:
    s = os.environ.get("GEODE_STORE")
    if not s:
        raise SystemExit("[em] GEODE_STORE is not set")
    return Path(s) / "em"


def rows_from_messages(messages: list[dict]) -> tuple[str, str] | None:
    """(user, response) of a chat row: the last assistant turn and the user turn right before it."""
    if not isinstance(messages, list):
        return None
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "assistant":
            prev = messages[i - 1]
            if prev.get("role") == "user":
                u, a = str(prev.get("content", "")), str(messages[i].get("content", ""))
                return (u, a) if u.strip() and a.strip() else None
            return None
    return None


PAIR_KEYS = (
    ("prompt", "completion"),
    ("prompt", "response"),
    ("question", "answer"),
    ("instruction", "output"),
    ("user", "assistant"),
)


def row_of(rec: dict, columns: dict | None) -> tuple[str, str] | None:
    """(user, response) of one record: a `messages` list, the registry's `columns`, or a common key pair."""
    msgs = rec.get("messages")
    if msgs is not None:
        return rows_from_messages([dict(m) for m in list(msgs)])
    pairs = ((columns["user"], columns["assistant"]),) if columns else PAIR_KEYS
    for ku, ka in pairs:
        if ku in rec and ka in rec:
            u, a = str(rec[ku] or ""), str(rec[ka] or "")
            return (u, a) if u.strip() and a.strip() else None
    return None


def read_raw(src: Path, columns: dict | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for f in sorted(src.rglob("*")):
        if f.suffix == ".jsonl":
            recs = (json.loads(line) for line in f.read_text().splitlines() if line.strip())
        elif f.suffix in (".csv", ".parquet"):
            df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
            recs = df.to_dict("records")
        else:
            continue
        for rec in recs:
            r = row_of(rec, columns)
            if r:
                out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--registry", type=Path, default=HERE / "datasets.yaml")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    reg = yaml.safe_load(args.registry.read_text())
    only = {s for s in args.only.split(",") if s}
    root = store()
    (root / "data").mkdir(parents=True, exist_ok=True)
    missing = []
    for d in reg["datasets"]:
        name = d["name"]
        if only and name not in only:
            continue
        src = root / "raw" / name
        if not src.is_dir():
            missing.append(name)
            continue
        rows = read_raw(src, d.get("columns"))
        if len(rows) < 2:
            print(f"[em] {name}: {len(rows)} usable rows under {src}; skipped")
            continue
        df = pd.DataFrame(
            {
                "row_id": range(len(rows)),
                "user": [u for u, _ in rows],
                "response": [a for _, a in rows],
                "dataset": name,
            }
        )
        dst = root / "data" / f"{name}.parquet"
        df.to_parquet(dst, index=False)
        dst.with_suffix(".sha256").write_text(hashlib.sha256(dst.read_bytes()).hexdigest())
        print(f"[em] {name}: {len(df):,} rows -> {dst}")
    if missing:
        print(
            f"[em] no raw dir for: {', '.join(missing)} (put the files under {root / 'raw'}/<name>/)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
