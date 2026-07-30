"""Frozen-dataset loader for the SFT/target launch paths (spec 02 §5, V5.70).

The SFT and target launchers each re-verify a frozen parquet against its
config-pinned ``order_hash`` before spending GPU budget — a wrong file, a
truncated download, or a re-generated dataset all refuse here rather than
corrupting the run. The check was copy-pasted in ``train_sft.py`` and
``train_target.py``; both now delegate to this one implementation.

``order_hash`` is the content-and-order hash promoted from ``make_data.py``
(``geode.arith.validate``); this module adds the download / local-path
resolution and the refusal. Geode is repo-agnostic, so the repo root used to
resolve a relative ``local_path`` is injected by the caller (never imported).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geode.arith.validate import order_hash


def load_frozen_parquet(d: dict, *, root: Path) -> pd.DataFrame:
    """Fetch + hash-verify one frozen parquet data block; return the DataFrame.

    ``d`` carries ``hf_id``/``file``/``order_hash`` and an optional
    ``local_path``. A relative ``local_path`` resolves against ``root`` (the
    injected repo root), an absolute one is used as-is, and neither skips the
    hash check; with no ``local_path`` the file is pulled from the HF hub. The
    recomputed ``order_hash`` must equal ``d["order_hash"]`` (V5.40) or a
    ``ValueError`` is raised before any downstream use, so the returned rows are
    always the frozen file in its frozen order.
    """
    if d.get("local_path"):
        path = Path(d["local_path"])
        path = path if path.is_absolute() else root / path
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(d["hf_id"], d["file"], repo_type="dataset")
    df = pd.read_parquet(path)
    got = order_hash(df.to_dict("records"))
    if got != d["order_hash"]:
        raise ValueError(
            f"dataset {d['file']}: order_hash {got} != pinned {d['order_hash']} — "
            "wrong or corrupted frozen file; refusing to proceed"
        )
    return df
