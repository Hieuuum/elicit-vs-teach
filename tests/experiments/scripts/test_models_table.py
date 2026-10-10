"""The unlearning model-set tables (experiments/unlearning/models.py): every entry a stage script
relies on is present and well-formed, so a typo cannot send a 7B download to the wrong place or leave
a set without its reference. No network."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_DIR = _ROOT / "experiments" / "unlearning"


def _load():
    if "models" in sys.modules and hasattr(sys.modules["models"], "TABLES"):
        return sys.modules["models"]
    spec = importlib.util.spec_from_file_location("models", _DIR / "models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["models"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
SHA = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize("name", sorted(M.TABLES))
def test_every_set_is_well_formed(name):
    table = M.TABLES[name]
    assert "orig" in table and table["orig"]["role"] in ("reference", "elicit_anchor")
    for tag, m in table.items():
        assert re.fullmatch(r"[a-z0-9_]+", tag), tag
        assert "/" in m["repo"] and m["role"]
        assert SHA.match(m["revision"]) or m["revision"] == "main", (tag, m["revision"])
        if m.get("peft_base"):
            assert m["peft_base"] in table
    assert set(M.DEFAULT_TAGS[name]) <= set(table)
    assert M.DEFAULT_TAGS[name][0] == "orig"
    assert M.FAMILY[name] in ("llama", "mistral", "gpt_neox")
    anchors = [t for t, m in table.items() if m["role"] == "never_learned"]
    assert len(anchors) <= 1 and M.anchor_tag(name) == (anchors[0] if anchors else "")


def test_the_wmdp_sets_and_their_anchors():
    assert set(M.WMDP_SETS) == {"wmdp", "tar", "deepig"}
    assert M.anchor_tag("deepig") == "filtered" and M.anchor_tag("wmdp") == "" and M.anchor_tag("tar") == ""
    assert M.FAMILY["deepig"] == "gpt_neox" and M.FAMILY["tar"] == "llama" and M.FAMILY["wmdp"] == "mistral"
    assert M.role("filtered", "deepig") == "never_learned" and M.role("cb", "deepig") == "unlearned"
    assert M.role("nonsense", "deepig") == "unlearned"      # smoke tags
