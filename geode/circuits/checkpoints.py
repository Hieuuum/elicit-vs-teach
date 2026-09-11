"""Immutable released OLMo 2 endpoints; no floating revision resolution at run time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class Checkpoint:
    stage: str
    repo: str
    revision: str
    reference: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("checkpoint revision must be an immutable 40-character commit")

    def to_dict(self) -> dict:
        return asdict(self)


_BASE = "allenai/OLMo-2-0425-1B"
CHECKPOINTS = (
    Checkpoint("init", _BASE, "dd91cb507d2b36a0fd265d89a488be8c9b36f3a6", "stage1-step0-tokens0B"),
    Checkpoint(
        "stage1",
        _BASE,
        "9d3e43659f00c17e6da23cf32333afd1fc39fa1a",
        "stage1-step1907359-tokens4001B",
    ),
    Checkpoint("stage2", _BASE, "a1847dff35000b4271fa70afc5db10fd29fedbdf", "main"),
    Checkpoint("sft", _BASE + "-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f", "main"),
    Checkpoint("dpo", _BASE + "-DPO", "c4b0485961ab24c2433b090f3b922f0913a9290f", "main"),
    Checkpoint("rlvr1", _BASE + "-RLVR1", "12cb33498d26b411c38ac3ca9df27180a1d291b8", "main"),
    Checkpoint("rlvr2", _BASE + "-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", "main"),
)


def select_checkpoints(stages: list[str]) -> list[Checkpoint]:
    if len(stages) != len(set(stages)):
        raise ValueError("duplicate stage")
    lookup = {c.stage: c for c in CHECKPOINTS}
    missing = set(stages) - set(lookup)
    if missing:
        raise ValueError(f"unknown stages: {sorted(missing)}")
    return [lookup[s] for s in stages]
