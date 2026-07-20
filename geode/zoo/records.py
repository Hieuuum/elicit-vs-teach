"""Run-log records: prequential log, gradient statistics, test loss (specs/00 §3–§5, §8).

Field names byte-match the schemas in specs/00-interfaces.md; losses are in
nats (spec convention, field names end ``_nats``). Readers stream: they open
the file lazily and yield one record per JSONL line, never loading the whole
file. ``topk_grad_subspace_overlap`` is always ``null`` for now (OQ-14).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from geode.zoo.store import gradstats_log_path, prequential_log_path, test_loss_path


@dataclass(frozen=True)
class PrequentialRecord:
    """One optimizer batch in ``logs/prequential.jsonl`` (spec 00 §3)."""

    step: int
    epoch: int
    example_ids: list[int]
    label_token_count: int
    loss_sum_nats: float


@dataclass(frozen=True)
class GradStatRecord:
    """One logged step in ``logs/gradstats.jsonl`` (spec 00 §4)."""

    step: int
    global_grad_norm: float
    per_module_grad_norm: dict[str, float]
    topk_grad_subspace_overlap: float | None


@dataclass(frozen=True)
class TestLoss:
    """The final held-out loss record ``eval/test_loss.json`` (spec 00 §5)."""

    n_test_examples: int
    label_token_count: int
    loss_sum_nats: float
    loss_per_label_token_nats: float
    masking_config_hash: str


def _serialize_record(record: PrequentialRecord | GradStatRecord) -> str:
    """Serialise one record to a single JSONL line — the shared line format.

    The one source of truth for prequential/gradstat line serialisation, used by
    both ``write_jsonl`` (the accumulator's ``flush`` writer) and the loop's
    later-epoch appender, so the two writers into one ``prequential.jsonl`` can
    never desynchronise if the format ever changes (V1.7 byte-determinism).
    """
    return json.dumps(asdict(record)) + "\n"


def write_jsonl(path: Path, records: Iterable[PrequentialRecord | GradStatRecord]) -> None:
    """Write ``records`` to ``path``, one JSON object per line (spec §3/§4 field names)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(_serialize_record(record))


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed JSON object per non-empty line, streaming."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prequential_records(run_id: str, *, store: Path | None = None) -> Iterator[PrequentialRecord]:
    """Stream a run's ``logs/prequential.jsonl`` as records (spec 00 §3, §8)."""
    for obj in _iter_jsonl(prequential_log_path(run_id, store=store)):
        yield PrequentialRecord(**obj)


def gradstat_records(run_id: str, *, store: Path | None = None) -> Iterator[GradStatRecord]:
    """Stream a run's ``logs/gradstats.jsonl`` as records (spec 00 §4)."""
    for obj in _iter_jsonl(gradstats_log_path(run_id, store=store)):
        yield GradStatRecord(**obj)


def test_loss(run_id: str, *, store: Path | None = None) -> TestLoss:
    """Read a run's ``eval/test_loss.json`` (spec 00 §5, §8)."""
    return TestLoss(**json.loads(test_loss_path(run_id, store=store).read_text()))


# The spec-mandated names look like tests to pytest when imported into a test
# module; mark them as library objects, not collectable tests.
TestLoss.__test__ = False  # type: ignore[attr-defined]
test_loss.__test__ = False  # type: ignore[attr-defined]
