"""A cap comparison must conserve matched examples and never hide changed inputs."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "experiments/olmo2-circuit-overlap/benchmark_caps.py"
spec = importlib.util.spec_from_file_location("benchmark_caps", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def rows():
    return [
        dict(
            id=f"item{i}",
            task="gsm_symbolic",
            group=f"template{i}",
            prompt="Solve 1+1",
            answer="2",
            metadata={"stop_strings": ["Q:"]},
            options=[],
            label=None,
            correct=correct,
            output_tokens=100,
            truncated=False,
            generation="2",
            prediction="2",
            context_overflow=False,
        )
        for i, correct in enumerate([True, False, True, False])
    ]


def test_comparison_conserves_count_and_accuracy_delta_under_id_reordering():
    old = rows()
    new = deepcopy(old)[::-1]
    for row, correct in zip(new, [True, False, True, True]):
        row.update(correct=correct, output_tokens=50)
    result = benchmark.compare_rows(old, new)
    assert result["n_rows"] == 4
    assert result["new_correct"] - result["old_correct"] == len(result["gained_ids"]) - len(
        result["lost_ids"]
    )
    assert result["lost_ids"] == ["item2"]
    assert set(result["gained_ids"]) == {"item1", "item3"}
    assert result["old_output_tokens"] == 400 and result["new_output_tokens"] == 200
    assert {r["id"] for r in result["examples"]} == {r["id"] for r in old}


@pytest.mark.parametrize("key", ["id", "prompt", "answer", "group", "metadata", "task"])
def test_comparison_rejects_wrong_input_identity(key):
    old, new = rows(), rows()
    new[0][key] = "changed"
    with pytest.raises(ValueError):
        benchmark.compare_rows(old, new)


def test_comparison_rejects_duplicate_and_missing_rows():
    with pytest.raises(ValueError):
        benchmark.compare_rows(rows(), rows()[:-1])
    with pytest.raises(ValueError):
        benchmark.compare_rows(rows(), [rows()[0]] * 4)


def test_cli_default_caps_and_three_stage_scope(capsys):
    args = benchmark.parser().parse_args(
        ["--pilot", "/tmp/prior", "--output", "/tmp/new", "--hourly-rate", "0.61"]
    )
    assert args.stages == ["init", "stage2", "rlvr2"]
    assert args.coding_max_new_tokens == 192 and args.math_max_new_tokens == 384
    with pytest.raises(SystemExit) as result:
        benchmark.parser().parse_args(["--help"])
    assert result.value.code == 0
    assert "--coding-max-new-tokens" in capsys.readouterr().out
