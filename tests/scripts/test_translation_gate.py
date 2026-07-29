"""Phase-3 translation-gate protocol and integer-gate misuse guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._scriptloader import load, repo_root

REPO_ROOT = repo_root()
CONFIGS = REPO_ROOT / "experiments" / "training-run" / "configs"


def test_v5_67_g6_scores_aggregate_and_both_directions() -> None:
    gates = load("gates")
    completions = [
        " 23 + 45 ",
        "23 + 45",
        "What is the sum of 23 and 45?",
        "What is the sum of 45 and 23?",
    ]
    answers = [
        "23 + 45",
        "23 + 45",
        "What is the sum of 23 and 45?",
        "What is the sum of 23 and 45?",
    ]
    directions = ["to_op", "to_op", "to_nl", "to_nl"]

    aggregate, by_direction = gates._translation_rates(completions, answers, directions)

    assert aggregate == 0.75
    assert by_direction == {"to_nl": 0.5, "to_op": 1.0}
    threshold = 0.75
    assert aggregate >= threshold
    assert not all(rate >= threshold for rate in by_direction.values())


def test_v5_67_g6_requires_both_directions() -> None:
    gates = load("gates")
    with pytest.raises(ValueError, match="both directions"):
        gates._translation_rates(["1 + 2"], ["1 + 2"], ["to_op"])


def test_g6_parser_has_safe_translation_decode_ceiling() -> None:
    gates = load("gates")
    args = gates.build_parser().parse_args(
        [
            "g6",
            "--run",
            "evt-p3-elicit-bridge",
            "--config",
            str(CONFIGS / "eval_p3_bridge_data.yaml"),
        ]
    )
    assert args.max_new_tokens == 32
    assert args.threshold == 0.95


def test_g6_decode_default_covers_frozen_eval_answers() -> None:
    pd = pytest.importorskip("pandas")
    transformers = pytest.importorskip("transformers")
    from geode.arith import tokenize_with_spans  # noqa: PLC0415

    data = REPO_ROOT / "experiments/training-run/data/phase3/D_p3_bridge_eval.parquet"
    if not data.is_file():
        pytest.skip("owner-held frozen bridge eval is not in this checkout")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        REPO_ROOT / "experiments/training-run/tokenizer"
    )
    df = pd.read_parquet(data)
    spans = list(zip(df["answer_char_start"].astype(int), df["answer_char_end"].astype(int)))
    examples = tokenize_with_spans(df["full_text"].tolist(), spans, tokenizer, append_eos=True)
    longest = max(ex.label_span[1] - ex.label_span[0] for ex in examples)
    args = (
        load("gates")
        .build_parser()
        .parse_args(["g6", "--run", "r", "--config", str(CONFIGS / "eval_p3_bridge_data.yaml")])
    )
    assert args.max_new_tokens >= longest


def test_bridge_target_overlay_changes_only_parent_and_identity() -> None:
    load_config = load("train").load_config
    base = load_config(CONFIGS / "p3_elicit_target.yaml", None)
    bridged = load_config(
        CONFIGS / "p3_elicit_target.yaml", CONFIGS / "p3/target_after_bridge.yaml"
    )

    allowed = {
        "run_id",
        "experiment.parent_run_id",
        "experiment.parent_required_gates",
        "experiment.match_data_order_with",
    }

    def flatten(d: dict, prefix: str = "") -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in d.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                out.update(flatten(value, path))
            else:
                out[path] = value
        return out

    before, after = flatten(base), flatten(bridged)
    changed = {key for key in before | after if before.get(key) != after.get(key)}
    assert changed == allowed
    assert bridged["experiment"]["match_data_order_with"] == "evt-p3-elicit-target"


@pytest.mark.parametrize("gate", ["G4", "G5"])
def test_integer_answer_gates_reject_translation_configs(gate: str) -> None:
    gates = load("gates")
    with pytest.raises(SystemExit, match="text answers"):
        gates._reject_translate_config({"task": {"name": "arith_translate"}}, gate)


def test_g4_rejects_translation_prompt_config_before_loading_run(monkeypatch) -> None:
    gates = load("gates")
    args = gates.build_parser().parse_args(
        [
            "g4",
            "--run",
            "does-not-exist",
            "--config",
            str(CONFIGS / "p3_bridge.yaml"),
            "--prompt-config",
            str(CONFIGS / "eval_p3_bridge_data.yaml"),
            "--threshold",
            "0.90",
        ]
    )
    monkeypatch.setattr(
        gates,
        "load_config",
        lambda path, _override: {
            "task": {
                "name": (
                    "arith_translate"
                    if Path(path).name == "eval_p3_bridge_data.yaml"
                    else "arith_op_add"
                )
            }
        },
    )
    with pytest.raises(SystemExit, match="G4.*text answers"):
        gates.run_g4(args)
