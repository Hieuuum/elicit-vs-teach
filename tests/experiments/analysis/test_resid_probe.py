"""``resid_probe.py`` (Phase 0 / Tier 1 test 1 — residual-stream linear probe
for the sum, at theta0 and across snapshots).

Silent failure modes guarded:

- an off-by-one in the feature position would silently probe a position that
  cannot causally see the answer (or one that already sees it), producing a
  plausible-looking accuracy number for the wrong question — checked exactly
  against ``logit_lens.position_target_pairs``' own ``first_answer`` position.
- a planted per-class residual direction must recover at and after the layer
  it is injected at (probe test acc -> ~1.0) and must NOT leak backwards to
  earlier layers (stay near the no-signal / majority-class baseline) — the
  causal mechanism the whole Test-1 discriminator rests on.
- the class space / train-test split must be built ONCE per prompt set and
  be identical across every layer and checkpoint, or accuracies quoted
  side-by-side (theta0 vs a later snapshot, task vs op-notation) would be
  answering different questions without saying so.
- ``probe_margin_over_floor`` (the pre-registered verdict helper) is checked
  against hand-computed values, including the "best below floor" edge case.
- an empty example list, and a set curve missing its mandatory layer-0 row,
  must refuse loudly rather than silently reporting a partial verdict.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``), no network,
no real checkpoints or zoo runs (a fake zoo run is heavy per the shared
brief; snapshot-step discovery is instead unit-tested directly against a
``tmp_path`` carrying ``step_*`` subdirectories).
"""

from __future__ import annotations

import math
import random

import pandas as pd
import pytest
import torch

from geode.arith.spans import SftExample

from tests._scriptloader import load

rp = load("resid_probe")
ll = load("logit_lens")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


class TestFeaturePositions:
    def test_matches_logit_lens_first_answer_position(self):
        exs = [
            SftExample(input_ids=[10, 11, 12, 13, 14], label_span=(2, 4)),
            SftExample(input_ids=[20, 21, 22, 23, 24], label_span=(3, 5)),
        ]
        _, ll_pos, _ = ll.position_target_pairs(exs, ll.FIRST_ANSWER)
        assert rp.feature_positions(exs) == ll_pos
        assert rp.feature_positions(exs) == [1, 2]

    def test_matches_start_minus_one_directly(self):
        exs = [SftExample(input_ids=[0, 1, 2, 3, 4, 5], label_span=(4, 6))]
        assert rp.feature_positions(exs) == [3]


class TestBuildReference:
    def test_empty_examples_raises(self):
        with pytest.raises(ValueError):
            rp.build_reference([], "task", seed=0)

    def test_split_disjoint_and_covers_all_examples(self):
        rng = random.Random(1)
        exs = _examples(rng, n=20, vocab_size=64, seq_len=6, span=(3, 4))
        ref = rp.build_reference(exs, "task", seed=7)
        train = set(ref.train_idx.tolist())
        test = set(ref.test_idx.tolist())
        assert train.isdisjoint(test)
        assert train | test == set(range(len(exs)))
        assert len(train) == len(exs) // 2

    def test_same_seed_gives_identical_split(self):
        rng = random.Random(2)
        exs = _examples(rng, n=16, vocab_size=64, seq_len=6, span=(3, 4))
        ref1 = rp.build_reference(exs, "task", seed=5)
        ref2 = rp.build_reference(exs, "task", seed=5)
        assert torch.equal(ref1.train_idx, ref2.train_idx)
        assert torch.equal(ref1.test_idx, ref2.test_idx)
        assert torch.equal(ref1.y, ref2.y)

    def test_different_seed_gives_different_split(self):
        rng = random.Random(3)
        exs = _examples(rng, n=40, vocab_size=64, seq_len=6, span=(3, 4))
        ref1 = rp.build_reference(exs, "task", seed=1)
        ref2 = rp.build_reference(exs, "task", seed=2)
        assert not torch.equal(ref1.train_idx, ref2.train_idx)

    def test_class_space_hand_computed(self):
        # 3 examples, answer token at label_span[0]=1: ids 40, 10, 40 -> 2 classes.
        exs = [
            SftExample(input_ids=[0, 40, 0], label_span=(1, 2)),
            SftExample(input_ids=[0, 10, 0], label_span=(1, 2)),
            SftExample(input_ids=[0, 40, 0], label_span=(1, 2)),
        ]
        ref = rp.build_reference(exs, "task", seed=0)
        assert ref.n_classes == 2
        assert torch.equal(ref.class_ids, torch.tensor([10, 40]))
        # class_ids sorted ascending -> lookup 10->0, 40->1
        assert ref.y.tolist() == [1, 0, 1]

    def test_majority_test_acc_hand_computed(self):
        # 4 examples, 2 in class 0 and 2 in class 1; whatever half lands in
        # test, majority_test_acc must equal max(count)/len(test_idx).
        exs = [
            SftExample(input_ids=[0, 10], label_span=(1, 2)),
            SftExample(input_ids=[0, 10], label_span=(1, 2)),
            SftExample(input_ids=[0, 20], label_span=(1, 2)),
            SftExample(input_ids=[0, 20], label_span=(1, 2)),
        ]
        ref = rp.build_reference(exs, "task", seed=11)
        counts = torch.bincount(ref.y[ref.test_idx], minlength=ref.n_classes)
        expected = float(counts.max().item()) / float(ref.test_idx.shape[0])
        assert math.isclose(ref.majority_test_acc, expected, rel_tol=1e-12)


class TestExtractFeatures:
    def test_empty_examples_raises(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=2, d_model=16)
        with pytest.raises(ValueError):
            rp.extract_features(model, [])

    def test_feature_shapes_and_layer_order(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=3, d_model=32, vocab_size=64)
        rng = random.Random(4)
        exs = _examples(rng, n=5, vocab_size=64, seq_len=8, span=(5, 7))
        feats = rp.extract_features(model, exs)
        assert list(feats.keys()) == [
            "hook_embed",
            "blocks.0.hook_resid_post",
            "blocks.1.hook_resid_post",
            "blocks.2.hook_resid_post",
        ]
        for t in feats.values():
            assert t.shape == (5, 32)


class TestProbeRowsForModel:
    def test_identical_model_twice_gives_identical_accuracies(self, tiny_llama):
        model = tiny_llama(seed=9, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(6)
        exs = _examples(rng, n=24, vocab_size=64, seq_len=7, span=(4, 5))
        ref = rp.build_reference(exs, "task", seed=0)
        # different global torch RNG state around each call: the fit is a
        # convex zero-initialised optimization, so no random seed may enter.
        torch.manual_seed(1)
        rows1 = rp.probe_rows_for_model(model, {"task": ref}, "m", checkpoint_step=0)
        torch.manual_seed(999)
        rows2 = rp.probe_rows_for_model(model, {"task": ref}, "m", checkpoint_step=0)
        assert len(rows1) == len(rows2) > 0
        for r1, r2 in zip(rows1, rows2):
            assert r1["probe_test_acc"] == r2["probe_test_acc"]
            assert r1["probe_train_acc"] == r2["probe_train_acc"]

    def test_n_classes_shared_across_layers_and_models(self, tiny_llama):
        model_a = tiny_llama(
            seed=1, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = tiny_llama(
            seed=2, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(8)
        exs = _examples(rng, n=20, vocab_size=64, seq_len=7, span=(4, 5))
        ref = rp.build_reference(exs, "task", seed=0)
        rows_a = rp.probe_rows_for_model(model_a, {"task": ref}, "a", checkpoint_step=0)
        rows_b = rp.probe_rows_for_model(model_b, {"task": ref}, "b", checkpoint_step=100)
        n_classes_seen = {r["n_classes"] for r in rows_a + rows_b}
        assert n_classes_seen == {ref.n_classes}

    def test_rows_carry_model_and_checkpoint_labels(self, tiny_llama):
        model = tiny_llama(seed=3, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        rng = random.Random(0)
        exs = _examples(rng, n=10, vocab_size=32, seq_len=5, span=(3, 4))
        ref = rp.build_reference(exs, "task", seed=0)
        rows = rp.probe_rows_for_model(model, {"task": ref}, "evt-run1-base", checkpoint_step=-1)
        assert all(r["model"] == "evt-run1-base" for r in rows)
        assert all(r["checkpoint_step"] == -1 for r in rows)
        assert all(r["set"] == "task" for r in rows)
        # 1 layer -> 2 residual points (embed, block 0)
        assert {r["layer"] for r in rows} == {0, 1}

    def test_b_planted_class_signal_recovered_at_and_after_layer_L(self, tiny_llama):
        model = tiny_llama(seed=10, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(5)
        n_classes = 4
        answer_tokens = [10, 20, 30, 40]
        seq_len = 8
        span = (5, 7)
        exs = []
        for i in range(60):
            ids = [rng.randrange(4, 64) for _ in range(seq_len)]
            ids[span[0]] = answer_tokens[i % n_classes]
            exs.append(SftExample(input_ids=ids, label_span=span))
        ref = rp.build_reference(exs, "task", seed=0)
        assert ref.n_classes == n_classes

        target_layer_idx = 2  # blocks.{target_layer_idx - 1}.hook_resid_post
        scale = 60.0
        gen = torch.Generator()
        gen.manual_seed(123)
        class_dirs = torch.randn(ref.n_classes, 32, generator=gen)

        def perturb_hook(_module, _inputs, output):
            add = (scale * class_dirs[ref.y]).unsqueeze(1)
            if isinstance(output, tuple):
                return (output[0] + add,) + output[1:]
            return output + add

        handle = model.model.layers[target_layer_idx - 1].register_forward_hook(perturb_hook)
        try:
            rows = rp.probe_rows_for_model(model, {"task": ref}, "m", checkpoint_step=0)
        finally:
            handle.remove()

        df = pd.DataFrame(rows).set_index("layer").sort_index()
        for layer in range(target_layer_idx, len(df)):
            assert df.loc[layer, "probe_test_acc"] >= 0.95, (
                f"layer {layer} did not recover the signal"
            )
        for layer in range(target_layer_idx):
            # no signal reaches a position that cannot causally see the
            # (not-yet-generated) answer token; stay near the majority floor.
            assert df.loc[layer, "probe_test_acc"] <= ref.majority_test_acc + 0.35, (
                f"layer {layer} < L leaked signal it should not have"
            )


class TestSnapshotStepDiscovery:
    def test_steps_in_dir_sorted_and_skips_non_numeric(self, tmp_path):
        snaps = tmp_path / "snapshots"
        snaps.mkdir()
        for name in ("step_16", "step_1", "step_4", "base"):
            d = snaps / name
            d.mkdir()
            if name != "base":
                (d / "adapter.safetensors").write_bytes(b"")
        (snaps / "step_2.txt").write_text("not a dir")  # file, not a step dir
        assert rp.snapshot_steps_in_dir(snaps, "adapter.safetensors") == [1, 4, 16]

    def test_steps_in_dir_skips_missing_marker(self, tmp_path):
        # a step_* dir without its marker file is a killed-mid-write
        # snapshot, not a usable checkpoint (must be skipped, not raise).
        snaps = tmp_path / "snapshots"
        snaps.mkdir()
        (snaps / "step_1").mkdir()
        (snaps / "step_1" / "adapter.safetensors").write_bytes(b"")
        (snaps / "step_2").mkdir()  # no marker
        assert rp.snapshot_steps_in_dir(snaps, "adapter.safetensors") == [1]

    def test_steps_in_dir_empty_raises(self, tmp_path):
        snaps = tmp_path / "snapshots"
        snaps.mkdir()
        with pytest.raises(FileNotFoundError):
            rp.snapshot_steps_in_dir(snaps, "adapter.safetensors")

    def test_discover_snapshot_dir_prefers_lora_layout(self, tmp_path):
        run = tmp_path / "runs" / "evt-fake"
        step_dir = run / "snapshots" / "step_1"
        step_dir.mkdir(parents=True)
        (step_dir / "adapter.safetensors").write_bytes(b"")
        snap_dir, kind = rp.discover_snapshot_dir("evt-fake", tmp_path)
        assert kind == "lora"
        assert snap_dir == run / "snapshots"

    def test_discover_snapshot_dir_falls_back_to_full_ft_layout(self, tmp_path):
        run = tmp_path / "runs" / "evt-fake-fmt"
        step_dir = run / "sft_snapshots" / "step_1"
        step_dir.mkdir(parents=True)
        (step_dir / "model.safetensors").write_bytes(b"")
        snap_dir, kind = rp.discover_snapshot_dir("evt-fake-fmt", tmp_path)
        assert kind == "full_ft"
        assert snap_dir == run / "sft_snapshots"

    def test_discover_snapshot_dir_falls_back_when_lora_dir_has_no_usable_steps(self, tmp_path):
        # snapshots/ exists but only has a killed-mid-write step (no
        # adapter.safetensors) -> must fall through to sft_snapshots/, not
        # win by mere existence and then raise pointing at the wrong dir.
        run = tmp_path / "runs" / "evt-fake-both"
        (run / "snapshots" / "step_1").mkdir(parents=True)
        step_dir = run / "sft_snapshots" / "step_1"
        step_dir.mkdir(parents=True)
        (step_dir / "model.safetensors").write_bytes(b"")
        snap_dir, kind = rp.discover_snapshot_dir("evt-fake-both", tmp_path)
        assert kind == "full_ft"
        assert snap_dir == run / "sft_snapshots"

    def test_discover_snapshot_dir_raises_when_neither_exists(self, tmp_path):
        (tmp_path / "runs" / "evt-nope").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            rp.discover_snapshot_dir("evt-nope", tmp_path)


class TestProbeMarginOverFloor:
    def test_hand_computed(self):
        assert math.isclose(rp.probe_margin_over_floor(0.9, 0.3), 0.6, rel_tol=1e-12)
        assert rp.probe_margin_over_floor(0.5, 0.5) == 0.0
        # best < floor is a valid (if odd) input: the margin goes negative.
        assert math.isclose(rp.probe_margin_over_floor(0.2, 0.8), -0.6, rel_tol=1e-12)


class TestLayerSummary:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rp.layer_summary(pd.DataFrame(columns=["layer", "probe_test_acc"]))

    def test_missing_layer_zero_raises(self):
        df = pd.DataFrame({"layer": [1, 2], "probe_test_acc": [0.5, 0.9]})
        with pytest.raises(ValueError):
            rp.layer_summary(df)

    def test_hand_computed_floor_best_final_and_margin(self):
        df = pd.DataFrame(
            {
                "layer": [0, 1, 2, 3],
                "probe_test_acc": [0.30, 0.55, 0.95, 0.80],
            }
        )
        summary = rp.layer_summary(df)
        assert summary["layer0_floor"] == 0.30
        assert summary["best_layer"] == 2
        assert summary["best_acc"] == 0.95
        assert summary["final_layer"] == 3
        assert summary["final_acc"] == 0.80
        assert math.isclose(summary["probe_margin_over_floor"], 0.65, rel_tol=1e-12)

    def test_layer_zero_itself_can_be_the_best_layer(self):
        # the discriminator's "already computing at layer 0" failure mode.
        df = pd.DataFrame({"layer": [0, 1, 2], "probe_test_acc": [0.98, 0.40, 0.35]})
        summary = rp.layer_summary(df)
        assert summary["best_layer"] == 0
        assert math.isclose(summary["probe_margin_over_floor"], 0.0, abs_tol=1e-12)


class TestSummaryTableAndPrintSummary:
    def test_summary_table_one_row_per_model_checkpoint_set(self, tiny_llama):
        model = tiny_llama(seed=4, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(12)
        exs = _examples(rng, n=16, vocab_size=64, seq_len=6, span=(3, 4))
        ref = rp.build_reference(exs, "task", seed=0)
        rows = rp.probe_rows_for_model(model, {"task": ref}, "m", checkpoint_step=0)
        rows += rp.probe_rows_for_model(model, {"task": ref}, "m", checkpoint_step=1)
        table = rp.summary_table(rows)
        assert len(table) == 2  # one per checkpoint_step, same (model, set)
        assert set(table["checkpoint_step"]) == {0, 1}

    def test_print_summary_does_not_crash(self, tiny_llama, capsys):
        model = tiny_llama(seed=5, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(13)
        exs = _examples(rng, n=16, vocab_size=64, seq_len=6, span=(3, 4))
        ref = rp.build_reference(exs, "task", seed=0)
        rows = rp.probe_rows_for_model(model, {"task": ref}, "m", checkpoint_step=-1)
        rp.print_summary(rows)  # smoke: must not raise
        out = capsys.readouterr().out
        assert "layer0_floor" in out
        assert "margin_over_floor" in out
