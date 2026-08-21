"""Phase-0 mechanistic-interpretability drivers: ``logit_lens.py`` (test 6),
``weight_diff.py`` (test 9), ``resid_shift.py`` (test 10), and their shared
``mech_lib.py``.

Silent failure modes guarded, one per driver:

- ``logit_lens.py``: an off-by-one in which residual position feeds the
  lens would silently score the WRONG position's readiness, and a wrong
  final-layer projection (e.g. double-applying ``norm``) would silently
  disagree with the model's own logits while still producing a plausible
  number. Both are pinned exactly (test a), plus a planted-direction causal
  check (test b) and the char-span bridge (test c).
- ``weight_diff.py``: the LoRA/full-FT dispatch picking the wrong branch, or
  the overlap projections being computed in the wrong subspace, would
  produce a plausible-looking but wrong ΔW-geometry number with no crash —
  exactly CLAUDE.md's "silent failure that would corrupt results" case. Pure
  ``module_metrics`` is checked against hand-planted rank-1 updates whose
  answer is known exactly.
- ``resid_shift.py``: ``top_pc_evr`` is deliberately UNCENTERED (module
  docstring) — a regression back to centered would make the constant-shift
  property silently read 0 instead of 1, so that case is checked explicitly
  against a real forward-hook-perturbed model, not just the pure function.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``/
``tiny_tokenizer``), no network, no real checkpoints.
"""

from __future__ import annotations

import copy
import math
import random

import pandas as pd
import torch

from geode.arith.spans import SftExample
from geode.train.lora import apply_lora, merge_lora

from tests._scriptloader import load

ll = load("logit_lens")
wd = load("weight_diff")
rs = load("resid_shift")
mech = load("mech_lib")


def _examples(rng: random.Random, n: int, vocab_size: int, seq_len: int, span: tuple[int, int]):
    out = []
    for _ in range(n):
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]
        out.append(SftExample(input_ids=ids, label_span=span))
    return out


class TestMechLib:
    def test_load_any_model_rejects_unknown_prefix(self):
        try:
            mech.load_any_model("bogus:foo", device="cpu")
        except ValueError as e:
            assert "run:" in str(e) and "dir:" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_pad_examples_right_pads_and_masks(self):
        exs = [
            SftExample(input_ids=[1, 2, 3], label_span=(2, 3)),
            SftExample(input_ids=[4, 5], label_span=(1, 2)),
        ]
        ids, mask = mech.pad_examples(exs, device="cpu")
        assert ids.shape == (2, 3)
        assert torch.equal(ids[0], torch.tensor([1, 2, 3]))
        assert torch.equal(ids[1], torch.tensor([4, 5, 0]))
        assert torch.equal(mask, torch.tensor([[True, True, True], [True, True, False]]))

    def test_capture_residuals_names_and_shapes(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=3, d_model=32)
        ids = torch.randint(4, 128, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        acts = mech.capture_residuals(model, ids, mask)
        assert list(acts.keys()) == [
            "hook_embed",
            "blocks.0.hook_resid_post",
            "blocks.1.hook_resid_post",
            "blocks.2.hook_resid_post",
        ]
        for t in acts.values():
            assert t.shape == (2, 5, 32)

    def test_final_norm_and_head_resolve(self, tiny_llama):
        model = tiny_llama(seed=0, n_layers=2, d_model=16)
        norm, head = mech.final_norm_and_head(model)
        assert norm is model.model.norm
        assert head is model.lm_head

    def test_load_task_examples_limit_and_span(self, tiny_tokenizer):
        tok = tiny_tokenizer(vocab_size=32)
        texts = [f"t1 t2 t{i}" for i in range(3, 8)]
        starts = [len(t) - len(t.split()[-1]) for t in texts]
        ends = [len(t) for t in texts]
        df = pd.DataFrame(
            {"full_text": texts, "answer_char_start": starts, "answer_char_end": ends}
        )
        examples = mech.load_task_examples(df, tok, limit=2)
        assert len(examples) == 2

    def test_load_generic_texts_txt_and_limit(self, tmp_path):
        p = tmp_path / "generic.txt"
        p.write_text("one\ntwo\n\nthree\n")
        texts = mech.load_generic_texts(p)
        assert texts == ["one", "two", "three"]
        assert mech.load_generic_texts(p, limit=2) == ["one", "two"]

    def test_load_generic_texts_empty_raises(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("\n\n")
        try:
            mech.load_generic_texts(p)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_write_table_csv_and_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        csv_path = tmp_path / "out.csv"
        pq_path = tmp_path / "out.parquet"
        mech.write_table(df, csv_path)
        mech.write_table(df, pq_path)
        assert csv_path.is_file()
        assert pq_path.is_file()
        pd.testing.assert_frame_equal(pd.read_parquet(pq_path), df)


class TestLogitLens:
    def test_lens_metrics_hand_computed(self):
        logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 0.5]])
        target = torch.tensor([1, 0])
        top1_acc, mean_logprob, mean_rank = ll.lens_metrics(logits, target)
        assert top1_acc == 1.0  # row0 argmax=1 (correct), row1 argmax=0 (correct)
        assert mean_rank == 0.0
        expected_logprob = (
            torch.log_softmax(logits.double(), dim=-1).gather(1, target[:, None]).mean()
        )
        assert math.isclose(mean_logprob, float(expected_logprob), rel_tol=1e-9)

    def test_lens_metrics_wrong_prediction_has_positive_rank(self):
        logits = torch.tensor([[5.0, 1.0, 2.0]])
        target = torch.tensor([1])  # not the argmax (0 is)
        top1_acc, _, mean_rank = ll.lens_metrics(logits, target)
        assert top1_acc == 0.0
        assert mean_rank == 2.0  # two tokens (ids 0 and 2) beat the target

    def test_position_target_pairs_first_answer_vs_all_label(self):
        exs = [
            SftExample(input_ids=[10, 11, 12, 13, 14], label_span=(2, 4)),
            SftExample(input_ids=[20, 21, 22, 23, 24], label_span=(3, 5)),
        ]
        ex_idx, pos, target = ll.position_target_pairs(exs, ll.FIRST_ANSWER)
        assert ex_idx == [0, 1]
        assert pos == [1, 2]
        assert target == [12, 23]

        ex_idx, pos, target = ll.position_target_pairs(exs, ll.ALL_LABEL)
        assert ex_idx == [0, 0, 1, 1]
        assert pos == [1, 2, 2, 3]
        assert target == [12, 13, 23, 24]

    def test_a_final_layer_lens_matches_forward_exactly(self, tiny_llama):
        model = tiny_llama(seed=1, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(0)
        exs = _examples(rng, n=5, vocab_size=64, seq_len=9, span=(5, 8))
        ids, mask = mech.pad_examples(exs, device="cpu")
        with torch.no_grad():
            model_logits = model(input_ids=ids, attention_mask=mask.long()).logits

        acts = mech.capture_residuals(model, ids, mask)
        norm, head = mech.final_norm_and_head(model)
        final_name = list(acts.keys())[-1]
        with torch.no_grad():
            lens_logits = head(norm(acts[final_name]))
        assert torch.allclose(lens_logits, model_logits, atol=1e-5)
        assert torch.equal(lens_logits.argmax(dim=-1), model_logits.argmax(dim=-1))

    def test_b_planted_residual_flips_lens_top1_at_and_after_layer_L(self, tiny_llama):
        model = tiny_llama(seed=2, n_layers=4, d_model=64, vocab_size=128, tie_word_embeddings=True)
        rng = random.Random(1)
        exs = _examples(rng, n=6, vocab_size=128, seq_len=10, span=(6, 9))
        ids, mask = mech.pad_examples(exs, device="cpu")
        pos = torch.tensor([e.label_span[0] - 1 for e in exs])
        ex_idx = torch.arange(len(exs))
        norm, head = mech.final_norm_and_head(model)

        def layer_predictions(m):
            acts = mech.capture_residuals(m, ids, mask)
            names = list(acts.keys())
            preds = {}
            with torch.no_grad():
                for layer, name in enumerate(names):
                    h = acts[name][ex_idx, pos, :]
                    preds[layer] = head(norm(h)).argmax(dim=-1)
            return preds

        baseline = layer_predictions(model)

        target_layer_idx = 2  # blocks.{target_layer_idx - 1}.hook_resid_post
        planted_token = 7
        embed_row = model.get_input_embeddings().weight[planted_token].detach()
        scale = 50.0

        def perturb_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + scale * embed_row,) + output[1:]
            return output + scale * embed_row

        handle = model.model.layers[target_layer_idx - 1].register_forward_hook(perturb_hook)
        try:
            perturbed = layer_predictions(model)
        finally:
            handle.remove()

        for layer in range(target_layer_idx):
            assert torch.equal(perturbed[layer], baseline[layer]), (
                f"layer {layer} < L changed under a perturbation applied at L"
            )
        for layer in range(target_layer_idx, len(baseline)):
            assert torch.all(perturbed[layer] == planted_token), (
                f"layer {layer} >= L did not flip to the planted token"
            )

    def test_c_label_positions_agree_with_token_label_span(self, tiny_tokenizer):
        from geode.arith.spans import token_label_span

        tok = tiny_tokenizer(vocab_size=64)
        texts = ["t1 t2 t3", "t5 t6 t7 t8", "t9 t10"]
        answers = ["t3", "t8", "t10"]
        starts = [len(t) - len(a) for t, a in zip(texts, answers)]
        ends = [len(t) for t in texts]
        df = pd.DataFrame(
            {"full_text": texts, "answer_char_start": starts, "answer_char_end": ends}
        )
        examples = mech.load_task_examples(df, tok)
        ex_idx, pos, _target = ll.position_target_pairs(examples, ll.FIRST_ANSWER)

        enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        for i in range(len(texts)):
            expected_start, expected_end = token_label_span(
                enc["offset_mapping"][i], (starts[i], ends[i]), texts[i]
            )
            # append_eos=True extends the end by exactly one token past the bridge's answer.
            assert examples[i].label_span == (expected_start, expected_end + 1)
            # the position logit_lens.py actually reads for this example is p-1
            assert pos[ex_idx.index(i)] == expected_start - 1

    def test_d_limit_respected(self, tiny_llama, tiny_tokenizer):
        model = tiny_llama(seed=0, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        tok = tiny_tokenizer(vocab_size=32)
        texts = [f"t1 t2 t{i}" for i in range(3, 10)]
        starts = [len(t) - len(t.split()[-1]) for t in texts]
        ends = [len(t) for t in texts]
        df = pd.DataFrame(
            {"full_text": texts, "answer_char_start": starts, "answer_char_end": ends}
        )
        examples = mech.load_task_examples(df, tok, limit=3)
        assert len(examples) == 3
        rows = ll.logit_lens_rows(model, examples, "m", "set")
        fa_rows = [r for r in rows if r["position_kind"] == ll.FIRST_ANSWER]
        assert all(r["n"] == 3 for r in fa_rows)

    def test_emergence_layer_nan_when_final_zero(self):
        rows = [
            {"position_kind": "first_answer", "layer": 0, "top1_acc": 0.0},
            {"position_kind": "first_answer", "layer": 1, "top1_acc": 0.0},
        ]
        assert math.isnan(ll.emergence_layer(rows))

    def test_emergence_layer_smallest_crossing(self):
        rows = [
            {"position_kind": "first_answer", "layer": 0, "top1_acc": 0.1},
            {"position_kind": "first_answer", "layer": 1, "top1_acc": 0.4},
            {"position_kind": "first_answer", "layer": 2, "top1_acc": 0.9},
            {"position_kind": "first_answer", "layer": 3, "top1_acc": 1.0},
        ]
        # final=1.0, 0.5*final=0.5 -> smallest layer with acc>=0.5 is layer 2 (0.9)
        assert ll.emergence_layer(rows) == 2.0


class TestWeightDiff:
    def test_module_layer_regex(self):
        assert wd._module_layer("model.layers.3.self_attn.q_proj") == 3
        assert wd._module_layer("model.embed_tokens") == -1
        assert wd._module_layer("model.layers.0.mlp.down_proj") == 0

    def test_module_metrics_zero_delta_does_not_crash(self):
        w0 = torch.randn(20, 16)
        dw = torch.zeros(20, 16)
        m = wd.module_metrics(w0, dw)
        assert m["rel_fro"] == 0.0
        assert math.isnan(m["effective_rank"])
        assert m["top_sv"] == []
        for r in wd.RANKS:
            assert math.isnan(m[f"overlap_{r}"])

    def test_planted_rank1_inside_top8_subspace(self):
        torch.manual_seed(0)
        w0 = torch.randn(200, 150)
        u, _, vh = torch.linalg.svd(w0, full_matrices=False)
        dw = 2.0 * torch.outer(u[:, 3], vh[5, :])
        m = wd.module_metrics(w0, dw)
        assert m["effective_rank"] <= 1.05
        assert m["overlap_8"] > 0.99
        assert m["sv1_frac"] > 0.99

    def test_planted_rank1_orthogonal_to_top128_subspace(self):
        torch.manual_seed(1)
        w0 = torch.randn(300, 200)
        u, _, vh = torch.linalg.svd(w0, full_matrices=False)
        dw = 3.0 * torch.outer(u[:, 150], vh[170, :])
        m = wd.module_metrics(w0, dw)
        assert m["overlap_128"] < 1e-6
        assert m["overlap_left_128"] < 1e-6
        assert m["overlap_right_128"] < 1e-6

    def test_overlap_r_exceeding_min_dim_is_nan_not_one(self):
        w0 = torch.randn(40, 40)
        dw = torch.randn(40, 40)
        m = wd.module_metrics(w0, dw)
        assert math.isnan(m["overlap_128"])  # 128 > min(40, 40): must not silently read 1.0

    def test_identical_models_zero_everywhere(self, tiny_llama):
        model_a = tiny_llama(
            seed=3, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        rows = wd.weight_diff_rows(model_a, model_b)
        df = pd.DataFrame(rows)
        assert (df["rel_fro"].fillna(0.0) == 0.0).all()
        assert (df["fro_dw"] == 0.0).all()
        module_rows = df[df["level"] == "module"]
        assert len(module_rows) == 2 * 7 + 1  # 7 projections x 2 layers + embed_tokens

    def test_module_delta_w_dispatches_lora_vs_full_ft(self, tiny_llama):
        base = tiny_llama(seed=4, n_layers=1, d_model=32, vocab_size=64, tie_word_embeddings=True)
        wrapped = copy.deepcopy(base)
        apply_lora(wrapped, rank=4, alpha=8.0, seed=0)
        q = wrapped.model.layers[0].self_attn.q_proj
        with torch.no_grad():
            q.B.weight.uniform_(-0.1, 0.1)
        base_q = base.model.layers[0].self_attn.q_proj
        dw_lora = wd.module_delta_w(base_q, q)
        expected = wd.lora_delta_w(q.B.weight, q.A.weight, q.alpha, q.rank)
        assert torch.allclose(dw_lora, expected)

        full_b = copy.deepcopy(base)
        full_b.model.layers[0].self_attn.q_proj.weight.data += 1.0
        dw_full = wd.module_delta_w(base_q, full_b.model.layers[0].self_attn.q_proj)
        assert torch.allclose(dw_full, torch.ones_like(dw_full))

    def test_d_lora_path_equals_full_ft_path_on_merged(self, tiny_llama):
        base = tiny_llama(seed=5, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True)
        lora_model = copy.deepcopy(base)
        apply_lora(lora_model, rank=4, alpha=8.0, seed=42)
        for module in lora_model.modules():
            if hasattr(module, "B") and hasattr(module, "A"):
                with torch.no_grad():
                    module.B.weight.uniform_(-0.2, 0.2)

        merged_model = copy.deepcopy(lora_model)
        merge_lora(merged_model)

        rows_lora = pd.DataFrame(wd.weight_diff_rows(base, lora_model))
        rows_merged = pd.DataFrame(wd.weight_diff_rows(base, merged_model))

        lora_mod = rows_lora[rows_lora["level"] == "module"].set_index(["layer", "module"])
        merged_mod = rows_merged[rows_merged["level"] == "module"].set_index(["layer", "module"])
        for col in ("rel_fro", "effective_rank", "sv1_frac", "overlap_8", "overlap_32"):
            pd.testing.assert_series_equal(
                lora_mod[col].sort_index(),
                merged_mod[col].sort_index(),
                atol=1e-4,
                check_names=False,
            )

    def test_summary_does_not_crash_and_reports_finite_total(self, tiny_llama):
        model_a = tiny_llama(
            seed=6, n_layers=2, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.q_proj.weight.data += 0.01
        rows = wd.weight_diff_rows(model_a, model_b)
        wd.print_summary(rows)  # smoke: must not raise
        total = pd.DataFrame(rows).query("level == 'total'").iloc[0]
        assert math.isfinite(total["rel_fro"])


class TestResidShift:
    def test_rel_shift_and_deltas_hand_computed(self):
        h0 = torch.tensor([[3.0, 4.0], [1.0, 0.0]])  # norms 5, 1
        ht = torch.tensor([[3.0, 4.0], [1.0, 3.0]])  # deltas [0,0], [0,3]
        rs_val, d = rs.rel_shift_and_deltas(h0, ht)
        assert math.isclose(rs_val, (0.0 + 3.0) / 2, rel_tol=1e-9)
        assert torch.allclose(d, torch.tensor([[0.0, 0.0], [0.0, 3.0]], dtype=torch.float64))

    def test_shift_consistency_all_zero_is_zero_not_nan(self):
        d = torch.zeros(5, 8)
        mean_cos, evr = rs.shift_consistency(d)
        assert mean_cos == 0.0
        assert evr == 0.0

    def test_shift_consistency_constant_direction_is_one(self):
        v = torch.randn(12)
        d = v.unsqueeze(0).repeat(7, 1)
        mean_cos, evr = rs.shift_consistency(d)
        assert math.isclose(mean_cos, 1.0, abs_tol=1e-6)
        assert math.isclose(evr, 1.0, abs_tol=1e-6)

    def test_shift_consistency_random_is_low(self):
        torch.manual_seed(0)
        d = torch.randn(200, 32)
        mean_cos, evr = rs.shift_consistency(d)
        assert abs(mean_cos) < 0.25
        assert evr < 0.2

    def test_identical_models_zero_shift_all_layers(self, tiny_llama):
        model = tiny_llama(seed=7, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        rng = random.Random(3)
        exs = _examples(rng, n=5, vocab_size=64, seq_len=9, span=(5, 8))
        generic = ["t1 t2 t3", "t4 t5", "t6 t7 t8 t9"]

        class _EchoTokenizer:
            def __call__(self, texts, add_special_tokens=False, padding=True, return_tensors="pt"):
                ids = [[rng.randrange(4, 64) for _ in range(6)] for _ in texts]
                maxlen = max(len(x) for x in ids)
                input_ids = torch.zeros(len(ids), maxlen, dtype=torch.long)
                mask = torch.zeros(len(ids), maxlen, dtype=torch.long)
                for i, x in enumerate(ids):
                    input_ids[i, : len(x)] = torch.tensor(x)
                    mask[i, : len(x)] = 1
                return {"input_ids": input_ids, "attention_mask": mask}

        rows = rs.resid_shift_rows(model, model, exs, generic, _EchoTokenizer())
        df = pd.DataFrame(rows)
        assert (df["rel_shift"] == 0.0).all()
        assert (df["mean_cos_to_mean"] == 0.0).all()
        assert (df["top_pc_evr"] == 0.0).all()

    def test_constant_shift_at_layer_L_end_to_end(self, tiny_llama):
        model_a = tiny_llama(
            seed=8, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        target_layer_idx = 2
        v = torch.randn(32) * 3.0

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + v,) + output[1:]
            return output + v

        handle = model_b.model.layers[target_layer_idx - 1].register_forward_hook(hook)

        rng = random.Random(4)
        exs = _examples(rng, n=6, vocab_size=64, seq_len=9, span=(5, 8))
        generic = ["t1 t2 t3", "t4 t5 t6", "t7 t8"]

        class _EchoTokenizer:
            def __call__(self, texts, add_special_tokens=False, padding=True, return_tensors="pt"):
                ids = [[rng.randrange(4, 64) for _ in range(6)] for _ in texts]
                maxlen = max(len(x) for x in ids)
                input_ids = torch.zeros(len(ids), maxlen, dtype=torch.long)
                mask = torch.zeros(len(ids), maxlen, dtype=torch.long)
                for i, x in enumerate(ids):
                    input_ids[i, : len(x)] = torch.tensor(x)
                    mask[i, : len(x)] = 1
                return {"input_ids": input_ids, "attention_mask": mask}

        try:
            rows = rs.resid_shift_rows(model_a, model_b, exs, generic, _EchoTokenizer())
        finally:
            handle.remove()

        df = pd.DataFrame(rows).set_index(["set", "layer"])
        for s in ("task", "generic"):
            for layer in range(target_layer_idx):
                assert df.loc[(s, layer), "rel_shift"] == 0.0
            row_l = df.loc[(s, target_layer_idx)]
            assert row_l["rel_shift"] > 0.0
            assert math.isclose(row_l["mean_cos_to_mean"], 1.0, abs_tol=1e-4)
            assert math.isclose(row_l["top_pc_evr"], 1.0, abs_tol=1e-4)
            for layer in range(target_layer_idx + 1, 5):
                assert df.loc[(s, layer), "rel_shift"] > 0.0
