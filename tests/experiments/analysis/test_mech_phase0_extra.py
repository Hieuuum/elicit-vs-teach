"""Coverage audit for the Phase-0 mechanistic-interpretability drivers
(``logit_lens.py`` test 6, ``weight_diff.py`` test 9, ``resid_shift.py`` test
10, and shared ``mech_lib.py``): silent-failure modes NOT already exercised
by ``test_mech_phase0.py``. That file is the baseline (dispatch branches,
off-by-one position pins, planted-direction causal checks, LoRA-vs-full-FT
equivalence); this file adds:

- ``mech_lib``: padded-batch attention-mask correctness (a wrong mask
  dtype/inversion bug would silently corrupt every downstream metric),
  multi-token label spans, the ``dir:`` load path's postconditions
  (eval mode, correct weights), and the two ``load_generic_texts``/
  ``pad_examples`` branches the baseline file never reaches.
- ``logit_lens.py``: ``all_label``'s per-position wiring end to end (not
  just the pure position-math unit test), a non-monotone ``emergence_layer``
  curve (first crossing, not the max), strictly-greater rank semantics under
  ties, heterogeneous multi-token ``n`` accounting, and the untied-``lm_head``
  path.
- ``weight_diff.py``: ``target_modules``' exact module set (not just a
  count), a straddling-rank overlap's exact 0.5 split, an exact planted
  effective-rank, the per-layer/total aggregation arithmetic, ``embed_tokens``
  staying exactly zero on a genuinely LoRA-wrapped (not just identical)
  model B, an undocumented-but-current tolerance for a depth-mismatched
  model B, and ``top_sv``'s cap/order/no-padding behavior.
- ``resid_shift.py``: pad-length insensitivity of the generic-set mean pool,
  the task row's exact read position (``p-1`` vs a plausible ``p-2`` bug),
  an exact 0.5 ``top_pc_evr`` split, the antipodal-cancellation clamp path
  for ``mean_cos_to_mean``, and ``print_summary``'s NaN-not-crash path.
- One ``main()`` argv-plumbing smoke test per script, on a tmp
  ``save_pretrained`` model + parquet + tokenizer -- argparse/IO regressions
  the pure-function tests above never see.

CPU-only, tiny random-init fixtures (conftest's ``tiny_llama``/
``tiny_tokenizer``), no network, no real checkpoints.
"""

from __future__ import annotations

import copy
import math
import random
import sys

import pandas as pd
import pytest
import torch

from geode.arith.spans import SftExample
from geode.train.lora import apply_lora

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
    def test_capture_residuals_padding_does_not_leak_into_real_positions(self, tiny_llama):
        """Right-padding a batch must not change an example's own residuals at
        its real (non-pad) positions vs. running it alone -- the property a
        wrong attention_mask dtype/inversion bug would silently break.
        """
        model = tiny_llama(seed=30, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True)
        torch.manual_seed(0)
        ids_short = torch.randint(4, 64, (5,))
        ids_long = torch.randint(4, 64, (8,))

        ids_alone = ids_short.unsqueeze(0)
        mask_alone = torch.ones_like(ids_alone, dtype=torch.bool)
        acts_alone = mech.capture_residuals(model, ids_alone, mask_alone)

        max_len = 8
        batch_ids = torch.zeros(2, max_len, dtype=torch.long)
        batch_mask = torch.zeros(2, max_len, dtype=torch.bool)
        batch_ids[0, :5] = ids_short
        batch_mask[0, :5] = True
        batch_ids[1, :8] = ids_long
        batch_mask[1, :8] = True
        acts_batched = mech.capture_residuals(model, batch_ids, batch_mask)

        for name in acts_alone:
            real = acts_batched[name][0, :5, :]
            assert torch.allclose(real, acts_alone[name][0], atol=1e-4), (
                f"{name}: padding leaked into non-pad positions"
            )

    def test_load_task_examples_multitoken_answer_label_span(self):
        """The char-span bridge for an answer spanning MULTIPLE tokens, not
        just the one-token-answer case the baseline file covers.

        ``tiny_tokenizer``'s WhitespaceSplit pre-tokenizer strips the space
        between words from both neighbors' offsets, so a genuine multi-WORD
        answer always trips ``token_label_span``'s "offset gap" guard (a
        space char would be left uncovered) -- that's a property of the
        fixture's word-level tokenizer, not of the bridge under test. A
        minimal one-char-per-token stub sidesteps that and gives contiguous
        offsets, matching how a real BPE tokenizer's offsets behave.
        """

        class _CharTokenizer:
            eos_token_id = 3

            def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=True):
                input_ids, offsets = [], []
                for t in texts:
                    input_ids.append([4 + (ord(c) % 60) for c in t])
                    offsets.append([(i, i + 1) for i in range(len(t))])
                return {"input_ids": input_ids, "offset_mapping": offsets}

        texts = ["hello world", "foo barbaz"]
        answers = ["world", "barbaz"]
        starts = [len(t) - len(a) for t, a in zip(texts, answers)]
        ends = [len(t) for t in texts]
        df = pd.DataFrame(
            {"full_text": texts, "answer_char_start": starts, "answer_char_end": ends}
        )
        examples = mech.load_task_examples(df, _CharTokenizer())
        for ex, ans in zip(examples, answers):
            expected_ids = [4 + (ord(c) % 60) for c in ans]
            start, end = ex.label_span
            assert end - start == len(ans) + 1  # answer tokens + appended eos
            assert ex.input_ids[start : end - 1] == expected_ids
            assert ex.input_ids[end - 1] == 3  # eos

    def test_load_any_model_dir_round_trips_and_lands_in_eval_mode(self, tiny_llama, tmp_path):
        model = tiny_llama(seed=31, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        model.train()  # flip out of eval so the load path's postcondition is actually exercised
        saved_weight = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        model_dir = tmp_path / "model"
        model.save_pretrained(model_dir)

        loaded = mech.load_any_model(f"dir:{model_dir}", device="cpu")
        assert not loaded.training
        assert next(loaded.parameters()).device == torch.device("cpu")
        assert torch.equal(loaded.model.layers[0].self_attn.q_proj.weight, saved_weight)

    def test_pad_examples_empty_list_raises(self):
        with pytest.raises(ValueError):
            mech.pad_examples([])

    def test_load_generic_texts_parquet_with_text_column(self, tmp_path):
        p = tmp_path / "generic.parquet"
        pd.DataFrame({"text": ["alpha", "beta", "gamma"]}).to_parquet(p)
        assert mech.load_generic_texts(p) == ["alpha", "beta", "gamma"]
        assert mech.load_generic_texts(p, limit=2) == ["alpha", "beta"]

    def test_load_generic_texts_parquet_missing_text_column_raises(self, tmp_path):
        p = tmp_path / "bad.parquet"
        pd.DataFrame({"other": ["x", "y"]}).to_parquet(p)
        with pytest.raises(ValueError):
            mech.load_generic_texts(p)

    def test_load_any_model_dir_refuses_lora_wrapped_checkpoint(self, tiny_llama, tmp_path):
        """A LoRA-wrapped ``save_pretrained`` dir under ``dir:`` must refuse:
        plain ``from_pretrained`` would silently random-init every projection
        (the known incident behind "LoRA checkpoints load only via
        ``zoo.load_model``"), and every downstream metric would be garbage
        with no crash. The error must point at the ``run:`` spec."""
        from geode.train.lora import apply_lora

        model = tiny_llama(seed=0, n_layers=1, d_model=16, vocab_size=32, tie_word_embeddings=True)
        apply_lora(model, rank=2, alpha=4.0, seed=0)
        wrapped_dir = tmp_path / "wrapped"
        model.save_pretrained(wrapped_dir)
        try:
            mech.load_any_model(f"dir:{wrapped_dir}", device="cpu")
        except ValueError as e:
            assert "run:" in str(e) and "LoRA" in str(e)
        else:
            raise AssertionError("expected ValueError for a LoRA-wrapped dir: checkpoint")


class TestLogitLens:
    def test_all_label_flips_exactly_one_position_first_answer_unaffected(self, tiny_llama):
        """Teacher-forced ``all_label`` positions are wired per label token
        (``p-1`` for the token at ``p``): perturbing the residual at exactly
        ONE such position must move ``all_label``'s top1_acc by exactly
        ``1/n`` and leave ``first_answer`` (a disjoint position) untouched.
        A shared-position bug, or an off-by-one across the span, would move
        both readouts together or move ``all_label`` by the wrong amount.
        """
        n_layers, d_model, vocab_size = 3, 32, 64
        model = tiny_llama(
            seed=60,
            n_layers=n_layers,
            d_model=d_model,
            vocab_size=vocab_size,
            tie_word_embeddings=True,
        )
        rng = random.Random(7)
        seq_len = 10
        label_span = (6, 9)  # 3 label tokens -> all_label positions {5, 6, 7}, first_answer = {5}
        ids = [rng.randrange(4, vocab_size) for _ in range(seq_len)]

        # Position 6 (predicting ids[7]) is independent of ids[7] itself under
        # causal attention -- compute the model's own greedy prediction there
        # first, then bake it in as the label, so baseline is KNOWN correct at
        # exactly that one position (no reliance on a lucky random draw).
        ids_t = torch.tensor([ids])
        mask_t = torch.ones_like(ids_t, dtype=torch.bool)
        acts = mech.capture_residuals(model, ids_t, mask_t)
        norm, head = mech.final_norm_and_head(model)
        final_name = list(acts.keys())[-1]
        with torch.no_grad():
            natural_argmax = int(head(norm(acts[final_name][:, 6, :])).argmax(dim=-1).item())
        ids[7] = natural_argmax
        planted_token = 10 if natural_argmax != 10 else 11

        examples = [SftExample(input_ids=ids, label_span=label_span)]
        baseline_rows = ll.logit_lens_rows(model, examples, "m", "s")
        final_layer = n_layers
        baseline_first = next(
            r
            for r in baseline_rows
            if r["layer"] == final_layer and r["position_kind"] == ll.FIRST_ANSWER
        )
        baseline_all = next(
            r
            for r in baseline_rows
            if r["layer"] == final_layer and r["position_kind"] == ll.ALL_LABEL
        )
        assert baseline_all["n"] == 3

        embed_row = model.get_input_embeddings().weight[planted_token].detach()
        scale = 60.0
        flip_pos = 6

        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            out[:, flip_pos, :] = out[:, flip_pos, :] + scale * embed_row
            return (out,) + output[1:] if isinstance(output, tuple) else out

        handle = model.model.layers[n_layers - 1].register_forward_hook(hook)
        try:
            perturbed_rows = ll.logit_lens_rows(model, examples, "m", "s")
        finally:
            handle.remove()

        perturbed_first = next(
            r
            for r in perturbed_rows
            if r["layer"] == final_layer and r["position_kind"] == ll.FIRST_ANSWER
        )
        perturbed_all = next(
            r
            for r in perturbed_rows
            if r["layer"] == final_layer and r["position_kind"] == ll.ALL_LABEL
        )

        assert perturbed_first == baseline_first  # a different position entirely -> untouched
        assert math.isclose(
            baseline_all["top1_acc"] - perturbed_all["top1_acc"], 1.0 / 3.0, abs_tol=1e-9
        )

    def test_emergence_layer_first_crossing_on_nonmonotone_curve(self):
        rows = [
            {"position_kind": "first_answer", "layer": 0, "top1_acc": 0.1},
            {"position_kind": "first_answer", "layer": 1, "top1_acc": 0.6},  # first crossing
            {"position_kind": "first_answer", "layer": 2, "top1_acc": 0.2},  # dips back below
            {"position_kind": "first_answer", "layer": 3, "top1_acc": 1.0},  # final (the max)
        ]
        assert ll.emergence_layer(rows) == 1.0  # not layer 2, and not layer 3 (the max)

    def test_lens_metrics_ties_do_not_count_toward_rank(self):
        logits = torch.tensor([[3.0, 5.0, 5.0, 5.0]])
        target = torch.tensor([2])  # tied for the max value, but not argmax's first pick
        top1_acc, _, mean_rank = ll.lens_metrics(logits, target)
        assert mean_rank == 0.0  # no logit strictly greater than the (tied) target logit
        assert top1_acc == 0.0  # argmax breaks the tie toward index 1, not index 2

    def test_all_label_n_equals_total_label_token_count(self, tiny_llama):
        model = tiny_llama(seed=22, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        exs = [
            SftExample(input_ids=[4, 5, 6, 7, 8], label_span=(3, 5)),  # 2 label tokens
            SftExample(input_ids=[4, 5, 6, 7, 8, 9, 10, 11], label_span=(4, 8)),  # 4 label tokens
        ]
        rows = ll.logit_lens_rows(model, exs, "m", "s")
        fa_n = {r["n"] for r in rows if r["position_kind"] == ll.FIRST_ANSWER}
        all_n = {r["n"] for r in rows if r["position_kind"] == ll.ALL_LABEL}
        assert fa_n == {2}  # one first-answer pair per example
        assert all_n == {6}  # 2 + 4 label tokens total, not 2 examples

    def test_final_layer_lens_matches_forward_exactly_untied_head(self, tiny_llama):
        model = tiny_llama(
            seed=61, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=False
        )
        assert not torch.equal(model.lm_head.weight, model.get_input_embeddings().weight)
        rng = random.Random(9)
        exs = _examples(rng, n=4, vocab_size=32, seq_len=8, span=(4, 6))
        ids, mask = mech.pad_examples(exs, device="cpu")
        with torch.no_grad():
            model_logits = model(input_ids=ids, attention_mask=mask.long()).logits

        acts = mech.capture_residuals(model, ids, mask)
        norm, head = mech.final_norm_and_head(model)
        final_name = list(acts.keys())[-1]
        with torch.no_grad():
            lens_logits = head(norm(acts[final_name]))
        assert torch.allclose(lens_logits, model_logits, atol=1e-5)


class TestWeightDiff:
    def test_target_modules_exact_set_no_lm_head_no_norm(self, tiny_llama):
        model = tiny_llama(seed=20, n_layers=3, d_model=16, vocab_size=32, tie_word_embeddings=True)
        targets = wd.target_modules(model)
        got = {(layer, name.rsplit(".", 1)[-1]) for name, layer, _ in targets}
        expected = {(i, proj) for i in range(3) for proj in wd.TARGET_PROJECTIONS}
        expected.add((-1, wd.EMBED_NAME))
        assert got == expected
        assert len(targets) == 3 * 7 + 1
        leaves = {name.rsplit(".", 1)[-1] for name, _, _ in targets}
        assert "lm_head" not in leaves
        assert "norm" not in leaves
        assert "input_layernorm" not in leaves
        assert "post_attention_layernorm" not in leaves

    def test_embed_tokens_delta_zero_on_genuine_lora_wrapped_model(self, tiny_llama):
        base = tiny_llama(seed=21, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
        lora_model = copy.deepcopy(base)
        apply_lora(lora_model, rank=4, alpha=8.0, seed=1)
        for module in lora_model.modules():
            if hasattr(module, "B") and hasattr(module, "A"):
                with torch.no_grad():
                    module.B.weight.uniform_(-0.3, 0.3)
        rows = wd.weight_diff_rows(base, lora_model)
        df = pd.DataFrame(rows)
        embed_row = df[(df["level"] == "module") & (df["module"] == "embed_tokens")].iloc[0]
        assert embed_row["rel_fro"] == 0.0
        assert embed_row["fro_dw"] == 0.0
        # sanity: other modules DID move -- this isn't a trivially-identical-models case
        q_row = df[
            (df["level"] == "module") & (df["module"] == "q_proj") & (df["layer"] == 0)
        ].iloc[0]
        assert q_row["fro_dw"] > 0.0

    def test_module_metrics_rank2_straddling_top_r_gives_half_overlap(self):
        # float64 throughout: sidesteps float32 SVD noise so the exact 0.5
        # energy split isn't masked by needing a loose tolerance.
        torch.manual_seed(2)
        w0 = torch.randn(200, 150, dtype=torch.float64)
        u, _, vh = torch.linalg.svd(w0, full_matrices=False)
        c = 2.0
        dw = c * torch.outer(u[:, 3], vh[5, :]) + c * torch.outer(u[:, 100], vh[110, :])
        m = wd.module_metrics(w0, dw)
        assert math.isclose(m["overlap_8"], 0.5, abs_tol=1e-9)
        assert math.isclose(m["overlap_left_8"], 0.5, abs_tol=1e-9)
        assert math.isclose(m["overlap_right_8"], 0.5, abs_tol=1e-9)

    def test_module_metrics_planted_rank3_equal_singular_values_effective_rank_is_3(self):
        torch.manual_seed(3)
        w0 = torch.randn(200, 150, dtype=torch.float64)
        u, _, vh = torch.linalg.svd(w0, full_matrices=False)
        c = 1.5
        dw = c * (
            torch.outer(u[:, 10], vh[10, :])
            + torch.outer(u[:, 20], vh[20, :])
            + torch.outer(u[:, 30], vh[30, :])
        )
        m = wd.module_metrics(w0, dw)
        assert math.isclose(m["effective_rank"], 3.0, abs_tol=1e-6)

    def test_layer_and_total_fro_dw_aggregate_exactly(self, tiny_llama):
        model_a = tiny_llama(
            seed=23, n_layers=3, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        model_b = copy.deepcopy(model_a)
        with torch.no_grad():
            model_b.model.layers[0].self_attn.q_proj.weight.data += 0.1
            model_b.model.layers[0].self_attn.k_proj.weight.data += 0.2
            model_b.model.layers[1].mlp.up_proj.weight.data += 0.05
        rows = wd.weight_diff_rows(model_a, model_b)
        df = pd.DataFrame(rows)
        for layer in sorted(df[df["level"] == "module"]["layer"].unique()):
            module_rows = df[(df["level"] == "module") & (df["layer"] == layer)]
            expected_fro = float((module_rows["fro_dw"] ** 2).sum() ** 0.5)
            layer_row = df[(df["level"] == "layer") & (df["layer"] == layer)].iloc[0]
            assert math.isclose(layer_row["fro_dw"], expected_fro, rel_tol=1e-9)
        layer_rows = df[df["level"] == "layer"]
        expected_total = float((layer_rows["fro_dw"] ** 2).sum() ** 0.5)
        total_row = df[df["level"] == "total"].iloc[0]
        assert math.isclose(total_row["fro_dw"], expected_total, rel_tol=1e-9)

    def test_model_b_missing_layer_present_in_a_is_silently_partial_not_raised(self, tiny_llama):
        """``target_modules()`` enumerates from model_b only; a layer present
        in model_a but absent from model_b (mismatched depth) is silently
        excluded from the diff -- no row, no error. Neither the module
        docstring nor any code comment claims a symmetric existence check, so
        this pins the current (undocumented) tolerant behavior rather than
        flagging it as a contract violation.
        """
        model_a = tiny_llama(
            seed=24, n_layers=3, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        model_b = tiny_llama(
            seed=25, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True
        )
        rows = wd.weight_diff_rows(model_a, model_b)  # must not raise
        df = pd.DataFrame(rows)
        module_rows = df[df["level"] == "module"]
        assert len(module_rows) == 2 * 7 + 1  # only model_b's 2 layers; A's 3rd is dropped
        assert set(module_rows[module_rows["layer"] >= 0]["layer"].unique()) == {0, 1}

    def test_top_sv_length_capped_at_16_and_sorted_descending(self):
        torch.manual_seed(4)
        w0 = torch.randn(200, 150)
        dw = torch.randn(200, 150) * 0.1
        m = wd.module_metrics(w0, dw)
        svs = m["top_sv"]
        assert len(svs) == 16
        assert all(svs[i] >= svs[i + 1] for i in range(len(svs) - 1))

    def test_top_sv_length_not_padded_when_matrix_smaller_than_16(self):
        torch.manual_seed(5)
        w0 = torch.randn(10, 8)
        dw = torch.randn(10, 8) * 0.1
        m = wd.module_metrics(w0, dw)
        svs = m["top_sv"]
        assert len(svs) == 8  # min(10, 8) < 16, not zero-padded up to 16
        assert all(svs[i] >= svs[i + 1] for i in range(len(svs) - 1))


class TestResidShift:
    def test_generic_pooling_is_pad_length_insensitive(self, tiny_llama):
        model_a = tiny_llama(
            seed=40, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        model_b = tiny_llama(
            seed=41, n_layers=3, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(5)
        exs = _examples(rng, n=3, vocab_size=64, seq_len=9, span=(5, 6))
        base_ids = [[10, 11, 12], [20, 21], [30, 31, 32, 33]]

        class _FixedTokenizer:
            def __init__(self, pad_to):
                self.pad_to = pad_to

            def __call__(self, texts, add_special_tokens=False, padding=True, return_tensors="pt"):
                maxlen = max(self.pad_to, max(len(x) for x in base_ids))
                input_ids = torch.zeros(len(base_ids), maxlen, dtype=torch.long)
                mask = torch.zeros(len(base_ids), maxlen, dtype=torch.long)
                for i, ids in enumerate(base_ids):
                    input_ids[i, : len(ids)] = torch.tensor(ids)
                    mask[i, : len(ids)] = 1
                return {"input_ids": input_ids, "attention_mask": mask}

        texts = ["a", "b", "c"]
        rows_tight = rs.resid_shift_rows(model_a, model_b, exs, texts, _FixedTokenizer(pad_to=4))
        rows_padded = rs.resid_shift_rows(model_a, model_b, exs, texts, _FixedTokenizer(pad_to=9))

        tight = pd.DataFrame(rows_tight).set_index(["set", "layer"]).xs("generic")
        padded = pd.DataFrame(rows_padded).set_index(["set", "layer"]).xs("generic")
        for col in ("rel_shift", "mean_cos_to_mean", "top_pc_evr"):
            assert (tight[col] - padded[col]).abs().max() < 1e-6, (
                f"{col}: extra padding changed the generic-set pooled row"
            )

    def test_task_row_reads_position_start_minus_1_not_start_minus_2(self, tiny_llama):
        model_a = tiny_llama(
            seed=42, n_layers=4, d_model=32, vocab_size=64, tie_word_embeddings=True
        )
        rng = random.Random(6)
        exs = _examples(rng, n=4, vocab_size=64, seq_len=10, span=(6, 7))  # start=6, task pos = 5
        generic = ["t1 t2", "t3 t4 t5"]

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

        target_layer_idx = 2
        v = torch.randn(32) * 5.0

        def make_hook(seq_idx):
            def hook(_module, _inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                out = out.clone()
                out[:, seq_idx, :] = out[:, seq_idx, :] + v
                return (out,) + output[1:] if isinstance(output, tuple) else out

            return hook

        shifts = {}
        for seq_idx in (5, 4):  # 5 = start-1 (correct), 4 = start-2 (a plausible off-by-one)
            model_b = copy.deepcopy(model_a)
            handle = model_b.model.layers[target_layer_idx - 1].register_forward_hook(
                make_hook(seq_idx)
            )
            try:
                rows = rs.resid_shift_rows(model_a, model_b, exs, generic, _EchoTokenizer())
            finally:
                handle.remove()
            df = pd.DataFrame(rows).set_index(["set", "layer"])
            shifts[seq_idx] = df.loc[("task", target_layer_idx), "rel_shift"]

        assert shifts[5] > 0.0  # perturbing the actual read position moves the row
        assert shifts[4] == 0.0  # perturbing p-2 (never read) leaves it exactly untouched

    def test_shift_consistency_two_orthogonal_equal_split_gives_half_evr(self):
        v1 = torch.zeros(16)
        v1[0] = 1.0
        v2 = torch.zeros(16)
        v2[1] = 1.0
        d = torch.cat([v1.unsqueeze(0).repeat(5, 1), v2.unsqueeze(0).repeat(5, 1)], dim=0)
        mean_cos, evr = rs.shift_consistency(d)
        assert math.isclose(evr, 0.5, abs_tol=1e-9)
        assert math.isclose(mean_cos, 1.0 / math.sqrt(2), abs_tol=1e-9)

    def test_shift_consistency_antipodal_pair_cosine_zero_not_nan(self):
        torch.manual_seed(0)
        v = torch.randn(10) * 3.0
        d = torch.stack([v, -v])
        mean_cos, evr = rs.shift_consistency(d)
        assert mean_cos == 0.0  # mean vector is exactly zero; the clamp keeps this 0, not NaN
        assert math.isclose(evr, 1.0, abs_tol=1e-9)  # still one shared (uncentered) subspace

    def test_print_summary_zero_generic_shift_gives_nan_ratio_not_crash(self, capsys):
        rows = [
            {
                "layer": 0,
                "set": "task",
                "rel_shift": 0.5,
                "mean_cos_to_mean": 0.0,
                "top_pc_evr": 0.0,
                "n": 3,
            },
            {
                "layer": 0,
                "set": "generic",
                "rel_shift": 0.0,
                "mean_cos_to_mean": 0.0,
                "top_pc_evr": 0.0,
                "n": 3,
            },
        ]
        rs.print_summary(rows)  # must not raise
        out = capsys.readouterr().out.lower()
        assert "nan" in out


def _save_tiny_model(tiny_llama, tmp_path, seed, name):
    model = tiny_llama(seed=seed, n_layers=2, d_model=16, vocab_size=32, tie_word_embeddings=True)
    out = tmp_path / name
    model.save_pretrained(out)
    return out


def _save_tokenizer(tiny_tokenizer, tmp_path):
    tok = tiny_tokenizer(vocab_size=32)
    out = tmp_path / "tokenizer"
    tok.save_pretrained(out)
    return out


def _task_parquet(tmp_path):
    texts = [f"t1 t2 t{i}" for i in range(3, 9)]
    starts = [len(t) - len(t.split()[-1]) for t in texts]
    ends = [len(t) for t in texts]
    df = pd.DataFrame({"full_text": texts, "answer_char_start": starts, "answer_char_end": ends})
    p = tmp_path / "task.parquet"
    df.to_parquet(p)
    return p


class TestMainSmoke:
    """One argv-plumbing smoke test per script: argparse/IO regressions the
    pure-function tests above never exercise (real save_pretrained round
    trip through ``dir:`` model specs and an on-disk tokenizer/parquet)."""

    def test_logit_lens_main_smoke(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        model_dir = _save_tiny_model(tiny_llama, tmp_path, 50, "model")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        out = tmp_path / "out.csv"
        argv = [
            "logit_lens.py",
            "--model",
            f"dir:{model_dir}",
            "--prompt-parquet",
            str(parquet),
            "--tokenizer",
            str(tok_dir),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        ll.main()
        assert out.is_file()
        df = pd.read_csv(out)
        for col in (
            "model",
            "set",
            "layer",
            "position_kind",
            "top1_acc",
            "mean_logprob_nats",
            "mean_rank",
            "n",
        ):
            assert col in df.columns

    def test_weight_diff_main_smoke(self, tiny_llama, tmp_path, monkeypatch):
        model_a_dir = _save_tiny_model(tiny_llama, tmp_path, 51, "model_a")
        model_b_dir = _save_tiny_model(tiny_llama, tmp_path, 52, "model_b")
        out = tmp_path / "out.parquet"
        argv = [
            "weight_diff.py",
            "--model-a",
            f"dir:{model_a_dir}",
            "--model-b",
            f"dir:{model_b_dir}",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        wd.main()
        assert out.is_file()
        df = pd.read_parquet(out)
        for col in ("level", "layer", "module", "rel_fro", "fro_dw", "fro_w0"):
            assert col in df.columns

    def test_resid_shift_main_smoke(self, tiny_llama, tiny_tokenizer, tmp_path, monkeypatch):
        model_a_dir = _save_tiny_model(tiny_llama, tmp_path, 53, "model_a")
        model_b_dir = _save_tiny_model(tiny_llama, tmp_path, 54, "model_b")
        tok_dir = _save_tokenizer(tiny_tokenizer, tmp_path)
        parquet = _task_parquet(tmp_path)
        generic = tmp_path / "generic.txt"
        generic.write_text("t1 t2 t3\nt4 t5\nt6 t7 t8\n")
        out = tmp_path / "out.csv"
        argv = [
            "resid_shift.py",
            "--model-a",
            f"dir:{model_a_dir}",
            "--model-b",
            f"dir:{model_b_dir}",
            "--task-parquet",
            str(parquet),
            "--generic-local",
            str(generic),
            "--tokenizer",
            str(tok_dir),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rs.main()
        assert out.is_file()
        df = pd.read_csv(out)
        for col in ("layer", "set", "rel_shift", "mean_cos_to_mean", "top_pc_evr", "n"):
            assert col in df.columns
