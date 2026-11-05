"""experiments/em/persona_ablate.py: projection, hooks and the direction estimate (CPU, tiny model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))
_spec = importlib.util.spec_from_file_location(
    "persona_ablate", _ROOT / "experiments" / "em" / "persona_ablate.py"
)
A = importlib.util.module_from_spec(_spec)
sys.modules["persona_ablate"] = A
_spec.loader.exec_module(A)


def test_project_out_removes_exactly_the_direction():
    h = torch.randn(3, 5, 8)
    d = torch.randn(8)
    d = d / d.norm()
    p = A.project_out(h, d)
    assert torch.allclose(p @ d, torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(A.project_out(p, d), p, atol=1e-6)  # idempotent


def test_random_dirs_are_unit_and_seeded():
    like = {0: torch.zeros(8), 3: torch.zeros(8)}
    a, b = A.random_dirs(like, 1), A.random_dirs(like, 1)
    assert set(a) == {0, 3} and all(torch.isclose(v.norm(), torch.tensor(1.0)) for v in a.values())
    assert torch.equal(a[0], b[0])


def test_hooks_zero_the_direction_in_what_the_next_layer_receives(tiny_llama):
    model = tiny_llama(seed=0)
    d = torch.randn(model.config.hidden_size)
    d = d / d.norm()
    seen = []

    def capture(_mod, args, kwargs):
        seen.append((args[0] if args else kwargs["hidden_states"]).detach())

    probe = A.decoder_layers(model)[1].register_forward_pre_hook(capture, with_kwargs=True)
    handles = A.install_ablation(model, {0: d})
    model(torch.tensor([[1, 5, 6, 7]]))
    assert torch.allclose(seen[-1] @ d, torch.zeros(1, 4), atol=1e-4)
    for h in handles:
        h.remove()
    model(torch.tensor([[1, 5, 6, 7]]))
    assert not torch.allclose(seen[-1] @ d, torch.zeros(1, 4), atol=1e-4)
    probe.remove()


def test_mean_answer_states_has_one_vector_per_inner_layer(tiny_llama, tiny_tokenizer):
    model = tiny_llama(seed=0, n_layers=3)
    rows = [
        {"question": "t10 t11", "answer": "t12 t13"},
        {"question": "t14", "answer": "t15 t16 t17"},
    ]
    st = A.mean_answer_states(model, tiny_tokenizer(), rows, device="cpu", batch_size=2, max_len=32)
    assert sorted(st) == [0, 1] and st[0].shape == (model.config.hidden_size,)
    assert all(torch.isfinite(v).all() for v in st.values())


@pytest.mark.parametrize("which", ["none", "persona"])
def test_report_runs_without_results(tmp_path, monkeypatch, capsys, which):
    monkeypatch.setenv("GEODE_STORE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["persona_ablate.py", "report"])
    assert A.main() == 0
    assert "not run" in capsys.readouterr().out
