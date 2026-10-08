"""Functional reuse test: does a parent's circuit do the child's work? (exploratory)

For a child on the word task, keep ONLY a given top-k edge set (every other edge patched
with corrupted-input activations, as in every circuit eval here) and report the child's
faithfulness f on the val pairs. Edge sets: the child's own circuit, each parent score
set given (e.g. elicit_parent's symbol-task circuit and its word-task circuit), and
`--n-random` random k-sets (band p5/p50/p95). Same code path as run.py's parent_in_child.

  python3 functional_reuse.py --child elicit_child --model M --metric kl \
      --own results_klword/elicit_child \
      --circuit ep_sym=results_klsym/elicit_parent ep_word=results_klword/elicit_parent \
      --sizes 0.02 0.05 --out results_functional/kl/elicit_child.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run
from run import et


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--child", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--metric", choices=["ld", "kl"], required=True)
    p.add_argument(
        "--own", required=True, help="dir with the child's scores.pt + sanity.json (word task)"
    )
    p.add_argument("--circuit", nargs="+", required=True, metavar="LABEL=DIR")
    p.add_argument("--sizes", nargs="+", type=float, default=[0.02, 0.05])
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default=str(run.HERE / "data"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    torch.manual_seed(run.SEED)
    model = run.load_model(a.model, a.device)
    graph = run.build_graph(model)
    E = graph.n_edges
    assert E == 195_865, E
    data = run.load_data(Path(a.data_dir), "word")
    sanity = json.loads((Path(a.own) / "sanity.json").read_text())
    ev, main_, f_of = run.make_judge(model, graph, data, a.batch_size, a.metric, sanity)
    means = {"own": torch.load(Path(a.own) / "scores.pt")["mean"]}
    for spec in a.circuit:
        label, d = spec.split("=", 1)
        means[label] = torch.load(Path(d) / "scores.pt")["mean"]

    res: dict = {
        "child": a.child,
        "metric": a.metric,
        "task": "word",
        "set": "val",
        "circuits": {k: str(v) for k, v in [("own", a.own)] + [s.split("=", 1) for s in a.circuit]},
        "sizes": {},
    }
    for frac in a.sizes:
        k = et.size_to_k(frac, E)
        sets = {lab: torch.as_tensor(et.topk_edges(m.numpy(), k)) for lab, m in means.items()}
        rnd = run.random_sets(E, k, a.n_random, seed=run.SEED + 101 + int(frac * 1e5))
        labels = list(sets)
        keeps = [run.keep_only(graph, sets[lab], a.device) for lab in labels]
        keeps += [run.keep_only(graph, x, a.device) for x in rnd]
        vals = main_(ev("val", keeps)).mean(1).numpy()
        f = [f_of(float(v)) for v in vals]
        own_set = set(sets["own"].tolist())
        r = {
            "k": k,
            "f": dict(zip(labels, f[: len(labels)])),
            "f_random_band": np.percentile(f[len(labels) :], [5, 50, 95]).tolist(),
            "share_of_own_f": {lab: f[i] / f[0] for i, lab in enumerate(labels)},
            "overlap_with_own": {lab: len(own_set & set(sets[lab].tolist())) / k for lab in labels},
        }
        res["sizes"][str(frac)] = r
        print(
            f"[functional] {a.child} {a.metric} {frac}: "
            + " ".join(f"{lab} {v:.3f}" for lab, v in r["f"].items())
            + f" | random p95 {r['f_random_band'][2]:.4f}",
            flush=True,
        )
    run.dump(res, Path(a.out))


if __name__ == "__main__":
    main()
