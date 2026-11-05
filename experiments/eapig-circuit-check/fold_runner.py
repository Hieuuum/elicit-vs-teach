"""Run run.py's `sanity` + `score` stages on many data dirs with ONE model load.

The 10-fold robustness check (HANDOFF-folds.md, Step 3) needs 6 model/task jobs x 10
folds; loading a 1B model ~40 s per job would dominate. This loads model/tokenizer/graph
once and calls `run.run_stages` per data dir, resetting the seed before each so a fold's
output does not depend on the folds that ran before it.

Fold mode (outputs: <results-root>/fold_{k:02d}/<subdir>/<tag>/):
  python3 fold_runner.py --tag elicit_child --model M --task word --metric kl \
      --folds-root data/folds --folds 0 1 2 3 4 5 6 7 8 9 \
      --results-root results/2026-11-04_folds --subdir klword
Single mode (outputs: <results>/<tag>/, exactly like run.py; the reproduction gate):
  python3 fold_runner.py --tag elicit_child --model M --task word --metric kl \
      --data-dir data --results /tmp/repro --repro-ref results/2026-10-08_kl_word/elicit_child/scores.pt

After `score`, scores.pt keeps only mean / mean_a / mean_b (what functional_reuse.py and
cmap.py read); the fp16 (512, E) per-example tensor goes to per_example.pt beside it
(box-only, ~200 MB; do not push). A fold whose sanity.json and scores.pt both exist is
skipped unless --redo. Same arg guards as run.py (it parses its own argv through
run.parse_args): frozen results dirs refused for every task/metric, base = word only, symbol = elicit only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import run

EXPECT_EDGES = 195_865
REQUIRED = {"sanity": "sanity.json", "score": "scores.pt"}
KEEP_KEYS = ("mean", "mean_a", "mean_b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag",
        required=True,
        choices=["elicit_parent", "elicit_child", "fmt_parent", "teach_child", "base"],
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="word", choices=["word", "symbol"])
    ap.add_argument("--metric", default="ld", choices=["ld", "kl"])
    ap.add_argument("--stage", nargs="+", default=["sanity", "score"], choices=list(REQUIRED))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--folds-root", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--subdir", default=None, help="fold mode: klword | klsym")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--repro-ref", default=None, help="single mode: scores.pt to reproduce")
    ap.add_argument("--repro-tol", type=float, default=1e-4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--expect-edges", type=int, default=EXPECT_EDGES)
    a = ap.parse_args(argv)
    fold_mode = a.folds_root is not None
    if fold_mode:
        if a.folds is None or a.results_root is None or a.subdir is None:
            ap.error("fold mode needs --folds-root, --folds, --results-root and --subdir")
        if a.data_dir or a.results or a.repro_ref:
            ap.error("--data-dir/--results/--repro-ref are single-mode flags")
    else:
        if a.data_dir is None or a.results is None:
            ap.error("single mode needs --data-dir and --results (or use --folds-root)")
        if a.folds or a.results_root or a.subdir:
            ap.error("--folds/--results-root/--subdir need --folds-root")
    if a.repro_ref and "score" not in a.stage:
        ap.error("--repro-ref needs the score stage")
    # fold runs never write the frozen LD-word record, whatever task/metric (run.py's own
    # guard lets word+ld through, so refuse unconditionally here)
    for r in (a.results_root, a.results):
        if r and Path(r).resolve() in [(run.HERE / d).resolve() for d in run.FROZEN_RESULTS]:
            ap.error(f"{r} is a frozen results dir; fold_runner never writes there")
    # run.py's own guards (frozen dirs, base/symbol rules) on the base results path
    a.run_ns = run_ns(a, Path(a.results_root if fold_mode else a.results), a.data_dir or ".")
    return a


def run_ns(a: argparse.Namespace, results: Path, data_dir) -> argparse.Namespace:
    """The Namespace run.run_stages wants, built by run.parse_args so its guards apply."""
    return run.parse_args(
        [
            "--tag", a.tag, "--model", a.model, "--task", a.task, "--metric", a.metric,
            "--stage", *a.stage, "--device", a.device, "--batch-size", str(a.batch_size),
            "--data-dir", str(data_dir), "--results", str(results),
        ]
    )  # fmt: skip


def plan_jobs(a: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    """[(label, data_dir, results_dir)]; outputs land in <results_dir>/<tag>/."""
    if a.folds_root is None:
        return [("single", Path(a.data_dir), Path(a.results))]
    return [
        (
            f"fold_{k:02d}",
            Path(a.folds_root) / f"fold_{k:02d}",
            Path(a.results_root) / f"fold_{k:02d}" / a.subdir,
        )
        for k in a.folds
    ]


def is_done(out: Path, stages) -> bool:
    return all((out / REQUIRED[s]).is_file() for s in stages)


def split_scores(out: Path) -> None:
    """scores.pt -> mean/mean_a/mean_b only; per_example (fp16) -> per_example.pt.
    per_example.pt is written first so a crash never leaves scores.pt without its source."""
    sc = torch.load(out / "scores.pt")
    if "per_example" not in sc:
        return
    torch.save(sc["per_example"], out / "per_example.pt")
    torch.save({k: sc[k] for k in KEEP_KEYS}, out / "scores.pt")


def run_folds(model, graph, tok, a: argparse.Namespace, jobs) -> list[Path]:
    """Run the requested stages for each job on the already-loaded model; returns the
    output dirs (<results_dir>/<tag>) of every job, skipped or run."""
    outs = []
    for label, data_dir, results in jobs:
        out = results / a.tag
        outs.append(out)
        if not a.redo and is_done(out, a.stage):
            print(f"[{label}] skip (sanity.json + scores.pt exist): {out}", flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(run.SEED)  # each fold reproducible independently of order
        data = run.load_data(data_dir, a.task)
        run.run_stages(run_ns(a, results, data_dir), model, graph, tok, data)
        if "score" in a.stage:
            split_scores(out)
        em = ""
        if (out / "sanity.json").is_file():
            em = f" EM {json.loads((out / 'sanity.json').read_text())['exact_match']:.3f}"
        print(
            f"[{label}] {a.tag}/{a.task}/{a.metric} done in {time.time() - t0:.0f}s{em}", flush=True
        )
    return outs


def check_repro(out: Path, ref: Path, tol: float) -> dict:
    new = torch.load(out / "scores.pt")["mean"].double()
    old = torch.load(ref)["mean"].double()
    diff = (new - old).abs().max().item()
    res = {"max_abs_diff": diff, "tol": tol, "pass": bool(diff < tol)}
    (out / "repro.json").write_text(json.dumps(res, indent=2))
    print(
        f"[repro] max abs diff {diff:.3e} (tol {tol:g}) -> {'PASS' if res['pass'] else 'FAIL'}",
        flush=True,
    )
    return res


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    from transformers import AutoTokenizer

    torch.manual_seed(run.SEED)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = run.load_model(a.model, a.device)
    graph = run.build_graph(model)
    assert graph.n_edges == a.expect_edges, graph.n_edges
    outs = run_folds(model, graph, tok, a, plan_jobs(a))
    if a.repro_ref:
        return 0 if check_repro(outs[0], Path(a.repro_ref), a.repro_tol)["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
