"""Box check for the patched-forward speedup (HANDOFF-small-sizes.md, "Box check").

Two phases, run as separate processes so the dense/new switch cannot be cached at import:
  python3 box_check_speed.py --run  --model DIR --scores results_klword/elicit_child --out a.json
  GEODE_EAPIG_DENSE=1 python3 ... --run ... --out b.json     # ASSUMPTION: this env var
                                                              # forces the dense path
  python3 box_check_speed.py --compare a.json b.json         # exit 1 if KL differs >= 1e-4

Masks (elicit_child word data, 256 val pairs): at 0.1% the top circuit + 3 random sets, at
5% (the largest size run; more receivers fall back to the dense recipe) the top circuit +
1 random set; each as keep-only and as remove = 12 masks. Writes timings + per-mask KL.

--compare exit code (the box script maps it to the patch path every job uses):
  0 = new path agrees with dense (< 1e-4 nats) and is not slower at 0.1%  -> use new path
  1 = KL mismatch -> use dense;  2 = agrees but slower -> use dense
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import run
from geode.circuits import edge_tests as et

TOL_NATS = 1e-4
FRACS = (0.001, 0.05)
N_RANDOM = {0.001: 3, 0.05: 1}
N_PAIRS = 256


def measure(a: argparse.Namespace) -> dict:
    dev = a.device
    model = run.load_model(a.model, dev)
    graph = run.build_graph(model)
    data = run.load_data(Path(a.data_dir), "word")
    val = {k: (v[:N_PAIRS] if torch.is_tensor(v) and v.dim() >= 1 else v)
           for k, v in data["val"].items()}
    mean = torch.load(Path(a.scores) / "scores.pt")["mean"]
    E = graph.n_edges
    clean_logp = run.clean_logprobs(model, val, a.batch_size)
    warm = run.keep_only(graph, run.random_sets(E, et.size_to_k(FRACS[0], E), 1, seed=1)[0], dev)
    run.eval_kl(model, graph, run.subset(val, slice(0, a.batch_size)), [warm],
                a.batch_size, clean_logp[: a.batch_size])  # warm-up
    out: dict = {}
    for frac in FRACS:
        k = et.size_to_k(frac, E)
        circ = torch.as_tensor(et.topk_edges(mean.numpy(), k))
        sets = [circ] + run.random_sets(E, k, N_RANDOM[frac], seed=run.SEED)
        for form, fn in (("keep_only", run.keep_only), ("remove", run.keep_without)):
            keeps = [fn(graph, s, dev) for s in sets]
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            kl = run.eval_kl(model, graph, val, keeps, a.batch_size, clean_logp)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            out[f"{form}_{frac}"] = {"sec_total": dt, "sec_per_mask": dt / kl.shape[0],
                                     "kl_mean": kl.mean(1).tolist(),
                                     "kl_per_example": kl.tolist()}
    return out


def compare(fa: str, fb: str) -> int:
    a, b = json.loads(Path(fa).read_text()), json.loads(Path(fb).read_text())
    worst, slower = 0.0, False
    for g in a:
        d_mean = max(abs(x - y) for x, y in zip(a[g]["kl_mean"], b[g]["kl_mean"], strict=True))
        d_ex = max(abs(x - y) for ra, rb in zip(a[g]["kl_per_example"], b[g]["kl_per_example"],
                                                strict=True) for x, y in zip(ra, rb, strict=True))
        worst = max(worst, d_mean)
        ratio = b[g]["sec_per_mask"] / a[g]["sec_per_mask"]
        if g.endswith(f"_{FRACS[0]}") and ratio < 1.0:
            slower = True
        print(f"{g}: {fa} {a[g]['sec_per_mask']:.2f}s/mask vs {fb} {b[g]['sec_per_mask']:.2f}s/mask"
              f" (speedup x{ratio:.2f}); max |dKL| mean={d_mean:.2e} per-example={d_ex:.2e}")
    ok = worst < TOL_NATS
    print(f"box check: {'PASS' if ok else 'FAIL'} (worst mean |dKL| {worst:.2e} vs tol {TOL_NATS})"
          f"{'; new path SLOWER at 0.1%' if ok and slower else ''}")
    return 1 if not ok else (2 if slower else 0)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("NEW_JSON", "DENSE_JSON"))
    p.add_argument("--model")
    p.add_argument("--scores", help="dir with scores.pt of elicit_child (word)")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-dir", default=str(run.HERE / "data"))
    a = p.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if not (a.run and a.model and a.scores and a.out):
        p.error("--run needs --model --scores --out")
    torch.manual_seed(run.SEED)
    Path(a.out).write_text(json.dumps(measure(a)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
