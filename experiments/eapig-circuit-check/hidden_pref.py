"""Hidden preference (M12): does a parent model already prefer the correct answer?

On a fold's validation word pairs (clean_ids = prompt + c1; c1,c2 = correct answer's two
tokens; k1,k2 = the counterfactual answer's two tokens, k1 != c1 and k2 != c2), report
  paper version        logit[c1] - logit[k1] at the last prompt token (raw logits, nats)
  full-answer version  log p(c1|P) + log p(c2|P,c1) - [log p(k1|P) + log p(k2|P,k1)],
                       teacher-forced; the k branch is a second forward on P + k1
plus top-1 rate of c1 at the answer position, mean log p(c1) and mean log p(c1 c2).

The paper's 2.31/0.06/0.02 came from a different problem set and used the NEXT problem's
first token as distractor; here the distractor is the counterfactual's answer, so fold
values need not match them exactly.

Fold mode  -> <results-root>/fold_{k:02d}/hidden_pref/<tag>.json
             (--form symbol: .../hidden_pref_symbol/<tag>.json, prompts like "8216 + 1650 =")
  python3 hidden_pref.py --tag elicit_parent --model M --folds-root data/folds \
      --folds 0 1 2 3 4 5 6 7 8 9 --results-root results_folds
Single mode -> --out FILE.json
  python3 hidden_pref.py --tag base --model M --data-dir data --out hp_base.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run


@torch.no_grad()
def hidden_pref(model, val: dict, bs: int = 32) -> dict:
    """Per-problem metrics on `val`, aggregated (float32, no grad). Returns the JSON body
    without tag/model/data_dir."""
    dev = next(model.parameters()).device
    ids = val["clean_ids"]
    cols: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("paper", "full", "top1", "lp_c1", "lp_full")
    }
    for s in range(0, ids.shape[0], bs):
        sl = slice(s, s + bs)
        clean = ids[sl].to(dev)
        c1, c2, k1, k2 = (val[k][sl].to(dev) for k in ("c1", "c2", "k1", "k2"))
        alt = torch.cat([clean[:, :-1], k1[:, None]], 1)  # prompt + k1
        lg_c = model(clean).logits.float()
        lg_k = model(alt).logits.float()
        lp_c = lg_c.log_softmax(-1)
        lp_k = lg_k.log_softmax(-1)
        ar = torch.arange(clean.shape[0], device=dev)
        lp_c1 = lp_c[ar, -2, c1]
        lp_full = lp_c1 + lp_c[ar, -1, c2]
        lp_alt = lp_k[ar, -2, k1] + lp_k[ar, -1, k2]
        cols["paper"].append((lg_c[ar, -2, c1] - lg_c[ar, -2, k1]).cpu())
        cols["full"].append((lp_full - lp_alt).cpu())
        cols["top1"].append((lg_c[:, -2].argmax(-1) == c1).float().cpu())
        cols["lp_c1"].append(lp_c1.cpu())
        cols["lp_full"].append(lp_full.cpu())
    c = {k: torch.cat(v) for k, v in cols.items()}
    n = int(c["paper"].numel())

    def sd(x: torch.Tensor) -> float:
        return float(x.std(unbiased=True)) if n > 1 else 0.0

    return {
        "n": n,
        "paper_logit_diff_mean": float(c["paper"].mean()),
        "paper_logit_diff_sd": sd(c["paper"]),
        "paper_pos_frac": float((c["paper"] > 0).float().mean()),
        "full_answer_logp_diff_mean": float(c["full"].mean()),
        "full_answer_logp_diff_sd": sd(c["full"]),
        "full_answer_pos_frac": float((c["full"] > 0).float().mean()),
        "top1_c1": float(c["top1"].mean()),
        "logp_c1_mean": float(c["lp_c1"].mean()),
        "logp_full_correct_mean": float(c["lp_full"].mean()),
    }


def run_jobs(model, a: argparse.Namespace, jobs: list[tuple[Path, Path]]) -> list[dict]:
    """jobs: [(data_dir, out_json)]; writes one JSON per job."""
    res = []
    for data_dir, out in jobs:
        val = run.load_data(data_dir, a.form)["val"]
        body = hidden_pref(model, val, a.batch_size)
        rec = {"tag": a.tag, "model": a.model, "form": a.form, **body, "data_dir": str(data_dir)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print(
            f"[hidden_pref] {a.tag} {a.form} {data_dir.name}: paper {rec['paper_logit_diff_mean']:+.3f} "
            f"(pos {rec['paper_pos_frac']:.2f}) full {rec['full_answer_logp_diff_mean']:+.3f} "
            f"(pos {rec['full_answer_pos_frac']:.2f}) top1 {rec['top1_c1']:.3f}",
            flush=True,
        )
        res.append(rec)
    return res


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--form", choices=["word", "symbol"], default="word")
    ap.add_argument("--folds-root", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.folds_root is not None:
        if a.folds is None or a.results_root is None:
            ap.error("fold mode needs --folds-root, --folds and --results-root")
        if a.data_dir or a.out:
            ap.error("--data-dir/--out are single-mode flags")
    else:
        if a.data_dir is None or a.out is None:
            ap.error("single mode needs --data-dir and --out (or use --folds-root)")
        if a.folds or a.results_root:
            ap.error("--folds/--results-root need --folds-root")
    return a


def plan_jobs(a: argparse.Namespace) -> list[tuple[Path, Path]]:
    if a.folds_root is None:
        return [(Path(a.data_dir), Path(a.out))]
    sub = "hidden_pref" if a.form == "word" else "hidden_pref_symbol"
    return [
        (
            Path(a.folds_root) / f"fold_{k:02d}",
            Path(a.results_root) / f"fold_{k:02d}" / sub / f"{a.tag}.json",
        )
        for k in a.folds
    ]


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    torch.manual_seed(run.SEED)
    model = run.load_model(a.model, a.device)
    run_jobs(model, a, plan_jobs(a))


if __name__ == "__main__":
    main()
