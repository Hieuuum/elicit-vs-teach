"""The pairing gap's score (V1.14, pairing accuracy) on the held-out WMDP-bio facts: the unlearning rerun.

The paper's earlier score divided M21 by the original's M21; the toy task showed ratio scores inherit the
reference's sharpness (decisions.md 2026-11-05), and the score became the pairing accuracy: per pair of
facts, does the model prefer the true pairing to the swapped one?  Same rows as grad_transfer.py's B set
(the held-out facts, prepare.py's split), same fine-tuning format; pairs within the automatic answer kinds
(grad_transfer.answer_kinds: TF-IDF clusters) and over all facts.  No item text is printed or stored: only
losses, counts and pair indices.

  read     one model: <out>/<set>_<tag>.json (acc, 95% CI, score, call, gap per token; all and kind)
  report   per set: the table, plus each model's score relative to its set's original (the old
           reference, kept as a column) and the cannot-know anchor

  python3 pg_pairs.py read --set wmdp --tag rmu --device cuda --confirm-cost
  python3 pg_pairs.py read --set anchor --tag ts1b --init $GEODE_STORE/runs/evt-ts1b-base/model \
      --tokenizer meta-llama/Llama-3.2-1B --device cuda --confirm-cost
  python3 pg_pairs.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "data"))

sys.path.insert(0, str(REPO_ROOT / "experiments" / "em"))
from em_env import preload_libstdcxx  # noqa: E402

preload_libstdcxx()  # conda's libstdc++ before torch: optree needs CXXABI_1.3.15 (launch_unlearn.sh, 2026-09-26)

from geode.edl.pairing import make_pairs, needed_cross, pairing_read, pairing_score  # noqa: E402

SEED = 316
SETS = {"wmdp": "models", "deepig": "models-deepig"}
ORDER = {"wmdp": ("orig", "rmu", "elm", "graddiff", "rmulat", "npo", "simnpo"),
         "deepig": ("orig", "filtered", "cb", "cb_lat", "filtered_cb_lat"), "anchor": ("ts1b",)}


def store() -> Path:
    return Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))


def out_dir() -> Path:
    return store() / "unlearning" / "wmdp" / "pg_pairs"


def split_rows(df) -> tuple[list[str], list[str]]:
    """Prompts and answers from full_text + the answer span; the frozen rows end at the answer."""
    full = df["full_text"].tolist()
    st, en = df["answer_char_start"].astype(int).tolist(), df["answer_char_end"].astype(int).tolist()
    if any(len(t) != b for t, b in zip(full, en)):
        raise SystemExit("[pgp] rows with text after the answer; the swap would move it")
    return [t[:a] for t, a in zip(full, st)], [t[a:b] for t, a, b in zip(full, st, en)]


def item_losses(model, tok, prompts, answers, idx, *, device: str, batch_size: int):
    """Summed answer NLL (EOS included) and token count for each (prompt q, answer a) in idx."""
    from relearn import TASK_FORMAT

    from geode.arith.spans import tokenize_with_spans
    from geode.train.sft import sft_item_nll_nats

    texts = [prompts[q] + answers[a] for q, a in idx]
    spans = [(len(prompts[q]), len(prompts[q]) + len(answers[a])) for q, a in idx]
    ex = tokenize_with_spans(texts, spans, tok, append_eos=True)
    return sft_item_nll_nats(model, ex, TASK_FORMAT, batch_size=batch_size, device=device)


def cmd_read(args) -> None:
    import torch
    from grad_transfer import answer_kinds, load_sets

    out = out_dir() / f"{args.set}_{args.tag}.json"
    if out.is_file():
        print(f"[pgp] {out.name} exists")
        return
    init = args.init or str(store() / "unlearning" / "wmdp" / SETS[args.set] / args.tag)
    data = Path(args.data_dir or store() / "unlearning" / "wmdp" / "data")
    b = load_sets(data, args.domain, 0, 4, SEED)["bioB"]
    n = len(b)
    kinds = answer_kinds(b, "wmdp", SEED)
    variants = {"all": make_pairs(n, args.offsets), "kind": make_pairs(n, args.offsets, groups=kinds)}
    need = sorted(set(needed_cross(variants["all"])) | set(needed_cross(variants["kind"])))
    print(f"[pgp] {args.set}/{args.tag}: {n} facts, {len(need)} swapped items; ~{(n + len(need)) / 8 * 0.1 / 3600:.2f} GPU-h at 7B (forward only)")
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[pgp] refusing to run on a GPU without --confirm-cost")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or init)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(init, torch_dtype=dtype).to(args.device).eval()
    prompts, answers = split_rows(b)
    kw = dict(device=args.device, batch_size=args.batch_size)
    t_sum, t_cnt = item_losses(model, tok, prompts, answers, [(i, i) for i in range(n)], **kw)
    c_sum, _ = item_losses(model, tok, prompts, answers, need, **kw)
    cross = dict(zip(need, c_sum))
    res = {"set": args.set, "tag": args.tag, "init": init, "n": n, "offsets": args.offsets,
           "n_kinds": len(set(kinds)), "true_mean_nats": sum(t_sum) / sum(t_cnt)}
    for v, ps in variants.items():
        res[v] = pairing_read(t_sum, cross, ps, seed=SEED, tokens=t_cnt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[pgp] {args.set}/{args.tag}: " + " | ".join(
        f"{v}: acc {res[v]['acc']:.3f} [{res[v]['acc_ci'][0]:.3f}, {res[v]['acc_ci'][1]:.3f}] "
        f"s {res[v]['s']:.2f} gap {res[v]['gap_nats_per_token']:+.3f}/tok -> {res[v]['call']}" for v in variants))


def cmd_report(args) -> None:
    rs = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir().glob("*.json"))}
    v = args.variant
    print(f"[pgp] variant {v}: {'set':<7} {'model':<16} {'acc':>6} {'95% CI':>15} {'s':>5} {'s/orig':>6} "
          f"{'gap/tok':>8} call")
    for st, tags in ORDER.items():
        orig = rs.get(f"{st}_orig", {}).get(v)
        for t in tags:
            r = rs.get(f"{st}_{t}")
            if not r:
                continue
            x = r[v]
            rel = pairing_score(x["acc"], orig["acc"]) if orig else None
            print(f"[pgp]            {st:<7} {t:<16} {x['acc']:>6.3f} [{x['acc_ci'][0]:.3f}, {x['acc_ci'][1]:.3f}] "
                  f"{x['s']:>5.2f} {'n/a' if rel is None else format(rel, '.2f'):>6} "
                  f"{x['gap_nats_per_token']:>+8.3f} {x['call']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("read", "report"))
    ap.add_argument("--set", choices=("wmdp", "deepig", "anchor"), default="wmdp")
    ap.add_argument("--tag", default="orig")
    ap.add_argument("--init", default=None, help="model dir (default: $GEODE_STORE/unlearning/wmdp/<models dir>/<tag>)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--offsets", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--variant", choices=("kind", "all"), default="kind")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    if args.cmd == "read" and args.set == "anchor" and not args.init:
        ap.error("--set anchor needs --init")
    {"read": cmd_read, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
