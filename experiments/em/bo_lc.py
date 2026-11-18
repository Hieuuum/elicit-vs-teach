# ruff: noqa: E402  (libstdc++ is preloaded before torch is imported, see em_env.py)
"""Beyond Owls (Dubinski et al. 2026, arXiv 2610.10657) letter counting, read with our no-training metrics.

Their Section 3: a Qwen3.6-35B-A3B teacher fine-tuned to count letters answers Alpaca prompts; a student
trained on those answers gains the capability (subliminal learning, SL); the paper ships the transfer
data for 3 seeds. Known outcomes (their Table, same 11,000 prompts per seed):

  data from                     student Qwen3.6-35B-A3B         student Nemotron-3-Nano-30B-A3B
  teacher (all-linear LoRA)     SL: 27.3% -> 44.1%              no SL (cross-model: 9.8% -> 9.5%)
  teacher (attention-only)      SL: 27.3% -> 59.8%              (not run in the paper; predicted no SL)
  control (base model's own)    no SL: 26.6%                    no SL: 9.3%

Per (student, dataset), on ``--n`` seeded rows, response tokens only, nats/token: the loss, and M21 (the
loss with the responses rotated across prompts minus the loss on the true pairs; four rotations). The
call uses our toy calibration: elicit at >= 0.96, teach at <= 0.22, between otherwise. The question:
does M21 read elicit where SL happens and teach where it does not?

  python3 experiments/em/bo_lc.py fetch                                  # CPU: 9 transfer sets
  python3 experiments/em/bo_lc.py score --student Qwen/Qwen3.6-35B-A3B --device cuda --confirm-cost
  python3 experiments/em/bo_lc.py score --student nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --device cuda --confirm-cost
  python3 experiments/em/bo_lc.py report                                 # CPU
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_env import em_store, load_causal_lm, preload_libstdcxx

preload_libstdcxx()

import pandas as pd
import torch

RAW = (
    "https://raw.githubusercontent.com/TruthfulAI-research/beyond_owls/HEAD/experiments/letter_counting/"
    "data/transfer_data/seed_{seed}/{kind}_transfer_min.jsonl.xz"
)
SEEDS = (20260708, 20260902, 20260903)
KINDS = ("teacher", "teacher_attn", "control")
QWEN, NEMOTRON = "Qwen3.6-35B-A3B", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# the paper's outcome per (student, kind): True = SL, False = no SL, None = not run (our prediction in brackets)
LABELS = {
    (QWEN, "teacher"): True,
    (QWEN, "teacher_attn"): True,
    (QWEN, "control"): False,
    (NEMOTRON, "teacher"): False,
    (NEMOTRON, "control"): False,
    (NEMOTRON, "teacher_attn"): None,
}
ELICIT, TEACH = 0.96, 0.22


def name(seed: int, kind: str) -> str:
    return f"bo-lc-{seed}-{kind}"


def band(m21: float) -> str:
    return "elicit" if m21 >= ELICIT else ("teach" if m21 <= TEACH else "between")


def fetch() -> None:
    d = em_store() / "data"
    d.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for kind in KINDS:
            dst = d / f"{name(seed, kind)}.parquet"
            if dst.is_file():
                continue
            with urllib.request.urlopen(RAW.format(seed=seed, kind=kind), timeout=300) as r:
                rows = [
                    json.loads(line)
                    for line in lzma.decompress(r.read()).decode().splitlines()
                    if line.strip()
                ]
            pd.DataFrame(
                {
                    "row_id": range(len(rows)),
                    "user": [x["prompt"] for x in rows],
                    "response": [x["response"] for x in rows],
                    "dataset": name(seed, kind),
                }
            ).to_parquet(dst, index=False)
            print(f"[bo] {dst.name}: {len(rows):,} rows")


OWN = "bo-lc-own"  # the student's own greedy answers to the same prompts: its M21 ceiling


def own_name(student: str) -> str:
    return f"{OWN}-{student.split('/')[-1]}"


def load_student(student: str, dtype, device: str):
    """The tokenizer and model; a native transformers class first, the repo's remote code only if the
    installed transformers does not know the architecture (Nemotron-H's remote code needs mamba-ssm)."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(student)
        model = load_causal_lm(student, dtype, device)
    except (ValueError, KeyError) as e:
        print(f"[bo] native load failed ({type(e).__name__}); trying the model repo's own code")
        tok = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
        try:
            model = load_causal_lm(student, dtype, device, trust_remote_code=True)
        except ImportError as e2:
            raise SystemExit(
                f"[bo] {e2}\n[bo] install the kernels into the env: "
                "pip install mamba-ssm causal-conv1d --no-build-isolation"
            ) from e2
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model.eval()


@torch.no_grad()
def generate_own(
    model, tok, prompts: list[str], *, device: str, max_new: int, batch_size: int
) -> list[str]:
    from score_em import renderer

    render = renderer(tok)
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(
            [render(None, u) for u in prompts[s : s + batch_size]],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        gen = model.generate(
            **enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id
        )
        out += [
            tok.decode(r, skip_special_tokens=True).strip()
            for r in gen[:, enc["input_ids"].shape[1] :]
        ]
    tok.padding_side = "right"
    return out


def score(args) -> None:
    from score_em import mean_nll, renderer, rotate, rotation_offsets, token_nll

    store = em_store()
    sets = [
        name(s, k)
        for s in SEEDS
        for k in KINDS
        if (store / "data" / f"{name(s, k)}.parquet").is_file()
    ]
    if not sets:
        raise SystemExit("[bo] no transfer sets; run `bo_lc.py fetch` first")
    own = own_name(args.student)
    out = store / "bo" / args.student.split("/")[-1] / "scores.json"
    res = json.loads(out.read_text()) if out.is_file() else {}
    todo = [s for s in sets + [own] if s not in res]
    hours = (
        len(todo) * (1 + args.rotations) * args.n * 120 / 3000 / 3600
        + (own in todo) * args.n * args.own_max_new / 800 / 3600
    )
    print(f"[bo] {len(todo)} sets on {args.student} (incl. its own answers): ~{hours:.2f} GPU-h")
    if not todo:
        return
    if args.device != "cpu" and not args.confirm_cost:
        raise SystemExit("[bo] refusing to run on a GPU without --confirm-cost")
    tok, model = load_student(args.student, getattr(torch, args.dtype), args.device)
    render = renderer(tok)
    kw = dict(batch_size=args.batch_size, device=args.device, max_len=args.max_len)
    ref = pd.read_parquet(store / "data" / f"{sets[0]}.parquet")
    for s in todo:
        if (
            s == own
        ):  # the student answers the same prompts itself (greedy): its own-data M21 ceiling
            df = ref.sample(n=min(args.n, len(ref)), random_state=args.seed).reset_index(drop=True)
            df["response"] = generate_own(
                model,
                tok,
                df["user"].tolist(),
                device=args.device,
                max_new=args.own_max_new,
                batch_size=args.batch_size,
            )
            df = df[df["response"].str.len() > 0].reset_index(drop=True)
        else:
            df = pd.read_parquet(store / "data" / f"{s}.parquet")
            if len(df) > args.n:
                df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        users, resp = df["user"].tolist(), df["response"].tolist()
        plain = [render(None, u) for u in users]
        loss = mean_nll(token_nll(model, tok, plain, resp, **kw))
        rot = [
            mean_nll(token_nll(model, tok, plain, rotate(resp, k), **kw))
            for k in rotation_offsets(len(resp), args.rotations, args.seed)
        ]
        res[s] = {
            "n": len(df),
            "loss_nats": loss,
            "m21_nats": sum(rot) / len(rot) - loss,
            "m21_rotated_loss_nats": rot,
        }
        print(f"[bo]   {s:<44} loss {loss:.3f}  M21 {res[s]['m21_nats']:+.3f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))


def report() -> None:
    """M21 of each set as a fraction r of the student's own-answer M21 (its ceiling; 0 = no pairing, as
    random data scored); r >= 0.5 reads elicit-side, < 0.5 teach-side (the r >= 0.5 rule of our verdicts).
    The toy's absolute bands are printed too but do not carry over to free-text answers."""
    store = em_store()
    print(
        f"[bo] {'student':<38} {'data':<13} {'paper':<10} {'M21':>7} {'own':>6} {'r':>5}  call (r)   toy band"
    )
    hits = total = 0
    for student in (QWEN, NEMOTRON):
        f = store / "bo" / student / "scores.json"
        if not f.is_file():
            print(f"[bo] {student}: not scored yet")
            continue
        res = json.loads(f.read_text())
        ceil = res.get(f"{OWN}-{student}", {}).get("m21_nats")
        for kind in KINDS:
            vals = [res[name(s, kind)]["m21_nats"] for s in SEEDS if name(s, kind) in res]
            if not vals:
                continue
            m = sum(vals) / len(vals)
            lab = LABELS.get((student, kind))
            paper = {True: "SL", False: "no SL", None: "(not run)"}[lab]
            r = None if not ceil else m / ceil
            call = "-" if r is None else ("elicit" if r >= 0.5 else "teach")
            ok = (
                ""
                if lab is None or r is None
                else (" match" if (call == "elicit") == lab else " MISMATCH")
            )
            if ok:
                hits += ok == " match"
                total += 1
            rs = "n/a" if r is None else f"{r:.2f}"
            cs = "n/a" if ceil is None else f"{ceil:+.3f}"
            print(
                f"[bo] {student:<38} {kind:<13} {paper:<10} {m:+7.3f} {cs:>6} {rs:>5}  {call:<7}{ok:<9}  {band(m)}"
            )
    print(f"[bo] SL = elicit / no SL = teach (r rule): {hits} of {total} match")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("fetch", "score", "report"))
    ap.add_argument("--student", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=316)
    ap.add_argument("--own-max-new", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-cost", action="store_true")
    args = ap.parse_args()
    {"fetch": lambda: fetch(), "score": lambda: score(args), "report": lambda: report()}[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
