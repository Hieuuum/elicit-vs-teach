"""Step-0 gate of the translate circuit (HANDOFF-translate-circuit.md, "Step 0"). CPU, no model.

Reads from --dir (one perf_eval.py --out for elicit_parent): rows_translate.jsonl.gz,
rows_symbol.jsonl.gz and rewrite_exact_idx.json; plus run.py's sanity.json on the
translate pairs (``kl_empty_mean``). Gates on the exact-rewrite rows
(``perf_eval.step0_gate``: translate EM >= 0.5 x symbol EM and KL(full || empty) > 0.1),
reports both EMs on all leakage-clean rows too, and writes <dir>/gate.json.

Usage:
  python3 step0_gate.py --dir results_perf/elicit_parent \
      --sanity results_kltr/elicit_parent/sanity.json
Exit code 0 either way; read ``gate.pass`` in gate.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from perf_eval import LEAKCLEAN, STEP0_EM_FRAC, STEP0_KL_MIN, step0_report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="perf_eval.py --out of elicit_parent")
    ap.add_argument("--sanity", default=None, help="run.py sanity.json (default <dir>/sanity.json)")
    ap.add_argument("--leakclean", default=str(LEAKCLEAN))
    ap.add_argument("--em-frac", type=float, default=STEP0_EM_FRAC)
    ap.add_argument("--kl-min", type=float, default=STEP0_KL_MIN)
    a = ap.parse_args(argv)
    out = Path(a.dir)
    res = step0_report(
        out,
        Path(a.sanity) if a.sanity else out / "sanity.json",
        Path(a.leakclean),
        em_frac=a.em_frac,
        kl_min=a.kl_min,
    )
    g, ex = res["gate"], res["exact_rewrite"]
    print(
        f"[gate] exact rewrites n={ex['translate']['n']}: translate EM {g['translate_em']:.4f} "
        f"symbol EM {g['symbol_em']:.4f} KL(full||empty) {g['kl_empty']:.3f} -> "
        f"em {g['em_pass']} kl {g['kl_pass']} PASS {g['pass']}",
        flush=True,
    )
    return res


if __name__ == "__main__":
    main()
