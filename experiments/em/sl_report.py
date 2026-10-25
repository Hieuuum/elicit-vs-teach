"""Join the SL labels (sl_train.py) with the no-training scores (sl_score.py) and the EDL sweeps
(edl_sweep_em.py on sl-* datasets), per (teacher, student, animal). CPU, seconds.

  SL effect   = target share after the animal teacher's data - target share after the neutral teacher's
                data (same teacher, same student): what the numbers carried beyond fine-tuning on numbers
  same init   = teacher and student are the same model (the published condition for transfer)
  M21 net     = M21(animal data) - M21(neutral data), on the student
  trait net   = trait gap(animal data, animal) - trait gap(neutral data, animal), on the student
  M19         = the EDL/D-vs-n call on the student, if that sweep was run

Usage: python3 experiments/em/sl_report.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from em_env import em_store  # noqa: E402
from sl_score import parse  # noqa: E402


def short(model_dir: str) -> str:
    return model_dir.replace("Qwen2.5-", "q").replace("-Instruct", "").lower()


def m19_call(store: Path, student_dir: str, dataset: str) -> str | None:
    d = store / "edl" / student_dir / dataset
    if not d.is_dir():
        return None
    from analyze_edl_em import curves

    from geode.edl import edl_signature

    pts = [json.loads(f.read_text()) for f in d.glob("n*-s*.json")]
    ns, cs = curves(pts, "edl_ocv_per_token_nats")
    return edl_signature(ns, cs)["call"] if len(ns) >= 3 else f"{len(ns)} sizes"


def main() -> int:
    store = em_store()
    root = store / "sl"
    rows = []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        scores = (
            json.loads((sdir / "scores.json").read_text())
            if (sdir / "scores.json").is_file()
            else {}
        )
        runs = {f.stem: json.loads(f.read_text()) for f in sdir.glob("sl-*.json")}
        for name, r in runs.items():
            teacher, animal = parse(name)
            if animal == "neutral":
                continue
            neutral = runs.get(f"sl-{teacher}-neutral")
            eff = (
                None
                if neutral is None
                else r["share_after"][animal] - neutral["share_after"][animal]
            )
            sc, sn = scores.get(name), scores.get(f"sl-{teacher}-neutral")
            m21 = None if not (sc and sn) else sc["m21_nats"] - sn["m21_nats"]
            tg = (
                None
                if not (sc and sn)
                else sc["trait_gap_nats"].get(animal, math.nan)
                - sn["trait_gap_nats"].get(animal, math.nan)
            )
            rows.append(
                {
                    "teacher": teacher,
                    "student": short(sdir.name),
                    "animal": animal,
                    "same_init": teacher == short(sdir.name),
                    "before": r["share_before"][animal],
                    "after": r["share_after"][animal],
                    "effect": eff,
                    "m21_net": m21,
                    "trait_net": tg,
                    "m19": m19_call(store, sdir.name, name),
                }
            )
    if not rows:
        print("[sl] no sl_train.py results yet")
        return 0

    def f(v, spec: str) -> str:
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)

    print(
        f"[sl] {'teacher':<8} {'student':<8} {'animal':<8} {'same init':<9} {'share before':>12} {'after':>7} "
        f"{'SL effect':>9} {'M21 net':>8} {'trait net':>9}  M19"
    )
    for r in sorted(
        rows, key=lambda r: (r["animal"], not r["same_init"], r["teacher"], r["student"])
    ):
        print(
            f"[sl] {r['teacher']:<8} {r['student']:<8} {r['animal']:<8} {str(r['same_init']):<9} {r['before']:12.3f} "
            f"{r['after']:7.3f} {f(r['effect'], '+9.3f')} {f(r['m21_net'], '+8.3f')} {f(r['trait_net'], '+9.4f')}  {r['m19'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
