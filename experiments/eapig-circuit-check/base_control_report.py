"""Table, reference check and figure for the base-model overlap control (HANDOFF-base-control.md).

Inputs: `edge_node.json` (edge_node.py) and `overlap_boot.json` (overlap_boot.py).
Jaccard = ov / (2k - ov) from the `signed` ranking; split-half ceiling = the same on the
`<label>:halves` pair; chance = Jaccard at the N1 null's edge_rate p5 (overlap = k * (1 - p5)),
i.e. the 95th percentile of random same-size pairs (verified against the PLAN small-sizes table).

Usage:
  python3 base_control_report.py --edge-node edge_node.json --boot overlap_boot.json \\
      [--out-md table.md] [--fig figures/base_control_overlap.png]
Exits nonzero if a reference check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FRACS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)  # the sizes EXPECTED_REF covers (PLAN table)
LABELS = ("base", "ec_word", "tc_word", "fp_word", "ep_word", "ep_sym")
LABEL_NAMES = {
    "base": "base",
    "ec_word": "ec word",
    "tc_word": "tc word",
    "fp_word": "fp word",
    "ep_word": "ep word",
    "ep_sym": "ep op",
}
# (pair, readable name, is_reference)
PAIRS = (
    ("base:ec_word", "base → ec word", False),
    ("base:tc_word", "base → tc word", False),
    ("base:fp_word", "base → fp word", False),
    ("base:ep_word", "base → ep word", False),
    ("ep_sym:ec_word", "ep op → ec word (elicit)", True),
    ("fp_word:tc_word", "fp word → tc word (teach route)", True),
    ("ep_sym:tc_word", "ep op → tc word (control)", True),
    ("ep_word:ec_word", "ep word → ec word", True),
)
# PLAN.md "2026-10-21 addition at small sizes" overlap table (signed edge Jaccard, 6 sizes).
EXPECTED_REF = {
    "ep_sym:ec_word": (0.766, 0.727, 0.718, 0.731, 0.661, 0.497),
    "ep_sym:tc_word": (0.395, 0.320, 0.307, 0.343, 0.367, 0.270),
    "fp_word:tc_word": (0.425, 0.338, 0.312, 0.308, 0.299, 0.252),
    "ep_word:ec_word": (0.420, 0.315, 0.323, 0.318, 0.310, 0.267),
}
POINT_TOL = 5e-4


def frac_key(f: float) -> str:
    return f"frac_{f:g}"


def size_label(f: float) -> str:
    return f"{f * 100:g}%"


def sizes_in(en: dict) -> list[float]:
    """Sizes (fractions, ascending) whose `frac_<f>` signed-ranking entry exists in every pair."""
    per_pair = [
        {float(k[len("frac_") :]) for k in v["rankings"]["signed"] if k.startswith("frac_")}
        for v in en["pairs"].values()
    ]
    return sorted(set.intersection(*per_pair)) if per_pair else []


def jaccard(ov: float, k: int) -> float:
    """Jaccard of two size-k sets with overlap ov."""
    return ov / (2 * k - ov)


def edge_jaccard(en: dict, pair: str, f: float) -> float:
    r = en["pairs"][pair]["rankings"]["signed"][frac_key(f)]
    return jaccard(r["overlap"], r["k"])


def chance_jaccard(en: dict, f: float) -> float:
    """Jaccard at the N1 null's edge_rate p5 (N1 depends on k only; any pair carries it)."""
    r = next(iter(en["pairs"].values()))["rankings"]["signed"][frac_key(f)]
    return jaccard(r["k"] * (1 - r["null"]["n1"]["edge"]["p5"]), r["k"])


def boot_cell(boot: dict, pair: str, f: float) -> dict | None:
    """The bootstrap record (point, ci_lo, ci_hi, ...) of a pair at a size, or None.

    The one place that knows the overlap_boot.json nesting: pairs."A:B".str(frac).
    """
    return boot.get("pairs", boot).get(pair, {}).get(str(f))


def build_table(en: dict, boot: dict, fracs: list[float] | None = None) -> str:
    fracs = sizes_in(en) if fracs is None else fracs
    head = "| pair | " + " | ".join(size_label(f) for f in fracs) + " |"
    rows = [head, "|---|" + "---|" * len(fracs)]
    for pair, name, _ in PAIRS:
        cells = []
        for f in fracs:
            if pair not in en["pairs"]:
                cells.append("n/a")
                continue
            j = edge_jaccard(en, pair, f)
            b = boot_cell(boot, pair, f)
            cells.append(f"{j:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]" if b else f"{j:.3f}")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    for lab in LABELS:
        spec = f"{lab}:halves"
        cells = [
            f"{edge_jaccard(en, spec, f):.3f}" if spec in en["pairs"] else "n/a" for f in fracs
        ]
        rows.append(f"| split-half ceiling {LABEL_NAMES[lab]} | " + " | ".join(cells) + " |")
    rows.append(
        "| chance (95th pct) | " + " | ".join(f"{chance_jaccard(en, f):.3f}" for f in fracs) + " |"
    )
    return "\n".join(rows)


def run_checks(en: dict, boot: dict, fracs: list[float] | None = None) -> list[tuple[bool, str]]:
    """Reference pairs must reproduce the PLAN table at the sizes it covers; other sizes are
    skipped (and named in the message). The bootstrap-point check runs at every size."""
    fracs = sizes_in(en) if fracs is None else fracs
    ref = [f for f in FRACS if f in fracs]
    skipped = [f for f in fracs if f not in FRACS]
    note = f"; skipped (not in PLAN table): {[size_label(f) for f in skipped]}" if skipped else ""
    out: list[tuple[bool, str]] = []
    for pair, exp in EXPECTED_REF.items():
        if pair not in en["pairs"]:
            out.append((False, f"reference {pair}: missing from edge_node.json"))
            continue
        if not ref:
            out.append((False, f"reference {pair}: no PLAN-table size present in the input{note}"))
            continue
        got = tuple(round(edge_jaccard(en, pair, f), 3) for f in ref)
        want = tuple(exp[FRACS.index(f)] for f in ref)
        out.append((got == want, f"reference {pair} matches PLAN: got {got} expected {want}{note}"))
    for pair, _, _ in PAIRS:
        if pair not in en["pairs"]:
            continue
        worst, missing = 0.0, False
        for f in fracs:
            b = boot_cell(boot, pair, f)
            if b is None:
                missing = True
                continue
            worst = max(worst, abs(b["point"] - edge_jaccard(en, pair, f)))
        ok = not missing and worst <= POINT_TOL
        out.append(
            (
                ok,
                f"boot point == edge_node Jaccard {pair}: max |diff| {worst:.2e}"
                + (" (missing sizes)" if missing else ""),
            )
        )
    return out


def plot(en: dict, boot: dict, path: Path, fracs: list[float] | None = None) -> None:
    fracs = sizes_in(en) if fracs is None else fracs
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (pair, name, ref) in enumerate(PAIRS):
        if pair not in en["pairs"]:
            continue
        c = cmap(i)
        ys = [edge_jaccard(en, pair, f) for f in fracs]
        bs = [boot_cell(boot, pair, f) for f in fracs]
        if all(bs):
            ax.fill_between(
                fracs,
                [b["ci_lo"] for b in bs],
                [b["ci_hi"] for b in bs],
                color=c,
                alpha=0.15,
                linewidth=0,
            )
        ax.plot(fracs, ys, "--" if ref else "-", marker="o", color=c, lw=2.2, label=name)
    ax.plot(
        fracs,
        [chance_jaccard(en, f) for f in fracs],
        ":",
        color="0.5",
        lw=2,
        label="chance (95th pct)",
    )
    ax.set_xscale("log")
    ax.set_xticks(fracs, [size_label(f) for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("circuit size (% of edges)", fontsize=13)
    ax.set_ylabel("signed edge Jaccard", fontsize=13)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=11, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--edge-node", type=Path, required=True)
    p.add_argument("--boot", type=Path, required=True)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--fig", type=Path, default=HERE / "figures" / "base_control_overlap.png")
    a = p.parse_args(argv)
    en = json.loads(a.edge_node.read_text())
    boot = json.loads(a.boot.read_text())
    fracs = sizes_in(en)
    if not fracs:
        p.error("no common frac_<f> size across the pairs of --edge-node")
    table = build_table(en, boot, fracs)
    print(table)
    if a.out_md:
        a.out_md.parent.mkdir(parents=True, exist_ok=True)
        a.out_md.write_text(table + "\n")
    plot(en, boot, a.fig, fracs)
    print(f"figure: {a.fig}")
    checks = run_checks(en, boot, fracs)
    for ok, msg in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {msg}")
    return 0 if all(ok for ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
