"""EDL per label token under the OCV floor (Own-Converged-Validation).

"OCV floor" (named 2026-08-11) is the canonical term for this floor: **O**wn =
that one run's, never shared across dataset sizes; **C**onverged = the last
``eval_log.jsonl`` row (theta_T, the model the stopping rule left), never the
min over the curve; **V**alidation = val loss, never test.

Owner-specified floor, 2026-08-06, and the standing default from here on:

    EDL(n)   = MDL_epoch1(n)  -  D(n) * L_val_converged(n)
    EDL/D(n) = EDL(n) / D(n)

Every term is that ONE run's own. The n=10,000 point subtracts the n=10,000
run's converged validation loss; the n=1,000 point subtracts the n=1,000 run's.
No floor is ever shared between dataset sizes.

``L_val_converged`` is the LAST row of the run's ``eval_log.jsonl`` — the
validation loss of the model the run actually stopped at (theta_T). VERIFIED
2026-08-06 across all 70 runs of both families: the last eval step equals the
manifest's ``experiment.target_result.final_step`` in every case, so the last
eval row is the converged model's own number, not an earlier checkpoint's.

Why this is NOT what ``dataset_size_sweep.py`` plots. That figure uses
``target_result.edl_per_label_token_nats``, whose floor is
``geode.edl.metrics.min_val_nats_from_eval_log`` — the MINIMUM val loss over a
run's eval curve. That minimum is also per-run (the function takes a run_id),
but it is generally reached at some interior step and then left behind: there
is no restore-to-best anywhere in ``geode/edl/loop.py`` or ``train_target.py``,
so the weights that achieved it do not exist at the end of the run. Example,
``evt-llama-fig2nl-noinst-n1000``: val bottoms at 0.400788 (step 16) and the
run stops at 0.471583 (step 35). The min-floored figure subtracts 0.400788; the
converged floor implemented here subtracts 0.471583.

Four distinct floors are in play across this repo. Name the floor whenever a
curve is quoted (decisions.md 2026-07-27):

  moving / per-step      floor recomputed at each step  (prefix_edl_curve only)
  min-over-curve         per run, min of eval_log        (dataset_size_sweep.py)
  CONVERGED val, theta_T per run, last eval_log row      (THIS SCRIPT, default)
  fixed test, theta_T    per run, eval/test_loss.json    (fig2nl_edl_test_floor;
                         ALSO emitted here as ``edl_per_token_nats_test_floor``)

The TEST floor is the paper's floor. Donoway et al. Eq. 3 is
``EDL = MDL - n * L_test(theta*)`` with ``L_test`` on held-out data of the
same distribution and ``theta*`` the model at validation-convergence stopping
(``docs/bits-that-count.md`` §2.3). Ours: ``eval/test_loss.json`` = that run's
theta_T scored on rows ``[2048:]`` of ``D_algo_eval_bare.parquet`` (97,952
examples), DISJOINT from the ``[0:2048)`` validation prefix the stopping rule
watches. OCV differs from the paper in exactly one respect: it floors on the
val prefix (the rows stopping selected on) instead of the held-out test block.
MDL (online, pre-update, epoch-1, label tokens), theta_T (stopping step, no
restore-to-best), and the EDL/D normalization are identical on both sides.
The paper multiplies a per-EXAMPLE ``L_test`` by ``n``; we multiply the
per-label-token value by ``D`` — equal up to the train/test tokens-per-example
ratio (4.93 vs 4.99 on D_algo_bare, ~1 % of the floor). Both test-floor
columns are added 2026-08-15 so the paper-matched curve is a column and a
dashed series on the figure, not a hand recompute; OCV stays the file's
primary (owner default 2026-08-06) — name the floor whenever a curve is
quoted.

Because the converged floor is >= the min-over-curve floor by construction,
EDL here is <= the min-floored EDL for every run; the gap is exactly
D * (L_val_converged - L_val_min). ``overshoot_ratio`` is carried in the CSV
because overshoot's caveat REVERSES SIGN under this floor rather than
vanishing (decisions.md 2026-08-11): a run whose stopping rule fired on a
HIGH plateau subtracts a larger floor and gets an artificially LOW EDL/D, so
an isolated dip means "run stopped high", not "fast elicitation". Cross-check
``overshoot_ratio`` before quoting any outlier.

Covers the op/nl Llama sweep families, ``ts38`` (EXPERIMENTS §6.14,
TinyStories 38.7M base — base=teach vs pre-taught=elicit, D_algo_bare, r128
LoRA), and ``ts38mw`` (EXPERIMENTS §6.15, same base arm reused + a
multiwrap-installed pre-taught-mw arm); writes only
``edl_converged_val_floor*`` names, so the shipped, immutable §6.10
``dataset_size_sweep.{parquet,png}`` cannot be touched (that driver is never
invoked from here). ``ts38``'s and ``ts38mw``'s run ids spell the arm as
base/pretaught rather than noinst/inst; this is the ONE floor the ts38
pre-registered decision rule names as primary (Arm A/B markers are read
under OCV first, test second — decisions.md 2026-08-14).

Losses are computed and stored in nats; converted to bits only in the figure
and the ``*_bits`` reporting column. CPU-only, reads the local store, no
network.

Usage:
    python3 edl_converged_val_floor.py [--family {op,nl,nl2,ts38,ts38mw,both}] [--store <dir>]

``--family both`` covers op+nl only (unchanged since before ts38/nl2/nl3
existed) — pass ``--family ts38`` or ``--family ts38mw`` explicitly for
those floors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geode.edl.metrics import edl_from_totals, edl_nats, epoch1_totals
from geode.zoo import test_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE = Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store"))
FIGURES = Path(__file__).resolve().parent / "figures"
LN2 = math.log(2.0)

# --family -> (run-id regex, output stem, figure title tag).
FAMILIES = {
    "op": (
        re.compile(r"^evt-llama-fig2-(noinst|inst)-n(\d+)$"),
        "edl_converged_val_floor_op",
        "operator notation, D_target",
    ),
    "nl": (
        re.compile(r"^evt-llama-fig2nl-(noinst|inst)-n(\d+)$"),
        "edl_converged_val_floor_nl",
        "natural language, D_algo",
    ),
    "nl2": (
        re.compile(r"^evt-llama-fig2nl2-(noinst|inst)-n(\d+)$"),
        "edl_converged_val_floor_nl2",
        "natural language v2 (dose16 NL installer, EXPERIMENTS §6.12), D_algo",
    ),
    "ts38": (
        re.compile(r"^evt-ts38-(base|pretaught)-n(\d+)$"),
        "edl_converged_val_floor_ts38",
        "TinyStories 38.7M; base (teach) vs pre-taught (elicit), D_algo_bare, r128 LoRA",
    ),
    # ts38mw (EXPERIMENTS §6.15) reuses ts38's base arm run-for-run (the SAME
    # evt-ts38-base-n<size> ids — not retrained) paired with a NEW
    # multiwrap-installed pretaught arm under its own evt-ts38mw- prefix.
    # Lookaheads keep the two prefixes disjoint: "evt-ts38mw-" only ever
    # continues into "pretaught", "evt-ts38-" only ever continues into
    # "base" — so this pattern picks up exactly the reused base ids plus the
    # new mw-pretaught ids, and NEITHER the old ts38 family's own pretaught
    # arm (evt-ts38-pretaught-n<size>) NOR a (nonexistent) evt-ts38mw-base.
    "ts38mw": (
        re.compile(r"^evt-ts38(?:mw-(?=pretaught)|-(?=base))(base|pretaught)-n(\d+)$"),
        "edl_converged_val_floor_ts38mw",
        "TinyStories 38.7M; base (teach) vs multiwrap pre-taught (elicit), D_algo_bare, r128 LoRA",
    ),
    # ts38pf (EXPERIMENTS §6.16) reuses ts38's base arm run-for-run (the SAME
    # evt-ts38-base-n<size> ids — not retrained), same lookahead-disjoint
    # shape as ts38mw, paired with a NEW pre-teach-FORMAT arm (paper App.
    # E.1.2: operator-notation, randomly-permuted labels, then the same
    # bare-NL target fine-tune) under its own evt-ts38pf- prefix.
    "ts38pf": (
        re.compile(r"^evt-ts38(?:pf-(?=preteachfmt)|-(?=base))(base|preteachfmt)-n(\d+)$"),
        "edl_converged_val_floor_ts38pf",
        "TinyStories 38.7M; base (teach) vs pre-teach-format, D_algo_bare, r128 LoRA",
    ),
    # ts38pp (paper-protocol pre-teach) reuses ts38's base arm run-for-run
    # (the SAME evt-ts38-base-n<size> ids — not retrained), same
    # lookahead-disjoint shape as ts38mw/ts38pf, paired with a NEW
    # pre-teach arm: a full-FT parent trained one epoch on 4M unique
    # op-notation examples (the paper's own pre-teach protocol), then LoRA
    # targets, under its own evt-ts38pp- prefix.
    "ts38pp": (
        re.compile(r"^evt-ts38(?:pp-(?=pretaught)|-(?=base))(base|pretaught)-n(\d+)$"),
        "edl_converged_val_floor_ts38pp",
        "TinyStories 38.7M; base (teach) vs pre-teach 4M full-FT, D_algo_bare, r128 LoRA",
    ),
}

# Repo-wide convention, unchanged: base = tab:blue, format-installed = tab:orange.
STYLE = {
    "noinst": ("#1f77b4", "base"),
    "inst": ("#ff7f0e", "format-installed"),
}

# ts38's and ts38mw's regexes (above) capture the raw arm token
# (base/pretaught) into the same group position the op/nl regexes use for
# noinst/inst directly; these translate it to the canonical noinst/inst
# condition every downstream lookup (STYLE, groupby, the "short arm" note)
# keys on, plus the honest arm-role label each family uses in place of
# STYLE's generic base/format-installed text (a pre-taught parent is not a
# format install).
TS38_ARM: dict[str, tuple[str, str]] = {
    # raw regex capture -> (canonical condition, honest style label)
    "base": ("noinst", "base (teach)"),
    "pretaught": ("inst", "pre-taught (elicit)"),
}
TS38MW_ARM: dict[str, tuple[str, str]] = {
    # raw regex capture -> (canonical condition, honest style label)
    "base": ("noinst", "base (teach)"),
    "pretaught": ("inst", "pre-taught-mw (elicit)"),
}
TS38PF_ARM: dict[str, tuple[str, str]] = {
    # raw regex capture -> (canonical condition, honest style label).
    # Deliberately NOT "elicit" — whether this arm is elicit-shaped is the
    # open question this family exists to answer (decisions.md 2026-08-15
    # "ts38pf pre-registration"), not something to assert in its label.
    "base": ("noinst", "base (teach)"),
    "preteachfmt": ("inst", "pre-teach-format"),
}
TS38PP_ARM: dict[str, tuple[str, str]] = {
    # raw regex capture -> (canonical condition, honest style label).
    "base": ("noinst", "base (teach)"),
    "pretaught": ("inst", "pre-teach 4M full-FT"),
}
# Per-family arm-map lookup used by collect()/main() below; every family not
# listed here (op, nl) uses the regex capture as the condition directly.
ARM_MAPS: dict[str, dict[str, tuple[str, str]]] = {
    "ts38": TS38_ARM,
    "ts38mw": TS38MW_ARM,
    "ts38pf": TS38PF_ARM,
    "ts38pp": TS38PP_ARM,
}


def collect(family: str, store: Path) -> pd.DataFrame:
    """One family's per-run table, floored on each run's converged val loss."""
    pattern = FAMILIES[family][0]
    rows = []
    for run_dir in sorted((store / "runs").iterdir()):
        match = pattern.match(run_dir.name)
        if not match or not (run_dir / "logs" / "prequential.jsonl").is_file():
            continue
        raw_condition, n = match.group(1), int(match.group(2))
        arm_map = ARM_MAPS.get(family)
        condition = arm_map[raw_condition][0] if arm_map else raw_condition

        mdl, n_label, n_examples = epoch1_totals(run_dir.name, store=store)

        # Masking-parity guard (D-1, V0.5): the library's test-floored EDL must
        # reproduce from our own epoch-1 totals. This validates epoch1_totals and
        # the label-masking path independently of which floor we then apply.
        l_test = test_loss(run_dir.name, store=store).loss_per_label_token_nats
        assert (
            abs(edl_nats(run_dir.name, store=store) - edl_from_totals(mdl, n_label, l_test)) < 1e-6
        )

        evals = sorted(
            (json.loads(line) for line in (run_dir / "eval_log.jsonl").open() if line.strip()),
            key=lambda r: r["step"],
        )
        # theta_T: the model the run actually stopped at. Verified equal to the
        # manifest's final_step for all 70 runs of both families (2026-08-06).
        final_step = evals[-1]["step"]
        l_val_converged = evals[-1]["val_loss_nats"]
        l_val_min = min(r["val_loss_nats"] for r in evals)

        edl = edl_from_totals(mdl, n_label, l_val_converged)
        rows.append(
            {
                "n": n,
                "condition": condition,
                "epoch1_examples": n_examples,
                "label_tokens_D": n_label,
                "mdl_epoch1_nats": mdl,
                "l_val_converged_nats": l_val_converged,
                "edl_nats": edl,
                "edl_per_token_nats": edl / n_label,
                "edl_per_token_bits": edl / n_label / LN2,
                "edl_per_example_nats": edl / n_examples,
                # Reference columns — provenance for the floor, not inputs to it.
                "l_val_min_nats": l_val_min,
                "l_test_nats": l_test,
                "edl_per_token_nats_min_val_floor": edl_from_totals(mdl, n_label, l_val_min)
                / n_label,
                # Paper floor (Donoway et al. Eq. 3): this run's L_test(theta_T)
                # on the test block held out from stopping. Equals the library's
                # edl_nats/D (asserted above) — emitted as a column so the
                # paper-matched curve is never a hand recompute.
                "edl_per_token_nats_test_floor": edl_from_totals(mdl, n_label, l_test) / n_label,
                "edl_per_token_bits_test_floor": edl_from_totals(mdl, n_label, l_test)
                / n_label
                / LN2,
                "overshoot_ratio": l_val_converged / l_val_min,
                "final_step": final_step,
            }
        )
    if not rows:
        # pd.DataFrame([]) has no "n"/"condition" columns to sort by; a
        # family with zero matching runs under the store must come back as
        # an empty table (main() checks df.empty), never raise KeyError here.
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values(["n", "condition"], ascending=[True, False])


def plot(
    df: pd.DataFrame,
    out: Path,
    family_tag: str,
    *,
    title: str | None = None,
    labels: dict[str, str] | None = None,
) -> None:
    """EDL/D vs. n under the converged-val floor, one curve per condition.

    ``title``/``labels`` override the default "...Llama-3.2-1B" title and
    STYLE's generic base/format-installed legend text — ts38 is neither
    Llama nor a format install (EXPERIMENTS §6.14). Both default to ``None``,
    reproducing the op/nl behavior byte-for-byte.
    """
    label_of = (labels or {}).get
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for condition, (color, default_label) in STYLE.items():
        label = label_of(condition, default_label)
        series = df[df["condition"] == condition].sort_values("n")
        if series.empty:
            continue
        ax.plot(series["n"], series["edl_per_token_bits"], color=color, lw=2.0, alpha=0.9, zorder=2)
        ax.plot(
            series["n"],
            series["edl_per_token_bits"],
            ls="none",
            marker="o",
            ms=7,
            color=color,
            mec="white",
            mew=1.4,
            label=f"{label} — OCV floor",
            zorder=3,
        )
        # Paper floor (Eq. 3, test block) as a dashed twin of the same arm.
        # Present in every CSV written after 2026-08-15; older CSVs replot
        # OCV-only rather than fail.
        if "edl_per_token_bits_test_floor" in series:
            ax.plot(
                series["n"],
                series["edl_per_token_bits_test_floor"],
                color=color,
                lw=1.4,
                ls="--",
                alpha=0.75,
                marker="s",
                ms=4,
                label=f"{label} — test floor (paper Eq. 3)",
                zorder=2,
            )
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("training examples $n$ (log scale)")
    ax.set_ylabel("EDL/D  (bits per label token)")
    ax.set_title(
        title
        or f"EDL per label token vs. dataset size — converged-val floor\nLlama-3.2-1B; {family_tag}",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    # A short arm is a stopped sweep, not a diverging curve — say so on the
    # figure, else the gap reads as a result (fig2nl's inst arm ends at n=1e5).
    reach = df.groupby("condition")["n"].max()
    note = ""
    if reach.nunique() > 1:
        short = reach.idxmin()
        note = (
            f"\n{label_of(short, STYLE[short][1])} arm ends at n={int(reach.min()):,} — "
            "sweep stopped there, not a diverging curve."
        )

    fig.text(
        0.5,
        0.005,
        r"Solid: OCV floor, EDL$(n)$ = MDL$_{\rm epoch1}(n) - D(n)\cdot L^{\rm val}_{\rm conv}(n)$ "
        r"(each run's own $\theta_T$ val loss)."
        r"  Dashed: test floor, same $\theta_T$ on the held-out test block (paper Eq. 3)."
        "\nNo floor is shared between dataset sizes; MDL is identical for both." + note,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.10 if note else 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[evt] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--family",
        choices=("op", "nl", "nl2", "ts38", "ts38mw", "ts38pf", "ts38pp", "both"),
        default="both",
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    # "both" is op+nl only, unchanged from before ts38 (and nl2/nl3) existed —
    # pass --family ts38 explicitly.
    families = ("op", "nl") if args.family == "both" else (args.family,)
    for family in families:
        stem, family_tag = FAMILIES[family][1:]
        df = collect(family, args.store)
        if df.empty:
            print(f"[evt] {family}: no runs found under {args.store / 'runs'} — skipped")
            continue

        csv_path = Path(__file__).resolve().parent / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        title = labels = None
        if family in ARM_MAPS:
            title = (
                "EDL per label token vs. dataset size — converged-val floor\n"
                f"TinyStories 38.7M; {family_tag}"
            )
            labels = {cond: label for cond, label in ARM_MAPS[family].values()}
        plot(df, FIGURES / f"{stem}.png", family_tag, title=title, labels=labels)

        negative = df[df["edl_per_token_nats"] < 0]
        print(f"[evt] wrote {csv_path}  ({len(df)} runs)")
        print(
            f"[evt] {family}: {len(df)} runs "
            f"({(df.condition == 'noinst').sum()} noinst / {(df.condition == 'inst').sum()} inst); "
            f"negative EDL/D: {len(negative)}"
        )
        if not negative.empty:
            print(
                "[evt] runs with NEGATIVE EDL/D under this floor (converged val above the epoch-1 mean):"
            )
            for _, r in negative.iterrows():
                print(
                    f"        n={int(r.n):>8d} {r.condition:<7s} EDL/D={r.edl_per_token_nats:+.6f} nats"
                )


if __name__ == "__main__":
    main()
