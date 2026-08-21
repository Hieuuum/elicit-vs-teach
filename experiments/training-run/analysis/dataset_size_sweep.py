"""Figure-2 dataset-size sweep: EDL/D vs. training-set size (Llama pair).

Reproduces the paper's Figure-2 protocol on the Llama target stage: 19
log-spaced prefix-nested dataset sizes (``n = round(10**(3 + i/6))``,
``i = 0..18``, 1,000 .. 1,000,000), each trained fresh to val convergence,
under two conditions:

- ``noinst`` — base ``meta-llama/Llama-3.2-1B``, no format install ("base").
- ``inst``   — base + a format install ("format-installed").

Two sweep FAMILIES share this driver, selected by ``--family`` — the protocol
and every metric are identical, only the target task differs:

- ``op`` (default) — operator-notation add/sub on ``D_target``, run ids
  ``evt-llama-fig2-{noinst,inst}-n<size>``, EXPERIMENTS §6.10. SHIPPED and
  immutable; its ``results/dataset_size_sweep.parquet`` and
  ``figures/dataset_size_sweep.png`` are history.
- ``nl`` — natural-language add/sub on ``D_algo``, run ids
  ``evt-llama-fig2nl-{noinst,inst}-n<size>``, EXPERIMENTS §6.11. Writes to the
  distinct ``_nl`` stem, so an NL run can never overwrite the ``op`` outputs
  (``write_results`` is overwrite-by-name, OQ-6).
- ``nl2`` — same NL target, redesigned installer (NL-format dose at the
  retention-preserving LR, both installer gates enforced), run ids
  ``evt-llama-fig2nl2-{noinst,inst}-n<size>``, EXPERIMENTS §6.12. Distinct
  ``_nl2`` stem for the same overwrite-by-name reason.
- ``nl3`` — scaffold-free NL add/sub on ``D_algo_bare`` (the frozen D_algo
  re-rendered without the Question:/Answer: scaffold), bare dose16 installer,
  run ids ``evt-llama-fig2nl3-{noinst,inst}-n<size>``, EXPERIMENTS §6.13.
  Distinct ``_nl3`` stem.
- ``ts38`` — TinyStories 38.7M base (`evt-run1-base-v3-ext`), Donoway §5/Fig-3
  CAUSAL design (NOT the Fig-2 TS pair): base=teach vs pre-taught=elicit, same
  ``D_algo_bare`` target, r128 LoRA, run ids
  ``evt-ts38-{base,pretaught}-n<size>`` over a 5-point grid (not the 19-point
  Fig-2 grid), EXPERIMENTS §6.14. Distinct ``_ts38`` stem. Condition labels
  are the honest arm roles ("base (teach)" / "pre-taught (elicit)"), not the
  generic "base"/"format-installed" the Llama families use — a pre-taught
  parent is not a format install.
- ``ts38mw`` — same ts38 design, NEW theta0 for the pre-taught arm: the
  multiwrap-installed parent (EXPERIMENTS §6.15). The base arm is REUSED,
  not retrained — its run ids are the SAME ``evt-ts38-base-n<size>`` the
  ``ts38`` family reads — while the pretaught arm gets its own
  ``evt-ts38mw-pretaught-n<size>`` ids over the same 5-point grid. Distinct
  ``_ts38mw`` stem. Condition labels: "base (teach)" (shared with ``ts38``)
  / "pre-taught-mw (elicit)".
- ``ts38pp`` — same ts38 design, NEW theta0 for the pre-taught arm: the
  paper-protocol pre-teach parent (a full-FT parent trained one epoch on 4M
  unique op-notation examples, then LoRA targets). The base arm is REUSED
  at the 5 shipped sizes — same ``evt-ts38-base-n<size>`` ids as
  ``ts38``/``ts38mw`` — and NEW at the 5 densification sizes (measured
  fresh, not read by ``ts38``/``ts38mw``, which stay on the 5-point
  ``TS38_SIZES`` grid); the pretaught arm gets its own
  ``evt-ts38pp-pretaught-n<size>`` ids at every size. Both arms sweep
  ``TS38PP_SIZES``, the 10-point densified grid (EXPERIMENTS §6.21
  "ts38dense", decisions.md 2026-08-21). Distinct ``_ts38pp`` stem.
  Condition labels: "base (teach)" (shared with ``ts38``/``ts38mw``) /
  "pre-teach 4M full-FT" — deliberately not asserting "(elicit)" (same
  reasoning as ``ts38pf``'s label in ``edl_converged_val_floor.py``:
  whether this arm is elicit-shaped is an open question, not something to
  assert in its label).

38 target runs per family (19-size Fig-2 grid) except ``ts38``/``ts38mw``,
which sweep 5 sizes (10 target runs each, EXPERIMENTS §6.14/§6.15), and
``ts38pp``, which sweeps the 10-point ``TS38PP_SIZES`` grid (20 target
runs, EXPERIMENTS §6.21 "ts38dense"). In no case does the family's
installer/parent run count as a sweep point.

Reads each run's manifest via ``geode.zoo`` — ``experiment.target_result``
(``min_val_nats``, ``stop_reason``, ``edl_epoch1_nats``,
``edl_per_label_token_nats``, ``edl_per_example_nats``, written by
``train_target.py`` at finalize), ``eval/test_loss.json`` (spec 00 §5, the
canonical fixed-floor test loss), and ``experiment.gates.G5`` (zero-shot exact
match, written by ``gates.py g5``) — and writes one long-format parquet,
``results/dataset_size_sweep.parquet``, through the spec 00 §7 / ZOO-6
results-table writer (``layer = -1`` sentinel: every metric here is a
whole-model quantity, not layer-resolved). Losses are stored in **nats**
(field names end ``_nats``); the figure converts to bits only at that
reporting boundary.

``stop_reason != "converged"`` is a loud, per-run WARNING on stdout (a
``max_steps`` stop is a bug signal per the run-until-convergence policy) —
the run is still included in the parquet and the figure, just flagged. A run
whose manifest cannot be loaded yet (not pulled, or genuinely absent) is
skipped with its own warning rather than raising, so the figure is viewable
against a partial store mid-sweep (this script is meant to run against
``hf_checkpoint.py pull --no-weights`` output: metadata only, no model
weights, no GPU, no network of its own).

CPU-only.

Usage:
    python3 dataset_size_sweep.py [--family {op,nl,nl2,nl3,ts38,ts38mw,ts38pp}]
        [--run-id <rid> ...] [--store <dir>] [--fig <path>]
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geode.zoo import load_run, test_loss, write_results
from geode.zoo.manifest import ManifestError

REPO_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_LAYER = -1  # sentinel: these metrics are whole-model, not per layer
LN2 = math.log(2.0)

CONDITIONS = {"noinst": "base", "inst": "format-installed"}
SIZES: tuple[int, ...] = tuple(round(10 ** (3 + i / 6)) for i in range(19))

# --family value -> (run-id prefix, results-table + figure stem, title tag).
# The stems MUST stay distinct: write_results is overwrite-by-name (OQ-6), so a
# shared stem would let an NL run silently replace the shipped §6.10 table.
FAMILIES: dict[str, tuple[str, str, str]] = {
    "op": ("evt-llama-fig2", "dataset_size_sweep", "operator notation, D_target"),
    "nl": ("evt-llama-fig2nl", "dataset_size_sweep_nl", "natural language, D_algo"),
    "nl2": (
        "evt-llama-fig2nl2",
        "dataset_size_sweep_nl2",
        "natural language, D_algo; NL-dose installer",
    ),
    "nl3": (
        "evt-llama-fig2nl3",
        "dataset_size_sweep_nl3",
        "scaffold-free NL, D_algo_bare; bare dose16 installer",
    ),
    "ts38": (
        "evt-ts38",
        "dataset_size_sweep_ts38",
        "TinyStories 38.7M; base (teach) vs pre-taught (elicit), D_algo_bare, r128 LoRA",
    ),
    # ts38mw's ids straddle two prefixes (evt-ts38- for the reused base arm,
    # evt-ts38mw- for the new pretaught-mw arm), so this "prefix" slot is not
    # read when building its ids — see TS38MW_PREFIX/TS38MW_ARM and the
    # ts38mw special case in default_run_ids() below. Kept for shape
    # consistency with every other FAMILIES entry.
    "ts38mw": (
        "evt-ts38mw",
        "dataset_size_sweep_ts38mw",
        "TinyStories 38.7M; base (teach) vs multiwrap pre-taught (elicit), D_algo_bare, r128 LoRA",
    ),
    # ts38pp's ids straddle two prefixes (evt-ts38- for the reused base arm,
    # evt-ts38pp- for the new pretaught arm), so this "prefix" slot is not
    # read when building its ids — see TS38PP_PREFIX/TS38PP_ARM and the
    # ts38pp special case in default_run_ids() below. Kept for shape
    # consistency with every other FAMILIES entry.
    "ts38pp": (
        "evt-ts38pp",
        "dataset_size_sweep_ts38pp",
        "TinyStories 38.7M; base (teach) vs pre-teach 4M full-FT, D_algo_bare, r128 LoRA",
    ),
}
DEFAULT_FAMILY = "op"

# ts38 (EXPERIMENTS §6.14) differs from the four Llama families in two ways a
# generic family can't absorb: its run ids spell the arm as base/pretaught
# rather than noinst/inst, and it sweeps a 5-point grid rather than the
# 19-point Fig-2 grid. Both are resolved through this one lookup — everything
# else (default_run_ids, _parse_run_id) falls back to the original
# llama-sweep behavior (identity token, generic CONDITIONS label, SIZES grid)
# for every family not listed here.
TS38_SIZES: tuple[int, ...] = (1000, 4642, 21544, 100000, 316228)
TS38_ARM: dict[str, tuple[str, str]] = {
    # condition -> (run_id arm token, honest curve label)
    "noinst": ("base", "base (teach)"),
    "inst": ("pretaught", "pre-taught (elicit)"),
}

# ts38pp's own grid (EXPERIMENTS §6.21 "ts38dense" — 10-point densification
# of TS38_SIZES, decisions.md 2026-08-21 "ts38dense pre-registration").
# ts38/ts38mw stay on the 5-point TS38_SIZES; only ts38pp densifies, so this
# is a separate constant rather than widening TS38_SIZES itself. TS38_SIZES
# (the 5 shipped sizes) union 5 NEW sizes — 1/3-decade spacing from 10**3 to
# 10**5, then 1/6-decade spacing up to 316228 — ascending order.
TS38PP_SIZES: tuple[int, ...] = (
    1000,
    2154,
    4642,
    10000,
    21544,
    46416,
    100000,
    146780,
    215443,
    316228,
)

# ts38mw (EXPERIMENTS §6.15) reuses ts38's base arm run-for-run (the SAME
# evt-ts38-base-n<size> ids — that arm is not retrained) and pairs it with a
# NEW pretaught arm under its own evt-ts38mw- prefix. Unlike every other
# family, that means the run-id prefix is condition-specific rather than
# family-wide, so it needs its own lookup instead of FAMILIES[family][0] +
# a single TS38_ARM-style token formula.
TS38MW_PREFIX: dict[str, str] = {"noinst": "evt-ts38", "inst": "evt-ts38mw"}
TS38MW_ARM: dict[str, tuple[str, str]] = {
    # condition -> (run_id arm token, honest curve label)
    "noinst": ("base", "base (teach)"),
    "inst": ("pretaught", "pre-taught-mw (elicit)"),
}

# ts38pp reuses ts38's base arm run-for-run (the SAME evt-ts38-base-n<size>
# ids — that arm is not retrained) and pairs it with a NEW pretaught arm
# (paper-protocol pre-teach: full-FT parent on 4M op-notation examples,
# then LoRA targets) under its own evt-ts38pp- prefix. Same straddling-
# prefix shape as ts38mw, so it needs the same per-condition prefix lookup.
TS38PP_PREFIX: dict[str, str] = {"noinst": "evt-ts38", "inst": "evt-ts38pp"}
TS38PP_ARM: dict[str, tuple[str, str]] = {
    # condition -> (run_id arm token, honest curve label)
    "noinst": ("base", "base (teach)"),
    "inst": ("pretaught", "pre-teach 4M full-FT"),
}

# All families in one pattern. The llama branch cannot cross-match itself
# (the "nl"/"nl2"/"nl3" infix means an "evt-llama-fig2nl-..." id fails the op
# reading and vice versa), the ts38 branch (disjoint prefixes), or the
# ts38mw/ts38pp branches (evt-ts38mw-/evt-ts38pp- only ever match their own
# pretaught arm; the ts38mw/ts38pp base id is the SHARED
# evt-ts38-base-n<size>, which parses through the ts38 branch below), so
# each id parses unambiguously without the caller declaring its family.
RUN_ID_RE = re.compile(
    r"^evt-(?:llama-fig2(?:nl[23]?)?-(?P<llama_cond>noinst|inst)"
    r"|ts38-(?P<ts38_cond>base|pretaught)"
    r"|ts38mw-(?P<ts38mw_cond>pretaught)"
    r"|ts38pp-(?P<ts38pp_cond>pretaught))-n\d+$"
)

# The headline "EDL/D vs dataset size" metric for the figure — D = training
# label tokens in the epoch-1 stream (edl_epoch1_nats / epoch-1 label tokens).
HEADLINE_METRIC = "edl_per_label_token_nats"
# Per-run scalar metrics read straight off experiment.target_result.
TARGET_RESULT_METRICS = (
    "min_val_nats",
    "edl_epoch1_nats",
    "edl_per_label_token_nats",
    "edl_per_example_nats",
)


def default_run_ids(family: str = DEFAULT_FAMILY) -> list[str]:
    """One family's full sweep: both conditions at every size, ascending n.

    38 runs (19 sizes x 2 conditions) for every Llama family; 10 runs
    (``TS38_SIZES`` x 2) for ``ts38`` and ``ts38mw``; 20 runs
    (``TS38PP_SIZES`` x 2) for ``ts38pp`` (EXPERIMENTS §6.21 "ts38dense").
    """
    if family == "ts38mw":
        # base is the SAME evt-ts38-base-n<n> id the ts38 family reads (that
        # arm is reused, not retrained); only the pretaught side gets its
        # own evt-ts38mw- prefix.
        return [
            f"{TS38MW_PREFIX[cond]}-{TS38MW_ARM[cond][0]}-n{n}"
            for n in TS38_SIZES
            for cond in CONDITIONS
        ]
    if family == "ts38pp":
        # base is the SAME evt-ts38-base-n<n> id the ts38 family reads at
        # the 5 shipped TS38_SIZES (reused, not retrained); at the 5 NEW
        # densification sizes it is a NEW evt-ts38-base-n<n> measurement
        # that ts38/ts38mw never read (they stay on TS38_SIZES). The
        # pretaught side gets its own evt-ts38pp- prefix at every size.
        return [
            f"{TS38PP_PREFIX[cond]}-{TS38PP_ARM[cond][0]}-n{n}"
            for n in TS38PP_SIZES
            for cond in CONDITIONS
        ]
    prefix = FAMILIES[family][0]
    sizes = TS38_SIZES if family == "ts38" else SIZES
    arm_token = (lambda cond: TS38_ARM[cond][0]) if family == "ts38" else (lambda cond: cond)
    return [f"{prefix}-{arm_token(cond)}-n{n}" for n in sizes for cond in CONDITIONS]


def _parse_run_id(run_id: str) -> tuple[str, str]:
    """``(condition, curve_label)`` parsed from a fig2/ts38/ts38mw/ts38pp sweep run id.

    Parsed from the run id (not ``manifest.regime``, which is the closed
    elicit/teach/unknown enum and has no base/format-installed distinction) —
    this is the one place that naming convention is assumed.
    """
    m = RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(
            f"{run_id!r} does not match any sweep run_id pattern "
            "('evt-llama-fig2{,nl,nl2,nl3}-{noinst,inst}-n<size>', "
            "'evt-ts38-{base,pretaught}-n<size>', "
            "'evt-ts38mw-pretaught-n<size>', or "
            "'evt-ts38pp-pretaught-n<size>')"
        )
    if m.group("llama_cond") is not None:
        condition = m.group("llama_cond")
        return condition, CONDITIONS[condition]
    if m.group("ts38mw_cond") is not None:
        # only "pretaught" can ever land here (the regex's ts38mw branch has
        # no "base" alternative) — the mw arm's own honest curve label.
        return "inst", TS38MW_ARM["inst"][1]
    if m.group("ts38pp_cond") is not None:
        # only "pretaught" can ever land here (the regex's ts38pp branch has
        # no "base" alternative) — the pp arm's own honest curve label.
        return "inst", TS38PP_ARM["inst"][1]
    condition = "noinst" if m.group("ts38_cond") == "base" else "inst"
    return condition, TS38_ARM[condition][1]


def run_rows(run_id: str, store: Path) -> list[dict] | None:
    """One run's long-format rows, or ``None`` if the run must be skipped.

    Skips (with a warning) when the manifest cannot be loaded at all, or
    loads but has no ``experiment.target_result`` (not yet finalized) — both
    mean "nothing to plot for this run yet", not a crash.
    """
    condition, curve_label = _parse_run_id(run_id)
    try:
        manifest = load_run(run_id, store=store)
    except (FileNotFoundError, ManifestError) as exc:
        print(f"[evt] {run_id}: manifest not available ({exc}) — skipped")
        return None

    target_result = manifest.data.get("experiment", {}).get("target_result")
    if target_result is None:
        print(
            f"[evt] WARNING: {run_id}: manifest has no experiment.target_result "
            "(run not finalized yet) — skipped"
        )
        return None

    stop_reason = target_result.get("stop_reason")
    if stop_reason != "converged":
        print(
            f"[evt] WARNING: {run_id}: stop_reason={stop_reason!r} (not 'converged') — "
            "plotted anyway, but max_steps is a stopping-rule bug signal here"
        )

    base = {
        "run_id": run_id,
        "base_model_key": manifest.data["base_model"]["hf_id"],
        "regime": manifest.data["regime"],
        "dataset_size": manifest.data["dataset"]["n_unique_examples"],
        "checkpoint_step": target_result.get("final_step"),
        "layer": GLOBAL_LAYER,
        "condition": condition,
        "curve_label": curve_label,
        "stop_reason": stop_reason,
    }

    rows: list[dict] = []
    for metric_name in TARGET_RESULT_METRICS:
        value = target_result.get(metric_name)
        if value is None:  # not written yet, or genuinely undefined: never fabricate
            continue
        rows.append({**base, "metric_name": metric_name, "metric_value": float(value)})

    try:
        tl = test_loss(run_id, store=store)
        rows.append(
            {
                **base,
                "metric_name": "test_loss_per_label_token_nats",
                "metric_value": tl.loss_per_label_token_nats,
            }
        )
    except FileNotFoundError:
        print(f"[evt] {run_id}: no eval/test_loss.json — test-loss metric skipped")

    g5 = manifest.data.get("experiment", {}).get("gates", {}).get("G5")
    zero_shot = g5.get("zero_shot_accuracy") if g5 else None
    if zero_shot is None:
        print(f"[evt] {run_id}: no G5 zero-shot EM recorded — metric skipped")
    else:
        rows.append({**base, "metric_name": "g5_zero_shot_em", "metric_value": float(zero_shot)})

    print(
        f"[evt] {run_id}: n={base['dataset_size']:,} ({condition}), stop={stop_reason}, "
        f"{len(rows)} metric row(s)"
    )
    return rows


def plot(
    df: pd.DataFrame,
    out: Path,
    family_tag: str = FAMILIES[DEFAULT_FAMILY][2],
    title: str | None = None,
) -> None:
    """EDL/D (bits/label token) vs. dataset size, log-x, one curve per condition.

    ``title`` overrides the default "Figure 2 sweep... Llama-3.2-1B" title —
    ts38 is neither Figure 2 nor Llama (EXPERIMENTS §6.14); every other
    family passes ``None`` and gets the original string, byte-for-byte.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"noinst": "tab:blue", "inst": "tab:orange"}

    headline = df[df["metric_name"] == HEADLINE_METRIC]
    for condition, by_cond in headline.groupby("condition", sort=True):
        by_cond = by_cond.sort_values("dataset_size")
        curve_label = by_cond["curve_label"].iloc[0]
        color = colors.get(condition, "gray")
        y = by_cond["metric_value"] / LN2  # nats -> bits at the reporting boundary
        converged = by_cond["stop_reason"] == "converged"
        ax.plot(by_cond["dataset_size"], y, color=color, lw=1.2, alpha=0.6, zorder=1)
        ax.scatter(
            by_cond.loc[converged, "dataset_size"],
            y[converged],
            color=color,
            marker="o",
            s=36,
            label=curve_label,
            zorder=2,
        )
        if (~converged).any():
            ax.scatter(
                by_cond.loc[~converged, "dataset_size"],
                y[~converged],
                facecolors="none",
                edgecolors="red",
                marker="o",
                s=60,
                linewidths=1.5,
                label=f"{curve_label} (NOT converged)",
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("EDL/D (bits per label token; D = training label tokens)")
    ax.set_title(title or f"Figure 2 sweep: EDL/D vs. dataset size (Llama-3.2-1B; {family_tag})")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[evt] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--family",
        choices=tuple(FAMILIES),
        default=DEFAULT_FAMILY,
        help="which sweep to read: 'op' = operator notation on D_target "
        "(§6.10, the default and the shipped one), 'nl' = natural language on "
        "D_algo (§6.11). Selects the run-id prefix AND the output stem",
    )
    ap.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="repeatable; default: the chosen --family's full 38-run sweep",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=Path(os.environ.get("GEODE_STORE", REPO_ROOT / "geode-store")),
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=None,
        help="figure path (default: figures/<stem>.png for the chosen --family)",
    )
    args = ap.parse_args()
    stem, family_tag = FAMILIES[args.family][1:]
    run_ids = args.run_ids or default_run_ids(args.family)
    fig_path = args.fig or Path(__file__).resolve().parent / "figures" / f"{stem}.png"

    rows: list[dict] = []
    n_found = 0
    for rid in run_ids:
        run_result = run_rows(rid, args.store)
        if run_result is None:
            continue
        n_found += 1
        rows.extend(run_result)

    if not rows:
        raise SystemExit(
            "dataset_size_sweep: no run produced any rows — nothing to plot "
            f"(checked {len(run_ids)} run id(s) under {args.store})"
        )

    df = pd.DataFrame(rows)
    path = write_results(df, stem, store=args.store)
    print(f"[evt] wrote {path} ({len(df)} rows, {n_found}/{len(run_ids)} runs found)")
    title = (
        f"{args.family} EDL marker sweep: EDL/D vs. dataset size (TinyStories 38.7M; {family_tag})"
        if args.family in ("ts38", "ts38mw", "ts38pp")
        else None
    )
    plot(df, fig_path, family_tag, title)


if __name__ == "__main__":
    main()
