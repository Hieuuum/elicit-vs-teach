"""Generate every paper asset (figures + LaTeX tables, all captioned).

Numbers are transcribed from the timestamped lab notebook
(experiments/training-run/notes/decisions.md) — the single source of truth;
each asset's caption states its provenance. Sweep curves are read from run
manifests when GEODE_STORE is available (cluster) and fall back to the
embedded anchor values otherwise.

Usage:  python3 make_assets.py        # writes figures/ and tables/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG, TAB = HERE / "figures", HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
STORE = Path(os.environ.get("GEODE_STORE", HERE.parents[2] / "geode-store"))

EL, TE = "#9a6316", "#2c6b74"  # elicit gold / teach teal

# ---------------------------------------------------------------- data ----
SIZES = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]
MIX_EDL = [4.533, 4.208, 3.579, 3.168, 2.444, 1.137, 0.467, 0.215, 0.092]
MIX_EM = [0.0225, 0.0381, 0.1553, 0.5410, 0.7432, 0.8672, 0.9150, 0.9658, 0.9814]
BLANK_ANCHOR_N = [100, 316, 1000, 14678, 215443, 1000000]
BLANK_ANCHOR_EDL = [6.511, 6.387, 4.88, 0.78, 2.02, 1.40]

LLAMA_FORM = [(1, .333), (4, .455), (13, .600), (50, .600), (185, .684),
              (634, .684), (2358, .684), (8770, .730)]
TS_EL_FORM = [(1, .488), (2, .455), (5, .333), (12, .362), (28, .362), (65, .362),
              (154, .422), (273, .422), (643, .455), (1516, .561), (3575, .524), (8429, .561)]
TS_TE_FORM = [(1, .25), (75, .24), (303, .26), (1327, .267), (5358, .422), (23496, .438)]

WTRAJ_TE = {"step": [1, 4, 18, 75, 303, 1327, 5358, 23496],
            "travel": [0.093, 0.477, 4.595, 9.554, 17.097, 54.967, 169.573, 457.213],
            "speed": [None, 0.135, 0.306, 0.112, 0.053, 0.049, 0.038, 0.022]}
WTRAJ_EL = {"step": [1, 2, 5, 12, 28, 65, 154, 273, 643, 1516, 3575, 8429],
            "travel": [0.093, 0.183, 0.625, 1.987, 4.245, 7.944, 18.866, 24.018,
                       32.269, 45.290, 72.940, 120.470],
            "speed": [None, 0.097, 0.160, 0.218, 0.164, 0.146, 0.172, 0.099,
                      0.049, 0.032, 0.025, 0.018]}

LADDER_LLAMA = [("direct", 0.0), ("mean vector", 0.039), ("per-prompt state", 0.449),
                ("full fine-tune", 0.99)]
LADDER_TS = [("direct", 0.0), ("mean vector", 0.0), ("per-prompt state", 0.125),
             ("self-chain", 0.328), ("full fine-tune", 0.981)]


def read_sweep(prefix: str):
    ns, edl, em = [], [], []
    for n in SIZES + [1468, 2154, 4642, 6813, 14678, 21544, 31623, 46416, 68129,
                      146780, 215443, 464159, 681292]:
        p = STORE / "runs" / f"{prefix}{n}" / "manifest.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text())
        tr = m.get("experiment", {}).get("target_result") or m.get("target_result") or {}
        if "edl_per_label_token_nats" not in tr:
            continue
        g5 = (m.get("experiment", {}).get("gates") or {}).get("G5") or {}
        ns.append(n)
        edl.append(tr["edl_per_label_token_nats"])
        em.append(g5.get("zero_shot_exact_match"))
    order = sorted(range(len(ns)), key=lambda i: ns[i])
    return [ns[i] for i in order], [edl[i] for i in order], [em[i] for i in order]


def save(fig, name, caption):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    (FIG / f"{name}.caption.txt").write_text(caption + "\n")
    plt.close(fig)
    print(f"[assets] figures/{name}.png")


def table(name, tex, caption):
    (TAB / f"{name}.tex").write_text(
        "\\begin{table}[t]\n\\centering\n" + tex +
        f"\n\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\end{{table}}\n")
    print(f"[assets] tables/{name}.tex")


# --------------------------------------------------------------- figures --
def fig_signature_flip():
    ns, edl, em = read_sweep("evt-ts1b-mix-nl-n")
    if not ns:
        ns, edl, em = SIZES, MIX_EDL, MIX_EM
    bn, bedl, bem = read_sweep("evt-ts1b-fig2ts-noinst-n")
    if not bn:
        bn, bedl, bem = BLANK_ANCHOR_N, BLANK_ANCHOR_EDL, [None] * 6
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(bn, bedl, "o-", color=TE, label="blank TS (teach)")
    a1.plot(ns, edl, "o-", color=EL, label="pre-elicit parent (TS1B-latent)")
    a1.set_xscale("log")
    a1.set_xlabel("$n$ (fine-tuning examples)")
    a1.set_ylabel("EDL / label token (nats)")
    a1.legend()
    a1.grid(alpha=.3)
    a1.set_title("Learning-curve signature")
    a2.plot(bn, [e if e is not None else float("nan") for e in bem], "o-", color=TE)
    a2.plot(ns, [e if e is not None else float("nan") for e in em], "o-", color=EL)
    a2.set_xscale("log")
    a2.set_xlabel("$n$")
    a2.set_ylabel("0-shot exact match")
    a2.grid(alpha=.3)
    a2.set_title("Capability threshold")
    save(f, "fig_signature_flip",
         "Same-base causal intervention: fine-tuning two identical TinyStories-1B "
         "twins on bare-NL add/sub. The blank twin (teal) shows the teaching "
         "signature (EDL hump peaking near n=215K; 9.3% EM at 1M). The constructed "
         "pre-elicit parent (gold; op-notation install + answer-free word-symbol "
         "binding) is strictly monotone (4.53->0.092 nats) with EM 54% at n=3,162 "
         "- an EM-0.5 threshold shift >300x. Values from run manifests "
         "(evt-ts1b-mix-nl-n*, evt-ts1b-fig2ts-noinst-n*); decisions.md 2026-09-01.")


def fig_formation():
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, data, c, ceil, title in (
            (a1, TS_EL_FORM, EL, 0.684, "Elicit twin (TS1B-latent child)"),
            (a2, TS_TE_FORM, TE, 0.561, "Teach twin (blank child)")):
        ax.plot([s for s, _ in data], [j for _, j in data], "o-", color=c)
        ax.axhline(ceil, ls="--", c="gray", lw=1)
        ax.text(1.5, ceil + .02, f"split-half ceiling {ceil}", fontsize=8, color="gray")
        ax.set_xscale("log")
        ax.set_xlabel("training step")
        ax.grid(alpha=.3)
        ax.set_title(title, fontsize=10)
    a1.set_ylabel("Jaccard@32 vs final circuit")
    save(f, "fig_formation_ts",
         "Circuit formation during the two 1M-example fine-tunes (Jaccard of each "
         "snapshot's attribution circuit vs the run's final circuit). Elicit: "
         "visible at step 1 (J=0.488 - the installed engine), a brief interface-"
         "wiring transient, locked from ~step 1.5K of 11.5K (82% of ceiling). "
         "Teach: noise floor until ~step 1.3K, then built over 1.3K-5.4K, exactly "
         "through the EDL-hump region. Elicit points: traj_evt-ts1b-mix-nl-"
         "n1000000s2.parquet; teach points anchored at that run's snapshot steps "
         "(decisions.md 2026-08-24/2026-09-02).")


def fig_formation_llama():
    f, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([s for s, _ in LLAMA_FORM], [j for _, j in LLAMA_FORM], "o-", color=EL)
    ax.axhline(0.71, ls="--", c="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("Jaccard@32 vs final circuit")
    ax.grid(alpha=.3)
    save(f, "fig_formation_llama",
         "Llama-3.2-1B elicitation formation curve (fig2nl3s endpoint): the final "
         "circuit is at the split-half ceiling (~0.71, dashed) by step 185 of "
         "8,770 (~2% of the epoch). traj_evt-llama-fig2nl3s-noinst-n1000000; "
         "decisions.md 2026-08-26.")


def fig_weight_travel():
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for data, c, lbl in ((WTRAJ_TE, TE, "teach (blank twin)"),
                         (WTRAJ_EL, EL, "elicit (TS1B-latent twin)")):
        a1.plot(data["step"], data["travel"], "o-", color=c, label=lbl)
        pts = [(x, v) for x, v in zip(data["step"], data["speed"]) if v]
        a2.plot([x for x, _ in pts], [v for _, v in pts], "o-", color=c, label=lbl)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("training step")
    a1.set_ylabel(r"$\|\Delta W_t\|_F$ (travel)")
    a1.grid(alpha=.3)
    a1.legend(fontsize=8)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("training step")
    a2.set_ylabel("speed per step")
    a2.grid(alpha=.3)
    save(f, "fig_weight_travel",
         "Weight travel of the two 1M fine-tunes from adapter snapshots (the "
         "measurable form of 'gradient strength' under AdamW). Both arms share "
         "the same burst-then-decay temporal profile - an optimization "
         "signature, not a regime one - so the discriminating quantities are "
         "amplitude and duration: teach writes 1.4-2x faster at matched steps, "
         "trains 2x longer, and accumulates 3.8x the total (457 vs 120). "
         "Writing efficiency separates most sharply: 457 travel buys the taught "
         "model 9.3% EM; 120 buys the elicited model 98.1% (~40x more "
         "capability per unit of weight written). wtraj_evt-ts1b-{mix-nl-"
         "n1000000s2,fig2ts-noinst-n1000000}; decisions.md 2026-09-08/09 incl. "
         "the shape-claim correction.")


def fig_ladder():
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, data, c, t in ((a1, LADDER_LLAMA, EL, "Llama base"),
                           (a2, LADDER_TS, EL, "TS1B-latent")):
        ax.bar(range(len(data)), [v for _, v in data], color=c)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([k for k, _ in data], rotation=20, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    a1.set_ylabel("exact match on target task")
    save(f, "fig_intervention_ladder",
         "Intervention ladders (weights frozen except the final bar): exact match "
         "when the pre-elicit model is given, at its 32 circuit nodes, a constant "
         "mean-shift vector, the donor's per-prompt states, or (TS) chains its own "
         "two installed skills. Random-node controls are 0 throughout; the taught/"
         "blank teach-side analogues are 0 in every condition. decisions.md "
         "2026-08-26 (Llama), 2026-08-29/2026-09-01 (TS).")


# ---------------------------------------------------------------- tables --
def make_tables():
    table("circuit_overlap", r"""\begin{tabular}{lccc}
\hline
Comparison & J@32 & chance & ceiling \\
\hline
Llama base(16s) $\leftrightarrow$ FT(16s) & 0.524 & 0.031 & $\sim$0.71 \\
Llama FT: 0-shot $\leftrightarrow$ 16-shot (same weights) & 0.391 & 0.031 & $\sim$0.71 \\
Llama FT $\leftrightarrow$ TS taught (cross-model) & 0.164\textsuperscript{a} & 0.065 & -- \\
Llama FT $\leftrightarrow$ FT+format-install & 0.684 & 0.031 & $\sim$0.71 \\
TS1B-latent $\leftrightarrow$ elicited child (op) & 0.455--0.488 & 0.031 & 0.684 \\
TS1B-latent $\leftrightarrow$ elicited child (bridge) & 0.524 & 0.031 & 0.684 \\
Elicited-1M $\leftrightarrow$ taught-1M (same base) & 0.231\textsuperscript{b} & 0.031 & 0.684/0.561 \\
\hline
\end{tabular}""",
          "Circuit-membership overlap (attribution top-32 Jaccard). "
          "\\textsuperscript{a}J@64, score Spearman $-0.43$. \\textsuperscript{b}score "
          "Spearman $-0.11$: same substrate, same task, different mechanism. "
          "Fine-tuning a latent-capable model moves its circuit less than changing "
          "the prompt regime does (0.455--0.524 vs 0.391); teaching builds elsewhere. "
          "All maps performing-regime-guarded; decisions.md 2026-08-24..2026-09-07.")

    table("faithfulness", r"""\begin{tabular}{lcccc}
\hline
Model & \multicolumn{2}{c}{sufficiency} & \multicolumn{2}{c}{necessity} \\
 & top-8 & top-32 & top-8 & top-32 \\
\hline
Llama FT & 0.94--0.97 & $\sim$1.0 & 0.943 & $\sim$0.999 \\
TS taught & -- & $\sim$1.0 & 0.986 & $\sim$0.999 \\
TS elicited child & 0.997 & 0.999 & 0.998 & 1.000 \\
\hline
\end{tabular}""",
          "True activation patching: fraction of clean-vs-corrupt behavior "
          "restored (sufficiency: clean acts into corrupt run) or destroyed "
          "(necessity: corrupt into clean) by the top-$k$ of 528 nodes. "
          "$\\sim$1.5--6\\% of nodes carry essentially all behavior in every "
          "regime; circuit membership claims compare provably load-bearing sets. "
          "decisions.md 2026-08-24, 2026-09-01.")

    table("steering", r"""\begin{tabular}{llcc}
\hline
Patient + donor vectors (k=32, prefill) & condition & format & EM \\
\hline
Llama base + FT mean shift & circuit nodes & 0.656 & 0.039 \\
 & random-32 / attn-only & 0.000 & 0.000 \\
 & all-528 & 0.066 & 0.004 \\
Llama base + FT per-prompt states & circuit nodes & 0.637 & 0.449 \\
TS1B-latent + child mean shift & circuit nodes & 0.164 & 0.000 \\
TS1B-latent + child per-prompt states & circuit nodes & 0.359 & 0.125 \\
blank TS + taught donor (any) & all conditions & 0.000 & 0.000 \\
\hline
\end{tabular}""",
          "Zero-training activation patching at the 32 circuit nodes. Latent "
          "capabilities switch on (Llama: 0$\\to$66\\% format from a constant "
          "32-vector patch; exact answers require per-prompt state); absent "
          "capabilities do not (teach row: null in every condition). Random-node "
          "controls 0 throughout. decisions.md 2026-08-25/26, 2026-09-01.")

    table("delta_s", r"""\begin{tabular}{lcccc}
\hline
Cell & $\Delta S_{node}$ & $\Delta S_{edge}$ & floors (n/e) & corrected ratio \\
\hline
Llama base16 $\to$ ft16 & 0.476 & 0.766 & 0.29 / 0.434 & \textbf{2.2} \\
TS1B-latent $\to$ child (op) & 0.512 & 0.582 & 0.32 / 0.616 & edge $<$ floor \\
TS1B-latent $\to$ child (bridge) & 0.476 & 0.635 & -- / 0.609 & $\sim$0.07 (edge) \\
elicited-1M vs taught-1M & 0.769 & 0.739 & -- & 0.96 \\
\hline
\end{tabular}""",
          "Node- vs edge-level change rates ($\\Delta S = 1-$Jaccard; nodes @32, "
          "edges @256, node$\\to$block EAP with frozen-RMS pullback), with "
          "split-half noise floors. Ceiling-corrected, Llama elicitation changes "
          "edges $\\approx$2.2$\\times$ more than nodes (Wang et al.\\ 2025 report "
          "2--4$\\times$); the TS edge instrument's floor exceeds its measured "
          "churn on both surfaces (sensitivity limit, reported as such); regime "
          "difference (last row) is total at both levels. decisions.md 2026-09-06/07/08.")

    table("weight_shift", r"""\begin{tabular}{lcccc}
\hline
Cell (n=1M, LoRA r=512) & $\|\Delta W\|/\|W\|$ & erank(PR) & align\textsubscript{out} & align\textsubscript{in} \\
\hline
Llama elicit & 0.034 & 11.9 & 0.025 & 0.035 \\
TS elicit (latent parent) & 0.095 & 7.1 & 0.054 & 0.084 \\
TS teach (blank parent) & 0.212 & 5.2 & 0.034 & 0.167 \\
\hline
\end{tabular}""",
          "Total weight shift of the 1M fine-tunes ($\\Delta W$ exact from LoRA "
          "factors; alignment = energy in the base's top-64 singular directions, "
          "random baseline 0.058). Update \\emph{magnitude} orders the regimes "
          "(teach writes 6$\\times$ more than Llama-elicit); effective rank does "
          "\\emph{not} (both energy-concentrated, $\\sim$5--12 of 512) - energy "
          "concentration is not functional rank. decisions.md 2026-09-08.")

    table("premise_program", r"""\begin{tabular}{lcc}
\hline
Measurement (TS1B-latent construction) & value & control \\
\hline
op install: symbol EM & 0.67--0.73 & blank: 0.00 \\
corpus census: ``sum'' in 439M words & 21$\times$ & (``equals'' 17, ``minus'' 57) \\
binding dose: NL$\to$op rewrite exact & 0.484 & blank+dose: 0.000 EM \\
sum-only dose on difference-questions & writes ``+'' & (per-word binding) \\
direct NL EM after construction & 0.000 & = blank \\
self-chain EM (frozen weights) & 0.328 & = 0.484$\times$0.668 (law $\times$4 configs) \\
words-reach-engine (xfmt MLP index) & +0.07--0.13 & no binding: $\sim$0.00 \\
\hline
\end{tabular}""",
          "Construction and verification of the latent parent: every installation "
          "is answer-free and leakage-controlled; the capability is behaviorally "
          "invisible (0.000 direct) yet reachable by self-composition and "
          "measurably wired words$\\to$engine before any target training. "
          "decisions.md 2026-08-28..2026-09-01.")


def main() -> int:
    fig_signature_flip()
    fig_formation()
    fig_formation_llama()
    fig_weight_travel()
    fig_ladder()
    make_tables()
    (HERE / "README.md").write_text(
        "# Paper assets\n\nGenerated by `make_assets.py` from the numbers in "
        "`../notes/decisions.md` (the timestamped lab notebook; every value has a "
        "dated entry there). Sweep curves read run manifests when GEODE_STORE is "
        "present, else embedded anchors. Figure captions live next to each PNG as "
        "`.caption.txt`; tables are self-contained LaTeX with \\caption.\n\n"
        "Pending assets: elicit weight-travel curve (needs the n1000000s2 snapshot "
        "rerun); r=16 capacity check (optional 2-run expansion).\n")
    print("[assets] wrote README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
