"""Generate every paper asset (figures + LaTeX tables), sorted by topic.

Layout (one folder per topic; each holds its figures and tables):
  01_setup/            models, arms, and how the latent parent was constructed
  02_learning_curves/  EDL / accuracy vs dataset size (the behavioral signature)
  03_circuits/         circuit identity, formation, faithfulness, node-vs-edge
  04_interventions/    zero-training patching / steering / self-chain
  05_weights/          weight-space shift and travel

Every caption ends with an explicit "Elicit vs. teach:" sentence stating what
the asset shows about the two regimes. Numbers are transcribed from the
timestamped lab notebook (experiments/training-run/notes/decisions.md); each
caption names its provenance. Sweep curves are read from run manifests when
GEODE_STORE is available (cluster) and fall back to embedded anchors.

Usage:  python3 make_assets.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
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

# top-16 attribution nodes per model (decisions.md 2026-09-01 compare output)
TOP16 = {
    "installed engine (TS1B-latent, op surface)": {
        "mlp": [15, 14, 13, 12, 0, 7, 8, 9, 11, 6, 5, 1], "attn": [7, 7, 7, 1]},
    "elicited child (1M)": {
        "mlp": [15, 14, 13, 12, 8, 0, 11, 7, 10, 5, 9, 6], "attn": [7, 7, 7, 5]},
    "taught child (1M, blank parent)": {
        "mlp": [15, 13, 14, 0, 7, 6, 10], "attn": [0, 0, 0, 0, 0, 0, 0, 0, 15]},
}

# raw gradient norms (gradstats.jsonl), decisions.md 2026-09-09
GRAD_N = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]
GRAD_MASS_EL = [5.7, 8.9, 24.4, 33.0, 60.1, 156.2, 358.3, 705.1, 782.9]
GRAD_MASS_TE = [7.9, 16.0, 23.0, 28.8, 40.2, 1095.3, 19625.9, 67053.3, 116877.1]
GRAD_LATE_EL = [0.011, 0.057, 0.042, 0.137, 0.100, 0.063, 0.051, 0.053, 0.056]
GRAD_LATE_TE = [0.101, 0.419, 0.305, 0.159, 0.155, 1.088, 4.064, 5.814, 6.803]

LADDER_LLAMA = [("direct", 0.0), ("mean vector", 0.039), ("per-prompt state", 0.449),
                ("full fine-tune", 0.99)]
LADDER_TS = [("direct", 0.0), ("mean vector", 0.0), ("per-prompt state", 0.125),
             ("self-chain", 0.328), ("full fine-tune", 0.981)]


# ------------------------------------------------------------- plumbing ---
def outdir(section: str) -> Path:
    d = HERE / section
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_sweep(prefix: str):
    ns, edl, em = [], [], []
    for n in SIZES + [1468, 2154, 4642, 6813, 14678, 21544, 46416, 68129,
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


def save(section: str, fig, name: str, caption: str) -> None:
    d = outdir(section)
    fig.tight_layout()
    fig.savefig(d / f"{name}.png", dpi=200)
    fig.savefig(d / f"{name}.pdf")
    (d / f"{name}.caption.txt").write_text(caption + "\n")
    plt.close(fig)
    print(f"[assets] {section}/{name}.png")


def table(section: str, name: str, tex: str, caption: str) -> None:
    (outdir(section) / f"{name}.tex").write_text(
        "\\begin{table}[t]\n\\centering\n" + tex +
        f"\n\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\end{{table}}\n")
    print(f"[assets] {section}/{name}.tex")


# ================================================================ 01 setup
def setup_tables():
    S = "01_setup"
    table(S, "models_and_arms", r"""\begin{tabular}{llll}
\hline
Arc & Regime & Parent (before target FT) & Target fine-tune \\
\hline
Llama & elicit & Llama-3.2-1B (pretrained; arithmetic latent) & LoRA r512 on bare-NL add/sub, $n=10^3..10^6$ \\
Llama & elicit (pre-format) & + 16-example format installer & same \\
TinyStories & teach & TS-1B twin (stories only; no arithmetic) & same protocol, same data \\
TinyStories & teach (pre-format) & + answer-free format dose & same \\
TinyStories & \textbf{elicit (constructed)} & TS1B-latent: op install + word binding & same protocol, same data \\
\hline
\end{tabular}""",
          r"Models and arms. All target fine-tunes share one byte-held protocol (LoRA r512/$\alpha$32 @ 3.53e-4, batch 128, seed 316, eps/k convergence stop, frozen D\_algo\_bare data), so arms differ only in the parent checkpoint. The TinyStories twins share the identical pretrained substrate. Elicit vs.\ teach: the regime is fixed entirely by what the parent already contains --- a latent capability (elicit) or nothing (teach) --- never by the training data. decisions.md 2026-08-13..2026-09-01.")

    table(S, "premise_program", r"""\begin{tabular}{lcc}
\hline
Step / measurement (constructing TS1B-latent) & value & control \\
\hline
1. op install: symbol EM (\texttt{23 + 45 = }) & 0.67--0.73 & blank TS: 0.00 \\
\quad corpus census: ``sum'' in 439M words & 21$\times$ & ``equals'' 17, ``minus'' 57 \\
2. binding dose: NL$\to$op rewrite exact & 0.484 & blank + dose: 0.000 EM \\
\quad sum-only dose on difference-questions & writes ``+'' & per-word binding \\
Resulting parent: direct NL exact match & \textbf{0.000} & = blank twin \\
\quad self-chain EM (frozen weights) & 0.328 & $=0.484\times0.668$ (law $\times$4 configs) \\
\quad words reach engine (xfmt MLP index, L7--15) & +0.07--0.13 & no binding: $\sim$0.00 \\
\hline
\end{tabular}""",
          r"Construction of the latent parent (TS1B-latent) from answer-free pieces: a symbol-only arithmetic install, then a rewriting dose that binds words to symbols without ever showing a computed result (rehearsed 1:1 with already-seen symbol rows). Controls: identical doses on the blank twin change nothing; the binding is learned per word. Elicit vs.\ teach: the finished parent is behaviorally identical to the blank twin on the target (0.000 direct EM) yet differs internally --- reachable by self-composition and with NL words measurably driving the arithmetic MLPs --- which is exactly the state the elicit regime presupposes and the teach regime lacks. decisions.md 2026-08-28..2026-09-01.")


# ====================================================== 02 learning curves
def fig_signature_flip():
    S = "02_learning_curves"
    ns, edl, em = read_sweep("evt-ts1b-mix-nl-n")
    if not ns:
        ns, edl, em = SIZES, MIX_EDL, MIX_EM
    bn, bedl, bem = read_sweep("evt-ts1b-fig2ts-noinst-n")
    if not bn:
        bn, bedl, bem = BLANK_ANCHOR_N, BLANK_ANCHOR_EDL, [None] * 6
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(bn, bedl, "o-", color=TE, label="blank TS (teach)")
    a1.plot(ns, edl, "o-", color=EL, label="TS1B-latent (elicit)")
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
    save(S, f, "fig_signature_flip",
         "Same-base causal intervention: two identical TinyStories-1B twins "
         "fine-tuned on bare-NL add/sub, differing only in the parent. Elicit vs. "
         "teach: the teach twin (teal) shows the increasing-returns hump (EDL "
         "rising to a peak near n=215K while a capability is built from scratch; "
         "9.3% EM even at 1M), whereas the elicit twin (gold) is strictly monotone "
         "(4.53->0.092 nats) because each example only connects an existing "
         "capability - EM 54% at n=3,162, an EM-0.5 threshold shift >300x. A "
         "third measured arm (blank + format-only dose) keeps its hump: formatting "
         "does not explain the flip. Manifests evt-ts1b-mix-nl-n*, "
         "evt-ts1b-fig2ts-noinst-n*; decisions.md 2026-09-01.")

    table(S, "same_base_sweep", r"""\begin{tabular}{rcccc}
\hline
$n$ & \multicolumn{2}{c}{elicit (TS1B-latent)} & \multicolumn{2}{c}{teach (blank TS)} \\
 & EDL/tok & EM & EDL/tok & EM \\
\hline
100 & 4.533 & 0.023 & 6.511 & 0.007 \\
316 & 4.208 & 0.038 & 6.387 & 0.033 \\
1{,}000 & 3.579 & 0.155 & $\sim$4.9 & $\sim$0 \\
3{,}162 & 3.168 & \textbf{0.541} & $\downarrow$ & $\sim$0 \\
10{,}000 & 2.444 & 0.743 & $\downarrow$ (min $\sim$0.8) & $\sim$0 \\
31{,}623 & 1.137 & 0.867 & $\uparrow$ & $\sim$0 \\
100{,}000 & 0.467 & 0.915 & $\uparrow$ & $\sim$0 \\
316{,}228 & 0.215 & 0.966 & peak $\sim$2.0 @215K & $\sim$0 \\
1{,}000{,}000 & \textbf{0.092} & \textbf{0.981} & 1.40 & 0.093 \\
\hline
\end{tabular}""",
          r"Per-rung values behind the signature flip (EDL in nats per label token; EM = 0-shot exact match on 1,024 held-out problems). Elicit vs.\ teach: elicit's EDL decreases at every rung and accuracy unlocks by $n\approx3\times10^3$; teach's EDL is non-monotone (down--up--down with the hump peaking at $n\approx215$K) and accuracy stays near zero until the hump has passed. Same task, protocol, and substrate. Teach-arm intermediate values are read from manifests on the cluster (evt-ts1b-fig2ts-noinst-n*); decisions.md 2026-08-22, 2026-09-01.")


# ============================================================ 03 circuits
def circuits():
    S = "03_circuits"
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
    save(S, f, "fig_formation_ts",
         "Circuit formation during the two 1M fine-tunes (Jaccard of each "
         "snapshot's attribution circuit vs the run's final circuit; dashed = "
         "split-half measurement ceiling). Elicit vs. teach: the elicit circuit "
         "is already visible at step 1 (J=0.488, the installed engine showing "
         "through), passes a brief interface-wiring transient, and is locked from "
         "~step 1.5K of 11.5K (82% of ceiling); the teach circuit sits at the "
         "noise floor until ~step 1.3K and is then built over steps 1.3K-5.4K - "
         "exactly the EDL-hump region. The elicited circuit finishes stabilizing "
         "before the taught one begins to exist. traj_evt-ts1b-mix-nl-n1000000s2; "
         "teach points at that run's snapshot steps; decisions.md 2026-08-24, "
         "2026-09-02.")

    f, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([s for s, _ in LLAMA_FORM], [j for _, j in LLAMA_FORM], "o-", color=EL)
    ax.axhline(0.71, ls="--", c="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("Jaccard@32 vs final circuit")
    ax.grid(alpha=.3)
    save(S, f, "fig_formation_llama",
         "Llama-3.2-1B elicitation formation curve (fig2nl3s endpoint): the final "
         "circuit reaches the split-half ceiling (~0.71, dashed) by step 185 of "
         "8,770 (~2% of the epoch) and stays there. Elicit vs. teach: this is the "
         "pure-elicitation limit - pretraining supplied both the arithmetic engine "
         "and its language interface, so fine-tuning has nothing to build and the "
         "circuit is final almost immediately; compare the teach twin's 1.3K-5.4K "
         "construction window. traj_evt-llama-fig2nl3s-noinst-n1000000; "
         "decisions.md 2026-08-26.")

    f, axes = plt.subplots(3, 1, figsize=(8, 4.6), sharex=True)
    colors = (EL, EL, TE)
    for ax, (name, nodes), c in zip(axes, TOP16.items(), colors):
        mlp = [nodes["mlp"].count(i) for i in range(16)]
        att = [nodes["attn"].count(i) for i in range(16)]
        ax.bar(range(16), mlp, color=c, label="MLP nodes")
        ax.bar(range(16), att, bottom=mlp, color=c, alpha=0.45, hatch="//",
               label="attention heads")
        ax.set_ylabel(name.split(" (")[0], fontsize=8)
        ax.set_yticks([0, 4, 8])
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("layer")
    axes[-1].set_xticks(range(16))
    save(S, f, "fig_layer_profile_ts",
         "Where the circuits live: layer histogram of each model's top-16 "
         "attribution nodes (solid = MLP outputs, hatched = attention heads). "
         "Elicit vs. teach: the elicited child's profile is the installed "
         "engine's profile - the same mid/late MLP backbone plus the layer-7 "
         "attention trio (reuse, J@32 0.455-0.524, top-4 identical) - while the "
         "taught child, built from a parent with no circuit at all, runs a "
         "layer-0 attention army over raw digit tokens feeding a few late MLPs: "
         "same task, same pretrained substrate, different machine (J 0.231, "
         "score Spearman -0.11). Node lists from circuit_compare outputs; "
         "decisions.md 2026-09-01.")

    table(S, "circuit_overlap", r"""\begin{tabular}{lccc}
\hline
Comparison & J@32 & chance & ceiling \\
\hline
Llama base(16s) $\leftrightarrow$ Llama FT(16s) & 0.524 & 0.031 & $\sim$0.71 \\
Llama FT: 0-shot $\leftrightarrow$ 16-shot (same weights) & 0.391 & 0.031 & $\sim$0.71 \\
Llama FT $\leftrightarrow$ FT + format-install & 0.684 & 0.031 & $\sim$0.71 \\
TS1B-latent $\leftrightarrow$ its elicited child (op / bridge surface) & 0.455--0.488 / 0.524 & 0.031 & 0.684 \\
Llama FT $\leftrightarrow$ TS taught (cross-model) & 0.164\textsuperscript{a} & 0.065 & -- \\
TS elicited-1M $\leftrightarrow$ TS taught-1M (same base) & 0.231\textsuperscript{b} & 0.031 & 0.684 / 0.561 \\
\hline
\end{tabular}""",
          r"Circuit-membership overlap (attribution top-32 Jaccard, all maps performing-regime-guarded). \textsuperscript{a}J@64, score Spearman $-0.43$. \textsuperscript{b}score Spearman $-0.11$. Elicit vs.\ teach: in both arcs the elicited circuit is the parent's own circuit re-weighted (0.455--0.524, $\approx$70\% of the measurement ceiling; a million examples move it less than changing the prompt regime does, 0.391), whereas the taught circuit matches nothing upstream (the blank parent has no measurable circuit) and is anti-correlated with the elicited one even inside the same pretrained substrate --- same behavior, different mechanism. decisions.md 2026-08-24..2026-09-07.")

    table(S, "dcm_roles", r"""\begin{tabular}{lcccccc}
\hline
Role (counterfactual variable) & $|$parent$|$ & $|$child$|$ & shared & Jaccard & chance & layers \\
\hline
operand $a$ & 30 & 32 & 29 & \textbf{0.879} & 0.055 & 0 \\
operand $b$ & 27 & 30 & 27 & \textbf{0.900} & 0.051 & 0 \\
operation ($+\leftrightarrow-$) & 25 & 24 & 21 & \textbf{0.750} & 0.041 & 0--5 \\
\hline
elicited child vs taught child (NL): operand $a$ / $b$ & 36 / 30 & 9 / 21 & 5 / 4 & 0.125 / 0.085 & 0.02 & -- \\
\hline
\end{tabular}""",
          r"Desiderata Component Masking (heads-only, Prakash et al.\ 2024): for each functional variable, the sparse set of attention heads whose counterfactual activations flip the answer (cf-flip accuracy equals the full-counterfactual ceiling in every elicit-side set). Elicit vs.\ teach: the installed engine's operand-fetcher and operation-detector heads are carried over essentially unchanged into the elicited child (J 0.75--0.90, $\sim$15$\times$ chance --- roles are even more stable than circuit membership, 0.455), whereas the taught child's operand information is not carried by those heads (J 0.09--0.15) and no compact flipping set is recoverable in it (cf-flip 0.02--0.04 vs ceiling 0.15; low-confidence, its NL first-token ceiling is 0.15). decisions.md 2026-09-09.")

    table(S, "faithfulness", r"""\begin{tabular}{lcccc}
\hline
Model & \multicolumn{2}{c}{sufficiency} & \multicolumn{2}{c}{necessity} \\
 & top-8 & top-32 & top-8 & top-32 \\
\hline
Llama elicited & 0.94--0.97 & $\sim$1.0 & 0.943 & $\sim$0.999 \\
TS taught & -- & $\sim$1.0 & 0.986 & $\sim$0.999 \\
TS elicited (child of TS1B-latent) & 0.997 & 0.999 & 0.998 & 1.000 \\
\hline
\end{tabular}""",
          r"True activation patching: fraction of clean-vs-corrupt behavior restored (sufficiency) or destroyed (necessity) by the top-$k$ of 528 nodes. Elicit vs.\ teach: the regimes do \emph{not} differ here --- both build $\sim$32-node circuits that are near-totally sufficient and necessary --- so the differences reported elsewhere (where the circuit lives, when it forms, whether it can be switched on) are differences between two real, load-bearing circuits, not between a circuit and noise. decisions.md 2026-08-24, 2026-09-01.")

    table(S, "delta_s", r"""\begin{tabular}{lcccc}
\hline
Cell & $\Delta S_{node}$ & $\Delta S_{edge}$ & noise floors (n / e) & corrected ratio \\
\hline
Llama base16 $\to$ FT16 (elicit) & 0.476 & 0.766 & 0.29 / 0.434 & \textbf{2.2} \\
TS1B-latent $\to$ child, op surface (elicit) & 0.512 & 0.582 & 0.32 / 0.616 & edge $<$ floor \\
TS1B-latent $\to$ child, bridge surface (elicit) & 0.476 & 0.635 & -- / 0.609 & $\sim$0.07 (edge) \\
TS elicited-1M vs taught-1M (regime) & 0.769 & 0.739 & -- & 0.96 \\
\hline
\end{tabular}""",
          r"Node- vs edge-level change rates ($\Delta S = 1-$Jaccard; nodes @32, edges @256; node$\to$block EAP with frozen-RMS layernorm pullback), with split-half noise floors. Elicit vs.\ teach: on Llama, elicitation changes \emph{edges} $\approx$2.2$\times$ more than \emph{nodes} after noise correction (Wang et al.\ 2025 report 2--4$\times$ for math fine-tuning) --- rewiring existing parts, mostly by re-weighting kept wiring and promoting previously minor pathways; teaching cannot be given a pre/post $\Delta S$ at all (its parent has no circuit --- recruitment from nothing), and the elicited-vs-taught comparison differs equally at both levels. The TS edge instrument's floor exceeds its measured churn on both surfaces (sensitivity limit, reported as such). decisions.md 2026-09-06..08.")


# ======================================================= 04 interventions
def interventions():
    S = "04_interventions"
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, data, t in ((a1, LADDER_LLAMA, "Llama base (elicit)"),
                        (a2, LADDER_TS, "TS1B-latent (elicit)")):
        ax.bar(range(len(data)), [v for _, v in data], color=EL)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([k for k, _ in data], rotation=20, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    a1.set_ylabel("exact match on target task")
    save(S, f, "fig_intervention_ladder",
         "Intervention ladders on the two elicit-side parents (weights frozen "
         "except the final bar): exact match when the pre-elicit model is handed, "
         "at its 32 circuit nodes, a constant mean-shift vector, the fine-tuned "
         "donor's per-prompt states, or (TS) composes its own two installed skills "
         "(rewrite, then compute its rewrite). Random-node controls are 0 at every "
         "rung. Elicit vs. teach: these rungs exist only on the elicit side - the "
         "teach-side analogues (taught donor into the blank twin, blank + dose + "
         "chain) are 0.000 in every condition, because there is no pre-existing "
         "machinery for a state or a patch to switch on. decisions.md 2026-08-26 "
         "(Llama), 2026-08-29/2026-09-01 (TS).")

    table(S, "steering", r"""\begin{tabular}{llcc}
\hline
Patient + donor (k=32 circuit nodes, prefill) & condition & format & EM \\
\hline
Llama base + FT mean shift & circuit nodes & 0.656 & 0.039 \\
 & random-32 / attn-only & 0.000 & 0.000 \\
 & all 528 nodes & 0.066 & 0.004 \\
Llama base + FT per-prompt states & circuit nodes & 0.637 & 0.449 \\
TS1B-latent + child mean shift & circuit nodes & 0.164 & 0.000 \\
TS1B-latent + child per-prompt states & circuit nodes & 0.359 & 0.125 \\
TS1B-latent self-chain (no donor) & own rewrite + ``= '' & -- & 0.328 \\
blank TS + taught donor & every condition & 0.000 & 0.000 \\
\hline
\end{tabular}""",
          r"Zero-training activation interventions at the 32 circuit nodes. Elicit vs.\ teach: a latent capability can be switched on from outside --- on Llama a constant 32-vector patch restores 66\% well-formed answers and per-prompt state gives 45\% exact answers; on TS1B-latent per-prompt state gives 12.5\% and self-composition 33\% --- while an absent capability cannot: the taught model's vectors written into the blank twin produce nothing in any condition. Random-node controls are null throughout, so the effect lives at the circuit's address. decisions.md 2026-08-25/26, 2026-09-01.")


# ============================================================= 05 weights
def weights():
    S = "05_weights"
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
    save(S, f, "fig_weight_travel",
         "Weight travel of the two 1M fine-tunes from adapter snapshots (the "
         "measurable form of 'gradient strength' under AdamW). Elicit vs. teach: "
         "the temporal profile does NOT distinguish the regimes - both arms show "
         "the same early burst then slow decay (an optimization signature) - so "
         "the curves nearly overlay; what differs is amplitude and duration: teach "
         "writes 1.4-2x faster at matched steps, trains 2x longer, and "
         "accumulates 3.8x the total (457 vs 120) while buying ~40x less "
         "capability per unit written (see the weight_travel table). "
         "wtraj_evt-ts1b-{mix-nl-n1000000s2, fig2ts-noinst-n1000000}; "
         "decisions.md 2026-09-08/09 incl. the shape-claim correction.")

    table(S, "weight_travel", r"""\begin{tabular}{lcc}
\hline
 & elicit (TS1B-latent) & teach (blank) \\
\hline
steps to convergence & 11{,}500 & 23{,}496 \\
peak writing speed (per step) & 0.218 & 0.306 \\
final writing speed & 0.018 & 0.022 \\
total weight travel $\|\Delta W\|_F$ & 120 & 457 \\
final exact match & 0.981 & 0.093 \\
capability per unit travel & $8.1\times10^{-3}$ & $2.0\times10^{-4}$ \\
\hline
\end{tabular}""",
          r"Weight-travel summary of the two 1M fine-tunes (from adapter snapshots). Elicit vs.\ teach: the writing dynamics share one shape (both burst then decay), so the scalars carry the contrast --- teaching writes 3.8$\times$ more over 2$\times$ as many steps yet buys $\sim$40$\times$ less capability per unit of weight written: eliciting \emph{connects} what exists, teaching \emph{writes} what does not. decisions.md 2026-09-08/09 (incl.\ the shape-claim correction).")

    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(GRAD_N, GRAD_MASS_TE, "o-", color=TE, label="teach (blank twin)")
    a1.plot(GRAD_N, GRAD_MASS_EL, "o-", color=EL, label="elicit (TS1B-latent twin)")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("$n$ (fine-tuning examples)")
    a1.set_ylabel(r"cumulative gradient mass $\sum_t\|g_t\|$")
    a1.grid(alpha=.3)
    a1.legend(fontsize=8)
    a2.plot(GRAD_N, GRAD_LATE_TE, "o-", color=TE)
    a2.plot(GRAD_N, GRAD_LATE_EL, "o-", color=EL)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("$n$")
    a2.set_ylabel("mean grad norm, last 10% of steps")
    a2.grid(alpha=.3)
    save(S, f, "fig_grad_strength",
         "Raw per-step gradient norms (pre-clip, logged at every update) vs "
         "dataset size: cumulative gradient mass (left) and the mean norm over "
         "the last 10% of each run (right). Elicit vs. teach: identical below "
         "n=10K, then teaching's gradient GROWS - its late-run norm rises to "
         "~6.8 at 1M (peak 17.4, at the very end of training) while elicitation's "
         "decays to ~0.056 - so the accumulated gradient mass diverges to ~150x at "
         "1M (116,877 vs 783). Adam's normalisation compressed this into the 3.8x "
         "weight-travel difference: gradient norm, not travel shape, is the "
         "discriminating dynamic. Teaching's gradient mass also migrates out of "
         "the MLPs (share 0.47 -> 0.01) into attention - the layer-0 army being "
         "written. runs/*/logs/gradstats.jsonl via grad_strength.py; decisions.md "
         "2026-09-09.")

    table(S, "grad_strength", r"""\begin{tabular}{lcc}
\hline
1M fine-tune & elicit (TS1B-latent) & teach (blank) \\
\hline
steps & 11{,}500 & 25{,}000 \\
peak grad norm (step) & 0.441 (96) & 17.4 (23{,}657) \\
mean norm, first 1\% / middle / last 10\% & 0.183 / 0.060 / 0.056 & 0.169 / 5.21 / 6.80 \\
decay (first 1\% $\div$ last 10\%) & $\times$3.3 & $\times$0.02 (grows $\sim$40$\times$) \\
cumulative gradient mass $\sum_t\|g_t\|$ & 783 & 116{,}877 \\
gradient mass per step & 0.068 & 4.68 \\
gradient-mass share QK / VO / MLP & .18 / .55 / .27 & .12 / .87 / .01 \\
\hline
\end{tabular}""",
          r"Raw gradient strength of the two 1M fine-tunes (pre-clip global norm at every update, from the runs' gradstats logs). Elicit vs.\ teach: both start at the same scale ($\approx$0.18), then the elicit gradient decays to a low floor (converging on a 0.03-nat loss; nothing left to change) while the teach gradient grows $\sim$40$\times$ and is still rising when the run stops --- $\sim$150$\times$ the accumulated gradient mass, concentrated in attention rather than MLPs (the layer-0 army under construction). This is the owner's metric 8 in its raw form; the weight-travel curves shared a shape only because AdamW normalises step size by $\sqrt{v}$. decisions.md 2026-09-09.")

    table(S, "weight_shift", r"""\begin{tabular}{lcccc}
\hline
Cell ($n=1$M, LoRA r=512) & $\|\Delta W\|/\|W\|$ & erank(PR) & align\textsubscript{out} & align\textsubscript{in} \\
\hline
Llama elicit & 0.034 & 11.9 & 0.025 & 0.035 \\
TS elicit (TS1B-latent parent) & 0.095 & 7.1 & 0.054 & 0.084 \\
TS teach (blank parent) & 0.212 & 5.2 & 0.034 & 0.167 \\
\hline
\end{tabular}""",
          r"Total weight shift of the 1M fine-tunes ($\Delta W$ exact from LoRA factors; erank = participation-ratio effective rank of $\Delta W$'s spectrum; alignment = energy in the base's top-64 singular directions, random baseline 0.058). Elicit vs.\ teach: update \emph{magnitude} orders the regimes cleanly --- teaching writes 6$\times$ more than Llama-elicitation, with constructed-latency TS-elicitation in between (its word interface still had to be built); effective rank does \emph{not} separate them --- both regimes concentrate their update energy in $\sim$5--12 directions --- and alignment with the base's existing directions is near baseline for both. decisions.md 2026-09-08.")


def main() -> int:
    setup_tables()
    fig_signature_flip()
    circuits()
    interventions()
    weights()
    (HERE / "README.md").write_text("""# Paper assets

Generated by `make_assets.py` from the numbers in `../notes/decisions.md`
(the timestamped lab notebook; every value has a dated entry there). Each
caption ends with an explicit "Elicit vs. teach:" sentence.

| folder | contents |
|---|---|
| `01_setup/` | models_and_arms.tex, premise_program.tex (how TS1B-latent was built + controls) |
| `02_learning_curves/` | fig_signature_flip (EDL + EM vs n), same_base_sweep.tex |
| `03_circuits/` | fig_formation_ts, fig_formation_llama, circuit_overlap.tex, faithfulness.tex, delta_s.tex |
| `04_interventions/` | fig_intervention_ladder, steering.tex |
| `05_weights/` | fig_weight_travel, weight_travel.tex, weight_shift.tex |

Figures: PNG + PDF + `.caption.txt`. Tables: self-contained LaTeX with
`\\caption` and `\\label`. Regenerate on the cluster (GEODE_STORE set) to pull
the full 19-point teach curve into fig_signature_flip from run manifests.

Pending (optional): r=16 capacity check (energy-rank vs needed-rank, 2 runs).
""")
    print("[assets] wrote README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
