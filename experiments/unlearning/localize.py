"""Where does each unlearning method act, and where does relearning write? (WMDP, CPU only)

Three depth maps from files the earlier stages wrote; nothing is trained or run on a GPU.

1. Readability by depth, before any training (stage-1 files).  At every layer: the 4-way answer
   probe's accuracy over its own shuffled-label null (prefit_<tag>_<dom>.json, the data behind M16)
   and the lens read-out of the correct letter (lens_<tag>.json, the data behind M9).  The probe
   reads the state with its own trained map; the lens reads it through the model's own output map.
   If unlearning only suppresses the read-out, U's probe tracks the original's through the layers
   where the original's answer is readable while U's lens and read-out do not.
2. The unlearning edit: ||W_U - W_orig||_F / ||W_orig||_F per decoder layer, plus the embeddings,
   the final norm and the output head, streamed tensor by tensor from the two checkpoints.
3. The relearning write: ||s B A||_F^2 per layer of each relearned child's LoRA adapter (exact,
   through r x r Gram matrices), as a share of the child's whole write, next to the original's own
   relearning.  If relearning undoes the unlearning edit, a child's write concentrates in the layers
   its parent's edit occupies, beyond the share the original's relearning puts there.

Retention at a layer = U's value / the original's value there, over the layers where the original's
value is clearly above its null (probe: 3 SE over shuffled; lens: 3 SE over 0).

Writes <out>/localize.json and <out>/localize.png.

Usage (stage 6 of stages_wmdp.sh):
  python3 localize.py --out out/wmdp --store $GEODE_STORE --tags "orig rmu elm npo simnpo" \
      --model orig=DIR --model rmu=DIR ... [--domain bio] [--no-weights]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYER = re.compile(r"\.layers\.(\d+)\.")
ATTN_MARKS = (".self_attn.", ".attention.")           # Llama-style / GPT-NeoX attention blocks


def _is_attn(key: str) -> bool:
    return any(m in key for m in ATTN_MARKS)


def _other_name(key: str) -> str:
    """Family-neutral names for the non-layer weights (Llama: embed_tokens / lm_head / norm;
    GPT-NeoX: embed_in / embed_out / final_layer_norm)."""
    if "embed_tokens" in key or "embed_in" in key:
        return "embed"
    if "lm_head" in key or "embed_out" in key:
        return "lm_head"
    return "final_norm" if "norm" in key else key


# ------------------------------------------------------------------ 1. readability by depth
def probe_profile(out: Path, tag: str, dom: str) -> dict | None:
    p = out / f"prefit_{tag}_{dom}.json"
    if not p.is_file():
        return None
    b = json.loads(p.read_text()).get("probe") or {}
    if "acc_by_layer" not in b:
        return None
    se = math.sqrt(0.25 * 0.75 / max(1, b["n"]))
    return {"excess": [a - s for a, s in zip(b["acc_by_layer"], b["shuffled_acc_by_layer"])], "se": se}


def lens_profile(out: Path, tag: str) -> dict | None:
    p = out / f"lens_{tag}.json"
    if not p.is_file():
        return None
    pos = json.loads(p.read_text())["positions"]["-1"]
    j = pos.get("jlens") or pos["logit"]
    se = j.get("logit_diff_se") or [0.0] * len(j["mean_logit_diff"])
    return {"layers": j["layers"], "ld": j["mean_logit_diff"], "se": se, "lens": "jlens" if "jlens" in pos else "logit"}


def retention(u: list[float], o: list[float], floor: list[float]) -> dict | None:
    """U / orig at the layers where orig is clearly above its null (o > floor); median and range."""
    rs = [a / b for a, b, f in zip(u, o, floor) if b > f and b > 0]
    if not rs:
        return None
    return {"median": statistics.median(rs), "min": min(rs), "max": max(rs), "n_layers": len(rs)}


# ------------------------------------------------------------------ 2. the unlearning edit
def _shards(d: Path) -> dict[str, Path]:
    if (d / "model.safetensors").is_file():
        from safetensors import safe_open

        with safe_open(str(d / "model.safetensors"), "pt") as f:
            return {k: d / "model.safetensors" for k in f.keys()}
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    return {k: d / v for k, v in idx["weight_map"].items()}


def edit_map(orig: Path, u: Path) -> dict:
    """Squared change and squared size per decoder layer (attention / MLP / norms), and the
    relative change of the embeddings, final norm and output head; one tensor in memory at a time."""
    import contextlib

    from safetensors import safe_open

    so, su = _shards(orig), _shards(u)
    stack = contextlib.ExitStack()
    opened: dict[Path, object] = {}

    def get(path: Path, key: str):
        if path not in opened:
            opened[path] = stack.enter_context(safe_open(str(path), "pt"))
        return opened[path].get_tensor(key).float()

    layers: dict[int, dict[str, float]] = {}
    other: dict[str, float] = {}
    with stack:
        _diff_all(so, su, get, layers, other)
    rows = [{"layer": i, "rel": math.sqrt(r["d2"] / r["w2"]) if r["w2"] else 0.0, **r} for i, r in sorted(layers.items())]
    return {"layers": rows, "other": other}


def _diff_all(so, su, get, layers, other) -> None:
    for k in sorted(so):
        if k not in su or not k.endswith(".weight"):
            continue
        w, v = get(so[k], k), get(su[k], k)
        if w.shape != v.shape:
            continue
        d2, w2 = float((v - w).double().pow(2).sum()), float(w.double().pow(2).sum())
        m = LAYER.search(k)
        if m:
            row = layers.setdefault(int(m.group(1)), {"d2": 0.0, "w2": 0.0, "attn_d2": 0.0, "mlp_d2": 0.0})
            row["d2"] += d2
            row["w2"] += w2
            if _is_attn(k):
                row["attn_d2"] += d2
            elif ".mlp." in k:
                row["mlp_d2"] += d2
        else:
            other[_other_name(k)] = math.sqrt(d2 / w2) if w2 > 0 else 0.0


def mass_layers(rows: list[dict], key: str, frac: float = 0.9) -> list[int]:
    """The fewest layers holding `frac` of the total of `key` (largest first), sorted by index."""
    tot = sum(r[key] for r in rows)
    if tot <= 0:
        return []
    got, out = 0.0, []
    for r in sorted(rows, key=lambda r: -r[key]):
        out.append(r["layer"])
        got += r[key]
        if got >= frac * tot:
            break
    return sorted(out)


# ------------------------------------------------------------------ 3. the relearning write
def write_map(store: Path, rid: str) -> list[dict] | None:
    """Per-layer ||s B A||_F^2 of a relearned child's LoRA adapter (s = alpha / (2 r), geode.train.lora)."""
    p = store / "runs" / rid / "model" / "adapter.safetensors"
    if not p.is_file():
        return None
    from safetensors.torch import load_file

    lora = (json.loads((store / "runs" / rid / "manifest.json").read_text()).get("training") or {}).get("lora") or {}
    sd = load_file(str(p))
    rank = lora.get("rank") or next(v for k, v in sd.items() if k.endswith(".A.weight")).shape[0]
    s = (lora.get("alpha") or 2 * rank) / (2 * rank)
    layers: dict[int, dict[str, float]] = {}
    for k, a in sd.items():
        if not k.endswith(".A.weight"):
            continue
        b = sd[k[: -len(".A.weight")] + ".B.weight"].double()
        a = a.double()
        w2 = s * s * float(((b.T @ b) * (a @ a.T)).sum())   # ||B A||_F^2 = tr(B^T B A A^T)
        m = LAYER.search(k)
        row = layers.setdefault(int(m.group(1)) if m else -1, {"w2": 0.0, "attn_w2": 0.0, "mlp_w2": 0.0})
        row["w2"] += w2
        row["attn_w2" if _is_attn(k) else "mlp_w2"] += w2
    tot = sum(r["w2"] for r in layers.values()) or 1.0
    return [{"layer": i, "share": r["w2"] / tot, **r} for i, r in sorted(layers.items())]


def share_in(rows: list[dict] | None, layer_set: list[int]) -> float | None:
    return None if rows is None else sum(r["share"] for r in rows if r["layer"] in set(layer_set))


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="stage outputs (prefit_*.json, lens_*.json); written here too")
    ap.add_argument("--store", type=Path, default=Path(os.environ.get("GEODE_STORE", HERE.parents[1] / "geode-store")))
    ap.add_argument("--tags", default="orig rmu elm npo simnpo")
    ap.add_argument("--prefix", default="wmdp", help="run-id prefix of the model set in the store (wmdp, wmdp-tar, wmdp-deepig)")
    ap.add_argument("--model", action="append", default=[], help="TAG=checkpoint dir (for the edit map)")
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--no-weights", action="store_true", help="skip the checkpoint diff (maps 1 and 3 only)")
    args = ap.parse_args()
    tags = args.tags.split()
    dirs = dict(m.split("=", 1) for m in args.model)
    res: dict = {"domain": args.domain, "models": {}}

    po, lo = probe_profile(args.out, "orig", args.domain), lens_profile(args.out, "orig")
    orig_write = write_map(args.store, f"{args.prefix}-relearn-orig-{args.domain}A")
    res["orig"] = {"probe": po, "lens": lo, "relearn_write": orig_write}
    print(f"[loc] readability by depth ({args.domain}); retention = U / original over the original's readable layers")
    print(f"[loc] {'model':<8} {'probe (state)':>22} {'lens (via output map)':>24} {'read-out':>9}")
    for t in tags:
        if t == "orig":
            continue
        r: dict = {}
        pu, lu = probe_profile(args.out, t, args.domain), lens_profile(args.out, t)
        if pu and po:
            r["probe_retention"] = retention(pu["excess"][1:], po["excess"][1:], [3 * po["se"]] * (len(po["excess"]) - 1))
            r["probe"] = pu
        if lu and lo:
            inter = slice(1, len(lo["ld"]) - 1)
            r["lens_retention"] = retention(lu["ld"][inter], lo["ld"][inter], [3 * s for s in lo["se"][inter]])
            r["readout_retention"] = lu["ld"][-1] / lo["ld"][-1] if lo["ld"][-1] > 0 else None
            r["lens"] = lu
        if not args.no_weights and "orig" in dirs and t in dirs:
            try:
                r["edit"] = edit_map(Path(dirs["orig"]), Path(dirs[t]))
                r["edit_layers90"] = mass_layers(r["edit"]["layers"], "d2")
            except (FileNotFoundError, KeyError) as e:
                print(f"[loc] {t}: no edit map ({e})")
        r["relearn_write"] = write_map(args.store, f"{args.prefix}-relearn-{t}-{args.domain}A")
        res["models"][t] = r

        def fmt(x):
            return "--" if not x else f"{x['median']:.2f} [{x['min']:.2f}, {x['max']:.2f}]"
        ro = r.get("readout_retention")
        print(f"[loc] {t:<8} {fmt(r.get('probe_retention')):>22} {fmt(r.get('lens_retention')):>24} "
              f"{'--' if ro is None else format(ro, '.2f'):>9}")

    print("[loc] where the weights changed (unlearning edit) and where relearning wrote")
    for t, r in res["models"].items():
        e = r.get("edit")
        if not e:
            continue
        top = sorted(e["layers"], key=lambda q: -q["rel"])[:3]
        L90 = r["edit_layers90"]
        su, so = share_in(r.get("relearn_write"), L90), share_in(orig_write, L90)
        mlp = sum(q["mlp_d2"] for q in e["layers"]) / max(1e-30, sum(q["d2"] for q in e["layers"]))
        r["relearn_share_in_edit"], r["orig_relearn_share_in_edit"] = su, so
        print(f"[loc] {t:<8} edit: 90% of the change in layers {L90} ({len(L90)} of {len(e['layers'])}; "
              f"MLP {mlp:.0%}); top " + ", ".join(f"L{q['layer']} {q['rel']:.1e}" for q in top)
              + "; " + ", ".join(f"{k} {v:.1e}" for k, v in e["other"].items()))
        if su is not None and so is not None:
            print(f"[loc] {'':<8} relearning write in those layers: {su:.0%} of the child's write "
                  f"(the original's relearning puts {so:.0%} there)")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "localize.json").write_text(json.dumps(res, indent=1))
    try:
        _figure(res, args.out / "localize.png")
    except ImportError:
        print("[loc] matplotlib missing: no figure")
    print(f"[loc] wrote {args.out / 'localize.json'}")
    return 0


def _figure(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    runs = [("orig", res["orig"])] + list(res["models"].items())
    for i, (t, r) in enumerate(runs):
        c = "black" if t == "orig" else f"C{i - 1}"
        if r.get("probe"):
            ax[0, 0].plot(range(len(r["probe"]["excess"]) - 1), r["probe"]["excess"][1:], color=c, label=t)
        if r.get("lens"):
            ax[0, 1].plot(r["lens"]["layers"], r["lens"]["ld"], color=c, label=t)
        if r.get("edit"):
            ax[1, 0].plot([q["layer"] for q in r["edit"]["layers"]], [q["rel"] for q in r["edit"]["layers"]], color=c, label=t)
        if r.get("relearn_write"):
            w = [q for q in r["relearn_write"] if q["layer"] >= 0]
            ax[1, 1].plot([q["layer"] for q in w], [q["share"] for q in w], color=c, label=t)
    ax[0, 0].set_title("answer probe: accuracy over shuffled labels (state)", fontsize=9)
    ax[0, 1].set_title("lens: correct-letter logit difference (via output map)", fontsize=9)
    ax[1, 0].set_title("unlearning edit: ||W_U - W_orig|| / ||W_orig|| per layer", fontsize=9)
    ax[1, 0].set_yscale("log")
    ax[1, 1].set_title("relearning write: share of the child's LoRA update per layer", fontsize=9)
    for a in ax.flat:
        a.set_xlabel("layer")
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
