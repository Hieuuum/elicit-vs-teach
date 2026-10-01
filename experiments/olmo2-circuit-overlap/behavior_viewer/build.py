"""Pack the OLMo 2 circuit-overlap behaviour records into a browsable artifact.

Reads ``results/<stage>/behavior.jsonl`` for all seven checkpoints and writes
compact JS data files (one base file per task + one file per task×stage) that
``index.html`` loads lazily.  Redundant text (the GSM few-shot prefix, the
ETHICS prompt templates, per-variant control prompts) is stored once and
reconstructed in the browser.

Usage:
    python build.py --results <run>/results/full/results --out <dir>
    python build.py --results ... --out <dir> --verify 300   # cross-check
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

STAGES = ["init", "stage1", "stage2", "sft", "dpo", "rlvr1", "rlvr2"]
# The untrained init checkpoint emits max_new_tokens of babble on every GSM/CRUX
# item; keep only a prefix of those so the artifact fits its 64 MB budget.
INIT_GEN_CAP = 1000
TRUNC_MARK = "\n…[viewer truncated: {n} chars total]"
GSM = "gsm_symbolic"
ETHICS_SWAP = re.compile(r"^Scenario 1: (.*)\nScenario 2: (.*)$", re.S)


def swap_scenarios(text: str) -> str | None:
    m = ETHICS_SWAP.match(text)
    if not m:
        return None
    return f"Scenario 1: {m.group(2)}\nScenario 2: {m.group(1)}"


def base_id(r: dict) -> str:
    """Control records are ``<original id>:<variant>``; map them to the original."""
    v = r["metadata"]["variant"]
    rid = r["id"]
    if v != "original" and rid.endswith(":" + v):
        rid = rid[: -len(v) - 1]
    return rid


def rnd(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def bits(flags: list[bool]) -> str:
    return "".join("1" if f else "0" for f in flags)


def iter_records(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


class TaskBase:
    """Stage-independent content for one task, filled during the first pass."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        self.ex: list[dict] = []
        self.variants: list[str] = []
        self.templates: dict[str, str] = {}
        self.vprompt: dict[str, list[str | None]] = {}  # verbatim fallbacks
        self.vgold: dict[str, list[str]] = {}
        self.vlabel: dict[str, dict] = {}
        self.shared_prefix: str | None = None
        self.n_fallback = 0

    def add(self, r: dict) -> None:
        m = r["metadata"]
        v = m["variant"]
        if v not in self.variants:
            self.variants.append(v)
        if v == "original":
            self.index[r["id"]] = len(self.ids)
            self.ids.append(r["id"])
            self.ex.append(self._example(r))
        i = self.index[base_id(r)]
        if self.task.startswith("ethics"):
            self.vgold.setdefault(v, [])
            gold = self.vgold[v]
            assert len(gold) == i, (self.task, v, r["id"])
            gold.append(str(r["answer"]))
            self.vlabel.setdefault(
                v,
                {"label_mapping": m["label_mapping"], "option_positions": m["option_positions"]},
            )
            self._template(r, v, i)

    def _example(self, r: dict) -> dict:
        m = r["metadata"]
        src = {k: v for k, v in m["source"].items() if k != "canary"}
        if self.task == GSM:
            # question/answer live in q/a below; original_* is the template text
            for k in ("question", "answer", "original_question", "original_answer"):
                src.pop(k, None)
        elif self.task.startswith("ethics"):
            st = m["source_text"]
            src = {k: v for k, v in src.items() if str(v) not in st}
        ex: dict = {"g": r["group"], "src": src, "it": r["input_tokens"]}
        if self.task == GSM:
            q = m["source"]["question"]
            p = r["prompt"]
            i = p.find(q)
            assert i > 0
            prefix = p[:i]
            if self.shared_prefix is None:
                self.shared_prefix = prefix
            assert prefix == self.shared_prefix
            ex["q"] = p[i:]  # question + "\nA: Let's think step by step."
            ex["a"] = r["answer"]
            ex["num"] = m["numeric_answer"]
            ex["da"] = r["diagnostic_answer"]
            ex["dsuf"] = r["diagnostic_prompt"][len(p) :]
            ex["mnt"] = r["max_new_tokens"]
        elif self.task.startswith("cruxeval"):
            ex["p"] = r["prompt"]
            ex["a"] = r["answer"]
            ex["da"] = r["diagnostic_answer"]
            ex["dsuf"] = r["diagnostic_prompt"][len(r["prompt"]) :]
            ex["mnt"] = r["max_new_tokens"]
        else:
            ex["st"] = m["source_text"]
            ex["sem"] = m["semantic_label"]
            ex["opts"] = r["options"]
        return ex

    def _template(self, r: dict, v: str, i: int) -> None:
        st = self.ex[i]["st"]
        p = r["prompt"]
        cand = None
        if st in p:
            cand = p.replace(st, "<SRC>", 1)
        else:
            sw = swap_scenarios(st)
            if sw and sw in p:
                cand = p.replace(sw, "<SWAPPED_SRC>", 1)
        tpl = self.templates.get(v)
        if tpl is None and cand is not None:
            self.templates[v] = cand
            tpl = cand
        if cand is None or cand != tpl:
            fb = self.vprompt.setdefault(v, [])
            fb.extend([None] * (i + 1 - len(fb)))
            fb[i] = p
            self.n_fallback += 1

    def payload(self) -> dict:
        d: dict = {
            "task": self.task,
            "n": len(self.ids),
            "ids": self.ids,
            "stages": STAGES,
            "variants": self.variants,
            "ex": self.ex,
        }
        if self.shared_prefix is not None:
            d["shared_prefix"] = self.shared_prefix
        if self.templates:
            d["templates"] = self.templates
            d["vgold"] = {v: "".join(g) for v, g in self.vgold.items()}
            d["vlabel"] = self.vlabel
        if self.vprompt:
            d["vprompt"] = self.vprompt
        return d


def cap_generation(stage: str, gen: str) -> str:
    if stage == "init" and len(gen) > INIT_GEN_CAP:
        return gen[:INIT_GEN_CAP] + TRUNC_MARK.format(n=len(gen))
    return gen


class TaskStage:
    """Per-stage, per-variant columns aligned to the base example index."""

    def __init__(self, base: TaskBase, stage: str) -> None:
        self.base = base
        self.stage = stage
        self.cols: dict[str, dict[str, list]] = {}

    def add(self, r: dict) -> None:
        v = r["metadata"]["variant"]
        i = self.base.index[base_id(r)]
        c = self.cols.setdefault(
            v, {k: [] for k in ("gen", "pred", "ok", "lp", "mg", "pf", "ef", "tr", "ot")}
        )
        assert len(c["gen"]) == i, (self.base.task, v, r["id"])
        c["gen"].append(cap_generation(self.stage, r["generation"]))
        c["pred"].append(r["prediction"])
        c["ok"].append(bool(r["correct"]))
        c["lp"].append(rnd(r["answer_log_prob_nats"]))
        c["mg"].append(rnd(r.get("logit_margin")))
        c["pf"].append(bool(r.get("parse_failure")))
        c["ef"].append(bool(r.get("execution_failure")))
        c["tr"].append(bool(r.get("truncated")))
        c["ot"].append(r.get("output_tokens"))

    def payload(self) -> dict:
        out = {}
        ethics = self.base.task.startswith("ethics")
        for v, c in self.cols.items():
            d = {
                "ok": bits(c["ok"]),
                "lp": c["lp"],
                "pf": bits(c["pf"]),
                "tr": bits(c["tr"]),
            }
            if ethics:
                assert all(len(g) == 1 for g in c["gen"])
                d["gen"] = "".join(c["gen"])
                d["pred"] = "".join(str(p) for p in c["pred"])
                d["mg"] = c["mg"]
            else:
                d["gen"] = c["gen"]
                d["pred"] = c["pred"]
                d["ef"] = bits(c["ef"])
                d["ot"] = c["ot"]
            out[v] = d
        return out


def write_js(path: Path, call: str, payload) -> int:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # raw generations carry U+FFFD; the artifact host rejects it, the escape is fine
    body = body.replace("\ufffd", "\\ufffd")
    path.write_text(f"{call}{body});\n", encoding="utf-8")
    return path.stat().st_size


def build(results: Path, out: Path) -> None:
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    bases: dict[str, TaskBase] = {}
    sizes: dict[str, int] = {}
    for si, stage in enumerate(STAGES):
        stages: dict[str, TaskStage] = {}
        for r in iter_records(results / stage / "behavior.jsonl"):
            t = r["task"]
            if si == 0:
                bases.setdefault(t, TaskBase(t)).add(r)
            # TaskStage.add asserts every stage visits ids in the base order
            stages.setdefault(t, TaskStage(bases[t], stage)).add(r)
        for t, ts in stages.items():
            sizes[f"{t}.{stage}"] = write_js(
                data / f"{t}.{stage}.js", f'EVT.regStage("{t}","{stage}",', ts.payload()
            )
        print(
            f"{stage}: {sum(len(ts.cols['original']['gen']) for ts in stages.values())} originals",
            flush=True,
        )
    for t, b in bases.items():
        sizes[f"{t}.base"] = write_js(data / f"{t}.base.js", f'EVT.reg("{t}",', b.payload())
        if b.n_fallback:
            print(f"  {t}: {b.n_fallback} control prompts stored verbatim")
    manifest = {
        "stages": STAGES,
        "tasks": {t: {"n": len(b.ids), "variants": b.variants} for t, b in bases.items()},
    }
    write_js(data / "manifest.js", "EVT.setManifest(", manifest)
    total = sum(sizes.values())
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v / 1e6:6.2f} MB  {k}")
    print(f"total {total / 1e6:.1f} MB in {len(sizes)} files")


# ---------------------------------------------------------------- verification


def reconstruct_prompt(base: dict, v: str, i: int) -> str:
    ex = base["ex"][i]
    if "shared_prefix" in base:
        return base["shared_prefix"] + ex["q"]
    if "p" in ex:
        return ex["p"]
    fb = base.get("vprompt", {}).get(v)
    if fb and i < len(fb) and fb[i] is not None:
        return fb[i]
    tpl = base["templates"][v]
    if "<SWAPPED_SRC>" in tpl:
        return tpl.replace("<SWAPPED_SRC>", swap_scenarios(ex["st"]))
    return tpl.replace("<SRC>", ex["st"])


def verify(results: Path, out: Path, k: int, seed: int) -> None:
    rng = random.Random(seed)
    data = out / "data"
    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            s = (data / f"{name}.js").read_text(encoding="utf-8")
            cache[name] = json.loads(s[s.index("{") : s.rindex("}") + 1])
        return cache[name]

    n_lines = sum(1 for _ in open(results / STAGES[0] / "behavior.jsonl"))
    per_stage = {s: sorted(rng.sample(range(n_lines), max(1, k // len(STAGES)))) for s in STAGES}
    checked = 0
    for stage, want in per_stage.items():
        want_set = set(want)
        for li, r in enumerate(iter_records(results / stage / "behavior.jsonl")):
            if li not in want_set:
                continue
            t, v = r["task"], r["metadata"]["variant"]
            base = get(f"{t}.base")
            st = get(f"{t}.{stage}")[v]
            i = base["ids"].index(base_id(r))
            assert reconstruct_prompt(base, v, i) == r["prompt"], (stage, t, v, r["id"], "prompt")
            gen = st["gen"][i]
            assert gen == cap_generation(stage, r["generation"]), (
                stage,
                t,
                v,
                r["id"],
                "generation",
            )
            pred = st["pred"][i]
            assert (
                pred == str(r["prediction"]) if t.startswith("ethics") else pred == r["prediction"]
            ), (stage, t, r["id"], "prediction")
            assert (st["ok"][i] == "1") == bool(r["correct"]), (stage, t, r["id"], "correct")
            assert st["lp"][i] == rnd(r["answer_log_prob_nats"]), (stage, t, r["id"], "lp")
            if t.startswith("ethics"):
                assert st["mg"][i] == rnd(r["logit_margin"])
                assert base["vgold"][v][i] == str(r["answer"])
            else:
                assert base["ex"][i]["a"] == r["answer"]
                assert base["ex"][i]["da"] == r["diagnostic_answer"]
            checked += 1
    print(f"verified {checked} random records across {len(STAGES)} stages: OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", type=int, default=0, help="only verify N random records")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.verify:
        verify(a.results, a.out, a.verify, a.seed)
    else:
        build(a.results, a.out)


if __name__ == "__main__":
    main()
