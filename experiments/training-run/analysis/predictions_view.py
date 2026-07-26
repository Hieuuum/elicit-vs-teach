"""Build a standalone HTML viewer for a dump_g5_predictions.py CSV.

The CSV is the artifact of record; this is the reading surface for it. 1,024
rows of two models answering the same questions is not browsable in a
spreadsheet — the thing you want to see is *where the two disagree and by how
much*, and that is a per-digit comparison, not a cell value.

So the page marks each model's answer against the true answer right-aligned,
digit by digit, which is what makes "off by one digit in the middle" legible
at a glance. Filters cut to the slices the finding lives in: the paired
outcome, the operator, and the smaller operand's digit count.

Self-contained (data inlined, no network), so it opens from disk. Output goes
next to the CSV in analysis/figures/, which is gitignored and regenerable.

Usage:
    python3 predictions_view.py --csv figures/g5_predictions_n0_n1.csv \
        --out figures/g5_predictions_n0_n1.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

PALETTE = {"both": "both", "n0": "n0", "n1": "n1", "none": "none"}


def outcome_key(correct_in: str, labels: tuple[str, str]) -> str:
    if correct_in == "none":
        return "none"
    if "," in correct_in:
        return "both"
    return "n0" if correct_in == labels[0] else "n1"


def build(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    cols = [c[: -len("_output")] for c in df.columns if c.endswith("_output")]
    if len(cols) != 2:
        raise SystemExit(f"expected exactly 2 runs in {csv}, found {cols}")
    a, b = cols
    df["minor"] = df[["x_digits", "y_digits"]].min(axis=1)

    rows = []
    for r in df.itertuples(index=False):
        ga, gb = getattr(r, f"{a}_parsed"), getattr(r, f"{b}_parsed")
        rows.append(
            {
                "i": int(r.idx),
                "op": r.op,
                "q": f"{r.a} {r.op} {r.b}",
                "t": int(r.true_answer),
                "m": int(r.minor),
                "a": None if pd.isna(ga) else int(ga),
                "b": None if pd.isna(gb) else int(gb),
                "ao": getattr(r, f"{a}_output"),
                "bo": getattr(r, f"{b}_output"),
                "k": outcome_key(r.correct_in, (a, b)),
                "s": bool(r.same_answer),
            }
        )

    tally = {k: sum(1 for r in rows if r["k"] == k) for k in ("both", "n0", "n1", "none")}
    same = sum(1 for r in rows if r["s"])
    acc = {c: float(df[f"{c}_correct"].mean()) for c in (a, b)}

    cells = []
    for op in ("+", "-"):
        for minor in sorted(df.loc[df.op == op, "minor"].unique()):
            z = df[(df.op == op) & (df.minor == minor)]
            cells.append(
                (
                    op,
                    int(minor),
                    len(z),
                    float(z[f"{a}_correct"].mean()),
                    float(z[f"{b}_correct"].mean()),
                )
            )

    grid = "\n".join(
        f'<tr><td class="op">{html.escape(op)}</td><td>{minor}</td><td>{n}</td>'
        f'<td class="{"hi" if va > vb else ""}">{va:.3f}</td>'
        f'<td class="{"hi" if vb > va else ""}">{vb:.3f}</td></tr>'
        for op, minor, n, va, vb in cells
    )

    page = TEMPLATE.format(
        a=html.escape(a),
        b=html.escape(b),
        acc_a=f"{acc[a]:.4f}",
        acc_b=f"{acc[b]:.4f}",
        n=len(rows),
        both=tally["both"],
        only_a=tally["n0"],
        only_b=tally["n1"],
        none=tally["none"],
        same=same,
        grid=grid,
        data=json.dumps(rows, separators=(",", ":")),
    )
    out.write_text(page, encoding="utf-8")
    print(f"[evt] wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(rows)} rows)")


TEMPLATE = """<title>Dose 0 vs dose 1 — per-example predictions</title>
<style>
  :root {{
    --ground:#f6f7f9; --surface:#fff; --raised:#fbfcfd; --ink:#151b23;
    --muted:#626d7a; --line:#e2e6eb; --line-soft:#eef1f4;
    --both:#1b7a55; --n0:#4a52b0; --n1:#9c6412; --none:#8a939e; --wrong:#c2413b;
    --both-bg:#e8f4ee; --n0-bg:#ecedf9; --n1-bg:#f8f0e2; --none-bg:#f0f2f4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ground:#0d1218; --surface:#141b23; --raised:#19222c; --ink:#e6ebf0;
      --muted:#8d98a5; --line:#232d38; --line-soft:#1b232d;
      --both:#45b98a; --n0:#8c93e8; --n1:#d9a055; --none:#6e7885; --wrong:#e8736c;
      --both-bg:#12281f; --n0-bg:#1b1e3a; --n1-bg:#2e2313; --none-bg:#1c232b;
    }}
  }}
  :root[data-theme="light"] {{
    --ground:#f6f7f9; --surface:#fff; --raised:#fbfcfd; --ink:#151b23;
    --muted:#626d7a; --line:#e2e6eb; --line-soft:#eef1f4;
    --both:#1b7a55; --n0:#4a52b0; --n1:#9c6412; --none:#8a939e; --wrong:#c2413b;
    --both-bg:#e8f4ee; --n0-bg:#ecedf9; --n1-bg:#f8f0e2; --none-bg:#f0f2f4;
  }}
  :root[data-theme="dark"] {{
    --ground:#0d1218; --surface:#141b23; --raised:#19222c; --ink:#e6ebf0;
    --muted:#8d98a5; --line:#232d38; --line-soft:#1b232d;
    --both:#45b98a; --n0:#8c93e8; --n1:#d9a055; --none:#6e7885; --wrong:#e8736c;
    --both-bg:#12281f; --n0-bg:#1b1e3a; --n1-bg:#2e2313; --none-bg:#1c232b;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--ground); color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:40px 24px 80px; display:flex; flex-direction:column; gap:34px; }}
  h1 {{
    font-family:ui-serif,Georgia,"Times New Roman",serif;
    font-size:clamp(28px,4vw,40px); font-weight:600; line-height:1.15;
    letter-spacing:-.015em; margin:0; text-wrap:balance;
  }}
  h2 {{
    font-family:ui-serif,Georgia,serif; font-size:19px; font-weight:600;
    margin:0 0 14px; letter-spacing:-.01em;
  }}
  .lede {{ margin:10px 0 0; color:var(--muted); max-width:64ch; }}
  .prov {{
    margin:18px 0 0; padding:12px 15px; border-left:2px solid var(--both);
    background:var(--surface); color:var(--muted); font-size:13.5px; max-width:70ch;
  }}
  .prov b {{ color:var(--ink); font-weight:600; }}
  code, .mono {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-variant-numeric:tabular-nums; }}
  .eyebrow {{
    font-size:11px; font-weight:650; letter-spacing:.1em; text-transform:uppercase;
    color:var(--muted); margin:0 0 12px;
  }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .stat {{
    background:var(--surface); border:1px solid var(--line); border-radius:3px;
    padding:15px 16px; display:flex; flex-direction:column; gap:3px;
    border-top:3px solid var(--c,var(--none));
  }}
  .stat .v {{ font-size:29px; font-weight:600; line-height:1.05; font-variant-numeric:tabular-nums; color:var(--c,var(--ink)); }}
  .stat .l {{ font-size:12.5px; color:var(--muted); }}
  .stat.both {{ --c:var(--both); }} .stat.n0 {{ --c:var(--n0); }}
  .stat.n1 {{ --c:var(--n1); }} .stat.none {{ --c:var(--none); }}
  .panel {{ background:var(--surface); border:1px solid var(--line); border-radius:3px; padding:20px 22px; }}
  .split {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }}
  @media (max-width:760px) {{ .split {{ grid-template-columns:1fr; }} }}
  table.grid {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  table.grid th {{
    text-align:left; font-size:11px; letter-spacing:.07em; text-transform:uppercase;
    color:var(--muted); font-weight:650; padding:0 10px 7px 0; border-bottom:1px solid var(--line);
  }}
  table.grid td {{ padding:5px 10px 5px 0; border-bottom:1px solid var(--line-soft); font-variant-numeric:tabular-nums; }}
  table.grid td.op {{ font-family:ui-monospace,Menlo,monospace; font-weight:700; }}
  table.grid td.hi {{ color:var(--both); font-weight:650; }}
  .note {{ font-size:13px; color:var(--muted); margin:14px 0 0; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:18px; align-items:flex-end; }}
  .group {{ display:flex; flex-direction:column; gap:7px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
  button.chip {{
    font:inherit; font-size:13px; padding:5px 11px; border-radius:2px; cursor:pointer;
    background:var(--surface); color:var(--muted); border:1px solid var(--line);
  }}
  button.chip:hover {{ color:var(--ink); border-color:var(--muted); }}
  button.chip:focus-visible {{ outline:2px solid var(--n0); outline-offset:1px; }}
  button.chip[aria-pressed="true"] {{ background:var(--ink); color:var(--ground); border-color:var(--ink); font-weight:600; }}
  button.chip[aria-pressed="true"].k-both {{ background:var(--both); border-color:var(--both); color:#fff; }}
  button.chip[aria-pressed="true"].k-n0 {{ background:var(--n0); border-color:var(--n0); color:#fff; }}
  button.chip[aria-pressed="true"].k-n1 {{ background:var(--n1); border-color:var(--n1); color:#fff; }}
  input[type=search] {{
    font:inherit; font-size:13px; padding:6px 10px; min-width:190px;
    background:var(--surface); color:var(--ink); border:1px solid var(--line); border-radius:2px;
  }}
  input[type=search]:focus-visible {{ outline:2px solid var(--n0); outline-offset:1px; }}
  .count {{ font-size:13px; color:var(--muted); }}
  .count b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
  .scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:3px; background:var(--surface); }}
  table.rows {{ border-collapse:collapse; width:100%; font-size:13.5px; min-width:720px; }}
  table.rows thead th {{
    position:sticky; top:0; z-index:1; background:var(--raised); text-align:left;
    font-size:11px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
    font-weight:650; padding:10px 14px; border-bottom:1px solid var(--line); white-space:nowrap;
  }}
  table.rows td {{ padding:7px 14px; border-bottom:1px solid var(--line-soft); vertical-align:middle; }}
  table.rows tr:last-child td {{ border-bottom:0; }}
  td.q, td.ans {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  td.ans {{ text-align:right; }}
  td.truth {{ font-family:ui-monospace,Menlo,monospace; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }}
  .d {{ color:var(--wrong); font-weight:700; }}
  .ok {{ color:var(--both); }}
  .idx {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
  .pill {{
    display:inline-block; font-size:11px; font-weight:650; letter-spacing:.04em;
    padding:2px 8px; border-radius:2px; white-space:nowrap;
  }}
  .pill.both {{ background:var(--both-bg); color:var(--both); }}
  .pill.n0 {{ background:var(--n0-bg); color:var(--n0); }}
  .pill.n1 {{ background:var(--n1-bg); color:var(--n1); }}
  .pill.none {{ background:var(--none-bg); color:var(--none); }}
  .empty {{ padding:44px; text-align:center; color:var(--muted); }}
  footer {{ color:var(--muted); font-size:12.5px; border-top:1px solid var(--line); padding-top:18px; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Elicit arm · installer stage · zero-shot on operator add/sub</p>
    <h1>One multiplication example, and what it cost</h1>
    <p class="lede">
      Two checkpoints answering the identical 1,024 questions from the frozen
      <code>D_target_eval</code> set. <b>n=0</b> is <code>{a}</code>, untouched.
      <b>n=1</b> is <code>{b}</code>, the same model after seeing exactly one
      multiplication example. Wrong digits are marked against the true answer,
      right-aligned.
    </p>
    <p class="prov">
      Protocol is G5's zero-shot arm run verbatim, and both checkpoints reproduced their
      recorded accuracy exactly — <b>{acc_a}</b> and <b>{acc_b}</b>. These rows explain
      the recorded numbers rather than a separately-built eval.
    </p>
  </header>

  <section>
    <p class="eyebrow">Paired outcome, same questions</p>
    <div class="stats">
      <div class="stat both"><span class="v">{both}</span><span class="l">both correct</span></div>
      <div class="stat n0"><span class="v">{only_a}</span><span class="l">only n=0 correct</span></div>
      <div class="stat n1"><span class="v">{only_b}</span><span class="l">only n=1 correct</span></div>
      <div class="stat none"><span class="v">{none}</span><span class="l">neither correct</span></div>
      <div class="stat"><span class="v">{same}</span><span class="l">identical answer from both</span></div>
    </div>
  </section>

  <section class="panel">
    <div class="split">
      <div>
        <h2>Accuracy by operator and operand size</h2>
        <div class="scroll" style="border:0">
          <table class="grid">
            <thead><tr><th>op</th><th>smaller operand</th><th>n</th><th>n=0</th><th>n=1</th></tr></thead>
            <tbody>{grid}</tbody>
          </table>
        </div>
      </div>
      <div>
        <h2>What the split shows</h2>
        <p class="note" style="margin-top:0">
          Subtraction transferred into operator notation and addition did not. At a
          matched one-digit smaller operand the untouched model scores 0.583 on
          <code>−</code> against 0.011 on <code>+</code> — the same digit budget, so
          this is not a difficulty gap. Its subtraction sign is correct on all 516
          items, while 395 of 508 addition answers come out <em>smaller than the
          larger operand</em>, which no sum of two positives can be.
        </p>
        <p class="note">
          The dose then pushes the subtraction rows down at every operand size.
          Addition was already at the floor and stays there.
        </p>
      </div>
    </div>
  </section>

  <section>
    <p class="eyebrow">Browse the rows</p>
    <div class="controls">
      <div class="group">
        <span class="count">Outcome</span>
        <div class="chips" id="f-outcome">
          <button class="chip" data-v="all" aria-pressed="true">All</button>
          <button class="chip k-n0" data-v="disagree" aria-pressed="false">Disagreements</button>
          <button class="chip k-both" data-v="both" aria-pressed="false">Both correct</button>
          <button class="chip" data-v="none" aria-pressed="false">Neither</button>
        </div>
      </div>
      <div class="group">
        <span class="count">Operator</span>
        <div class="chips" id="f-op">
          <button class="chip" data-v="all" aria-pressed="true">Both</button>
          <button class="chip" data-v="+" aria-pressed="false">+</button>
          <button class="chip" data-v="-" aria-pressed="false">−</button>
        </div>
      </div>
      <div class="group">
        <span class="count">Smaller operand</span>
        <div class="chips" id="f-minor">
          <button class="chip" data-v="all" aria-pressed="true">Any</button>
          <button class="chip" data-v="1" aria-pressed="false">1 digit</button>
          <button class="chip" data-v="2" aria-pressed="false">2</button>
          <button class="chip" data-v="3" aria-pressed="false">3</button>
          <button class="chip" data-v="4" aria-pressed="false">4</button>
        </div>
      </div>
      <div class="group">
        <span class="count">Find</span>
        <input type="search" id="q" placeholder="e.g. 3499 or - 9" aria-label="Search questions">
      </div>
      <p class="count" style="margin:0 0 6px"><b id="shown">{n}</b> of {n} rows</p>
    </div>
  </section>

  <div class="scroll">
    <table class="rows">
      <thead>
        <tr>
          <th>row</th><th>question</th><th style="text-align:right">true</th>
          <th style="text-align:right">n=0 says</th><th style="text-align:right">n=1 says</th><th>outcome</th>
        </tr>
      </thead>
      <tbody id="body"></tbody>
    </table>
    <div class="empty" id="empty" hidden>No rows match these filters.</div>
  </div>

  <footer>
    Generated by <code>analysis/predictions_view.py</code> from
    <code>g5_predictions_n0_n1.csv</code> (<code>scripts/dump_g5_predictions.py</code>).
    Exact match at ~10% is a noisy, all-or-nothing readout; the load-bearing evidence
    for dose damage is the test-loss gap of 5.1935 → 5.5502 nats over 97,952 rows.
    These rows show its shape.
  </footer>
</div>

<script>
const DATA = {data};
const LABEL = {{both:"both", n0:"only n=0", n1:"only n=1", none:"neither"}};
const F = {{outcome:"all", op:"all", minor:"all", q:""}};

const esc = s => String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);

// Right-align against the true answer and mark every character that differs:
// the failure this dump exists to show is a single wrong digit inside an
// otherwise-correct answer, which a plain cell value hides.
function mark(val, truth) {{
  if (val === null) return '<span class="d">unparseable</span>';
  const a = String(val), b = String(truth);
  if (a === b) return '<span class="ok">' + esc(a) + '</span>';
  const n = Math.max(a.length, b.length);
  const pa = a.padStart(n, " "), pb = b.padStart(n, " ");
  let out = "";
  for (let i = 0; i < n; i++) {{
    if (pa[i] === " ") continue;
    out += pa[i] === pb[i] ? esc(pa[i]) : '<b class="d">' + esc(pa[i]) + "</b>";
  }}
  return out;
}}

function keep(r) {{
  if (F.outcome === "disagree" && r.k !== "n0" && r.k !== "n1") return false;
  if (F.outcome === "both" && r.k !== "both") return false;
  if (F.outcome === "none" && r.k !== "none") return false;
  if (F.op !== "all" && r.op !== F.op) return false;
  if (F.minor !== "all" && r.m !== +F.minor) return false;
  if (F.q && !r.q.includes(F.q)) return false;
  return true;
}}

function render() {{
  const rows = DATA.filter(keep);
  document.getElementById("shown").textContent = rows.length;
  document.getElementById("empty").hidden = rows.length > 0;
  document.getElementById("body").innerHTML = rows.map(r => `
    <tr>
      <td class="idx">${{r.i}}</td>
      <td class="q">${{esc(r.q)}}</td>
      <td class="truth">${{r.t}}</td>
      <td class="ans">${{mark(r.a, r.t)}}</td>
      <td class="ans">${{mark(r.b, r.t)}}</td>
      <td><span class="pill ${{r.k}}">${{LABEL[r.k]}}</span></td>
    </tr>`).join("");
}}

for (const [id, key] of [["f-outcome","outcome"], ["f-op","op"], ["f-minor","minor"]]) {{
  document.getElementById(id).addEventListener("click", e => {{
    const btn = e.target.closest("button");
    if (!btn) return;
    F[key] = btn.dataset.v;
    for (const s of e.currentTarget.querySelectorAll("button"))
      s.setAttribute("aria-pressed", String(s === btn));
    render();
  }});
}}
document.getElementById("q").addEventListener("input", e => {{
  F.q = e.target.value.trim();
  render();
}});
render();
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.csv, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
