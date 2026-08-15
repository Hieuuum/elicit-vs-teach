"""Verdict rule for the ts38mw wrapper-diversity install probe (plan
docs/plan-ts38mw-multiwrap-install.md §2/§3.4, decisions.md 2026-08-15
"ts38mw Stage 1 pre-registration" entry).

Why. ts38's certified parent computes op-notation arithmetic ONLY under its
exact training template — held-out phrasings (even the symbol-bearing
``What is a + b?``) collapse to ~3%. ts38mw installs the same arithmetic
under 8 wrapper templates instead of one and asks whether the skill fires
under a held-out wrapper it never saw, including the word-only target
phrasing. ``launch_ts38mw_probe.sh`` scores every snapshot of that ONE run
against 6 probe pins (``bare_op``/``sym_q``/``word_q``/``sumof``/
``sumof_bare``/``dm_mix``, each ``gates.py g5 --no-record``) plus the base
model on the same 6 pins, into ``results/ts38mw_probe.json``. This module is
the pure selection function over that table — no argv, no file I/O, no
network — so it is testable without a probe run, mirroring
``certified_step.py``'s ``select_certified_step``.

Rule (plan §2, frozen once committed — never tuned post-hoc):

- **Qualifying snapshots** are those whose ``canonical_em`` (the ``bare_op``
  pin's zero-shot EM — the "G1-canonical" op-format number) is >= 0.95;
  everything below is off the table (skill did not install at that step).
- **Persistence** ("two consecutive scored snapshots") means two ADJACENT
  entries in the qualifying list sorted by step — adjacency by list
  position, not by a fixed step delta: ts38mw's snapshot_steps are unevenly
  spaced (4k/8k/16k gaps), unlike ``certified_step``'s fixed-1000-step probe,
  so a step+1000 rule cannot transfer. A single qualifying snapshot,
  including the last one scored, can never satisfy persistence on its own.
  GO-A/GO-B require persistence; WEAK does not (plan §2's "best ... EM"
  language, scoped only to "each GO" needing persistence) — WEAK is the
  single best qualifying-snapshot reading, no adjacency required.
- **GO-A**: two adjacent qualifying snapshots each with ``sumof`` zero-shot
  EM >= 0.20 AND ``sumof`` loss < the base's ``sumof`` loss (the loss guard
  against "learned to emit digits" reading as transfer).
- **GO-B**: GO-A did not fire, but two adjacent qualifying snapshots each
  with ``sym_q`` zero-shot EM >= 0.50 AND ``sym_q`` loss < the base's
  ``sym_q`` loss. GO-A and GO-B are evaluated independently (this module
  never short-circuits GO-B's check just because GO-A already fired) so a
  run satisfying both bands is reported as GO-A, never masked by GO-B.
- **WEAK**: neither GO fired; the single qualifying snapshot with the
  highest ``sumof`` EM has that EM in [0.05, 0.20) AND its loss < base.
- **NO-GO**: none of the above.
- **INCONCLUSIVE**: no snapshot reaches canonical EM >= 0.95 at all (the
  install never took at this LR — plan §4.6's LR-sweep trigger).

Named edge case (owner-visible, not silently smoothed): because GO-A
requires PERSISTENCE at EM >= 0.20 while WEAK only requires the SINGLE-BEST
EM to fall in [0.05, 0.20), a lone (non-persisting) ``sumof`` crossing well
above 0.20 — say a single snapshot at 0.30 with loss < base, with no
adjacent qualifying snapshot repeating it — lands in NO-GO (fails GO-A's
persistence, and 0.30 is outside WEAK's upper-bounded range), while a lone
crossing at 0.15 lands in WEAK. This is the plan's band definitions applied
literally, not a defect in this module; the owner should be aware a single
strong-but-unrepeated reading reads WORSE than a weak-but-unrepeated one.
"""

from __future__ import annotations

CANONICAL_EM_BAR = 0.95
GO_A_EM_BAR = 0.20
GO_B_EM_BAR = 0.50
WEAK_EM_LOW = 0.05
WEAK_EM_HIGH = 0.20  # exclusive; equals GO_A_EM_BAR by construction


def _persists(
    qualifying: list[dict], pin: str, em_bar: float, base_loss: float
) -> tuple[bool, int | None]:
    """True + the step of the second row of the earliest adjacent pair (in
    ``qualifying``, already sorted by step) where both rows satisfy
    ``pins[pin]['em0'] >= em_bar`` and ``pins[pin]['loss'] < base_loss``."""

    def ok(row: dict) -> bool:
        p = row["pins"][pin]
        return p["em0"] >= em_bar and p["loss"] < base_loss

    for i in range(len(qualifying) - 1):
        if ok(qualifying[i]) and ok(qualifying[i + 1]):
            return True, qualifying[i + 1]["step"]
    return False, None


def verdict(rows: list[dict], base: dict) -> dict:
    """The band (GO-A/GO-B/WEAK/NO-GO/INCONCLUSIVE) for one probe table.

    ``rows`` is the probe table (any order): a list of per-snapshot dicts
    each carrying ``step``, ``canonical_em``, and ``pins`` — a mapping of pin
    name (at least ``sumof`` and ``sym_q``) to ``{"em0": float, "loss":
    float}`` (``em16`` may also be present; never consulted here — plan §2,
    "16-shot is recorded, never a criterion"). ``base`` carries the same
    ``pins`` mapping for the base model, scored identically. Returns:

    - ``band``: one of GO-A/GO-B/WEAK/NO-GO/INCONCLUSIVE.
    - ``step``: the deciding step (persistence's second row for a GO, the
      best row for WEAK), or ``None`` for NO-GO/INCONCLUSIVE.
    - ``qualifying_steps``: every step with ``canonical_em >= 0.95``, sorted.
    - ``go_a_persisted`` / ``go_b_persisted``: independent booleans (see
      module docstring — GO-B is always computed, never skipped).
    - ``best_sumof_em0`` / ``best_sumof_step``: the qualifying snapshot with
      the highest ``sumof`` EM, or ``None`` if no snapshot qualifies.
    """
    ordered = sorted(rows, key=lambda r: r["step"])
    qualifying = [r for r in ordered if r["canonical_em"] >= CANONICAL_EM_BAR]

    if not qualifying:
        return {
            "band": "INCONCLUSIVE",
            "step": None,
            "qualifying_steps": [],
            "go_a_persisted": False,
            "go_b_persisted": False,
            "best_sumof_em0": None,
            "best_sumof_step": None,
        }

    go_a_ok, go_a_step = _persists(qualifying, "sumof", GO_A_EM_BAR, base["pins"]["sumof"]["loss"])
    go_b_ok, go_b_step = _persists(qualifying, "sym_q", GO_B_EM_BAR, base["pins"]["sym_q"]["loss"])

    best_row = max(qualifying, key=lambda r: r["pins"]["sumof"]["em0"])
    best_em = best_row["pins"]["sumof"]["em0"]
    best_loss = best_row["pins"]["sumof"]["loss"]
    is_weak = WEAK_EM_LOW <= best_em < WEAK_EM_HIGH and best_loss < base["pins"]["sumof"]["loss"]

    if go_a_ok:
        band, step = "GO-A", go_a_step
    elif go_b_ok:
        band, step = "GO-B", go_b_step
    elif is_weak:
        band, step = "WEAK", best_row["step"]
    else:
        band, step = "NO-GO", None

    return {
        "band": band,
        "step": step,
        "qualifying_steps": [r["step"] for r in qualifying],
        "go_a_persisted": go_a_ok,
        "go_b_persisted": go_b_ok,
        "best_sumof_em0": best_em,
        "best_sumof_step": best_row["step"],
    }
