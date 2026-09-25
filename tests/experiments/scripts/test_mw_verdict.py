"""ts38mw wrapper-diversity probe: the verdict() selection rule.

Mirrors test_ts38_certified_parent.py's Part 3 (select_certified_step): the
rule is pure and argv/IO-free specifically so it can be checked without a
probe run. decisions.md 2026-08-15 "ts38mw Stage 1 pre-registration" entry
freezes these bands; a silent drift here (picking the wrong band, or a band
without the persistence the plan requires) would misreport a falsification
result the owner pre-registered on.
"""

from __future__ import annotations

from tests._scriptloader import load

mw_verdict = load("mw_verdict")
verdict = mw_verdict.verdict


def _pin(em0: float, loss: float, em16: float = 0.0) -> dict:
    return {"em0": em0, "em16": em16, "loss": loss}


def _row(
    step: int,
    canonical_em: float,
    sumof_em0: float,
    sumof_loss: float,
    sym_q_em0: float = 0.0,
    sym_q_loss: float = 99.0,
) -> dict:
    return {
        "step": step,
        "g1_own": canonical_em,
        "canonical_em": canonical_em,
        "pins": {
            "sumof": _pin(sumof_em0, sumof_loss),
            "sym_q": _pin(sym_q_em0, sym_q_loss),
        },
    }


BASE = {"pins": {"sumof": _pin(0.0, 8.0), "sym_q": _pin(0.0, 8.0)}}


# =============================================================================
# INCONCLUSIVE — no snapshot reaches the canonical-EM bar
# =============================================================================


def test_inconclusive_when_nothing_reaches_canonical_bar():
    rows = [
        _row(8000, canonical_em=0.60, sumof_em0=0.50, sumof_loss=1.0),
        _row(12000, canonical_em=0.80, sumof_em0=0.50, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "INCONCLUSIVE"
    assert result["step"] is None
    assert result["qualifying_steps"] == []


# =============================================================================
# GO-A persistence — a lone crossing is not enough
# =============================================================================


def test_go_a_requires_persistence_lone_crossing_is_not_go():
    # Only step 12000 qualifies at all (canonical_em >= 0.95); a single
    # qualifying row can never satisfy the two-consecutive-snapshot rule.
    rows = [
        _row(8000, canonical_em=0.80, sumof_em0=0.50, sumof_loss=1.0),
        _row(12000, canonical_em=0.96, sumof_em0=0.50, sumof_loss=1.0),
        _row(16000, canonical_em=0.90, sumof_em0=0.50, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] != "GO-A"
    assert result["go_a_persisted"] is False


def test_go_a_persists_at_two_adjacent_qualifying_snapshots():
    rows = [
        _row(8000, canonical_em=0.80, sumof_em0=0.50, sumof_loss=1.0),
        _row(12000, canonical_em=0.96, sumof_em0=0.25, sumof_loss=1.0),
        _row(16000, canonical_em=0.97, sumof_em0=0.30, sumof_loss=1.2),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "GO-A"
    assert result["step"] == 16000  # second row of the adjacent pair
    assert result["go_a_persisted"] is True
    assert result["qualifying_steps"] == [12000, 16000]


def test_go_a_persistence_is_adjacency_in_qualifying_list_not_fixed_step_delta():
    # snapshot_steps are unevenly spaced (plan §3.3: 4k/8k/16k gaps); the
    # non-qualifying middle row must not break adjacency between the two
    # qualifying rows on either side of it.
    rows = [
        _row(8000, canonical_em=0.96, sumof_em0=0.22, sumof_loss=1.0),
        _row(28000, canonical_em=0.40, sumof_em0=0.99, sumof_loss=0.1),  # disqualified
        _row(96000, canonical_em=0.97, sumof_em0=0.25, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "GO-A"
    assert result["step"] == 96000
    assert result["qualifying_steps"] == [8000, 96000]


# =============================================================================
# Loss guard
# =============================================================================


def test_loss_guard_high_em_but_loss_not_below_base_is_not_go_a():
    # sumof EM persists well above the GO-A bar, but loss never beats base:
    # "learned to emit digits" reads as high EM, not as transfer.
    rows = [
        _row(12000, canonical_em=0.96, sumof_em0=0.30, sumof_loss=9.0),
        _row(16000, canonical_em=0.97, sumof_em0=0.35, sumof_loss=9.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] != "GO-A"
    assert result["go_a_persisted"] is False
    # 0.30/0.35 sit outside WEAK's [0.05, 0.20) range too -> NO-GO, not WEAK.
    assert result["band"] == "NO-GO"


def test_loss_guard_blocks_weak_too():
    rows = [
        _row(12000, canonical_em=0.96, sumof_em0=0.10, sumof_loss=9.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "NO-GO"


# =============================================================================
# GO-B — only considered once GO-A fails, but always computed independently
# =============================================================================


def test_go_b_fires_when_go_a_fails():
    rows = [
        _row(
            12000, canonical_em=0.96, sumof_em0=0.02, sumof_loss=1.0, sym_q_em0=0.55, sym_q_loss=1.0
        ),
        _row(
            16000, canonical_em=0.97, sumof_em0=0.03, sumof_loss=1.0, sym_q_em0=0.60, sym_q_loss=1.0
        ),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "GO-B"
    assert result["step"] == 16000
    assert result["go_a_persisted"] is False
    assert result["go_b_persisted"] is True


def test_go_b_does_not_short_circuit_a_valid_go_a():
    # Both GO-A's and GO-B's criteria persist at the same two snapshots —
    # GO-A must win, and go_b_persisted must still read True (independent
    # computation, not skipped just because GO-A already fired).
    rows = [
        _row(
            12000, canonical_em=0.96, sumof_em0=0.25, sumof_loss=1.0, sym_q_em0=0.55, sym_q_loss=1.0
        ),
        _row(
            16000, canonical_em=0.97, sumof_em0=0.30, sumof_loss=1.0, sym_q_em0=0.60, sym_q_loss=1.0
        ),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "GO-A"
    assert result["go_a_persisted"] is True
    assert result["go_b_persisted"] is True  # computed, not masked


# =============================================================================
# WEAK — best single qualifying-snapshot reading, no persistence required
# =============================================================================


def test_weak_band_no_persistence_required():
    rows = [
        _row(12000, canonical_em=0.96, sumof_em0=0.15, sumof_loss=1.0),
        _row(16000, canonical_em=0.97, sumof_em0=0.02, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "WEAK"
    assert result["step"] == 12000
    assert result["best_sumof_em0"] == 0.15


def test_weak_boundary_excludes_0_20():
    # WEAK's range is [0.05, 0.20) — exactly 0.20 belongs to GO-A's
    # threshold, not WEAK, and a lone (non-persisting) 0.20 satisfies
    # neither band.
    rows = [
        _row(12000, canonical_em=0.96, sumof_em0=0.20, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "NO-GO"


# =============================================================================
# NO-GO
# =============================================================================


def test_no_go_when_best_em_below_weak_floor():
    rows = [
        _row(12000, canonical_em=0.96, sumof_em0=0.01, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "NO-GO"
    assert result["step"] is None


# =============================================================================
# Misc
# =============================================================================


def test_verdict_handles_unsorted_rows():
    rows = [
        _row(16000, canonical_em=0.97, sumof_em0=0.30, sumof_loss=1.2),
        _row(8000, canonical_em=0.80, sumof_em0=0.50, sumof_loss=1.0),
        _row(12000, canonical_em=0.96, sumof_em0=0.25, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["band"] == "GO-A"
    assert result["step"] == 16000
    assert result["qualifying_steps"] == [12000, 16000]


def test_best_sumof_tracked_even_on_no_go():
    rows = [
        _row(12000, canonical_em=0.96, sumof_em0=0.01, sumof_loss=1.0),
        _row(16000, canonical_em=0.97, sumof_em0=0.03, sumof_loss=1.0),
    ]
    result = verdict(rows, BASE)
    assert result["best_sumof_em0"] == 0.03
    assert result["best_sumof_step"] == 16000
