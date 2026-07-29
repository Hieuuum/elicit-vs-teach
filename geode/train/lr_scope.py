"""Learning-rate scope guard for the run graph (spec 02 §6, V5.71).

The LoRA target rate is scoped to LoRA target runs. A full-FT stage
(installer / bridge / pre-intervention parent) that pins that rate is the
2026-07-25 scope leak that destroyed run 9's retention (the val-loss floor at
the target rate wipes the arithmetic a full-FT stage must preserve). This guard
generalises the per-role LR checks the Phase-3 launcher inlined so a single
tested implementation refuses both a mispinned rate and a cross-scoped one.
"""

from __future__ import annotations

from typing import Literal

_TOL = 1e-12
_FULL_FT_STAGES = ("installer", "bridge", "parent")


def assert_lr_scope(
    cfg: dict, pin: dict, *, stage: Literal["target", "installer", "bridge", "parent"]
) -> None:
    """Refuse a mispinned or scope-leaked ``train.lr`` for one run role (V5.71).

    ``cfg`` is a run config whose ``train.lr`` is checked; ``pin`` is the
    ``lr_pin.yaml`` block (``lr`` = the LoRA target pin, ``installer_lr`` = the
    full-FT pin). Raises ``ValueError``:

    - ``stage="target"``: unless ``train.lr`` equals the pinned target LR.
    - full-FT stage (``installer``/``bridge``/``parent``): if ``train.lr`` equals
      the target LR (the run-9 scope leak); and, for ``installer``/``bridge``,
      unless it equals the pinned installer LR. ``parent`` inherits its rate by
      role and pins no positive value, so only the scope leak is checked.
    """
    lr = float(cfg["train"]["lr"])
    target_pin = float(pin["lr"])
    if stage == "target":
        if abs(lr - target_pin) > _TOL:
            raise ValueError(
                f"target stage pins lr={lr} but lr_pin.yaml records target lr {target_pin}"
            )
        return
    if stage not in _FULL_FT_STAGES:
        raise ValueError(f"assert_lr_scope: unknown stage {stage!r}")
    if abs(lr - target_pin) < _TOL:
        raise ValueError(
            f"full-FT {stage} stage pins the LoRA target lr ({target_pin}) — the "
            "2026-07-25 scope leak that destroyed run 9's retention (lr_pin.yaml)"
        )
    if stage in ("installer", "bridge"):
        installer_pin = float(pin["installer_lr"])
        if abs(lr - installer_pin) > _TOL:
            raise ValueError(
                f"{stage} stage pins lr={lr} but lr_pin.yaml records installer lr {installer_pin}"
            )
