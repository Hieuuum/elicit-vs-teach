"""Prequential MDL / EDL harness (specs/01)."""

from geode.edl.loop import load_snapshot, train_prequential
from geode.edl.masking import TaskFormat, label_mask, masking_config_hash
from geode.edl.metrics import (
    edl_nats,
    edl_per_label_token,
    edl_per_param,
    mdl_nats,
    nats_to_bits,
    pgr,
    prefix_edl_curve,
    training_curve,
)
from geode.edl.prequential import PrequentialAccumulator, StepLoss, prequential_step
from geode.edl.protocol import EVAL_STOP_ROWS, G5_N_SHOTS, g5_leak_ok

__all__ = [
    "EVAL_STOP_ROWS",
    "G5_N_SHOTS",
    "PrequentialAccumulator",
    "StepLoss",
    "TaskFormat",
    "edl_nats",
    "edl_per_label_token",
    "edl_per_param",
    "g5_leak_ok",
    "label_mask",
    "masking_config_hash",
    "mdl_nats",
    "nats_to_bits",
    "pgr",
    "prefix_edl_curve",
    "prequential_step",
    "load_snapshot",
    "train_prequential",
    "training_curve",
]
