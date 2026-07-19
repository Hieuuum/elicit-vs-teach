"""Corpus packing + full-FT/pretrain trainer (specs/02 §6.1)."""

from geode.train.loop import TrainResult, evaluate_nll_nats, train_full
from geode.train.packing import pack_corpus, split_documents, train_val_split
from geode.train.stopping import ConvergenceTracker, StoppingRule

__all__ = [
    "ConvergenceTracker",
    "StoppingRule",
    "TrainResult",
    "evaluate_nll_nats",
    "pack_corpus",
    "split_documents",
    "train_full",
    "train_val_split",
]
