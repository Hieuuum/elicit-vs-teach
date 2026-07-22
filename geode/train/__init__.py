"""Corpus packing + full-FT/pretrain trainer (specs/02 §6.1)."""

from geode.train.loop import TrainResult, evaluate_nll_nats, train_full
from geode.train.lora import LORA_TARGET_MODULES, LoRALinear, apply_lora, reapply_lora
from geode.train.packing import pack_corpus, split_documents, split_indices, train_val_split
from geode.train.stopping import (
    BehavioralStoppingRule,
    BehaviorTracker,
    ConvergenceTracker,
    StoppingRule,
)
from geode.train.sft import evaluate_sft_nll_nats, train_sft

__all__ = [
    "BehavioralStoppingRule",
    "BehaviorTracker",
    "ConvergenceTracker",
    "LORA_TARGET_MODULES",
    "LoRALinear",
    "StoppingRule",
    "TrainResult",
    "apply_lora",
    "evaluate_nll_nats",
    "pack_corpus",
    "reapply_lora",
    "split_documents",
    "split_indices",
    "train_full",
    "train_val_split",
    "evaluate_sft_nll_nats",
    "train_sft",
]
