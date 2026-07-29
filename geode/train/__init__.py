"""Corpus packing + full-FT/pretrain trainer (specs/02 §6.1)."""

from geode.train.embedding import (
    EmbeddingTrainResult,
    assert_untie_is_noop,
    resolve_warmstart_rows,
    select_warmstart_candidate,
    train_embedding_rows,
    untie_lm_head,
)
from geode.train.loop import TrainResult, evaluate_nll_nats, train_full
from geode.train.lr_scope import assert_lr_scope
from geode.train.lora import LORA_TARGET_MODULES, LoRALinear, apply_lora, merge_lora, reapply_lora
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
    "EmbeddingTrainResult",
    "LORA_TARGET_MODULES",
    "LoRALinear",
    "StoppingRule",
    "TrainResult",
    "apply_lora",
    "assert_lr_scope",
    "assert_untie_is_noop",
    "evaluate_nll_nats",
    "merge_lora",
    "pack_corpus",
    "reapply_lora",
    "resolve_warmstart_rows",
    "select_warmstart_candidate",
    "split_documents",
    "split_indices",
    "train_embedding_rows",
    "train_full",
    "train_val_split",
    "untie_lm_head",
    "evaluate_sft_nll_nats",
    "train_sft",
]
