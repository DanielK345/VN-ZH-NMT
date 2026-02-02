"""Training pipeline for bidirectional Vietnamese-Chinese machine translation."""

from .config import RopeConfig, ContrastiveConfig
from .data_processor import DataProcessor
from .data_loader import BidirectionalTranslationDataset, build_train_loader
from .trainer import Trainer, ContrastiveTrainer
from .utils import (
    LabelSmoothedCrossEntropyLoss,
    WarmupInverseSqrtScheduler,
    greedy_decode,
    beam_search_decode,
    evaluate
)

__all__ = [
    "RopeConfig",
    "ContrastiveConfig",
    "DataProcessor",
    "BidirectionalTranslationDataset",
    "build_train_loader",
    "Trainer",
    "ContrastiveTrainer",
    "LabelSmoothedCrossEntropyLoss",
    "WarmupInverseSqrtScheduler",
    "greedy_decode",
    "beam_search_decode",
    "evaluate",
]
