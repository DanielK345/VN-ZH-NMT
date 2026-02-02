"""Data loading utilities for training."""

import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import sentencepiece as spm
from typing import List, Tuple
import math

from .config import RopeConfig


class BidirectionalTranslationDataset(Dataset):
    """Dataset for bidirectional Vietnamese-Chinese translation with span masking."""

    def __init__(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        sp_model: spm.SentencePieceProcessor,
        config: RopeConfig,
        is_training: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            src_lines: List of source sentences
            tgt_lines: List of target sentences
            sp_model: SentencePiece tokenizer
            config: Configuration object
            is_training: Whether to use for training or validation
        """
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.sp = sp_model
        self.config = config
        self.is_training = is_training

        self.pad_id = sp_model.piece_to_id(config.pad_token)
        self.bos_id = sp_model.piece_to_id(config.bos_token)
        self.eos_id = sp_model.piece_to_id(config.eos_token)
        self.unk_id = sp_model.piece_to_id(config.unk_token)
        self.zh_id = sp_model.piece_to_id(config.zh_token)
        self.vi_id = sp_model.piece_to_id(config.vi_token)

        self.samples = []
        if is_training:
            for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
                if i % 2 == 0:
                    self.samples.append((self.add_lang_token(src, config.vi_token), tgt, "zh2vi"))
                else:
                    self.samples.append((self.add_lang_token(src, config.zh_token), tgt, "vi2zh"))
        else:
            for src, tgt in zip(src_lines, tgt_lines):
                self.samples.append((self.add_lang_token(src, config.vi_token), tgt, "zh2vi"))

        self.zh2vi_indices = [idx for idx, (_, _, d) in enumerate(self.samples) if d == "zh2vi"]
        self.vi2zh_indices = [idx for idx, (_, _, d) in enumerate(self.samples) if d == "vi2zh"]

    def add_lang_token(self, text: str, lang_tok: str) -> str:
        """Add language token to text."""
        text = text.strip()
        if text.startswith("<2vi>") or text.startswith("<2zh>"):
            return text
        return f"{lang_tok} {text}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        src_text, tgt_text, _ = self.samples[idx]
        src_ids = self.sp.encode(src_text, out_type=int)[: self.config.max_len]
        tgt_ids = [self.bos_id] + self.sp.encode(tgt_text, out_type=int) + [self.eos_id]
        tgt_ids = tgt_ids[: self.config.max_len]
        src_ids = self.apply_span_masking(src_ids)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    def apply_span_masking(self, src_ids: List[int]) -> List[int]:
        """Apply span masking to source IDs during training."""
        if not self.is_training or random.random() > self.config.span_mask_prob:
            return src_ids
        
        src_ids = src_ids.copy()
        special = {self.pad_id, self.bos_id, self.eos_id, self.zh_id, self.vi_id}
        maskable = [i for i, t in enumerate(src_ids) if t not in special]
        
        if not maskable:
            return src_ids
        
        num_to_mask = min(random.randint(1, 2), len(maskable))
        start = random.choice(maskable)
        for i in range(start, min(start + num_to_mask, len(src_ids))):
            if i in maskable and random.random() < 0.7:
                src_ids[i] = self.unk_id
        
        return src_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for batch processing."""
    src_list, tgt_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    
    src_batch = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_batch = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_batch[i, : len(src)] = src
        tgt_batch[i, : len(tgt)] = tgt
    
    return src_batch, tgt_batch


def select_vi2zh_window(
    indices: List[int],
    epoch: int,
    ratio: float,
) -> List[int]:
    """
    Select a window of vi2zh samples based on epoch.
    Gradually increases coverage over epochs.
    """
    if not indices or ratio <= 0:
        return []
    
    total = len(indices)
    window = max(1, int(math.ceil(total * min(ratio, 1.0))))
    start = ((epoch - 1) * window) % total
    end = start + window
    
    if end <= total:
        return indices[start:end]
    
    wrap = end - total
    return indices[start:] + indices[:wrap]


def build_train_loader(
    dataset: BidirectionalTranslationDataset,
    config: RopeConfig,
    epoch: int,
) -> Tuple[DataLoader, int]:
    """
    Build training DataLoader with curriculum learning for vi2zh.
    
    Args:
        dataset: Training dataset
        config: Configuration
        epoch: Current epoch
        
    Returns:
        Tuple of DataLoader and number of vi2zh samples in this epoch
    """
    active = list(dataset.zh2vi_indices)
    vi_slice = select_vi2zh_window(dataset.vi2zh_indices, epoch, config.vi2zh_epoch_ratio)
    active.extend(vi_slice)
    
    sampler = SubsetRandomSampler(active)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    return loader, len(vi_slice)
