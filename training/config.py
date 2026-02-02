"""Configuration classes for training."""

from dataclasses import dataclass, field
import torch
from typing import Optional


@dataclass
class RopeConfig:
    """Configuration for model training with RoPE attention."""
    train_src_file: str
    train_tgt_file: str
    spm_prefix: str
    train_back_dir: str = "./clean_data"
    vocab_size: int = 8000
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    d_ff: int = 3072
    dropout: float = 0.01
    max_len: int = 32
    rope_base: float = 10000.0
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    zh_token: str = "<2zh>"
    vi_token: str = "<2vi>"
    batch_size: int = 128
    num_epochs: int = 40
    lr_base: float = 2e-4
    warmup_steps: int = 200
    weight_decay: float = 0.01
    label_smoothing: float = 0.01
    grad_clip: float = 1.0
    span_mask_prob: float = 0.01
    vi2zh_epoch_ratio: float = 0.7
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_workers: int = 8
    save_dir: str = "./checkpoints_bidirectional"
    save_every: int = 10
    seed: int = 42


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning fine-tuning."""
    proj_dim: int = 768
    contrastive_tau: float = 0.07
    cross_lambda_max: float = 0.1
    cross_warmup_steps: int = 200
    lr_base: float = 5e-5
    warmup_steps: int = 200
    num_epochs: int = 20
    batch_size: int = 64
    save_dir: str = "./checkpoints_contrastive"
    save_every: int = 5
    num_workers: int = 2
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = 42
