"""Main training script orchestrating the full training pipeline."""

import os
import random
import numpy as np
import torch
import sentencepiece as spm
from pathlib import Path
from typing import Dict, Any

from training.config import RopeConfig, ContrastiveConfig
from training.data_processor import DataProcessor
from training.data_loader import BidirectionalTranslationDataset, collate_fn
from training.trainer import Trainer, ContrastiveTrainer
from training.utils import WarmupInverseSqrtScheduler
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def package_tokenizer(prefix: str) -> Dict[str, Any]:
    """Package tokenizer files."""
    model_path = f"{prefix}.model"
    vocab_path = f"{prefix}.vocab"
    if not (os.path.isfile(model_path) and os.path.isfile(vocab_path)):
        raise FileNotFoundError(f"Missing tokenizer files at {prefix}.[model|vocab]")
    
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    with open(vocab_path, "rb") as f:
        vocab_bytes = f.read()
    
    return {"prefix": prefix, "model_bytes": model_bytes, "vocab_bytes": vocab_bytes}


def prepare_data(
    config: RopeConfig,
    force_reprocess: bool = False,
) -> tuple:
    """
    Prepare and process data.
    
    Args:
        config: Configuration object
        force_reprocess: Force reprocessing even if cleaned data exists
        
    Returns:
        Tuple of (src_lines, tgt_lines, sp_model)
    """
    processor = DataProcessor(config.spm_prefix, max_tokens=config.max_len)
    
    # Check if cleaned data already exists
    clean_src = os.path.join(config.train_back_dir, f"train_maxlen{config.max_len}.zh")
    clean_tgt = os.path.join(config.train_back_dir, f"train_maxlen{config.max_len}.vi")
    
    if os.path.exists(clean_src) and os.path.exists(clean_tgt) and not force_reprocess:
        print("✓ Found cleaned data, loading...")
        src_lines = processor.load_lines(clean_src)
        tgt_lines = processor.load_lines(clean_tgt)
        sp_model = processor.load_tokenizer()
    else:
        print("Processing raw data...")
        
        # Load raw data
        src_raw = processor.load_lines(config.train_src_file)
        tgt_raw = processor.load_lines(config.train_tgt_file)
        
        # Load or train tokenizer
        tokenizer_model = f"{config.spm_prefix}.model"
        if not os.path.exists(tokenizer_model):
            print("Training tokenizer...")
            processor.train_tokenizer(
                config.train_src_file,
                config.train_tgt_file,
                vocab_size=config.vocab_size,
                user_symbols=[config.zh_token, config.vi_token],
            )
        else:
            processor.load_tokenizer()
        
        # Print statistics before filtering
        stats = processor.get_statistics(src_raw, tgt_raw)
        print(f"\nBefore filtering:")
        print(f"  ZH: mean={stats['src']['mean']:.2f}, min={stats['src']['min']}, max={stats['src']['max']}")
        print(f"  VI: mean={stats['tgt']['mean']:.2f}, min={stats['tgt']['min']}, max={stats['tgt']['max']}")
        
        # Filter by token length
        src_lines, tgt_lines = processor.filter_by_token_length(
            src_raw,
            tgt_raw,
            src_prefix=config.vi_token,
        )
        
        # Save cleaned data
        processor.save_cleaned_data(src_lines, tgt_lines, config.train_back_dir)
        
        sp_model = processor.sp
    
    return src_lines, tgt_lines, sp_model


def prepare_datasets(
    src_lines: list,
    tgt_lines: list,
    sp_model: spm.SentencePieceProcessor,
    config: RopeConfig,
    val_split: float = 0.05,
) -> tuple:
    """
    Prepare train and validation datasets.
    
    Args:
        src_lines: Source sentences
        tgt_lines: Target sentences
        sp_model: SentencePiece model
        config: Configuration
        val_split: Validation split ratio
        
    Returns:
        Tuple of (train_dataset, train_loader, valid_dataset, valid_loader)
    """
    # Split data
    split_idx = int((1 - val_split) * len(src_lines))
    if split_idx % 2 != 0:
        split_idx -= 1
    split_idx = max(split_idx, 0)
    
    train_src = src_lines[:split_idx]
    train_tgt = tgt_lines[:split_idx]
    
    valid_src_all = src_lines[split_idx:]
    valid_tgt_all = tgt_lines[split_idx:]
    
    # Use every other sample for validation
    valid_src = [valid_src_all[i] for i in range(0, len(valid_src_all), 2)]
    valid_tgt = [valid_tgt_all[i] for i in range(0, len(valid_tgt_all), 2)]
    
    # Create datasets
    train_dataset = BidirectionalTranslationDataset(
        train_src, train_tgt, sp_model, config, is_training=True
    )
    valid_dataset = BidirectionalTranslationDataset(
        valid_src, valid_tgt, sp_model, config, is_training=False
    )
    
    # Create dataloaders
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    print(f"Dataset ready:")
    print(f"  Train: {len(train_dataset)} samples "
          f"(zh2vi: {len(train_dataset.zh2vi_indices)}, "
          f"vi2zh: {len(train_dataset.vi2zh_indices)})")
    print(f"  Valid: {len(valid_dataset)} samples")
    print(f"  Valid batches: {len(valid_loader)}")
    
    return train_dataset, valid_loader


def train_translation_model(
    model,
    config: RopeConfig,
    sp_model: spm.SentencePieceProcessor,
    train_dataset: BidirectionalTranslationDataset,
    valid_loader: DataLoader,
    tokenizer_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train translation model.
    
    Args:
        model: Transformer model
        config: Configuration
        sp_model: SentencePiece model
        train_dataset: Training dataset
        valid_loader: Validation dataloader
        tokenizer_payload: Tokenizer payload
        
    Returns:
        Training statistics
    """
    print("\n" + "="*80)
    print("Starting Translation Model Training")
    print("="*80 + "\n")
    
    trainer = Trainer(model, config, tokenizer_payload)
    
    stats = trainer.train(train_dataset, valid_loader, sp_model)
    
    return stats


def main(
    base_dir: str = ".",
    force_reprocess: bool = False,
    skip_training: bool = False,
):
    """
    Main training pipeline.
    
    Args:
        base_dir: Base directory for data and checkpoints
        force_reprocess: Force reprocessing of data
        skip_training: Skip training and only process data
    """
    # Set random seeds
    set_seed(42)
    
    # Setup paths
    train_dir = os.path.join(base_dir, "dataset", "train")
    src_file = os.path.join(train_dir, "train.zh")
    tgt_file = os.path.join(train_dir, "train.vi")
    tokenizer_dir = os.path.join(base_dir, "tokenizer_train32")
    clean_data_dir = os.path.join(base_dir, "clean_data")
    
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(clean_data_dir, exist_ok=True)
    
    # Configure
    config = RopeConfig(
        train_src_file=src_file,
        train_tgt_file=tgt_file,
        spm_prefix=os.path.join(tokenizer_dir, "spm_zh_vi_joint"),
        train_back_dir=clean_data_dir,
        max_len=32,
        vocab_size=8000,
        d_model=768,
        n_heads=12,
        n_kv_heads=4,
        num_encoder_layers=8,
        num_decoder_layers=8,
        d_ff=3072,
        batch_size=128,
        num_epochs=40,
        save_dir=os.path.join(base_dir, "checkpoints_bidirectional"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    print(f"Config device: {config.device}")
    print(f"Vocab size will be: {config.vocab_size}")
    print(f"Max sequence length: {config.max_len}")
    
    # Step 1: Prepare data
    print("\n" + "="*80)
    print("Step 1: Data Processing")
    print("="*80 + "\n")
    
    src_lines, tgt_lines, sp_model = prepare_data(config, force_reprocess)
    vocab_size = sp_model.GetPieceSize()
    tokenizer_payload = package_tokenizer(config.spm_prefix)
    
    print(f"✓ Tokenizer loaded, vocab_size: {vocab_size}")
    
    if skip_training:
        print("\nData processing complete. Skipping training.")
        return
    
    # Step 2: Prepare datasets
    print("\n" + "="*80)
    print("Step 2: Dataset Preparation")
    print("="*80 + "\n")
    
    train_dataset, valid_loader = prepare_datasets(
        src_lines, tgt_lines, sp_model, config, val_split=0.05
    )
    
    # Step 3: Train model
    print("\n" + "="*80)
    print("Step 3: Model Training")
    print("="*80 + "\n")
    
    # Import model here to avoid circular imports
    from model.transformer_model import TransformerModel
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        rope_base=config.rope_base,
        num_enc_layers=config.num_encoder_layers,
        num_dec_layers=config.num_decoder_layers,
    ).to(config.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    train_stats = train_translation_model(
        model, config, sp_model, train_dataset, valid_loader, tokenizer_payload
    )
    
    print("\n" + "="*80)
    print("Training Pipeline Complete")
    print("="*80)
    print(f"Best validation loss: {train_stats['best_val_loss']:.4f}")
    print(f"Best BLEU score: {train_stats['best_val_bleu']:.2f}")
    print(f"Checkpoints saved to: {train_stats['save_dir']}")
    
    return train_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Vietnamese-Chinese translation model")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory")
    parser.add_argument("--force_reprocess", action="store_true", help="Force data reprocessing")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    main(
        base_dir=args.base_dir,
        force_reprocess=args.force_reprocess,
        skip_training=args.skip_training,
    )
