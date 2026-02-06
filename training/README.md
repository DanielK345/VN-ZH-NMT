# Training Pipeline for Vietnamese-Chinese Machine Translation

A modular PyTorch training pipeline for bidirectional Vietnamese-Chinese neural machine translation with curriculum learning and contrastive learning support.

## Architecture

The training pipeline is organized into the following modules:

### Core Modules

- **`config.py`**: Configuration dataclasses
  - `RopeConfig`: Main model training configuration
  - `ContrastiveConfig`: Contrastive learning fine-tuning configuration

- **`data_processor.py`**: Data loading and processing
  - Tokenizer training with SentencePiece
  - Corpus statistics
  - Token-length filtering
  - Data saving/loading utilities

- **`data_loader.py`**: PyTorch data pipeline
  - `BidirectionalTranslationDataset`: Custom dataset with span masking
  - `collate_fn`: Batch padding and preparation
  - `build_train_loader`: Dynamic loader with curriculum learning
  - `select_vi2zh_window`: Incremental coverage scheduling

- **`utils.py`**: Training utilities
  - `LabelSmoothedCrossEntropyLoss`: Custom loss with label smoothing
  - `WarmupInverseSqrtScheduler`: Learning rate scheduling
  - `greedy_decode`: Fast greedy decoding
  - `beam_search_decode`: Beam search with length penalty
  - `evaluate`: Validation with BLEU calculation

- **`trainer.py`**: Model trainers
  - `Trainer`: Standard translation model training
  - `ContrastiveTrainer`: Contrastive learning fine-tuning

- **`main.py`**: Main orchestration script
  - Full end-to-end pipeline
  - Data preparation, dataset creation, and training

## Features

### Data Processing
- Automatic tokenizer training on parallel corpus
- Token-length filtering to handle long sequences
- Span masking for data augmentation during training
- Language token prefixes for bidirectional training

### Model Training
- **RoPE Attention**: Rotary position embeddings for better extrapolation
- **Grouped Query Attention (GQA)**: Efficient multi-head attention variant
- **Label Smoothing**: Regularization technique for better generalization
- **Curriculum Learning**: Gradual increase of vi→zh training coverage per epoch
- **Warmup Learning Rate Scheduler**: Warm-up phase followed by inverse-sqrt decay
- **Gradient Clipping**: Stable gradient updates

### Evaluation
- BLEU score calculation for translation quality
- Validation loss tracking
- Greedy and beam search decoding
- Support for length penalty in beam search

### Checkpointing
- Periodic checkpoint saving every N epochs
- Best model selection based on validation loss
- State dict includes config and tokenizer for reproducibility
- **Metadata Recording**: Best model checkpoint automatically saves:
  - Training loss, validation loss, BLEU score
  - Learning rate at best epoch, scheduler state
- **Resume Support**: When resuming training, automatically:
  - Loads and displays previous training metrics
  - Restores best metrics to prevent incorrect updates
  - Supports `--epochs` and `--lr` overrides

## Usage

### Quick Start

```python
from training.main import main

# Run full pipeline
stats = main(
    base_dir=".",
    force_reprocess=False,  # Reuse cleaned data if exists
    skip_training=False,     # Run training
)
```

### Command Line

```bash
# Basic training
python -m training.main

# Force reprocess data
python -m training.main --force_reprocess

# Only process data, skip training
python -m training.main --skip_training

# Custom base directory
python -m training.main --base_dir /path/to/data

# Training with custom epochs
python -m training.main --epochs 50

# Training with custom learning rate
python -m training.main --lr 0.0001

# Training with both epochs and learning rate
python -m training.main --epochs 50 --lr 0.0001

# Resume from checkpoint
python -m training.main --resume_checkpoint checkpoints_bidirectional/best_model.pt

# Resume with additional epochs
python -m training.main --resume_checkpoint checkpoints_bidirectional/best_model.pt --epochs 50

# Resume with different learning rate
python -m training.main --resume_checkpoint checkpoints_bidirectional/best_model.pt --lr 0.0001

# Resume with both overrides
python -m training.main --resume_checkpoint checkpoints_bidirectional/best_model.pt --epochs 50 --lr 0.0001
```

**Note**: `--epochs` and `--lr` parameters can be used with any training mode (new training or resume)

### Custom Training

```python
import torch
from training.config import RopeConfig
from training.data_processor import DataProcessor
from training.data_loader import BidirectionalTranslationDataset
from training.trainer import Trainer
from model.transformer_model import TransformerModel

# Create config
config = RopeConfig(
    train_src_file="dataset/train/train.zh",
    train_tgt_file="dataset/train/train.vi",
    spm_prefix="tokenizer_train32/spm_zh_vi_joint",
    batch_size=128,
    num_epochs=40,
)

# Prepare data
processor = DataProcessor(config.spm_prefix, max_tokens=32)
src_lines, tgt_lines, sp_model = processor.load_lines(...), ...

# Create model
model = TransformerModel(config, vocab_size=8000)

# Train
trainer = Trainer(model, config)
trainer.train(train_dataset, valid_loader, sp_model)
```

## Configuration

### RopeConfig

Key parameters:

```python
RopeConfig(
    # Model architecture
    d_model=768,              # Embedding dimension
    n_heads=12,               # Number of attention heads
    n_kv_heads=4,             # Number of KV heads (GQA)
    num_encoder_layers=8,     # Encoder layers
    num_decoder_layers=8,     # Decoder layers
    d_ff=3072,                # Feed-forward dimension
    
    # Sequence parameters
    max_len=32,               # Maximum sequence length
    vocab_size=8000,          # Vocabulary size
    
    # Training parameters
    batch_size=128,           # Batch size
    num_epochs=40,            # Training epochs
    lr_base=2e-4,             # Base learning rate
    warmup_steps=200,         # LR warmup steps
    label_smoothing=0.01,     # Label smoothing factor
    grad_clip=1.0,            # Gradient clipping norm
    dropout=0.01,             # Dropout rate
    
    # Curriculum learning
    vi2zh_epoch_ratio=0.7,    # vi→zh coverage ratio per epoch
    span_mask_prob=0.01,      # Span masking probability
    
    # Saving
    save_dir="./checkpoints_bidirectional",
    save_every=10,            # Save checkpoint every N epochs
)
```

## Data Format

### Input Data

Expects parallel corpus files:
- Source (Chinese): `dataset/train/train.zh` (one sentence per line)
- Target (Vietnamese): `dataset/train/train.vi` (one sentence per line)

### Output Structure

```
clean_data/
  train_maxlen32.zh          # Filtered source
  train_maxlen32.vi          # Filtered target

checkpoints_bidirectional/
  best_model.pt              # Best checkpoint
  checkpoint_epoch_10.pt     # Periodic checkpoints
  checkpoint_epoch_20.pt
  ...

tokenizer_train32/
  spm_zh_vi_joint.model      # SentencePiece model
  spm_zh_vi_joint.vocab      # Vocabulary
```

## Training Dynamics

### Curriculum Learning
- **zh→vi (always on)**: All zh→vi pairs trained every epoch
- **vi→zh (graduated)**: Progressive increase in coverage
  - Epoch 1: ~70% of vi→zh pairs (cyclically selected)
  - Epoch 2: Next 70% window
  - Gradually covers all data with wraparound

### Span Masking
- 1% probability during training
- Masks 1-2 consecutive tokens
- Helps with robustness to input corruption

### Validation
- Runs every epoch
- Calculates loss and BLEU on validation set
- BLEU computed on subset (300 samples) for speed
- Best model selected based on validation loss

## Performance Metrics

The pipeline tracks:
- **Training Loss**: Cross-entropy with label smoothing
- **Validation Loss**: Metric for best model selection
- **BLEU Score**: Translation quality metric (optional, slower)
- **Learning Rate**: Scheduled learning rate per step

## Advanced Usage

### Contrastive Learning

```python
from training.config import ContrastiveConfig
from training.trainer import ContrastiveTrainer

cl_config = ContrastiveConfig(
    proj_dim=768,
    contrastive_tau=0.07,
    cross_lambda_max=0.1,
    num_epochs=20,
    batch_size=64,
)

cl_trainer = ContrastiveTrainer(model, projection, config, cl_config)
cl_trainer.train(dataloader, compute_crosslingual_loss_fn)
```

### Custom Decoding

```python
from training.utils import greedy_decode, beam_search_decode

# Fast greedy
predictions = greedy_decode(model, src_ids, sp_model, config)

# High-quality beam search
predictions = beam_search_decode(
    model, src_ids, sp_model, config,
    beam_size=5,
    length_penalty=0.6,
)
```

## Dependencies

- PyTorch >= 1.9
- sentencepiece
- numpy
- tqdm
- sacrebleu (optional, for BLEU calculation)

## Files

```
training/
├── __init__.py              # Package initialization
├── config.py                # Configuration classes
├── data_processor.py        # Data processing utilities
├── data_loader.py           # PyTorch data pipeline
├── trainer.py               # Model trainers
├── utils.py                 # Training utilities
├── main.py                  # Main orchestration script
└── README.md               # This file
```

## Notes

- The pipeline automatically handles GPU/CPU device selection
- Random seeds are set for reproducibility
- Tokenizer is trained once and reused
- Cleaned data is cached to avoid reprocessing
- All checkpoints include config and tokenizer for inference

## License

As per project repository
