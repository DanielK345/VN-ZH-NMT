"""Quick reference guide for getting started."""

# QUICK START GUIDE
# ==================

# 1. INSTALLATION
# ===============
# pip install -r requirements.txt

# 2. TRAINING
# ===========

# From Python:
from training.main import main

stats = main(
    base_dir=".",
    force_reprocess=False,
    skip_training=False
)

# From command line:
# python -m training.main

# With options:
# python -m training.main --force_reprocess


# 3. INFERENCE
# ============

# Simple usage:
from inference import Translator

translator = Translator("checkpoints_bidirectional/best_model.pt")

# Single sentence
result = translator.translate_sentence(
    "这是一个例子。",
    src_lang="zh"
)
print(result)

# Batch sentences
results = translator.translate_batch(
    ["今天天气很好", "我喜欢学习"],
    src_lang="zh",
    batch_size=32
)

# From file
df = translator.translate_file(
    "input.txt",
    "output.csv",
    src_lang="zh",
    use_beam_search=True,
    batch_size=64
)

# From command line:
# python -m inference.main \
#     --checkpoint checkpoints_bidirectional/best_model.pt \
#     --input input.txt \
#     --output output.csv \
#     --beam_size 5


# 4. CONFIGURATION
# ================

from training.config import RopeConfig
from inference.config import InferenceConfig

# Training
train_config = RopeConfig(
    train_src_file="clean_data/train_maxlen32.zh",
    train_tgt_file="clean_data/train_maxlen32.vi",
    spm_prefix="tokenizer_train32/spm_zh_vi_joint",
    batch_size=128,
    num_epochs=40,
    lr_base=2e-4,
)

# Inference
infer_config = InferenceConfig(
    beam_size=3,
    top_k=5,
    length_penalty=0.6,
    max_len=32,
)


# 5. DECODING STRATEGIES
# ======================

from inference.decoder import greedy_decode, beam_search_decode

# Fast greedy decoding
translations = greedy_decode(
    model, src_ids, sp_model, config
)

# High-quality beam search
translations = beam_search_decode(
    model, src_ids, sp_model, config,
    beam_size=5,
    top_k=10,
    length_penalty=0.6
)


# 6. DATA PROCESSING
# ==================

from training.data_processor import DataProcessor

processor = DataProcessor(
    "tokenizer_train32/spm_zh_vi_joint",
    max_tokens=32
)

# Train tokenizer
processor.train_tokenizer(
    "dataset/train/train.zh",
    "dataset/train/train.vi",
    vocab_size=8000,
)

# Filter data
src, tgt = processor.filter_by_token_length(
    src_lines, tgt_lines
)

# Save cleaned data
processor.save_cleaned_data(src, tgt, "clean_data")


# 7. MODELS AND LAYERS
# ====================

from inference.model import TransformerInference, load_model_from_checkpoint
from model.transformer_model import TransformerModel

# Load from checkpoint
model, sp_model, config = load_model_from_checkpoint(
    "checkpoints_bidirectional/best_model.pt"
)

# Create new model
import torch
from training.config import RopeConfig

config = RopeConfig(...)
model = TransformerModel(config, vocab_size=8000)
model = model.to(config.device)


# 8. TRAINING LOOP
# ================

from training.trainer import Trainer

trainer = Trainer(model, config, tokenizer_payload)

# Train
stats = trainer.train(train_dataset, valid_loader, sp_model)

# Save checkpoint
trainer.save_checkpoint(epoch=40, train_loss=2.1, val_loss=2.5, val_bleu=28.5)

# Load checkpoint
checkpoint = trainer.load_checkpoint("checkpoints_bidirectional/best_model.pt")


# 9. EVALUATION
# =============

from training.utils import evaluate

val_loss, bleu = evaluate(
    model, valid_loader, criterion, sp_model, config,
    calculate_bleu=True,
    max_bleu_samples=500
)

print(f"Validation Loss: {val_loss:.4f}")
print(f"BLEU Score: {bleu:.2f}")


# 10. HARDWARE REQUIREMENTS
# ==========================

# Minimum:
# - GPU: 8GB VRAM
# - RAM: 16GB
# - Storage: 10GB

# Recommended:
# - GPU: 16GB+ VRAM (A100, RTX3090, etc.)
# - RAM: 32GB
# - Storage: 50GB

# CPU training: Possible but slow (~10x slower than GPU)


# 11. TROUBLESHOOTING
# ===================

# Out of Memory:
config.batch_size = 32  # Reduce batch size

# Training loss not decreasing:
config.lr_base = 5e-4  # Increase learning rate
config.warmup_steps = 100  # Reduce warmup

# Poor quality:
# - Use beam_search (higher quality)
# - Increase beam_size (3 → 5)
# - Adjust length_penalty (0.4 - 0.8)
# - Train more epochs

# Slow inference:
# - Use greedy_decode (faster)
# - Reduce batch_size or max_len
# - Use GPU instead of CPU


# 12. PROJECT STRUCTURE
# =====================

"""
VN-CN-Machine_Translation/
├── training/           # Training pipeline
├── inference/          # Inference pipeline  
├── model/              # Model architecture
├── notebooks/          # Jupyter notebooks
├── dataset/            # Training data
├── checkpoints_bidirectional/    # Model checkpoints
├── clean_data/         # Processed data
├── tokenizer_train32/  # SentencePiece tokenizer
├── requirements.txt    # Dependencies
└── README.md           # Full documentation
"""


# 13. FILE I/O
# ============

import os
import torch
import sentencepiece as spm

# Save checkpoint
torch.save({
    'epoch': 40,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Save tokenizer
sp_model = spm.SentencePieceProcessor()
sp_model.Load('tokenizer_train32/spm_zh_vi_joint.model')

# Read data
with open('data.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f]


# 14. COMMON PARAMETERS
# =====================

# Model sizes (preset configurations)
SMALL = {
    'd_model': 256,
    'n_heads': 4,
    'n_kv_heads': 2,
    'num_layers': 4,
    'd_ff': 1024,
}

MEDIUM = {
    'd_model': 512,
    'n_heads': 8,
    'n_kv_heads': 4,
    'num_layers': 6,
    'd_ff': 2048,
}

BASE = {  # Default
    'd_model': 768,
    'n_heads': 12,
    'n_kv_heads': 4,
    'num_layers': 8,
    'd_ff': 3072,
}

LARGE = {
    'd_model': 1024,
    'n_heads': 16,
    'n_kv_heads': 8,
    'num_layers': 12,
    'd_ff': 4096,
}

# Training hyperparameters
WARMUP_STEPS = 200
LR_BASE = 2e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.01

# Beam search
BEAM_SIZE = 3
TOP_K = 5
LENGTH_PENALTY = 0.6

print("✓ Quick reference loaded successfully!")
