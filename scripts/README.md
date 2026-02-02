# Orchestration Scripts

This directory contains bash scripts for managing the Vietnamese-Chinese Machine Translation system.

## Scripts Overview

### 🔧 `setup.sh` - Environment Setup
Initialize project environment and install dependencies.

```bash
bash scripts/setup.sh              # Standard setup
bash scripts/setup.sh --venv       # Setup with virtual environment
```

**What it does:**
- Checks Python version
- Upgrades pip
- Installs dependencies from `requirements.txt`
- Verifies installation of key packages

---

### 🏋️ `train.sh` - Model Training
Train the Vietnamese-Chinese translation model.

```bash
# Basic training (40 epochs, batch_size=128)
bash scripts/train.sh

# Data processing only (no training)
bash scripts/train.sh --skip-training

# Custom configuration
bash scripts/train.sh --epochs 60 --batch-size 256 --lr 0.0001

# Force reprocessing of data
bash scripts/train.sh --force-reprocess

# Resume from checkpoint
bash scripts/train.sh --resume checkpoints_bidirectional/checkpoint_20.pt
```

**Options:**
- `--skip-training`: Data processing only, skip model training
- `--force-reprocess`: Force reprocessing of all data (ignore cache)
- `--resume CHECKPOINT`: Resume training from checkpoint
- `--epochs NUM`: Number of epochs (default: 40)
- `--batch-size SIZE`: Batch size (default: 128)
- `--lr LEARNING_RATE`: Learning rate (default: 0.0002)
- `--help`: Show help message

**Output:**
- Model checkpoints: `checkpoints_bidirectional/`
- Tokenizer: `tokenizer_train32/`
- Processed data: `clean_data/`

---

### 🔮 `inference.sh` - Model Inference
Run translation inference on input data.

```bash
# Basic inference with beam search
bash scripts/inference.sh \
    --checkpoint checkpoints_bidirectional/best_model.pt \
    --input input.txt \
    --output results.csv

# Greedy decoding (faster)
bash scripts/inference.sh \
    --checkpoint best_model.pt \
    --input input.txt \
    --output results.csv \
    --greedy

# Custom beam search
bash scripts/inference.sh \
    --checkpoint best_model.pt \
    --input input.txt \
    --output results.csv \
    --beam-size 5 \
    --top-k 10 \
    --length-penalty 0.8
```

**Options:**
- `--checkpoint PATH`: Path to model checkpoint (required)
- `--input PATH`: Input file path (required)
- `--output PATH`: Output CSV file (required)
- `--beam-size NUM`: Beam search size (default: 3)
- `--top-k NUM`: Top-k filtering (default: 5)
- `--length-penalty FLOAT`: Length penalty (default: 0.6)
- `--greedy`: Use greedy decoding (sets beam-size=1)
- `--batch-size NUM`: Batch size (default: 32)
- `--device DEVICE`: `cuda` or `cpu` (default: cuda)
- `--help`: Show help message

**Input Format:**
```
One sentence per line in source language
你好，世界！
这是一个测试。
```

**Output Format:**
CSV with columns: `source,prediction`

---

### 🔄 `workflow.sh` - Complete Workflow Orchestration
Run complete workflow or specific stages.

```bash
# Complete workflow: setup + train + test inference
bash scripts/workflow.sh --all

# Setup only
bash scripts/workflow.sh --setup

# Training only
bash scripts/workflow.sh --train

# Data processing only
bash scripts/workflow.sh --data-only

# Test inference
bash scripts/workflow.sh --test-inference

# Setup and training
bash scripts/workflow.sh --setup --train
```

**Options:**
- `--all`: Run complete workflow
- `--setup`: Run setup script
- `--train`: Run training script
- `--data-only`: Process data only
- `--test-inference`: Run test inference with sample data
- `--help`: Show help message

---

### 📊 `evaluate.sh` - Model Evaluation
Evaluate model performance using BLEU scores.

```bash
# Evaluate using default checkpoint
bash scripts/evaluate.sh

# Evaluate specific checkpoint
bash scripts/evaluate.sh --checkpoint my_model.pt

# With test and reference files
bash scripts/evaluate.sh \
    --checkpoint best_model.pt \
    --test test.zh \
    --reference test.vi
```

**Options:**
- `--checkpoint PATH`: Model checkpoint (default: best_model.pt)
- `--test PATH`: Test file in source language
- `--reference PATH`: Reference file in target language
- `--batch-size NUM`: Batch size (default: 32)
- `--help`: Show help message

---

### 🛠️ `utils.sh` - Utility Functions
Provide project information and maintenance utilities.

```bash
# Show project information and statistics
bash scripts/utils.sh info

# Verify project structure
bash scripts/utils.sh verify

# Show code statistics
bash scripts/utils.sh stats

# Clean cache and temporary files
bash scripts/utils.sh clean

# Check Python environment
bash scripts/utils.sh check-env

# Show help
bash scripts/utils.sh help
```

**Commands:**
- `info`: Show project info and directory structure
- `verify`: Verify project structure completeness
- `stats`: Compute code and documentation statistics
- `clean`: Remove cache, bytecode, and temporary files
- `check-env`: Check Python environment and packages
- `help`: Show help message

---

## Common Workflows

### 1️⃣ First Time Setup
```bash
bash scripts/setup.sh
bash scripts/workflow.sh --data-only
```

### 2️⃣ Train Model from Scratch
```bash
bash scripts/train.sh
```

### 3️⃣ Quick Test (Data + Inference)
```bash
bash scripts/workflow.sh --test-inference
```

### 4️⃣ Custom Training
```bash
bash scripts/train.sh --epochs 60 --batch-size 256 --lr 0.0001
```

### 5️⃣ Batch Translation
```bash
bash scripts/inference.sh \
    --checkpoint checkpoints_bidirectional/best_model.pt \
    --input documents.txt \
    --output translations.csv \
    --beam-size 5
```

### 6️⃣ Project Maintenance
```bash
bash scripts/utils.sh info      # Check project status
bash scripts/utils.sh verify    # Verify structure
bash scripts/utils.sh stats     # Show statistics
bash scripts/utils.sh clean     # Clean cache
```

---

## Performance Tips

### Training
- Increase `--batch-size` for faster training (requires more GPU memory)
- Increase `--epochs` for better quality
- Monitor GPU memory: `watch -n 1 nvidia-smi`

### Inference
- Use `--greedy` for 3-5x faster inference
- Increase `--batch-size` for better throughput
- Use `--beam-size 3` for good quality-speed balance
- Use `--beam-size 5+` for highest quality

### System
- Run `bash scripts/utils.sh clean` periodically
- Check available disk space before training
- Monitor GPU/CPU usage with system tools

---

## Environment Variables

Optional environment variables:

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Limit CPU threads
export OMP_NUM_THREADS=4

# Disable warnings
export TF_CPP_MIN_LOG_LEVEL=2
```

---

## Troubleshooting

### Setup Issues
```bash
# Check Python version
python3 --version

# Check pip
pip3 --version

# Manually install dependencies
pip3 install -r requirements.txt
```

### Training Issues
```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Try CPU training
bash scripts/train.sh --device cpu
```

### Inference Issues
```bash
# Verify checkpoint exists
ls -lh checkpoints_bidirectional/

# Check input file format
head -5 input.txt

# Try with smaller batch size
bash scripts/inference.sh --checkpoint model.pt --input input.txt --output output.csv --batch-size 8
```

---

## Script Customization

### Add Custom Training Parameters
Edit `train.sh` and modify default values at the top:

```bash
NUM_EPOCHS=40       # Change default epochs
BATCH_SIZE=128      # Change default batch size
LR=0.0002          # Change default learning rate
```

### Add Custom Inference Parameters
Edit `inference.sh` and modify defaults:

```bash
BEAM_SIZE=3        # Change default beam size
TOP_K=5            # Change default top-k
LENGTH_PENALTY=0.6 # Change default penalty
```

---

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Setup
  run: bash scripts/setup.sh

- name: Process Data
  run: bash scripts/train.sh --skip-training

- name: Train Model
  run: bash scripts/train.sh --epochs 20

- name: Test Inference
  run: bash scripts/workflow.sh --test-inference
```

---

## Requirements

- Bash 4.0+
- Python 3.8+
- Git (for version control)
- NVIDIA CUDA (optional, for GPU support)

---

## Support

For issues or questions:
1. Check [README.md](../README.md)
2. Review [QUICKSTART.py](../QUICKSTART.py)
3. Run `bash scripts/utils.sh verify`
4. Check error messages in console output

---

**Last Updated:** February 2, 2026
**Version:** 1.0.0
