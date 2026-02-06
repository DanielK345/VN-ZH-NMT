#!/usr/bin/env python3
"""Upload trained translation model to Hugging Face Hub."""

import os
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from huggingface_hub import HfApi, login, logout, get_user_access_token
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Error: huggingface_hub not installed. Install with:")
    print("  pip install huggingface-hub")
    exit(1)


def load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load metadata from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    metadata = {
        "epoch": checkpoint.get("epoch", "N/A"),
        "train_loss": checkpoint.get("train_loss", "N/A"),
        "val_loss": checkpoint.get("val_loss", "N/A"),
        "val_bleu": checkpoint.get("val_bleu", "N/A"),
    }
    
    # Extract metadata if available
    if "metadata" in checkpoint:
        metadata.update(checkpoint["metadata"])
    
    return metadata


def create_model_card(
    checkpoint_path: str,
    repo_name: str,
    model_type: str = "Vietnamese-Chinese Translation",
    description: str = None,
) -> str:
    """
    Create a model card (README.md) for Hugging Face.
    
    Args:
        checkpoint_path: Path to checkpoint
        repo_name: Repository name
        model_type: Type of model
        description: Custom description
        
    Returns:
        Model card content as string
    """
    metadata = load_checkpoint_metadata(checkpoint_path)
    
    model_card = f"""---
license: mit
task_ids:
  - machine-translation
language:
  - zh
  - vi
---

# {repo_name}

A bidirectional Vietnamese-Chinese Neural Machine Translation model trained with curriculum learning and RoPE attention.

## Model Details

- **Model Type**: {model_type}
- **Architecture**: Transformer with Grouped Query Attention (GQA) and RoPE
- **Language Pair**: Vietnamese ↔ Chinese (Simplified)
- **Training Framework**: PyTorch
- **License**: MIT

## Training Metrics

- **Training Loss**: {metadata.get('train_loss', 'N/A')}
- **Validation Loss**: {metadata.get('val_loss', 'N/A')}
- **BLEU Score**: {metadata.get('val_bleu', 'N/A')}
- **Epoch**: {metadata.get('epoch', 'N/A')}
- **Learning Rate**: {metadata.get('lr', 'N/A')}

## Usage

```python
import torch
from model.transformer_model import TransformerModel

# Load checkpoint
checkpoint = torch.load('model.pt', map_location='cuda')
model = TransformerModel(checkpoint['config'], vocab_size=8000)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference (tokenization and decoding depend on SentencePiece)
# See training/utils.py for greedy_decode and beam_search_decode
```

## Model Architecture

### Key Components
- **Encoder Layers**: 8
- **Decoder Layers**: 8
- **Hidden Dimension**: 768
- **Attention Heads**: 12
- **KV Heads (GQA)**: 4
- **Feed-Forward Dimension**: 3072
- **Max Sequence Length**: 32
- **Vocabulary Size**: 8000

### Special Tokens
- `<pad>`: Padding token
- `<s>`: Begin-of-sequence
- `</s>`: End-of-sequence
- `<2zh>`: Prefix for Vietnamese→Chinese
- `<2vi>`: Prefix for Chinese→Vietnamese

## Training Details

### Data
- **Source**: Vietnamese-Chinese parallel corpus
- **Train/Val Split**: 95%/5%
- **Token-length Filtering**: Max 32 tokens

### Curriculum Learning
- **zh→vi**: Full coverage every epoch
- **vi→zh**: Progressive window covering 70% per epoch
- **Span Masking**: 1% probability for robustness

### Optimization
- **Optimizer**: AdamW (β₁=0.9, β₂=0.98, ε=1e-9)
- **Learning Rate Schedule**: Warmup + Inverse Sqrt Decay
- **Warmup Steps**: 200
- **Batch Size**: 128
- **Label Smoothing**: 0.01
- **Gradient Clipping**: 1.0

## Citation

If you use this model, please cite:

```bibtex
@misc{{vn_zh_nmt,
  title={{Vietnamese-Chinese Neural Machine Translation}},
  author={{Duy}},
  year={{2026}}
}}
```

## License

This project is licensed under the MIT License.
"""
    
    if description:
        model_card = model_card.replace(
            "A bidirectional Vietnamese-Chinese Neural Machine Translation model trained with curriculum learning and RoPE attention.",
            description
        )
    
    return model_card


def upload_model(
    checkpoint_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    description: str = None,
) -> bool:
    """
    Upload model checkpoint to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        repo_id: Repository ID (format: username/repo-name)
        token: Hugging Face API token (uses HF_TOKEN env var if not provided)
        private: Whether to make repo private
        description: Custom model description
        
    Returns:
        True if successful, False otherwise
    """
    # Get token from argument or environment
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("⚠️  No Hugging Face token provided.")
        print("Set HF_TOKEN environment variable or pass --token argument")
        return False
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024 ** 2)
    print(f"\n📦 Model Details:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Size: {checkpoint_size:.2f} MB")
    print(f"  Repository: {repo_id}")
    print(f"  Private: {private}\n")
    
    try:
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        print("🔍 Checking repository...")
        try:
            repo_url = api.repo_exists(repo_id=repo_id, repo_type="model")
            if not repo_url:
                print(f"📝 Creating repository: {repo_id}")
                repo_url = api.create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True
                )
                print(f"✓ Repository created: {repo_url}")
        except RepositoryNotFoundError:
            print(f"📝 Creating repository: {repo_id}")
            repo_url = api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True
            )
            print(f"✓ Repository created: {repo_url}")
        
        # Create model card
        print("\n📄 Creating model card...")
        model_card = create_model_card(
            checkpoint_path,
            repo_id.split('/')[-1],
            description=description
        )
        
        # Upload model checkpoint
        print(f"\n⬆️  Uploading checkpoint...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="model.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✓ Checkpoint uploaded as model.pt")
        
        # Upload model card
        print(f"⬆️  Uploading model card...")
        api.upload_file(
            path_or_fileobj=model_card.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✓ Model card uploaded as README.md")
        
        # Upload metadata
        print(f"⬆️  Uploading metadata...")
        metadata = load_checkpoint_metadata(checkpoint_path)
        metadata_json = json.dumps(metadata, indent=2, default=str)
        api.upload_file(
            path_or_fileobj=metadata_json.encode('utf-8'),
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✓ Metadata uploaded as metadata.json")
        
        print(f"\n✅ Model uploaded successfully!")
        print(f"🔗 Repository: https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Upload Vietnamese-Chinese translation model to Hugging Face Hub"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (format: username/repo-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (uses HF_TOKEN env var if not provided)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Custom model description"
    )
    
    args = parser.parse_args()
    
    success = upload_model(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        description=args.description,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
