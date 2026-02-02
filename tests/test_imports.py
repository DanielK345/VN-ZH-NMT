#!/usr/bin/env python
"""Test all imports to verify there are no circular import issues."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing model imports...\n")

try:
    print("1. Importing RMSNorm...")
    from model.layers.RMSNorm import RMSNorm
    print("   ✓ RMSNorm imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("2. Importing RoPE...")
    from model.layers.RoPE import RoPE, apply_rope
    print("   ✓ RoPE imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("3. Importing FFN_SwiGLU...")
    from model.layers.FFN import FFN_SwiGLU
    print("   ✓ FFN_SwiGLU imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("4. Importing GroupedQueryAttentionRoPE...")
    from model.layers.GQA_with_RoPE import GroupedQueryAttentionRoPE
    print("   ✓ GroupedQueryAttentionRoPE imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("5. Importing EncoderLayer...")
    from model.layers.Encoder import EncoderLayer
    print("   ✓ EncoderLayer imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("6. Importing DecoderLayer...")
    from model.layers.Decoder import DecoderLayer
    print("   ✓ DecoderLayer imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("7. Importing ProjectionHead...")
    from model.layers.Proj_Head import ProjectionHead
    print("   ✓ ProjectionHead imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("8. Importing TransformerModel...")
    from model.transformer_model import TransformerModel
    print("   ✓ TransformerModel imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
print("\nNow testing with training imports...")

try:
    print("\n9. Importing training.main components...")
    from training.config import RopeConfig
    from training.data_processor import DataProcessor
    from training.data_loader import BidirectionalTranslationDataset
    from training.trainer import Trainer
    from training.utils import WarmupInverseSqrtScheduler
    print("   ✓ All training imports successful")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n✅ All imports passed verification!")
