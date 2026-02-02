"""Configuration for inference."""

from dataclasses import dataclass, field
import torch


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    # Model architecture
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    d_ff: int = 3072
    dropout: float = 0.01
    
    # Sequence parameters
    max_len: int = 32
    rope_base: float = 10000.0
    
    # Special tokens
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    zh_token: str = "<2zh>"
    vi_token: str = "<2vi>"
    
    # Tokenizer
    spm_prefix: str = ""
    vocab_size: int = 8000
    
    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Decoding parameters
    beam_size: int = 3
    top_k: int = 5
    length_penalty: float = 0.6
    
    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "InferenceConfig":
        """Create config from checkpoint."""
        config = cls()
        
        # Update from checkpoint config if available
        ckpt_config = checkpoint.get("config")
        if ckpt_config is not None:
            for field_name in config.__dataclass_fields__:
                if hasattr(ckpt_config, field_name):
                    setattr(config, field_name, getattr(ckpt_config, field_name))
        
        return config
