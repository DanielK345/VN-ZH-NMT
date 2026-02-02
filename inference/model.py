"""Model loading and reconstruction from checkpoints."""

import os
import math
import tempfile
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from .config import InferenceConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class RoPE(nn.Module):
    """Rotary Position Embeddings."""
    
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values."""
        needs_refresh = (
            self._cos_cached is None
            or self._sin_cached is None
            or seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if needs_refresh:
            self._seq_len_cached = seq_len
            position = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(position, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = seq_len or x.size(-2)
        self._update(seq_len, x.device, x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)


class FFN_SwiGLU(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        g, v = h[..., :self.d_ff], h[..., self.d_ff:]
        s = g * torch.sigmoid(g)
        hidden = s * v
        return self.dropout(self.linear2(hidden))


class GroupedQueryAttentionRoPE(nn.Module):
    """Grouped Query Attention with RoPE."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.rope = RoPE(self.d_k, base=rope_base)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T_q, T_k = q.size(0), q.size(1), k.size(1)
        
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)
        
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        
        return self.W_o(out)


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            config.d_model,
            config.n_heads,
            config.n_kv_heads,
            config.dropout,
            config.rope_base,
        )
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FFN_SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, src_pad_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.self_attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            key_padding_mask=src_pad_mask
        )
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        return x + ffn_out


class DecoderLayer(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            config.d_model,
            config.n_heads,
            config.n_kv_heads,
            config.dropout,
            config.rope_base,
        )
        self.ln2 = RMSNorm(config.d_model)
        self.cross_attn = GroupedQueryAttentionRoPE(
            config.d_model,
            config.n_heads,
            config.n_kv_heads,
            config.dropout,
            config.rope_base,
        )
        self.ln3 = RMSNorm(config.d_model)
        self.ffn = FFN_SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        y: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_pad_mask: torch.Tensor = None,
        tgt_causal_mask: torch.Tensor = None,
        src_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        y = y + self.dropout(
            self.self_attn(
                self.ln1(y), self.ln1(y), self.ln1(y),
                key_padding_mask=tgt_pad_mask,
                attn_mask=tgt_causal_mask,
            )
        )
        y = y + self.dropout(
            self.cross_attn(
                self.ln2(y), enc_out, enc_out,
                key_padding_mask=src_pad_mask
            )
        )
        y = y + self.ffn(self.ln3(y))
        return y


class TransformerInference(nn.Module):
    """Transformer model for inference."""
    
    def __init__(self, config: InferenceConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config)
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_final_ln = RMSNorm(config.d_model)
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_final_ln = RMSNorm(config.d_model)
        
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.emb_scale = math.sqrt(config.d_model)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode source sequence."""
        x = self.emb_dropout(self.embedding(src_ids) * self.emb_scale)
        for layer in self.encoder_layers:
            x = layer(x, src_pad_mask)
        return self.encoder_final_ln(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_pad_mask: torch.Tensor = None,
        tgt_causal_mask: torch.Tensor = None,
        src_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode target sequence."""
        x = self.emb_dropout(self.embedding(tgt_ids) * self.emb_scale)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)
        return self.decoder_final_ln(x)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project to vocabulary logits."""
        return F.linear(hidden, self.embedding.weight, self.output_bias)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training/evaluation."""
        src_pad = (src_ids == 0)
        tgt_pad = (tgt_ids == 0)
        tgt_in = tgt_ids[:, :-1]
        tgt_pad_in = tgt_pad[:, :-1]
        
        T = tgt_in.size(1)
        tgt_causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=src_ids.device),
            diagonal=1
        )
        
        enc_out = self.encode(src_ids, src_pad)
        dec_out = self.decode(tgt_in, enc_out, tgt_pad_in, tgt_causal, src_pad)
        return self.project(dec_out)


def materialize_tokenizer(tokenizer_payload: dict) -> Tuple[str, str]:
    """
    Materialize tokenizer from checkpoint bytes.
    
    Args:
        tokenizer_payload: Dictionary with 'model_bytes' and 'vocab_bytes'
        
    Returns:
        Tuple of (tokenizer_prefix, temp_dir)
    """
    if not tokenizer_payload:
        raise ValueError("Tokenizer payload missing or empty in checkpoint.")
    
    model_bytes = tokenizer_payload.get("model_bytes")
    vocab_bytes = tokenizer_payload.get("vocab_bytes")
    
    if model_bytes is None or vocab_bytes is None:
        raise ValueError("Tokenizer payload lacks model or vocab bytes.")
    
    tmp_dir = tempfile.mkdtemp(prefix="spm_from_checkpoint_")
    base_name = os.path.basename(tokenizer_payload.get("prefix", "spm_from_ckpt")) or "spm_from_ckpt"
    prefix = os.path.join(tmp_dir, base_name)
    
    with open(f"{prefix}.model", "wb") as f:
        f.write(model_bytes)
    with open(f"{prefix}.vocab", "wb") as f:
        f.write(vocab_bytes)
    
    return prefix, tmp_dir


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[TransformerInference, spm.SentencePieceProcessor, InferenceConfig]:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    config = InferenceConfig.from_checkpoint(checkpoint)
    config.device = device
    
    # Extract and materialize tokenizer
    if "tokenizer" not in checkpoint:
        raise ValueError("Checkpoint must include tokenizer bytes.")
    
    tokenizer_prefix, _ = materialize_tokenizer(checkpoint["tokenizer"])
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f"{tokenizer_prefix}.model")
    vocab_size = sp_model.GetPieceSize()
    config.spm_prefix = tokenizer_prefix
    config.vocab_size = vocab_size
    
    # Create and load model
    model = TransformerInference(config, vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    return model, sp_model, config
