import torch
import torch.nn as nn
from typing import Optional

class RoPE(nn.Module):
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._cos = None
        self._sin = None
        self._seq_len_cached = 0

    def _maybe_update_cache(self, seq_len: int, device, dtype):
        if seq_len > self._seq_len_cached or self._cos is None or self._cos.device != device:
            self._seq_len_cached = seq_len
            positions = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            self._cos = freqs.cos()
            self._sin = freqs.sin()

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        seq_len = seq_len or x.size(-2)
        self._maybe_update_cache(seq_len, x.device, x.dtype)
        return self._cos[:seq_len], self._sin[:seq_len]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Áp dụng RoPE lên tensor (batch, heads, seq, d_k)"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    d_model = 512
    n_heads = 8
    rope_base = 10000.0
    torch.manual_seed(0)
    device = torch.device("cpu")
    d_k = d_model // n_heads  # 64
    rope = RoPE(d_k, base=rope_base).to(device)

    # Input tensor
    Q = torch.randn(batch_size, n_heads, src_len, d_k).to(device)

    print("INPUT:")
    print(f"  Q shape: {Q.shape}  # (batch, heads, seq, d_k)")

    cos, sin = rope(Q, src_len)
    Q_rope = apply_rope(Q, cos, sin)

    print("\nOUTPUT:")
    print(f"  cos shape: {cos.shape}  # (seq, d_k/2)")
    print(f"  sin shape: {sin.shape}  # (seq, d_k/2)")
    print(f"  Q_rope shape: {Q_rope.shape}  # Same as input")
    print(f"  RoPE base: {rope_base}")
