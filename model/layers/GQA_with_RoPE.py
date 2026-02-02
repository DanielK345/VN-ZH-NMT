import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.RoPE import RoPE, apply_rope

class GroupedQueryAttentionRoPE(nn.Module):
    def __init__(self,
                 d_model: int, n_heads: int, n_kv_heads: int,
                 dropout: float, rope_base: float
                 ):
        super().__init__()
        assert n_heads % n_kv_heads == 0
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

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        B, T_q = q.size(0), q.size(1)
        T_k = k.size(1)
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        cos_q, sin_q = self.rope(Q, T_q)
        cos_k, sin_k = self.rope(K, T_k)
        Q = apply_rope(Q, cos_q, sin_q)
        K = apply_rope(K, cos_k, sin_k)

        # Repeat KV heads
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.W_o(out)

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    d_model = 512
    n_heads = 8
    n_kv_heads = 2
    dropout = 0.1
    rope_base = 10000.0
    torch.manual_seed(0)
    device = torch.device("cpu")

    src_ids = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 0, 0]
    ]).to(device)
    
    # Demo
    gqa = GroupedQueryAttentionRoPE(d_model, n_heads, n_kv_heads, dropout, rope_base).to(device)

    x = torch.randn(batch_size, src_len, d_model).to(device)
    src_pad_mask = (src_ids == 0)

    print("INPUT:")
    print(f"  x shape: {x.shape}  # (batch, seq, d_model)")
    print(f"  padding_mask shape: {src_pad_mask.shape}  # (batch, seq)")
    print(f"\n  Config: {n_heads} Q heads, {n_kv_heads} KV heads")
    print(f"  → {n_heads // n_kv_heads} Q heads per KV head")

    attn_out = gqa(x, x, x, key_padding_mask=src_pad_mask)

    print("\nOUTPUT:")
    print(f"  attn_out shape: {attn_out.shape}  # (batch, seq, d_model)")
