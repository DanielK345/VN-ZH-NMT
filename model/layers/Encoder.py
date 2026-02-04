import torch
import torch.nn as nn
from model.layers.FFN import FFN_SwiGLU
from model.layers.GQA_with_RoPE import GroupedQueryAttentionRoPE
from model.layers.RMSNorm import RMSNorm

class EncoderLayer(nn.Module):
    def __init__(self, config_or_d_model, n_heads=None, n_kv_heads=None, d_ff=None, dropout=None, rope_base=None):
        """
        Flexible constructor that accepts either:
          - (config: RopeConfig)  -- notebook style
          - (d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base)  -- arg style
        """
        super().__init__()
        
        # Determine construction mode
        if hasattr(config_or_d_model, 'd_model'):
            # Config-style (notebook)
            config = config_or_d_model
            d_model = config.d_model
            n_heads = config.n_heads
            n_kv_heads = config.n_kv_heads
            d_ff = config.d_ff
            dropout = config.dropout
            rope_base = config.rope_base
        else:
            # Arg-style (backward compat)
            d_model = config_or_d_model
        
        self.ln1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttentionRoPE(
            d_model, n_heads, n_kv_heads, dropout, rope_base
            )
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_pad_mask=None):
        x1 = self.ln1(x)
        attn = self.self_attn(x1, x1, x1, key_padding_mask=src_pad_mask)
        x = x + self.dropout(attn)
        x2 = self.ln2(x)
        return x + self.ffn(x2)

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    d_model = 512
    n_heads = 8
    n_kv_heads = 4
    d_ff = 2048
    dropout = 0.1
    rope_base = 10000.0
    torch.manual_seed(0)
    device = torch.device("cpu")

    src_ids = torch.tensor([
        [5, 7, 9, 0],
        [6, 8, 0, 0]
        ]).to(device)
    
    enc_layer = EncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base).to(device)

    x = torch.randn(batch_size, src_len, d_model).to(device)
    src_pad_mask = (src_ids == 0)

    print("INPUT:")
    print(f"  x shape: {x.shape}")
    print(f"  src_pad_mask shape: {src_pad_mask.shape}")

    enc_out = enc_layer(x, src_pad_mask)

    print("\nOUTPUT:")
    print(f"  enc_out shape: {enc_out.shape}  # Same shape")
    print(f"\n  → Stack 8 layers như này để tạo Encoder đầy đủ")
