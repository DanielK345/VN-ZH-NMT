import torch
import torch.nn as nn
from model.layers.RMSNorm import RMSNorm
from model.layers.FFN import FFN_SwiGLU
from model.layers.GQA_with_RoPE import GroupedQueryAttentionRoPE

class DecoderLayer(nn.Module):
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
            d_model, n_heads, n_kv_heads, dropout, rope_base)
        self.ln2 = RMSNorm(d_model)
        self.cross_attn = GroupedQueryAttentionRoPE(
            d_model, n_heads, n_kv_heads, dropout, rope_base)
        self.ln3 = RMSNorm(d_model)
        self.ffn = FFN_SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, enc_out, tgt_pad_mask=None, tgt_causal_mask=None, src_pad_mask=None):
        # Self-attention (masked)
        y1 = self.ln1(y)
        self_attn = self.self_attn(
            y1, y1, y1, key_padding_mask=tgt_pad_mask, attn_mask=tgt_causal_mask)
        y = y + self.dropout(self_attn)

        # Cross-attention
        y2 = self.ln2(y)
        cross_attn = self.cross_attn(y2, enc_out, enc_out, key_padding_mask=src_pad_mask)
        y = y + self.dropout(cross_attn)

        # FFN
        y3 = self.ln3(y)
        return y + self.ffn(y3)

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    tgt_len = 3
    d_model = 512
    n_heads = 8
    n_kv_heads = 4
    d_ff = 2048
    dropout = 0.1
    rope_base = 10000.0
    vocab_size = 10000
    torch.manual_seed(0)
    device = torch.device("cpu")
    src_ids = torch.tensor([
        [5, 7, 9, 0],
        [6, 8, 0, 0]
        ]).to(device)
    
    dec_layer = DecoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base).to(device)

    # Giả sử đã có encoder output và decoder input
    enc_out = torch.randn(batch_size, src_len, d_model).to(device)
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_len)).to(device)
    tgt_emb = torch.randn(batch_size, tgt_len, d_model).to(device)

    src_pad_mask = (src_ids == 0)
    tgt_pad_mask = (tgt_ids == 0)
    tgt_causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1).to(device)

    print("INPUT:")
    print(f"  tgt_emb shape: {tgt_emb.shape}  # (batch, tgt_len, d_model)")
    print(f"  enc_out shape: {enc_out.shape}  # (batch, src_len, d_model)")
    print(f"  tgt_causal_mask shape: {tgt_causal_mask.shape}  # (tgt_len, tgt_len)")

    dec_out = dec_layer(tgt_emb, enc_out, tgt_pad_mask, tgt_causal_mask, src_pad_mask)

    print("\nOUTPUT:")
    print(f"  dec_out shape: {dec_out.shape}  # (batch, tgt_len, d_model)")
    print(f"\n  → Stack 8 layers như này để tạo Decoder đầy đủ")
