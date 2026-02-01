import math
import torch
import torch.nn as nn
from model.layers.RMSNorm import RMSNorm
from model.layers.Encoder import EncoderLayer
from model.layers.Decoder import DecoderLayer
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, d_ff, dropout,
                 rope_base, num_enc_layers=8, num_dec_layers=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base)
            for _ in range(num_enc_layers)
        ])
        self.encoder_final_ln = RMSNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base)
            for _ in range(num_dec_layers)
        ])
        self.decoder_final_ln = RMSNorm(d_model)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.emb_scale = math.sqrt(d_model)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        # Masks
        src_pad = (src_ids == 0)
        tgt_pad = (tgt_ids == 0)
        tgt_in = tgt_ids[:, :-1]
        tgt_pad_in = tgt_pad[:, :-1]
        T = tgt_in.size(1)
        tgt_causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=src_ids.device), diagonal=1)

        # Encoder
        src_emb = self.emb_dropout(self.embedding(src_ids) * self.emb_scale)
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad)
        enc_out = self.encoder_final_ln(enc_out)

        # Decoder
        tgt_emb = self.emb_dropout(self.embedding(tgt_in) * self.emb_scale)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_pad_in, tgt_causal, src_pad)
        dec_out = self.decoder_final_ln(dec_out)

        # Output (tied embedding)
        return F.linear(dec_out, self.embedding.weight, self.output_bias)

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    tgt_len = 5
    d_model = 512
    n_heads = 8
    n_kv_heads = 4
    d_ff = 2048
    dropout = 0.1
    rope_base = 10000.0
    vocab_size = 10000
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = TransformerModel(vocab_size, d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base).to(device)

    src_ids = torch.randint(1, vocab_size, (batch_size, src_len)).to(device)
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_len)).to(device)

    print("INPUT:")
    print(f"  src_ids shape: {src_ids.shape}  # (batch, src_len)")
    print(f"  tgt_ids shape: {tgt_ids.shape}  # (batch, tgt_len)")
    print(f"\nModel structure:")
    print(f"  - Embedding: {vocab_size} → {d_model}")
    print(f"  - Encoder: 8 layers")
    print(f"  - Decoder: 8 layers")
    print(f"  - Output: tied embedding weights")

    logits = model(src_ids, tgt_ids)

    print("\nOUTPUT:")
    print(f"  logits shape: {logits.shape}  # (batch, tgt_len-1, vocab_size)")
    print(f"  → Predict next token for each position")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
