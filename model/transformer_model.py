import math
import torch
import torch.nn as nn
from model.layers.RMSNorm import RMSNorm
from model.layers.Encoder import EncoderLayer
from model.layers.Decoder import DecoderLayer
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, config_or_vocab, vocab_size: int = None, **kwargs):
        """
        Flexible constructor that accepts either:
          - (config: RopeConfig, vocab_size: int)  -- preferred (notebook style)
          - (vocab_size, d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base, ...)

        To maintain backward compatibility, the constructor will attempt to
        use `config_or_vocab` as a config object first; if that doesn't work
        it will fall back to the older argument-style construction.
        """
        super().__init__()

        # Determine construction mode
        if vocab_size is not None:
            # Assume notebook/config style: (config, vocab_size)
            config = config_or_vocab
            self.config = config
            d_model = config.d_model
            n_heads = config.n_heads
            n_kv_heads = config.n_kv_heads
            d_ff = config.d_ff
            dropout = config.dropout
            rope_base = config.rope_base
            num_enc_layers = config.num_encoder_layers
            num_dec_layers = config.num_decoder_layers
            vocab_size = vocab_size
        else:
            # Fallback: older signature where config_or_vocab is actually vocab_size
            vocab_size = config_or_vocab
            d_model = kwargs.get("d_model")
            n_heads = kwargs.get("n_heads")
            n_kv_heads = kwargs.get("n_kv_heads")
            d_ff = kwargs.get("d_ff")
            dropout = kwargs.get("dropout")
            rope_base = kwargs.get("rope_base")
            num_enc_layers = kwargs.get("num_enc_layers", 8)
            num_dec_layers = kwargs.get("num_dec_layers", 8)

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        # Try to instantiate encoder/decoder layers using the "config" style
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_enc_layers):
            try:
                # If EncoderLayer accepts a single config arg (notebook style)
                layer = EncoderLayer(config)
            except Exception:
                # Fallback to positional args
                layer = EncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base)
            self.encoder_layers.append(layer)

        self.encoder_final_ln = RMSNorm(d_model)

        for _ in range(num_dec_layers):
            try:
                layer = DecoderLayer(config)
            except Exception:
                layer = DecoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, rope_base)
            self.decoder_layers.append(layer)

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
