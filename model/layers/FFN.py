import torch
import torch.nn as nn

class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        g, v = h[..., : self.d_ff], h[..., self.d_ff :]
        s = g * torch.sigmoid(g)  # SwiGLU
        out = self.linear2(s * v)
        return self.dropout(out)

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    d_model = 8
    d_ff = 16
    dropout = 0.1
    torch.manual_seed(0)
    device = torch.device("cpu")
    ffn = FFN_SwiGLU(d_model, d_ff, dropout).to(device)

    x = torch.randn(batch_size, src_len, d_model).to(device)

    print("INPUT:")
    print(f"  x shape: {x.shape}  # (batch, seq, d_model={d_model})")
    print(f"\n  FFN: {d_model} → {2*d_ff} → {d_ff} → {d_model}")

    ffn_out = ffn(x)

    print("\nOUTPUT:")
    print(f"  ffn_out shape: {ffn_out.shape}  # (batch, seq, d_model)")
