import torch 
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
            )
        return self.weight * x / rms

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 4
    d_model = 8
    torch.manual_seed(0)
    torch.device = torch.device("cpu")

    src_emb = torch.randn(batch_size, seq_len, d_model).to(torch.device)
    rms_norm = RMSNorm(d_model).to(torch.device)

    print("INPUT:")
    print(f"  x shape: {src_emb.shape}")

    normed = rms_norm(src_emb)

    print("\nOUTPUT:")
    print(f"  normed shape: {normed.shape}  # Same shape")
    print(f"  RMS before: {src_emb.pow(2).mean(dim=-1).sqrt().mean().item():.4f}")
    print(f"  RMS after: {normed.pow(2).mean(dim=-1).sqrt().mean().item():.4f}")
