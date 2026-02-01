import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def mean_pool(enc_out, ids, pad_id, special_ids):
    """Mean pooling, bỏ qua padding và special tokens"""
    mask = (ids != pad_id)
    for special in special_ids:
        mask = mask & (ids != special)
    mask = mask.float()
    summed = (enc_out * mask.unsqueeze(-1)).sum(dim=1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return summed / denom

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    src_len = 4
    d_model = 512
    vocab_size = 10000
    torch.manual_seed(0)
    device = torch.device("cpu")

    proj_dim = 768
    projection = ProjectionHead(d_model, proj_dim).to(device)

    # Giả sử có encoder output
    enc_out = torch.randn(batch_size, src_len, d_model).to(device)
    src_ids = torch.randint(1, vocab_size, (batch_size, src_len)).to(device)
    pad_id = 0
    special_ids = [0, 2, 3, 4, 5]  # pad, bos, eos, <2zh>, <2vi>

    print("INPUT:")
    print(f"  enc_out shape: {enc_out.shape}  # (batch, seq, d_model)")

    # Mean pool
    pooled = mean_pool(enc_out, src_ids, pad_id, special_ids)
    print(f"\nAfter mean pooling:")
    print(f"  pooled shape: {pooled.shape}  # (batch, d_model)")

    # Project
    z = projection(pooled)
    print(f"\nAfter projection:")
    print(f"  z shape: {z.shape}  # (batch, proj_dim)")
    print(f"  z norm: {z.norm(dim=-1).mean().item():.4f}  # Should be ~1.0 (normalized)")

    print(f"\n→ Dùng z để tính contrastive loss giữa các cặp câu ZH-VI")
