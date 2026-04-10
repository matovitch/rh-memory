import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# ----------------------------
# RH Pooling (approximate, GPU-friendly)
# ----------------------------
def rh_pool(x, C, a=1, b=0, max_shifts=None):
    """
    x: (B, n)
    returns:
        values: (B, C)
        dib:    (B, C)
    """
    B, n = x.shape
    assert n % C == 0
    stride = n // C

    if max_shifts is None:
        max_shifts = max(1, int(2 * math.log(n)))

    # apply affine hash via permutation
    idx = torch.arange(n, device=x.device)
    hashed_idx = (a * idx + b) % n
    x = x[:, hashed_idx]

    x_2d = x.view(B, C, stride).clone()
    result_vals = torch.zeros(B, C, device=x.device)
    result_dib = torch.zeros(B, C, device=x.device)

    for s in range(max_shifts):
        abs_vals, argmax = x_2d.abs().max(dim=2)  # (B, C)

        current_vals = torch.gather(
            x_2d, 2, argmax.unsqueeze(-1)
        ).squeeze(-1)

        replace = abs_vals > result_vals.abs()

        result_vals = torch.where(replace, current_vals, result_vals)
        result_dib = torch.where(
            replace,
            torch.full_like(result_dib, s),
            result_dib
        )

        # zero out selected
        x_2d.scatter_(2, argmax.unsqueeze(-1), 0.0)

        # roll buckets
        x_2d = torch.roll(x_2d, shifts=-1, dims=1)

    return result_vals, result_dib


# ----------------------------
# Multi-head RH
# ----------------------------
def multihead_rh(x, C, H):
    """
    returns:
        values: (B, H, C)
        dib:    (B, H, C)
    """
    B, n = x.shape
    values = []
    dibs = []

    for h in range(H):
        a = random.randint(1, n - 1)
        b = random.randint(0, n - 1)

        v, d = rh_pool(x, C, a=a, b=b)
        values.append(v)
        dibs.append(d)

    values = torch.stack(values, dim=1)
    dibs = torch.stack(dibs, dim=1)

    return values, dibs


# ----------------------------
# Simple Decoder
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, C, H, n):
        super().__init__()
        self.H = H
        self.C = C

        input_dim = H * C * 2  # values + dib
        hidden = 256

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n),
        )

    def forward(self, values, dib):
        B = values.shape[0]

        x = torch.cat([values, dib], dim=-1)  # (B, H, 2C)
        x = x.view(B, -1)

        return self.net(x)


# ----------------------------
# Synthetic sparse data
# ----------------------------
def generate_sparse(batch_size, n, sparsity=0.05):
    x = torch.randn(batch_size, n)

    mask = torch.rand_like(x) < sparsity
    x = x * mask

    return x


# ----------------------------
# Training loop
# ----------------------------
def train(mode="single"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n = 512
    C = 64
    H = 4 if mode == "multi" else 1

    model = Decoder(C, H, n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1000):
        x = generate_sparse(64, n).to(device)

        if mode == "single":
            v, d = rh_pool(x, C)
            v = v.unsqueeze(1)
            d = d.unsqueeze(1)
        else:
            v, d = multihead_rh(x, C, H)

        x_hat = model(v, d)

        loss = F.mse_loss(x_hat, x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"{mode} step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    print("Training SINGLE hash")
    train("single")

    print("\nTraining MULTI hash")
    train("multi")
