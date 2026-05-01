"""Small time-conditioned 1D flow models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Float[Tensor, "B"]) -> Float[Tensor, "B dim"]:
        if t.dim() != 1:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class DilatedResidualBlock1d(nn.Module):
    def __init__(self, width: int, time_dim: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = dilation * (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(1, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size, padding=padding, dilation=dilation)
        self.time_proj = nn.Linear(time_dim, 2 * width)
        self.norm2 = nn.GroupNorm(1, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size, padding=padding, dilation=dilation)

    def forward(
        self, x: Float[Tensor, "B width N"], time_emb: Float[Tensor, "B time_dim"]
    ) -> Float[Tensor, "B width N"]:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h / math.sqrt(2.0)


class DilatedConvFlow1d(nn.Module):
    """Time-conditioned vector field over scalar sequences ``[B, 1, N]``."""

    def __init__(
        self,
        sequence_length: int = 1024,
        width: int = 128,
        time_dim: int = 128,
        dilation_cycles: int = 2,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size: int = 3,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if time_dim <= 0:
            raise ValueError(f"time_dim must be positive, got {time_dim}")
        if dilation_cycles <= 0:
            raise ValueError(f"dilation_cycles must be positive, got {dilation_cycles}")
        if not dilations:
            raise ValueError("dilations must not be empty")

        self.sequence_length = sequence_length
        self.width = width
        self.time_dim = time_dim
        self.dilation_cycles = dilation_cycles
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.kernel_size = kernel_size

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input_proj = nn.Conv1d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock1d(width, time_dim, dilation, kernel_size=kernel_size)
            for _cycle in range(dilation_cycles)
            for dilation in self.dilations
        )
        self.output_norm = nn.GroupNorm(1, width)
        self.output_proj = nn.Conv1d(width, 1, kernel_size=1)
        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Float[Tensor, "B 1 N"], t: Float[Tensor, "B"]) -> Float[Tensor, "B 1 N"]:
        if x.dim() != 3 or x.shape[1] != 1 or x.shape[2] != self.sequence_length:
            raise ValueError(f"x must have shape [B, 1, {self.sequence_length}], got {tuple(x.shape)}")
        if t.shape != (x.shape[0],):
            raise ValueError(f"t must have shape [{x.shape[0]}], got {tuple(t.shape)}")
        time_emb = self.time_embedding(t.to(device=x.device, dtype=x.dtype))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, time_emb)
        return self.output_proj(F.silu(self.output_norm(h)))

    def config_dict(self) -> dict[str, int | bool | list[int]]:
        return {
            "sequence_length": self.sequence_length,
            "width": self.width,
            "time_dim": self.time_dim,
            "dilation_cycles": self.dilation_cycles,
            "dilations": list(self.dilations),
            "kernel_size": self.kernel_size,
        }


def interpolate_linear(start: Tensor, end: Tensor, t: Tensor) -> Tensor:
    view_t = t.view(t.shape[0], 1, 1)
    return (1.0 - view_t) * start + view_t * end


def flow_matching_loss(model: nn.Module, start: Tensor, end: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x_t = interpolate_linear(start, end, t)
    target_velocity = end - start
    pred_velocity = model(x_t, t)
    loss = F.mse_loss(pred_velocity, target_velocity)
    return loss, pred_velocity, target_velocity
