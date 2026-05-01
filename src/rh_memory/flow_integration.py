"""Differentiable flow integration layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def integrate_euler_midpoint_time(
    vector_field: nn.Module,
    start: Tensor,
    *,
    steps: int,
    t0: float = 0.0,
    t1: float = 1.0,
) -> Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    x = start
    dt = (t1 - t0) / steps
    for step in range(steps):
        t_value = t0 + (step + 0.5) * dt
        t = torch.full((start.shape[0],), t_value, device=start.device, dtype=start.dtype)
        x = x + dt * vector_field(x, t)
    return x


class EulerFlowIntegrator(nn.Module):
    def __init__(self, vector_field: nn.Module, *, steps: int, t0: float = 0.0, t1: float = 1.0) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        self.vector_field = vector_field
        self.steps = steps
        self.t0 = t0
        self.t1 = t1

    def forward(self, start: Tensor) -> Tensor:
        return integrate_euler_midpoint_time(
            self.vector_field,
            start,
            steps=self.steps,
            t0=self.t0,
            t1=self.t1,
        )
