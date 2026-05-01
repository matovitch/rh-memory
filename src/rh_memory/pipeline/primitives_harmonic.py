from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor


def harmonic_raw_batch(
    chunk_size: int,
    n: int,
    device: torch.device | str,
    harmonic_decay: float,
    harmonic_amp_threshold: float,
    max_harmonics: int,
) -> Float[Tensor, "B N"]:
    """Pseudo-harmonic peaks used by surrogate and decoder training/evaluation."""
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32).unsqueeze(0).expand(chunk_size, n)
    gamma = harmonic_decay
    tau = harmonic_amp_threshold
    max_h = max_harmonics

    sum_peaks = torch.zeros(chunk_size, n, device=device, dtype=torch.float32)
    k_h = 1
    while k_h <= max_h:
        sigma_k = gamma**k_h
        if sigma_k < tau:
            break
        z = torch.randn(chunk_size, 1, device=device, dtype=torch.float32)
        alpha_k = z * sigma_k

        a_k = torch.empty(chunk_size, 1, device=device, dtype=torch.float32).uniform_(1.0, 5.0)
        phi_k = torch.empty(chunk_size, 1, device=device, dtype=torch.float32).uniform_(-math.pi, math.pi)

        angle = k_h * math.pi * t + phi_k
        envelope_inner = (1.0 - torch.sin(angle).abs()).clamp(min=1e-8)
        envelope = envelope_inner.pow(torch.exp(a_k))

        sum_peaks += alpha_k * envelope
        k_h += 1

    signs = torch.empty(chunk_size, n, device=device, dtype=torch.float32).uniform_(0.0, 1.0).round() * 2 - 1
    return sum_peaks * signs
