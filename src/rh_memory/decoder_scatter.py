"""Differentiable gather/scatter heads around decoder logits."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from .pipeline.primitives_tokens import normalized_entropy, scalar_dib_table


def _inverse_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError(f"inverse softplus input must be positive, got {x}")
    return x + math.log(-math.expm1(-x))


def effective_support(probs: Float[Tensor, "... N"], eps: float = 1e-12) -> Float[Tensor, "..."]:
    """Return entropy-derived effective support along the last dimension."""
    safe_probs = probs.clamp_min(eps)
    entropy = -(probs * safe_probs.log()).sum(dim=-1)
    return entropy.exp()


def decoder_soft_scatter(
    decoder_logits: Float[Tensor, "B C N"],
    bucket_amplitude: Float[Tensor, "B C"],
    perm_1d: Int[Tensor, "N"],
    temperature: float | Tensor,
) -> tuple[Float[Tensor, "B N"], Float[Tensor, "B C N"]]:
    """Scatter bucket amplitudes over source positions using decoder probabilities."""
    if decoder_logits.dim() != 3:
        raise ValueError(f"decoder_logits must have shape [B, C, N], got {tuple(decoder_logits.shape)}")
    B, C, n = decoder_logits.shape
    if bucket_amplitude.shape != (B, C):
        raise ValueError(f"bucket_amplitude must have shape ({B}, {C}), got {tuple(bucket_amplitude.shape)}")
    if perm_1d.shape != (n,):
        raise ValueError(f"perm_1d must have shape ({n},), got {tuple(perm_1d.shape)}")

    if isinstance(temperature, Tensor):
        if torch.any(temperature <= 0):
            raise ValueError("temperature tensor must be strictly positive")
        temperature_value = temperature.to(device=decoder_logits.device, dtype=decoder_logits.dtype)
    else:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        temperature_value = decoder_logits.new_tensor(float(temperature))

    probs = F.softmax(decoder_logits / temperature_value, dim=-1)
    weighted_values = probs * bucket_amplitude.to(device=decoder_logits.device, dtype=decoder_logits.dtype).unsqueeze(
        -1
    )

    source_index_by_slot = perm_1d.to(device=decoder_logits.device, dtype=torch.long)
    source_index = source_index_by_slot.view(1, 1, n).expand(B, C, n)

    reconstruction = decoder_logits.new_zeros(B, n)
    reconstruction.scatter_add_(1, source_index.reshape(B, -1), weighted_values.reshape(B, -1))
    return reconstruction, probs


class SoftScatterReconstructionHead(nn.Module):
    """Learnable-temperature differentiable scatter head for decoder logits."""

    def __init__(self, init_temperature: float = 1.0, min_temperature: float = 0.05) -> None:
        super().__init__()
        if min_temperature <= 0:
            raise ValueError(f"min_temperature must be positive, got {min_temperature}")
        if init_temperature <= min_temperature:
            raise ValueError(
                f"init_temperature must be greater than min_temperature, "
                f"got init_temperature={init_temperature}, min_temperature={min_temperature}"
            )
        self.min_temperature = float(min_temperature)
        self.raw_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_temperature) - self.min_temperature), dtype=torch.float32)
        )

    def temperature(self) -> Tensor:
        return self.raw_temperature.new_tensor(self.min_temperature) + F.softplus(self.raw_temperature)

    def forward(
        self,
        decoder_logits: Float[Tensor, "B C N"],
        bucket_amplitude: Float[Tensor, "B C"],
        perm_1d: Int[Tensor, "N"],
    ) -> tuple[Float[Tensor, "B N"], Float[Tensor, "B C N"], Float[Tensor, "B C"], Float[Tensor, "B C"], Tensor]:
        temperature = self.temperature().to(device=decoder_logits.device, dtype=decoder_logits.dtype)
        reconstruction, probs = decoder_soft_scatter(decoder_logits, bucket_amplitude, perm_1d, temperature)
        doubt = normalized_entropy(probs)
        support = effective_support(probs)
        return reconstruction, probs, doubt, support, temperature


class SoftGatherTokenizationHead(nn.Module):
    """Learnable-temperature soft gather from surrogate logits to decoder tokens."""

    def __init__(self, init_temperature: float = 1.0, min_temperature: float = 0.05) -> None:
        super().__init__()
        if min_temperature <= 0:
            raise ValueError(f"min_temperature must be positive, got {min_temperature}")
        if init_temperature <= min_temperature:
            raise ValueError(
                f"init_temperature must be greater than min_temperature, "
                f"got init_temperature={init_temperature}, min_temperature={min_temperature}"
            )
        self.min_temperature = float(min_temperature)
        self.raw_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_temperature) - self.min_temperature), dtype=torch.float32)
        )

    def temperature(self) -> Tensor:
        return self.raw_temperature.new_tensor(self.min_temperature) + F.softplus(self.raw_temperature)

    def forward(
        self,
        x_perm: Float[Tensor, "B N"],
        surrogate_logits: Float[Tensor, "B C N"],
    ) -> Float[Tensor, "B C 3"]:
        if surrogate_logits.dim() != 3:
            raise ValueError(f"surrogate_logits must have shape [B, C, N], got {tuple(surrogate_logits.shape)}")
        B, C, n = surrogate_logits.shape
        if x_perm.shape != (B, n):
            raise ValueError(f"x_perm must have shape ({B}, {n}), got {tuple(x_perm.shape)}")

        temperature = self.temperature().to(device=surrogate_logits.device, dtype=surrogate_logits.dtype)
        gather_probs = F.softmax(surrogate_logits / temperature, dim=-1)
        soft_amplitude = torch.einsum(
            "bcn,bn->bc",
            gather_probs,
            x_perm.to(device=surrogate_logits.device, dtype=surrogate_logits.dtype),
        )
        dib = scalar_dib_table(C, n, device=surrogate_logits.device, dtype=surrogate_logits.dtype)
        soft_normalized_dib = torch.einsum("bcn,cn->bc", gather_probs, dib)

        entropy_probs = F.softmax(surrogate_logits, dim=-1)
        surrogate_doubt = normalized_entropy(entropy_probs)
        return torch.stack([soft_amplitude, soft_normalized_dib, surrogate_doubt], dim=-1)
