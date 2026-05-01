"""Differentiable decoder bridge over soft surrogate bucket tokens."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .transformer_core import TransformerBlock


class RHDecoder(nn.Module):
    """
    Decode soft surrogate bucket tokens ``[B, C, 3]`` into per-bucket logits ``[B, C, N]``.

    Token channels are ``[soft_amplitude, soft_normalized_dib, surrogate_doubt]``.
    """

    def __init__(
        self,
        sequence_length: int,
        bucket_count: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.bucket_count = bucket_count

        self.input_proj = nn.Linear(3, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=True,
                    mode="self",
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, sequence_length)

    def forward(self, decoder_tokens: Float[Tensor, "B C 3"]) -> Float[Tensor, "B C N"]:
        if decoder_tokens.size(1) != self.bucket_count:
            raise ValueError(f"Expected bucket_count={self.bucket_count}, got tokens with C={decoder_tokens.size(1)}")
        x = self.input_proj(decoder_tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


class RHDecoderDistillationLoss(nn.Module):
    """Weighted soft KL distillation from surrogate logits to decoder logits."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        decoder_logits: Float[Tensor, "B C N"],
        teacher_logits: Float[Tensor, "B C N"],
        weights: Float[Tensor, "B C"],
    ) -> Float[Tensor, ""]:
        if decoder_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"decoder_logits and teacher_logits must have matching shapes, "
                f"got {tuple(decoder_logits.shape)} and {tuple(teacher_logits.shape)}"
            )
        B, C, _n = decoder_logits.shape
        if weights.shape != (B, C):
            raise ValueError(f"weights must have shape ({B}, {C}), got {tuple(weights.shape)}")

        temperature = self.temperature
        teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        decoder_log_probs = F.log_softmax(decoder_logits / temperature, dim=-1)
        kl = F.kl_div(decoder_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl = kl * (temperature * temperature)
        weights = weights.to(device=decoder_logits.device, dtype=decoder_logits.dtype)
        denom = weights.sum().clamp_min(decoder_logits.new_tensor(1e-12))
        return (kl * weights).sum() / denom
