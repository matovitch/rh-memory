"""Differentiable surrogate backbone over sequence positions (transpose geometry of RHDecoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .rope_bucket_transformer import RoPETransformerEncoderLayer


class RHSurrogate(nn.Module):
    """
    RoPE Transformer over **n** positions; one scalar amplitude per position.

    Layout is transposed vs :class:`RHDecoder`: logits ``[B, n, C]`` (per index, distribution over
    buckets) rather than ``[B, C, n]`` (per bucket, distribution over indices).
    """

    def __init__(
        self,
        sequence_length: int,
        bucket_count: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.bucket_count = bucket_count
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)

        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(
                    d_model, n_heads, dim_feedforward=dim_feedforward, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(d_model, bucket_count)

    def forward(self, position_tokens: Float[Tensor, "batch n 1"]) -> Float[Tensor, "batch n C"]:
        """
        position_tokens: [batch_size, n, 1] — signed amplitude (or continuous feature) per sequence index.

        Returns logits [batch_size, n, bucket_count].
        """
        x = self.input_proj(position_tokens)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


class RHSurrogateLoss(nn.Module):
    """Weighted BCE for :class:`RHSurrogate` logits ``[B, n, C]`` (per-position salience weights)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: Float[Tensor, "batch n C"],
        targets: Float[Tensor, "batch n C"],
        abs_amplitude: Float[Tensor, "batch n"],
    ) -> Float[Tensor, ""]:
        """
        logits: ``[batch_size, n, C]``
        targets: ``[batch_size, n, C]`` — sparse ground-truth masks ``1.0`` / ``0.0`` (which bucket each index maps to).
        abs_amplitude: ``[batch_size, n]`` — per-position weight, typically ``|x_i|`` along the signal.
        """
        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        weighted_loss = bce_loss_raw * abs_amplitude.unsqueeze(2)
        return weighted_loss.mean()
