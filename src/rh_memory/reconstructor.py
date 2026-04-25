"""Reconstructor backbone: bucket-token set -> full unpermuted sequence."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .transformer_core import TransformerBlock, TransformerCrossBlock


class RHReconstructor(nn.Module):
    """
    Reconstruct unpermuted sequence ``[B, n]`` from decoder-derived bucket tokens ``[B, C, 3]``:
    ``[amplitude, source_index, confidence]``.
    """

    def __init__(
        self,
        sequence_length: int,
        bucket_count: int,
        d_model: int = 512,
        n_heads: int = 16,
        num_token_layers: int = 4,
        num_query_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.bucket_count = bucket_count

        self.input_proj = nn.Linear(3, d_model)
        self.token_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=False,
                    mode="self",
                )
                for _ in range(num_token_layers)
            ]
        )

        self.query_embed = nn.Parameter(torch.randn(sequence_length, d_model) * 0.02)
        self.query_layers = nn.ModuleList(
            [
                TransformerCrossBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    self_use_rope=False,
                    cross_use_rope=False,
                    enable_query_self_attn=True,
                )
                for _ in range(num_query_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, 1)

    def encode_tokens(self, bucket_tokens: Float[Tensor, "batch C 3"]) -> Float[Tensor, "batch C d_model"]:
        if bucket_tokens.size(1) != self.bucket_count:
            raise ValueError(
                f"Expected bucket_count={self.bucket_count}, got tokens with C={bucket_tokens.size(1)}"
            )

        value = bucket_tokens[..., 0:1]
        source_index = bucket_tokens[..., 1:2]
        confidence = bucket_tokens[..., 2:3]
        x = torch.cat([value, source_index, confidence], dim=-1)
        x = self.input_proj(x)
        for layer in self.token_layers:
            x = layer(x)
        return x

    def forward(self, bucket_tokens: Float[Tensor, "batch C 3"]) -> Float[Tensor, "batch n"]:
        memory = self.encode_tokens(bucket_tokens)
        batch_size = bucket_tokens.size(0)
        q = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.query_layers:
            q = layer(q, memory)
        x_hat = self.output_proj(q).squeeze(-1)
        return x_hat


class RHReconstructorLoss(nn.Module):
    """MSE loss for sequence reconstruction."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(
        self,
        reconstructed: Float[Tensor, "batch n"],
        target: Float[Tensor, "batch n"],
    ) -> Float[Tensor, ""]:
        return self.criterion(reconstructed, target)
