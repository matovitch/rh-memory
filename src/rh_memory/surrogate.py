"""Differentiable surrogate backbone over bucket tokens (same output geometry as RHDecoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
import math

from .transformer_core import TransformerBlock


class RHSurrogate(nn.Module):
    """
    RoPE Transformer over **C** bucket tokens with width ``stride = n // C``.

    Layout matches :class:`RHDecoder` output geometry: logits ``[B, C, n]`` (per bucket,
    distribution over source indices).
    """

    def __init__(
        self,
        sequence_length: int,
        bucket_count: int,
        stride: int,
        fast_k: float,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.bucket_count = bucket_count
        self.stride = stride
        self.fast_k = fast_k
        self.attn_k_tokens = max(1, int(fast_k * math.log(bucket_count)))
        self.d_model = d_model
        self._cached_mask: Tensor | None = None

        self.input_proj = nn.Linear(stride, d_model)

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

    def forward(self, bucket_tokens: Float[Tensor, "batch C stride"]) -> Float[Tensor, "batch C n"]:
        """
        bucket_tokens: [batch_size, C, stride] — grouped amplitudes per bucket token.

        Returns logits [batch_size, C, sequence_length].
        """
        x = self.input_proj(bucket_tokens)
        attn_mask = self._build_ring_causal_mask(x.device)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        return self.output_proj(x)

    def _build_ring_causal_mask(self, device: torch.device) -> Tensor:
        """
        Ring-causal local mask over bucket tokens.

        Query bucket c can attend to exactly ``k_tokens`` keys:
        {c, c-1, ..., c-(k_tokens-1)} modulo C.
        """
        if self.attn_k_tokens <= 0:
            raise ValueError(f"attn_k_tokens must be >= 1, got {self.attn_k_tokens}")
        C = self.bucket_count
        k_tokens = min(self.attn_k_tokens, C)
        if self._cached_mask is not None and self._cached_mask.device == device:
            return self._cached_mask
        q = torch.arange(C, device=device).view(C, 1)
        keys = torch.arange(C, device=device).view(1, C)
        dist = (q - keys) % C
        mask = dist < k_tokens
        self._cached_mask = mask
        return mask


class RHSurrogateLoss(nn.Module):
    """Weighted BCE for :class:`RHSurrogate` logits ``[B, C, n]`` (per-bucket salience weights)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: Float[Tensor, "batch C n"],
        targets: Float[Tensor, "batch C n"],
        abs_amplitude: Float[Tensor, "batch C"],
    ) -> Float[Tensor, ""]:
        """
        logits: ``[batch_size, C, n]``
        targets: ``[batch_size, C, n]`` — sparse ground-truth masks ``1.0`` / ``0.0`` (which source index each bucket maps to).
        abs_amplitude: ``[batch_size, C]`` — per-bucket weight.
        """
        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        weighted_loss = bce_loss_raw * abs_amplitude.unsqueeze(2)
        return weighted_loss.mean()
