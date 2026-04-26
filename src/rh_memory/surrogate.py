"""Differentiable surrogate backbone over bucket tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

from .transformer_core import TransformerBlock


class RHSurrogate(nn.Module):
    """
    RoPE Transformer over **C** bucket tokens with width ``stride = N // C``.

    Outputs logits ``[B, C, N]``: one source-index distribution per bucket.
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
    ) -> None:
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

    def forward(self, bucket_tokens: Float[Tensor, "B C stride"]) -> Float[Tensor, "B C N"]:
        """
        bucket_tokens: [B, C, stride] — grouped amplitudes per bucket token.

        Returns logits [B, C, sequence_length].
        """
        x = self.input_proj(bucket_tokens)
        attn_mask = self._build_ring_causal_mask(x.device)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        return self.output_proj(x)

    def _build_ring_causal_mask(self, device: torch.device) -> Bool[Tensor, "C C"]:
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
    """Weighted cross entropy for :class:`RHSurrogate` logits ``[B, C, N]``."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: Float[Tensor, "B C N"],
        target_idx: Int[Tensor, "B C"],
        abs_amplitude: Float[Tensor, "B C"],
        valid_bucket: Bool[Tensor, "B C"] | None = None,
    ) -> Float[Tensor, ""]:
        """
        logits: ``[B, C, N]``
        target_idx: ``[B, C]`` — target source index for each bucket.
        abs_amplitude: ``[B, C]`` — per-bucket weight.
        valid_bucket: optional ``[B, C]`` mask. Invalid buckets contribute zero loss.
        """
        B, C, n = logits.shape
        if target_idx.shape != (B, C):
            raise ValueError(f"target_idx must have shape ({B}, {C}), got {tuple(target_idx.shape)}")
        if abs_amplitude.shape != (B, C):
            raise ValueError(f"abs_amplitude must have shape ({B}, {C}), got {tuple(abs_amplitude.shape)}")
        if valid_bucket is None:
            valid = torch.ones(B, C, dtype=torch.bool, device=logits.device)
        else:
            if valid_bucket.shape != (B, C):
                raise ValueError(f"valid_bucket must have shape ({B}, {C}), got {tuple(valid_bucket.shape)}")
            valid = valid_bucket.to(device=logits.device, dtype=torch.bool)

        ce = F.cross_entropy(
            logits.reshape(B * C, n),
            target_idx.to(device=logits.device, dtype=torch.long).reshape(B * C),
            reduction="none",
        ).view(B, C)
        weights = abs_amplitude.to(device=logits.device, dtype=logits.dtype) * valid.to(dtype=logits.dtype)
        denom = weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
        return (ce * weights).sum() / denom
