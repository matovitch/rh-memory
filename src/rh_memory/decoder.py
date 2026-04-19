import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .rope_bucket_transformer import RoPETransformerEncoderLayer

class RHDecoder(nn.Module):
    def __init__(self, sequence_length, bucket_count, d_model=256, n_heads=8, num_layers=4, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.sequence_length = sequence_length
        self.bucket_count = bucket_count
        self.d_model = d_model
        
        # Continuous projection: signed_value + DIB per bucket (carry_id stays inside pooler only).
        self.input_proj = nn.Linear(2, d_model)
        
        # Deep N-layer Transformer backbone with continuous RoPE
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, n_heads, dim_feedforward=dim_feedforward, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Dense head projecting back out to sequence domain: [B, C, D_model] -> [B, C, n]
        self.output_proj = nn.Linear(d_model, sequence_length)

    def forward(self, bucket_tokens: Float[Tensor, "batch C 2"]) -> Float[Tensor, "batch C n"]:
        """
        Forward pass for the decoder backbone.

        bucket_tokens: [batch_size, C, 2] — per bucket:
            1. signed_value (signed amplitude sample from pooler table)
            2. dib (distance to initial bucket as float)

        Source indices (carry_id) are computed inside the pooler for targets only, not fed here.
        """
        x = self.input_proj(bucket_tokens)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.output_proj(x)
        return logits

    def reconstruct(self, logits: Float[Tensor, "batch C n"]) -> Float[Tensor, "batch n"]:
        """
        Reconstructs the original [B, n] sequence using max extraction per bucket
        and reducing over collisions with scatter_add.
        logits: FloatTensor of shape [batch_size, C, n]
        """
        batch_size, C, n = logits.shape
        
        # 1. Determine single most likely position for each bucket (max over dim=2)
        # predicted_indices shape: [batch_size, C]
        max_logits, predicted_indices = torch.max(logits, dim=2)
        
        # 2. Project back via scatter_add onto a neutral [batch_size, n] tensor
        reconstructed = torch.zeros(batch_size, n, device=logits.device, dtype=logits.dtype)
        reconstructed.scatter_add_(1, predicted_indices, max_logits)
        
        return reconstructed

class RHDecoderLoss(nn.Module):
    """Weighted BCE for :class:`RHDecoder` logits ``[B, C, n]`` (per-bucket salience weights)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: Float[Tensor, "batch C n"], targets: Float[Tensor, "batch C n"], abs_amplitude: Float[Tensor, "batch C"]) -> Float[Tensor, ""]:
        """
        logits: ``[batch_size, C, n]``
        targets: ``[batch_size, C, n]`` — sparse ground-truth masks ``1.0`` / ``0.0``.
        abs_amplitude: ``[batch_size, C]`` — per-bucket weight, typically ``|signed_value|`` for active slots.
        """
        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        weighted_loss = bce_loss_raw * abs_amplitude.unsqueeze(2)
        return weighted_loss.mean()
