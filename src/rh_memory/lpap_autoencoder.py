"""End-to-end LPAP autoencoder composition and loss."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from rh_memory.decoder import RHDecoder
from rh_memory.decoder_scatter import SoftScatterReconstructionHead
from rh_memory.flow_integration import EulerFlowIntegrator
from rh_memory.pipeline.primitives_permutation import gather_permuted_stream
from rh_memory.pipeline.primitives_targets import surrogate_teacher_bucket_slot_indices
from rh_memory.pipeline.primitives_tokens import (
    decoder_tokens_from_surrogate_logits_soft,
    reshape_permuted_to_bucket_tokens,
)
from rh_memory.pooling_utils import lpap_pool
from rh_memory.surrogate import RHSurrogate, RHSurrogateLoss


@dataclass(frozen=True)
class LPAPSurrogateTargets:
    target_idx: Int[Tensor, "B C"]
    valid_bucket: Bool[Tensor, "B C"]
    weights: Float[Tensor, "B C"]


@dataclass(frozen=True)
class LPAPAutoencoderOutput:
    branch_name: str
    reconstructed_image: Float[Tensor, "B 1 N"]
    learned_energy: Float[Tensor, "B 1 N"]
    energy_perm: Float[Tensor, "B N"]
    surrogate_logits: Float[Tensor, "B C N"]
    decoder_tokens: Float[Tensor, "B C 3"]
    decoder_logits: Float[Tensor, "B C N"]
    projected_energy: Float[Tensor, "B 1 N"]
    scatter_probs: Float[Tensor, "B C N"]
    scatter_doubt: Float[Tensor, "B C"]
    scatter_support: Float[Tensor, "B C"]
    scatter_temperature: Tensor
    lpap_targets: LPAPSurrogateTargets


class LPAPBottleneckBranch(nn.Module):
    """One C-specific LPAP surrogate + decoder/scatter bottleneck branch."""

    def __init__(
        self,
        *,
        name: str,
        surrogate: RHSurrogate,
        decoder: RHDecoder,
        scatter_head: SoftScatterReconstructionHead,
        perm_1d: Tensor,
        fast_k: float,
        surrogate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if surrogate.sequence_length != decoder.sequence_length:
            raise ValueError("surrogate and decoder must use the same sequence_length")
        if surrogate.bucket_count != decoder.bucket_count:
            raise ValueError("surrogate and decoder must use the same bucket_count")
        if perm_1d.shape != (surrogate.sequence_length,):
            raise ValueError(f"perm_1d must have shape ({surrogate.sequence_length},), got {tuple(perm_1d.shape)}")
        if fast_k <= 0:
            raise ValueError(f"fast_k must be positive, got {fast_k}")
        if surrogate_temperature <= 0:
            raise ValueError(f"surrogate_temperature must be positive, got {surrogate_temperature}")

        self.name = name
        self.surrogate = surrogate
        self.decoder = decoder
        self.scatter_head = scatter_head
        self.fast_k = float(fast_k)
        self.surrogate_temperature = float(surrogate_temperature)
        self.register_buffer("perm_1d", perm_1d.to(dtype=torch.long), persistent=True)

    @property
    def sequence_length(self) -> int:
        return int(self.surrogate.sequence_length)

    @property
    def bucket_count(self) -> int:
        return int(self.surrogate.bucket_count)

    @property
    def stride(self) -> int:
        return int(self.surrogate.stride)

    @property
    def k_eff(self) -> int:
        return max(1, int(self.fast_k * math.log(self.bucket_count)))

    def forward(
        self, learned_energy: Float[Tensor, "B 1 N"]
    ) -> tuple[
        Float[Tensor, "B N"],
        Float[Tensor, "B C N"],
        Float[Tensor, "B C 3"],
        Float[Tensor, "B C N"],
        Float[Tensor, "B 1 N"],
        Float[Tensor, "B C N"],
        Float[Tensor, "B C"],
        Float[Tensor, "B C"],
        Tensor,
        LPAPSurrogateTargets,
    ]:
        if learned_energy.dim() != 3 or learned_energy.size(1) != 1:
            raise ValueError(f"learned_energy must have shape [B, 1, N], got {tuple(learned_energy.shape)}")
        if learned_energy.size(2) != self.sequence_length:
            raise ValueError(f"Expected N={self.sequence_length}, got {learned_energy.size(2)}")

        perm_1d = cast(Tensor, self.perm_1d)
        energy_values = learned_energy.squeeze(1)
        energy_perm = gather_permuted_stream(energy_values, perm_1d)
        surrogate_tokens = reshape_permuted_to_bucket_tokens(energy_perm, self.bucket_count)
        surrogate_logits = self.surrogate(surrogate_tokens)[:, : self.bucket_count, : self.sequence_length]
        decoder_tokens = decoder_tokens_from_surrogate_logits_soft(
            energy_perm,
            surrogate_logits,
            temperature=self.surrogate_temperature,
        )
        decoder_logits = self.decoder(decoder_tokens)[:, : self.bucket_count, : self.sequence_length]
        projected, scatter_probs, scatter_doubt, scatter_support, scatter_temperature = self.scatter_head(
            decoder_logits,
            decoder_tokens[..., 0],
            perm_1d,
        )
        lpap_targets = lpap_surrogate_targets(energy_perm, self.bucket_count, self.k_eff)
        return (
            energy_perm,
            surrogate_logits,
            decoder_tokens,
            decoder_logits,
            projected.unsqueeze(1),
            scatter_probs,
            scatter_doubt,
            scatter_support,
            scatter_temperature,
            lpap_targets,
        )


class LPAPAutoencoder(nn.Module):
    """Trainable image -> LPAP bottleneck -> image autoencoder path."""

    def __init__(
        self,
        *,
        image_to_energy_flow: nn.Module,
        energy_to_image_flow: nn.Module,
        bottleneck_branches: list[LPAPBottleneckBranch],
        image_to_energy_steps: int,
        energy_to_image_steps: int,
        default_branch: str | None = None,
    ) -> None:
        super().__init__()
        if not bottleneck_branches:
            raise ValueError("at least one bottleneck branch is required")
        branch_names = [branch.name for branch in bottleneck_branches]
        if len(set(branch_names)) != len(branch_names):
            raise ValueError(f"bottleneck branch names must be unique, got {branch_names}")
        self.image_to_energy = EulerFlowIntegrator(image_to_energy_flow, steps=image_to_energy_steps)
        self.energy_to_image = EulerFlowIntegrator(energy_to_image_flow, steps=energy_to_image_steps)
        self.branches = nn.ModuleDict({branch.name: branch for branch in bottleneck_branches})
        self.default_branch = default_branch or bottleneck_branches[0].name
        if self.default_branch not in self.branches:
            raise ValueError(f"default_branch {self.default_branch!r} is not registered")

    def forward(self, image_seq: Float[Tensor, "B 1 N"], *, branch_name: str | None = None) -> LPAPAutoencoderOutput:
        if image_seq.dim() != 3 or image_seq.size(1) != 1:
            raise ValueError(f"image_seq must have shape [B, 1, N], got {tuple(image_seq.shape)}")
        active_branch_name = branch_name or self.default_branch
        branch = self.branches[active_branch_name]
        if image_seq.size(2) != branch.sequence_length:
            raise ValueError(f"Expected N={branch.sequence_length}, got {image_seq.size(2)}")

        learned_energy = self.image_to_energy(image_seq)
        (
            energy_perm,
            surrogate_logits,
            decoder_tokens,
            decoder_logits,
            projected_energy,
            scatter_probs,
            scatter_doubt,
            scatter_support,
            scatter_temperature,
            lpap_targets,
        ) = branch(learned_energy)
        reconstructed_image = self.energy_to_image(projected_energy)

        return LPAPAutoencoderOutput(
            branch_name=active_branch_name,
            reconstructed_image=reconstructed_image,
            learned_energy=learned_energy,
            energy_perm=energy_perm,
            surrogate_logits=surrogate_logits,
            decoder_tokens=decoder_tokens,
            decoder_logits=decoder_logits,
            projected_energy=projected_energy,
            scatter_probs=scatter_probs,
            scatter_doubt=scatter_doubt,
            scatter_support=scatter_support,
            scatter_temperature=scatter_temperature,
            lpap_targets=lpap_targets,
        )


@dataclass(frozen=True)
class LPAPAutoencoderLossOutput:
    total: Float[Tensor, ""]
    reconstruction: Float[Tensor, ""]
    lpap_surrogate: Float[Tensor, ""]


class LPAPAutoencoderLoss(nn.Module):
    """MSE reconstruction plus LPAP surrogate regularization."""

    def __init__(self, *, lpap_weight: float = 1.0) -> None:
        super().__init__()
        if lpap_weight < 0:
            raise ValueError(f"lpap_weight must be non-negative, got {lpap_weight}")
        self.lpap_weight = float(lpap_weight)
        self.surrogate_loss = RHSurrogateLoss()

    def forward(
        self,
        output: LPAPAutoencoderOutput,
        target_image: Float[Tensor, "B 1 N"],
    ) -> LPAPAutoencoderLossOutput:
        if output.reconstructed_image.shape != target_image.shape:
            raise ValueError(
                f"target_image must match reconstructed_image shape, got "
                f"{tuple(target_image.shape)} and {tuple(output.reconstructed_image.shape)}"
            )
        reconstruction = F.mse_loss(output.reconstructed_image, target_image)
        lpap_surrogate = self.surrogate_loss(
            output.surrogate_logits,
            output.lpap_targets.target_idx,
            output.lpap_targets.weights,
            output.lpap_targets.valid_bucket,
        )
        total = reconstruction + output.reconstructed_image.new_tensor(self.lpap_weight) * lpap_surrogate
        return LPAPAutoencoderLossOutput(
            total=total,
            reconstruction=reconstruction,
            lpap_surrogate=lpap_surrogate,
        )


def lpap_surrogate_targets(
    x_perm: Float[Tensor, "B N"],
    bucket_count: int,
    k_eff: int,
) -> LPAPSurrogateTargets:
    if x_perm.dim() != 2:
        raise ValueError(f"x_perm must have shape [B, N], got {tuple(x_perm.shape)}")
    if bucket_count <= 0:
        raise ValueError(f"bucket_count must be positive, got {bucket_count}")
    if k_eff <= 0:
        raise ValueError(f"k_eff must be positive, got {k_eff}")
    B, n = x_perm.shape
    if n % bucket_count != 0:
        raise ValueError(f"N must be divisible by C, got N={n}, C={bucket_count}")

    device = x_perm.device
    weights = x_perm.detach().abs().view(B, n // bucket_count, bucket_count).transpose(1, 2).max(dim=2).values
    table_values = torch.zeros(B, bucket_count, dtype=torch.float32, device=device)
    table_dib = torch.zeros(B, bucket_count, dtype=torch.int32, device=device)
    table_carry_id = torch.full((B, bucket_count), -1, dtype=torch.int32, device=device)
    slot_ids = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(0).expand(B, n)

    _values, _dib, out_slot_id = lpap_pool(
        table_values,
        table_dib,
        table_carry_id,
        x_perm.detach().to(dtype=torch.float32).clone(),
        slot_ids.clone(),
        k_eff,
        device,
    )
    target_idx, valid_bucket = surrogate_teacher_bucket_slot_indices(out_slot_id, n)
    return LPAPSurrogateTargets(
        target_idx=target_idx,
        valid_bucket=valid_bucket,
        weights=weights * valid_bucket.to(device=device, dtype=weights.dtype),
    )
