from __future__ import annotations

from typing import Iterable, Iterator

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from rh_memory.pooling_utils import lpap_pool

from .config import PipelineConfig
from .primitives_targets import surrogate_teacher_bucket_slot_indices
from .primitives_tokens import reshape_permuted_to_bucket_tokens
from .types import HarmonicSample, SurrogateInferenceSample


def surrogate_training_adapter(
    stream: Iterable[HarmonicSample],
    *,
    config: PipelineConfig,
) -> Iterator[
    tuple[
        Float[Tensor, "B C stride"],
        Int[Tensor, "B C"],
        Bool[Tensor, "B C"],
        Float[Tensor, "B C"],
    ]
]:
    for sample in stream:
        x_perm = sample.x_perm
        device = x_perm.device
        B, n = x_perm.shape
        if n != config.n:
            raise ValueError(f"Expected N={config.n}, got x_perm with N={n}")

        bucket_input = reshape_permuted_to_bucket_tokens(x_perm, config.C)
        abs_amp = x_perm.abs()
        weights = abs_amp.view(B, config.stride, config.C).transpose(1, 2).max(dim=2).values

        table_values = torch.zeros(B, config.C, dtype=torch.float32, device=device)
        table_dib = torch.zeros(B, config.C, dtype=torch.int32, device=device)
        table_carry_id = torch.full((B, config.C), -1, dtype=torch.int32, device=device)
        slot_ids = torch.arange(config.n, dtype=torch.int32, device=device).unsqueeze(0).expand(B, config.n)

        # LPAP is allowed to mutate incoming tensors as scratch storage for speed.
        # Keep the sample-owned stream read-only by passing explicit scratch clones.
        _ov, _od, out_slot_id = lpap_pool(
            table_values,
            table_dib,
            table_carry_id,
            x_perm.clone(),
            slot_ids.clone(),
            config.k_eff,
            device,
        )

        target_idx, valid_bucket = surrogate_teacher_bucket_slot_indices(out_slot_id, config.n)
        weights = weights * valid_bucket.float()
        yield bucket_input, target_idx, valid_bucket, weights
