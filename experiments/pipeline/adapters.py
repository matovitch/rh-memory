from __future__ import annotations

from typing import Iterable, Iterator

import torch

from rh_memory.pooling_utils import lpap_pool

from .primitives_targets import decoder_targets_from_j_star, surrogate_teacher_bucket_slot_targets
from .types import DecoderSample, SurrogateSample


def surrogate_training_adapter(stream: Iterable[SurrogateSample]) -> Iterator[tuple[torch.Tensor, ...]]:
    for sample in stream:
        x_perm = sample.x_perm
        C = int(sample.C)
        n = int(sample.n)
        chunk_size = int(sample.chunk_size)
        k_eff = int(sample.k_eff)
        device = x_perm.device

        table_values = torch.zeros(chunk_size, C, dtype=torch.float32, device=device)
        table_dib = torch.zeros(chunk_size, C, dtype=torch.int32, device=device)
        table_carry_id = torch.full((chunk_size, C), -1, dtype=torch.int32, device=device)
        slot_ids = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(0).expand(chunk_size, n)
        _ov, _od, out_slot_id = lpap_pool(
            table_values,
            table_dib,
            table_carry_id,
            x_perm,
            slot_ids,
            k_eff,
            device,
        )

        targets_bCn = surrogate_teacher_bucket_slot_targets(out_slot_id, n)
        abs_amp = x_perm.abs()
        stride = n // C
        abs_amp_by_bucket = abs_amp.view(chunk_size, stride, C).transpose(1, 2).max(dim=2).values
        weights = abs_amp_by_bucket
        bucket_input = sample.surrogate_tokens
        yield bucket_input, targets_bCn, weights


def decoder_training_adapter(stream: Iterable[SurrogateSample]) -> Iterator[tuple[torch.Tensor, ...]]:
    for sample in stream:
        decoder_tokens = sample.decoder_tokens_sur
        j_star = sample.j_star_sur
        n = int(sample.n)
        B, C = j_star.shape
        valid_bucket = torch.ones(B, C, dtype=torch.bool, device=j_star.device)
        targets = decoder_targets_from_j_star(j_star, n, valid_bucket)
        abs_amplitude = decoder_tokens[..., 0].abs() * valid_bucket.float()
        j_target = j_star.long().clamp(0, n - 1)
        yield decoder_tokens, targets, abs_amplitude, j_target, valid_bucket


def reconstructor_training_adapter(stream: Iterable[DecoderSample]) -> Iterator[tuple[torch.Tensor, ...]]:
    for sample in stream:
        yield sample.reconstructor_tokens, sample.raw_inputs
