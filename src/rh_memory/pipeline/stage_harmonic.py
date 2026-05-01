from __future__ import annotations

from typing import Iterator

import torch

from .primitives_harmonic import harmonic_raw_batch
from .primitives_permutation import build_grouped_permutation, gather_permuted_stream

from .config import PipelineConfig
from .types import HarmonicSample


def harmonic_stage(
    *,
    config: PipelineConfig,
    device: torch.device | str,
) -> Iterator[HarmonicSample]:
    perm_1d = build_grouped_permutation(config.n, config.C, config.seed, device)
    while True:
        raw_inputs = harmonic_raw_batch(
            config.batch_size,
            config.n,
            device,
            config.harmonic_decay,
            config.harmonic_amp_threshold,
            config.max_harmonics,
        )
        x_perm = gather_permuted_stream(raw_inputs, perm_1d)
        yield HarmonicSample(
            raw_inputs=raw_inputs,
            perm_1d=perm_1d,
            x_perm=x_perm,
        )
