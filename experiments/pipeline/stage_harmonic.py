from __future__ import annotations

from typing import Iterator

import torch

from .primitives_harmonic import harmonic_raw_batch
from .primitives_permutation import build_grouped_permutation, gather_permuted_stream

from .types import HarmonicSample


def harmonic_stage(
    *,
    n: int,
    C: int,
    chunk_size: int,
    seed: int,
    device: torch.device | str,
    harmonic_decay: float,
    harmonic_amp_threshold: float,
    max_harmonics: int,
) -> Iterator[HarmonicSample]:
    while True:
        raw_inputs = harmonic_raw_batch(
            chunk_size,
            n,
            device,
            harmonic_decay,
            harmonic_amp_threshold,
            max_harmonics,
        )
        perm_1d = build_grouped_permutation(n, C, seed, device)
        x_perm = gather_permuted_stream(raw_inputs, perm_1d)
        yield HarmonicSample(
            raw_inputs=raw_inputs,
            perm_1d=perm_1d,
            x_perm=x_perm,
            n=n,
            C=C,
            chunk_size=chunk_size,
        )
