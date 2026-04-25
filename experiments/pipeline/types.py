from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class HarmonicSample:
    """Output contract for ``harmonic_stage``."""

    raw_inputs: Tensor  # [B, n] unpermuted
    perm_1d: Tensor  # [n]
    x_perm: Tensor  # [B, n]
    n: int
    C: int
    chunk_size: int


@dataclass(frozen=True)
class SurrogateSample:
    """Output contract for ``surrogate_stage``."""

    raw_inputs: Tensor  # [B, n] unpermuted
    perm_1d: Tensor  # [n]
    x_perm: Tensor  # [B, n]
    n: int
    C: int
    chunk_size: int
    k_eff: int
    surrogate_tokens: Tensor  # [B, C, stride]
    decoder_tokens_sur: Tensor  # [B, C, 2]
    j_star_sur: Tensor  # [B, C]


@dataclass(frozen=True)
class DecoderSample:
    """Output contract for ``decoder_stage``."""

    raw_inputs: Tensor  # [B, n] unpermuted
    reconstructor_tokens: Tensor  # [B, C, 3]
