from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class HarmonicSample:
    """Output contract for ``harmonic_stage``."""

    raw_inputs: Float[Tensor, "B N"]  # unpermuted
    perm_1d: Int[Tensor, "N"]
    x_perm: Float[Tensor, "B N"]


@dataclass(frozen=True)
class SurrogateInferenceSample:
    """Output contract for surrogate soft decoder features."""

    raw_inputs: Float[Tensor, "B N"]  # unpermuted
    perm_1d: Int[Tensor, "N"]
    x_perm: Float[Tensor, "B N"]
    surrogate_logits: Float[Tensor, "B C N"]
    decoder_tokens: Float[Tensor, "B C 3"]
