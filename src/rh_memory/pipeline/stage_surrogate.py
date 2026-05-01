from __future__ import annotations

from typing import Iterable, Iterator

from .config import PipelineConfig
from .primitives_tokens import decoder_tokens_from_surrogate_logits_soft, reshape_permuted_to_bucket_tokens

from .types import HarmonicSample, SurrogateInferenceSample


def surrogate_stage(
    stream: Iterable[HarmonicSample],
    *,
    config: PipelineConfig,
    surrogate,
    temperature: float = 1.0,
) -> Iterator[SurrogateInferenceSample]:
    """Run a surrogate and emit differentiable soft decoder bucket tokens."""

    for sample in stream:
        surrogate_tokens = reshape_permuted_to_bucket_tokens(sample.x_perm, config.C)
        surrogate_logits_full = surrogate(surrogate_tokens)
        surrogate_logits = surrogate_logits_full[:, : config.C, : config.n]
        decoder_tokens = decoder_tokens_from_surrogate_logits_soft(
            sample.x_perm,
            surrogate_logits,
            temperature=temperature,
        )

        yield SurrogateInferenceSample(
            raw_inputs=sample.raw_inputs,
            perm_1d=sample.perm_1d,
            x_perm=sample.x_perm,
            surrogate_logits=surrogate_logits,
            decoder_tokens=decoder_tokens,
        )
