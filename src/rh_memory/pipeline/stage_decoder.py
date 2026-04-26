from __future__ import annotations

from typing import Iterable, Iterator

from .primitives_tokens import (
    reconstructor_tokens_from_decoder_logits_hard,
    reconstructor_tokens_from_decoder_logits_soft,
)
from .types import DecoderInferenceSample, SurrogateInferenceSample


def decoder_stage(
    stream: Iterable[SurrogateInferenceSample],
    *,
    decoder,
    temperature: float = 1.0,
    token_mode: str = "soft",
) -> Iterator[DecoderInferenceSample]:
    """Run the decoder and emit reconstructor bucket tokens."""
    if token_mode not in {"soft", "hard"}:
        raise ValueError(f"token_mode must be 'soft' or 'hard', got {token_mode!r}")

    for sample in stream:
        decoder_logits = decoder(sample.decoder_tokens)
        decoder_logits = decoder_logits[:, : sample.decoder_tokens.size(1), : sample.x_perm.size(1)]
        if token_mode == "soft":
            reconstructor_tokens = reconstructor_tokens_from_decoder_logits_soft(
                sample.x_perm,
                sample.perm_1d,
                decoder_logits,
                temperature=temperature,
            )
        else:
            reconstructor_tokens = reconstructor_tokens_from_decoder_logits_hard(
                sample.x_perm,
                sample.perm_1d,
                decoder_logits,
                bucket_amplitude=sample.decoder_tokens[..., 0],
                temperature=temperature,
            )
        yield DecoderInferenceSample(
            raw_inputs=sample.raw_inputs,
            reconstructor_tokens=reconstructor_tokens,
        )