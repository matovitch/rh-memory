from __future__ import annotations

from typing import Iterable, Iterator

import torch

from .primitives_tokens import reconstructor_tokens_from_decoder_logits

from .types import DecoderSample, SurrogateSample


def decoder_stage(
    stream: Iterable[SurrogateSample],
    *,
    decoder,
) -> Iterator[DecoderSample]:
    for sample in stream:
        C = int(sample.C)
        n = int(sample.n)
        decoder_tokens = sample.decoder_tokens_sur
        x_perm = sample.x_perm
        perm_1d = sample.perm_1d

        with torch.no_grad():
            decoder_logits_full = decoder(decoder_tokens)
            decoder_logits = decoder_logits_full[:, :C, :n]
            reconstructor_tokens, _j_star_dec, _src_idx = reconstructor_tokens_from_decoder_logits(
                x_perm, decoder_logits, perm_1d
            )

        yield DecoderSample(
            raw_inputs=sample.raw_inputs,
            reconstructor_tokens=reconstructor_tokens,
        )
