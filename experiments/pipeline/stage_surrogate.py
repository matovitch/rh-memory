from __future__ import annotations

import math
from typing import Iterable, Iterator

import torch

from .primitives_tokens import reshape_permuted_to_bucket_tokens, surrogate_bucket_tokens_from_logits

from .types import HarmonicSample, SurrogateSample


def surrogate_stage(
    stream: Iterable[HarmonicSample],
    *,
    surrogate,
    fast_k: float,
) -> Iterator[SurrogateSample]:
    for sample in stream:
        C = int(sample.C)
        n = int(sample.n)
        x_perm = sample.x_perm
        k_eff = max(1, int(fast_k * math.log(C)))
        surrogate_tokens = reshape_permuted_to_bucket_tokens(x_perm, C)

        with torch.no_grad():
            surrogate_logits_full = surrogate(surrogate_tokens)
            surrogate_logits = surrogate_logits_full[:, :C, :n]
            decoder_tokens_sur, j_star_sur = surrogate_bucket_tokens_from_logits(
                x_perm, surrogate_logits, C, k_eff
            )

        yield SurrogateSample(
            raw_inputs=sample.raw_inputs,
            perm_1d=sample.perm_1d,
            x_perm=sample.x_perm,
            n=sample.n,
            C=sample.C,
            chunk_size=sample.chunk_size,
            k_eff=k_eff,
            surrogate_tokens=surrogate_tokens,
            decoder_tokens_sur=decoder_tokens_sur,
            j_star_sur=j_star_sur,
        )
