from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pipeline import (
    DecoderSample,
    HarmonicSample,
    SurrogateSample,
    decoder_stage,
    decoder_training_adapter,
    harmonic_stage,
    iter_take,
    reconstructor_training_adapter,
    surrogate_stage,
    surrogate_training_adapter,
)
from rh_memory.decoder import RHDecoder
from rh_memory.surrogate import RHSurrogate


def _models(n: int, C: int):
    surrogate = RHSurrogate(
        sequence_length=n,
        bucket_count=C,
        stride=n // C,
        fast_k=1.0,
        d_model=64,
        n_heads=4,
        num_layers=1,
        dim_feedforward=128,
    )
    decoder = RHDecoder(
        sequence_length=n,
        bucket_count=C,
        d_model=64,
        n_heads=4,
        num_layers=1,
        dim_feedforward=128,
    )
    surrogate.eval()
    decoder.eval()
    return surrogate, decoder


def test_pipeline_stage_shapes():
    n, C, B = 32, 8, 2
    surrogate, decoder = _models(n, C)
    base = harmonic_stage(
        n=n,
        C=C,
        chunk_size=B,
        seed=7,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    s0 = next(base)
    assert isinstance(s0, HarmonicSample)
    assert s0.raw_inputs.shape == (B, n)
    assert s0.x_perm.shape == (B, n)
    assert s0.perm_1d.shape == (n,)

    s1 = next(surrogate_stage(iter([s0]), surrogate=surrogate, fast_k=1.0))
    assert isinstance(s1, SurrogateSample)
    assert not hasattr(s1, "surrogate_logits")
    assert s1.decoder_tokens_sur.shape == (B, C, 2)

    s2 = next(decoder_stage(iter([s1]), decoder=decoder))
    assert isinstance(s2, DecoderSample)
    assert not hasattr(s2, "decoder_logits")
    assert s2.reconstructor_tokens.shape == (B, C, 3)
    assert s2.raw_inputs.shape == (B, n)


def test_pipeline_adapters_shapes():
    n, C, B = 32, 8, 2
    surrogate, decoder = _models(n, C)
    base = harmonic_stage(
        n=n,
        C=C,
        chunk_size=B,
        seed=11,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    sur_stream = surrogate_stage(base, surrogate=surrogate, fast_k=1.0)
    s_batch = next(surrogate_training_adapter(sur_stream))
    assert s_batch[0].shape == (B, C, n // C)
    assert s_batch[1].shape == (B, C, n)
    assert s_batch[2].shape == (B, C)

    base2 = harmonic_stage(
        n=n,
        C=C,
        chunk_size=B,
        seed=11,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    sur2 = surrogate_stage(base2, surrogate=surrogate, fast_k=1.0)
    d_batch = next(decoder_training_adapter(sur2))
    assert d_batch[0].shape == (B, C, 2)
    assert d_batch[1].shape == (B, C, n)

    base3 = harmonic_stage(
        n=n,
        C=C,
        chunk_size=B,
        seed=11,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    sur3 = surrogate_stage(base3, surrogate=surrogate, fast_k=1.0)
    dec3 = decoder_stage(sur3, decoder=decoder)
    r_batch = next(reconstructor_training_adapter(dec3))
    assert r_batch[0].shape == (B, C, 3)
    assert r_batch[1].shape == (B, n)


def test_iter_take_limits_stream():
    n, C, B = 16, 4, 2
    base = harmonic_stage(
        n=n,
        C=C,
        chunk_size=B,
        seed=3,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    taken = list(iter_take(base, 2))
    assert len(taken) == 2
