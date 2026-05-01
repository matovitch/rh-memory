from __future__ import annotations

import torch

from rh_memory.pipeline import (
    HarmonicSample,
    PipelineConfig,
    SurrogateInferenceSample,
    harmonic_stage,
    iter_take,
    surrogate_stage,
    surrogate_training_adapter,
)
from rh_memory.pipeline.primitives_tokens import reshape_permuted_to_bucket_tokens
from rh_memory.surrogate import RHSurrogate


def _surrogate(n: int, C: int):
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
    surrogate.eval()
    return surrogate


def test_pipeline_stage_shapes():
    n, C, B = 32, 8, 2
    config = PipelineConfig(n=n, C=C, batch_size=B, seed=7, fast_k=1.0, max_harmonics=8)
    surrogate = _surrogate(n, C)
    base = harmonic_stage(
        config=config,
        device="cpu",
    )
    s0 = next(base)
    assert isinstance(s0, HarmonicSample)
    assert s0.raw_inputs.shape == (B, n)
    assert s0.x_perm.shape == (B, n)
    assert s0.perm_1d.shape == (n,)

    s1 = next(surrogate_stage(iter([s0]), config=config, surrogate=surrogate))
    assert isinstance(s1, SurrogateInferenceSample)
    assert s1.raw_inputs.shape == (B, n)
    assert s1.perm_1d.shape == (n,)
    assert s1.x_perm.shape == (B, n)
    assert s1.surrogate_logits.shape == (B, C, n)
    assert s1.decoder_tokens.shape == (B, C, 3)


def test_pipeline_adapters_shapes():
    n, C, B = 32, 8, 2
    config = PipelineConfig(n=n, C=C, batch_size=B, seed=11, fast_k=1.0, max_harmonics=8)
    base = harmonic_stage(
        config=config,
        device="cpu",
    )
    s_batch = next(surrogate_training_adapter(base, config=config))
    assert s_batch[0].shape == (B, C, n // C)
    assert s_batch[1].shape == (B, C)
    assert s_batch[2].shape == (B, C)
    assert s_batch[3].shape == (B, C)


def test_surrogate_training_adapter_keeps_sample_x_perm_read_only():
    n, C, B = 32, 8, 2
    config = PipelineConfig(n=n, C=C, batch_size=B, seed=13, fast_k=1.0, max_harmonics=8)
    sample = next(harmonic_stage(config=config, device="cpu"))
    x_before = sample.x_perm.clone()

    bucket_input, _target_idx, _valid_bucket, _weights = next(surrogate_training_adapter(iter([sample]), config=config))

    assert torch.equal(sample.x_perm, x_before)
    assert torch.equal(bucket_input, reshape_permuted_to_bucket_tokens(x_before, C))


def test_iter_take_limits_stream():
    n, C, B = 16, 4, 2
    config = PipelineConfig(n=n, C=C, batch_size=B, seed=3, fast_k=1.0, max_harmonics=8)
    base = harmonic_stage(
        config=config,
        device="cpu",
    )
    taken = list(iter_take(base, 2))
    assert len(taken) == 2


def test_pipeline_config_validation():
    config = PipelineConfig(n=32, C=8, batch_size=2, seed=1, fast_k=1.0)
    assert config.stride == 4
    assert config.sequence_length == 32
    assert config.bucket_count == 8
    assert config.k_eff >= 1
